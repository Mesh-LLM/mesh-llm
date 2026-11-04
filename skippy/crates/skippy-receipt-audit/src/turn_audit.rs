//! The audit of one turn a client received: its binding, the challenge the
//! VRF of its root draws, the node's opening, and every check of the opened
//! values against the claimed GGUF. `skippy audit-receipt` runs it on a
//! turn it asked for; a Mesh node runs it on turns its peers served.
//!
//! An audit ends one of four ways ([`Outcome`]), and which one depends on
//! who stopped it:
//!
//! - Before the challenge, the client decides from the receipt alone
//!   whether it can audit the turn at all. A receipt it cannot audit is no
//!   audit. A receipt that is not bound to its key and signed, does not
//!   describe a turn of the claimed model, or contradicts the request, is
//!   a failure.
//! - A node that gives no usable signed answer to the challenge, or to the
//!   request for attention rows, leaves the audit unanswered.
//! - Once the node signed an answer, every defect of it is a failure, and
//!   the evidence shows it.
//! - What then still stops a check is the client's own: the claimed GGUF
//!   out of reach, a layer that does not load. That is no audit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::binding;
use skippy_receipts::challenge;
use skippy_receipts::coverage;
use skippy_receipts::evidence::Evidence;
use skippy_receipts::node_fault::NodeFault;
use skippy_receipts::opened_values::{self, ModelShapes};
use skippy_receipts::sampler_request::RequestedSampling;
use skippy_receipts::turn::{self, VerifiedBinding};
use skippy_receipts::vrf;
use skippy_receipts::weights::WeightFile;
use skippy_receipts::wire::{Challenge, Opening, Pick, Receipt};

pub use crate::attention_audit::OpenedRows;

use crate::attention_audit::{self, RowsAnswer};
use crate::chat_reply::ChatReply;
use crate::claimed_model::{ClaimedModel, LoadClaimedModel};
use crate::opening_checks::{Checks, check_opening};
use crate::outcome::{Check, Outcome};
use crate::output::{AuditOutput, NoAuditOutput, UnansweredOutput};
use crate::{claimed_file, head_key_file};

/// An opening as the node sent it: decoded, its bytes, and the node's
/// signature of them.
pub struct Opened {
    pub opening: Opening,
    pub bytes: Vec<u8>,
    pub signature: Option<String>,
    pub seconds: f64,
}

/// Sends a challenge to the node that served the turn.
///
/// An error from either call means the node gave no usable answer: the
/// request did not get through, timed out, came back with an error status,
/// or its body does not decode. The audit is then unanswered.
pub trait OpenChallenge {
    fn open(&self, challenge: &Challenge) -> Result<Opened>;
    /// Asks for the K and V rows of the attention blocks the client drew
    /// after the opening.
    fn rows(&self, challenge: &BlockChallenge) -> Result<OpenedRows>;
}

/// The claimed model, ready to audit turns against.
pub struct ClaimedSetup {
    pub weights: WeightFile,
    pub model: Box<dyn ClaimedModel>,
    /// Executable layers, from the GGUF's metadata.
    pub layer_count: u32,
    /// The client's head key file for the model.
    pub head_key: PathBuf,
}

impl ClaimedSetup {
    /// Opens the claimed GGUF: a file here, or `repo@revision/file` on
    /// Hugging Face, read by range into a sparse copy under `cache_root`.
    /// A first read of a remote header takes seconds, so this comes before
    /// any challenge.
    pub fn open(claimed: &Path, cache_root: &Path, load: LoadClaimedModel<'_>) -> Result<Self> {
        Self::from_file(claimed_file::open(claimed, cache_root)?, load)
    }

    /// The claimed model from an opened claimed file.
    pub fn from_file(
        claimed: claimed_file::ClaimedFile,
        load: LoadClaimedModel<'_>,
    ) -> Result<Self> {
        let layer_count = claimed_layer_count(&claimed.path)?;
        Ok(Self {
            head_key: head_key_file::path_for(&claimed.identity),
            weights: claimed.weights,
            model: load(claimed.path),
            layer_count,
        })
    }
}

/// What the client received in a turn, and what it asked for.
pub struct ReceivedTurn<'a> {
    pub receipt: Receipt,
    /// The key the request named.
    pub key: &'a vrf::SecretKey,
    /// The prompt and the reply, when the auditor has them.
    pub response: Option<(&'a str, &'a ChatReply)>,
    /// The sampling parameters the request set. The node's sampler
    /// statement must not contradict them.
    pub sampling: RequestedSampling,
    /// The id the request asked for the receipt under, when it named one.
    pub asked_id: Option<&'a str>,
}

/// How an audit runs its checks.
pub struct AuditOptions {
    /// Challenge this layer alone instead of the ones the VRF picks.
    pub layer: Option<u32>,
    /// Times to compute each challenged layer for the cost report.
    pub timing_runs: u32,
}

impl Default for AuditOptions {
    fn default() -> Self {
        Self {
            layer: None,
            timing_runs: 1,
        }
    }
}

/// What an audit of a turn found.
pub enum TurnAudit {
    /// A check this client cannot run: it says nothing about the node.
    NoAudit(NoAuditOutput),
    /// The node gave no usable signed answer to the challenge, or to the
    /// request for attention rows. Nothing it committed was checked, and
    /// that is the node's doing: not a pass, a failure or no audit.
    Unanswered(UnansweredOutput),
    /// The node answered. The output's outcome is a pass, a failure, or no
    /// audit when a check could not run here.
    Audited {
        output: Box<AuditOutput>,
        /// The signed receipt, the challenge, the opening and both VRF
        /// proofs: what `skippy recheck-receipt` checks again. Absent for
        /// a challenge picked by hand, which is not evidence.
        evidence: Option<Box<Evidence>>,
    },
}

impl TurnAudit {
    pub fn outcome(&self) -> Outcome {
        match self {
            Self::NoAudit(_) => Outcome::NoAudit,
            Self::Unanswered(_) => Outcome::Unanswered,
            Self::Audited { output, .. } => output.outcome,
        }
    }

    fn no_audit(receipt_id: &str, error: &anyhow::Error) -> Self {
        Self::NoAudit(NoAuditOutput::new(receipt_id, format!("{error:#}")))
    }
}

/// Whether the client audits the turn `receipt` records, at `rate`: the VRF
/// of the turn root decides, so the node cannot know in advance. A receipt
/// that is not bound to `key` is never audited.
pub fn audit_drawn(receipt: &Receipt, key: &vrf::SecretKey, rate: f64) -> bool {
    turn::verify(receipt, Some(&key.public())).is_ok_and(|bound| {
        let (seed, _) = key.evaluate(&turn::challenge_input(&bound.turn_root));
        challenge::audits(&seed, rate)
    })
}

/// Audits `turn` against the claimed model, opening the challenge with
/// `opener`.
pub fn audit_turn(
    setup: &ClaimedSetup,
    turn: ReceivedTurn<'_>,
    opener: &dyn OpenChallenge,
    options: &AuditOptions,
) -> TurnAudit {
    let ReceivedTurn {
        receipt,
        key,
        response,
        sampling,
        asked_id,
    } = turn;
    let weights = &setup.weights;
    // each audit fetches within its own budget, however long the model
    // stays open
    weights.reset_fetch_budget();
    let model = setup.model.as_ref();
    let id = receipt.receipt_id.clone();
    // a check that cannot run here would fail an honest node; this reads
    // the receipt alone, so the node cannot make it true after the challenge
    if let Some(reason) = coverage::unauditable(&receipt, weights, &model.runtime()) {
        return TurnAudit::NoAudit(NoAuditOutput::new(id, reason));
    }
    let content = response.map(|(_, reply)| reply.content.clone());
    let asked = Asked {
        sampling: &sampling,
        id: asked_id,
    };
    // a receipt that is refused is not challenged: there is nothing to ask
    let bound = match turn::verify(&receipt, Some(&key.public())) {
        Ok(bound) => bound,
        Err(error) => {
            return refused(content, receipt, |output| {
                output.binding_error = Some(format!("{error:#}"));
            });
        }
    };
    let drawn = fits_claimed_model(weights, setup.layer_count, &receipt)
        .and_then(|()| draw_challenge(options.layer, &receipt, setup, key, &bound));
    let challenge = match drawn {
        Ok(challenge) => challenge,
        Err(error) if NodeFault::find(&error).is_some() => {
            return refused(content, receipt, |output| {
                output.receipt_error = Some(format!("{error:#}"));
            });
        }
        Err(error) => return TurnAudit::no_audit(&id, &error),
    };
    let request_errors = asked.contradictions(&receipt);
    let mut output = AuditOutput::new(content.unwrap_or_default(), receipt, challenge);
    output.request_errors = request_errors;
    output.hand_picked = options.layer.is_some();
    let opened = match open_signed(opener, &bound, &output.challenge) {
        Ok(opened) => opened,
        Err(reason) => return unanswered(output, "open", reason),
    };
    output.open_seconds = Some(opened.answer.seconds);
    // the rows come from the opening exactly as it arrived
    let (rows_seed, rows_proof) = key.evaluate(&turn::rows_input(&opened.answer.bytes));
    output.rows_proof = Some(hex::encode(rows_proof));
    let mut evidence = (!output.hand_picked).then(|| {
        Evidence::new(
            output.receipt.clone(),
            output.challenge.clone(),
            &opened.answer.bytes,
            opened.signature.clone(),
            &rows_proof,
        )
    });
    let (receipt, challenge) = (output.receipt.clone(), output.challenge.clone());
    let validated = binding::verify_paths(&receipt, &challenge, &opened.answer.opening)
        .and_then(|verified| opened_values::validate(verified, weights));
    let validated = match validated {
        Ok(validated) => validated,
        Err(error) if NodeFault::find(&error).is_some() => {
            output.opening_error = Some(format!("{error:#}"));
            return audited(output, evidence);
        }
        Err(error) => return TurnAudit::no_audit(&id, &error),
    };
    let checks = Checks {
        timing_runs: options.timing_runs,
        response,
        head_key: Some(setup.head_key.clone()),
    };
    let layouts = check_opening(&checks, model, weights, &validated, &rows_seed, &mut output);
    note_node_seed(&sampling, &mut output);
    let open_rows = |blocks: &BlockChallenge| -> RowsAnswer {
        let mut blocks = blocks.clone();
        blocks.proof = Some(hex::encode(rows_proof));
        signed_rows(opener, &bound, &blocks)
    };
    let (attention, rows) =
        attention_audit::check_attention(&validated, &layouts, &rows_seed, &open_rows);
    output.attention = Some(attention);
    if let Some(rows) = rows {
        evidence = evidence.map(|evidence| evidence.with_rows(&rows.bytes, rows.signature));
    }
    audited(output, evidence)
}

/// A draw replayed with a seed the request did not set shows that the node
/// followed the seed it states, and not that it had no choice of seed: the
/// report says so.
fn note_node_seed(sampling: &RequestedSampling, output: &mut AuditOutput) {
    let replayed = matches!(
        &output.head,
        Some(Check::Verdict(head)) if head.positions.iter().any(|p| p.draw.is_some())
    );
    if replayed && sampling.client_seed().is_none() {
        output.notes.push(
            "the request set no seed: the draws were replayed with the seed the node chose".into(),
        );
    }
}

/// Adds the checks up. An audit that failed nothing and whose rows request
/// went unanswered is unanswered.
fn audited(mut output: AuditOutput, evidence: Option<Evidence>) -> TurnAudit {
    output.finish();
    let unanswered = output
        .attention
        .as_ref()
        .and_then(|a| a.rows_unanswered.clone());
    match (output.outcome, unanswered) {
        (Outcome::Unanswered, Some(reason)) => TurnAudit::Unanswered(UnansweredOutput::new(
            output.receipt.receipt_id,
            "rows",
            reason,
        )),
        _ => TurnAudit::Audited {
            output: Box::new(output),
            evidence: evidence.map(Box::new),
        },
    }
}

/// A node that did not answer `request`. When the receipt already
/// contradicts the request, that failure stands.
fn unanswered(mut output: AuditOutput, request: &'static str, reason: String) -> TurnAudit {
    if output.request_errors.is_empty() {
        return TurnAudit::Unanswered(UnansweredOutput::new(
            output.receipt.receipt_id,
            request,
            reason,
        ));
    }
    output.notes.push(format!(
        "the node did not answer /receipts/{request}: {reason}"
    ));
    audited(output, None)
}

/// A receipt the audit refuses before any challenge, with what `record`
/// says of it: a failure with nothing asked and nothing opened.
fn refused(
    content: Option<String>,
    receipt: Receipt,
    record: impl FnOnce(&mut AuditOutput),
) -> TurnAudit {
    let nothing_asked = Challenge {
        receipt_id: receipt.receipt_id.clone(),
        picks: Vec::new(),
        proof: None,
        head: Vec::new(),
        attention: Vec::new(),
    };
    let mut output = AuditOutput::new(content.unwrap_or_default(), receipt, nothing_asked);
    record(&mut output);
    audited(output, None)
}

/// What the client asked for besides the turn.
struct Asked<'a> {
    sampling: &'a RequestedSampling,
    id: Option<&'a str>,
}

impl Asked<'_> {
    /// How the receipt contradicts the request: it is for another id, or
    /// its sampler statement does not fit the parameters the client set.
    fn contradictions(&self, receipt: &Receipt) -> Vec<String> {
        let mut found = Vec::new();
        if let Some(id) = self.id.filter(|id| *id != receipt.receipt_id) {
            found.push(format!(
                "the receipt is for request {:?}; the client asked under {id:?}",
                receipt.receipt_id
            ));
        }
        let sampler = receipt.binding.as_ref().map(|b| b.sampler.as_str());
        found.extend(sampler.and_then(|sampler| self.sampling.contradiction(sampler)));
        found
    }
}

/// Checks that a receipt the node signed describes a turn of the claimed
/// model, before anything is asked of the node: its stages, positions and
/// cached prefix, and every token it names is one the model embeds.
///
/// An error that blames the node refuses the receipt. Any other is the
/// client's: its claimed GGUF does not name the model's shapes.
pub(crate) fn fits_claimed_model(
    weights: &WeightFile,
    layer_count: u32,
    receipt: &Receipt,
) -> Result<()> {
    binding::verify_shape(receipt, layer_count)?;
    let shapes = ModelShapes::of(weights)?;
    opened_values::check_tokens(receipt, &shapes)
}

/// An opening the node signed for the challenge.
struct SignedOpening {
    /// The answer as it arrived.
    answer: Opened,
    /// The node's signature of it, which verified.
    signature: String,
}

/// Asks the node to open `challenge`. An error is why its answer is not
/// one the node can be held to: none came, or it is not signed for this
/// challenge.
fn open_signed(
    opener: &dyn OpenChallenge,
    bound: &VerifiedBinding,
    challenge: &Challenge,
) -> Result<SignedOpening, String> {
    let answer = opener.open(challenge).map_err(|e| format!("{e:#}"))?;
    let signature = answer
        .signature
        .clone()
        .ok_or("the node did not sign its opening")?;
    turn::verify_opening(bound, challenge, &answer.bytes, &signature)
        .map_err(|e| format!("{e:#}"))?;
    Ok(SignedOpening { answer, signature })
}

/// Asks the node for the rows of `blocks`.
fn signed_rows(
    opener: &dyn OpenChallenge,
    bound: &VerifiedBinding,
    blocks: &BlockChallenge,
) -> RowsAnswer {
    let rows = match opener.rows(blocks) {
        Ok(rows) => rows,
        Err(error) => return RowsAnswer::Unanswered(format!("{error:#}")),
    };
    let Some(signature) = rows.signature.as_deref() else {
        return RowsAnswer::Unanswered("the node did not sign its attention rows".into());
    };
    match turn::verify_rows(bound, blocks, &rows.bytes, signature) {
        Ok(()) => RowsAnswer::Signed(rows),
        Err(error) => RowsAnswer::Unanswered(format!("{error:#}")),
    }
}

/// Executable layers of a claimed GGUF, from its metadata: a stage plan
/// covers these and never a next-token prediction layer.
pub fn claimed_layer_count(claimed_model: &std::path::Path) -> Result<u32> {
    let meta = skippy_model_artifact::gguf::scan_gguf_compact_meta(claimed_model)
        .with_context(|| format!("read GGUF metadata {}", claimed_model.display()))?;
    meta.executable_layer_count()
        .context("the claimed GGUF names no executable layers")
}

/// Weight bytes of each executable layer, which a challenge draws from.
pub(crate) fn executable_layer_bytes(weights: &WeightFile, layer_count: u32) -> BTreeMap<u32, u64> {
    weights
        .layer_bytes()
        .into_iter()
        .filter(|(layer, _)| *layer < layer_count)
        .collect()
}

/// The challenge the VRF of the turn root picks, with its proof.
fn draw_challenge(
    layer: Option<u32>,
    receipt: &Receipt,
    setup: &ClaimedSetup,
    key: &vrf::SecretKey,
    bound: &VerifiedBinding,
) -> Result<Challenge> {
    let layer_bytes = executable_layer_bytes(&setup.weights, setup.layer_count);
    let (seed, proof) = key.evaluate(&turn::challenge_input(&bound.turn_root));
    let mut challenge = challenge::draw(receipt, &layer_bytes, &seed)?;
    challenge.proof = Some(hex::encode(proof));
    if let Some(layer) = layer {
        if layer >= setup.layer_count {
            bail!(
                "layer {layer} is not one of the model's {} layers",
                setup.layer_count
            );
        }
        let positions = challenge
            .picks
            .first()
            .map(|pick| pick.positions.clone())
            .context("the challenge picks no layer")?;
        challenge.picks = vec![Pick { layer, positions }];
        challenge.attention = challenge::draw_attention(&challenge.picks, &seed);
    }
    Ok(challenge)
}

#[cfg(test)]
mod tests;
