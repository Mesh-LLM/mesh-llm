//! The audit of one turn a client received: its binding, the challenge the
//! VRF of its root draws, the node's opening, and every check of the opened
//! values against the claimed GGUF. `skippy audit-receipt` runs it on a
//! turn it asked for; a Mesh node runs it on turns its peers served.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_receipts::attention::AttentionDescriptor;
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::audit::{self, AuditReport};
use skippy_receipts::binding;
use skippy_receipts::challenge;
use skippy_receipts::claimed_reads;
use skippy_receipts::coverage;
use skippy_receipts::evidence::Evidence;
use skippy_receipts::glue;
use skippy_receipts::head;
use skippy_receipts::turn;
use skippy_receipts::vrf;
use skippy_receipts::weights::WeightFile;
use skippy_receipts::wire::{Challenge, LayerOpening, MixingKind, Opening, Pick, Receipt};

pub use super::attention_audit::OpenedRows;

use super::output::{
    AuditOutput, CostOutput, GlueOutput, HeadOutput, LayerOutput, MatrixOutput, NoAuditOutput,
    RoleOutput, TokenOutput,
};
use super::{CacheTypes, ChatReply, ClaimedModel, LoadClaimedModel, claimed_file, head_key_file};
use super::{attention_audit, recurrent_audit};

/// An opening as the node sent it: decoded, its bytes, and the node's
/// signature of them.
pub struct Opened {
    pub opening: Opening,
    pub bytes: Vec<u8>,
    pub signature: Option<String>,
    pub seconds: f64,
}

/// Sends a challenge to the node that served the turn.
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

/// What the client received in a turn.
pub struct ReceivedTurn<'a> {
    pub receipt: Receipt,
    /// The key the request named.
    pub key: &'a vrf::SecretKey,
    /// The prompt and the reply, when the auditor has them.
    pub response: Option<(&'a str, &'a ChatReply)>,
    /// The request asked for greedy decoding, so the node must say it
    /// decoded greedily.
    pub asked_greedy: bool,
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
    Audited {
        output: Box<AuditOutput>,
        /// The signed receipt, the challenge, the opening and both VRF
        /// proofs, when the node signed its opening.
        evidence: Option<Box<Evidence>>,
    },
}

impl TurnAudit {
    /// `None` for no audit, else whether every check passed.
    pub fn passed(&self) -> Option<bool> {
        match self {
            Self::NoAudit(_) => None,
            Self::Audited { output, .. } => Some(output.pass),
        }
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
) -> Result<TurnAudit> {
    let ReceivedTurn {
        receipt,
        key,
        response,
        asked_greedy,
    } = turn;
    let weights = &setup.weights;
    let model = setup.model.as_ref();
    let bound = turn::verify(&receipt, Some(&key.public())).and_then(|verified| {
        match receipt.binding.as_ref().map(|b| b.sampler.as_str()) {
            Some(sampler) if asked_greedy && sampler != turn::GREEDY => bail!(
                "the node says its sampler was {sampler:?}; the request asked for greedy decoding"
            ),
            _ => Ok(verified),
        }
    });
    let shape = bound
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e:#}"))
        .and_then(|_| binding::verify_shape(&receipt, setup.layer_count));
    // a check that cannot run here would fail an honest node
    if shape.is_ok()
        && let Some(reason) = coverage::unauditable(&receipt, weights, &model.runtime())
    {
        return Ok(TurnAudit::NoAudit(NoAuditOutput::new(
            receipt.receipt_id,
            reason,
        )));
    }
    let challenge = match &bound {
        Ok(verified) => draw_challenge(
            options.layer,
            &receipt,
            weights,
            setup.layer_count,
            key,
            verified,
        )?,
        Err(_) => Challenge {
            receipt_id: receipt.receipt_id.clone(),
            picks: Vec::new(),
            proof: None,
            head: Vec::new(),
            attention: Vec::new(),
        },
    };
    let opened = match &shape {
        Ok(()) => Some(opener.open(&challenge)?),
        Err(_) => None,
    };
    let binding = shape.and_then(|()| {
        let (Ok(verified), Some(opened)) = (&bound, &opened) else {
            bail!("opened only for a bound receipt of the right shape");
        };
        let signature = opened
            .signature
            .as_deref()
            .context("the node did not sign its opening")?;
        turn::verify_opening(verified, &opened.bytes, signature)?;
        binding::verify_paths(&receipt, &challenge, &opened.opening)
    });
    let content = response.map(|(_, reply)| reply.content.clone());
    let mut output = AuditOutput::new(content.unwrap_or_default(), receipt, challenge, &binding);
    output.hand_picked = options.layer.is_some();
    output.open_seconds = opened.as_ref().map(|opened| opened.seconds);
    let mut evidence = None;
    if let (Ok(()), Some(opened)) = (&binding, &opened) {
        // the rows come from the opening exactly as it arrived
        let (rows_seed, rows_proof) = key.evaluate(&turn::rows_input(&opened.bytes));
        output.rows_proof = Some(hex::encode(rows_proof));
        let checks = Checks {
            timing_runs: options.timing_runs,
            response,
            head_key: Some(setup.head_key.clone()),
        };
        let layouts = check_opening(
            &checks,
            model,
            weights,
            &opened.opening,
            &rows_seed,
            &mut output,
        )?;
        let verified = bound.as_ref().ok();
        let open_rows = |blocks: &BlockChallenge| -> Result<OpenedRows> {
            let mut blocks = blocks.clone();
            blocks.proof = Some(hex::encode(rows_proof));
            let rows = opener.rows(&blocks)?;
            if let Some(verified) = verified {
                let signature = rows
                    .signature
                    .as_deref()
                    .context("the node did not sign its attention rows")?;
                turn::verify_rows(verified, &rows.bytes, signature)?;
            }
            Ok(rows)
        };
        let (attention, rows) = attention_audit::check_attention(
            &output.receipt,
            &output.challenge,
            &opened.opening,
            &layouts,
            &rows_seed,
            &open_rows,
        );
        output.attention = Some(attention);
        evidence = opened.signature.as_ref().map(|signature| {
            let evidence = Evidence::new(
                output.receipt.clone(),
                output.challenge.clone(),
                &opened.bytes,
                signature.clone(),
                &rows_proof,
            );
            Box::new(match rows {
                Some(rows) => evidence.with_rows(&rows.bytes, rows.signature),
                None => evidence,
            })
        });
    }
    output.finish();
    Ok(TurnAudit::Audited {
        output: Box::new(output),
        evidence,
    })
}

/// Executable layers of a claimed GGUF, from its metadata: a stage plan
/// covers these and never a next-token prediction layer.
pub(super) fn claimed_layer_count(claimed_model: &std::path::Path) -> Result<u32> {
    let meta = skippy_model_artifact::gguf::scan_gguf_compact_meta(claimed_model)
        .with_context(|| format!("read GGUF metadata {}", claimed_model.display()))?;
    meta.executable_layer_count()
        .context("the claimed GGUF names no executable layers")
}

/// Weight bytes of each executable layer, which a challenge draws from.
pub(super) fn executable_layer_bytes(weights: &WeightFile, layer_count: u32) -> BTreeMap<u32, u64> {
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
    weights: &WeightFile,
    layer_count: u32,
    key: &vrf::SecretKey,
    bound: &turn::VerifiedBinding,
) -> Result<Challenge> {
    let layer_bytes = executable_layer_bytes(weights, layer_count);
    let (seed, proof) = key.evaluate(&turn::challenge_input(&bound.turn_root));
    let mut challenge = challenge::draw(receipt, &layer_bytes, &seed)?;
    challenge.proof = Some(hex::encode(proof));
    if let Some(layer) = layer {
        let positions = challenge.picks[0].positions.clone();
        challenge.picks = vec![Pick { layer, positions }];
        challenge.attention = challenge::draw_attention(&challenge.picks, &seed);
    }
    Ok(challenge)
}

/// What the checks of an opening need beyond it.
pub(super) struct Checks<'a> {
    pub(super) timing_runs: u32,
    /// The prompt and the response, when the auditor received them.
    pub(super) response: Option<(&'a str, &'a ChatReply)>,
    /// The client's head key file, which checks the node's logits. Without
    /// it, the head check computes every logit from the output matrix.
    pub(super) head_key: Option<PathBuf>,
}

/// Every check that needs the opened values: each picked layer, the input
/// of layer 0, and the tokens. Returns each picked layer's attention ops,
/// as the auditor's graph describes them.
pub(super) fn check_opening(
    checks: &Checks<'_>,
    model: &dyn ClaimedModel,
    weights: &WeightFile,
    opening: &Opening,
    rows_seed: &[u8],
    output: &mut AuditOutput,
) -> Result<BTreeMap<u32, Vec<AttentionDescriptor>>> {
    // the layers load whole: what they read must be here first
    let layers: Vec<u32> = output.challenge.picks.iter().map(|p| p.layer).collect();
    weights
        .prefetch(&claimed_reads::glue_ranges(weights, opening, &layers)?)
        .context("fetch the tensors the glue check reads")?;
    let mut reports = Vec::new();
    let mut layouts = BTreeMap::new();
    let mut state_layers = BTreeSet::new();
    for pick in &output.challenge.picks.clone() {
        let layer = opening
            .stages
            .iter()
            .flat_map(|s| &s.layers)
            .find(|l| l.layer == pick.layer)
            .context("a verified opening holds every picked layer")?;
        let backend = output
            .receipt
            .stage_for_layer(pick.layer)
            .map(|s| s.backend.clone())
            .context("a verified receipt has a stage for every layer")?;
        let (checked, rows, glue_facts) = check_layer(
            checks.timing_runs,
            model,
            weights,
            layer,
            rows_seed,
            &backend,
        )?;
        output.layers.push(checked);
        reports.push(rows);
        if let Some((attention, states)) = glue_facts {
            layouts.insert(pick.layer, attention);
            if states > 0 {
                state_layers.insert(pick.layer);
            }
        }
    }
    output.roles = audit::pool_roles(&reports)
        .iter()
        .map(RoleOutput::from)
        .collect();
    let anchors = &opening.stages[0].anchors;
    let mut verdict = None;
    let checked = model.with_layer(0, CacheTypes::default(), &mut |evaluator| {
        verdict = Some(glue::check_anchors(evaluator, anchors)?);
        Ok(())
    });
    output.anchors = match (checked, verdict) {
        (Ok(()), Some(verdict)) => Ok(GlueOutput::from(&verdict)),
        (Err(error), _) => Err(format!("{error:#}")),
        (Ok(()), None) => Err("the anchors were not checked".into()),
    };
    output.head = check_head(
        weights,
        &output.receipt,
        opening,
        checks.head_key.as_deref(),
        rows_seed,
    );
    output.tokens = checks
        .response
        .map(|(prompt, reply)| check_tokens(model, &output.receipt, prompt, reply));
    output.recurrent = recurrent_audit::check_recurrent(
        model,
        &output.receipt,
        &output.challenge,
        opening,
        &state_layers,
    );
    Ok(layouts)
}

/// The sampled tokens at the head positions: against the node's logits,
/// which the head key checks, when the node opened them and the key is at
/// hand; else against every logit of the claimed output head on the opened
/// final residuals.
fn check_head(
    weights: &WeightFile,
    receipt: &Receipt,
    opening: &Opening,
    head_key: Option<&std::path::Path>,
    rows_seed: &[u8],
) -> Option<Result<HeadOutput, String>> {
    let last = receipt.stages.last()?;
    let openings = &opening.stage(last.stage_index)?.head;
    if openings.is_empty() {
        return None;
    }
    let started = Instant::now();
    let keyed = head_key.filter(|_| {
        openings.iter().all(|o| o.logits.is_some()) && head::logits_are_linear(weights)
    });
    let rule = match keyed {
        Some(_) => head::token_rule(receipt),
        None => head::token_rule(receipt).on_untransformed_logits(head::transforms_logits(weights)),
    };
    let checked = match keyed {
        Some(path) => head_key_file::load_or_build(path, weights).and_then(|(key, built)| {
            head::key::check_head_with_key(
                weights,
                &key,
                receipt,
                openings,
                &last.backend,
                rows_seed,
            )
            .map(|v| (v, built))
        }),
        None => head::check_head(weights, receipt, openings).map(|v| (v, None)),
    };
    Some(
        checked
            .map(|(verdict, built)| {
                let seconds = started.elapsed().as_secs_f64() - built.unwrap_or(0.0);
                let mut output = HeadOutput::new(&verdict, seconds);
                output.rule = rule.describe();
                output.key_build_seconds = built;
                output
            })
            .map_err(|e| format!("{e:#}")),
    )
}

/// The attention ops of a layer, and how many recurrent states it reads.
type GlueFacts = (Vec<AttentionDescriptor>, usize);

/// Row checks, then the glue check on the layer loaded alone. Returns the
/// layer's attention ops and recurrent states when the glue ran.
fn check_layer(
    timing_runs: u32,
    model: &dyn ClaimedModel,
    weights: &WeightFile,
    opening: &LayerOpening,
    rows_seed: &[u8],
    backend: &str,
) -> Result<(LayerOutput, AuditReport, Option<GlueFacts>)> {
    let started = Instant::now();
    let rows = audit::check_rows(weights, opening, rows_seed, backend)?;
    let row_check_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut glue_eval_seconds = Vec::new();
    let mut glue_check_seconds = 0.0;
    let mut report = None;
    let checked = model.with_layer(
        opening.layer,
        layer_cache_types(opening),
        &mut |evaluator| {
            let check_started = Instant::now();
            let checked = glue::check_glue(evaluator, opening)?;
            glue_check_seconds = check_started.elapsed().as_secs_f64();
            glue_eval_seconds.push(checked.eval_seconds);
            let request = glue::glue_request(opening)?;
            for _ in 1..timing_runs {
                let run_started = Instant::now();
                evaluator.evaluate(&request)?;
                glue_eval_seconds.push(run_started.elapsed().as_secs_f64());
            }
            report = Some(checked);
            Ok(())
        },
    );
    let layer_seconds = started.elapsed().as_secs_f64();
    let glue = match (checked, report) {
        (Ok(()), Some(report)) => Ok(report),
        (Err(error), _) => Err(format!("{error:#}")),
        (Ok(()), None) => Err("the glue was not checked".into()),
    };
    let glue_seconds: f64 = glue_check_seconds + glue_eval_seconds.iter().skip(1).sum::<f64>();
    let checked = LayerOutput {
        layer: opening.layer,
        matrices: rows.matrices.iter().map(MatrixOutput::from).collect(),
        glue: glue
            .as_ref()
            .map(|g| g.verdicts.iter().map(GlueOutput::from).collect())
            .unwrap_or_default(),
        glue_pass: glue.as_ref().is_ok_and(|g| g.pass),
        glue_error: glue.as_ref().err().cloned(),
        unchecked_weights: glue
            .as_ref()
            .map(|g| g.unchecked_weights.clone())
            .unwrap_or_default(),
        cost: CostOutput {
            positions: opening.leaves.len(),
            layer_load_seconds: (layer_seconds - glue_seconds).max(0.0),
            glue_eval_seconds,
            glue_check_seconds,
            row_check_seconds,
        },
    };
    let facts = glue.ok().map(|g| (g.attention, g.states));
    Ok((checked, rows, facts))
}

/// The response must be exactly the text of the tokens the receipt says
/// were sampled: the reasoning and the content together, up to whitespace
/// and the reasoning markers the output parser removes. A response that
/// drops or adds text fails, and so does an empty one for tokens that make
/// text. The prompt must be part of the text of the tokens before them,
/// because the chat template adds its own text around it.
fn check_tokens(
    model: &dyn ClaimedModel,
    receipt: &Receipt,
    prompt: &str,
    reply: &ChatReply,
) -> Result<TokenOutput, String> {
    let split = (receipt.generation_start - receipt.position_start) as usize;
    let mut sampled = receipt.sampled_tokens();
    while let Some(&last) = sampled.last() {
        if !model
            .is_end_of_generation(last)
            .map_err(|e| format!("{e:#}"))?
        {
            break;
        }
        sampled.pop();
    }
    let generated = model.detokenize(&sampled).map_err(|e| format!("{e:#}"))?;
    let prompt_text = model
        .detokenize(&receipt.tokens[..split])
        .map_err(|e| format!("{e:#}"))?;
    let answered = format!("{}{}", reply.reasoning, reply.content);
    Ok(TokenOutput {
        response_ok: response_matches(&generated, &answered),
        // a turn that continued a cached prefix receipts only its suffix
        prompt_ok: receipt.position_start > 0
            || comparable(&prompt_text).contains(&comparable(prompt)),
        generated,
    })
}

/// Text compared without whitespace and reasoning markers.
fn comparable(text: &str) -> String {
    text.replace("<think>", "")
        .replace("</think>", "")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn response_matches(generated: &str, answered: &str) -> bool {
    comparable(generated) == comparable(answered)
}

/// The KV cache types of the rows `opening`'s layer stored: its first
/// stored row is the key, its second the value.
fn layer_cache_types(opening: &LayerOpening) -> CacheTypes {
    let stored = |index| {
        opening
            .leaves
            .iter()
            .flat_map(|leaf| &leaf.mixing)
            .find(|m| m.kind == MixingKind::Stored && m.index == index && m.values.is_some())
            .and_then(|m| u32::try_from(m.ggml_type).ok())
    };
    let k = stored(0);
    CacheTypes {
        k,
        v: stored(1).or(k),
    }
}

#[cfg(test)]
mod tests {
    use super::response_matches;

    #[test]
    fn the_response_must_be_all_of_the_generated_text() {
        assert!(response_matches(
            "<think>\nhm\n</think>\n\nParis.",
            "hmParis."
        ));
        assert!(response_matches("Paris is big.", " Paris is  big. "));
        // dropped, added or empty text fails
        assert!(!response_matches("Paris is big.", "Paris"));
        assert!(!response_matches("Paris.", "Paris. Buy now."));
        assert!(!response_matches("Paris.", ""));
        assert!(response_matches("", ""));
    }
}
