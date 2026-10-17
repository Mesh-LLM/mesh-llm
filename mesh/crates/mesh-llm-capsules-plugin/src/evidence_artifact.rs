//! The artifact response in the interop profile's form (`evidence-interop/0`,
//! agent-action-capsule `spec/interop-profile-v0.md` §6.2): this node builds
//! one from its ledger index ([`build`]) and checks one a peer sent
//! ([`check`]).
//!
//! The envelope is the `evidence-request` crate's (`envelope::sign`,
//! `envelope::parse_reply`, `Envelope::verify_for`), and so is the profile's
//! check of a record's Checkpointed Local Log material
//! (`cll_material::check_anchor_inclusion`). Building the artifact is the
//! responder's business, not that crate's. evidencebook has a builder, but
//! over its own `CllSubstrate`, which opens the durable node store this
//! plugin's checkpointer owns: so the builder is here, over the read-only
//! index (`evidence_log`), in evidencebook's (and the Go responder's)
//! artifact forms:
//!
//! - `record`, no derivation: the record's line in the ledger, proven by the
//!   anchor's `cll-checkpoint`, its `cll-inclusion-proof` and, when the
//!   checkpoint extends another, its `cll-consistency-proof` (profile
//!   §6.2.1). The entry the proof covers is the record's `capsule_id`, which
//!   is not the SHA-256 of the line (that is the line without its
//!   `capsule_id` and producer envelope), so a profile verifier reports this
//!   check `NOT_CHECKED` (§6.2.1 step 5), never `FAIL`; this node's
//!   requester recomputes the `capsule_id` and checks the proof itself. The
//!   line is served whole so the requester keeps the producer's signature
//!   over it;
//! - `full_history`, `range`, `correlation`, `exchange`: RFC 8785 of
//!   `{anchor, evidence_stream, records, subject}`, each record
//!   `{body, digest, leaf_index}` with `body` in hex;
//! - `checkpoints`: RFC 8785 of `{anchor, checkpoints, evidence_stream,
//!   subject}`, each checkpoint its COSE statement in hex;
//! - `history_card/1`: RFC 8785 of `{anchor, derivation, evidence_stream,
//!   history, subject}`, each entry `{checkpoint, consistency_from_previous}`;
//! - `served_summary/1` (this node's own derivation): RFC 8785 of `{anchor,
//!   derivation, evidence_stream, served_summary, subject}`;
//! - `exchange_corroboration/1` (this node's own, named by its definition
//!   digest): RFC 8785 of `{anchor, derivation, entries, evidence_stream,
//!   subject}`, each entry the derivation's members with `inclusion` the
//!   half's `cll-inclusion-proof` at the anchor, in hex
//!   (`crate::exchange_corroboration`).
//!
//! **Beyond the profile.** The profile defines no material for a record set,
//! the checkpoint list or a derivation (its P-13), and no carrier for a
//! witness's receipt. A requester here checks them all the same, so this
//! node sends, where a profile verifier reads nothing:
//!
//! - for a record set, the anchor's `cll-checkpoint` and one
//!   `cll-inclusion-proof` per record, in the artifact's order; for
//!   `served_summary/1` and `exchange_corroboration/1`, the anchor's
//!   `cll-checkpoint`. A profile verifier reports `anchor_inclusion`
//!   `NOT_CHECKED` for these subjects before it reads any material
//!   (§6.2.1 step 1), so the extra items change nothing for it;
//! - the receipts its witnesses issued for the checkpoints the answer
//!   carries, as an unsigned top-level `witness_receipts` member of the
//!   reply, which a profile reader ignores. A receipt authenticates itself:
//!   it is bound to its checkpoint's digest and signed by its witness.
//!
//! The artifact is the same for every requester (-00 §5); the material and
//! the receipts may differ, as the profile allows.

use cll::checkpoint::{verify_checkpoint_cose_offline, DecodedCheckpointCose, WitnessRecord};
use cll::mmr::{leaf_count, Hash};
use ed25519_dalek::VerifyingKey;
use evidence_request::cll_material::{
    self, encode_consistency_proof, encode_inclusion_proof, KeyDecision, LogKeyPolicy,
    KIND_CHECKPOINT, KIND_CONSISTENCY, KIND_INCLUSION,
};
use evidence_request::envelope::{
    self, subject_value, Anchor, AnchorFreshness, Binding, Envelope, EnvelopeParts, Reply,
};
use evidence_request::registry::{Reason, DERIVATION_HISTORY_CARD};
use evidence_request::request::{Coverage, Derivation, Freshness, Request, Subject};
use evidence_request::resolve::ResolvedAnchor;
use evidence_request::time::parse_utc;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::evidence_log::{ChainCheckpoint, LogView, Record};

pub(crate) mod cll_proofs;

use crate::served_summary::SERVED_SUMMARY_DERIVATION_TOKEN;
use cll_proofs::{consistent, inclusion_leaf, proves};

/// The reply member that carries witness receipts beside the envelope.
pub const WITNESS_RECEIPTS: &str = "witness_receipts";

/// The most records, checkpoints or receipts a received answer may carry
/// before anything in it is checked (the responder's limits are lower).
const MAX_ITEMS: usize = 10_000;

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// The responder's side.
// ---------------------------------------------------------------------------

/// What an answer is built from, once the request is resolved and the
/// sharing policy has passed it.
pub struct Build<'a> {
    pub request: &'a Request,
    /// The request's octets as received.
    pub request_bytes: &'a [u8],
    /// The anchor as `resolve::resolve` resolved it: what the envelope names.
    pub resolved: &'a ResolvedAnchor,
    /// The chain position of the checkpoint the answer is served under
    /// (a stated one).
    pub at: usize,
    /// The leaf indices of the records the answer carries, ascending, all
    /// under that checkpoint. Empty for a checkpoint list or a derivation.
    pub records: &'a [u64],
    pub signing_key: &'a ed25519_dalek::SigningKey,
    pub issued_at: &'a str,
}

/// Build and sign the answer: the profile's flat reply, carried in a JSON
/// frame, with this node's witness receipts beside the envelope. A reason when no answer can
/// be built.
pub fn build(b: &Build<'_>, view: &LogView<'_>) -> Result<Value, Reason> {
    let index = view.index;
    let served = index.chain.get(b.at).ok_or(Reason::CoverageUnsatisfiable)?;
    let statement = &served.statement;
    let anchor_digest = served.anchor_digest();
    // The checkpoints up to the serving one: a checkpoint list and a history
    // card.
    let listed: Vec<&ChainCheckpoint> = index.chain[..=b.at].iter().collect();

    let mut head = Map::new();
    head.insert("anchor".into(), json!(anchor_digest));
    head.insert("evidence_stream".into(), json!(served.record.log_id));
    head.insert("subject".into(), subject_value(&b.request.subject));
    let mut material: Vec<(String, Vec<u8>)> = Vec::new();
    // The checkpoints whose receipts go with the answer.
    let mut carried: Vec<&ChainCheckpoint> = vec![served];

    let artifact = match (&b.request.derivation, &b.request.subject) {
        (Some(Derivation::Token(t)), _) if t == DERIVATION_HISTORY_CARD => {
            carried = listed.clone();
            history_card(view, &listed, head)?
        }
        (Some(d), Subject::FullHistory) if crate::served_summary::is_derivation(Some(d)) => {
            material.push((KIND_CHECKPOINT.into(), statement.clone()));
            jcs(served_summary(view, served, head)?)?
        }
        (Some(d), Subject::FullHistory) if crate::service_history::is_derivation(Some(d)) => {
            let card = history_card(view, &listed, head.clone())?;
            // Only the deliberately broken build that drops the member gets
            // here under any other coverage (`evidence_answer`).
            let pinned_to_latest = matches!(b.request.coverage, Coverage::ExpectedPin(_))
                && b.at + 1 == index.chain.len();
            let summary = if pinned_to_latest {
                Some(Value::Object(served_summary(view, served, head)?))
            } else {
                None
            };
            material.push((KIND_CHECKPOINT.into(), statement.clone()));
            carried = listed.clone();
            crate::service_history::artifact(&card, summary, &b.request.subject, &anchor_digest)?
                .into_bytes()
        }
        (Some(Derivation::Digest(d)), Subject::Correlation(id))
            if d == crate::exchange_corroboration::DEFINITION_DIGEST =>
        {
            let entries =
                crate::exchange_corroboration::entries(id, &served.record, b.records, view)?;
            head.insert("derivation".into(), json!(d));
            head.insert("entries".into(), entries);
            material.push((KIND_CHECKPOINT.into(), statement.clone()));
            jcs(head)?
        }
        (Some(_), _) => return Err(Reason::DerivationUnsupported),
        (None, Subject::Checkpoints) => {
            let list = listed
                .iter()
                .map(|cp| json!(hex::encode(&cp.statement)))
                .collect();
            head.insert("checkpoints".into(), Value::Array(list));
            carried = listed.clone();
            jcs(head)?
        }
        (None, Subject::Record(digest)) => {
            let [leaf] = b.records else {
                return Err(Reason::NoSuchSubject);
            };
            let record = view
                .record_at(*leaf)
                .filter(|r| r.digest == *digest)
                .ok_or(Reason::NoSuchSubject)?;
            material.push((KIND_CHECKPOINT.into(), statement.clone()));
            material.push((
                KIND_INCLUSION.into(),
                inclusion(view, record.leaf_index, served.record.mmr_size)?,
            ));
            if served.record.prev_size > 0 {
                material.push((
                    KIND_CONSISTENCY.into(),
                    consistency(view, served.record.prev_size, served.record.mmr_size)?,
                ));
            }
            record.body
        }
        (None, _) => {
            if b.records.is_empty() {
                return Err(Reason::NoSuchSubject);
            }
            let records = b
                .records
                .iter()
                .map(|&i| view.record_at(i).ok_or(Reason::NoSuchSubject))
                .collect::<Result<Vec<Record>, _>>()?;
            material.push((KIND_CHECKPOINT.into(), statement.clone()));
            for r in &records {
                material.push((
                    KIND_INCLUSION.into(),
                    inclusion(view, r.leaf_index, served.record.mmr_size)?,
                ));
            }
            head.insert(
                "records".into(),
                Value::Array(
                    records
                        .iter()
                        .map(|r| json!({"body": hex::encode(&r.body), "digest": r.digest, "leaf_index": r.leaf_index}))
                        .collect(),
                ),
            );
            jcs(head)?
        }
    };

    let freshness = match &b.request.coverage {
        Coverage::MinFreshness(Freshness::Size(_)) => {
            Some(AnchorFreshness::Size(served.record.mmr_size))
        }
        Coverage::MinFreshness(Freshness::Time(_)) => Some(AnchorFreshness::Time(profile_time(
            &served.record.timestamp,
        ))),
        Coverage::ExpectedPin(_) => None,
    };
    let signed = envelope::sign(
        b.request_bytes,
        EnvelopeParts {
            subject: b.request.subject.clone(),
            derivation: b.request.derivation.clone(),
            // Under a half pin the envelope names the pin, the requester's
            // own half (profile §7.1 anchor_match), as evidencebook and the
            // Go responder do; the artifact names the serving checkpoint.
            anchor: Anchor {
                digest: b.resolved.digest().to_string(),
                freshness,
            },
            artifact,
            verification: material,
            issued_at: b.issued_at.to_string(),
        },
        b.signing_key,
    )
    .map_err(|e| {
        tracing::warn!(error = %e, "an evidence answer could not be signed -- declining");
        Reason::PolicyDeclined
    })?;
    let mut reply: Value = serde_json::from_slice(&signed.encode(Binding::Json))
        .map_err(|_| Reason::PolicyDeclined)?;
    let receipts: Vec<Value> = carried
        .iter()
        .flat_map(|cp| {
            let digest = cp.anchor_digest();
            cp.record
                .witnesses
                .iter()
                .map(move |w| json!({"checkpoint": digest, "witness": w}))
        })
        .collect();
    if !receipts.is_empty() {
        reply[WITNESS_RECEIPTS] = Value::Array(receipts);
    }
    Ok(reply)
}

/// The `history_card/1` artifact over `listed` (the checkpoints up to the
/// serving one), from `head` (`anchor`, `evidence_stream`, `subject`): what
/// a direct request gets, and what `service_history/1` embeds.
fn history_card(
    view: &LogView<'_>,
    listed: &[&ChainCheckpoint],
    mut head: Map<String, Value>,
) -> Result<Vec<u8>, Reason> {
    let mut history = Vec::with_capacity(listed.len());
    for (i, cp) in listed.iter().enumerate() {
        // The chain grows strictly, so every checkpoint after the first
        // extends the one before it by a proof.
        let consistency = match i.checked_sub(1).map(|p| listed[p]) {
            Some(prev) => json!(hex::encode(consistency(
                view,
                prev.record.mmr_size,
                cp.record.mmr_size
            )?)),
            None => Value::Null,
        };
        history.push(json!({
            "checkpoint": hex::encode(&cp.statement),
            "consistency_from_previous": consistency,
        }));
    }
    head.insert("derivation".into(), json!(DERIVATION_HISTORY_CARD));
    head.insert("history".into(), Value::Array(history));
    jcs(head)
}

/// The `served_summary/1` artifact under `served`, from `head`. Its
/// `derivation` names the definition digest, however the request named it.
fn served_summary(
    view: &LogView<'_>,
    served: &ChainCheckpoint,
    mut head: Map<String, Value>,
) -> Result<Map<String, Value>, Reason> {
    let index = view.index;
    let covered = leaf_count(served.record.mmr_size).map_err(|_| Reason::CoverageUnsatisfiable)?;
    let facts: Vec<_> = index.leaves.iter().map(|l| &l.facts).collect();
    let summary = crate::served_summary::summary_value(
        &facts,
        &crate::served_summary::Coverage {
            checkpoint: &served.record,
            covered,
        },
        &|id| index.leaf_index_of(id),
    );
    head.insert(
        "derivation".into(),
        json!(crate::served_summary::SERVED_SUMMARY_DEFINITION_DIGEST),
    );
    head.insert("served_summary".into(), summary);
    Ok(head)
}

/// A checkpoint time in the profile's form: UTC, any fraction dropped
/// (profile §5.2).
fn profile_time(timestamp: &str) -> String {
    match timestamp.split_once('.') {
        Some((seconds, _)) => format!("{seconds}Z"),
        None => timestamp.to_string(),
    }
}

fn jcs(object: Map<String, Value>) -> Result<Vec<u8>, Reason> {
    evidence_request::jcs::to_string(&Value::Object(object))
        .map(String::into_bytes)
        .map_err(|_| Reason::PolicyDeclined)
}

pub(crate) fn inclusion(view: &LogView<'_>, leaf: u64, size: u64) -> Result<Vec<u8>, Reason> {
    view.index
        .inclusion_path(leaf, size)
        .and_then(|path| encode_inclusion_proof(leaf, &path))
        .ok_or(Reason::PolicyDeclined)
}

fn consistency(view: &LogView<'_>, from: u64, to: u64) -> Result<Vec<u8>, Reason> {
    let (old, paths, new) = view
        .index
        .consistency_parts(from, to)
        .ok_or(Reason::PolicyDeclined)?;
    encode_consistency_proof(from, to, &old, &paths, &new).ok_or(Reason::PolicyDeclined)
}

// ---------------------------------------------------------------------------
// The requester's side.
// ---------------------------------------------------------------------------

/// A received answer that checked out.
#[derive(Debug)]
pub struct Checked {
    /// The digest of the checkpoint the answer is served under.
    pub anchor: String,
    /// The leaf indices of the records it carries.
    pub records: Vec<u64>,
    /// The checkpoints it lists (`checkpoints`, `history_card/1`).
    pub checkpoints: usize,
    /// The halves an `exchange_corroboration/1` answer states, each proven
    /// under the anchor. Empty for any other answer.
    pub halves: Vec<crate::exchange_corroboration::Half>,
    /// The witness receipts it carries, each as checked.
    pub receipts: Vec<ReceiptCheck>,
}

/// What became of one witness receipt.
#[derive(Clone, Debug)]
pub struct ReceiptCheck {
    /// The digest of the checkpoint statement it is for.
    pub checkpoint: String,
    pub ts_url: String,
    pub status: ReceiptStatus,
}

/// A receipt's standing. Only `Verified` counts as witnessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptStatus {
    /// It proves its checkpoint under the key configured for its witness.
    Verified,
    /// Well formed and bound to its checkpoint, but no key is configured
    /// for its witness.
    NoKey,
    /// A placeholder that was never registered.
    Stub,
}

/// Why a received answer is not evidence.
#[derive(Debug, PartialEq, Eq)]
pub enum CheckError {
    /// The answer itself does not hold.
    Answer(String),
    /// A witness receipt is malformed, or not for the checkpoint it names.
    ReceiptMalformed,
    /// A witness receipt does not verify under the key configured for its
    /// witness, or is bound to another checkpoint.
    ReceiptBinding,
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckError::Answer(why) => f.write_str(why),
            CheckError::ReceiptMalformed => {
                f.write_str("a witness receipt on a checkpoint is malformed")
            }
            CheckError::ReceiptBinding => {
                f.write_str("a witness receipt does not cover the checkpoint it is carried on")
            }
        }
    }
}

fn fail<T>(why: impl Into<String>) -> Result<T, CheckError> {
    Err(CheckError::Answer(why.into()))
}

/// A checkpoint statement from the answer, verified under the responder's
/// key.
pub(crate) struct Stated {
    pub(crate) digest: String,
    pub(crate) cp: DecodedCheckpointCose,
}

/// The profile identifier a request names: its `profile` member, or
/// `evidence-interop/0` when it has none (profile §3.3 step 2). `None` when
/// the member is not text. This node answers and checks `evidence-interop/0`
/// only.
pub fn requested_profile(request: &Map<String, Value>) -> Option<&str> {
    match request.get("profile") {
        None => Some(envelope::PROFILE),
        Some(Value::String(profile)) => Some(profile),
        Some(_) => None,
    }
}

/// Check a reply `response` to the request whose octets as sent are `sent`
/// (parsed as `request`), from the responder whose key is `key`. Every check
/// the old pre-profile verifier made is made here: the envelope (the
/// responder, this request, its subject and derivation, the coverage asked
/// for), the artifact against the anchor checkpoint (signed by the same key),
/// every record against its digest and its inclusion proof, every
/// checkpoint list against its chain, and every witness receipt against its
/// checkpoint (under the key `witness_key` gives for its witness, if any).
pub fn check(
    sent: &[u8],
    request: &Request,
    response: &Value,
    key: &VerifyingKey,
    witness_key: &dyn Fn(&str) -> Option<VerifyingKey>,
) -> Result<Checked, CheckError> {
    let env = verified_envelope(sent, request, response, key)?;

    let derivation = match &env.derivation {
        None => None,
        Some(Derivation::Token(t))
            if t == DERIVATION_HISTORY_CARD || t == SERVED_SUMMARY_DERIVATION_TOKEN =>
        {
            Some(t.as_str())
        }
        Some(Derivation::Digest(d))
            if d == crate::exchange_corroboration::DEFINITION_DIGEST
                || d == crate::served_summary::SERVED_SUMMARY_DEFINITION_DIGEST =>
        {
            Some(d.as_str())
        }
        Some(_) => return fail("the answer is for a derivation this node does not check"),
    };

    let mut halves = Vec::new();
    let (served, records, listed) = match (derivation, &env.subject) {
        (None, Subject::Record(digest)) => {
            let report = cll_material::check_anchor_inclusion(&env, &ResponderLogKey(*key));
            let (served, leaf) = match report.status {
                // A record that is its own octets, as the profile proves one.
                cll_material::Status::Pass => {
                    // The check passed, so each kind is there exactly once.
                    let item = |kind: &str| {
                        env.verification
                            .iter()
                            .position(|v| v.kind == kind)
                            .map(|i| env.verification_content[i].as_slice())
                            .unwrap_or_default()
                    };
                    let proof = item(KIND_INCLUSION);
                    let leaf = inclusion_leaf(proof).ok_or_else(|| {
                        CheckError::Answer("the inclusion proof is malformed".into())
                    })?;
                    // The profile's check finds the path's root among the
                    // peaks; whether the leaf sits at its stated position
                    // under the anchor's size is checked here.
                    let served = stated(item(KIND_CHECKPOINT), key)?;
                    if !proves(
                        proof,
                        leaf,
                        &raw_digest(digest)?,
                        served.cp.mmr_size,
                        &served.cp.new_peak_hashes,
                    ) {
                        return fail("the record's inclusion proof does not hold");
                    }
                    (served, leaf)
                }
                // A record of this format, named by its `capsule_id`.
                cll_material::Status::NotChecked if env.artifact_digest != *digest => {
                    check_named_record(&env, digest, key)?
                }
                _ => return fail(format!("anchor_inclusion: {}", report.detail)),
            };
            (served, vec![leaf], Vec::new())
        }
        (Some(t), _) if t == DERIVATION_HISTORY_CARD => {
            let art = artifact_map(&env, Some(t))?;
            let served = check_history_card(&art, key)?;
            (served.0, Vec::new(), served.1)
        }
        (Some(t), Subject::FullHistory)
            if t == SERVED_SUMMARY_DERIVATION_TOKEN
                || t == crate::served_summary::SERVED_SUMMARY_DEFINITION_DIGEST =>
        {
            // served_summary/1: a static export, answered only under a pin.
            if !matches!(request.coverage, Coverage::ExpectedPin(_)) {
                return fail("a served summary is answered only under an expected_pin");
            }
            // It names its definition digest however it was asked for; a
            // node before the digest named the token.
            let art = artifact_map(
                &env,
                Some(crate::served_summary::SERVED_SUMMARY_DEFINITION_DIGEST),
            )
            .or_else(|_| artifact_map(&env, Some(SERVED_SUMMARY_DERIVATION_TOKEN)))?;
            if !art.get("served_summary").is_some_and(Value::is_object) {
                return fail("the artifact carries no served summary");
            }
            let served = anchor_checkpoint_only(&env, key)?;
            check_names(&art, &served)?;
            (served, Vec::new(), Vec::new())
        }
        (Some(d), Subject::Correlation(id))
            if d == crate::exchange_corroboration::DEFINITION_DIGEST =>
        {
            let art = artifact_map(&env, derivation)?;
            let served = anchor_checkpoint_only(&env, key)?;
            check_names(&art, &served)?;
            let leaves;
            (halves, leaves) = crate::exchange_corroboration::check_entries(&art, id, &served.cp)
                .map_err(|why| CheckError::Answer(why.into()))?;
            (served, leaves, Vec::new())
        }
        (Some(_), _) => return fail("the answer is for a derivation of another subject"),
        (None, Subject::Checkpoints) => {
            let art = artifact_map(&env, None)?;
            let list = art
                .get("checkpoints")
                .and_then(Value::as_array)
                .filter(|l| !l.is_empty() && l.len() <= MAX_ITEMS)
                .ok_or_else(|| CheckError::Answer("the checkpoint list is malformed".into()))?;
            let mut listed = Vec::with_capacity(list.len());
            for item in list {
                let statement = item
                    .as_str()
                    .and_then(|h| hex::decode(h).ok())
                    .ok_or_else(|| CheckError::Answer("a listed checkpoint is malformed".into()))?;
                listed.push(stated(&statement, key)?);
            }
            let served = check_chain(listed, &art)?;
            (served.0, Vec::new(), served.1)
        }
        (None, subject) => {
            let art = artifact_map(&env, None)?;
            let (served, leaves) = check_record_set(&env, &art, subject, key)?;
            (served, leaves, Vec::new())
        }
    };

    check_coverage(&env, request, &served, &records)?;
    let carried: Vec<&Stated> = if listed.is_empty() {
        vec![&served]
    } else {
        listed.iter().collect()
    };
    let receipts = check_receipts(response.get(WITNESS_RECEIPTS), &carried, witness_key)?;
    Ok(Checked {
        anchor: served.digest,
        records,
        checkpoints: listed.len(),
        halves,
        receipts,
    })
}

/// The material of an answer that carries only its anchor checkpoint (a
/// served summary, an exchange corroboration), verified under `key`.
pub(crate) fn anchor_checkpoint_only(
    env: &Envelope,
    key: &VerifyingKey,
) -> Result<Stated, CheckError> {
    match (
        env.verification.as_slice(),
        env.verification_content.as_slice(),
    ) {
        ([item], [statement]) if item.kind == KIND_CHECKPOINT => stated(statement, key),
        _ => fail("the answer carries no anchor checkpoint"),
    }
}

/// The envelope of a reply `response` to the request sent as `sent` (parsed
/// as `request`): under the profile the request named (profile §3.3 step 5),
/// signed by `key`, for these octets, and for the subject and derivation
/// asked for.
pub(crate) fn verified_envelope(
    sent: &[u8],
    request: &Request,
    response: &Value,
    key: &VerifyingKey,
) -> Result<Envelope, CheckError> {
    let octets = serde_json::to_vec(response).map_err(|e| CheckError::Answer(e.to_string()))?;
    let env = match envelope::parse_reply(&octets, Binding::Json) {
        Ok(Reply::Envelope(env)) => *env,
        Ok(Reply::Refusal(_)) => return fail("a refusal, not an answer"),
        Err(e) => return fail(e.to_string()),
    };
    // Profile §3.3 step 5: the envelope's profile is the one asked for.
    let asked = match serde_json::from_slice::<Value>(sent) {
        Ok(Value::Object(map)) => requested_profile(&map).map(str::to_string),
        _ => None,
    };
    if asked.as_deref() != Some(env.profile.as_str()) {
        return fail("the answer is not under the profile the request named");
    }
    env.verify_for(key, sent)
        .map_err(|e| CheckError::Answer(e.to_string()))?;
    if env.subject != request.subject || env.derivation != request.derivation {
        return fail("the answer is not for the requested subject or derivation");
    }
    if env.verification.len() > MAX_ITEMS + 2 {
        return fail("the answer carries too much material");
    }
    Ok(env)
}

/// The relying party's key policy for the anchor checkpoint: the
/// responder's own key, which signs this node's checkpoints.
struct ResponderLogKey(VerifyingKey);

impl LogKeyPolicy for ResponderLogKey {
    fn resolve(&self, _iss: &str, _sub: &str, kid: &[u8], _issued_at: &str) -> KeyDecision {
        if kid == self.0.as_bytes() {
            KeyDecision::Accept(self.0)
        } else {
            KeyDecision::Reject
        }
    }
}

/// A checkpoint statement that verifies under `key` (its `kid`), with its
/// own consistency claim checked (`verify_checkpoint_cose_offline`).
fn stated(statement: &[u8], key: &VerifyingKey) -> Result<Stated, CheckError> {
    let verified = verify_checkpoint_cose_offline(statement);
    match verified.decoded {
        Some(cp) if verified.ok && cp.key_id == hex::encode(key.as_bytes()) => Ok(Stated {
            digest: sha256_hex(statement),
            cp,
        }),
        _ => fail("a checkpoint is not the responder's, or does not verify"),
    }
}

/// The artifact as a JSON map, naming this subject and, exactly when one
/// was asked for, this derivation.
fn artifact_map(
    env: &Envelope,
    derivation: Option<&str>,
) -> Result<Map<String, Value>, CheckError> {
    let art: Value = serde_json::from_slice(&env.artifact)
        .map_err(|_| CheckError::Answer("the artifact is malformed".into()))?;
    let Value::Object(art) = art else {
        return fail("the artifact is malformed");
    };
    if art.get("subject") != Some(&subject_value(&env.subject))
        || art.get("derivation").and_then(Value::as_str) != derivation
        || (derivation.is_none() && art.contains_key("derivation"))
    {
        return fail("the artifact is not for the requested subject or derivation");
    }
    Ok(art)
}

/// The artifact names `served` as its anchor and its log as its stream.
pub(crate) fn check_names(art: &Map<String, Value>, served: &Stated) -> Result<(), CheckError> {
    if art.get("anchor").and_then(Value::as_str) != Some(served.digest.as_str()) {
        return fail("the artifact names another anchor than its checkpoint");
    }
    if art.get("evidence_stream").and_then(Value::as_str) != Some(served.cp.log_id.as_str()) {
        return fail("the artifact names another stream than its checkpoint's");
    }
    Ok(())
}

/// A `history_card/1` artifact's history, each checkpoint the responder's
/// (under `key`) and each extending the one before it by its consistency
/// proof, as a chain ending at the anchor the artifact names. Returns the
/// anchor and the list.
pub(crate) fn check_history_card(
    art: &Map<String, Value>,
    key: &VerifyingKey,
) -> Result<(Stated, Vec<Stated>), CheckError> {
    let history = art
        .get("history")
        .and_then(Value::as_array)
        .filter(|h| !h.is_empty() && h.len() <= MAX_ITEMS)
        .ok_or_else(|| CheckError::Answer("the history is malformed".into()))?;
    let mut listed: Vec<Stated> = Vec::with_capacity(history.len());
    for entry in history {
        let statement = hex_member(entry, "checkpoint")?;
        let cp = stated(&statement, key)?;
        match (listed.last(), entry.get("consistency_from_previous")) {
            (None, Some(Value::Null)) => {}
            (Some(prev), Some(Value::String(proof))) => {
                let proof = hex::decode(proof)
                    .map_err(|_| CheckError::Answer("a consistency proof is malformed".into()))?;
                if !consistent(
                    prev.cp.mmr_size,
                    &prev.cp.new_peak_hashes,
                    cp.cp.mmr_size,
                    &cp.cp.new_peak_hashes,
                    &proof,
                ) {
                    return fail("a consistency proof does not hold");
                }
            }
            _ => return fail("a history entry's consistency proof is missing or out of place"),
        }
        listed.push(cp);
    }
    check_chain(listed, art)
}

/// A listed chain (`checkpoints`, `history_card/1`) starts the stream, each
/// checkpoint extends the one before it, and the last is the anchor the
/// artifact names. Returns the anchor and the list.
fn check_chain(
    listed: Vec<Stated>,
    art: &Map<String, Value>,
) -> Result<(Stated, Vec<Stated>), CheckError> {
    let first = listed
        .first()
        .ok_or_else(|| CheckError::Answer("no checkpoints".into()))?;
    if first.cp.prev_size != 0 || !first.cp.prev_root.is_empty() {
        return fail("the checkpoint list does not start the stream");
    }
    for pair in listed.windows(2) {
        let (prev, next) = (&pair[0].cp, &pair[1].cp);
        if next.prev_size != prev.mmr_size
            || next.prev_root != prev.root
            || next.log_id != prev.log_id
            || next.mmr_size <= prev.mmr_size
        {
            return fail("a checkpoint does not extend the one listed before it");
        }
    }
    let last = listed.last().expect("not empty");
    check_names(art, last)?;
    let served = Stated {
        digest: last.digest.clone(),
        cp: last.cp.clone(),
    };
    Ok((served, listed))
}

/// A record of this format answering a `record` subject: the artifact is a
/// record whose recomputed `capsule_id` is the subject, and the material is
/// the anchor checkpoint, the record's inclusion proof and, when there is
/// one, the checkpoint's consistency proof, in that order. Returns the
/// anchor and the record's leaf.
fn check_named_record(
    env: &Envelope,
    digest: &str,
    key: &VerifyingKey,
) -> Result<(Stated, u64), CheckError> {
    if crate::evidence_log::record_digest(&env.artifact).as_deref() != Some(digest) {
        return fail("the artifact is not the record the subject names");
    }
    let kinds: Vec<&str> = env.verification.iter().map(|v| v.kind.as_str()).collect();
    if !matches!(
        kinds.as_slice(),
        [KIND_CHECKPOINT, KIND_INCLUSION] | [KIND_CHECKPOINT, KIND_INCLUSION, KIND_CONSISTENCY]
    ) {
        return fail("the material is not the anchor checkpoint and the record's proofs");
    }
    let served = stated(&env.verification_content[0], key)?;
    let leaf = inclusion_leaf(&env.verification_content[1])
        .ok_or_else(|| CheckError::Answer("the inclusion proof is malformed".into()))?;
    let covered = leaf_count(served.cp.mmr_size)
        .map_err(|_| CheckError::Answer("the anchor's size is not an MMR size".into()))?;
    let entry = raw_digest(digest)?;
    if leaf >= covered
        || !proves(
            &env.verification_content[1],
            leaf,
            &entry,
            served.cp.mmr_size,
            &served.cp.new_peak_hashes,
        )
    {
        return fail("the record's inclusion proof does not hold");
    }
    if let Some(proof) = env.verification_content.get(2) {
        if !consistent(
            served.cp.prev_size,
            &served.cp.prev_peak_hashes,
            served.cp.mmr_size,
            &served.cp.new_peak_hashes,
            proof,
        ) {
            return fail("the anchor's consistency proof does not hold");
        }
    }
    Ok((served, leaf))
}

fn raw_digest(digest: &str) -> Result<Hash, CheckError> {
    hex::decode(digest)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| CheckError::Answer("a record's digest is malformed".into()))
}

/// A record set: each record's body is the record its digest names, the
/// anchor checkpoint and one inclusion proof per record prove each under it,
/// and the records are the ones the subject names. A range or the full
/// history must be exactly those positions. A `correlation` or `exchange`
/// answer's records must each carry the identifier or cite the half, by the
/// rules the responder's index uses; that the set is complete is not
/// checkable here, and is what the requester's own half proves (-00 §6).
/// Returns the anchor and the leaves.
fn check_record_set(
    env: &Envelope,
    art: &Map<String, Value>,
    subject: &Subject,
    key: &VerifyingKey,
) -> Result<(Stated, Vec<u64>), CheckError> {
    let records = art
        .get("records")
        .and_then(Value::as_array)
        .filter(|r| !r.is_empty() && r.len() <= MAX_ITEMS)
        .ok_or_else(|| CheckError::Answer("the records are malformed".into()))?;
    let kinds: Vec<&str> = env.verification.iter().map(|v| v.kind.as_str()).collect();
    let expected =
        std::iter::once(KIND_CHECKPOINT).chain(std::iter::repeat_n(KIND_INCLUSION, records.len()));
    if !kinds.iter().copied().eq(expected) {
        return fail(
            "the material is not the anchor checkpoint and one inclusion proof per record",
        );
    }
    let served = stated(&env.verification_content[0], key)?;
    check_names(art, &served)?;
    let covered = leaf_count(served.cp.mmr_size)
        .map_err(|_| CheckError::Answer("the anchor's size is not an MMR size".into()))?;

    let mut leaves = Vec::with_capacity(records.len());
    let mut bodies = Vec::with_capacity(records.len());
    for (record, proof) in records.iter().zip(&env.verification_content[1..]) {
        let body = hex_member(record, "body")?;
        // A record of this format, or one that is its own octets.
        let digest = record
            .get("digest")
            .and_then(Value::as_str)
            .filter(|d| {
                *d == sha256_hex(&body)
                    || crate::evidence_log::record_digest(&body).as_deref() == Some(*d)
            })
            .ok_or_else(|| {
                CheckError::Answer("a record's body is not the record its digest names".into())
            })?;
        let leaf = record
            .get("leaf_index")
            .and_then(Value::as_u64)
            .filter(|&i| i < covered)
            .ok_or_else(|| CheckError::Answer("a record is not under the anchor".into()))?;
        if leaves.last().is_some_and(|&prev| prev >= leaf) {
            return fail("the records are not in log order");
        }
        let raw = raw_digest(digest)?;
        if !proves(
            proof,
            leaf,
            &raw,
            served.cp.mmr_size,
            &served.cp.new_peak_hashes,
        ) {
            return fail("a record's inclusion proof does not hold");
        }
        leaves.push(leaf);
        bodies.push(body);
    }
    let fits = match subject {
        Subject::Range(a, b) => leaves.iter().copied().eq(*a..=*b),
        Subject::FullHistory => leaves.iter().copied().eq(0..covered),
        Subject::Correlation(_) | Subject::Exchange(_) => {
            bodies.iter().all(|b| record_fits(subject, b))
        }
        // Answered before a record set is read.
        Subject::Record(_) | Subject::Checkpoints => false,
    };
    if !fits {
        return fail("the records are not the ones the subject names");
    }
    Ok((served, leaves))
}

/// Whether a record body is one the index files under a `correlation` or
/// `exchange` subject: a JSON object, not padding, that carries the
/// identifier or cites the half. It mirrors this plugin's own index, so it
/// holds only for answers from plugin peers (the one live caller is
/// `mesh_evidence_bridge`); -00 §3 does not yet define what "carries" means
/// for other responders.
fn record_fits(subject: &Subject, body: &[u8]) -> bool {
    let Some(record) = serde_json::from_slice::<Value>(body)
        .ok()
        .filter(|r| r.is_object() && !crate::producer::padding::is_padding(r))
    else {
        return false;
    };
    match subject {
        Subject::Correlation(id) => crate::evidence_log::record_correlation_ids(&record)
            .iter()
            .any(|c| c == id),
        Subject::Exchange(half) => crate::evidence_log::counterparty_half_citations(&record)
            .iter()
            .any(|h| h == half),
        _ => false,
    }
}

/// The coverage the request asked for: the pinned anchor, or one at least
/// as fresh as asked, which the envelope states as the anchor attains it.
pub(crate) fn check_coverage(
    env: &Envelope,
    request: &Request,
    served: &Stated,
    records: &[u64],
) -> Result<(), CheckError> {
    let half_pin = matches!((&request.subject, &request.coverage),
        (Subject::Exchange(half), Coverage::ExpectedPin(pin)) if half == pin);
    if !half_pin && env.anchor.digest != served.digest {
        return fail("the envelope names another anchor than the artifact's");
    }
    match &request.coverage {
        Coverage::ExpectedPin(pin) => {
            if env.anchor.digest != *pin || env.anchor.freshness.is_some() {
                return fail("the anchor is not the pinned one");
            }
        }
        Coverage::MinFreshness(Freshness::Size(n)) => {
            let covered = leaf_count(served.cp.mmr_size).unwrap_or(0);
            if env.anchor.freshness != Some(AnchorFreshness::Size(served.cp.mmr_size))
                || covered < *n
            {
                return fail("the anchor is not as fresh as asked");
            }
        }
        Coverage::MinFreshness(Freshness::Time(t)) => {
            let at = profile_time(&served.cp.timestamp);
            if env.anchor.freshness != Some(AnchorFreshness::Time(at.clone()))
                || parse_utc(&at).is_none_or(|at| at < *t)
            {
                return fail("the anchor is not as fresh as asked");
            }
        }
    }
    // A half pinned by itself is answered under the earliest checkpoint
    // covering the records served: the anchor's predecessor must not
    // already cover the last of them.
    if half_pin {
        let before = leaf_count(served.cp.prev_size)
            .map_err(|_| CheckError::Answer("the anchor's size is not an MMR size".into()))?;
        if records.iter().max().is_none_or(|&last| last < before) {
            return fail("the anchor is not the earliest covering the records");
        }
    }
    Ok(())
}

fn hex_member(value: &Value, member: &str) -> Result<Vec<u8>, CheckError> {
    value
        .get(member)
        .and_then(Value::as_str)
        .and_then(|h| hex::decode(h).ok())
        .ok_or_else(|| CheckError::Answer(format!("{member} is not hex")))
}

/// Check every receipt the reply carries. Each must name one of the
/// checkpoints the answer carries and be bound to it: its entry hash is
/// `SHA-256(checkpoint digest)`, the digest of the checkpoint's signed JSON
/// form, which its statement states. It must be a receipt this node reads.
/// When a key is configured for its witness, its proof and signature must
/// cover that entry under the key. Stubs were never registered and are
/// never verified.
pub(crate) fn check_receipts(
    carried: Option<&Value>,
    checkpoints: &[&Stated],
    witness_key: &dyn Fn(&str) -> Option<VerifyingKey>,
) -> Result<Vec<ReceiptCheck>, CheckError> {
    use base64::Engine as _;
    let Some(carried) = carried else {
        return Ok(Vec::new());
    };
    let list = carried
        .as_array()
        .filter(|l| l.len() <= MAX_ITEMS)
        .ok_or(CheckError::ReceiptMalformed)?;
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        let checkpoint = item
            .get("checkpoint")
            .and_then(Value::as_str)
            .and_then(|d| checkpoints.iter().find(|cp| cp.digest == d))
            .ok_or(CheckError::ReceiptMalformed)?;
        let w: WitnessRecord = item
            .get("witness")
            .and_then(|w| serde_json::from_value(w.clone()).ok())
            .ok_or(CheckError::ReceiptMalformed)?;
        let check = |status| ReceiptCheck {
            checkpoint: checkpoint.digest.clone(),
            ts_url: w.ts_url.clone(),
            status,
        };
        if w.is_stub {
            out.push(check(ReceiptStatus::Stub));
            continue;
        }
        let digest = hex::decode(checkpoint.cp.to_checkpoint_record().digest())
            .map_err(|_| CheckError::ReceiptMalformed)?;
        let entry: [u8; 32] = Sha256::digest(&digest).into();
        if w.entry_hash != hex::encode(entry) {
            return Err(CheckError::ReceiptBinding);
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&w.receipt_b64)
            .map_err(|_| CheckError::ReceiptMalformed)?;
        match witness_key(&w.ts_url) {
            Some(key) => match evidence_request::receipt::verify(&bytes, &entry, &key) {
                Ok(()) => out.push(check(ReceiptStatus::Verified)),
                Err(evidence_request::receipt::ReceiptError::Malformed(_)) => {
                    return Err(CheckError::ReceiptMalformed)
                }
                Err(evidence_request::receipt::ReceiptError::NotForThisLeaf) => {
                    return Err(CheckError::ReceiptBinding)
                }
            },
            None => {
                evidence_request::receipt::check_form(&bytes)
                    .map_err(|_| CheckError::ReceiptMalformed)?;
                out.push(check(ReceiptStatus::NoKey));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
