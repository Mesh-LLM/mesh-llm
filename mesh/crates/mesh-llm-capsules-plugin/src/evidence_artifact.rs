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
//!   derivation, evidence_stream, served_summary, subject}`.
//!
//! **Beyond the profile.** The profile defines no material for a record set,
//! the checkpoint list or a derivation (its P-13), and no carrier for a
//! witness's receipt. A requester here checks them all the same, so this
//! node sends, where a profile verifier reads nothing:
//!
//! - for a record set, the anchor's `cll-checkpoint` and one
//!   `cll-inclusion-proof` per record, in the artifact's order; for
//!   `served_summary/1`, the anchor's `cll-checkpoint`. A profile verifier
//!   reports `anchor_inclusion` `NOT_CHECKED` for these subjects before it
//!   reads any material (§6.2.1 step 1), so the extra items change nothing
//!   for it;
//! - the receipts its witnesses issued for the checkpoints the answer
//!   carries, as an unsigned top-level `witness_receipts` member of the
//!   reply, which a profile reader ignores. A receipt authenticates itself:
//!   it is bound to its checkpoint's digest and signed by its witness.
//!
//! The artifact is the same for every requester (-00 §5); the material and
//! the receipts may differ, as the profile allows.

use cll::checkpoint::{verify_checkpoint_cose_offline, DecodedCheckpointCose, WitnessRecord};
use cll::mmr::{height_at, interior_hash, leaf_count, leaf_hash, pos_to_leaf_index, Hash};
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
use crate::served_summary::SERVED_SUMMARY_DERIVATION_TOKEN;

/// The reply member that carries witness receipts beside the envelope.
pub const WITNESS_RECEIPTS: &str = "witness_receipts";

/// The most records, checkpoints or receipts a received answer may carry
/// before anything in it is checked (the responder's limits are lower).
const MAX_ITEMS: usize = 10_000;

/// The longest inclusion or consistency path an MMR of profile size can
/// have.
const MAX_PATH: usize = 64;

/// The largest MMR node index read from a peer: the profile's integers.
const MAX_NODE_INDEX: u64 = (1 << 53) - 1;

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

/// Build and sign the answer: the reply in the JSON binding, with this
/// node's witness receipts beside the envelope. A reason when no answer can
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
            let mut history = Vec::with_capacity(listed.len());
            for (i, cp) in listed.iter().enumerate() {
                // The chain grows strictly, so every checkpoint after the
                // first extends the one before it by a proof.
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
            carried = listed.clone();
            jcs(head)?
        }
        (Some(Derivation::Token(t)), Subject::FullHistory)
            if t == SERVED_SUMMARY_DERIVATION_TOKEN =>
        {
            let covered =
                leaf_count(served.record.mmr_size).map_err(|_| Reason::CoverageUnsatisfiable)?;
            let facts: Vec<_> = index.leaves.iter().map(|l| &l.facts).collect();
            let summary = crate::served_summary::summary_value(
                &facts,
                &crate::served_summary::Coverage {
                    checkpoint: &served.record,
                    covered,
                },
                &|id| index.leaf_index_of(id).is_some(),
            );
            head.insert("derivation".into(), json!(SERVED_SUMMARY_DERIVATION_TOKEN));
            head.insert("served_summary".into(), summary);
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

fn inclusion(view: &LogView<'_>, leaf: u64, size: u64) -> Result<Vec<u8>, Reason> {
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
struct Stated {
    digest: String,
    cp: DecodedCheckpointCose,
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
    let octets = serde_json::to_vec(response).map_err(|e| CheckError::Answer(e.to_string()))?;
    let env = match envelope::parse_reply(&octets, Binding::Json) {
        Ok(Reply::Envelope(env)) => env,
        Ok(Reply::Refusal(_)) => return fail("a refusal, not an answer"),
        Err(e) => return fail(e.to_string()),
    };
    env.verify_for(key, sent)
        .map_err(|e| CheckError::Answer(e.to_string()))?;
    if env.subject != request.subject || env.derivation != request.derivation {
        return fail("the answer is not for the requested subject or derivation");
    }
    if env.verification.len() > MAX_ITEMS + 2 {
        return fail("the answer carries too much material");
    }

    let derivation = match &env.derivation {
        None => None,
        Some(Derivation::Token(t))
            if t == DERIVATION_HISTORY_CARD || t == SERVED_SUMMARY_DERIVATION_TOKEN =>
        {
            Some(t.as_str())
        }
        Some(_) => return fail("the answer is for a derivation this node does not check"),
    };

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
                    let leaf = inclusion_leaf(item(KIND_INCLUSION)).ok_or_else(|| {
                        CheckError::Answer("the inclusion proof is malformed".into())
                    })?;
                    (stated(item(KIND_CHECKPOINT), key)?, leaf)
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
                        let proof = hex::decode(proof).map_err(|_| {
                            CheckError::Answer("a consistency proof is malformed".into())
                        })?;
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
                    _ => {
                        return fail(
                            "a history entry's consistency proof is missing or out of place",
                        )
                    }
                }
                listed.push(cp);
            }
            let served = check_chain(listed, &art)?;
            (served.0, Vec::new(), served.1)
        }
        (Some(_), Subject::FullHistory) => {
            // served_summary/1: a static export, answered only under a pin.
            if !matches!(request.coverage, Coverage::ExpectedPin(_)) {
                return fail("a served summary is answered only under an expected_pin");
            }
            let art = artifact_map(&env, derivation)?;
            if !art.get("served_summary").is_some_and(Value::is_object) {
                return fail("the artifact carries no served summary");
            }
            let [statement] = env.verification_content.as_slice() else {
                return fail("the answer carries no anchor checkpoint");
            };
            if env.verification[0].kind != KIND_CHECKPOINT {
                return fail("the answer carries no anchor checkpoint");
            }
            let served = stated(statement, key)?;
            check_names(&art, &served)?;
            (served, Vec::new(), Vec::new())
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
        receipts,
    })
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
fn check_names(art: &Map<String, Value>, served: &Stated) -> Result<(), CheckError> {
    if art.get("anchor").and_then(Value::as_str) != Some(served.digest.as_str()) {
        return fail("the artifact names another anchor than its checkpoint");
    }
    if art.get("evidence_stream").and_then(Value::as_str) != Some(served.cp.log_id.as_str()) {
        return fail("the artifact names another stream than its checkpoint's");
    }
    Ok(())
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

/// A record set: each record's body is the record its digest names, the leaves are
/// what the subject asks for, and the anchor checkpoint and one inclusion
/// proof per record prove each under it. Returns the anchor and the leaves.
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
    }
    let fits = match subject {
        Subject::Range(a, b) => leaves.iter().copied().eq(*a..=*b),
        Subject::FullHistory => leaves.iter().copied().eq(0..covered),
        _ => true,
    };
    if !fits {
        return fail("the records are not the ones the subject names");
    }
    Ok((served, leaves))
}

/// The coverage the request asked for: the pinned anchor, or one at least
/// as fresh as asked, which the envelope states as the anchor attains it.
fn check_coverage(
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

/// A `cll-inclusion-proof` item: `[MMR node index, inclusion path]`.
fn inclusion_parts(octets: &[u8]) -> Option<(u64, Vec<Hash>)> {
    let value: ciborium::Value = ciborium::from_reader(octets).ok()?;
    let ciborium::Value::Array(items) = value else {
        return None;
    };
    let [index, path] = items.as_slice() else {
        return None;
    };
    let index = u64::try_from(index.as_integer()?).ok()?;
    if index > MAX_NODE_INDEX {
        return None;
    }
    Some((index, hash_list(path)?))
}

/// The leaf an inclusion proof is for.
fn inclusion_leaf(octets: &[u8]) -> Option<u64> {
    let (index, _) = inclusion_parts(octets)?;
    pos_to_leaf_index(index).ok()
}

/// Whether `proof` proves the leaf `leaf` with entry `entry` in the MMR of
/// `size` nodes whose peaks are `peaks` (profile §6.2.1 step 12). The path
/// must climb from the leaf's own node to the one peak that holds it, so a
/// leaf cannot be passed off at another position: a peak that is itself a
/// leaf would otherwise match any position given an empty path.
fn proves(proof: &[u8], leaf: u64, entry: &Hash, size: u64, peaks: &[Hash]) -> bool {
    let Some((index, path)) = inclusion_parts(proof) else {
        return false;
    };
    if cll::mmr::leaf_index_to_pos(leaf).ok() != Some(index) || path.len() > MAX_PATH {
        return false;
    }
    let Ok(positions) = cll::mmr::peaks(size) else {
        return false;
    };
    let Some(k) = positions.iter().position(|&p| p >= index) else {
        return false;
    };
    positions.len() == peaks.len()
        && included_root(index, leaf_hash(entry), &path) == Some((positions[k], peaks[k]))
}

/// MMR's `included_root`: the node at `index` with hash `node`, carried up
/// its path, as the position it reaches and its hash. `None` when the path
/// climbs past any MMR of profile size.
fn included_root(mut index: u64, mut node: Hash, path: &[Hash]) -> Option<(u64, Hash)> {
    for (height, sibling) in (height_at(index)..).zip(path) {
        if index.checked_add(1)? > MAX_NODE_INDEX {
            return None;
        }
        if height_at(index + 1) > height {
            index += 1;
            node = interior_hash(sibling, &node, index);
        } else {
            index = index.checked_add(2u64.checked_shl(height)?)?;
            if index > MAX_NODE_INDEX {
                return None;
            }
            node = interior_hash(&node, sibling, index);
        }
    }
    Some((index, node))
}

/// Whether `proof` (a `cll-consistency-proof` item) proves that the MMR of
/// `to` nodes with peaks `to_peaks` extends the one of `from` nodes with
/// peaks `from_peaks` (profile §6.2.1 step 13).
fn consistent(
    from_size: u64,
    from_peaks: &[Hash],
    to_size: u64,
    to_peaks: &[Hash],
    proof: &[u8],
) -> bool {
    (|| -> Option<bool> {
        let value: ciborium::Value = ciborium::from_reader(proof).ok()?;
        let ciborium::Value::Array(items) = value else {
            return None;
        };
        let [from, to, paths, right] = items.as_slice() else {
            return None;
        };
        let from = u64::try_from(from.as_integer()?).ok()?;
        let to = u64::try_from(to.as_integer()?).ok()?;
        if from != from_size || to != to_size || from == 0 || from > to {
            return Some(false);
        }
        let ciborium::Value::Array(paths) = paths else {
            return None;
        };
        let peak_indices = cll::mmr::peaks(from).ok()?;
        if paths.len() != peak_indices.len() || paths.len() != from_peaks.len() {
            return Some(false);
        }
        let mut roots: Vec<Hash> = Vec::new();
        for ((index, path), peak) in peak_indices.iter().zip(paths).zip(from_peaks) {
            let path = hash_list(path)?;
            if path.len() > MAX_PATH {
                return Some(false);
            }
            let (_, root) = included_root(*index, *peak, &path)?;
            if roots.last() != Some(&root) {
                roots.push(root);
            }
        }
        roots.extend(hash_list(right)?);
        Some(roots == to_peaks)
    })()
    .unwrap_or(false)
}

fn hash_list(value: &ciborium::Value) -> Option<Vec<Hash>> {
    let ciborium::Value::Array(items) = value else {
        return None;
    };
    if items.len() > MAX_PATH {
        return None;
    }
    items
        .iter()
        .map(|item| match item {
            ciborium::Value::Bytes(b) => b.as_slice().try_into().ok(),
            _ => None,
        })
        .collect()
}

/// Check every receipt the reply carries. Each must name one of the
/// checkpoints the answer carries and be bound to it: its entry hash is
/// `SHA-256(checkpoint digest)`, the digest of the checkpoint's signed JSON
/// form, which its statement states. It must be a receipt this node reads.
/// When a key is configured for its witness, its proof and signature must
/// cover that entry under the key. Stubs were never registered and are
/// never verified.
fn check_receipts(
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
mod tests {
    use super::*;
    use crate::evidence_answer::{self, Outcome, Responder};
    use crate::evidence_request_parity::{
        announced_keys, corpus_ledger, corpus_ledger_dir, NODE_KEY_SEED, RESPONDER_NODE,
    };
    use ed25519_dalek::SigningKey;
    use evidence_request::request;

    const NOW: &str = "2026-09-29T00:00:00Z";

    fn key() -> SigningKey {
        SigningKey::from_bytes(&NODE_KEY_SEED)
    }

    /// A node over the corpus that answers anyone everything.
    struct Node {
        _dir: tempfile::TempDir,
        ledger: std::path::PathBuf,
        anchors: Vec<String>,
    }

    fn node() -> Node {
        let dir = tempfile::tempdir().unwrap();
        let ledger = dir.path().join("ledger");
        let anchors = corpus_ledger_dir(&ledger, &key())
            .iter()
            .map(|c| c.anchor())
            .collect();
        Node {
            _dir: dir,
            ledger,
            anchors,
        }
    }

    impl Node {
        fn ask(&self, request: &Value) -> (Vec<u8>, Value) {
            let bytes = serde_json::to_vec(request).unwrap();
            let announced = announced_keys();
            let replay = crate::requester_auth::ReplayCache::default();
            let key = key();
            let outcome = evidence_answer::answer_and_log(
                &bytes,
                false,
                &Responder {
                    ledger_dir: &self.ledger,
                    signing_key: &key,
                    now: NOW,
                    history_segments: "peers",
                    share_standard_history: true,
                    keys: crate::requester_auth::Keys {
                        announced: &announced,
                        configured: None,
                    },
                    replay: &replay,
                    node_id: Some(RESPONDER_NODE),
                },
                None,
            );
            let Outcome::Answered { wire, .. } = outcome else {
                panic!("{request}: {outcome:?}");
            };
            (bytes, wire)
        }

        fn pin(&self, i: usize) -> Value {
            json!({"expected_pin": self.anchors[i]})
        }
    }

    fn check_reply(sent: &[u8], reply: &Value) -> Result<Checked, CheckError> {
        let req = request::parse_json(sent).unwrap();
        check(sent, &req, reply, &key().verifying_key(), &|_| None)
    }

    fn envelope_of(reply: &Value) -> Envelope {
        match envelope::parse_reply(&serde_json::to_vec(reply).unwrap(), Binding::Json) {
            Ok(Reply::Envelope(env)) => *env,
            other => panic!("{other:?}"),
        }
    }

    /// The same answer with `change` made to its parts, signed again by the
    /// responder: what a responder that lies, with its own key, can send.
    fn resigned(sent: &[u8], reply: &Value, change: impl FnOnce(&mut EnvelopeParts)) -> Value {
        let env = envelope_of(reply);
        let mut parts = EnvelopeParts {
            subject: env.subject.clone(),
            derivation: env.derivation.clone(),
            anchor: env.anchor.clone(),
            artifact: env.artifact.clone(),
            verification: env
                .verification
                .iter()
                .zip(&env.verification_content)
                .map(|(v, o)| (v.kind.clone(), o.clone()))
                .collect(),
            issued_at: env.issued_at.clone(),
        };
        change(&mut parts);
        let signed = envelope::sign(sent, parts, &key()).unwrap();
        let mut out: Value = serde_json::from_slice(&signed.encode(Binding::Json)).unwrap();
        if let Some(r) = reply.get(WITNESS_RECEIPTS) {
            out[WITNESS_RECEIPTS] = r.clone();
        }
        out
    }

    fn artifact_json(parts: &EnvelopeParts) -> Value {
        serde_json::from_slice(&parts.artifact).unwrap()
    }

    fn jcs_bytes(v: &Value) -> Vec<u8> {
        evidence_request::jcs::to_string(v).unwrap().into_bytes()
    }

    /// A record answer carries the record's line and the profile's
    /// Checkpointed Local Log material for its `capsule_id`. A profile
    /// verifier cannot tie that name to the line and reports `NOT_CHECKED`,
    /// never `FAIL`; this node's requester checks it, for every record under
    /// either anchor.
    #[test]
    fn a_record_answer_is_checked_here_and_never_failed_by_the_profile() {
        let node = node();
        let ledger = corpus_ledger();
        for (i, record) in ledger.iter().enumerate().take(300) {
            if record.get("record_type") == Some(&json!("padding")) || i == 21 {
                continue;
            }
            let at = usize::from(i >= 16);
            let id = record["capsule_id"].as_str().unwrap();
            let (sent, reply) =
                node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(at)}));
            let env = envelope_of(&reply);
            assert_eq!(
                env.artifact,
                record.to_string().into_bytes(),
                "the line, whole"
            );
            let report =
                cll_material::check_anchor_inclusion(&env, &ResponderLogKey(key().verifying_key()));
            assert_eq!(report.status, cll_material::Status::NotChecked, "{i}");
            assert_eq!(
                env.verification
                    .iter()
                    .map(|v| v.kind.as_str())
                    .collect::<Vec<_>>(),
                if at == 0 {
                    vec![KIND_CHECKPOINT, KIND_INCLUSION]
                } else {
                    vec![KIND_CHECKPOINT, KIND_INCLUSION, KIND_CONSISTENCY]
                }
            );
            let checked = check_reply(&sent, &reply).unwrap();
            assert_eq!(
                (checked.anchor.as_str(), checked.records.as_slice()),
                (node.anchors[at].as_str(), &[i as u64][..])
            );
        }
    }

    /// The same material proves the record's `capsule_id` preimage, a record
    /// that is its own octets: the profile's check passes on it, and this
    /// node's requester accepts it too, as it would from evidencebook.
    #[test]
    fn a_record_that_is_its_own_octets_passes_the_profile_and_is_accepted() {
        let node = node();
        let record = &corpus_ledger()[17];
        let id = record["capsule_id"].as_str().unwrap();
        let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(1)}));
        let mut preimage = record.clone();
        let members = preimage.as_object_mut().unwrap();
        for local in ["capsule_id", "signature", "key_id"] {
            members.remove(local);
        }
        let own = resigned(&sent, &reply, |p| p.artifact = jcs_bytes(&preimage));
        let report = cll_material::check_anchor_inclusion(
            &envelope_of(&own),
            &ResponderLogKey(key().verifying_key()),
        );
        assert_eq!(
            report.status,
            cll_material::Status::Pass,
            "{}",
            report.detail
        );
        assert_eq!(check_reply(&sent, &own).unwrap().records, [17]);
    }

    /// A record answer whose line, proof or consistency proof is not the one
    /// it claims is not evidence, even signed by the responder.
    #[test]
    fn a_resigned_record_that_does_not_hold_is_refused() {
        let node = node();
        let ledger = corpus_ledger();
        let id = ledger[17]["capsule_id"].as_str().unwrap();
        let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(1)}));
        assert!(check_reply(&sent, &reply).is_ok());
        type Change = Box<dyn Fn(&mut EnvelopeParts)>;
        let other = ledger[18].to_string().into_bytes();
        let tampered: Vec<(&str, Change)> = vec![
            (
                "another record's line",
                Box::new(move |p| p.artifact = other.clone()),
            ),
            (
                "its own line, changed",
                Box::new(|p| {
                    let mut line: Value = serde_json::from_slice(&p.artifact).unwrap();
                    line["effect"]["status"] = json!("failed");
                    p.artifact = line.to_string().into_bytes();
                }),
            ),
            (
                "no inclusion proof",
                Box::new(|p| {
                    p.verification.truncate(1);
                }),
            ),
            (
                "a consistency proof that does not hold",
                Box::new(|p| {
                    let last = p.verification[2].1.len() - 1;
                    p.verification[2].1[last] ^= 1;
                }),
            ),
        ];
        for (what, change) in tampered {
            let bad = resigned(&sent, &reply, |p| change(p));
            assert!(check_reply(&sent, &bad).is_err(), "{what}");
        }
    }

    /// The material a record set carries beyond the profile is read by no
    /// profile verifier: it reports `NOT_CHECKED` before it looks.
    #[test]
    fn a_record_sets_extra_material_is_not_checked_by_the_profile() {
        let node = node();
        let (sent, reply) =
            node.ask(&json!({"subject": {"range": [5, 12]}, "coverage": node.pin(0)}));
        let env = envelope_of(&reply);
        let report =
            cll_material::check_anchor_inclusion(&env, &ResponderLogKey(key().verifying_key()));
        assert_eq!(report.status, cll_material::Status::NotChecked);
        assert_eq!(
            check_reply(&sent, &reply).unwrap().records,
            (5..=12).collect::<Vec<_>>()
        );
    }

    /// The artifact is the same for every requester (-00 §5): two requests
    /// for the same subject under the same anchor, differing in their nonce
    /// and coverage form, get byte-identical artifacts.
    #[test]
    fn the_artifact_is_the_same_whoever_asks() {
        let node = node();
        for subject in [
            json!({"range": [16, 20]}),
            json!({"checkpoints": null}),
            json!({"correlation": "n-1"}),
        ] {
            let a = node
                .ask(&json!({"subject": subject, "coverage": node.pin(1), "nonce": "a"}))
                .1;
            let b = node
                .ask(&json!({"subject": subject, "coverage": {"min_freshness": 300}, "nonce": "b"}))
                .1;
            assert_eq!(a["artifact"], b["artifact"], "{subject}");
            assert_ne!(a["request_digest"], b["request_digest"]);
        }
    }

    /// What a responder signs with its own key is still checked against its
    /// log: a record set whose body, leaf, proof or anchor is not the one it
    /// claims is not evidence.
    #[test]
    fn a_resigned_record_set_that_does_not_hold_is_refused() {
        let node = node();
        let (sent, reply) =
            node.ask(&json!({"subject": {"range": [16, 20]}, "coverage": node.pin(1)}));
        assert!(check_reply(&sent, &reply).is_ok());
        type Change = Box<dyn Fn(&mut EnvelopeParts)>;
        let tampered: Vec<(&str, Change)> = vec![
            (
                "a changed body",
                Box::new(|p| {
                    let mut art = artifact_json(p);
                    art["records"][1]["body"] = json!(hex::encode(b"{}"));
                    p.artifact = jcs_bytes(&art);
                }),
            ),
            (
                "a body under another record's digest",
                Box::new(|p| {
                    let mut art = artifact_json(p);
                    art["records"][1]["body"] = art["records"][2]["body"].clone();
                    art["records"][1]["digest"] = art["records"][2]["digest"].clone();
                    p.artifact = jcs_bytes(&art);
                }),
            ),
            (
                "a record dropped from the range",
                Box::new(|p| {
                    let mut art = artifact_json(p);
                    art["records"].as_array_mut().unwrap().remove(2);
                    p.artifact = jcs_bytes(&art);
                    p.verification.remove(3);
                }),
            ),
            ("proofs swapped", Box::new(|p| p.verification.swap(1, 2))),
            (
                "a proof missing",
                Box::new(|p| {
                    p.verification.pop();
                }),
            ),
            (
                "another anchor named",
                Box::new(|p| {
                    let mut art = artifact_json(p);
                    art["anchor"] = json!("e".repeat(64));
                    p.artifact = jcs_bytes(&art);
                }),
            ),
            (
                "the anchor checkpoint signed by another key",
                Box::new(|p| {
                    let other = SigningKey::from_bytes(&[9; 32]);
                    let ledger = corpus_ledger();
                    let cps = crate::evidence_request_parity::corpus_checkpoints(&ledger, &other);
                    p.verification[0].1 = cps[1].statement.clone();
                }),
            ),
        ];
        for (what, change) in tampered {
            let bad = resigned(&sent, &reply, |p| change(p));
            assert!(
                matches!(check_reply(&sent, &bad), Err(CheckError::Answer(_))),
                "{what}"
            );
        }
    }

    /// A checkpoint list and a history card must start the stream and chain
    /// to the anchor, and a history card's consistency proofs must hold.
    #[test]
    fn a_resigned_checkpoint_list_or_history_that_does_not_chain_is_refused() {
        let node = node();
        let (sent, reply) =
            node.ask(&json!({"subject": {"checkpoints": null}, "coverage": node.pin(1)}));
        assert_eq!(check_reply(&sent, &reply).unwrap().checkpoints, 2);
        let bad = resigned(&sent, &reply, |p| {
            let mut art = artifact_json(p);
            art["checkpoints"].as_array_mut().unwrap().remove(0);
            p.artifact = jcs_bytes(&art);
        });
        assert!(
            check_reply(&sent, &bad).is_err(),
            "a list that does not start the stream"
        );

        let card = json!({"subject": {"full_history": null}, "coverage": node.pin(1), "derivation": "history_card/1"});
        let (sent, reply) = node.ask(&card);
        assert_eq!(check_reply(&sent, &reply).unwrap().checkpoints, 2);
        let bad = resigned(&sent, &reply, |p| {
            let mut art = artifact_json(p);
            let proof = art["history"][1]["consistency_from_previous"]
                .as_str()
                .unwrap()
                .to_string();
            let mut bytes = hex::decode(proof).unwrap();
            let last = bytes.len() - 1;
            bytes[last] ^= 1;
            art["history"][1]["consistency_from_previous"] = json!(hex::encode(bytes));
            p.artifact = jcs_bytes(&art);
        });
        assert!(
            check_reply(&sent, &bad).is_err(),
            "a consistency proof that does not hold"
        );
        let bad = resigned(&sent, &reply, |p| {
            let mut art = artifact_json(p);
            art["history"][1]["consistency_from_previous"] = Value::Null;
            p.artifact = jcs_bytes(&art);
        });
        assert!(
            check_reply(&sent, &bad).is_err(),
            "a missing consistency proof"
        );
    }

    /// The envelope must answer what was asked: the subject, the pin, and
    /// the freshness the anchor attains as the envelope states it.
    #[test]
    fn an_answer_to_another_question_is_refused() {
        let node = node();
        let ledger = corpus_ledger();
        let id = ledger[0]["capsule_id"].as_str().unwrap();
        let (sent, reply) =
            node.ask(&json!({"subject": {"record": id}, "coverage": {"min_freshness": 300}}));
        assert!(check_reply(&sent, &reply).is_ok());
        let bad = resigned(&sent, &reply, |p| {
            p.anchor.freshness = Some(AnchorFreshness::Size(1))
        });
        assert!(
            check_reply(&sent, &bad).is_err(),
            "a freshness the anchor does not attain"
        );
        // Asked under the other anchor: the same answer does not satisfy it.
        let other =
            serde_json::to_vec(&json!({"subject": {"record": id}, "coverage": node.pin(0)}))
                .unwrap();
        let req = request::parse_json(&other).unwrap();
        assert!(check(&other, &req, &reply, &key().verifying_key(), &|_| None).is_err());
    }

    /// A served summary is under the newest anchor, named in its material.
    #[test]
    fn a_served_summary_names_its_checkpoint() {
        let node = node();
        let (sent, reply) = node.ask(&json!({"subject": {"full_history": null}, "coverage": node.pin(1), "derivation": "served_summary/1"}));
        let checked = check_reply(&sent, &reply).unwrap();
        assert_eq!(checked.anchor, node.anchors[1]);
        let bad = resigned(&sent, &reply, |p| p.verification.clear());
        assert!(check_reply(&sent, &bad).is_err());
    }

    /// A receipt is for one of the checkpoints the answer carries, by its
    /// statement's digest; naming any other is malformed.
    #[test]
    fn a_receipt_must_name_a_checkpoint_the_answer_carries() {
        let node = node();
        let (sent, mut reply) =
            node.ask(&json!({"subject": {"checkpoints": null}, "coverage": node.pin(1)}));
        let receipts = reply[WITNESS_RECEIPTS].as_array().unwrap().clone();
        assert_eq!(receipts.len(), 1, "the corpus's one stub witness");
        assert_eq!(receipts[0]["checkpoint"], json!(node.anchors[1]));
        let checked = check_reply(&sent, &reply).unwrap();
        assert_eq!(checked.receipts[0].status, ReceiptStatus::Stub);
        reply[WITNESS_RECEIPTS][0]["checkpoint"] = json!("e".repeat(64));
        assert_eq!(
            check_reply(&sent, &reply).unwrap_err(),
            CheckError::ReceiptMalformed
        );
    }

    /// A ledger the producer itself wrote: `records` sealed exchanges, with
    /// capsule-emit's own checkpointer cutting a checkpoint after each
    /// record listed in `cuts`. Returns the directory, the ledger and the
    /// node key.
    fn producer_ledger(
        records: usize,
        cuts: &[usize],
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        SigningKey,
        Vec<String>,
    ) {
        use crate::producer::anchor::AnchorClient;
        use crate::producer::checkpoint::{CheckpointCadenceConfig, CheckpointState};
        let dir = tempfile::tempdir().unwrap();
        let state = crate::capsule_emit::CapsuleState::open(dir.path(), "node-under-test").unwrap();
        let ledger = state.ledger_dir().to_path_buf();
        let (mut checkpoints, _) = CheckpointState::load(
            &ledger,
            "node-under-test",
            CheckpointCadenceConfig::default(),
        )
        .unwrap();
        let anchor = AnchorClient::new("http://127.0.0.1:1");
        let mut ids = Vec::new();
        for i in 0..records {
            let id = format!("e-{i}");
            let emitted = state
                .emit_for_exchange(&crate::capsule_emit::ExchangeRecord {
                    model: "m",
                    client_nonce: None,
                    request_bytes: b"{}",
                    response_bytes: b"{}",
                    latency_ms: 1.0,
                    exchange_id: Some(&id),
                    requesting_party: Some("party"),
                    host_provenance: None,
                })
                .unwrap();
            ids.push(emitted.capsule_id);
            if cuts.contains(&i) {
                checkpoints
                    .checkpoint_covering(&ids[i], state.signing_key(), &anchor)
                    .unwrap();
            }
        }
        let key = state.signing_key().clone();
        (dir, ledger, key, ids)
    }

    /// The anchors of `ledger` (stated checkpoints) and its chain length.
    fn anchors_of(ledger: &std::path::Path, key: &SigningKey) -> (Vec<String>, usize) {
        let key_id = hex::encode(key.verifying_key().to_bytes());
        crate::evidence_log::with_index(ledger, &key_id, |index| {
            let stated = index.chain.iter().map(|c| c.anchor_digest()).collect();
            (stated, index.chain.len())
        })
        .unwrap()
    }

    /// `request` answered from `ledger`, openly, and checked as a requester
    /// checks it.
    fn ask_ledger(
        ledger: &std::path::Path,
        key: &SigningKey,
        request: &Value,
    ) -> (Result<Checked, CheckError>, Value) {
        let bytes = serde_json::to_vec(request).unwrap();
        let announced = announced_keys();
        let replay = crate::requester_auth::ReplayCache::default();
        let outcome = evidence_answer::answer_and_log(
            &bytes,
            false,
            &Responder {
                ledger_dir: ledger,
                signing_key: key,
                now: NOW,
                history_segments: "peers",
                share_standard_history: true,
                keys: crate::requester_auth::Keys {
                    announced: &announced,
                    configured: None,
                },
                replay: &replay,
                node_id: Some(RESPONDER_NODE),
            },
            None,
        );
        let Outcome::Answered { wire, .. } = outcome else {
            panic!("{request}: {outcome:?}");
        };
        let req = request::parse_json(&bytes).unwrap();
        (
            check(&bytes, &req, &wire, &key.verifying_key(), &|_| None),
            wire,
        )
    }

    /// A ledger the producer itself wrote, checkpointed by capsule-emit's
    /// own checkpointer: its checkpoints anchor answers, and a record, the
    /// full history and a history card served from it verify. A profile
    /// verifier reports the record `NOT_CHECKED`, never `FAIL`.
    #[test]
    fn a_ledger_the_producer_wrote_is_answered_and_verifies() {
        let (_dir, ledger, key, ids) = producer_ledger(5, &[2, 4]);
        let (stated, total) = anchors_of(&ledger, &key);
        assert_eq!(
            (stated.len(), total),
            (2, 2),
            "both checkpoints are stated anchors"
        );
        let (checked, wire) = ask_ledger(
            &ledger,
            &key,
            &json!({"subject": {"record": ids[3]}, "coverage": {"expected_pin": stated[1]}}),
        );
        assert_eq!(checked.unwrap().anchor, stated[1]);
        let report = cll_material::check_anchor_inclusion(
            &envelope_of(&wire),
            &ResponderLogKey(key.verifying_key()),
        );
        assert_eq!(
            report.status,
            cll_material::Status::NotChecked,
            "{}",
            report.detail
        );
        let full = json!({"subject": {"full_history": null}, "coverage": {"min_freshness": 1}});
        assert!(ask_ledger(&ledger, &key, &full).0.unwrap().records.len() >= 5);
        let card = json!({"subject": {"full_history": null}, "coverage": {"expected_pin": stated[1]}, "derivation": "history_card/1"});
        assert_eq!(ask_ledger(&ledger, &key, &card).0.unwrap().checkpoints, 2);
    }

    /// A checkpoint whose line lost its COSE statement ends the chain, as
    /// any checkpoint this index cannot take does: what came before it is
    /// still answered and verifies, and nothing after it is served, so every
    /// list an answer carries extends checkpoint by checkpoint.
    #[test]
    fn a_checkpoint_without_its_statement_ends_the_chain() {
        let (_dir, ledger, key, _) = producer_ledger(6, &[1, 3, 5]);
        let path = ledger.join(crate::evidence_log::CHECKPOINTS_FILE);
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        let last = hex::encode(Sha256::digest(
            hex::decode(lines[2]["checkpoint_cose"].as_str().unwrap()).unwrap(),
        ));
        let mut stripped = lines.clone();
        stripped[1]
            .as_object_mut()
            .unwrap()
            .remove("checkpoint_cose");
        crate::evidence_request_parity::write_jsonl(&path, &stripped);
        let (anchors, total) = anchors_of(&ledger, &key);
        assert_eq!(total, 1, "the chain ends before the unstated checkpoint");

        let card = json!({"subject": {"full_history": null}, "coverage": {"expected_pin": anchors[0]}, "derivation": "history_card/1"});
        assert_eq!(ask_ledger(&ledger, &key, &card).0.unwrap().checkpoints, 1);
        let bytes = serde_json::to_vec(
            &json!({"subject": {"checkpoints": null}, "coverage": {"expected_pin": last}}),
        )
        .unwrap();
        let announced = announced_keys();
        let replay = crate::requester_auth::ReplayCache::default();
        let outcome = evidence_answer::answer_and_log(
            &bytes,
            false,
            &Responder {
                ledger_dir: &ledger,
                signing_key: &key,
                now: NOW,
                history_segments: "peers",
                share_standard_history: true,
                keys: crate::requester_auth::Keys {
                    announced: &announced,
                    configured: None,
                },
                replay: &replay,
                node_id: Some(RESPONDER_NODE),
            },
            None,
        );
        assert!(
            matches!(
                outcome,
                Outcome::Refused {
                    reason: Reason::CoverageUnsatisfiable,
                    ..
                }
            ),
            "{outcome:?}"
        );
    }

    /// A leaf that is itself a peak cannot be passed off at another
    /// position: in a three-leaf MMR, leaf 2 is the right peak, and an empty
    /// path from leaf 0's position must not reach it.
    #[test]
    fn an_inclusion_proof_proves_its_leafs_position() {
        use cll::mmr::{add_leaf, peaks, MemoryNodeStore, NodeReader};
        let entries = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let mut nodes = MemoryNodeStore::new();
        for e in &entries {
            add_leaf(&mut nodes, leaf_hash(e)).unwrap();
        }
        let size = nodes.size();
        let peak_hashes: Vec<Hash> = peaks(size)
            .unwrap()
            .iter()
            .map(|&p| nodes.node(p))
            .collect();
        let proof = |leaf: u64| {
            let path = cll::mmr::inclusion_proof(&nodes, leaf, size)
                .unwrap()
                .witness;
            let path: Vec<Hash> = path
                .iter()
                .map(|h| hex::decode(h).unwrap().try_into().unwrap())
                .collect();
            encode_inclusion_proof(leaf, &path).unwrap()
        };
        for (leaf, entry) in entries.iter().enumerate() {
            assert!(
                proves(&proof(leaf as u64), leaf as u64, entry, size, &peak_hashes),
                "{leaf}"
            );
        }
        let forged = encode_inclusion_proof(0, &[]).unwrap();
        assert!(!proves(&forged, 0, &entries[2], size, &peak_hashes));
        assert!(!proves(&proof(2), 2, &entries[2], size, &peak_hashes[..1]));
    }

    /// `included_root` stays inside the profile's integers on a hostile
    /// path, and an index that is not a leaf proves nothing.
    #[test]
    fn a_hostile_proof_is_refused_without_panicking() {
        assert!(included_root(MAX_NODE_INDEX, [0; 32], &[[1; 32]; 64]).is_none());
        let entry = [3u8; 32];
        let mut proof = Vec::new();
        ciborium::into_writer(
            &ciborium::Value::Array(vec![
                ciborium::Value::Integer(2.into()),
                ciborium::Value::Array(vec![]),
            ]),
            &mut proof,
        )
        .unwrap();
        assert!(!proves(&proof, 1, &entry, 3, &[leaf_hash(&entry)]));
        assert!(inclusion_parts(b"not cbor").is_none());
        let ledger = corpus_ledger();
        let cps = crate::evidence_request_parity::corpus_checkpoints(&ledger, &key());
        let decoded = |i: usize| {
            verify_checkpoint_cose_offline(&cps[i].statement)
                .decoded
                .unwrap()
        };
        let (a, b) = (decoded(0), decoded(1));
        assert!(!consistent(
            a.mmr_size,
            &a.new_peak_hashes,
            b.mmr_size,
            &b.new_peak_hashes,
            b"\x84\x00"
        ));
    }
}
