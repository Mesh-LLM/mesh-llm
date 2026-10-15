//! Receive a peer's pushed record, in-process: the checks of the Python
//! reference receiver (`record_push.py`), in the same order, with the same
//! answers.
//!
//! **What arrives.** A peer pushes its own sealed half of an exchange, either
//! bare or as a bundle (`{"record_push_bundle": 1, "capsule", "inclusion",
//! "checkpoint"}`) that also proves the half is in the peer's log under a
//! checkpoint it signed. A split's coordinator may add `split_stage_records`.
//! This is plugin-internal behaviour, not a standardized wire, and may change.
//!
//! **The order of the checks** (each refusal is signed by this node):
//! 1. The bytes parse strictly ([`crate::strict_json`]), else
//!    `request_malformed`.
//! 2. A bundle carries exactly its members with exactly their types, else
//!    `bundle_malformed`. From here on, a bundle's structural refusals are
//!    `bundle_malformed` and a bare push's are `request_malformed`: that
//!    reason in reply to a bundle is how a sender recognises a receiver that
//!    predates bundles.
//! 3. `record_at_completion: off` refuses `policy_decline`.
//! 4. A push with no sender id, or from a sender with no announced key
//!    ([`crate::peer_keys`]), is refused `signature_unverified`. These cheap
//!    refusals come before any work on the record's bytes: an unknown sender
//!    is refused before its record is checked.
//! 5. The half passes the Class 1 checks on its own bytes, as the reference
//!    verifier runs them (capsule-emit's `structure`, including that its
//!    `capsule_id` recomputes), else malformed.
//! 6. The announced key is the half's `key_id` and the half's signature
//!    verifies under it, else `signature_unverified`.
//! 7. What a served half claims about the exchange agrees with this node's
//!    own record of it, else `served_by_mismatch` / `model_mismatch`.
//! 8. A split's carried stage records are each valid and named by the main
//!    record's receipt, else `bundle_malformed`.
//! 9. A bundle's checkpoint is signed by the same announced key and its proof
//!    puts the half under that checkpoint's root, else `inclusion_unverified`;
//!    and it does not contradict a checkpoint already held for the same key
//!    and log (`checkpoint_equivocation`) or come from before the newest one
//!    at a size never vouched for (`checkpoint_stale`).
//!
//! Every refusal from step 4 on is recorded in `rejected-record-pushes.jsonl`,
//! up to [`MAX_REJECTED_LOG_BYTES`]; past that, the refusals still go out,
//! signed, and the ones not recorded are counted and logged. A refusal
//! issued before the record's signature verified is recorded only while the
//! log is under a sixteenth of that ([`UNAUTHENTICATED_LOG_SHARE`]), so
//! anyone's flood leaves room for the refusals of authenticated pushes.
//!
//! **What is kept.** A received half is evidence this node holds, not a
//! record it made: it goes to the held-artifact store
//! `received-capsules.jsonl`, as transmitted, with a provenance line; a
//! bundle's proof and checkpoint go to `received-inclusion.jsonl`. None of it
//! enters this node's own chain. The chained record of the receipt is the
//! citing record the bridge seals after this returns. A record already held
//! from the same sender is not stored again: a resent push is acknowledged
//! and adds nothing to disk. A line of a held store that does not parse (a
//! write torn by a crash) is skipped, counted and logged; the next line
//! written after such a fragment starts on its own line.
//!
//! **Bounded work.** The body is at most `MAX_SIDE_STREAM_BYTES` before it
//! gets here. Every size a peer states is checked before anything is sized by
//! it: a proof is verified by `cll`, which refuses a tree of 2^50 nodes or
//! more before it builds a path, and a split carries at most
//! [`MAX_SPLIT_STAGE_RECORDS`] records, counted before any is read.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::strict_json;

/// The top-level member that marks a push body as a bundle, and the one
/// bundle version this receiver reads.
pub const BUNDLE_MARKER: &str = "record_push_bundle";
pub const BUNDLE_VERSION: u64 = 1;

/// The bundle member a split's coordinator carries its stage records in.
pub const SPLIT_STAGE_RECORDS: &str = "split_stage_records";
/// More stage records than this in one bundle is refused.
pub const MAX_SPLIT_STAGE_RECORDS: usize = 64;

pub const REASON_REQUEST_MALFORMED: &str = "request_malformed";
pub const REASON_BUNDLE_MALFORMED: &str = "bundle_malformed";
pub const REASON_POLICY_DECLINE: &str = "policy_decline";
pub const REASON_SIGNATURE_UNVERIFIED: &str = "signature_unverified";
pub const REASON_SERVED_BY_MISMATCH: &str = "served_by_mismatch";
pub const REASON_MODEL_MISMATCH: &str = "model_mismatch";
pub const REASON_INCLUSION_UNVERIFIED: &str = "inclusion_unverified";
pub const REASON_CHECKPOINT_STALE: &str = "checkpoint_stale";
pub const REASON_CHECKPOINT_EQUIVOCATION: &str = "checkpoint_equivocation";

/// The held-artifact stores, beside `capsules.jsonl`. Nothing chains or
/// checkpoints them.
pub const RECEIVED_CAPSULES_FILENAME: &str = "received-capsules.jsonl";
pub const RECEIVED_PROVENANCE_FILENAME: &str = "received-provenance.jsonl";
pub const RECEIVED_INCLUSION_FILENAME: &str = "received-inclusion.jsonl";
pub const RECEIVED_SPLIT_STAGE_FILENAME: &str = "received-split-stage-records.jsonl";
pub const REJECTED_PUSHES_FILENAME: &str = "rejected-record-pushes.jsonl";
pub const CHECKPOINT_EQUIVOCATIONS_FILENAME: &str = "checkpoint-equivocations.jsonl";

const PROOF_FIELDS: [&str; 7] = [
    "v",
    "kind",
    "size",
    "leaf_index",
    "witness",
    "peaks_left",
    "peaks_right",
];
const CHECKPOINT_FIELDS: [&str; 10] = [
    "v",
    "kind",
    "log_id",
    "mmr_size",
    "root",
    "prev_size",
    "prev_root",
    "key_id",
    "timestamp",
    "signature",
];

static RECEIVING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// This node, as the receiver sees it.
pub struct Receiver<'a> {
    /// Where `capsules.jsonl` and the held-artifact stores live.
    pub ledger_dir: &'a Path,
    /// This node's key: it signs every refusal.
    pub signing_key: &'a SigningKey,
    /// The raw `CAPSULES_PEER_KEYS` value (`None` when unset).
    pub peer_keys: Option<&'a str>,
    /// `record_at_completion: off`: receive nothing.
    pub record_at_completion_off: bool,
    /// The most `rejected-record-pushes.jsonl` may grow to, in bytes
    /// ([`MAX_REJECTED_LOG_BYTES`] in the plugin). Past it, rejections are
    /// counted, and the count logged, not written.
    pub rejected_log_limit: u64,
}

/// How large the rejected-push log may grow: 4 MiB, some tens of thousands
/// of rejections. A peer that keeps pushing what this node refuses cannot
/// fill the disk through it; past the cap the refusals still go out, signed,
/// and only the local record of them stops.
pub const MAX_REJECTED_LOG_BYTES: u64 = 4 * 1024 * 1024;

/// Refusals issued before a record's signature has verified are logged only
/// while the log is under this fraction of its cap (a sixteenth: 256 KiB of
/// the 4 MiB), so an unauthenticated flood leaves the rest for refusals of
/// authenticated pushes.
pub const UNAUTHENTICATED_LOG_SHARE: u64 = 16;

static REJECTED_LOG_DROPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TORN_LINES_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Rejections not written because the rejected-push log is at its cap,
/// since this process started.
#[cfg(test)]
pub fn rejected_log_drops() -> u64 {
    REJECTED_LOG_DROPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Lines of a held store that did not parse (a torn write) and were skipped,
/// since this process started.
#[cfg(test)]
pub fn torn_lines_skipped() -> u64 {
    TORN_LINES_SKIPPED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Receive one push body from `sender` (its self-declared peer id, `None`
/// when it declared none), at `now` (RFC 3339 UTC). The reply is
/// `{"status": "received"}`, the same with the verified `inclusion` facts for
/// a bundle, or a signed refusal. Never panics on any body; an `Err` is a
/// local write that failed, and the caller must not acknowledge the push.
pub fn receive(
    node: &Receiver<'_>,
    body: &[u8],
    sender: Option<&str>,
    now: &str,
) -> std::io::Result<Value> {
    // One push at a time: the held-checkpoint judgement (step 8) reads what
    // is stored and then stores more, and two pushes interleaving there could
    // each miss the other's checkpoint.
    let _one_at_a_time = RECEIVING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let request_digest = hex::encode(Sha256::digest(body));
    let refuse = |reason: &str| refusal(node.signing_key, &request_digest, reason, now);

    let Ok(parsed) = strict_json::parse(body) else {
        return Ok(refuse(REASON_REQUEST_MALFORMED));
    };

    // A referee's verdict, delivered by a courier (`crate::referee::hold`):
    // the same policy gate and sender requirement as a pushed half.
    if parsed.get(crate::referee::hold::DELIVERY_MARKER).is_some() && parsed.is_object() {
        if node.record_at_completion_off {
            return Ok(refuse(REASON_POLICY_DECLINE));
        }
        let Some(sender) = sender.filter(|s| !s.is_empty()) else {
            return Ok(refuse(REASON_SIGNATURE_UNVERIFIED));
        };
        let own_key_id = hex::encode(node.signing_key.verifying_key().to_bytes());
        let door = crate::referee::hold::Node {
            ledger_dir: node.ledger_dir,
            peer_keys: node.peer_keys,
            own_key_id: &own_key_id,
        };
        return match crate::referee::hold::hold_delivered_verdict(&door, &parsed, sender, now)? {
            Ok(reply) => Ok(reply),
            Err(reason) => {
                let entry = json!({
                    "capsule_id": null,
                    "claimed_sender_peer_id": sender,
                    "reason": reason,
                    "rejected_at": now,
                });
                // Refused before, or because, the verdict's signature is
                // checked: the small share of the log, like any other
                // unauthenticated refusal.
                let limit = node.rejected_log_limit / UNAUTHENTICATED_LOG_SHARE;
                append_capped(node.ledger_dir, REJECTED_PUSHES_FILENAME, &entry, limit)?;
                Ok(refuse(reason))
            }
        };
    }

    let (half, bundle, malformed) = match parsed.get(BUNDLE_MARKER) {
        Some(_) if parsed.is_object() => {
            if !bundle_exact(&parsed) {
                return Ok(refuse(REASON_BUNDLE_MALFORMED));
            }
            (&parsed["capsule"], Some(&parsed), REASON_BUNDLE_MALFORMED)
        }
        _ => (&parsed, None, REASON_REQUEST_MALFORMED),
    };

    // The cheap refusals come before any work on the record's bytes: this
    // node's policy, then a sender with no announced key.
    if node.record_at_completion_off {
        return Ok(refuse(REASON_POLICY_DECLINE));
    }

    // The id the half claims (not yet checked) until it is verified below.
    let mut logged_id = half.get("capsule_id").and_then(Value::as_str);
    // A refusal issued before the record's signature has verified (no sender,
    // a sender with no announced key, or a key or signature that does not
    // match) may use only a small share of the log: anyone can fill that
    // share, but never crowd out the refusals of authenticated pushes.
    let reject = |logged_id: Option<&str>,
                  reason: &str,
                  exchange: Option<&str>,
                  authenticated: bool|
     -> std::io::Result<Value> {
        let mut entry = json!({
            "capsule_id": logged_id,
            "claimed_sender_peer_id": sender,
            "reason": reason,
            "rejected_at": now,
        });
        if let Some(digest) = exchange {
            entry["request_digest"] = json!(digest);
        }
        let limit = if authenticated {
            node.rejected_log_limit
        } else {
            node.rejected_log_limit / UNAUTHENTICATED_LOG_SHARE
        };
        append_capped(node.ledger_dir, REJECTED_PUSHES_FILENAME, &entry, limit)?;
        Ok(refuse(reason))
    };

    let Some(sender) = sender.filter(|s| !s.is_empty()) else {
        return reject(logged_id, REASON_SIGNATURE_UNVERIFIED, None, false);
    };
    let Some(announced) = crate::peer_keys::announced_key_in(node.peer_keys, sender) else {
        return reject(logged_id, REASON_SIGNATURE_UNVERIFIED, None, false);
    };

    let Some(capsule_id) = record_id(half) else {
        return Ok(refuse(malformed));
    };
    logged_id = Some(capsule_id);
    if half.get("key_id").and_then(Value::as_str) != Some(announced.as_str())
        || !signed_by(half, &announced)
    {
        return reject(logged_id, REASON_SIGNATURE_UNVERIFIED, None, false);
    }
    // From here on the record is signed by the sender's announced key.
    let reject = |reason: &str, exchange: Option<&str>| reject(logged_id, reason, exchange, true);

    if let Some(reason) = claims_verdict(node.ledger_dir, half, sender)? {
        let exchange = half
            .pointer("/effect/request_digest")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty());
        return reject(reason, exchange);
    }

    let mut split_records: &[Value] = &[];
    if let Some(records) = bundle.and_then(|b| b.get(SPLIT_STAGE_RECORDS)) {
        if !split_stage_records_ok(half, records) {
            return reject(REASON_BUNDLE_MALFORMED, None);
        }
        split_records = records.as_array().map(Vec::as_slice).unwrap_or_default();
    }

    let mut inclusion = None;
    if let Some(bundle) = bundle {
        let Some(verified) = verify_bundle_inclusion(capsule_id, &announced, bundle) else {
            return reject(REASON_INCLUSION_UNVERIFIED, None);
        };
        match history_verdict(&held_inclusions(node.ledger_dir)?, capsule_id, &verified) {
            History::Duplicate(row) => return Ok(received_reply(&row)),
            History::Equivocation(evidence) => {
                let mut evidence = evidence;
                evidence["received_from"] = json!(sender);
                evidence["detected_at"] = json!(now);
                append(
                    node.ledger_dir,
                    CHECKPOINT_EQUIVOCATIONS_FILENAME,
                    &evidence,
                )?;
                return reject(REASON_CHECKPOINT_EQUIVOCATION, None);
            }
            History::Stale => return reject(REASON_CHECKPOINT_STALE, None),
            History::Ok => {}
        }
        inclusion = Some(verified);
    }

    // A record already held from this sender is not stored again: a resent
    // push (a retry, or a replay) adds nothing to disk. A bundle's proof is
    // still judged and held on its own terms below.
    if !already_held(node.ledger_dir, capsule_id, sender)? {
        append(node.ledger_dir, RECEIVED_CAPSULES_FILENAME, half)?;
        append(
            node.ledger_dir,
            RECEIVED_PROVENANCE_FILENAME,
            &json!({
                "capsule_id": capsule_id,
                "received_from": sender,
                "via": "push",
                "received_at": now,
                "signature_ok": true,
            }),
        )?;
        for record in split_records {
            append(
                node.ledger_dir,
                RECEIVED_SPLIT_STAGE_FILENAME,
                &json!({"main_capsule_id": capsule_id, "received_at": now, "capsule": record}),
            )?;
        }
    }
    let Some(inclusion) = inclusion else {
        return Ok(json!({"status": "received"}));
    };
    let mut held = json!({
        "half_capsule_id": capsule_id,
        "received_from": sender,
        "via": "push",
        "received_at": now,
    });
    for (key, value) in inclusion {
        held[key] = value;
    }
    append(node.ledger_dir, RECEIVED_INCLUSION_FILENAME, &held)?;
    Ok(received_reply(&held))
}

/// A refusal of `body` for `reason`, signed by this node's key: for a push
/// this module does not take at all (a delivered verdict).
pub fn refuse(signing_key: &SigningKey, body: &[u8], reason: &str, now: &str) -> Value {
    refusal(signing_key, &hex::encode(Sha256::digest(body)), reason, now)
}

/// The success reply for a held bundle: the facts the bridge cites, never the
/// artifacts themselves.
fn received_reply(held: &Value) -> Value {
    let mut facts = Map::new();
    for key in [
        "half_capsule_id",
        "leaf_index",
        "mmr_size",
        "checkpoint_digest",
        "inclusion_proof_digest",
        "received_at",
    ] {
        facts.insert(key.into(), held.get(key).cloned().unwrap_or(Value::Null));
    }
    json!({"status": "received", "inclusion": facts})
}

/// A refusal signed by this node: Ed25519 over the RFC 8785 form of
/// `{issued_at, reason, request_digest}` (three ASCII strings, so sorted
/// compact JSON), `key_id` the raw public key. `policy_decline` also carries
/// the fabric's `status`.
fn refusal(key: &SigningKey, request_digest: &str, reason: &str, issued_at: &str) -> Value {
    let body = json!({"issued_at": issued_at, "reason": reason, "request_digest": request_digest});
    let signed = sorted_compact(&body);
    let mut reply = json!({
        "request_digest": request_digest,
        "reason": reason,
        "issued_at": issued_at,
        "key_id": hex::encode(key.verifying_key().to_bytes()),
        "sig": hex::encode(key.sign(signed.as_bytes()).to_bytes()),
    });
    if reason == REASON_POLICY_DECLINE {
        reply["status"] = json!("WITHHELD");
    }
    reply
}

/// Compact JSON with every object's keys sorted: RFC 8785 for values that
/// hold only ASCII strings, integers, and arrays and objects of them.
fn sorted_compact(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let ordered: BTreeMap<&String, Value> =
                    map.iter().map(|(k, v)| (k, sorted(v))).collect();
                Value::Object(ordered.into_iter().map(|(k, v)| (k.clone(), v)).collect())
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(value).to_string()
}

/// [`append`], unless the file would pass `limit` bytes: then the line is
/// counted as dropped instead, and the count logged (on the first drop and
/// every thousandth).
fn append_capped(
    ledger_dir: &Path,
    filename: &str,
    entry: &Value,
    limit: u64,
) -> std::io::Result<()> {
    let size = match std::fs::metadata(ledger_dir.join(filename)) {
        Ok(meta) => meta.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(e),
    };
    let line_len = u64::try_from(entry.to_string().len() + 1).unwrap_or(u64::MAX);
    if size.saturating_add(line_len) > limit {
        let dropped = REJECTED_LOG_DROPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if dropped == 1 || dropped.is_multiple_of(1000) {
            tracing::warn!(
                dropped,
                limit,
                "{filename} is at its cap; rejected pushes are counted, not logged"
            );
        }
        return Ok(());
    }
    append(ledger_dir, filename, entry)
}

/// Each line of a held store that parses as JSON. A line that does not (a
/// write torn by a crash) is skipped, counted and logged, never allowed to
/// fail every later push.
fn json_lines(ledger_dir: &Path, filename: &str) -> std::io::Result<Vec<Value>> {
    let file = match std::fs::File::open(ledger_dir.join(filename)) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut rows = Vec::new();
    for line in std::io::BufReader::new(file).split(b'\n') {
        let line = line?;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice(&line) {
            Ok(row) => rows.push(row),
            Err(error) => {
                TORN_LINES_SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(%error, "{filename}: skipped a line that does not parse");
            }
        }
    }
    Ok(rows)
}

/// Whether this sender's record `capsule_id` is held already.
fn already_held(ledger_dir: &Path, capsule_id: &str, sender: &str) -> std::io::Result<bool> {
    Ok(json_lines(ledger_dir, RECEIVED_PROVENANCE_FILENAME)?
        .iter()
        .any(|row| row["capsule_id"] == capsule_id && row["received_from"] == sender))
}

fn append(ledger_dir: &Path, filename: &str, entry: &Value) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    std::fs::create_dir_all(ledger_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(ledger_dir.join(filename))?;
    // A write torn by a crash leaves a fragment with no newline after it.
    // End it first, so this line starts on its own and the fragment stays
    // one line that `json_lines` skips, instead of swallowing this one.
    let mut line = String::new();
    if file.metadata()?.len() > 0 {
        let mut last = [0u8; 1];
        file.seek(SeekFrom::End(-1))?;
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            line.push('\n');
        }
    }
    line.push_str(&entry.to_string());
    line.push('\n');
    file.write_all(line.as_bytes())?;
    file.sync_data()
}

// ---------------------------------------------------------------------------
// The record itself
// ---------------------------------------------------------------------------

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The record's `capsule_id`, when `half` passes the Class 1 checks on its
/// own bytes (capsule-emit's `structure`: the reference verifier's gating
/// checks, including that the `capsule_id` recomputes).
fn record_id(half: &Value) -> Option<&str> {
    let carried = half.as_object()?.get("capsule_id")?.as_str()?;
    // The checks refuse a malformed id and any profile but format 4 with
    // JCS, and under that profile they recompute the id: passing them means
    // `carried` is the record's own id.
    (is_lower_hex(carried, 64) && capsule_emit_lib::structure::check_structure(half).ok())
        .then_some(carried)
}

/// Whether `half`'s producer envelope verifies and was made by `key_id`.
/// The caller has already required `half["key_id"] == key_id`.
fn signed_by(half: &Value, key_id: &str) -> bool {
    capsule_emit_lib::cose::verify_producer_envelope(half).is_ok_and(|signer| signer == key_id)
}

// ---------------------------------------------------------------------------
// Bundle shape
// ---------------------------------------------------------------------------

fn exact_members(value: &Value, members: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|m| m.len() == members.len() && members.iter().all(|k| m.contains_key(*k)))
}

/// An exact non-negative integer: never a bool, a float or a string.
fn count(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn hex_str(value: &Value, len: usize) -> bool {
    value.as_str().is_some_and(|s| is_lower_hex(s, len))
}

/// A bundle carries exactly its members, each with exactly its type and
/// canonical spelling: nothing unsigned rides along, and nothing is coerced.
fn bundle_exact(bundle: &Value) -> bool {
    let Some(map) = bundle.as_object() else {
        return false;
    };
    let top = ["record_push_bundle", "capsule", "inclusion", "checkpoint"];
    let expected = top.len() + usize::from(map.contains_key(SPLIT_STAGE_RECORDS));
    if map.len() != expected || !top.iter().all(|k| map.contains_key(*k)) {
        return false;
    }
    if count(&map[BUNDLE_MARKER]) != Some(BUNDLE_VERSION) {
        return false;
    }
    let inclusion = &map["inclusion"];
    if !exact_members(inclusion, &["leaf_index", "proof"]) {
        return false;
    }
    let proof = &inclusion["proof"];
    let cp = &map["checkpoint"];
    if !exact_members(proof, &PROOF_FIELDS) || !exact_members(cp, &CHECKPOINT_FIELDS) {
        return false;
    }
    let hashes_ok = ["witness", "peaks_left", "peaks_right"].iter().all(|k| {
        proof[*k]
            .as_array()
            .is_some_and(|hs| hs.iter().all(|h| hex_str(h, 64)))
    });
    let prev_ok = match count(&cp["prev_size"]) {
        Some(0) => cp["prev_root"].as_str() == Some(""),
        Some(_) => hex_str(&cp["prev_root"], 64),
        None => false,
    };
    count(&inclusion["leaf_index"]).is_some()
        && count(&proof["v"]) == Some(1)
        && proof["kind"].as_str() == Some("inclusion")
        && count(&proof["size"]).is_some()
        && count(&proof["leaf_index"]).is_some()
        && hashes_ok
        && count(&cp["v"]) == Some(1)
        && cp["kind"].as_str() == Some("mmr_checkpoint")
        && cp["log_id"].is_string()
        && count(&cp["mmr_size"]).is_some()
        && hex_str(&cp["root"], 64)
        && prev_ok
        && hex_str(&cp["key_id"], 64)
        && cp["timestamp"].is_string()
        && hex_str(&cp["signature"], 128)
}

// ---------------------------------------------------------------------------
// Inclusion and held checkpoints
// ---------------------------------------------------------------------------

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// A bundle's checkpoint and proof, once [`bundle_exact`] held: the checkpoint
/// is signed by the announced key (which is the half's own), and the proof
/// puts the half's `capsule_id` at `leaf_index` under the checkpoint's root.
/// Returns the facts to hold and cite, rebuilt from the parsed values.
fn verify_bundle_inclusion(
    capsule_id: &str,
    announced: &str,
    bundle: &Value,
) -> Option<Map<String, Value>> {
    let cp = &bundle["checkpoint"];
    let checkpoint = cll::checkpoint::CheckpointRecord {
        v: u32::try_from(count(&cp["v"])?).ok()?,
        kind: cp["kind"].as_str()?.to_string(),
        log_id: cp["log_id"].as_str()?.to_string(),
        mmr_size: count(&cp["mmr_size"])?,
        root: cp["root"].as_str()?.to_string(),
        prev_size: count(&cp["prev_size"])?,
        prev_root: cp["prev_root"].as_str()?.to_string(),
        key_id: cp["key_id"].as_str()?.to_string(),
        timestamp: cp["timestamp"].as_str()?.to_string(),
        signature: cp["signature"].as_str()?.to_string(),
        witnesses: Vec::new(),
    };
    if checkpoint.key_id != announced {
        return None;
    }
    let key_bytes: [u8; 32] = hex::decode(announced).ok()?.try_into().ok()?;
    let key = VerifyingKey::from_bytes(&key_bytes).ok()?;
    let signature_bytes: [u8; 64] = hex::decode(&checkpoint.signature).ok()?.try_into().ok()?;
    let signature = Signature::from_bytes(&signature_bytes);
    key.verify_strict(checkpoint.digest().as_bytes(), &signature)
        .ok()?;

    let p = &bundle["inclusion"]["proof"];
    let proof = cll::mmr::InclusionProof {
        v: u32::try_from(count(&p["v"])?).ok()?,
        kind: p["kind"].as_str()?.to_string(),
        size: count(&p["size"])?,
        leaf_index: count(&p["leaf_index"])?,
        witness: strings(&p["witness"]),
        peaks_left: strings(&p["peaks_left"]),
        peaks_right: strings(&p["peaks_right"]),
    };
    let leaf_index = count(&bundle["inclusion"]["leaf_index"])?;
    let root: [u8; 32] = hex::decode(&checkpoint.root).ok()?.try_into().ok()?;
    let body_digest: [u8; 32] = hex::decode(capsule_id).ok()?.try_into().ok()?;
    if !cll::mmr::verify_inclusion(&root, checkpoint.mmr_size, leaf_index, &body_digest, &proof) {
        return None;
    }

    let held_checkpoint = json!({
        "v": checkpoint.v, "kind": checkpoint.kind, "log_id": checkpoint.log_id,
        "mmr_size": checkpoint.mmr_size, "root": checkpoint.root, "prev_size": checkpoint.prev_size,
        "prev_root": checkpoint.prev_root, "key_id": checkpoint.key_id,
        "timestamp": checkpoint.timestamp, "signature": checkpoint.signature,
    });
    let held_proof = json!({
        "v": proof.v, "kind": proof.kind, "size": proof.size, "leaf_index": proof.leaf_index,
        "witness": proof.witness, "peaks_left": proof.peaks_left, "peaks_right": proof.peaks_right,
    });
    let mut facts = Map::new();
    facts.insert("leaf_index".into(), json!(leaf_index));
    facts.insert("mmr_size".into(), json!(checkpoint.mmr_size));
    facts.insert("checkpoint_digest".into(), json!(checkpoint.digest()));
    facts.insert(
        "inclusion_proof_digest".into(),
        json!(hex::encode(Sha256::digest(
            sorted_compact(&held_proof).as_bytes()
        ))),
    );
    facts.insert("checkpoint".into(), held_checkpoint);
    facts.insert("inclusion_proof".into(), held_proof);
    Some(facts)
}

/// The inclusion facts already held, one per line of `received-inclusion.jsonl`.
fn held_inclusions(ledger_dir: &Path) -> std::io::Result<Vec<Value>> {
    json_lines(ledger_dir, RECEIVED_INCLUSION_FILENAME)
}

enum History {
    /// This half under this checkpoint is held already: the held row.
    Duplicate(Value),
    /// The checkpoint, or its prev link, gives another root for a size a held
    /// checkpoint already signed: the evidence to keep.
    Equivocation(Value),
    /// Older than the newest held checkpoint, at a size none vouches for.
    Stale,
    Ok,
}

/// The (size, root) pairs a checkpoint signs: its own, and its prev link.
fn signed_roots(cp: &Value) -> Vec<(u64, String)> {
    let mut pairs = Vec::new();
    if let (Some(size), Some(root)) = (count(&cp["mmr_size"]), cp["root"].as_str()) {
        pairs.push((size, root.to_string()));
    }
    if let (Some(size), Some(root)) = (count(&cp["prev_size"]), cp["prev_root"].as_str()) {
        if size > 0 {
            pairs.push((size, root.to_string()));
        }
    }
    pairs
}

/// Judge a verified bundle's checkpoint against those held for the same
/// sender key and log. A burst sharing one checkpoint, and an older
/// checkpoint arriving after a newer one that links back to it, are fine.
fn history_verdict(
    held: &[Value],
    half_capsule_id: &str,
    verified: &Map<String, Value>,
) -> History {
    let new = &verified["checkpoint"];
    let same_log: Vec<&Value> = held
        .iter()
        .filter(|row| {
            row["checkpoint"]["key_id"] == new["key_id"]
                && row["checkpoint"]["log_id"] == new["log_id"]
        })
        .collect();
    if let Some(row) = same_log.iter().find(|row| {
        row["half_capsule_id"].as_str() == Some(half_capsule_id)
            && row["checkpoint_digest"] == verified["checkpoint_digest"]
    }) {
        return History::Duplicate((*row).clone());
    }

    // The first held checkpoint to sign a size is the one a contradiction is
    // reported against.
    let mut known: BTreeMap<u64, (String, &Value)> = BTreeMap::new();
    for row in &same_log {
        for (size, root) in signed_roots(&row["checkpoint"]) {
            known.entry(size).or_insert((root, &row["checkpoint"]));
        }
    }
    for (size, root) in signed_roots(new) {
        if let Some((held_root, held_checkpoint)) = known.get(&size) {
            if *held_root != root {
                return History::Equivocation(json!({
                    "key_id": new["key_id"],
                    "log_id": new["log_id"],
                    "mmr_size": size,
                    "held_root": held_root,
                    "claimed_root": root,
                    "held_checkpoint": held_checkpoint,
                    "claimed_checkpoint": new,
                }));
            }
        }
    }

    let newest = same_log
        .iter()
        .filter_map(|row| count(&row["checkpoint"]["mmr_size"]))
        .max()
        .unwrap_or(0);
    match count(&new["mmr_size"]) {
        Some(size) if size < newest && !known.contains_key(&size) => History::Stale,
        _ => History::Ok,
    }
}

// ---------------------------------------------------------------------------
// Claims on a served half
// ---------------------------------------------------------------------------

fn poc(record: &Value) -> Option<&Map<String, Value>> {
    record
        .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1")
        .and_then(Value::as_object)
}

/// The node a record names as having served it, if any.
fn named_server(record: &Value) -> Option<String> {
    let served_by = poc(record)?
        .get("serving_provenance")?
        .get("served_by_node_id")?
        .as_str()?
        .trim();
    (!served_by.is_empty() && served_by != "unknown").then(|| served_by.to_string())
}

/// Every weights file digest a record names: the producer's
/// `weights_digest.digest` and the host's
/// `serving_provenance.model.weights_digest`. Both are the SHA-256 of the
/// loaded file.
pub(crate) fn weights_claims(record: &Value) -> BTreeSet<String> {
    let ca = record.pointer("/model_attestation/compute_attestation");
    let mut candidates: Vec<String> = Vec::new();
    if let Some(d) = ca
        .and_then(|ca| ca.get("weights_digest"))
        .and_then(|w| w.get("digest"))
        .and_then(Value::as_str)
    {
        candidates.push(d.to_string());
    }
    if let Some(d) = poc(record)
        .and_then(|p| p.get("serving_provenance"))
        .and_then(|sp| sp.get("model"))
        .and_then(|m| m.get("weights_digest"))
        .and_then(Value::as_str)
    {
        candidates.push(d.to_string());
    }
    candidates
        .into_iter()
        .map(|c| c.trim().to_lowercase())
        .filter(|c| is_lower_hex(c, 64))
        .collect()
}

/// The `sha256-<hex>` / `sha256:<hex>` inside `model_attestation.model_id`.
/// mesh-llm names a GGUF model `local-gguf/sha256-<source identity>`. That
/// digest is the file's own SHA-256 only for a single local file; for a
/// Hugging Face GGUF it hashes the repository, revision and file list, and
/// for a split GGUF it hashes the parts. So it names the model, not the
/// weights file.
pub(crate) fn model_id_digest(record: &Value) -> Option<String> {
    let low = record
        .pointer("/model_attestation/model_id")
        .and_then(Value::as_str)?
        .to_lowercase();
    ["sha256-", "sha256:"].iter().find_map(|marker| {
        let at = low.find(marker)?;
        let hex: String = low[at + marker.len()..].chars().take(64).collect();
        is_lower_hex(&hex, 64).then_some(hex)
    })
}

/// Whether a served half names a different model than the requester's own
/// records of the exchange. Weights are held to weights and model ids to
/// model ids. Only a half that names no model id is held, by its weights, to
/// the id the requester asked for: the one thing left to compare.
pub(crate) fn names_other_model(half: &Value, own: &[&Value]) -> bool {
    let theirs = weights_claims(half);
    if theirs.len() > 1 {
        return true;
    }
    let mut asked_weights = BTreeSet::new();
    let mut asked_ids = BTreeSet::new();
    for record in own {
        asked_weights.extend(weights_claims(record));
        asked_ids.extend(model_id_digest(record));
    }
    let disagree = |asked: &BTreeSet<String>, claimed: &BTreeSet<String>| {
        !asked.is_empty() && !claimed.is_empty() && asked.is_disjoint(claimed)
    };
    if disagree(&asked_weights, &theirs) {
        return true;
    }
    match model_id_digest(half) {
        Some(id) => disagree(&asked_ids, &BTreeSet::from([id])),
        None => asked_weights.is_empty() && disagree(&asked_ids, &theirs),
    }
}

/// This node's own requester records of the exchange `request_digest`.
fn own_requested_records(ledger_dir: &Path, request_digest: &str) -> std::io::Result<Vec<Value>> {
    let file = match std::fs::File::open(ledger_dir.join("capsules.jsonl")) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut own = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(record) = serde_json::from_str::<Value>(&line?) else {
            continue;
        };
        let requested = poc(&record)
            .and_then(|p| p.get("role"))
            .and_then(Value::as_str)
            == Some("requested");
        if requested
            && record
                .pointer("/effect/request_digest")
                .and_then(Value::as_str)
                == Some(request_digest)
        {
            own.push(record);
        }
    }
    Ok(own)
}

/// The claim checks on a signature-verified served half. The signature only
/// proves who signed; this checks what the signer claims about the exchange
/// (who served it, which model) against this node's own record of it.
fn claims_verdict(
    ledger_dir: &Path,
    half: &Value,
    sender: &str,
) -> std::io::Result<Option<&'static str>> {
    if cfg!(feature = "mutant-record-push-skips-claims") {
        return Ok(None);
    }
    if poc(half)
        .and_then(|p| p.get("role"))
        .and_then(Value::as_str)
        != Some("served")
    {
        return Ok(None);
    }
    if named_server(half).is_some_and(|named| named != sender) {
        return Ok(Some(REASON_SERVED_BY_MISMATCH));
    }
    // No "our own record routed this elsewhere" refusal: the provider's push
    // can arrive before this node seals its own record, and one prompt sent
    // to several peers shares a digest.
    let own = match half
        .pointer("/effect/request_digest")
        .and_then(Value::as_str)
    {
        Some(digest) => own_requested_records(ledger_dir, digest)?,
        None => Vec::new(),
    };
    let own: Vec<&Value> = own
        .iter()
        .filter(|record| named_server(record).is_none_or(|named| named == sender))
        .collect();
    Ok(names_other_model(half, &own).then_some(REASON_MODEL_MISMATCH))
}

// ---------------------------------------------------------------------------
// A split's carried stage records
// ---------------------------------------------------------------------------

/// Every carried stage record verifies on its own, is a valid stage-side
/// block of the main record's split, and is one its receipt names, with no
/// record carried twice. The stage's key is not linked to a node here: the
/// requester never learned the stages' announced keys.
fn split_stage_records_ok(main: &Value, records: &Value) -> bool {
    use crate::producer::stage::{
        stage_block_of, CoordinatorReceipt, Side, COORDINATOR_RECEIPT_BLOCK,
    };

    let Some(records) = records.as_array() else {
        return false;
    };
    // Counted before any record is read.
    if records.is_empty() || records.len() > MAX_SPLIT_STAGE_RECORDS {
        return false;
    }
    let Some(receipt) = main
        .pointer("/model_attestation/compute_attestation")
        .and_then(|ca| ca.get(COORDINATOR_RECEIPT_BLOCK))
    else {
        return false;
    };
    let Ok(receipt) = CoordinatorReceipt::from_value(receipt) else {
        return false;
    };
    let named: BTreeSet<&str> = receipt
        .stages
        .iter()
        .flat_map(|stage| {
            stage
                .bundle_ref
                .iter()
                .chain(stage.bundle_refs.iter().flatten())
        })
        .map(|r| r.digest.as_str())
        .collect();
    let key = receipt.split_key();
    let mut seen = BTreeSet::new();
    for record in records {
        let Some(id) = record_id(record) else {
            return false;
        };
        if !named.contains(id) || !seen.insert(id) {
            return false;
        }
        if capsule_emit_lib::cose::verify_producer_envelope(record).is_err() {
            return false;
        }
        match stage_block_of(record) {
            Ok(Some(block)) if block.side == Side::Stage && block.split_key() == key => {}
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_push_parity::{body_bytes, parity_dir, read_json};

    const NOW: &str = "2026-09-29T00:00:00Z";

    fn corpus() -> Value {
        read_json(&parity_dir().join("corpus/record_push.json"))
    }

    fn case<'a>(corpus: &'a Value, name: &str) -> &'a Value {
        corpus["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|c| c["name"] == name))
            .unwrap_or_else(|| panic!("no corpus case {name}"))
    }

    /// Every body in the corpus, cut short at many points, with single bits
    /// flipped, and wrapped as a bundle's half: the receiver answers every
    /// one (received or a signed refusal) and never panics.
    #[test]
    fn no_body_derived_from_the_corpus_makes_the_receiver_panic() {
        let corpus = corpus();
        let dir = tempfile::tempdir().unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        let mut answered = 0usize;
        for case in corpus["cases"].as_array().unwrap() {
            let receiver = Receiver {
                ledger_dir: dir.path(),
                signing_key: &key,
                peer_keys: case["node"]["peer_keys_env"].as_str(),
                record_at_completion_off: false,
                rejected_log_limit: MAX_REJECTED_LOG_BYTES,
            };
            for push in case["pushes"].as_array().unwrap() {
                let body = body_bytes(push);
                let mut variants: Vec<Vec<u8>> = (0..body.len())
                    .step_by(97)
                    .map(|cut| body[..cut].to_vec())
                    .collect();
                for at in (0..body.len()).step_by(89) {
                    let mut flipped = body.clone();
                    flipped[at] ^= 0x20;
                    variants.push(flipped);
                }
                if let Ok(half) = serde_json::from_slice::<Value>(&body) {
                    variants.push(json!({"record_push_bundle": 1, "capsule": half, "inclusion": {}, "checkpoint": {}}).to_string().into_bytes());
                }
                for variant in variants {
                    let reply = receive(&receiver, &variant, push["sender"].as_str(), NOW)
                        .expect("local writes succeed");
                    assert!(
                        reply["status"] == "received"
                            || reply["sig"].as_str().is_some_and(|s| s.len() == 128),
                        "every answer is a receipt or a signed refusal: {reply}"
                    );
                    answered += 1;
                }
            }
        }
        assert!(answered > 5_000, "the sweep covered {answered} bodies");
    }

    /// A push that verifies but cannot be stored is an error, never a
    /// success reply: the bridge then acks nothing and the peer retries.
    #[test]
    fn a_verified_push_that_cannot_be_stored_is_not_received() {
        let corpus = corpus();
        let valid = case(&corpus, "bundle_valid");
        let dir = tempfile::tempdir().unwrap();
        let not_a_directory = dir.path().join("ledger");
        std::fs::write(
            &not_a_directory,
            b"a file where the ledger directory should be",
        )
        .unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        let receiver = Receiver {
            ledger_dir: &not_a_directory,
            signing_key: &key,
            peer_keys: valid["node"]["peer_keys_env"].as_str(),
            record_at_completion_off: false,
            rejected_log_limit: MAX_REJECTED_LOG_BYTES,
        };
        let push = &valid["pushes"][0];
        assert!(receive(&receiver, &body_bytes(push), push["sender"].as_str(), NOW).is_err());
    }

    /// The receiver holds nothing it refused: after every refused push in the
    /// corpus, no held-artifact store has grown.
    #[test]
    fn a_refused_push_holds_nothing() {
        let corpus = corpus();
        let key = SigningKey::from_bytes(&[9; 32]);
        for case in corpus["cases"].as_array().unwrap() {
            let dir = tempfile::tempdir().unwrap();
            let receiver = Receiver {
                ledger_dir: dir.path(),
                signing_key: &key,
                peer_keys: case["node"]["peer_keys_env"].as_str(),
                record_at_completion_off: case["node"]["record_at_completion"] == "off",
                rejected_log_limit: MAX_REJECTED_LOG_BYTES,
            };
            for push in case["pushes"].as_array().unwrap() {
                let held = |name: &str| {
                    std::fs::read_to_string(dir.path().join(name))
                        .map(|t| t.lines().count())
                        .unwrap_or(0)
                };
                let before: Vec<usize> = [
                    RECEIVED_CAPSULES_FILENAME,
                    RECEIVED_PROVENANCE_FILENAME,
                    RECEIVED_INCLUSION_FILENAME,
                    RECEIVED_SPLIT_STAGE_FILENAME,
                ]
                .iter()
                .map(|n| held(n))
                .collect();
                let reply =
                    receive(&receiver, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
                if reply.get("reason").is_some() {
                    let after: Vec<usize> = [
                        RECEIVED_CAPSULES_FILENAME,
                        RECEIVED_PROVENANCE_FILENAME,
                        RECEIVED_INCLUSION_FILENAME,
                        RECEIVED_SPLIT_STAGE_FILENAME,
                    ]
                    .iter()
                    .map(|n| held(n))
                    .collect();
                    assert_eq!(before, after, "{} held something it refused", case["name"]);
                }
            }
        }
    }

    fn receiver<'a>(
        dir: &'a Path,
        key: &'a SigningKey,
        case: &'a Value,
        limit: u64,
    ) -> Receiver<'a> {
        Receiver {
            ledger_dir: dir,
            signing_key: key,
            peer_keys: case["node"]["peer_keys_env"].as_str(),
            record_at_completion_off: false,
            rejected_log_limit: limit,
        }
    }

    fn lines(dir: &Path, name: &str) -> usize {
        std::fs::read_to_string(dir.join(name))
            .map(|t| t.lines().count())
            .unwrap_or(0)
    }

    /// A resent genuine record (a retry, or a replay) is acknowledged every
    /// time and stored once.
    #[test]
    fn a_hundred_resends_of_one_record_hold_it_once() {
        let corpus = corpus();
        let key = SigningKey::from_bytes(&[9; 32]);
        for name in ["bare_valid", "bundle_valid"] {
            let valid = case(&corpus, name);
            let dir = tempfile::tempdir().unwrap();
            let node = receiver(dir.path(), &key, valid, MAX_REJECTED_LOG_BYTES);
            let push = &valid["pushes"][0];
            for _ in 0..100 {
                let reply =
                    receive(&node, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
                assert_eq!(reply["status"], "received", "{name}");
            }
            for store in [RECEIVED_CAPSULES_FILENAME, RECEIVED_PROVENANCE_FILENAME] {
                assert_eq!(lines(dir.path(), store), 1, "{name}: {store}");
            }
            let inclusions = if name == "bundle_valid" { 1 } else { 0 };
            assert_eq!(
                lines(dir.path(), RECEIVED_INCLUSION_FILENAME),
                inclusions,
                "{name}"
            );
        }
    }

    /// The rejected-push log stops growing at its cap; the refusals still go
    /// out, signed, and the drops are counted.
    #[test]
    fn the_rejected_log_stops_at_its_cap_and_counts_what_it_drops() {
        let corpus = corpus();
        let unknown = case(&corpus, "bare_unknown_sender");
        let dir = tempfile::tempdir().unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        // An unknown sender's refusals get a sixteenth of the cap: 2,000 bytes.
        let limit = 32_000;
        let node = receiver(dir.path(), &key, unknown, limit);
        let push = &unknown["pushes"][0];
        let before = rejected_log_drops();
        for _ in 0..200 {
            let reply = receive(&node, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
            assert_eq!(reply["reason"], REASON_SIGNATURE_UNVERIFIED);
            assert!(reply["sig"].as_str().is_some_and(|s| s.len() == 128));
        }
        let size = std::fs::metadata(dir.path().join(REJECTED_PUSHES_FILENAME))
            .unwrap()
            .len();
        assert!(size <= limit / UNAUTHENTICATED_LOG_SHARE, "{size} bytes");
        let written = lines(dir.path(), REJECTED_PUSHES_FILENAME);
        assert!(written > 0 && written < 200);
        assert!(rejected_log_drops() - before >= u64::try_from(200 - written).unwrap());
    }

    /// One torn line in a held store (a crash mid-write) is skipped and
    /// counted; it does not fail later pushes.
    #[test]
    fn a_torn_line_in_the_held_stores_is_skipped_not_fatal() {
        let corpus = corpus();
        let valid = case(&corpus, "bundle_valid");
        let dir = tempfile::tempdir().unwrap();
        // A torn write: a fragment with NO newline after it.
        for store in [RECEIVED_INCLUSION_FILENAME, RECEIVED_PROVENANCE_FILENAME] {
            std::fs::write(dir.path().join(store), b"{\"half_capsule_id\": \"ab").unwrap();
        }
        let key = SigningKey::from_bytes(&[9; 32]);
        let node = receiver(dir.path(), &key, valid, MAX_REJECTED_LOG_BYTES);
        let push = &valid["pushes"][0];
        let before = torn_lines_skipped();
        let reply = receive(&node, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
        assert_eq!(reply["status"], "received");
        assert!(reply["inclusion"].is_object());
        assert!(torn_lines_skipped() - before >= 2);
        // The genuine lines written after the fragment survive it: each store
        // now holds the fragment and one whole line that parses.
        for store in [RECEIVED_INCLUSION_FILENAME, RECEIVED_PROVENANCE_FILENAME] {
            let rows = json_lines(dir.path(), store).unwrap();
            assert_eq!(rows.len(), 1, "{store}");
        }
        assert_eq!(
            held_inclusions(dir.path()).unwrap()[0]["half_capsule_id"],
            reply["inclusion"]["half_capsule_id"]
        );
        // And they are read back as held: a resend is a replay, stored once.
        let again = receive(&node, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
        assert_eq!(again, reply);
        assert_eq!(
            json_lines(dir.path(), RECEIVED_PROVENANCE_FILENAME)
                .unwrap()
                .len(),
            1
        );
    }

    /// Anyone can send unauthenticated pushes; their refusals use only a
    /// small share of the log, and never crowd out an authenticated one.
    #[test]
    fn an_unauthenticated_flood_cannot_crowd_out_an_authenticated_refusal() {
        let corpus = corpus();
        let unknown = case(&corpus, "bare_unknown_sender");
        let lie = case(&corpus, "claims_served_by_other");
        let dir = tempfile::tempdir().unwrap();
        let own: String = lie["node"]["own_records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.to_string() + "\n")
            .collect();
        std::fs::write(dir.path().join("capsules.jsonl"), own).unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        let limit = 32_000;
        let flood = receiver(dir.path(), &key, unknown, limit);
        let push = &unknown["pushes"][0];
        for _ in 0..2_000 {
            receive(&flood, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
        }
        let size = std::fs::metadata(dir.path().join(REJECTED_PUSHES_FILENAME))
            .unwrap()
            .len();
        assert!(
            size <= limit / UNAUTHENTICATED_LOG_SHARE,
            "the flood kept to its share: {size} bytes"
        );
        let known = receiver(dir.path(), &key, lie, limit);
        let push = &lie["pushes"][0];
        let reply = receive(&known, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
        assert_eq!(reply["reason"], REASON_SERVED_BY_MISMATCH);
        let logged = json_lines(dir.path(), REJECTED_PUSHES_FILENAME).unwrap();
        assert_eq!(logged.last().unwrap()["reason"], REASON_SERVED_BY_MISMATCH);
    }

    /// An unknown sender is refused before its record's bytes are checked:
    /// even a record that would fail every check gets the identity refusal.
    #[test]
    fn an_unknown_sender_is_refused_before_its_bytes_are_checked() {
        let corpus = corpus();
        let malformed = case(&corpus, "structure_approver");
        let dir = tempfile::tempdir().unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        let node = receiver(dir.path(), &key, malformed, MAX_REJECTED_LOG_BYTES);
        let push = &malformed["pushes"][0];
        let known = receive(&node, &body_bytes(push), push["sender"].as_str(), NOW).unwrap();
        assert_eq!(
            known["reason"], REASON_REQUEST_MALFORMED,
            "a known sender reaches the checks"
        );
        let unknown = receive(&node, &body_bytes(push), Some("nobody"), NOW).unwrap();
        assert_eq!(unknown["reason"], REASON_SIGNATURE_UNVERIFIED);
    }

    /// A delivered verdict goes to the referee's door, not the record
    /// checks: a malformed one is refused, signed, and logged without an id.
    #[test]
    fn a_malformed_delivered_verdict_is_refused_signed() {
        let dir = tempfile::tempdir().unwrap();
        let key = SigningKey::from_bytes(&[9; 32]);
        let node = Receiver {
            ledger_dir: dir.path(),
            signing_key: &key,
            peer_keys: None,
            record_at_completion_off: false,
            rejected_log_limit: MAX_REJECTED_LOG_BYTES,
        };
        let body = br#"{"adjudication_delivery": 1, "verdict_capsule": {}, "extra": 1}"#;
        let reply = receive(&node, body, Some("node-c"), NOW).unwrap();
        assert_eq!(reply["reason"], REASON_REQUEST_MALFORMED);
        assert!(reply.get("status").is_none(), "never a receipt");
        assert_eq!(reply["request_digest"], hex::encode(Sha256::digest(body)));
        let signed = sorted_compact(&json!({
            "issued_at": NOW, "reason": REASON_REQUEST_MALFORMED, "request_digest": reply["request_digest"],
        }));
        let sig: [u8; 64] = hex::decode(reply["sig"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        key.verifying_key()
            .verify_strict(signed.as_bytes(), &Signature::from_bytes(&sig))
            .expect("signed by this node");
        let logged = std::fs::read_to_string(dir.path().join(REJECTED_PUSHES_FILENAME)).unwrap();
        assert!(logged.contains("\"capsule_id\":null"));
    }

    // A half and the requester's record of one exchange, with the model id
    // and weights fields each names. The u844 live run's shape: a Hugging
    // Face GGUF named `local-gguf/sha256-<identity>`, whose host and
    // producer both name the file's own SHA-256.
    const HF_ID: &str =
        "local-gguf/sha256-cd3f391dbff00e71a7ee53da8deb2510119fc16ab703f310c301c3a60357a7a9";
    const Q8_FILE: &str = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
    const OTHER_FILE: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn requested(model_id: &str) -> Value {
        json!({
            "effect": {"request_digest": "b".repeat(64)},
            "model_attestation": {
                "model_id": model_id,
                "compute_attestation": {"x-mesh-poc-v1": {"role": "requested"}}
            }
        })
    }

    fn served(model_id: Option<&str>, producer: &str, host: &str) -> Value {
        let mut half = json!({
            "effect": {"request_digest": "b".repeat(64)},
            "model_attestation": {"compute_attestation": {
                "weights_digest": {"digest_alg": "SHA-256", "digest": producer, "scope": "file"},
                "x-mesh-poc-v1": {
                    "role": "served",
                    "serving_provenance": {"served_by_node_id": "node-a", "model": {"weights_digest": host}}
                }
            }}
        });
        if let Some(id) = model_id {
            half["model_attestation"]["model_id"] = json!(id);
        }
        half
    }

    /// MUTANT: count the model id's digest as a weights claim again and this
    /// fails: the half then names two weights and is refused.
    #[test]
    fn a_hugging_face_half_naming_the_asked_model_id_is_received() {
        let half = served(Some(HF_ID), Q8_FILE, Q8_FILE);
        assert!(!names_other_model(&half, &[&requested(HF_ID)]));
        assert!(
            !names_other_model(&half, &[]),
            "before our record is sealed"
        );

        let dir = tempfile::tempdir().unwrap();
        let line = serde_json::to_string(&requested(HF_ID)).unwrap();
        std::fs::write(dir.path().join("capsules.jsonl"), line + "\n").unwrap();
        assert_eq!(claims_verdict(dir.path(), &half, "node-a").unwrap(), None);
    }

    #[test]
    fn a_half_naming_another_model_id_than_asked_is_refused() {
        let other_id = format!("local-gguf/sha256-{OTHER_FILE}");
        let half = served(Some(&other_id), Q8_FILE, Q8_FILE);
        assert!(names_other_model(&half, &[&requested(HF_ID)]));
    }

    /// Attack D, case 1: one weights field changed.
    #[test]
    fn a_half_whose_weights_fields_disagree_is_refused() {
        let half = served(Some(HF_ID), Q8_FILE, OTHER_FILE);
        assert!(names_other_model(&half, &[&requested(HF_ID)]));
        assert!(names_other_model(&half, &[]));
    }

    /// A half that names no model id is held by its weights to the digest in
    /// the id we asked for (a single local GGUF names its file).
    #[test]
    fn a_half_naming_no_model_id_is_held_to_the_asked_id_digest() {
        let local_id = format!("local-gguf/sha256-{Q8_FILE}");
        assert!(!names_other_model(
            &served(None, Q8_FILE, Q8_FILE),
            &[&requested(&local_id)]
        ));
        assert!(names_other_model(
            &served(None, OTHER_FILE, OTHER_FILE),
            &[&requested(&local_id)]
        ));
    }

    #[test]
    fn a_model_id_without_a_full_digest_names_none() {
        assert_eq!(
            model_id_digest(&requested("local-gguf/sha256-4ff195f73917d9c2")),
            None
        );
        assert_eq!(model_id_digest(&requested("qwen2.5-0.5b-instruct")), None);
        assert_eq!(
            model_id_digest(&requested(&HF_ID.to_uppercase())).as_deref(),
            Some(&HF_ID["local-gguf/sha256-".len()..])
        );
    }
}
