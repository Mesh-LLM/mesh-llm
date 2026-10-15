//! The referee's side: a node asked to settle two twins checks what it can
//! check itself, decides, and signs the verdict with its own key.
//!
//! The requester sends, over the `evidence-request/1` carrier (plugin-internal,
//! not a -00 subject form):
//!
//! ```text
//! {"subject": {"kind": "adjudicate"},
//!  "selection_tier": 1 | 2,
//!  "twin_bracket_id": "<the host's bracket id>"        (optional),
//!  "halves": [{"capsule", "request_body", "response_body"}, <the other twin's>],
//!  "referee_answer": {"request_body", "response_body"}  (when the twins differ)}
//! ```
//!
//! Before it signs anything, this node checks:
//!
//! 1. The request states the tier it was asked from (1 or 2).
//! 2. This node can name itself: exactly one announced peer has its key.
//! 3. Each half is a record its serving node signed with that node's
//!    announced key, and carries the request and answer bodies.
//! 4. This node is neither twin, and both twins answered the same request.
//! 5. Each answer is the body its record's signature covers.
//! 6. When the twins differ, the referee answer is one THIS node served: a
//!    record on its own chain, signed with its own key, naming it as server,
//!    whose `response_digest` is the answer's digest. That record is cited.
//! 7. Both twins served the same weights, as their records say; otherwise,
//!    or when either was sampled, the ruling is `not_comparable`, with why.
//!
//! The verdict seals the tier the requester stated (`selection_tier`) and the
//! model the twins served (`model_hash`), and names this node as referee. How
//! the referee's prompt is known is recorded as the requester's word
//! (`referee_prompt: requester_attested`): the host rewrites a request before
//! a provider seals it, so the tie to the twins' prompt is not checked here.
//!
//! **Who asks** is checked first, before any of these and before anything is
//! read ([`handle_asked_adjudicate_request`]): the requester signs its
//! request as it signs every evidence request (`requester_auth`), and a
//! signature that fails its deadline, window, key, `audience` (this node),
//! nonce or stated id is refused `not_authorized`. A verdict does not depend
//! on who asks, so an unsigned request, or a valid signature under a key no
//! host binds, is answered as before.
//!
//! The verdict is held beside the ledger (`issued-adjudications.jsonl`),
//! never on the chain; the bridge seals this node's `adjudication_issued`
//! record citing it before the reply leaves. A repeated request about the
//! same pair returns the verdict already issued.

use std::path::Path;

use ed25519_dalek::SigningKey;
use serde_json::{json, Map, Value};

use super::half::{model_hash, record_checks, served_by, signer, weights_digest, Half, HalfError};
use super::verdict::{
    self, adjudicate, referee_verdict, Outcome, RefereeAnswer, UnknownWeights, Unreachable,
    ADJUDICATION_SCHEMA, CAPTURE_METHOD_DETERMINISTIC_REPLAY, NO_VERDICT_NOT_COMPARABLE,
    NO_VERDICT_REFEREE_UNREACHABLE, SOURCE_TWIN_COMPARISON, VERDICT_NOT_COMPARABLE,
};
use crate::peer_keys::{announced_key_in, peer_id_for_key_in};

pub const ADJUDICATE_SUBJECT_KIND: &str = "adjudicate";
/// The member that marks a reply as an issued verdict.
pub const ADJUDICATION_VERDICT_MARKER: &str = "adjudication_verdict";
pub const ISSUED_ADJUDICATIONS_FILENAME: &str =
    crate::adjudication_records::ISSUED_ADJUDICATIONS_FILENAME;
/// The referee's prompt is the requester's word (see the module doc).
pub const REFEREE_PROMPT_REQUESTER_ATTESTED: &str = "requester_attested";

pub const NOT_COMPARABLE_SAMPLED: &str = "sampled";
pub const NOT_COMPARABLE_WEIGHTS_DIFFER: &str = "weights_differ";
pub const NOT_COMPARABLE_WEIGHTS_UNKNOWN: &str = "weights_unknown";

pub const REASON_REQUEST_MALFORMED: &str = "request_malformed";
pub const REASON_HALF_UNVERIFIED: &str = "half_unverified";
pub const REASON_TWINS_DIFFER_IN_REQUEST: &str = "twins_differ_in_request";
pub const REASON_REFEREE_RECORD_NOT_FOUND: &str = "referee_record_not_found";
pub const REASON_REFEREE_NOT_INDEPENDENT: &str = "referee_not_independent";
pub const REASON_REFEREE_UNNAMED: &str = "referee_unnamed";
pub const REASON_NO_VERDICT: &str = "no_verdict";
/// A signed request whose signature does not prove its asker
/// (`requester_auth`): the evidence-request refusal token of the same name.
pub const REASON_NOT_AUTHORIZED: &str = "not_authorized";

/// The two tiers a referee can be asked from (`crate::referee` selection):
/// 1, a node with a corroboration for the model; 2, one with none yet.
pub const SELECTION_TIERS: [u64; 2] = [1, 2];

pub fn is_adjudicate_request(request: &Value) -> bool {
    request.pointer("/subject/kind").and_then(Value::as_str) == Some(ADJUDICATE_SUBJECT_KIND)
}

/// This node, as the referee's door sees it.
pub struct Referee<'a> {
    pub ledger_dir: &'a Path,
    pub signing_key: &'a SigningKey,
    /// The raw announced-key registry.
    pub peer_keys: Option<&'a str>,
}

/// A signed refusal, with the reason's `detail` when it has one.
fn refuse(node: &Referee<'_>, body: &[u8], reason: &str, detail: Option<&str>, now: &str) -> Value {
    let mut reply = crate::record_push_receive::refuse(node.signing_key, body, reason, now);
    if let Some(detail) = detail {
        reply["detail"] = json!(detail);
    }
    reply
}

/// The half `entry` names, when its serving node signed its record with
/// that node's announced key and the entry carries both bodies.
fn half(entry: &Value, peer_keys: Option<&str>) -> Option<Half> {
    let capsule = entry.get("capsule")?;
    capsule.get("capsule_id").and_then(Value::as_str)?;
    let server = served_by(capsule)?;
    let key = signer(capsule)?;
    if !record_checks(capsule)
        || capsule.get("key_id").and_then(Value::as_str) != Some(key.as_str())
        || announced_key_in(peer_keys, server).as_deref() != Some(key.as_str())
    {
        return None;
    }
    let response_body = entry.get("response_body").filter(|b| b.is_object())?;
    let request_body = entry.get("request_body").filter(|b| b.is_object())?;
    Some(Half {
        capsule: capsule.clone(),
        request_body: request_body.clone(),
        response_body: Some(response_body.clone()),
        response_text: None,
        node_id: Some(server.to_string()),
        weights_digest: weights_digest(capsule).map(str::to_string),
    })
}

/// This node's own served record of the referee answer whose digest is
/// `answer_digest`, newest first.
fn own_referee_record(
    ledger_dir: &Path,
    own_key: &str,
    self_id: &str,
    answer_digest: &str,
) -> Option<Value> {
    crate::evidence_panes::read_capsule_records(ledger_dir)
        .into_iter()
        .rev()
        .find(|record| {
            record
                .pointer("/effect/response_digest")
                .and_then(Value::as_str)
                == Some(answer_digest)
                && served_by(record) == Some(self_id)
                && record.get("key_id").and_then(Value::as_str) == Some(own_key)
                && record_checks(record)
                && signer(record).as_deref() == Some(own_key)
        })
}

fn held_lines(ledger_dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(ledger_dir.join(ISSUED_ADJUDICATIONS_FILENAME))
        .map(|text| {
            text.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn already_issued(ledger_dir: &Path, halves: &[String; 2]) -> Option<Value> {
    let mut want = halves.clone();
    want.sort();
    held_lines(ledger_dir).into_iter().find(|held| {
        let mut got: Vec<String> = held
            .get("halves")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|h| h.as_str().map(str::to_string))
            .collect();
        got.sort();
        got == want
    })
}

fn reply(held: &Value) -> Value {
    let mut out = Map::new();
    out.insert(ADJUDICATION_VERDICT_MARKER.into(), json!(1));
    if let Some(held) = held.as_object() {
        out.extend(held.clone());
    }
    Value::Object(out)
}

/// The ruling block the referee signs.
fn ruling_block(
    outcome: &Outcome,
    tier: u64,
    model: Option<&str>,
    extra: Map<String, Value>,
) -> Map<String, Value> {
    let verdict = outcome.verdict.as_deref().unwrap_or_default();
    let mut block = Map::new();
    block.insert("schema".into(), json!(ADJUDICATION_SCHEMA));
    block.insert("source".into(), json!(SOURCE_TWIN_COMPARISON));
    block.insert(
        "capture_method".into(),
        json!(CAPTURE_METHOD_DETERMINISTIC_REPLAY),
    );
    block.insert("verdict".into(), json!(verdict));
    block.insert("status".into(), json!(verdict::status_for_verdict(verdict)));
    block.insert("divergence_index".into(), json!(outcome.divergence_index));
    block.insert("prefix_digest".into(), json!(outcome.prefix_digest));
    block.insert(
        "twin_owner_distinct".into(),
        json!(outcome.twin_owner_distinct),
    );
    block.insert("weights_digest".into(), json!(outcome.weights_digest));
    block.insert("half_a_capsule_id".into(), json!(outcome.half_a_capsule_id));
    block.insert("half_b_capsule_id".into(), json!(outcome.half_b_capsule_id));
    if outcome.referee_called {
        block.insert(
            "referee_capsule_id".into(),
            json!(outcome.referee_capsule_id),
        );
        block.insert("referee_id".into(), json!(outcome.referee_id));
    }
    if !cfg!(feature = "mutant-referee-verdict-tier-not-sealed") {
        block.insert("selection_tier".into(), json!(tier));
    }
    block.insert("model_hash".into(), json!(model));
    for (key, value) in extra {
        block.entry(key).or_insert(value);
    }
    block
}

/// What a request's signature is checked against.
pub struct Asked<'a> {
    pub keys: crate::requester_auth::Keys<'a>,
    pub replay: &'a crate::requester_auth::ReplayCache,
    /// This node's own mesh id, as the host reported it.
    pub own_node: Option<&'a str>,
}

/// [`handle_adjudicate_request`], once the request's signature, if it
/// carries one, proved its asker (`requester_auth::principal`; a verdict
/// needs no identity). A malformed `requester_auth` is
/// [`REASON_REQUEST_MALFORMED`]; any other failure is
/// [`REASON_NOT_AUTHORIZED`], given before anything is read.
pub fn handle_asked_adjudicate_request(
    node: &Referee<'_>,
    body: &[u8],
    now: &str,
    asked: &Asked<'_>,
) -> std::io::Result<Value> {
    if let Ok(Value::Object(map)) = crate::strict_json::parse(body) {
        let checked = capsule_emit_evidence_request::time::parse_utc(now)
            .ok_or(crate::requester_auth::Refused::NotAuthorized)
            .and_then(|now| {
                crate::requester_auth::principal(
                    crate::requester_auth::Channel::RefereeAdjudicate,
                    &map,
                    now,
                    asked.keys,
                    asked.replay,
                    asked.own_node,
                    false,
                )
            });
        match checked {
            Ok(_) => {}
            Err(crate::requester_auth::Refused::Malformed) => {
                return Ok(refuse(node, body, REASON_REQUEST_MALFORMED, None, now))
            }
            Err(crate::requester_auth::Refused::NotAuthorized) => {
                return Ok(refuse(node, body, REASON_NOT_AUTHORIZED, None, now))
            }
        }
    }
    handle_adjudicate_request(node, body, now)
}

/// Answer one adjudicate request (`body`, as received) at `now`: a signed
/// verdict, or a signed refusal. Never panics on any body; an `Err` is a
/// local write that failed. Who asks is not checked here
/// ([`handle_asked_adjudicate_request`]).
pub fn handle_adjudicate_request(
    node: &Referee<'_>,
    body: &[u8],
    now: &str,
) -> std::io::Result<Value> {
    let refuse = |reason: &str, detail: Option<&str>| refuse(node, body, reason, detail, now);
    let Ok(request) = crate::strict_json::parse(body) else {
        return Ok(refuse(REASON_REQUEST_MALFORMED, None));
    };
    if !is_adjudicate_request(&request) {
        return Ok(refuse(REASON_REQUEST_MALFORMED, None));
    }
    let tier = request
        .get("selection_tier")
        .filter(|t| !t.is_f64())
        .and_then(Value::as_u64)
        .filter(|t| SELECTION_TIERS.contains(t));
    let entries = request
        .get("halves")
        .and_then(Value::as_array)
        .filter(|h| h.len() == 2);
    let answer = request.get("referee_answer").filter(|a| !a.is_null());
    let bracket = request.get("twin_bracket_id").filter(|b| !b.is_null());
    let (Some(tier), Some(entries)) = (tier, entries) else {
        return Ok(refuse(REASON_REQUEST_MALFORMED, None));
    };
    if answer.is_some_and(|a| !a.get("response_body").is_some_and(Value::is_object))
        || bracket.is_some_and(|b| !b.is_string())
    {
        return Ok(refuse(REASON_REQUEST_MALFORMED, None));
    }
    let bracket = bracket.and_then(Value::as_str);

    let own_key = hex::encode(node.signing_key.verifying_key().to_bytes());
    let Some(self_id) = peer_id_for_key_in(node.peer_keys, &own_key) else {
        return Ok(refuse(REASON_REFEREE_UNNAMED, None));
    };
    let (Some(mut half_a), Some(mut half_b)) = (
        half(&entries[0], node.peer_keys),
        half(&entries[1], node.peer_keys),
    ) else {
        return Ok(refuse(REASON_HALF_UNVERIFIED, None));
    };
    let (node_a, node_b) = (
        half_a.node_id.clone().unwrap_or_default(),
        half_b.node_id.clone().unwrap_or_default(),
    );
    if self_id == node_a || self_id == node_b {
        return Ok(refuse(REASON_REFEREE_NOT_INDEPENDENT, None));
    }
    let request_digest = |h: &Half| {
        h.capsule
            .pointer("/effect/request_digest")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
            .map(str::to_string)
    };
    let twins_request = request_digest(&half_a);
    if twins_request.is_none() || twins_request != request_digest(&half_b) {
        return Ok(refuse(REASON_TWINS_DIFFER_IN_REQUEST, None));
    }

    let halves = [
        half_a.capsule_id().to_string(),
        half_b.capsule_id().to_string(),
    ];
    if let Some(held) = already_issued(node.ledger_dir, &halves) {
        return Ok(reply(&held));
    }

    // The referee answer decides only when the twins differ; it must then be
    // one this node served.
    let answer_body = answer.map(|a| a["response_body"].clone());
    let answer_digest = answer_body
        .as_ref()
        .and_then(|b| crate::capsule_emit::digest_json(b).ok());
    let record = answer_digest
        .as_deref()
        .and_then(|digest| own_referee_record(node.ledger_dir, &own_key, &self_id, digest));
    let referee_text = answer_body
        .as_ref()
        .and_then(|b| b.pointer("/choices/0/message/content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    // Twins are compared only on the same weights, as their records say.
    let mut not_comparable_because = None;
    let (weights_a, weights_b) = (half_a.weights_digest.clone(), half_b.weights_digest.clone());
    if !(weights_a.is_some() && weights_a == weights_b) {
        not_comparable_because = Some(if weights_a.is_some() && weights_b.is_some() {
            NOT_COMPARABLE_WEIGHTS_DIFFER
        } else {
            NOT_COMPARABLE_WEIGHTS_UNKNOWN
        });
        // The checks on each half still run: a forged half is refused, never ruled on.
        half_a.weights_digest = None;
        half_b.weights_digest = None;
    }
    let referee_capsule_id = record
        .as_ref()
        .and_then(|r| r.get("capsule_id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut ask = |a: &Half, b: &Half, comparison: &verdict::Comparison| match &referee_capsule_id {
        None => Err(Unreachable),
        Some(capsule_id) => Ok(RefereeAnswer {
            verdict: referee_verdict(a, b, comparison, &referee_text),
            capsule_id: Some(capsule_id.clone()),
            referee_id: self_id.clone(),
        }),
    };
    let referee: Option<verdict::Referee<'_>> = if not_comparable_because.is_some() {
        None
    } else {
        Some(&mut ask)
    };
    let mut outcome = match adjudicate(
        &half_a,
        &half_b,
        referee,
        Some(&self_id),
        UnknownWeights::Compare,
    ) {
        Ok(outcome) => outcome,
        Err(HalfError::AnswerNotTheSignedBody) => {
            return Ok(refuse(
                REASON_HALF_UNVERIFIED,
                Some("an answer is not the body its record signs"),
            ))
        }
        Err(HalfError::Forged) => return Ok(refuse(REASON_HALF_UNVERIFIED, None)),
    };
    if outcome.no_verdict_reason == Some(NO_VERDICT_NOT_COMPARABLE) {
        not_comparable_because = Some(NOT_COMPARABLE_SAMPLED);
    }
    if not_comparable_because.is_some() {
        // A signed ruling that the twins cannot be compared: never a
        // corroboration or a contradiction.
        outcome.verdict = Some(VERDICT_NOT_COMPARABLE.to_string());
        outcome.no_verdict_reason = None;
        outcome.referee_called = false;
    } else if outcome.verdict.is_none() {
        if outcome.no_verdict_reason == Some(NO_VERDICT_REFEREE_UNREACHABLE) && record.is_none() {
            return Ok(refuse(REASON_REFEREE_RECORD_NOT_FOUND, None));
        }
        return Ok(refuse(REASON_NO_VERDICT, outcome.no_verdict_reason));
    }

    let mut extra = Map::new();
    extra.insert("referee_node_id".into(), json!(self_id));
    extra.insert("half_a_node_id".into(), json!(node_a));
    extra.insert("half_b_node_id".into(), json!(node_b));
    extra.insert("twins_request_digest".into(), json!(twins_request));
    extra.insert(
        "referee_prompt".into(),
        json!(REFEREE_PROMPT_REQUESTER_ATTESTED),
    );
    if let Some(because) = not_comparable_because {
        extra.insert("not_comparable_because".into(), json!(because));
    }
    if outcome.referee_called {
        if let Some(digest) = &answer_digest {
            extra.insert("referee_answer_digest".into(), json!(digest));
        }
    }
    if let Some(bracket) = bracket.filter(|b| !b.is_empty()) {
        extra.insert("twin_bracket_id".into(), json!(bracket));
    }
    let block = ruling_block(&outcome, tier, model_hash(&half_a.capsule), extra);
    let Ok(capsule) =
        crate::producer::capsule::seal_verdict_record(block, &halves[0], node.signing_key)
    else {
        return Ok(refuse(REASON_NO_VERDICT, None));
    };
    let held = json!({
        "verdict": outcome.verdict,
        "verdict_capsule_id": capsule["capsule_id"],
        "verdict_capsule": capsule,
        "referee_node_id": self_id,
        // Cited only when the referee's answer decided (the twins differed).
        "referee_capsule_id": if outcome.referee_called { referee_capsule_id } else { None },
        "halves": halves,
        "half_node_ids": [node_a, node_b],
        "twin_bracket_id": bracket,
        "issued_at": now,
    });
    append(node.ledger_dir, &held)?;
    Ok(reply(&held))
}

fn append(ledger_dir: &Path, line: &Value) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(ledger_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(ISSUED_ADJUDICATIONS_FILENAME))?;
    writeln!(file, "{line}")?;
    file.sync_data()
}
