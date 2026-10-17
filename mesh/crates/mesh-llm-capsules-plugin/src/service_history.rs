//! The Service History (`service_history/1`, PRD-SERVICE-EVIDENCE §2, M2):
//! one versioned projection that embeds the `history_card/1` artifact
//! unchanged and declares which further members it carries.
//!
//! **The derivation.** Unregistered, and named on the wire by the digest of
//! its definition document (`derivations/service-history-1.json`,
//! [`DEFINITION_DIGEST`]). The token `service_history/1` is the document's
//! `name`, never a request's `derivation`: a request naming it is refused
//! `derivation_unsupported`. The subject is `full_history`.
//!
//! **The artifact** is one object, as its RFC 8785 octets:
//! `projection` (the definition digest), `subject`, `anchor`, `declared`
//! (the members it carries, sorted), `history_card`, `history_card_digest`
//! and, when declared, `served_summary`. A member is present exactly when it
//! is declared, and no other key appears. This node always declares both
//! members, so the bytes depend only on (subject, anchor, definition): the
//! same for every requester (R-15).
//!
//! - `history_card` is the `history_card/1` artifact this node serves for
//!   the same subject and anchor, as a JSON value. RFC 8785 is
//!   compositional, so its canonical octets are the direct artifact's octets,
//!   and `history_card_digest` is their SHA-256 (PRD T6).
//! - `served_summary` is the direct `served_summary/1` artifact
//!   (`crate::served_summary`), as a JSON value.
//!
//! **Answered** under the served summary's coverage rule: an `expected_pin`
//! naming this node's latest checkpoint. A requester learns that pin from a
//! direct `history_card/1` request under `min_freshness`. `min_freshness`
//! here is `policy_declined`, an older pin `coverage_unsatisfiable`. It is
//! Standard History (`share_standard_history`), and when either member cannot
//! be built the whole projection is refused, never served with a member
//! dropped (`crate::evidence_answer`).
//!
//! **The wire** is the interop profile's envelope, as for every other answer
//! (`crate::evidence_artifact`): the envelope names the anchor, its material
//! is the anchor's `cll-checkpoint`, and the witness receipts of the card's
//! checkpoints ride beside it in `witness_receipts`.
//!
//! **Verified** by [`verify`]. The card is bound by the envelope's signed
//! artifact digest, which covers these octets; there is no second envelope.
//! Its content is checked as a direct `history_card/1` answer's is
//! (`evidence_artifact::check_history_card`): every checkpoint signed by the
//! responder, the list starting at the stream's first checkpoint and ending
//! at the anchor, and each consistency proof; then the witness receipts. A
//! requester that needs the card as a standalone -00 artifact asks for
//! `history_card/1` under the same pin and compares its digest with
//! `history_card_digest`.
//!
//! **Rendered** by [`render`]: every count with its unit, every model,
//! quantization and weights-digest figure as what the provider states, and
//! nothing that reads continuity, a signature or a receipt as execution
//! (PRD R-1, R-6, T5).

use std::collections::BTreeMap;

use ed25519_dalek::VerifyingKey;
use evidence_request::envelope::subject_value;
use evidence_request::jcs;
use evidence_request::registry::{Reason, DERIVATION_HISTORY_CARD};
use evidence_request::request::{Coverage, Derivation, Request, Subject};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::evidence_artifact::{self, CheckError, ReceiptCheck, ReceiptStatus, Stated};

/// The derivation's name, as its definition document states it.
#[cfg(test)]
pub const DERIVATION_NAME: &str = "service_history/1";

/// The definition document, as published beside this crate. Only its
/// digest is used at run time; a test pins one to the other.
#[cfg(test)]
pub const DEFINITION_DOCUMENT: &str = include_str!("../derivations/service-history-1.json");

/// SHA-256 over the RFC 8785 bytes of [`DEFINITION_DOCUMENT`]: the
/// derivation's name on the wire, and the artifact's `projection`.
pub const DEFINITION_DIGEST: &str =
    "6f8d6d058322b909e249e49bbeaae921abcff63b828c36c41202c75e392ba834";

/// The keys every artifact carries, whatever it declares.
const FIXED_KEYS: [&str; 5] = [
    "anchor",
    "declared",
    "history_card_digest",
    "projection",
    "subject",
];

/// The definition's members, in order, and whether each is required.
const MEMBERS: [(&str, bool); 2] = [("history_card", true), ("served_summary", false)];

/// The members this verifier checks.
const CHECKED_HERE: [&str; 2] = ["history_card", "served_summary"];

/// Whether `derivation` names this one.
pub fn is_derivation(derivation: Option<&Derivation>) -> bool {
    matches!(derivation, Some(Derivation::Digest(d)) if d == DEFINITION_DIGEST)
}

// ---------------------------------------------------------------------
// Responder
// ---------------------------------------------------------------------

/// The artifact, as RFC 8785 text: `card` is the direct `history_card/1`
/// artifact's octets for `subject` under `anchor`, and `summary` the
/// served-summary artifact (`None` only in the mutant that drops it).
/// `policy_declined` when the card's octets are not its own canonical form,
/// so the embedded card could not be the direct one.
pub(crate) fn artifact(
    card: &[u8],
    summary: Option<Value>,
    subject: &Subject,
    anchor: &str,
) -> Result<String, Reason> {
    let mut card_value: Value = serde_json::from_slice(card).map_err(|_| Reason::PolicyDeclined)?;
    if jcs::to_string(&card_value)
        .map_err(|_| Reason::PolicyDeclined)?
        .as_bytes()
        != card
    {
        return Err(Reason::PolicyDeclined);
    }
    // Deliberately broken in one CI build: the card is rebuilt from its
    // checkpoint list, without the consistency proofs.
    if cfg!(feature = "mutant-service-history-card-rebuilt") {
        if let Some(history) = card_value.get_mut("history").and_then(Value::as_array_mut) {
            for entry in history {
                entry["consistency_from_previous"] = Value::Null;
            }
        }
    }
    let card_text = jcs::to_string(&card_value).map_err(|_| Reason::PolicyDeclined)?;
    let mut declared = vec!["history_card"];
    let mut out = Map::new();
    out.insert("projection".into(), json!(DEFINITION_DIGEST));
    out.insert("subject".into(), subject_value(subject));
    out.insert("anchor".into(), json!(anchor));
    out.insert(
        "history_card_digest".into(),
        json!(hex::encode(Sha256::digest(card_text.as_bytes()))),
    );
    out.insert("history_card".into(), card_value);
    if let Some(summary) = summary {
        declared.push("served_summary");
        out.insert("served_summary".into(), summary);
    }
    declared.sort_unstable();
    out.insert("declared".into(), json!(declared));
    jcs::to_string(&Value::Object(out)).map_err(|_| Reason::PolicyDeclined)
}

// ---------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------

/// What one verified answer establishes.
#[derive(Clone, Debug)]
pub struct Verified {
    /// The anchor checkpoint's digest.
    pub anchor: String,
    /// Each declared member: `"checked"`, or `"not_checked"` for one this
    /// verifier does not implement.
    pub members: BTreeMap<String, &'static str>,
    /// The card's checkpoints, oldest first, each signed by the responder.
    pub checkpoints: usize,
    /// Every witness receipt the card's checkpoints carry, as checked.
    pub receipts: Vec<ReceiptCheck>,
    /// The embedded served summary's `served_summary` value, when checked.
    pub served_summary: Option<Value>,
}

impl Verified {
    /// The requester's result: counts per unit, nothing summed across
    /// members, and the History as text ([`render`]).
    pub fn to_json(&self) -> Value {
        let count =
            |status: ReceiptStatus| self.receipts.iter().filter(|r| r.status == status).count();
        json!({
            "state": "artifact",
            "anchor": self.anchor,
            "members": self.members,
            "checkpoints": self.checkpoints,
            "witness_receipts": {
                "verified": count(ReceiptStatus::Verified),
                "no_key": count(ReceiptStatus::NoKey),
                "stub": count(ReceiptStatus::Stub),
            },
            "lines": render(self),
        })
    }
}

/// Check a Service History reply `response` to the request sent as `sent`
/// (parsed as `req`) against the responder's `key`. `witness_key` is the
/// requester's key for a witness, by its URL.
pub fn verify(
    sent: &[u8],
    req: &Request,
    response: &Value,
    key: &VerifyingKey,
    witness_key: &dyn Fn(&str) -> Option<VerifyingKey>,
) -> Result<Verified, String> {
    verify_with(sent, req, response, key, witness_key, &CHECKED_HERE)
}

/// [`verify`], for a verifier that checks only the members in `checked`.
/// The envelope, the anchor checkpoint, the coverage, the card and the
/// receipts are checked as `evidence_artifact` checks any answer.
fn verify_with(
    sent: &[u8],
    req: &Request,
    response: &Value,
    key: &VerifyingKey,
    witness_key: &dyn Fn(&str) -> Option<VerifyingKey>,
    checked: &[&str],
) -> Result<Verified, String> {
    let fail = |why: &str| why.to_string();
    if !is_derivation(req.derivation.as_ref()) {
        return Err(fail("the request sent does not name this derivation"));
    }
    if !matches!(req.coverage, Coverage::ExpectedPin(_)) {
        return Err(fail("the anchor does not satisfy the requested coverage"));
    }
    let failed = |e: CheckError| e.to_string();
    let env = evidence_artifact::verified_envelope(sent, req, response, key).map_err(failed)?;
    let served = evidence_artifact::anchor_checkpoint_only(&env, key).map_err(failed)?;
    evidence_artifact::check_coverage(&env, req, &served, &[]).map_err(failed)?;
    let anchor = served.digest.clone();
    let art: Value =
        serde_json::from_slice(&env.artifact).map_err(|_| fail("the artifact is malformed"))?;
    let art = art
        .as_object()
        .ok_or_else(|| fail("the artifact is malformed"))?;
    let subject = subject_value(&req.subject);
    if art.get("projection").and_then(Value::as_str) != Some(DEFINITION_DIGEST)
        || art.get("subject") != Some(&subject)
        || art.get("anchor").and_then(Value::as_str) != Some(anchor.as_str())
    {
        return Err(fail(
            "the artifact is not for the requested subject or derivation",
        ));
    }
    let declared = declared_members(art)?;

    let mut verified = Verified {
        anchor,
        members: BTreeMap::new(),
        checkpoints: 0,
        receipts: Vec::new(),
        served_summary: None,
    };
    // The checkpoints whose receipts are checked: the card's, once checked.
    let mut listed: Vec<Stated> = Vec::new();
    for member in &declared {
        if !checked.contains(member) {
            verified.members.insert(member.to_string(), "not_checked");
            continue;
        }
        match *member {
            "history_card" => {
                listed = check_card(art, &subject, &served, key)?;
                verified.checkpoints = listed.len();
            }
            "served_summary" => {
                let summary = &art["served_summary"];
                if !served_summary_matches(summary, &subject, &served) {
                    return Err(fail(
                        "the served summary is not for this subject and anchor",
                    ));
                }
                verified.served_summary = summary.get("served_summary").cloned();
            }
            _ => unreachable!("every member checked here has a check"),
        }
        verified.members.insert(member.to_string(), "checked");
    }
    let carried: Vec<&Stated> = if listed.is_empty() {
        vec![&served]
    } else {
        listed.iter().collect()
    };
    verified.receipts = evidence_artifact::check_receipts(
        response.get(evidence_artifact::WITNESS_RECEIPTS),
        &carried,
        witness_key,
    )
    .map_err(failed)?;
    Ok(verified)
}

/// The members `art` declares: sorted, each one the definition has, the
/// required ones among them, each present, and no other key.
fn declared_members(art: &Map<String, Value>) -> Result<Vec<&str>, String> {
    let malformed = || "the declared members are malformed".to_string();
    let declared: Vec<&str> = art
        .get("declared")
        .and_then(Value::as_array)
        .and_then(|d| d.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
        .ok_or_else(malformed)?;
    if declared.windows(2).any(|w| w[0] >= w[1])
        || declared
            .iter()
            .any(|d| !MEMBERS.iter().any(|(name, _)| name == d))
        || MEMBERS
            .iter()
            .any(|(name, required)| *required && !declared.contains(name))
    {
        return Err(malformed());
    }
    let mut expected: Vec<&str> = FIXED_KEYS
        .iter()
        .copied()
        .chain(declared.iter().copied())
        .collect();
    expected.sort_unstable();
    let mut keys: Vec<&str> = art.keys().map(String::as_str).collect();
    keys.sort_unstable();
    if keys != expected {
        return Err("the artifact's members are not the ones it declares".into());
    }
    Ok(declared)
}

/// The embedded card: the octets `history_card_digest` names, a
/// `history_card/1` artifact for `subject`, and a chain of the responder's
/// checkpoints ending at the anchor `served` (`evidence_artifact`'s check of
/// a direct card). Returns the chain.
fn check_card(
    art: &Map<String, Value>,
    subject: &Value,
    served: &Stated,
    key: &VerifyingKey,
) -> Result<Vec<Stated>, String> {
    let card = &art["history_card"];
    let text = jcs::to_string(card).map_err(|_| "the card is malformed".to_string())?;
    if art.get("history_card_digest").and_then(Value::as_str)
        != Some(hex::encode(Sha256::digest(text.as_bytes())).as_str())
    {
        return Err("the card does not match history_card_digest".into());
    }
    let card = card
        .as_object()
        .ok_or_else(|| "the card is malformed".to_string())?;
    if card.get("subject") != Some(subject)
        || card.get("derivation").and_then(Value::as_str) != Some(DERIVATION_HISTORY_CARD)
    {
        return Err("the card is not for the requested subject".into());
    }
    let (anchor, listed) =
        evidence_artifact::check_history_card(card, key).map_err(|e| e.to_string())?;
    if anchor.digest != served.digest {
        return Err("the card does not end at the anchor".into());
    }
    Ok(listed)
}

/// Whether `summary` is a served-summary artifact for `subject` under the
/// anchor `served`. Its `derivation` is the definition digest, or the token
/// a node before the digest wrote.
fn served_summary_matches(summary: &Value, subject: &Value, served: &Stated) -> bool {
    let derivation = summary.get("derivation").and_then(Value::as_str);
    summary.get("anchor").and_then(Value::as_str) == Some(served.digest.as_str())
        && summary.get("evidence_stream").and_then(Value::as_str) == Some(served.cp.log_id.as_str())
        && summary.get("subject") == Some(subject)
        && (derivation == Some(crate::served_summary::SERVED_SUMMARY_DEFINITION_DIGEST)
            || derivation == Some(crate::served_summary::SERVED_SUMMARY_DERIVATION_TOKEN))
        && summary.get("served_summary").is_some_and(Value::is_object)
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

/// The words a rendered History never uses: continuity, a signature or a
/// receipt is never execution, and nothing ranks a node (PRD T5, §4).
#[cfg(test)]
pub const DENY_LIST: [&str; 8] = [
    "model executed",
    "verified model",
    "ran on",
    "attested",
    "score",
    "rating",
    "rank",
    "reputation",
];

/// The verified History as lines of text: each count with its unit, each
/// claim the host reported as what the provider states.
pub fn render(v: &Verified) -> Vec<String> {
    let mut lines = vec![
        format!("Service History, anchored at checkpoint {}", v.anchor),
        format!(
            "{} checkpoints: this log's own continuity, which says nothing about what was computed",
            v.checkpoints
        ),
    ];
    let verified = v
        .receipts
        .iter()
        .filter(|r| r.status == ReceiptStatus::Verified)
        .count();
    lines.push(if v.receipts.is_empty() {
        "witness receipts: no witness named".to_string()
    } else {
        format!(
            "witness receipts checked: {verified} of {} (a later rewrite of this log is detectable)",
            v.receipts.len()
        )
    });
    for (member, status) in &v.members {
        if *status == "not_checked" {
            lines.push(format!("{member}: carried, not checked by this verifier"));
        }
    }
    if let Some(summary) = &v.served_summary {
        let by_model = summary
            .pointer("/derivation/by_model")
            .and_then(Value::as_object);
        for (model, m) in by_model.into_iter().flatten() {
            let figure = |k: &str| m.get(k).map_or("unknown".to_string(), plain);
            lines.push(format!("model, as the provider states it: {model}"));
            lines.push(format!(
                "  served exchanges: {} ({} completed, {} failed)",
                figure("served"),
                figure("completed"),
                figure("failed")
            ));
            let quantizations: Vec<String> = m
                .get("quantizations")
                .and_then(Value::as_array)
                .map(|q| q.iter().map(plain).collect())
                .unwrap_or_default();
            if !quantizations.is_empty() {
                lines.push(format!(
                    "  quantization, as the provider states it: {}",
                    quantizations.join(", ")
                ));
            }
            let weights = |k: &str| {
                m.pointer(&format!("/weights_digest/{k}"))
                    .map_or("unknown".to_string(), plain)
            };
            lines.push(format!(
                "  weights digest, as the provider states it: present in {} served exchanges, absent in {}",
                weights("present"),
                weights("absent")
            ));
        }
        if let Some(n) = summary.pointer("/adjudications_received/value") {
            lines.push(format!(
                "referee verdicts received about this node's served exchanges (self-held): {}",
                plain(n)
            ));
        }
        if let Some(n) = summary.pointer("/selection/covered_entries") {
            lines.push(format!("records covered: {}", plain(n)));
        }
    }
    lines
}

/// A JSON value as text: a string without its quotes.
fn plain(v: &Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_string)
}

#[cfg(test)]
mod tests;
