//! Answering an evidence request in-process
//! (draft-mih-agent-evidence-request-00), for every subject. A referee's
//! `adjudicate` request is not a -00 subject and is refused, signed: this
//! node has no referee yet.
//!
//! The protocol is the `capsule-emit-evidence-request` crate's: this module
//! parses with `request::parse_json`, resolves with `resolve::resolve`,
//! builds with `answer::build` and signs refusals with `refusal::sign`. What
//! stays here is this node's own glue: the limits, the sharing-policy gate,
//! the served summary (a derivation the crate does not build), and the wire.
//!
//! **Order, cheapest first.** Every step that can refuse runs before the
//! work after it, so a request is turned away before it costs anything:
//!
//! 1. the request's size ([`MAX_REQUEST_BYTES`]) → `policy_declined`, before
//!    it is parsed;
//! 2. parsing → `request_malformed` / `coverage_unsatisfiable`, and a
//!    malformed `requester_auth` (or a signed request with no `audience`)
//!    → `request_malformed`;
//! 3. a deadline already past → `deadline_unmet`;
//! 4. the sharing policy switched `off` for any request that carries record
//!    bodies → `not_authorized`, before anything is looked up;
//! 5. who asks (`requester_auth`), before the ledger is read: a signature
//!    that fails its window, its key, its `audience` (this node), its nonce
//!    or its stated id → refused; a valid signature under a key no host
//!    binds → read as unsigned where the answer needs no identity, refused
//!    otherwise; an unsigned request whose answer depends on who asks →
//!    refused (see below);
//! 6. an answer that depends on who asks, to a node no record of this one
//!    names → refused, before the subject is looked up;
//! 7. resolution against the index → the crate's reason;
//! 8. the number of records (or checkpoints) the answer would carry, counted
//!    from the index without reading one → `policy_declined` over
//!    [`MAX_ANSWER_RECORDS`] / [`MAX_ANSWER_CHECKPOINTS`];
//! 9. the sharing-policy gate on each of those records → `not_authorized`;
//! 10. building and signing; an answer larger than [`MAX_ANSWER_BYTES`] on
//!     the wire → `policy_declined`.
//!
//! Every refusal is signed by this node's key and binds the digest of the
//! request bytes as received (`refusal::sign`). Nothing here unwraps or
//! panics on the request.
//!
//! **The sharing policy** (`share_policy::history_segments`) decides who gets
//! record bodies, by the same rule the plugin's `ledger-fetch/1` channel uses:
//! `off` serves none; a local block or an owner-maintenance record is never
//! served; under `peers` every other record goes to anyone; otherwise a
//! record goes only to the node it names as the other side of its exchange.
//! An answer is served whole or refused whole (`not_authorized`): an
//! artifact is the same for every requester (§5), so it is never filtered
//! per asker. Checkpoints, the history card and the served summary carry no
//! record bodies and are not gated by it.
//!
//! **Who asks** is the node a signed request proved it came from: the
//! `requester_auth` signature verifies under a key a host announced for that
//! node, and the request names this node as its `audience`
//! (`requester_auth`). A request's `requester_id` is only the asker's word:
//! it picks among nodes that announced the same key, one that names any
//! other node is refused, and an unsigned request has no principal at all. So an answer that depends on who asks (record bodies
//! unless they go to any peer, Standard History unless it is shared) is
//! refused to an unsigned request, before the ledger is read. Under `peers`,
//! or with Standard History shared, an unsigned request is answered as
//! before.
//!
//! **Standard History** is every answer that carries no record bodies:
//! `checkpoints`, `history_card/1`, `served_summary/1`, and any other
//! derivation. It shows how much and how often this node serves, so with
//! `share_standard_history` off (the default) it goes only to a requester
//! that has proven it is a node some record of this one names as the other
//! side of an exchange. Every other requester (unsigned, a signature that
//! fails, or a proven node no record names) gets `policy_declined`, decided
//! before the subject is resolved, so the refusal is the same whether or not
//! what was asked for exists. It is `refusal::sign` of the request digest,
//! the reason and `issued_at` alone: no member names or depends on the
//! asker, so two askers sending the same bytes in the same second get the
//! same bytes back. What differs between two refusals is only what differs
//! between the requests (their digest, which covers every member each
//! carries) and when they were answered (`issued_at`), and the signature
//! over those.
//!
//! A record request from a proven node that no record names is refused
//! `not_authorized` before the subject is resolved too, so a stranger cannot
//! read this log's size or cadence from which refusal it gets. A real
//! counterparty asking for more than its own records still can.
//!
//! **The wire.** An artifact answer is `{"artifact", "material", "envelope"}`:
//! the artifact and the verification material as the exact RFC 8785 text the
//! envelope's digest and signature cover (strings, so no re-serialization
//! can change a byte), and the signed envelope. A refusal is the crate's
//! refusal object. The carrier is plugin-internal, not a standardized wire.

use capsule_emit_evidence_request::answer::{self, BuildError, MAX_RECORDS};
use capsule_emit_evidence_request::digest::request_digest;
use capsule_emit_evidence_request::jcs;
use capsule_emit_evidence_request::refusal;
use capsule_emit_evidence_request::registry::Reason;
use capsule_emit_evidence_request::request::{self, Coverage, Derivation, Request, Subject};
use capsule_emit_evidence_request::resolve::{self, Resolution, ResolvedAnchor};
use capsule_emit_evidence_request::time::parse_utc;
use cll::mmr::leaf_count;
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::evidence_log::{self, LedgerIndex, LogView};
use crate::requester_auth::{self, Form, Principal};
use crate::served_summary::{self, SERVED_SUMMARY_DERIVATION_TOKEN};

/// The largest request answered. A -00 request is a few hundred bytes; this
/// turns a large one away before it is parsed.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;

/// The most records one answer carries. A larger subject is refused
/// `policy_declined`: ask for a range.
pub const MAX_ANSWER_RECORDS: u64 = 256;

/// The most checkpoints one answer carries (`checkpoints`, `history_card/1`).
pub const MAX_ANSWER_CHECKPOINTS: u64 = 1024;

/// The largest answer on the wire. The requester reads at most
/// `mesh_evidence_bridge::MAX_SIDE_STREAM_BYTES`, so a larger answer could
/// never arrive; it is refused here instead.
pub const MAX_ANSWER_BYTES: usize = crate::mesh_evidence_bridge::MAX_SIDE_STREAM_BYTES as usize;

const _: () = assert!(MAX_ANSWER_RECORDS <= MAX_RECORDS && MAX_ANSWER_CHECKPOINTS <= MAX_RECORDS);

/// What one request is answered against.
pub struct Responder<'a> {
    pub ledger_dir: &'a Path,
    pub signing_key: &'a SigningKey,
    /// Now, RFC 3339 UTC (`YYYY-MM-DDTHH:MM:SSZ`): every `issued_at`.
    pub now: &'a str,
    /// `share_policy::history_segments()`.
    pub history_segments: &'a str,
    /// `share_policy::shares_standard_history()`.
    pub share_standard_history: bool,
    /// The keys a requester's signature is bound through
    /// (`requester_auth::bound_node`).
    pub keys: requester_auth::Keys<'a>,
    /// The (key_id, nonce) pairs already answered
    /// (`requester_auth::replay_cache()`).
    pub replay: &'a requester_auth::ReplayCache,
    /// This node's own mesh node id, which a signed request's `audience`
    /// must name. `None` until the host has reported it: then no signed
    /// request is addressed to this node.
    pub node_id: Option<&'a str>,
}

/// The one outcome of a request.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// An artifact answer, on the wire.
    Answered {
        wire: Value,
        subject_kind: &'static str,
    },
    /// A signed refusal, on the wire.
    Refused {
        wire: Value,
        reason: Reason,
        subject_kind: Option<&'static str>,
    },
    /// No answer can be given (this node cannot sign one: its clock or its
    /// ledger could not be read). Nothing is sent; the requester records an
    /// absence, never a refusal this node did not sign.
    Unanswerable(String),
}

impl Outcome {
    /// The `received_log` status and reason.
    pub fn log_status(&self) -> (&'static str, Option<&'static str>) {
        match self {
            Outcome::Answered { .. } => ("answered", None),
            Outcome::Refused { reason, .. } => ("refused", Some(reason.token())),
            Outcome::Unanswerable(_) => ("unanswered", None),
        }
    }

    pub fn subject_kind(&self) -> Option<&'static str> {
        match self {
            Outcome::Answered { subject_kind, .. } => Some(subject_kind),
            Outcome::Refused { subject_kind, .. } => *subject_kind,
            Outcome::Unanswerable(_) => None,
        }
    }

    /// The bytes to send, if any.
    pub fn wire_bytes(&self) -> Option<Vec<u8>> {
        match self {
            Outcome::Answered { wire, .. } | Outcome::Refused { wire, .. } => {
                serde_json::to_vec(wire).ok()
            }
            Outcome::Unanswerable(_) => None,
        }
    }
}

/// The requester's self-declared id: the request's optional top-level
/// `requester_id` text member. Read only from a request within
/// [`MAX_REQUEST_BYTES`].
pub fn requester_id(request_bytes: &[u8]) -> Option<String> {
    if request_bytes.len() > MAX_REQUEST_BYTES {
        return None;
    }
    let value: Value = serde_json::from_slice(request_bytes).ok()?;
    value.get("requester_id")?.as_str().map(str::to_string)
}

/// Who a request proved it came from, as far as it got: the key its
/// signature verified under, and the node that key is bound to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Asker {
    pub key_id: Option<String>,
    pub node_id: Option<String>,
    /// `bound` when the key is bound to `node_id`; `unbound` when the
    /// signature verified under a key no host binds (never a principal).
    pub binding: Option<&'static str>,
    /// The `audience` a signed request named, as checked.
    pub audience: Option<String>,
    /// `checked` when the request's nonce went through the replay check;
    /// `unchecked` when it could not, because the unbound pool was full
    /// (ruling u290 §2).
    pub replay: Option<&'static str>,
}

/// Answer `request_bytes`, exactly as received.
#[cfg(test)]
pub fn answer(request_bytes: &[u8], responder: &Responder<'_>) -> Outcome {
    answer_as(request_bytes, responder).0
}

/// [`answer`], with who asked.
fn answer_as(request_bytes: &[u8], responder: &Responder<'_>) -> (Outcome, Asker) {
    let mut asker = Asker::default();
    let outcome = answer_inner(request_bytes, responder, &mut asker);
    (outcome, asker)
}

fn answer_inner(request_bytes: &[u8], responder: &Responder<'_>, asker: &mut Asker) -> Outcome {
    let digest = request_digest(request_bytes);
    let refuse = |reason: Reason, subject_kind: Option<&'static str>| match refusal::sign(
        &digest,
        reason,
        responder.now,
        responder.signing_key,
    ) {
        Ok(r) => Outcome::Refused {
            wire: r.to_json(),
            reason,
            subject_kind,
        },
        Err(e) => Outcome::Unanswerable(format!("could not sign a {reason} refusal: {e}")),
    };

    // 1. Size, before parsing.
    if request_bytes.len() > MAX_REQUEST_BYTES {
        return refuse(Reason::PolicyDeclined, None);
    }
    // 2. Parse.
    let req = match request::parse_json(request_bytes) {
        Ok(req) => req,
        Err(e) => return refuse(e.reason(), None),
    };
    let kind = Some(req.subject.form().token());
    let (map, form) = match serde_json::from_slice::<Value>(request_bytes) {
        Ok(Value::Object(map)) => {
            let form = requester_auth::form(&map);
            (map, form)
        }
        _ => return refuse(Reason::RequestMalformed, kind),
    };
    if form == Form::Malformed {
        return refuse(Reason::RequestMalformed, kind);
    }
    // 3. Deadline.
    let Some(now) = parse_utc(responder.now) else {
        return Outcome::Unanswerable("this node's clock is not an RFC 3339 UTC time".into());
    };
    if req.deadline.is_some_and(|deadline| deadline < now) {
        return refuse(Reason::DeadlineUnmet, kind);
    }
    // 4. Sharing switched off: no record bodies to anyone.
    if carries_records(&req) && responder.history_segments == "off" && policy_gate_applies() {
        return refuse(Reason::NotAuthorized, kind);
    }
    // 5. Who asks: the requester's signature, before anything is looked up.
    let principal = match requester(&map, form, &req, now, responder, asker) {
        Ok(principal) => principal,
        Err(reason) => return refuse(reason, kind),
    };
    let principal_node = principal.as_ref().map(|p| p.node_id.as_str());

    let key_id = hex::encode(responder.signing_key.verifying_key().to_bytes());
    let built = evidence_log::with_index(responder.ledger_dir, &key_id, |index| {
        let view = index.view()?;
        Ok::<_, std::io::Error>(answer_from(&req, &digest, principal_node, responder, &view))
    });
    match built {
        Ok(Ok(Ok(wire))) => match serde_json::to_vec(&wire) {
            Ok(bytes) if bytes.len() <= MAX_ANSWER_BYTES => Outcome::Answered {
                wire,
                subject_kind: req.subject.form().token(),
            },
            _ => refuse(Reason::PolicyDeclined, kind),
        },
        Ok(Ok(Err(reason))) => refuse(reason, kind),
        Ok(Err(e)) | Err(e) => Outcome::Unanswerable(format!("the ledger could not be read: {e}")),
    }
}

/// Whether the answer to `req` carries record bodies.
fn carries_records(req: &Request) -> bool {
    req.derivation.is_none() && !matches!(req.subject, Subject::Checkpoints)
}

/// Whether the answer to `req` depends on who asks: record bodies, unless
/// the operator serves them to any peer (`peers`); Standard History, unless
/// the operator shares it.
fn needs_requester(req: &Request, responder: &Responder<'_>) -> bool {
    if carries_records(req) {
        responder.history_segments != "peers"
    } else {
        !responder.share_standard_history
    }
}

/// Step 5: the principal a signed request proves (`requester_auth::check`),
/// `None` for an unsigned one this answer does not need, or the refusal.
///
/// A malformed `requester_auth` is `request_malformed`. A signature that
/// fails any check is refused whatever was asked. An unsigned request is
/// refused when the answer depends on who asks. The refusal for a record is
/// `not_authorized`; for Standard History it is `policy_declined`, the one
/// refusal every requester that is not a proven counterparty gets (R-8).
/// None of these reads the ledger, so each is the same whether or not what
/// was asked for exists.
fn requester(
    map: &Map<String, Value>,
    form: Form,
    req: &Request,
    now: capsule_emit_evidence_request::time::UtcTime,
    responder: &Responder<'_>,
    asker: &mut Asker,
) -> Result<Option<Principal>, Reason> {
    let needed = needs_requester(req, responder);
    let refused = if carries_records(req) {
        Reason::NotAuthorized
    } else {
        Reason::PolicyDeclined
    };
    match form {
        Form::Malformed => Err(Reason::RequestMalformed),
        Form::Unsigned if needed => Err(refused),
        Form::Unsigned => Ok(None),
        Form::Signed { key_id, sig } => {
            let deadline = req.deadline.ok_or(Reason::RequestMalformed)?;
            let audience = map
                .get(requester_auth::AUDIENCE)
                .and_then(Value::as_str)
                .map(str::to_string);
            let checked = requester_auth::check(
                requester_auth::Channel::EvidenceRequest,
                map,
                &key_id,
                &sig,
                deadline,
                now,
                responder.keys,
                responder.replay,
                responder.node_id,
            );
            // The audience is recorded as checked: once the signature
            // verified.
            if checked
                .as_ref()
                .map_or_else(|f| f.verified_key().is_some(), |_| true)
            {
                asker.audience = audience;
            }
            match checked {
                Ok(principal) => {
                    asker.key_id = Some(principal.key_id.clone());
                    asker.node_id = Some(principal.node_id.clone());
                    asker.binding = Some("bound");
                    asker.replay = Some("checked");
                    Ok(Some(principal))
                }
                Err(failure) => {
                    asker.key_id = failure.verified_key().map(str::to_string);
                    asker.node_id = failure.verified_node().map(str::to_string);
                    match failure {
                        requester_auth::Failure::ClaimMismatch { .. } => {
                            asker.binding = Some("bound");
                            asker.replay = Some("checked");
                        }
                        requester_auth::Failure::Unbound { replay_checked, .. } => {
                            asker.binding = Some("unbound");
                            asker.replay = Some(if replay_checked {
                                "checked"
                            } else {
                                "unchecked"
                            });
                        }
                        _ => {}
                    }
                    // A valid signature under a key no host binds reads as
                    // an unsigned request where the answer needs no
                    // identity, and gets nothing more (ruling addendum §2),
                    // also when its nonce could not be held (u290 §2).
                    if failure.reads_as_unsigned() && !needed {
                        return Ok(None);
                    }
                    Err(if needed {
                        refused
                    } else {
                        Reason::NotAuthorized
                    })
                }
            }
        }
    }
}

/// The sharing-policy gate, except in the deliberately broken build CI uses
/// to show the parity run catches a responder that drops it.
fn policy_gate_applies() -> bool {
    !cfg!(feature = "mutant-evidence-skips-policy-gate")
}

/// Steps 6 to 10, over the index.
fn answer_from(
    req: &Request,
    digest: &str,
    principal: Option<&str>,
    responder: &Responder<'_>,
    view: &LogView<'_>,
) -> Result<Value, Reason> {
    let index = view.index;
    // 6. An answer that depends on who asks goes only to a counterparty,
    // decided before the subject is resolved: Standard History unless it is
    // shared, and record bodies unless they go to any peer. A node no record
    // names learns nothing of this log's size or cadence from which
    // refusal it gets.
    if needs_requester(req, responder)
        && policy_gate_applies()
        && !is_counterparty(index, principal)
    {
        return Err(if carries_records(req) {
            Reason::NotAuthorized
        } else {
            Reason::PolicyDeclined
        });
    }
    // 7. Resolve.
    let anchor = match resolve::resolve(req, view) {
        Resolution::Artifact(anchor) => anchor,
        Resolution::Refuse(reason) => return Err(reason),
    };
    if matches!(&req.derivation, Some(Derivation::Token(t)) if t == SERVED_SUMMARY_DERIVATION_TOKEN)
    {
        return served_summary_answer(req, digest, &anchor, responder, index);
    }

    // 8. Size, from the index.
    let anchor_position = anchor_position(index, &anchor).ok_or(Reason::CoverageUnsatisfiable)?;
    let covered = leaf_count(index.chain[anchor_position].mmr_size)
        .map_err(|_| Reason::CoverageUnsatisfiable)?;
    let records: Vec<u64> = if carries_records(req) {
        let count = match &req.subject {
            Subject::Range(a, b) => b.saturating_sub(*a).saturating_add(1),
            Subject::FullHistory => covered,
            Subject::Record(_) => 1,
            Subject::Correlation(id) => index.correlated(id).len() as u64,
            Subject::Exchange(half) => index.citing(half).len() as u64,
            Subject::Checkpoints => 0,
        };
        if count > MAX_ANSWER_RECORDS {
            return Err(Reason::PolicyDeclined);
        }
        match &req.subject {
            Subject::Range(a, b) => (*a..=*b).collect(),
            Subject::FullHistory => (0..covered).collect(),
            // Held, but after the anchor: not yet committed under it.
            Subject::Record(d) => match index.leaf_index_of(d) {
                Some(i) if i >= covered => return Err(Reason::CoverageUnsatisfiable),
                found => found.into_iter().collect(),
            },
            Subject::Correlation(id) => index.correlated(id).to_vec(),
            Subject::Exchange(half) => index.citing(half).to_vec(),
            Subject::Checkpoints => Vec::new(),
        }
        .into_iter()
        .filter(|&i| i < covered)
        .collect()
    } else {
        if anchor_position as u64 + 1 > MAX_ANSWER_CHECKPOINTS {
            return Err(Reason::PolicyDeclined);
        }
        Vec::new()
    };

    // 9. Who may have these records.
    if policy_gate_applies()
        && !may_serve_all(index, &records, principal, responder.history_segments)
    {
        return Err(Reason::NotAuthorized);
    }

    // 10. Build and sign.
    let limit = if carries_records(req) {
        MAX_ANSWER_RECORDS
    } else {
        MAX_ANSWER_CHECKPOINTS
    };
    let built = answer::build(
        req,
        digest,
        &anchor,
        view,
        limit,
        responder.signing_key,
        responder.now,
    )
    .map_err(|e| {
        if !matches!(
            e,
            BuildError::OverLimit | BuildError::RecordNotFound | BuildError::NoRecords
        ) {
            tracing::warn!(error = %e, "an evidence answer could not be built -- declining");
        }
        build_error_reason(&e)
    })?;
    wire(built.artifact, built.material, built.envelope)
}

/// The refusal for a build that did not complete.
fn build_error_reason(e: &BuildError) -> Reason {
    match e {
        BuildError::RecordNotFound | BuildError::NoRecords => Reason::NoSuchSubject,
        BuildError::AnchorNotFound => Reason::CoverageUnsatisfiable,
        BuildError::DerivationUnsupported => Reason::DerivationUnsupported,
        BuildError::OverLimit | BuildError::Malformed | BuildError::Proof(_) => {
            Reason::PolicyDeclined
        }
    }
}

fn anchor_position(index: &LedgerIndex, anchor: &ResolvedAnchor) -> Option<usize> {
    match anchor {
        ResolvedAnchor::Anchor(a) => index.chain.iter().position(|cp| cp.digest() == a.digest),
        ResolvedAnchor::ExchangeHalf(_) => index.chain.len().checked_sub(1),
    }
}

/// Whether some record of this node names `principal`, the node a signed
/// request proved it came from, as the other side of its exchange (a local
/// block or an owner-maintenance record names no counterparty it was
/// exchanged with, and padding names none).
fn is_counterparty(index: &LedgerIndex, principal: Option<&str>) -> bool {
    let Some(asker) = principal.map(str::trim).filter(|id| !id.is_empty()) else {
        return false;
    };
    index.leaves.iter().any(|leaf| {
        !leaf.padding && !leaf.never_leaves && leaf.counterparty.as_deref() == Some(asker)
    })
}

/// Whether every record in `records` may go to `principal` under
/// `history_segments` (see the module doc).
fn may_serve_all(
    index: &LedgerIndex,
    records: &[u64],
    principal: Option<&str>,
    history_segments: &str,
) -> bool {
    if records.is_empty() {
        return true;
    }
    if history_segments == "off" {
        return false;
    }
    let asker = principal.map(str::trim).filter(|id| !id.is_empty());
    records.iter().all(|&i| {
        let Some(leaf) = index.leaves.get(i as usize) else {
            return false;
        };
        !leaf.never_leaves
            && (history_segments == "peers"
                || (!leaf.padding && asker.is_some() && leaf.counterparty.as_deref() == asker))
    })
}

/// The `served_summary/1` derivation: a static export, answered only under
/// an `expected_pin` naming this node's latest checkpoint (the Python
/// reference's rule). A `min_freshness` request asks for an on-demand export, which this
/// node declines.
fn served_summary_answer(
    req: &Request,
    digest: &str,
    anchor: &ResolvedAnchor,
    responder: &Responder<'_>,
    index: &LedgerIndex,
) -> Result<Value, Reason> {
    if !matches!(req.coverage, Coverage::ExpectedPin(_)) {
        return Err(Reason::PolicyDeclined);
    }
    let latest = index.chain.last().ok_or(Reason::CoverageUnsatisfiable)?;
    if anchor_position(index, anchor) != Some(index.chain.len() - 1) {
        return Err(Reason::CoverageUnsatisfiable);
    }
    let covered = leaf_count(latest.mmr_size).map_err(|_| Reason::CoverageUnsatisfiable)?;
    let facts: Vec<_> = index.leaves.iter().map(|l| &l.facts).collect();
    let summary = served_summary::summary_value(
        &facts,
        &served_summary::Coverage {
            checkpoint: latest,
            covered,
        },
        &|id| index.leaf_index_of(id).is_some(),
    );
    let anchor_digest = latest.digest();
    let artifact = json!({
        "anchor": anchor_digest,
        "evidence_stream": latest.log_id,
        "subject": answer::subject_json(&req.subject),
        "derivation": SERVED_SUMMARY_DERIVATION_TOKEN,
        "served_summary": summary,
    });
    let material = json!({ "anchor_checkpoint": serde_json::to_value(latest).map_err(|_| Reason::PolicyDeclined)? });
    let artifact = jcs::to_string(&artifact).map_err(|_| Reason::PolicyDeclined)?;
    let material = jcs::to_string(&material).map_err(|_| Reason::PolicyDeclined)?;
    let envelope = sign_envelope(digest, &anchor_digest, artifact.as_bytes(), responder)?;
    wire(artifact.into_bytes(), material.into_bytes(), envelope)
}

/// The envelope of `answer::build`, over a derivation the crate does not
/// build: Ed25519 over RFC 8785 of `{request_digest, anchor,
/// artifact_digest, issued_at}`, with `key_id` and `sig` beside them.
fn sign_envelope(
    digest: &str,
    anchor: &str,
    artifact: &[u8],
    responder: &Responder<'_>,
) -> Result<Value, Reason> {
    let mut signed = Map::new();
    signed.insert("request_digest".into(), json!(digest));
    signed.insert("anchor".into(), json!(anchor));
    signed.insert(
        "artifact_digest".into(),
        json!(hex::encode(Sha256::digest(artifact))),
    );
    signed.insert("issued_at".into(), json!(responder.now));
    let body =
        jcs::to_string(&Value::Object(signed.clone())).map_err(|_| Reason::PolicyDeclined)?;
    let sig = responder.signing_key.sign(body.as_bytes());
    signed.insert(
        "key_id".into(),
        json!(hex::encode(
            responder.signing_key.verifying_key().to_bytes()
        )),
    );
    signed.insert("sig".into(), json!(hex::encode(sig.to_bytes())));
    Ok(Value::Object(signed))
}

fn wire(artifact: Vec<u8>, material: Vec<u8>, envelope: Value) -> Result<Value, Reason> {
    let text = |b: Vec<u8>| String::from_utf8(b).map_err(|_| Reason::PolicyDeclined);
    Ok(json!({
        "artifact": text(artifact)?,
        "material": text(material)?,
        "envelope": envelope,
    }))
}

/// Split a received artifact answer into what `answer::verify` takes:
/// `(envelope, artifact bytes, material bytes)`. `None` for anything else.
pub fn split_wire(answer: &Value) -> Option<(&Value, &[u8], &[u8])> {
    Some((
        answer.get("envelope")?,
        answer.get("artifact")?.as_str()?.as_bytes(),
        answer.get("material")?.as_str()?.as_bytes(),
    ))
}

/// Answer `request_bytes` (or, when `busy`, decline it unread), and log the
/// outcome to the "asked of you" log in `log_dir`. The one path the mesh
/// responder takes.
pub fn answer_and_log(
    request_bytes: &[u8],
    busy: bool,
    responder: &Responder<'_>,
    log_dir: Option<&Path>,
) -> Outcome {
    let requester_id = requester_id(request_bytes);
    let (outcome, asker) = if busy {
        (decline_busy(request_bytes, responder), Asker::default())
    } else {
        answer_as(request_bytes, responder)
    };
    let (status, reason) = outcome.log_status();
    let digest = request_digest(request_bytes);
    crate::received_log::append(
        log_dir,
        &crate::received_log::Entry {
            ts: responder.now,
            path: "evidence-request",
            requester_id: requester_id.as_deref(),
            subject_kind: outcome.subject_kind(),
            status,
            reason,
            request_digest: Some(&digest),
            principal_key_id: asker.key_id.as_deref(),
            principal_node_id: asker.node_id.as_deref(),
            principal_binding: asker.binding,
            audience: asker.audience.as_deref(),
            replay: asker.replay,
        },
    );
    outcome
}

/// Now, as every `issued_at` is written.
pub fn now_utc() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// A signed `policy_declined` refusal of `request_bytes`, given without
/// reading the ledger: this node is already answering as many requests as it
/// takes at once.
pub fn decline_busy(request_bytes: &[u8], responder: &Responder<'_>) -> Outcome {
    let digest = request_digest(request_bytes);
    match refusal::sign(
        &digest,
        Reason::PolicyDeclined,
        responder.now,
        responder.signing_key,
    ) {
        Ok(r) => Outcome::Refused {
            wire: r.to_json(),
            reason: Reason::PolicyDeclined,
            subject_kind: None,
        },
        Err(e) => Outcome::Unanswerable(format!("could not sign a policy_declined refusal: {e}")),
    }
}

// ---------------------------------------------------------------------------
// The requester's side: is what came back evidence?
// ---------------------------------------------------------------------------

/// What a requester makes of a response to the request it sent (`sent`,
/// exactly the bytes written) from the responder whose key it expects:
///
/// - `{"state": "refusal", "reason"}`: a refusal that key signed, of this
///   request (`refusal::verify_for`);
/// - `{"state": "artifact", "anchor", "records", "checkpoints"}`: an answer
///   that verifies (`answer::verify`, or for `served_summary/1` its envelope,
///   artifact digest and anchor checkpoint);
/// - `{"state": "not_evidence", "why"}`: anything else. It is not a refusal
///   and not an answer (§4.1).
///
/// `witness_keys` are the witness keys the operator configured, by witness
/// URL (`checkpoint_cadence::active_witness_keys`): configured keys only,
/// never a key this node pinned on first contact with a witness. For an
/// answer `answer::verify` checks, each receipt is checked under the key
/// configured for ITS witness's URL (URLs normalized, and duplicates resolved,
/// as `witness_status` does), never under another witness's key. A receipt
/// that fails only because of a configured key makes the answer not
/// evidence with `reason: "witness_receipt_unverified"`; one that fails with
/// no key at all fails as before. With keys configured, such an artifact also
/// reports `witnessed` (receipts verified) and each receipt's status. With
/// none, the result is exactly what it was before keys were passed. A
/// `served_summary/1` answer's receipts are not checked here.
pub fn verify_response(
    sent: &[u8],
    response: &Value,
    key: &ed25519_dalek::VerifyingKey,
    witness_keys: &std::collections::BTreeMap<String, String>,
) -> Value {
    let not_evidence = |why: String| json!({"state": "not_evidence", "why": why});
    let digest = request_digest(sent);
    if response.get("reason").is_some() {
        return match refusal::verify_for(response, key, &digest) {
            Ok(reason) => json!({"state": "refusal", "reason": reason.token()}),
            Err(e) => not_evidence(e.to_string()),
        };
    }
    let Ok(req) = request::parse_json(sent) else {
        return not_evidence("the request sent is not a -00 request".into());
    };
    let Some((envelope, artifact, material)) = split_wire(response) else {
        return not_evidence("neither a refusal nor an artifact answer".into());
    };
    if matches!(&req.derivation, Some(Derivation::Token(t)) if t == SERVED_SUMMARY_DERIVATION_TOKEN)
    {
        return match verify_served_summary(envelope, artifact, material, &req, &digest, key) {
            Ok(anchor) => {
                json!({"state": "artifact", "anchor": anchor, "records": 0, "checkpoints": 0})
            }
            Err(why) => not_evidence(why.into()),
        };
    }
    let witness_key = |ts_url: &str| {
        if cfg!(feature = "mutant-answer-ignores-witness-keys") {
            return None;
        }
        configured_witness_key(witness_keys, ts_url)
    };
    match answer::verify(
        envelope,
        artifact,
        material,
        &req,
        &digest,
        key,
        &evidence_log::record_digest,
        &witness_key,
    ) {
        Ok(v) => {
            let mut out = json!({
                "state": "artifact",
                "anchor": v.anchor.digest(),
                "records": v.records.iter().map(|r| r.leaf_index).collect::<Vec<_>>(),
                "checkpoints": v.checkpoints.len(),
            });
            if !witness_keys.is_empty() {
                out["witnessed"] = json!(v
                    .receipts
                    .iter()
                    .filter(|r| r.status == answer::ReceiptStatus::Verified)
                    .count());
                out["receipts"] = v
                    .receipts
                    .iter()
                    .map(|r| {
                        json!({
                            "checkpoint": r.checkpoint_digest,
                            "ts_url": r.ts_url,
                            "status": match r.status {
                                answer::ReceiptStatus::Verified => "verified",
                                answer::ReceiptStatus::NoKey => "no_key",
                                answer::ReceiptStatus::Stub => "stub",
                            },
                        })
                    })
                    .collect();
            }
            out
        }
        // A receipt failing under a configured key: the same answer passes
        // with no keys, so the key is what refused it.
        Err(e @ (answer::VerifyError::ReceiptBinding | answer::VerifyError::ReceiptMalformed))
            if !witness_keys.is_empty()
                && answer::verify(
                    envelope,
                    artifact,
                    material,
                    &req,
                    &digest,
                    key,
                    &evidence_log::record_digest,
                    &|_| None,
                )
                .is_ok() =>
        {
            json!({"state": "not_evidence", "why": e.to_string(), "reason": "witness_receipt_unverified"})
        }
        Err(e) => not_evidence(e.to_string()),
    }
}

/// The key the operator configured for the witness at `ts_url`: the entry
/// whose URL is the same once normalized (`witness_status::normalize_url`).
/// A value that is not an Ed25519 key is no key.
fn configured_witness_key(
    witness_keys: &std::collections::BTreeMap<String, String>,
    ts_url: &str,
) -> Option<ed25519_dalek::VerifyingKey> {
    let normalized: std::collections::BTreeMap<String, &String> = witness_keys
        .iter()
        .map(|(url, key)| (crate::witness_status::normalize_url(url), key))
        .collect();
    let hex_key = normalized.get(&crate::witness_status::normalize_url(ts_url))?;
    let bytes: [u8; 32] = hex::decode(hex_key).ok()?.try_into().ok()?;
    ed25519_dalek::VerifyingKey::from_bytes(&bytes).ok()
}

/// The checks `answer::verify` makes of any answer, for the served-summary
/// derivation it does not know: the envelope is `key`'s and names this
/// request, the artifact is the one it signed, and the anchor checkpoint is
/// `key`'s, is the one named, and is the pinned one. Returns the anchor.
fn verify_served_summary(
    envelope: &Value,
    artifact: &[u8],
    material: &[u8],
    req: &Request,
    digest: &str,
    key: &ed25519_dalek::VerifyingKey,
) -> Result<String, &'static str> {
    let text = |k: &str| {
        envelope
            .get(k)
            .and_then(Value::as_str)
            .ok_or("the envelope is malformed")
    };
    let (env_request, env_anchor, env_artifact, issued_at) = (
        text("request_digest")?,
        text("anchor")?,
        text("artifact_digest")?,
        text("issued_at")?,
    );
    if text("key_id")? != hex::encode(key.to_bytes()) {
        return Err("the envelope is not signed by the expected responder key");
    }
    let body = jcs::to_string(&json!({
        "request_digest": env_request, "anchor": env_anchor,
        "artifact_digest": env_artifact, "issued_at": issued_at,
    }))
    .map_err(|_| "the envelope is malformed")?;
    let sig: [u8; 64] = hex::decode(text("sig")?)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or("the envelope is malformed")?;
    key.verify_strict(body.as_bytes(), &ed25519_dalek::Signature::from_bytes(&sig))
        .map_err(|_| "the envelope is not signed by the expected responder key")?;
    if env_request != digest {
        return Err("the envelope names a different request");
    }
    if hex::encode(Sha256::digest(artifact)) != env_artifact {
        return Err("the artifact does not match the envelope's artifact digest");
    }
    let art: Value = serde_json::from_slice(artifact).map_err(|_| "the artifact is malformed")?;
    if art.get("anchor").and_then(Value::as_str) != Some(env_anchor)
        || art.get("subject") != Some(&answer::subject_json(&req.subject))
        || art.get("derivation").and_then(Value::as_str) != Some(SERVED_SUMMARY_DERIVATION_TOKEN)
        || !art.get("served_summary").is_some_and(Value::is_object)
    {
        return Err("the artifact is not for the requested subject or derivation");
    }
    let mat: Value = serde_json::from_slice(material).map_err(|_| "the material is malformed")?;
    let cp: cll::checkpoint::CheckpointRecord = mat
        .get("anchor_checkpoint")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or("the anchor checkpoint is malformed")?;
    if cp.digest() != env_anchor
        || cp.key_id != hex::encode(key.to_bytes())
        || !cp.verify_signature_offline()
    {
        return Err("the anchor checkpoint is not the responder's, or not the one named");
    }
    match &req.coverage {
        Coverage::ExpectedPin(pin) if pin == env_anchor => Ok(env_anchor.to_string()),
        _ => Err("the anchor does not satisfy the requested coverage"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_request_parity::{
        announced_keys as announced, asker_key, corpus_checkpoints, corpus_ledger, sealed,
        write_jsonl, NODE_KEY_SEED,
    };

    use crate::requester_auth::{Keys, ReplayCache};

    const NOW: &str = "2026-09-29T00:00:00Z";

    /// This node's own mesh id, which a signed request's audience names.
    const RESPONDER: &str = crate::evidence_request_parity::RESPONDER_NODE;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&NODE_KEY_SEED)
    }

    fn key_id_of(key: &SigningKey) -> String {
        hex::encode(key.verifying_key().to_bytes())
    }

    fn now() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into()
    }

    /// `request` signed with `key` at [`NOW`] (`requester_auth::sign`).
    fn signed_with(key: &SigningKey, request: &Value) -> Vec<u8> {
        let mut request = request.clone();
        requester_auth::sign(
            requester_auth::Channel::EvidenceRequest,
            &mut request,
            key,
            now(),
            RESPONDER,
        );
        serde_json::to_vec(&request).unwrap()
    }

    /// `request` signed by the node `node`.
    fn signed_by(node: &str, request: &Value) -> Vec<u8> {
        signed_with(&asker_key(node), request)
    }

    /// Asked of a node with these settings and a fresh replay cache.
    fn ask_with(dir: &Path, bytes: &[u8], segments: &str, share_standard_history: bool) -> Outcome {
        ask_cached(
            dir,
            bytes,
            segments,
            share_standard_history,
            &ReplayCache::default(),
        )
    }

    fn ask_cached(
        dir: &Path,
        bytes: &[u8],
        segments: &str,
        share_standard_history: bool,
        replay: &ReplayCache,
    ) -> Outcome {
        let key = key();
        let announced = announced();
        answer(
            bytes,
            &Responder {
                ledger_dir: dir,
                signing_key: &key,
                now: NOW,
                history_segments: segments,
                share_standard_history,
                keys: Keys {
                    announced: &announced,
                    configured: None,
                },
                replay,
                node_id: Some(RESPONDER),
            },
        )
    }

    /// A ledger directory holding `ledger` and checkpoints cut at `cuts`
    /// (the corpus's own cutter, which cuts after 16 and 300 lines).
    fn node(ledger: &[Value]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_jsonl(&dir.path().join("capsules.jsonl"), ledger);
        let cps: Vec<Value> = corpus_checkpoints(ledger, &key())
            .into_iter()
            .map(|cp| serde_json::to_value(cp).unwrap())
            .collect();
        write_jsonl(&dir.path().join("checkpoints.jsonl"), &cps);
        dir
    }

    /// Asked with Standard History shared, as every node answered before it
    /// had the switch: these tests are about everything else.
    fn ask(dir: &Path, bytes: &[u8], segments: &str) -> Outcome {
        ask_with(dir, bytes, segments, true)
    }

    fn refused(outcome: &Outcome) -> Option<Reason> {
        match outcome {
            Outcome::Refused { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    /// No request, however formed, panics the responder or goes unanswered:
    /// each is a signed refusal bound to its own bytes.
    #[test]
    fn hostile_requests_are_signed_refusals_never_panics() {
        let dir = node(&corpus_ledger());
        let deep = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
        let long_id = format!(
            r#"{{"subject":{{"correlation":"{}"}},"coverage":{{"min_freshness":1}}}}"#,
            "x".repeat(15_000)
        );
        let inputs: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"null".to_vec(),
            b"[]".to_vec(),
            deep.into_bytes(),
            vec![0xff, 0xfe, 0x00],
            br#"{"subject":{"range":[0,18446744073709551615]},"coverage":{"min_freshness":1}}"#.to_vec(),
            br#"{"subject":{"range":[5,2]},"coverage":{"min_freshness":1}}"#.to_vec(),
            br#"{"subject":{"range":[-1,2]},"coverage":{"min_freshness":1}}"#.to_vec(),
            br#"{"subject":{"record":"a","range":[0,1]},"coverage":{"min_freshness":1}}"#.to_vec(),
            br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":18446744073709551615}}"#.to_vec(),
            br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":"not a time"}}"#.to_vec(),
            long_id.into_bytes(),
        ];
        for bytes in inputs {
            let outcome = ask(dir.path(), &bytes, "peers");
            let Outcome::Refused { wire, reason, .. } = &outcome else {
                panic!(
                    "{:?} was not refused: {outcome:?}",
                    String::from_utf8_lossy(&bytes)
                );
            };
            assert_eq!(
                refusal::verify_for(wire, &key().verifying_key(), &request_digest(&bytes)),
                Ok(*reason),
                "signed and bound to the request"
            );
        }
    }

    /// A `requester_id` that is not text is no identity at all: ignored, not
    /// a reason to refuse what needs none.
    #[test]
    fn a_requester_id_that_is_not_text_is_ignored() {
        let dir = node(&corpus_ledger());
        let bytes =
            br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":1},"requester_id":5}"#;
        assert_eq!(requester_id(bytes), None);
        assert!(matches!(
            ask(dir.path(), bytes, "peers"),
            Outcome::Answered { .. }
        ));
    }

    /// A refusal names the request it refuses: it does not verify as the
    /// answer to any other request.
    #[test]
    fn a_refusal_is_bound_to_its_request_digest() {
        let dir = node(&corpus_ledger());
        let bytes = br#"{"subject":{"record":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"coverage":{"min_freshness":1}}"#;
        let Outcome::Refused { wire, .. } = ask(dir.path(), bytes, "peers") else {
            panic!("refused");
        };
        let other = request_digest(b"another request");
        assert_eq!(
            refusal::verify_for(&wire, &key().verifying_key(), &other),
            Err(refusal::VerifyError::WrongRequest)
        );
    }

    /// Size and the `off` switch are decided before the ledger is read: with
    /// no ledger at all, the answer is still the policy refusal, not a
    /// failure to read one.
    #[test]
    fn cheap_refusals_come_before_the_ledger_is_read() {
        let missing = tempfile::tempdir().unwrap();
        let nowhere = missing.path().join("no-such-ledger");
        let big = vec![b' '; MAX_REQUEST_BYTES + 1];
        assert_eq!(
            refused(&ask(&nowhere, &big, "peers")),
            Some(Reason::PolicyDeclined)
        );
        let record = format!(
            r#"{{"subject":{{"record":"{}"}},"coverage":{{"min_freshness":1}}}}"#,
            "a".repeat(64)
        );
        assert_eq!(
            refused(&ask(&nowhere, record.as_bytes(), "off")),
            Some(Reason::NotAuthorized)
        );
    }

    /// An answer that would not fit the stream the requester reads is
    /// declined, not sent to be cut off.
    #[test]
    fn an_answer_over_the_stream_cap_is_declined() {
        let mut ledger = vec![sealed(
            json!({"action_type": "fyi", "blob": "x".repeat(MAX_ANSWER_BYTES)}),
        )];
        let rest = corpus_ledger();
        ledger.extend(rest.into_iter().skip(1).take(15));
        let dir = node(&ledger);
        let record = format!(
            r#"{{"subject":{{"record":"{}"}},"coverage":{{"min_freshness":1}}}}"#,
            ledger[0]["capsule_id"].as_str().unwrap()
        );
        assert_eq!(
            refused(&ask(dir.path(), record.as_bytes(), "peers")),
            Some(Reason::PolicyDeclined)
        );
    }

    /// The live paid run: the provider's served half carries the host's
    /// placeholder nonce, not the one the payer forwarded, so the payer's ask
    /// by its nonce is a signed "no such record". Asked by the exchange's two
    /// digests, the provider answers with that half, proven in its log; and
    /// nobody is answered by the placeholder.
    #[test]
    fn a_host_served_half_is_answered_by_its_digests_when_its_nonce_is_the_placeholder() {
        let (req, resp) = ("4b".repeat(32), "33".repeat(32));
        let mut ledger = corpus_ledger();
        ledger[3] = sealed(json!({
            "action_type": "fyi",
            "effect": {"status": "confirmed", "request_digest": req, "response_digest": resp},
            "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {
                "role": "served",
                "client_nonce": crate::evidence_log::HOST_SERVED_NO_NONCE,
                "client_nonce_source": "host_served_observed",
            }}},
        }));
        let dir = node(&ledger);
        let by = |id: &str| {
            json!({"subject": {"correlation": id}, "coverage": {"min_freshness": 1}}).to_string()
        };

        let by_nonce = by("payer-forwarded-nonce");
        assert_eq!(
            refused(&ask(dir.path(), by_nonce.as_bytes(), "peers")),
            Some(Reason::NoSuchSubject)
        );
        let by_placeholder = by(crate::evidence_log::HOST_SERVED_NO_NONCE);
        assert_eq!(
            refused(&ask(dir.path(), by_placeholder.as_bytes(), "peers")),
            Some(Reason::NoSuchSubject)
        );
        let by_one_wrong_digest = by(&crate::evidence_log::exchange_binding(&req, &req));
        assert_eq!(
            refused(&ask(dir.path(), by_one_wrong_digest.as_bytes(), "peers")),
            Some(Reason::NoSuchSubject)
        );

        let by_digests = by(&crate::evidence_log::exchange_binding(&req, &resp));
        let Outcome::Answered { wire, .. } = ask(dir.path(), by_digests.as_bytes(), "peers") else {
            panic!("answered");
        };
        let verification = verify_response(
            by_digests.as_bytes(),
            &wire,
            &key().verifying_key(),
            &Default::default(),
        );
        assert_eq!(verification["state"], json!("artifact"), "{verification}");
        assert_eq!(verification["records"], json!([3]));
    }

    /// A requester that is too busy gets a signed refusal, never a wait.
    #[test]
    fn a_busy_node_declines_with_a_signed_refusal() {
        let key = key();
        let (announced, replay) = (announced(), ReplayCache::default());
        let responder = Responder {
            ledger_dir: Path::new("/nonexistent"),
            signing_key: &key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: true,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER),
        };
        let bytes = br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":1}}"#;
        let Outcome::Refused { wire, reason, .. } = decline_busy(bytes, &responder) else {
            panic!("declined");
        };
        assert_eq!(reason, Reason::PolicyDeclined);
        assert!(refusal::verify_for(&wire, &key.verifying_key(), &request_digest(bytes)).is_ok());
    }

    /// A node whose clock cannot be written as an `issued_at` signs nothing.
    #[test]
    fn a_bad_clock_answers_nothing() {
        let key = key();
        let (announced, replay) = (announced(), ReplayCache::default());
        let responder = Responder {
            ledger_dir: Path::new("/nonexistent"),
            signing_key: &key,
            now: "yesterday",
            history_segments: "peers",
            share_standard_history: true,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER),
        };
        assert!(matches!(
            answer(b"{}", &responder),
            Outcome::Unanswerable(_)
        ));
        assert_eq!(answer(b"{}", &responder).wire_bytes(), None);
    }

    /// This node has no referee: a referee's `adjudicate` request is answered
    /// in-process with a signed refusal, never a verdict, never silence.
    #[test]
    fn an_adjudicate_request_is_refused_signed_never_a_verdict() {
        let dir = tempfile::tempdir().unwrap();
        let request = json!({
            "v": 1,
            "subject": {"kind": "adjudicate", "halves": ["a".repeat(64), "b".repeat(64)]},
            "requester_id": "peer-x",
        })
        .to_string();
        let outcome = ask(dir.path(), request.as_bytes(), "peers");
        assert!(refused(&outcome).is_some(), "refused");
        let wire = outcome.wire_bytes().expect("something signed is sent");
        let reply: Value = serde_json::from_slice(&wire).unwrap();
        assert!(reply.get("adjudication_verdict").is_none());
        let key_id = hex::encode(key().verifying_key().to_bytes());
        assert_eq!(reply["key_id"], json!(key_id));
        let check = refusal::verify_for(
            &reply,
            &key().verifying_key(),
            &request_digest(request.as_bytes()),
        );
        assert!(
            check.is_ok(),
            "the refusal verifies under this node's key: {check:?}"
        );
    }

    const WITNESS: &str = "https://witness.example";

    /// A real witness receipt for `entry` under `key`: a COSE_Sign1 (EdDSA)
    /// over the root of a one-leaf RFC 9162 tree, with its inclusion proof.
    fn mint_receipt(entry: &[u8; 32], key: &SigningKey) -> String {
        use base64::Engine as _;
        use coset::cbor::Value as Cbor;
        use coset::{iana, CoseSign1Builder, HeaderBuilder, TaggedCborSerializable};
        let mut leaf = Sha256::new();
        leaf.update([0u8]);
        leaf.update(entry);
        let root = leaf.finalize();
        let mut proof = Vec::new();
        coset::cbor::ser::into_writer(
            &Cbor::Array(vec![1.into(), 0.into(), Cbor::Array(vec![])]),
            &mut proof,
        )
        .unwrap();
        let protected = HeaderBuilder::new()
            .algorithm(iana::Algorithm::EdDSA)
            .value(395, 1.into())
            .build();
        let unprotected = HeaderBuilder::new()
            .value(
                396,
                Cbor::Map(vec![((-1).into(), Cbor::Array(vec![Cbor::Bytes(proof)]))]),
            )
            .build();
        let receipt = CoseSign1Builder::new()
            .protected(protected)
            .unprotected(unprotected)
            .create_detached_signature(&root, &[], |tbs| key.sign(tbs).to_bytes().to_vec())
            .build();
        base64::engine::general_purpose::STANDARD.encode(receipt.to_tagged_vec().unwrap())
    }

    /// A node whose every checkpoint carries a receipt from [`WITNESS`],
    /// signed by `witness`, for that checkpoint's own entry.
    fn witnessed_node(witness: &SigningKey) -> tempfile::TempDir {
        node_with_witnesses(|entry| {
            json!([{
                "ts_url": WITNESS, "entry_hash": hex::encode(entry),
                "receipt_b64": mint_receipt(entry, witness),
                "leaf_index": 0, "tree_size": 1,
            }])
        })
    }

    /// A node whose every checkpoint carries `witnesses(its entry)`.
    fn node_with_witnesses(witnesses: impl Fn(&[u8; 32]) -> Value) -> tempfile::TempDir {
        let dir = node(&corpus_ledger());
        let path = dir.path().join("checkpoints.jsonl");
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| {
                let mut cp: Value = serde_json::from_str(l).unwrap();
                let record: cll::checkpoint::CheckpointRecord =
                    serde_json::from_value(cp.clone()).unwrap();
                let entry: [u8; 32] = Sha256::digest(hex::decode(record.digest()).unwrap()).into();
                cp["witnesses"] = witnesses(&entry);
                cp
            })
            .collect();
        write_jsonl(&path, &lines);
        dir
    }

    const CHECKPOINTS: &[u8] =
        br#"{"subject":{"checkpoints":null},"coverage":{"min_freshness":1}}"#;

    fn verified_with(dir: &Path, keys: &[(&str, &SigningKey)]) -> Value {
        let Outcome::Answered { wire, .. } = ask(dir, CHECKPOINTS, "peers") else {
            panic!("answered");
        };
        let keys = keys
            .iter()
            .map(|(url, k)| (url.to_string(), hex::encode(k.verifying_key().to_bytes())))
            .collect();
        verify_response(CHECKPOINTS, &wire, &key().verifying_key(), &keys)
    }

    #[test]
    fn a_receipt_under_its_witnesss_configured_key_is_counted_witnessed() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let dir = witnessed_node(&witness);
        let v = verified_with(dir.path(), &[(WITNESS, &witness)]);
        assert_eq!(v["state"], json!("artifact"), "{v}");
        let receipts = v["receipts"].as_array().unwrap();
        assert!(!receipts.is_empty());
        assert!(receipts
            .iter()
            .all(|r| r["status"] == json!("verified") && r["ts_url"] == json!(WITNESS)));
        assert_eq!(v["witnessed"], json!(receipts.len()));
        // The configured URL is matched as the cadence matches it: normalized.
        let v = verified_with(dir.path(), &[("HTTPS://Witness.Example/", &witness)]);
        assert_eq!(v["witnessed"], json!(receipts.len()), "{v}");
    }

    #[test]
    fn a_receipt_that_does_not_verify_under_the_configured_key_is_refused_by_name() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let impostor = SigningKey::from_bytes(&[22; 32]);
        let dir = witnessed_node(&witness);
        let v = verified_with(dir.path(), &[(WITNESS, &impostor)]);
        assert_eq!(v["state"], json!("not_evidence"), "{v}");
        assert_eq!(v["reason"], json!("witness_receipt_unverified"));
        assert_eq!(
            v["why"],
            json!(answer::VerifyError::ReceiptBinding.to_string())
        );
    }

    #[test]
    fn only_the_key_configured_for_that_witness_is_used() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let dir = witnessed_node(&witness);
        // The right key, configured for another witness: never tried on this one.
        let v = verified_with(dir.path(), &[("https://other.example", &witness)]);
        assert_eq!(v["state"], json!("artifact"), "{v}");
        assert_eq!(v["witnessed"], json!(0));
        assert!(v["receipts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == json!("no_key")));
    }

    #[test]
    fn with_no_witness_keys_the_result_is_what_it_was_before() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let dir = witnessed_node(&witness);
        let v = verified_with(dir.path(), &[]);
        assert_eq!(v["state"], json!("artifact"), "{v}");
        let mut members: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        members.sort_unstable();
        assert_eq!(members, ["anchor", "checkpoints", "records", "state"]);
    }

    #[test]
    fn a_stub_receipt_is_listed_as_a_stub_and_not_counted() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let dir = node_with_witnesses(|entry| {
            json!([{"ts_url": WITNESS, "entry_hash": hex::encode(entry), "receipt_b64": "",
                    "leaf_index": 0, "tree_size": 1, "is_stub": true}])
        });
        let v = verified_with(dir.path(), &[(WITNESS, &witness)]);
        assert_eq!(v["state"], json!("artifact"), "{v}");
        assert_eq!(v["witnessed"], json!(0));
        assert!(v["receipts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == json!("stub")));
    }

    #[test]
    fn a_bad_receipt_from_a_witness_with_no_key_fails_as_before_not_as_a_key_failure() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        // A receipt from an unkeyed witness, carried on the wrong checkpoint.
        let dir = node_with_witnesses(|_| {
            json!([{"ts_url": "https://unkeyed.example", "entry_hash": "00".repeat(32),
                    "receipt_b64": mint_receipt(&[0; 32], &witness), "leaf_index": 0, "tree_size": 1}])
        });
        let v = verified_with(dir.path(), &[(WITNESS, &witness)]);
        assert_eq!(v["state"], json!("not_evidence"), "{v}");
        assert_eq!(v.get("reason"), None, "no configured key was involved: {v}");
        assert_eq!(
            verified_with(dir.path(), &[]),
            v,
            "the same answer as with no keys"
        );
    }

    #[test]
    fn two_entries_for_one_witness_resolve_as_the_witness_status_rows_do() {
        let witness = SigningKey::from_bytes(&[21; 32]);
        let impostor = SigningKey::from_bytes(&[22; 32]);
        let dir = witnessed_node(&witness);
        // Both normalize to WITNESS; as in `witness_status`, the entry that
        // comes last in the map's order wins ("https" sorts after "HTTPS").
        let v = verified_with(
            dir.path(),
            &[("HTTPS://Witness.Example", &impostor), (WITNESS, &witness)],
        );
        assert_eq!(v["state"], json!("artifact"), "{v}");
        assert!(v["witnessed"].as_u64().unwrap() > 0);
    }

    /// Asked of a node with the default settings: both switches unset,
    /// resolved by the plugin's own readers but not from this process's env
    /// or config file, so an operator's settings cannot change the result.
    /// Records to counterparties, Standard History not shared.
    fn ask_by_default(dir: &Path, bytes: &[u8]) -> Outcome {
        ask_with(
            dir,
            bytes,
            crate::share_policy::history_segments_for(None),
            crate::share_policy::shares_standard_history_for(None),
        )
    }

    /// The three Standard History requests, each pinned to the corpus's
    /// newest checkpoint.
    fn standard_history_requests() -> Vec<Value> {
        let ledger = corpus_ledger();
        let pin = corpus_checkpoints(&ledger, &key())[1].digest();
        let coverage = json!({"expected_pin": pin});
        vec![
            json!({"subject": {"checkpoints": null}, "coverage": coverage}),
            json!({"subject": {"full_history": null}, "coverage": coverage, "derivation": "history_card/1"}),
            json!({"subject": {"full_history": null}, "coverage": coverage, "derivation": "served_summary/1"}),
        ]
    }

    /// The refusal of exactly these bytes, now: nothing else goes into it.
    fn refusal_of(bytes: &[u8], reason: Reason) -> Vec<u8> {
        let wire = refusal::sign(&request_digest(bytes), reason, NOW, &key())
            .unwrap()
            .to_json();
        serde_json::to_vec(&wire).unwrap()
    }

    /// PRD T7, the default half (R-8): checkpoints, `history_card/1` and
    /// `served_summary/1` asked by anyone who has not proven it is a
    /// counterparty are refused `policy_declined`, signed, and the refusal is
    /// the refusal of the request's own bytes, now, and nothing else: an
    /// unsigned asker naming no id, one naming a counterparty's id, two
    /// signed strangers, a stranger signing while naming a counterparty, and
    /// the same bytes asked of a node that holds nothing at all. (Two askers
    /// never send the same signed bytes; unsigned bytes are the same
    /// whoever sends them, and get the same refusal.)
    #[test]
    fn standard_history_to_a_non_counterparty_is_the_refusal_of_its_own_bytes_whoever_asks() {
        let dir = node(&corpus_ledger());
        let empty = node(&[]);
        for request in standard_history_requests() {
            let mut claims_a = request.clone();
            claims_a["requester_id"] = json!("node-a");
            let asks: Vec<Vec<u8>> = vec![
                serde_json::to_vec(&request).unwrap(),
                serde_json::to_vec(&claims_a).unwrap(),
                signed_by("node-y", &request),
                signed_by("node-z", &request),
                signed_by("node-y", &claims_a),
            ];
            for bytes in &asks {
                for ledger in [dir.path(), empty.path()] {
                    let outcome = ask_by_default(ledger, bytes);
                    assert_eq!(
                        outcome.wire_bytes(),
                        Some(refusal_of(bytes, Reason::PolicyDeclined)),
                        "{request}: {outcome:?}"
                    );
                }
            }
        }
    }

    /// Two strangers asking the same thing get refusals that differ only
    /// where their requests do: the request digest, and the signature over
    /// it.
    #[test]
    fn two_askers_refusals_differ_only_by_their_own_request() {
        let dir = node(&corpus_ledger());
        for request in standard_history_requests() {
            let refusal = |node: &str| match ask_by_default(dir.path(), &signed_by(node, &request))
            {
                Outcome::Refused { wire, .. } => wire,
                other => panic!("{other:?}"),
            };
            let (mut a, mut b) = (refusal("node-y"), refusal("node-z"));
            for wire in [&mut a, &mut b] {
                let members = wire.as_object_mut().unwrap();
                members.remove("request_digest").expect("binds its request");
                members.remove("sig").expect("signed");
            }
            assert_eq!(a, b, "{request}");
        }
    }

    /// The default half's other side: a node that proves it is one some
    /// record names as the other side of an exchange is answered all three.
    #[test]
    fn a_counterparty_is_still_answered_standard_history() {
        let dir = node(&corpus_ledger());
        for request in standard_history_requests() {
            let bytes = signed_by("node-a", &request);
            let outcome = ask_by_default(dir.path(), &bytes);
            let Outcome::Answered { wire, .. } = &outcome else {
                panic!("{request}: {outcome:?}");
            };
            let verified =
                verify_response(&bytes, wire, &key().verifying_key(), &Default::default());
            assert_eq!(
                verified["state"],
                json!("artifact"),
                "{request}: {verified}"
            );
        }
    }

    /// A counterparty is a node some record names as the other side, by
    /// its whole id; an empty or absent id is nobody's.
    #[test]
    fn a_counterparty_is_named_by_a_record() {
        let dir = node(&corpus_ledger());
        let key_id = hex::encode(key().verifying_key().to_bytes());
        evidence_log::with_index(dir.path(), &key_id, |index| {
            assert!(is_counterparty(index, Some("node-a")));
            assert!(is_counterparty(index, Some(" node-b ")));
            assert!(!is_counterparty(index, Some("node")));
            assert!(!is_counterparty(index, Some("node-z")));
            assert!(!is_counterparty(index, Some(" ")));
            assert!(!is_counterparty(index, None));
            Ok::<_, std::io::Error>(())
        })
        .unwrap()
        .unwrap();
    }

    /// `share_standard_history` on restores what every node did before:
    /// Standard History to anyone, signed or not, named or not.
    #[test]
    fn sharing_standard_history_answers_anyone_as_before() {
        let dir = node(&corpus_ledger());
        for request in standard_history_requests() {
            let mut names_z = request.clone();
            names_z["requester_id"] = json!("node-z");
            for bytes in [
                serde_json::to_vec(&request).unwrap(),
                serde_json::to_vec(&names_z).unwrap(),
                signed_by("node-z", &request),
            ] {
                let outcome = ask_with(dir.path(), &bytes, "counterparties", true);
                assert!(
                    matches!(outcome, Outcome::Answered { .. }),
                    "{request}: {outcome:?}"
                );
            }
        }
    }

    /// The switch governs Standard History only: record bodies stay under
    /// `share_history_segments`, whichever way it is set. node-b proves who
    /// it is and is some record's counterparty, just not this one's.
    #[test]
    fn sharing_standard_history_never_releases_a_record() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let record = record_request(&ledger, 0);
        let outcome = ask_with(
            dir.path(),
            &signed_by("node-b", &record),
            "counterparties",
            true,
        );
        assert_eq!(refused(&outcome), Some(Reason::NotAuthorized));
    }

    /// A request for record `i` of `ledger`, pinned to its newest checkpoint.
    fn record_request(ledger: &[Value], i: usize) -> Value {
        let pin = corpus_checkpoints(ledger, &key())[1].digest();
        json!({
            "subject": {"record": ledger[i]["capsule_id"]},
            "coverage": {"expected_pin": pin},
        })
    }

    /// The corpus record node-a is the counterparty of, and a request for it.
    fn node_a_record(ledger: &[Value]) -> Value {
        let dir = node(ledger);
        let key_id = hex::encode(key().verifying_key().to_bytes());
        let i = evidence_log::with_index(dir.path(), &key_id, |index| {
            index
                .leaves
                .iter()
                .position(|leaf| leaf.counterparty.as_deref() == Some("node-a"))
        })
        .unwrap()
        .expect("the corpus names node-a");
        record_request(ledger, i)
    }

    // -----------------------------------------------------------------------
    // Who asks (requester_auth): the ruling's tests, T3 and D3.
    // -----------------------------------------------------------------------

    /// A valid signed request from the record's counterparty is answered,
    /// and the answer verifies.
    #[test]
    fn a_signed_request_from_the_counterparty_is_answered() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let bytes = signed_by("node-a", &node_a_record(&ledger));
        let Outcome::Answered { wire, .. } = ask_by_default(dir.path(), &bytes) else {
            panic!("answered");
        };
        let verified = verify_response(&bytes, &wire, &key().verifying_key(), &Default::default());
        assert_eq!(verified["state"], json!("artifact"), "{verified}");
    }

    /// evidencebook#69 finding 1, end to end: this node seals a host-served
    /// exchange node-a asked for (the host names node-a as
    /// `requested_by_node_id`), and under the default settings node-a's
    /// signed request for that record is answered with it. node-b, signing
    /// for itself while stating node-a's `requester_id` (T3's spoof), is
    /// refused `not_authorized` before the ledger is read: the same request
    /// is refused the same way with no ledger there at all.
    #[test]
    fn a_host_served_record_goes_to_the_peer_that_asked_and_not_to_a_spoofer() {
        use crate::capsule_emit::{CapsuleState, HostProvenance, ObservedHostExchange};
        use crate::lifecycle_channel::DispatchPath;
        use crate::producer::anchor::AnchorClient;
        use crate::producer::checkpoint::{CheckpointCadenceConfig, CheckpointState};

        let data = tempfile::tempdir().unwrap();
        let state = CapsuleState::open(data.path(), RESPONDER).unwrap();
        let emitted = state
            .emit_for_observed_host_exchange(&ObservedHostExchange {
                model: "m",
                exchange_id: Some("e-1"),
                request_digest: Some(&"4b".repeat(32)),
                response_digest: Some(&"33".repeat(32)),
                tool_calls_digest: None,
                reasoning_digest: None,
                usage: None,
                host_provenance: HostProvenance {
                    served_by_node_id: Some(RESPONDER.to_string()),
                    weights_digest: Some("ab".repeat(32)),
                    requested_by_node_id: Some("node-a".to_string()),
                    ..HostProvenance::default()
                },
                dispatch_path: DispatchPath::RawProxy,
                nonce: None,
                peer_capsule_id: None,
                peer_capsule_id_provenance: None,
                twin_bracket_id: None,
                response_text_digest: None,
            })
            .unwrap();
        let (mut checkpoints, _) = CheckpointState::load(
            state.ledger_dir(),
            RESPONDER,
            CheckpointCadenceConfig::default(),
        )
        .unwrap();
        let covering = checkpoints
            .checkpoint_covering(
                &emitted.capsule_id,
                state.signing_key(),
                &AnchorClient::new("http://127.0.0.1:1"),
            )
            .unwrap();
        let request = json!({
            "subject": {"record": emitted.capsule_id},
            "coverage": {"expected_pin": covering.checkpoint.digest()},
        });
        let announced = announced();
        let ask_node = |ledger_dir: &Path, bytes: &[u8]| {
            answer(
                bytes,
                &Responder {
                    ledger_dir,
                    signing_key: state.signing_key(),
                    now: NOW,
                    history_segments: crate::share_policy::history_segments_for(None),
                    share_standard_history: crate::share_policy::shares_standard_history_for(None),
                    keys: Keys {
                        announced: &announced,
                        configured: None,
                    },
                    replay: &ReplayCache::default(),
                    node_id: Some(RESPONDER),
                },
            )
        };

        let bytes = signed_by("node-a", &request);
        let outcome = ask_node(state.ledger_dir(), &bytes);
        let Outcome::Answered { wire, .. } = outcome else {
            panic!("node-a asked for the record of its own exchange: {outcome:?}");
        };
        let verified = verify_response(
            &bytes,
            &wire,
            &state.signing_key().verifying_key(),
            &Default::default(),
        );
        assert_eq!(verified["state"], json!("artifact"), "{verified}");

        let mut spoof = request.clone();
        spoof["requester_id"] = json!("node-a");
        let spoof = signed_by("node-b", &spoof);
        let nowhere = data.path().join("no-such-ledger");
        for ledger_dir in [state.ledger_dir(), nowhere.as_path()] {
            assert_eq!(
                refused(&ask_node(ledger_dir, &spoof)),
                Some(Reason::NotAuthorized),
                "{}",
                ledger_dir.display()
            );
        }
    }

    /// One member changed after signing, and the signature no longer covers
    /// the request: refused as an asker that proved nothing, `not_authorized`
    /// for a record and `policy_declined` once the change makes it Standard
    /// History.
    #[test]
    fn a_member_changed_after_signing_is_refused() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = node_a_record(&ledger);
        request["nonce"] = json!("nonce-1");
        let signed: Value = serde_json::from_slice(&signed_by("node-a", &request)).unwrap();
        // Each change alone would still be answered to node-a (or refused for
        // another reason than who asks) if the signature were not checked.
        let changes: Vec<(&str, Value, Reason)> = vec![
            (
                "subject",
                json!({"correlation": "n-1"}),
                Reason::NotAuthorized,
            ),
            (
                "derivation",
                json!("history_card/1"),
                Reason::PolicyDeclined,
            ),
            (
                "coverage",
                json!({"min_freshness": 1}),
                Reason::NotAuthorized,
            ),
            ("nonce", json!("nonce-2"), Reason::NotAuthorized),
            (
                "deadline",
                json!("2026-09-29T00:02:01Z"),
                Reason::NotAuthorized,
            ),
            ("route", json!("r"), Reason::NotAuthorized),
        ];
        for (member, value, reason) in changes {
            let mut changed = signed.clone();
            changed[member] = value;
            let bytes = serde_json::to_vec(&changed).unwrap();
            assert_eq!(
                refused(&ask_by_default(dir.path(), &bytes)),
                Some(reason),
                "{member}"
            );
        }
        let bytes = serde_json::to_vec(&signed).unwrap();
        assert!(matches!(
            ask_by_default(dir.path(), &bytes),
            Outcome::Answered { .. }
        ));
    }

    /// The same signed request twice within its deadline: the second is
    /// refused `not_authorized`.
    #[test]
    fn a_replayed_nonce_is_not_authorized() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let bytes = signed_by("node-a", &node_a_record(&ledger));
        let replay = ReplayCache::default();
        let segments = crate::share_policy::history_segments_for(None);
        assert!(matches!(
            ask_cached(dir.path(), &bytes, segments, false, &replay),
            Outcome::Answered { .. }
        ));
        assert_eq!(
            refused(&ask_cached(dir.path(), &bytes, segments, false, &replay)),
            Some(Reason::NotAuthorized)
        );
    }

    /// A signed request whose deadline has passed is `deadline_unmet`; one
    /// whose deadline is further ahead than the window is `not_authorized`.
    #[test]
    fn a_deadline_outside_the_window_is_refused() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = node_a_record(&ledger);
        request["deadline"] = json!("2026-09-28T23:59:59Z");
        assert_eq!(
            refused(&ask_by_default(dir.path(), &signed_by("node-a", &request))),
            Some(Reason::DeadlineUnmet)
        );
        request["deadline"] = json!("2026-09-29T00:05:01Z");
        assert_eq!(
            refused(&ask_by_default(dir.path(), &signed_by("node-a", &request))),
            Some(Reason::NotAuthorized)
        );
        request["deadline"] = json!("2026-09-29T00:05:00Z");
        assert!(matches!(
            ask_by_default(dir.path(), &signed_by("node-a", &request)),
            Outcome::Answered { .. }
        ));
    }

    /// A valid signature under a key no host announced (ruling addendum §2):
    /// where an unsigned request would be answered, it is answered exactly as
    /// one, the same artifact and nothing more, and the log says `unbound`
    /// with no node. Where the answer needs an identity it is refused, before
    /// the ledger is read.
    #[test]
    fn an_unbound_key_reads_as_unsigned_only_where_no_identity_is_needed() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let stranger = SigningKey::from_bytes(&[0x77; 32]);
        let record = node_a_record(&ledger);
        let unsigned = serde_json::to_vec(&record).unwrap();
        let unbound = signed_with(&stranger, &record);
        let artifact = |outcome: Outcome| match outcome {
            Outcome::Answered { wire, .. } => wire["artifact"].clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            artifact(ask_with(dir.path(), &unbound, "peers", false)),
            artifact(ask_with(dir.path(), &unsigned, "peers", false)),
            "under peers: the unsigned answer"
        );
        for request in standard_history_requests() {
            let unbound = signed_with(&stranger, &request);
            let unsigned = serde_json::to_vec(&request).unwrap();
            assert_eq!(
                artifact(ask_with(dir.path(), &unbound, "counterparties", true)),
                artifact(ask_with(dir.path(), &unsigned, "counterparties", true)),
                "Standard History shared: the unsigned answer, {request}"
            );
            let reads = evidence_log::index_reads();
            assert_eq!(
                refused(&ask_by_default(dir.path(), &unbound)),
                Some(Reason::PolicyDeclined),
                "Standard History not shared: {request}"
            );
            assert_eq!(
                evidence_log::index_reads(),
                reads,
                "refused before the ledger is read"
            );
        }
        let reads = evidence_log::index_reads();
        assert_eq!(
            refused(&ask_by_default(dir.path(), &unbound)),
            Some(Reason::NotAuthorized),
            "a record under counterparties"
        );
        assert_eq!(
            evidence_log::index_reads(),
            reads,
            "refused before the ledger is read"
        );

        let log = tempfile::tempdir().unwrap();
        let key = key();
        let (announced, replay) = (announced(), ReplayCache::default());
        let responder = Responder {
            ledger_dir: dir.path(),
            signing_key: &key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: false,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER),
        };
        let outcome = answer_and_log(&unbound, false, &responder, Some(log.path()));
        assert!(matches!(outcome, Outcome::Answered { .. }));
        let text = std::fs::read_to_string(log.path().join(crate::received_log::RECEIVED_LOG_FILE))
            .unwrap();
        let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(line["principal_key_id"], json!(key_id_of(&stranger)));
        assert_eq!(line["principal_binding"], json!("unbound"));
        assert_eq!(line["principal_node_id"], Value::Null);
        assert_eq!(line["audience"], json!(RESPONDER));
        assert_eq!(line["replay"], json!("checked"));
    }

    /// The unbound pool full of live nonces (ruling u290 §2): a validly
    /// signed request under a key no host announced is answered as an
    /// unsigned one would be, not refused, and logged `unbound` with
    /// `replay` `unchecked`. A replay the pool still holds is refused, and a
    /// bound key's replay is refused whatever that pool holds.
    #[test]
    fn a_full_unbound_pool_answers_as_unsigned_and_still_refuses_a_replay() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let stranger = SigningKey::from_bytes(&[0x77; 32]);
        let record = node_a_record(&ledger);
        let unsigned = serde_json::to_vec(&record).unwrap();
        let held = signed_with(&stranger, &record);
        let bound = signed_by("node-a", &record);
        let (key, announced, replay) = (key(), announced(), ReplayCache::default());
        let ask = |bytes: &[u8], segments: &str, log: Option<&Path>| {
            let responder = Responder {
                ledger_dir: dir.path(),
                signing_key: &key,
                now: NOW,
                history_segments: segments,
                share_standard_history: false,
                keys: Keys {
                    announced: &announced,
                    configured: None,
                },
                replay: &replay,
                node_id: Some(RESPONDER),
            };
            answer_and_log(bytes, false, &responder, log)
        };
        let artifact = |outcome: Outcome| match outcome {
            Outcome::Answered { wire, .. } => wire["artifact"].clone(),
            other => panic!("{other:?}"),
        };
        let expected = artifact(ask(&unsigned, "peers", None));
        assert_eq!(artifact(ask(&held, "peers", None)), expected);
        assert!(matches!(
            ask(&bound, "counterparties", None),
            Outcome::Answered { .. }
        ));
        let now = parse_utc(NOW).unwrap().seconds;
        for i in 1..requester_auth::MAX_UNBOUND_NONCES {
            assert_eq!(
                replay.first_use(&format!("filler-{i}"), "n", now + 60, now, false),
                requester_auth::Use::First
            );
        }

        let log = tempfile::tempdir().unwrap();
        let fresh = signed_with(&stranger, &record);
        assert_eq!(
            artifact(ask(&fresh, "peers", Some(log.path()))),
            expected,
            "answered as unsigned, not refused"
        );
        let text = std::fs::read_to_string(log.path().join(crate::received_log::RECEIVED_LOG_FILE))
            .unwrap();
        let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(line["principal_key_id"], json!(key_id_of(&stranger)));
        assert_eq!(line["principal_binding"], json!("unbound"));
        assert_eq!(line["replay"], json!("unchecked"));
        assert_eq!(
            refused(&ask(&held, "peers", None)),
            Some(Reason::NotAuthorized),
            "a replay the pool holds"
        );
        assert_eq!(
            refused(&ask(&bound, "counterparties", None)),
            Some(Reason::NotAuthorized),
            "a bound key's replay"
        );
        assert_eq!(
            refused(&ask(
                &signed_with(&stranger, &record),
                "counterparties",
                None
            )),
            Some(Reason::NotAuthorized),
            "still nothing where the answer needs an identity"
        );
    }

    /// A signed request with no `audience` is `request_malformed`, before
    /// anything else, whatever the settings.
    #[test]
    fn a_signed_request_without_an_audience_is_malformed() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut signed: Value =
            serde_json::from_slice(&signed_by("node-a", &node_a_record(&ledger))).unwrap();
        signed.as_object_mut().unwrap().remove("audience");
        let bytes = serde_json::to_vec(&signed).unwrap();
        for segments in ["counterparties", "peers", "off"] {
            assert_eq!(
                refused(&ask_with(dir.path(), &bytes, segments, true)),
                Some(Reason::RequestMalformed),
                "{segments}"
            );
        }
    }

    /// node-a's request for its own record, addressed here but signed for
    /// another channel (a `ledger-fetch/1` fetch, a referee's `adjudicate`),
    /// is refused `not_authorized` before the ledger is read, even where an
    /// unsigned request would be answered: the signature fails on the
    /// Evidence Request's tag (ruling u338), and a failed signature is never
    /// read as unsigned. MUTANT: one tag shared by every channel and it is
    /// answered.
    #[test]
    fn a_request_signed_for_another_channel_is_refused_before_the_ledger_is_read() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let record = node_a_record(&ledger);
        for channel in [
            requester_auth::Channel::LedgerFetch,
            requester_auth::Channel::RefereeAdjudicate,
        ] {
            let mut request = record.clone();
            requester_auth::sign(
                channel,
                &mut request,
                &asker_key("node-a"),
                now(),
                RESPONDER,
            );
            let bytes = serde_json::to_vec(&request).unwrap();
            for segments in ["counterparties", "peers"] {
                let reads = evidence_log::index_reads();
                let outcome = ask_with(dir.path(), &bytes, segments, false);
                assert_eq!(evidence_log::index_reads(), reads, "the ledger was read");
                assert_eq!(
                    outcome.wire_bytes(),
                    Some(refusal_of(&bytes, Reason::NotAuthorized)),
                    "{channel:?} ({segments})"
                );
            }
        }
        assert!(matches!(
            ask_by_default(dir.path(), &signed_by("node-a", &record)),
            Outcome::Answered { .. }
        ));
    }

    /// The relay (ruling addendum §1): node-a's request to node-x, forwarded
    /// by node-x to this node, names node-x as its audience. It is refused
    /// `not_authorized` before the ledger is read, even where an unsigned
    /// request would be answered. Addressed to this node, it is answered.
    #[test]
    fn a_request_addressed_to_another_node_is_refused_before_the_ledger_is_read() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let record = node_a_record(&ledger);
        let mut to_x = record.clone();
        to_x["nonce"] = json!("relayed");
        requester_auth::sign(
            requester_auth::Channel::EvidenceRequest,
            &mut to_x,
            &asker_key("node-a"),
            now(),
            "node-x",
        );
        let to_x = serde_json::to_vec(&to_x).unwrap();
        for segments in ["counterparties", "peers"] {
            let reads = evidence_log::index_reads();
            let outcome = ask_with(dir.path(), &to_x, segments, false);
            assert_eq!(evidence_log::index_reads(), reads, "the ledger was read");
            assert_eq!(
                outcome.wire_bytes(),
                Some(refusal_of(&to_x, Reason::NotAuthorized)),
                "{segments}"
            );
        }
        assert!(matches!(
            ask_by_default(dir.path(), &signed_by("node-a", &record)),
            Outcome::Answered { .. }
        ));
        // A node that does not know its own id yet is no signed request's
        // audience.
        let key = key();
        let (announced, replay) = (announced(), ReplayCache::default());
        let responder = Responder {
            ledger_dir: dir.path(),
            signing_key: &key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: true,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: None,
        };
        assert_eq!(
            refused(&answer(&signed_by("node-a", &record), &responder)),
            Some(Reason::NotAuthorized)
        );
    }

    /// What is never read as unsigned (ruling addendum §2): under `peers`,
    /// where an unsigned request is answered, each of these is still refused,
    /// signed by a bound key and by an unbound one alike. (A malformed
    /// `requester_auth`, a missing audience and a passed deadline are refused
    /// before the signature is looked at; the u289 hand mutants N1 and N2
    /// show those two rules bite.)
    #[test]
    fn a_failed_request_is_never_answered_as_unsigned() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let record = record_request(&ledger, 0);
        let stranger = SigningKey::from_bytes(&[0x77; 32]);
        for key in [asker_key("node-z"), stranger] {
            let signed: Value = serde_json::from_slice(&signed_with(&key, &record)).unwrap();
            let with = |member: &str, value: Value| {
                let mut r = signed.clone();
                r[member] = value;
                serde_json::to_vec(&r).unwrap()
            };
            let resign = |member: &str, value: Value| {
                let mut r = record.clone();
                r[member] = value;
                signed_with(&key, &r)
            };
            let mut no_audience = signed.clone();
            no_audience.as_object_mut().unwrap().remove("audience");
            let mut to_x = record.clone();
            requester_auth::sign(
                requester_auth::Channel::EvidenceRequest,
                &mut to_x,
                &key,
                now(),
                "node-x",
            );
            let cases: Vec<(&str, Vec<u8>, Reason)> = vec![
                (
                    "malformed requester_auth",
                    with("requester_auth", json!({"key_id": "ab"})),
                    Reason::RequestMalformed,
                ),
                (
                    "no audience",
                    serde_json::to_vec(&no_audience).unwrap(),
                    Reason::RequestMalformed,
                ),
                (
                    "deadline passed",
                    resign("deadline", json!("2026-09-28T23:59:59Z")),
                    Reason::DeadlineUnmet,
                ),
                (
                    "deadline outside the window",
                    resign("deadline", json!("2026-09-29T00:05:01Z")),
                    Reason::NotAuthorized,
                ),
                (
                    "bad signature",
                    with("nonce", json!("not-the-signed-one")),
                    Reason::NotAuthorized,
                ),
                (
                    "audience mismatch",
                    serde_json::to_vec(&to_x).unwrap(),
                    Reason::NotAuthorized,
                ),
            ];
            for (what, bytes, reason) in cases {
                assert_eq!(
                    refused(&ask_with(dir.path(), &bytes, "peers", true)),
                    Some(reason),
                    "{what}"
                );
            }
            // A replay: the first is answered, the second refused.
            let bytes = serde_json::to_vec(&signed).unwrap();
            let replay = ReplayCache::default();
            assert!(matches!(
                ask_cached(dir.path(), &bytes, "peers", true, &replay),
                Outcome::Answered { .. }
            ));
            assert_eq!(
                refused(&ask_cached(dir.path(), &bytes, "peers", true, &replay)),
                Some(Reason::NotAuthorized),
                "replay"
            );
        }
    }

    /// A node announcing node-a's public key as its own does not shut node-a
    /// out: node-a's request, signed and naming node-a, is still node-a's.
    /// The same signature naming no node is bound to neither, and naming a
    /// third node is a false claim, refused even under `peers`.
    #[test]
    fn a_node_announcing_another_nodes_key_does_not_unbind_it() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut announced = announced();
        announced.insert("node-m".into(), key_id_of(&asker_key("node-a")));
        let key = key();
        let ask = |bytes: &[u8], segments: &str| {
            let replay = ReplayCache::default();
            answer(
                bytes,
                &Responder {
                    ledger_dir: dir.path(),
                    signing_key: &key,
                    now: NOW,
                    history_segments: segments,
                    share_standard_history: false,
                    keys: Keys {
                        announced: &announced,
                        configured: None,
                    },
                    replay: &replay,
                    node_id: Some(RESPONDER),
                },
            )
        };
        let mut request = node_a_record(&ledger);
        assert_eq!(
            refused(&ask(&signed_by("node-a", &request), "counterparties")),
            Some(Reason::NotAuthorized),
            "naming no node: bound to neither"
        );
        request["requester_id"] = json!("node-a");
        assert!(matches!(
            ask(&signed_by("node-a", &request), "counterparties"),
            Outcome::Answered { .. }
        ));
        request["requester_id"] = json!("node-z");
        assert_eq!(
            refused(&ask(&signed_by("node-a", &request), "peers")),
            Some(Reason::NotAuthorized),
            "naming neither node: a false claim, never read as unsigned"
        );
    }

    /// A node stating another node's id under its own bound key is refused
    /// under `peers` too: a false claim is never read as unsigned.
    #[test]
    fn a_false_claim_is_refused_even_where_unsigned_is_answered() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = record_request(&ledger, 0);
        request["requester_id"] = json!("node-a");
        assert_eq!(
            refused(&ask_with(
                dir.path(),
                &signed_by("node-z", &request),
                "peers",
                true
            )),
            Some(Reason::NotAuthorized)
        );
    }

    /// An unsigned request may carry an `audience`; it is ignored there, as
    /// any member an unsigned request carries that no check reads.
    #[test]
    fn an_unsigned_request_naming_another_audience_is_answered_as_unsigned() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = record_request(&ledger, 0);
        request["audience"] = json!("node-x");
        assert!(matches!(
            ask_with(
                dir.path(),
                &serde_json::to_vec(&request).unwrap(),
                "peers",
                false
            ),
            Outcome::Answered { .. }
        ));
    }

    /// `requester_auth` with a 63-hex key is `request_malformed`, as is a
    /// signed request with no nonce.
    #[test]
    fn a_malformed_requester_auth_is_request_malformed() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut signed: Value =
            serde_json::from_slice(&signed_by("node-a", &node_a_record(&ledger))).unwrap();
        let mut short = signed.clone();
        let key_id = short["requester_auth"]["key_id"].as_str().unwrap()[1..].to_string();
        short["requester_auth"]["key_id"] = json!(key_id);
        assert_eq!(
            refused(&ask_by_default(
                dir.path(),
                &serde_json::to_vec(&short).unwrap()
            )),
            Some(Reason::RequestMalformed)
        );
        signed.as_object_mut().unwrap().remove("nonce");
        assert_eq!(
            refused(&ask_by_default(
                dir.path(),
                &serde_json::to_vec(&signed).unwrap()
            )),
            Some(Reason::RequestMalformed)
        );
    }

    /// A signature over the same map without the domain tag does not verify.
    #[test]
    fn a_signature_without_the_domain_tag_is_not_authorized() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = node_a_record(&ledger);
        request["nonce"] = json!("nonce-1");
        request["deadline"] = json!("2026-09-29T00:02:00Z");
        request["audience"] = json!(RESPONDER);
        let body = jcs::to_string(&request).unwrap();
        let a = asker_key("node-a");
        request["requester_auth"] = json!({
            "key_id": key_id_of(&a),
            "sig": hex::encode(a.sign(body.as_bytes()).to_bytes()),
        });
        let bytes = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            refused(&ask_by_default(dir.path(), &bytes)),
            Some(Reason::NotAuthorized)
        );
    }

    /// An unsigned request for a record is `not_authorized` before the ledger
    /// is read: the same bytes back for a record that exists and one that
    /// does not, and for a node with a ledger and one with none at all.
    #[test]
    fn an_unsigned_record_request_is_refused_alike_for_any_subject_before_the_ledger_is_read() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let missing = tempfile::tempdir().unwrap();
        let nowhere = missing.path().join("no-such-ledger");
        let held = node_a_record(&ledger);
        let mut absent = held.clone();
        absent["subject"] = json!({"record": "a".repeat(64)});
        for mut request in [held, absent] {
            request["requester_id"] = json!("node-a");
            let bytes = serde_json::to_vec(&request).unwrap();
            for ledger_dir in [dir.path(), nowhere.as_path()] {
                let reads = evidence_log::index_reads();
                let outcome = ask_by_default(ledger_dir, &bytes);
                assert_eq!(evidence_log::index_reads(), reads, "the ledger was read");
                assert_eq!(
                    outcome.wire_bytes(),
                    Some(refusal_of(&bytes, Reason::NotAuthorized)),
                    "{request}"
                );
            }
        }
    }

    /// PRD T3. Node y asks for node-a's record claiming to be node-a: with
    /// its own key and node-a's id, or with node-a's key id over its own
    /// signature. Each is a signed `not_authorized` bound to its request,
    /// given before the ledger is read. The same request from node-a,
    /// signed by node-a, is answered.
    #[test]
    fn a_node_claiming_to_be_another_is_refused_before_the_ledger_is_read() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let mut request = node_a_record(&ledger);
        request["requester_id"] = json!("node-a");

        let own_key_claims_a = signed_by("node-y", &request);
        let mut a_key_y_sig: Value = serde_json::from_slice(&own_key_claims_a).unwrap();
        a_key_y_sig["requester_auth"]["key_id"] = json!(key_id_of(&asker_key("node-a")));
        let a_key_y_sig = serde_json::to_vec(&a_key_y_sig).unwrap();

        for bytes in [&own_key_claims_a, &a_key_y_sig] {
            let reads = evidence_log::index_reads();
            let outcome = ask_by_default(dir.path(), bytes);
            assert_eq!(evidence_log::index_reads(), reads, "the ledger was read");
            let Outcome::Refused { wire, reason, .. } = &outcome else {
                panic!("{outcome:?}");
            };
            assert_eq!(*reason, Reason::NotAuthorized);
            assert_eq!(
                refusal::verify_for(wire, &key().verifying_key(), &request_digest(bytes)),
                Ok(Reason::NotAuthorized),
                "signed and bound to the request"
            );
        }
        assert!(matches!(
            ask_by_default(dir.path(), &signed_by("node-a", &request)),
            Outcome::Answered { .. }
        ));
    }

    /// D3: a proven node that no record of this one names gets the same
    /// `not_authorized` for any record subject, held or not, in range or
    /// past it, small or over the answer limit: the refusal tells it
    /// nothing about this log's size or cadence.
    #[test]
    fn a_stranger_learns_nothing_of_the_logs_size_from_a_record_refusal() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let held = node_a_record(&ledger);
        let pin = held["coverage"].clone();
        let subjects = [
            held["subject"].clone(),
            json!({"record": "a".repeat(64)}),
            json!({"range": [0, 3]}),
            json!({"range": [100_000, 100_001]}),
            json!({"range": [0, 100_000]}),
            json!({"full_history": null}),
            json!({"correlation": "no-such-nonce"}),
        ];
        for subject in subjects {
            let request = json!({"subject": subject, "coverage": pin});
            let bytes = signed_by("node-z", &request);
            let outcome = ask_by_default(dir.path(), &bytes);
            assert_eq!(
                outcome.wire_bytes(),
                Some(refusal_of(&bytes, Reason::NotAuthorized)),
                "{subject}"
            );
        }
    }

    /// The artifact a subject is answered with is the same bytes for two
    /// different authorized requesters.
    #[test]
    fn two_authorized_requesters_get_the_same_artifact() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let artifact =
            |bytes: &[u8], segments: &str| match ask_with(dir.path(), bytes, segments, false) {
                Outcome::Answered { wire, .. } => wire["artifact"].clone(),
                other => panic!("{other:?}"),
            };
        let checkpoints = &standard_history_requests()[0];
        assert_eq!(
            artifact(&signed_by("node-a", checkpoints), "counterparties"),
            artifact(&signed_by("node-b", checkpoints), "counterparties")
        );
        let record = record_request(&ledger, 0);
        assert_eq!(
            artifact(&signed_by("node-y", &record), "peers"),
            artifact(&signed_by("node-z", &record), "peers")
        );
    }

    /// Under `peers` an unsigned request is answered a record, as before
    /// (R-13's operator exception); a signed one is checked all the same.
    #[test]
    fn under_peers_an_unsigned_request_is_answered_and_a_bad_signature_is_not() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let record = record_request(&ledger, 0);
        assert!(matches!(
            ask_with(
                dir.path(),
                &serde_json::to_vec(&record).unwrap(),
                "peers",
                false
            ),
            Outcome::Answered { .. }
        ));
        let mut tampered: Value = serde_json::from_slice(&signed_by("node-z", &record)).unwrap();
        tampered["nonce"] = json!("not-the-signed-one");
        assert_eq!(
            refused(&ask_with(
                dir.path(),
                &serde_json::to_vec(&tampered).unwrap(),
                "peers",
                false
            )),
            Some(Reason::NotAuthorized)
        );
    }

    /// Who asked is recorded with the request's digest: the verified key and
    /// node when the signature and binding held, even when the request is
    /// refused for stating another node's id; nothing when it was unsigned.
    #[test]
    fn the_log_records_the_verified_principal_with_the_request_digest() {
        let ledger = corpus_ledger();
        let dir = node(&ledger);
        let log = tempfile::tempdir().unwrap();
        let key = key();
        let announced = announced();
        let replay = ReplayCache::default();
        let responder = Responder {
            ledger_dir: dir.path(),
            signing_key: &key,
            now: NOW,
            history_segments: "counterparties",
            share_standard_history: false,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER),
        };
        let request = node_a_record(&ledger);
        let mut claims_a = request.clone();
        claims_a["requester_id"] = json!("node-a");
        let asks = [
            signed_by("node-a", &request),
            serde_json::to_vec(&claims_a).unwrap(),
            signed_by("node-y", &claims_a),
        ];
        for bytes in &asks {
            answer_and_log(bytes, false, &responder, Some(log.path()));
        }
        let lines: Vec<Value> =
            std::fs::read_to_string(log.path().join(crate::received_log::RECEIVED_LOG_FILE))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
        let y = key_id_of(&asker_key("node-y"));
        let a = key_id_of(&asker_key("node-a"));
        let expected = [
            (Some(a.as_str()), Some("node-a"), None, "answered"),
            (None, None, Some("node-a"), "refused"),
            (Some(y.as_str()), Some("node-y"), Some("node-a"), "refused"),
        ];
        for ((line, bytes), (key_id, node_id, stated, status)) in
            lines.iter().zip(&asks).zip(expected)
        {
            assert_eq!(line["request_digest"], json!(request_digest(bytes)));
            assert_eq!(line["principal_key_id"], json!(key_id), "{line}");
            assert_eq!(line["principal_node_id"], json!(node_id), "{line}");
            assert_eq!(line["requester_id"], json!(stated), "{line}");
            assert_eq!(line["status"], json!(status), "{line}");
        }
        assert_eq!(lines.len(), 3);
    }
}
