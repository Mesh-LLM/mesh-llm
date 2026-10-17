//! Historical-reference corroboration (PRD-SERVICE-EVIDENCE §5.7(a), R-25 to
//! R-28 and R-30; the Q7 ruling): a node Z asks X for its half of an earlier
//! X–Y exchange, then asks Y, independently, for Y's half, and reports per Y
//! whether the two independently held accounts agree.
//!
//! **The derivation, `exchange_corroboration/1`.** Unregistered, and named
//! on the wire by the digest of its definition document
//! (`derivations/exchange-corroboration-1.json`, [`DEFINITION_DIGEST`]). The
//! subject is the existing `correlation` form, whose value is the SHA-256 of
//! the exchange's client nonce. The responder matches that value against the
//! nonce its own records bind and nothing else
//! ([`crate::evidence_log::LedgerIndex::by_client_nonce`]): never an
//! `exchange_id`, a request nonce or a digest binding. The artifact carries one
//! entry per matching half, each exactly `half`, `client_nonce_sha256`,
//! `role`, `request_digest`, `response_digest` and `inclusion`: digests and a
//! proof, never a record body, a counterparty or who asked. It is the same
//! bytes for every requester at the same anchor, so it is answered whole or
//! refused whole (`crate::evidence_answer` decides which, before the subject
//! is looked up). No match is a signed `no_such_subject`, never an empty
//! artifact.
//!
//! **What Z reports, per Y, never merged** ([`Outcome`]): halves agree;
//! halves disagree (naming the fields); not joinable (no bound client nonce
//! to join on, decided before anyone is asked); one-sided (Y's signed
//! `no_such_subject`); a signed refusal with its reason; a recorded absence
//! (nothing came back); an unverifiable answer (something came back and
//! failed). Every Y also carries its own independence statement, "not
//! established" unless Z's operator declared a basis (R-30), and the frame Z
//! chose it under (R-28), which stays in Z's own log and is never sent. One
//! ask per Y. No count, rate or figure across Ys or about X is computed.
//!
//! The run is split into pure steps ([`plan`], [`after_x`], [`finish`]) so
//! the mesh glue only sends what they say to send.

use std::collections::BTreeSet;

use cll::checkpoint::{CheckpointRecord, DecodedCheckpointCose};
use cll::mmr::leaf_count;
use evidence_request::refusal;
use evidence_request::registry::{Reason, SubjectForm};
use evidence_request::request::{self, Derivation, Request, Subject};
use evidence_request::resolve::{self, Anchor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::evidence_log::{LogView, HOST_SERVED_NO_NONCE};

/// The derivation's name, as its definition document states it.
#[cfg(test)]
pub const DERIVATION_NAME: &str = "exchange_corroboration/1";

/// The definition document, as published beside this crate. Only its
/// digest is used at run time; a test pins one to the other.
#[cfg(test)]
pub const DEFINITION_DOCUMENT: &str = include_str!("../derivations/exchange-corroboration-1.json");

/// SHA-256 over the RFC 8785 bytes of [`DEFINITION_DOCUMENT`]: the
/// derivation's name on the wire. Pinned to the file by a test.
pub const DEFINITION_DIGEST: &str =
    "7fed77bebd92247302067dc9eda0c34f96497b6521fd473a19c0fafec0993a0b";

/// The members every entry carries, and nothing else.
const ENTRY_MEMBERS: [&str; 6] = [
    "client_nonce_sha256",
    "half",
    "inclusion",
    "request_digest",
    "response_digest",
    "role",
];

/// The log Z keeps of each run, beside its ledger: the frames it asked
/// under and the absences it recorded.
pub const RUNS_FILENAME: &str = "historical-references.jsonl";

/// The `client_nonce_source` of a nonce this node's plugin minted itself
/// because the client sent none: it was never shared with the peer, so a
/// half carrying it joins nothing.
const FALLBACK_NONCE_SOURCE: &str = "plugin_generated_fallback";

/// Whether `derivation` names this one.
pub fn is_derivation(derivation: Option<&Derivation>) -> bool {
    matches!(derivation, Some(Derivation::Digest(d)) if d == DEFINITION_DIGEST)
}

// ---------------------------------------------------------------------
// Responder
// ---------------------------------------------------------------------

/// Resolution for this derivation: the ledger as `resolve::resolve` sees it,
/// except that a correlation is held only when some record's bound client
/// nonce hashes to it.
pub(crate) struct NonceView<'a, 'b>(pub &'a LogView<'b>);

impl resolve::Responder for NonceView<'_, '_> {
    fn holds_record(&self, digest: &str) -> bool {
        self.0.holds_record(digest)
    }
    fn holds_correlation(&self, id: &str) -> bool {
        !self.0.index.by_client_nonce(id).is_empty()
    }
    fn cites_exchange(&self, half_digest: &str) -> bool {
        self.0.cites_exchange(half_digest)
    }
    fn positional_ordering(&self) -> bool {
        self.0.positional_ordering()
    }
    fn anchors(&self) -> Vec<Anchor> {
        self.0.anchors()
    }
    fn derivation_forms(&self, derivation: &Derivation) -> Option<Vec<SubjectForm>> {
        is_derivation(Some(derivation)).then(|| vec![SubjectForm::Correlation])
    }
}

/// The leaf indices this derivation answers for `req` under `anchor`: every
/// record committed under the anchor whose bound nonce hashes to the subject.
pub(crate) fn matching_records(
    req: &Request,
    anchor: &CheckpointRecord,
    view: &LogView<'_>,
) -> Vec<u64> {
    let Subject::Correlation(id) = &req.subject else {
        return Vec::new();
    };
    let covered = leaf_count(anchor.mmr_size).unwrap_or(0);
    view.index
        .by_client_nonce(id)
        .iter()
        .copied()
        .filter(|&i| i < covered)
        .collect()
}

/// The artifact's `entries` for `records` under `anchor`, one per half, each
/// with the half's inclusion proof at the anchor in the profile's
/// `cll-inclusion-proof` form, in hex. `NoSuchSubject` when none is committed
/// there; `PolicyDeclined` when a record or its proof cannot be read back.
pub(crate) fn entries(
    id: &str,
    anchor: &CheckpointRecord,
    records: &[u64],
    view: &LogView<'_>,
) -> Result<Value, Reason> {
    if records.is_empty() {
        return Err(Reason::NoSuchSubject);
    }
    let mut entries = Vec::with_capacity(records.len());
    for &i in records {
        let record = view.record_at(i).ok_or(Reason::PolicyDeclined)?;
        let body: Value =
            serde_json::from_slice(&record.body).map_err(|_| Reason::PolicyDeclined)?;
        let proof = crate::evidence_artifact::inclusion(view, i, anchor.mmr_size)?;
        entries.push(json!({
            "half": record.digest,
            "client_nonce_sha256": id,
            "role": body.pointer("/model_attestation/compute_attestation/x-mesh-poc-v1/role")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            "request_digest": effect_digest(&body, "request_digest"),
            "response_digest": effect_digest(&body, "response_digest"),
            "inclusion": hex::encode(proof),
        }));
    }
    Ok(Value::Array(entries))
}

/// An `effect` digest the record binds, or `null` when it binds none (a half
/// sealed before the response arrived).
fn effect_digest(record: &Value, member: &str) -> Value {
    record
        .get("effect")
        .and_then(|e| e.get(member))
        .and_then(Value::as_str)
        .filter(|d| is_digest(d))
        .map_or(Value::Null, |d| json!(d))
}

// ---------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------

/// One half, as an answer under this derivation states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Half {
    pub half: String,
    pub client_nonce_sha256: String,
    pub role: String,
    pub request_digest: Option<String>,
    pub response_digest: Option<String>,
}

impl Half {
    fn to_json(&self) -> Value {
        json!({
            "half": self.half,
            "client_nonce_sha256": self.client_nonce_sha256,
            "role": self.role,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
        })
    }
}

/// What one answer to this derivation proved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verified {
    /// An artifact that verifies: its anchor and its halves.
    Halves { anchor: String, halves: Vec<Half> },
    /// A refusal signed by the responder, bound to the request sent.
    Refusal(Reason),
    /// Something came back and did not verify.
    Failed(String),
}

/// Check `response` to the request `sent` against the responder's `key`.
pub fn verify(sent: &[u8], response: &Value, key: &ed25519_dalek::VerifyingKey) -> Verified {
    let digest = evidence_request::digest::request_digest(sent);
    if response.get("reason").is_some() {
        return match refusal::verify_for(response, key, &digest) {
            Ok(reason) => Verified::Refusal(reason),
            Err(e) => Verified::Failed(format!("the refusal does not verify: {e}")),
        };
    }
    match verify_artifact(sent, response, key) {
        Ok((anchor, halves)) => Verified::Halves { anchor, halves },
        Err(why) => Verified::Failed(why),
    }
}

/// The answer, checked as any answer is (`evidence_artifact::check`): the
/// envelope, the anchor checkpoint, the coverage asked for, and each entry
/// ([`check_entries`]).
fn verify_artifact(
    sent: &[u8],
    response: &Value,
    key: &ed25519_dalek::VerifyingKey,
) -> Result<(String, Vec<Half>), String> {
    let req = request::parse_json(sent).map_err(|_| "the request sent is not a -00 request")?;
    if !matches!(req.subject, Subject::Correlation(_)) {
        return Err("the request sent is not a correlation request".into());
    }
    if !is_derivation(req.derivation.as_ref()) {
        return Err("the request sent does not name this derivation".into());
    }
    let checked = crate::evidence_artifact::check(sent, &req, response, key, &|_| None)
        .map_err(|e| e.to_string())?;
    Ok((checked.anchor, checked.halves))
}

/// The artifact's entries, for the client nonce digest `id`, under the
/// anchor checkpoint `cp`: each carries exactly the derivation's members,
/// is for `id`, and is proven under `cp`. Returns the halves and their
/// leaves.
pub(crate) fn check_entries(
    art: &Map<String, Value>,
    id: &str,
    cp: &DecodedCheckpointCose,
) -> Result<(Vec<Half>, Vec<u64>), &'static str> {
    let covered = leaf_count(cp.mmr_size).map_err(|_| "the anchor checkpoint is malformed")?;
    let entries = art
        .get("entries")
        .and_then(Value::as_array)
        .filter(|e| !e.is_empty())
        .ok_or("the artifact carries no entries")?;
    let mut halves = Vec::with_capacity(entries.len());
    let mut leaves = Vec::with_capacity(entries.len());
    for entry in entries {
        let map = entry.as_object().ok_or("an entry is malformed")?;
        let mut members: Vec<&str> = map.keys().map(String::as_str).collect();
        members.sort_unstable();
        if members != ENTRY_MEMBERS {
            return Err("an entry does not carry exactly the derivation's members");
        }
        let text = |k: &str| map.get(k).and_then(Value::as_str);
        let digest_or_null = |k: &str| match map.get(k) {
            Some(Value::Null) => Ok(None),
            Some(Value::String(d)) if is_digest(d) => Ok(Some(d.clone())),
            _ => Err("an entry's digest is malformed"),
        };
        let half = text("half")
            .filter(|h| is_digest(h))
            .ok_or("an entry's half is malformed")?;
        if text("client_nonce_sha256") != Some(id) {
            return Err("an entry is for another client nonce");
        }
        let role = text("role").ok_or("an entry's role is malformed")?;
        let proof = text("inclusion")
            .and_then(|h| hex::decode(h).ok())
            .ok_or("an entry's inclusion proof is malformed")?;
        let leaf = crate::evidence_artifact::cll_proofs::inclusion_leaf(&proof)
            .ok_or("an entry's inclusion proof is malformed")?;
        let half_bytes: [u8; 32] = hex::decode(half)
            .ok()
            .and_then(|b| b.try_into().ok())
            .ok_or("an entry's half is malformed")?;
        if leaf >= covered
            || !crate::evidence_artifact::cll_proofs::proves(
                &proof,
                leaf,
                &half_bytes,
                cp.mmr_size,
                &cp.new_peak_hashes,
            )
        {
            return Err("an entry's half is not included under the anchor");
        }
        halves.push(Half {
            half: half.to_string(),
            client_nonce_sha256: id.to_string(),
            role: role.to_string(),
            request_digest: digest_or_null("request_digest")?,
            response_digest: digest_or_null("response_digest")?,
        });
        leaves.push(leaf);
    }
    Ok((halves, leaves))
}

// ---------------------------------------------------------------------
// Z's comparison, per Y
// ---------------------------------------------------------------------

/// One field two halves state differently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    pub field: &'static str,
    pub x: Option<String>,
    pub y: Option<String>,
}

/// What Z reports for one node, never merged with another's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Agree {
        y_half: Half,
    },
    Disagree {
        differences: Vec<Difference>,
    },
    NotJoinable {
        why: String,
    },
    OneSided,
    SignedRefusal {
        reason: Reason,
    },
    RecordedAbsence {
        why: String,
    },
    Unverifiable {
        why: String,
    },
    /// X's half was not obtained, so this Y was not asked.
    NotAsked {
        why: String,
    },
}

impl Outcome {
    pub fn to_json(&self) -> Value {
        match self {
            Outcome::Agree { y_half } => json!({"state": "agree", "y_half": y_half.to_json()}),
            Outcome::Disagree { differences } => json!({
                "state": "disagree",
                "differences": differences.iter().map(|d| json!({"field": d.field, "x": d.x, "y": d.y})).collect::<Vec<_>>(),
            }),
            Outcome::NotJoinable { why } => json!({"state": "not_joinable", "why": why}),
            Outcome::OneSided => {
                json!({"state": "one_sided", "refusal": {"reason": Reason::NoSuchSubject.token()}})
            }
            Outcome::SignedRefusal { reason } => {
                json!({"state": "signed_refusal", "reason": reason.token()})
            }
            Outcome::RecordedAbsence { why } => json!({"state": "recorded_absence", "why": why}),
            Outcome::Unverifiable { why } => json!({"state": "unverifiable", "why": why}),
            Outcome::NotAsked { why } => json!({"state": "not_asked", "why": why}),
        }
    }
}

/// Whether two halves of one exchange state each other's side: one sealed as
/// the node that asked, the other as the node that served.
fn complementary(x: &str, y: &str) -> bool {
    if cfg!(feature = "mutant-corroboration-role-equality") {
        return x == y;
    }
    matches!((x, y), ("requested", "served") | ("served", "requested"))
}

/// The differences between X's half and one of Y's, empty when they agree.
/// A digest either side does not bind is a difference, never a match.
fn differences(x: &Half, y: &Half) -> Vec<Difference> {
    let mut out = Vec::new();
    if !complementary(&x.role, &y.role) {
        out.push(Difference {
            field: "role",
            x: Some(x.role.clone()),
            y: Some(y.role.clone()),
        });
    }
    for (field, a, b) in [
        ("request_digest", &x.request_digest, &y.request_digest),
        ("response_digest", &x.response_digest, &y.response_digest),
    ] {
        if a.is_none() || a != b {
            out.push(Difference {
                field,
                x: a.clone(),
                y: b.clone(),
            });
        }
    }
    out
}

/// Compare X's half with what Y's answer proved.
pub fn corroborate(x: &Half, y: &Verified) -> Outcome {
    match y {
        Verified::Halves { halves, .. } => {
            if let Some(agreeing) = halves.iter().find(|h| differences(x, h).is_empty()) {
                return Outcome::Agree {
                    y_half: agreeing.clone(),
                };
            }
            Outcome::Disagree {
                differences: halves.iter().flat_map(|h| differences(x, h)).collect(),
            }
        }
        Verified::Refusal(reason) if cfg!(feature = "mutant-corroboration-refusal-as-absence") => {
            Outcome::RecordedAbsence {
                why: reason.token().to_string(),
            }
        }
        Verified::Refusal(Reason::NoSuchSubject) => Outcome::OneSided,
        Verified::Refusal(reason) => Outcome::SignedRefusal { reason: *reason },
        Verified::Failed(why) => Outcome::Unverifiable { why: why.clone() },
    }
}

// ---------------------------------------------------------------------
// Z's run
// ---------------------------------------------------------------------

/// Why Z chose a Y (R-28). Recorded in Z's own log; never sent.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    /// `x_record_names_y` (X's record names Y as the other side),
    /// `operator_list`, or `other` with a `note`.
    pub basis: String,
    #[serde(default)]
    pub note: Option<String>,
}

const FRAME_BASES: [&str; 3] = ["x_record_names_y", "operator_list", "other"];

/// One Y to ask, once.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AskY {
    pub peer_id: String,
    pub frame: Frame,
    /// A basis Z's operator declares for this Y being independent of X. A
    /// claim, shown as one; absent, independence is not established.
    #[serde(default)]
    pub independence_basis: Option<String>,
}

/// One run's input. The join key is either stated (`client_nonce_sha256`) or
/// read from X's own record that Z already holds (`x_record`), never both.
#[derive(Clone, Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunArgs {
    /// Hex-encoded mesh id of X.
    pub x_peer: String,
    /// SHA-256 (64 lowercase hex) of the exchange's client nonce.
    #[serde(default)]
    pub client_nonce_sha256: Option<String>,
    /// X's record of the exchange, as Z holds it: its bound nonce is the key.
    #[serde(default)]
    pub x_record: Option<Value>,
    pub ys: Vec<AskY>,
}

/// The key two halves of one exchange are joined on, or why there is none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinKey {
    ClientNonce(String),
    /// No key: the reason, and nobody is asked.
    Unjoinable(String),
}

/// The join key X's record carries: the digest of its bound client nonce,
/// unless it binds none, the host-served placeholder, or a nonce this node's
/// plugin minted itself (never shared with the peer). Nothing else is ever a
/// key: not `exchange_id`, not the digest binding, not a time.
pub fn join_key_of(record: &Value) -> JoinKey {
    let poc = record.pointer("/model_attestation/compute_attestation/x-mesh-poc-v1");
    if cfg!(feature = "mutant-corroboration-joins-by-binding") {
        if let Some(effect) = record.get("effect") {
            if let (Some(a), Some(b)) = (
                effect.get("request_digest").and_then(Value::as_str),
                effect.get("response_digest").and_then(Value::as_str),
            ) {
                return JoinKey::ClientNonce(hex::encode(Sha256::digest(
                    crate::evidence_log::exchange_binding(a, b).as_bytes(),
                )));
            }
        }
    }
    let source = poc
        .and_then(|p| p.get("client_nonce_source"))
        .and_then(Value::as_str);
    if source == Some(FALLBACK_NONCE_SOURCE) {
        return JoinKey::Unjoinable(
            "X's half binds a nonce its own plugin minted, never shared".into(),
        );
    }
    match poc
        .and_then(|p| p.get("client_nonce"))
        .and_then(Value::as_str)
    {
        None | Some("") => JoinKey::Unjoinable("X's half binds no client nonce".into()),
        Some(HOST_SERVED_NO_NONCE) => {
            JoinKey::Unjoinable("X's half is host-served with no client nonce".into())
        }
        Some(_) => match crate::evidence_log::client_nonce_digest(record) {
            Some(d) => JoinKey::ClientNonce(d),
            None => JoinKey::Unjoinable("X's half binds no indexable client nonce".into()),
        },
    }
}

/// A run, checked: who to ask and on what key.
#[derive(Clone, Debug)]
pub struct Plan {
    pub x_peer: String,
    pub join_key: JoinKey,
    pub ys: Vec<AskY>,
}

/// Check `args`. An input that names a Y twice, no Y, X as a Y, an unknown
/// frame basis, or both or neither join-key source is refused: one bounded
/// ask per Y, stated up front.
pub fn plan(args: RunArgs) -> Result<Plan, String> {
    let x_peer = args.x_peer.trim().to_ascii_lowercase();
    if x_peer.is_empty() {
        return Err("x_peer must not be empty".into());
    }
    if args.ys.is_empty() {
        return Err("name at least one Y".into());
    }
    let mut seen = BTreeSet::new();
    let mut ys = Vec::with_capacity(args.ys.len());
    for mut y in args.ys {
        y.peer_id = y.peer_id.trim().to_ascii_lowercase();
        if y.peer_id.is_empty() || y.peer_id == x_peer {
            return Err("each Y must be a peer other than X".into());
        }
        if !seen.insert(y.peer_id.clone()) {
            return Err(format!("{} is named twice: one ask per Y", y.peer_id));
        }
        if !FRAME_BASES.contains(&y.frame.basis.as_str()) {
            return Err(format!(
                "frame basis {:?} is not one of {}",
                y.frame.basis,
                FRAME_BASES.join(", ")
            ));
        }
        if y.frame.basis == "other" && y.frame.note.as_deref().is_none_or(str::is_empty) {
            return Err("an `other` frame states its reason in `note`".into());
        }
        ys.push(y);
    }
    let join_key = match (args.client_nonce_sha256, args.x_record) {
        (Some(d), None) => {
            let d = d.trim().to_ascii_lowercase();
            if !is_digest(&d) {
                return Err("client_nonce_sha256 must be 64 hex".into());
            }
            if d == hex::encode(Sha256::digest(HOST_SERVED_NO_NONCE.as_bytes())) {
                JoinKey::Unjoinable(
                    "the key is the host-served placeholder, which names no exchange".into(),
                )
            } else {
                JoinKey::ClientNonce(d)
            }
        }
        (None, Some(record)) => join_key_of(&record),
        _ => return Err("give exactly one of client_nonce_sha256 and x_record".into()),
    };
    Ok(Plan {
        x_peer,
        join_key,
        ys,
    })
}

/// The request Z sends X and each Y: the same bytes but for what the
/// sender's signature adds.
pub fn request_for(key: &str) -> Value {
    json!({
        "subject": {"correlation": key},
        "coverage": {"min_freshness": 1},
        "derivation": DEFINITION_DIGEST,
    })
}

/// What came back from one peer, as the glue saw it.
#[derive(Clone, Debug)]
pub enum Reply {
    /// An answer arrived: the bytes sent, the JSON received, and the peer's
    /// key from Z's trust policy (`None` when the policy names none).
    Answered {
        sent: Vec<u8>,
        answer: Value,
        key: Option<Box<ed25519_dalek::VerifyingKey>>,
    },
    /// Nothing arrived by the deadline (unreachable, no channel, silence).
    Nothing(String),
    /// Bytes arrived that are not an answer.
    Malformed(String),
}

fn verified(reply: &Reply) -> Result<Verified, Outcome> {
    match reply {
        Reply::Answered { sent, answer, key } => match key {
            Some(key) => Ok(verify(sent, answer, key)),
            None => Err(Outcome::Unverifiable {
                why: "Z's trust policy names no key for this peer".into(),
            }),
        },
        Reply::Nothing(why) => Err(Outcome::RecordedAbsence { why: why.clone() }),
        Reply::Malformed(why) => Err(Outcome::Unverifiable { why: why.clone() }),
    }
}

/// After X answered: X's half, or the outcome that stops the run.
pub fn after_x(plan: &Plan, x: &Reply) -> Result<Half, Outcome> {
    let JoinKey::ClientNonce(_) = &plan.join_key else {
        return Err(Outcome::NotJoinable {
            why: "no join key".into(),
        });
    };
    match verified(x)? {
        Verified::Halves { halves, .. } => match halves.as_slice() {
            [one] => Ok(one.clone()),
            _ => Err(Outcome::NotJoinable {
                why: "X holds more than one half under this client nonce".into(),
            }),
        },
        other => Err(corroborate_x(&other)),
    }
}

/// X's own refusal or failure, as X's outcome.
fn corroborate_x(x: &Verified) -> Outcome {
    match x {
        Verified::Refusal(Reason::NoSuchSubject) => Outcome::OneSided,
        Verified::Refusal(reason) => Outcome::SignedRefusal { reason: *reason },
        Verified::Failed(why) => Outcome::Unverifiable { why: why.clone() },
        Verified::Halves { .. } => Outcome::Unverifiable {
            why: "unexpected".into(),
        },
    }
}

/// The run's result: X's outcome, then each Y's, in the order given. When X's
/// half was not obtained (or there is no key), `y_replies` is empty and no Y
/// was asked. No member summarizes across Ys.
pub fn finish(plan: &Plan, x: Result<Half, Outcome>, y_replies: &[Reply]) -> Value {
    let join_key = match &plan.join_key {
        JoinKey::ClientNonce(d) => json!({"kind": "client_nonce_sha256", "value": d}),
        JoinKey::Unjoinable(why) => json!({"kind": "none", "why": why}),
    };
    let x_json = match &x {
        Ok(half) => {
            json!({"peer_id": plan.x_peer, "outcome": {"state": "half_obtained"}, "half": half.to_json()})
        }
        Err(outcome) => json!({"peer_id": plan.x_peer, "outcome": outcome.to_json()}),
    };
    let ys: Vec<Value> = plan
        .ys
        .iter()
        .enumerate()
        .map(|(i, y)| {
            let outcome = match (&plan.join_key, &x, y_replies.get(i)) {
                (JoinKey::Unjoinable(why), _, _) => Outcome::NotJoinable { why: why.clone() },
                (_, Err(_), _) => Outcome::NotAsked {
                    why: "X's half was not obtained".into(),
                },
                (_, Ok(_), None) => Outcome::NotAsked {
                    why: "no reply was collected".into(),
                },
                (_, Ok(x_half), Some(reply)) => match verified(reply) {
                    Ok(v) => corroborate(x_half, &v),
                    Err(outcome) => outcome,
                },
            };
            let mut frame = json!({"basis": y.frame.basis});
            if let Some(note) = &y.frame.note {
                frame["note"] = json!(note);
            }
            json!({
                "peer_id": y.peer_id,
                "frame": frame,
                "outcome": outcome.to_json(),
                "independence": independence_json(y.independence_basis.as_deref()),
            })
        })
        .collect();
    json!({"join_key": join_key, "x": x_json, "ys": ys})
}

/// A Y's independence from X: never derived from its key, its signature, its
/// node id or its anchor (R-30).
fn independence_json(basis: Option<&str>) -> Value {
    match basis.map(str::trim).filter(|b| !b.is_empty()) {
        Some(basis) => json!({"state": "declared", "basis": basis, "declared_by": "operator"}),
        None => json!({"state": "not_established"}),
    }
}

/// Append `result` to Z's run log beside its ledger, with when it ran.
pub fn record_run(
    ledger_dir: &std::path::Path,
    result: &Value,
    ran_at: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut line = Map::new();
    line.insert("ran_at".into(), json!(ran_at));
    line.insert("result".into(), result.clone());
    let mut bytes = serde_json::to_vec(&Value::Object(line))?;
    bytes.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(RUNS_FILENAME))?;
    file.write_all(&bytes)
}

// ---------------------------------------------------------------------
// Mesh glue: the `historical_reference` operation
// ---------------------------------------------------------------------

/// The operation a local client calls to run one historical reference.
pub const HISTORICAL_REFERENCE_OPERATION: &str = "historical_reference";

/// Run one historical reference: ask X once, then (only when X's half was
/// obtained) each Y once, in order, over the evidence stream; record the run
/// in Z's own log; return the per-Y result.
pub async fn handle_historical_reference(
    args: RunArgs,
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    ledger_dir: std::path::PathBuf,
) -> mesh_llm_plugin::PluginResult<Value> {
    let plan = plan(args).map_err(mesh_llm_plugin::PluginError::invalid_params)?;
    let registry = crate::peer_keys::registry();
    let x = match &plan.join_key {
        JoinKey::ClientNonce(key) => {
            let reply = ask(context, &plan.x_peer, key, registry.as_deref()).await;
            after_x(&plan, &reply)
        }
        JoinKey::Unjoinable(why) => Err(Outcome::NotJoinable { why: why.clone() }),
    };
    let mut y_replies = Vec::new();
    if let (Ok(_), JoinKey::ClientNonce(key)) = (&x, &plan.join_key) {
        for y in &plan.ys {
            y_replies.push(ask(context, &y.peer_id, key, registry.as_deref()).await);
        }
    }
    let result = finish(&plan, x, &y_replies);
    record_run(&ledger_dir, &result, &crate::evidence_answer::now_utc()).map_err(|e| {
        mesh_llm_plugin::PluginError::internal(format!("could not record the run: {e}"))
    })?;
    Ok(result)
}

/// One ask of `peer` for its halves under `key`, as a [`Reply`].
async fn ask(
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    peer: &str,
    key: &str,
    registry: Option<&str>,
) -> Reply {
    use crate::mesh_evidence_bridge::{send_evidence_request, verifying_key, SendError};
    match send_evidence_request(context, peer, &request_for(key)).await {
        Ok((sent, answer)) => Reply::Answered {
            sent,
            answer,
            key: crate::peer_keys::announced_key_in(registry, peer)
                .and_then(|key_id| verifying_key(&key_id))
                .map(Box::new),
        },
        Err(SendError::NotOpened(e)) => Reply::Nothing(format!("could not reach the peer: {e}")),
        Err(SendError::NoAnswer(e)) => Reply::Nothing(format!("no answer by the deadline{e}")),
        Err(SendError::Malformed(e)) => Reply::Malformed(format!("not an answer: {e}")),
    }
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests;
