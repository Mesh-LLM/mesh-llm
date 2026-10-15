//! Split-inference stage records (`docs/DESIGN-split-stage-records.md`).
//!
//! A split run is modelled as nested exchanges: the requester's exchange with
//! the coordinator stays as it is, and each coordinator ⟷ stage pair becomes an
//! exchange of its own. This module owns the three record shapes that model
//! needs, all inside the existing exchange-record vocabulary:
//!
//! - the **stage record** a stage node seals for one request
//!   (`x-mesh-poc-v1.role = "stage"`, `x-mesh-stage-v1.side = "stage"`);
//! - the coordinator's **stage-exchange record** for each remote stage
//!   (`role = "stage"`, `side = "coordinator"`): what it assigned, what it saw
//!   first-hand, and a `counterparty_half` reference to the stage's own record
//!   once that arrived;
//! - the coordinator's **main record** of the requester's exchange (`role =
//!   "served"`, sealed through [`crate::producer::capsule::seal`]'s body) extended with
//!   its own slice (`stage_index: 0`), one `split_stage` reference per
//!   stage-exchange record, and the `x-mesh-coordinator-receipt-v1` block.
//!
//! The per-hop, per-lane frame fold ([`FrameFold`]) is defined here too, so the
//! producer, the fixtures and the verifier ([`crate::producer::stage_verify`]) share one
//! definition. What the records establish is stated narrowly in the design
//! doc §0: both ends of a hop committing to the same bytes, and which key
//! signed for which slice. Never that a stage computed its slice correctly.
//!
//! Nothing here reads a host event yet: the host seam (`skippy.stage.v1`) does
//! not exist, so [`StageBlock`] is deserialised from fixture events until it
//! does (design doc §8, §11 step 2).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::producer::capsule::{
    finish_seal, seal_body, seal_local_record, CapsuleInput, ChainLink, SealError,
    CHAIN_RELATION_FOLLOWS, CITATION_PURPOSE_COUNTERPARTY_HALF, REFERENCE_DIGEST_ALG,
    REFERENCE_TYPE_CAPSULE,
};

/// `x-mesh-poc-v1.role` of a record of a coordinator ⟷ stage exchange, on
/// either side of it. A reader that does not know this value must read the
/// record as `unknown`, never as `served`.
pub const ROLE_STAGE: &str = "stage";
/// The `compute_attestation` block naming one slice of a split.
pub const STAGE_BLOCK: &str = "x-mesh-stage-v1";
/// The `compute_attestation` block on the coordinator's main record: stage
/// order (`topology[]`) kept apart from what came back (`stages[]`).
pub const COORDINATOR_RECEIPT_BLOCK: &str = "x-mesh-coordinator-receipt-v1";
/// `references[].citation_purpose` from the coordinator's main record to each
/// of its stage-exchange records. PROVISIONAL: not registered
/// yet, hard-coded ahead of registration the way `counterparty_half` was.
pub const CITATION_PURPOSE_SPLIT_STAGE: &str = "split_stage";
/// Domain tag of the hop frame fold (design §4.2).
pub const FRAME_FOLD_DOMAIN: &[u8] = b"skippy-stage-frames/v1";
/// The receipt block's `kind`, unchanged from the Python reference emitter.
pub const COORDINATOR_RECEIPT_KIND: &str = "mesh-coordinator-receipt";
/// `model_attestation.model_id` of a stage record: a stage record describes a
/// slice of a package, never a whole model it served.
const STAGE_RECORD_MODEL_ID: &str = "n/a-split-stage";

/// The closed `terminal_state` set, identical to `mesh_record_verifier.py`'s
/// `TERMINAL_STATES`.
pub const TERMINAL_STATES: &[&str] = &[
    "completed",
    "policy_denied",
    "request_invalid",
    "backend_error",
    "transport_error",
    "client_cancelled",
    "timed_out",
    "evidence_unavailable",
];

/// The receipt's per-stage bundle states. `present` / `absent` /
/// `not_requested` are the Python reference's three; `conflict` is added for
/// two distinct records under one stage key (design §3): both are cited,
/// neither counts as present.
pub const BUNDLE_STATES: &[&str] = &["present", "absent", "not_requested", "conflict"];

/// Why a stage block, a receipt or a split record was refused.
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("x-mesh-stage-v1: {0}")]
    Block(String),
    #[error("x-mesh-coordinator-receipt-v1: {0}")]
    Receipt(String),
    #[error("split main record: {0}")]
    Main(String),
    #[error(transparent)]
    Seal(#[from] SealError),
    #[error(transparent)]
    Jcs(#[from] crate::producer::jcs::JcsError),
}

fn block_err(msg: impl Into<String>) -> StageError {
    StageError::Block(msg.into())
}

fn receipt_err(msg: impl Into<String>) -> StageError {
    StageError::Receipt(msg.into())
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Stage,
    Coordinator,
}

impl Side {
    fn wire(self) -> &'static str {
        match self {
            Side::Stage => "stage",
            Side::Coordinator => "coordinator",
        }
    }
}

/// How predicted tokens got back to stage 0: straight from the final stage
/// (`direct`) or hop by hop (`relayed`, the reverse fallback).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReturnMode {
    Direct,
    Relayed,
}

/// One of the three lanes a hop's frames are folded on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lane {
    Forward,
    Reply,
    DirectReturn,
}

impl Lane {
    pub fn wire(self) -> &'static str {
        match self {
            Lane::Forward => "forward",
            Lane::Reply => "reply",
            Lane::DirectReturn => "direct_return",
        }
    }
}

/// One lane's fold as a record carries it: how many frames, and the running
/// digest over them (`sha256:` + 64 lowercase hex).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaneFold {
    pub frames: u64,
    pub digest: String,
}

/// The hop to the previous stage, as this stage saw it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UpstreamHop {
    pub received: LaneFold,
    pub replies_sent: LaneFold,
}

/// The hop to the next stage, as this stage saw it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DownstreamHop {
    pub sent: LaneFold,
    pub replies_received: LaneFold,
}

/// The direct return lane: `sent` on the final stage, `received` on stage 0.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DirectReturn {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sent: Option<LaneFold>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub received: Option<LaneFold>,
}

/// Token counts named for what they count: prefill includes tokens restored
/// from the prefix cache, so it counts tokens received, not computed.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageTokens {
    pub prefill_received: u64,
    pub decode: u64,
}

/// What a coordinator saw first-hand of a remote stage's data path: its own
/// downstream end when the stage is stage 1, its own direct-return receive
/// end when the stage is the final one under `direct` return. Never anything
/// it only heard about.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorObserved {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downstream: Option<DownstreamHop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_return: Option<DirectReturn>,
}

/// The `x-mesh-stage-v1` block (design §4.1).
///
/// Three placements, told apart by `side` and `stage_index`:
/// - `side: stage`, `stage_index >= 1`: a stage node's own record;
/// - `side: coordinator`, `stage_index: 0`: the coordinator's own slice, on
///   its main record;
/// - `side: coordinator`, `stage_index >= 1`: the coordinator's
///   stage-exchange record for that stage. It carries the assignment and
///   `data_path_observed`, never the stage's hop fields.
///
/// `layer_end` is exclusive, as upstream counts it. `request_id` is a decimal
/// string: a u64 does not survive as a JSON number in every reader.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageBlock {
    pub v: u32,
    pub side: Side,
    pub coordinator_node_id: String,
    pub run_id: String,
    pub request_id: String,
    pub topology_hash: String,
    pub stage_index: u32,
    pub stage_count: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub package_id: String,
    pub manifest_sha256: String,
    pub source_model_sha256: String,
    pub return_mode: ReturnMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<UpstreamHop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downstream: Option<DownstreamHop>,
    /// Always serialised (`null` when this position has no direct lane), as
    /// the design's record shape shows it.
    #[serde(default)]
    pub direct_return: Option<DirectReturn>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<StageTokens>,
    pub terminal_state: String,
    /// Stage-exchange records only: the term the coordinator assigned under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinator_term: Option<u64>,
    /// Stage-exchange records only: `false` says outright that the coordinator
    /// saw nothing of this stage's data path, rather than leaving empty
    /// fields that could read as a pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_path_observed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinator_observed: Option<CoordinatorObserved>,
}

/// The key one stage exchange is recorded under (design §3). Two distinct
/// records under one key are a `conflict`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageKey {
    pub coordinator_node_id: String,
    pub run_id: String,
    pub request_id: String,
    pub stage_index: u32,
}

/// The key every record of one split request carries; rows nest on it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SplitKey {
    pub coordinator_node_id: String,
    pub run_id: String,
    pub request_id: String,
}

impl SplitKey {
    /// The console's bracket key, `split:<coordinator>/<run_id>/<request_id>`.
    pub fn bracket(&self) -> String {
        format!(
            "split:{}/{}/{}",
            self.coordinator_node_id, self.run_id, self.request_id
        )
    }
}

fn is_lower_hex(v: &str, len: usize) -> bool {
    v.len() == len
        && v.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_prefixed_sha256(v: &str) -> bool {
    v.strip_prefix("sha256:")
        .is_some_and(|h| is_lower_hex(h, 64))
}

/// A canonical decimal u64: digits only, no sign, no leading zero.
pub fn parse_request_id(raw: &str) -> Option<u64> {
    let canonical = !raw.is_empty()
        && raw.bytes().all(|b| b.is_ascii_digit())
        && (raw == "0" || !raw.starts_with('0'));
    if canonical {
        raw.parse().ok()
    } else {
        None
    }
}

fn check_fold(name: &str, fold: &LaneFold) -> Result<(), StageError> {
    if is_prefixed_sha256(&fold.digest) {
        Ok(())
    } else {
        Err(block_err(format!(
            "{name}.digest must be sha256: + 64 lowercase hex"
        )))
    }
}

fn check_upstream(hop: &UpstreamHop) -> Result<(), StageError> {
    check_fold("upstream.received", &hop.received)?;
    check_fold("upstream.replies_sent", &hop.replies_sent)
}

fn check_downstream(name: &str, hop: &DownstreamHop) -> Result<(), StageError> {
    check_fold(&format!("{name}.sent"), &hop.sent)?;
    check_fold(&format!("{name}.replies_received"), &hop.replies_received)
}

/// `direct_return` must hold exactly the one end this position owns.
fn check_direct_end(
    name: &str,
    direct: Option<&DirectReturn>,
    want: Option<Lane>,
    want_sent: bool,
) -> Result<(), StageError> {
    match (direct, want) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(block_err(format!("{name} must be null at this position"))),
        (None, Some(_)) => Err(block_err(format!("{name} is required at this position"))),
        (Some(d), Some(_)) => match (want_sent, &d.sent, &d.received) {
            (true, Some(sent), None) => check_fold(&format!("{name}.sent"), sent),
            (false, None, Some(received)) => check_fold(&format!("{name}.received"), received),
            (true, _, _) => Err(block_err(format!("{name} must carry sent only"))),
            (false, _, _) => Err(block_err(format!("{name} must carry received only"))),
        },
    }
}

impl StageBlock {
    pub fn key(&self) -> StageKey {
        StageKey {
            coordinator_node_id: self.coordinator_node_id.clone(),
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
            stage_index: self.stage_index,
        }
    }

    pub fn split_key(&self) -> SplitKey {
        SplitKey {
            coordinator_node_id: self.coordinator_node_id.clone(),
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
        }
    }

    pub fn is_final(&self) -> bool {
        self.stage_index + 1 == self.stage_count
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("a stage block is always JSON-serialisable")
    }

    /// Parse a block from a record or a fixture event, then [`Self::validate`] it.
    pub fn from_value(value: &Value) -> Result<Self, StageError> {
        let block: StageBlock = serde_json::from_value(value.clone())
            .map_err(|e| block_err(format!("does not parse: {e}")))?;
        block.validate()?;
        Ok(block)
    }

    /// The field and position rules of design §4.1 and §5. A block that fails
    /// here is never sealed, and a received one is never read as evidence.
    pub fn validate(&self) -> Result<(), StageError> {
        if self.v != 1 {
            return Err(block_err("v must be 1"));
        }
        for (name, value) in [
            ("coordinator_node_id", &self.coordinator_node_id),
            ("run_id", &self.run_id),
            ("topology_hash", &self.topology_hash),
        ] {
            if value.is_empty() {
                return Err(block_err(format!("{name} must be non-empty")));
            }
        }
        if parse_request_id(&self.request_id).is_none() {
            return Err(block_err(
                "request_id must be a canonical decimal u64 string",
            ));
        }
        if self.stage_count < 2 {
            return Err(block_err("stage_count must be at least 2 for a split"));
        }
        if self.stage_index >= self.stage_count {
            return Err(block_err("stage_index must be below stage_count"));
        }
        if self.layer_end <= self.layer_start {
            return Err(block_err(
                "layer_end is exclusive and must exceed layer_start",
            ));
        }
        if !is_prefixed_sha256(&self.package_id) {
            return Err(block_err("package_id must be sha256: + 64 lowercase hex"));
        }
        for (name, value) in [
            ("manifest_sha256", &self.manifest_sha256),
            ("source_model_sha256", &self.source_model_sha256),
        ] {
            if !is_lower_hex(value, 64) {
                return Err(block_err(format!("{name} must be 64 lowercase hex")));
            }
        }
        if !TERMINAL_STATES.contains(&self.terminal_state.as_str()) {
            return Err(block_err(format!(
                "terminal_state {:?} is not in the closed set",
                self.terminal_state
            )));
        }
        // Relayed return has no direct lane anywhere, so the position rules
        // below reject a non-null direct_return on every relayed block.
        let direct = self.return_mode == ReturnMode::Direct;
        match (self.side, self.stage_index) {
            (Side::Stage, 0) => Err(block_err(
                "stage 0 is the coordinator's own slice; a stage-side block starts at 1",
            )),
            (Side::Stage, _) => self.validate_stage_side(direct),
            (Side::Coordinator, 0) => self.validate_own_slice(direct),
            (Side::Coordinator, _) => self.validate_stage_exchange(direct),
        }
    }

    fn validate_stage_side(&self, direct: bool) -> Result<(), StageError> {
        self.forbid_exchange_fields()?;
        let upstream = self
            .upstream
            .as_ref()
            .ok_or_else(|| block_err("a stage past 0 must carry upstream"))?;
        check_upstream(upstream)?;
        match (&self.downstream, self.is_final()) {
            (Some(hop), false) => check_downstream("downstream", hop)?,
            (None, false) => {
                return Err(block_err("a stage before the last must carry downstream"))
            }
            (Some(_), true) => return Err(block_err("the final stage has no downstream")),
            (None, true) => {}
        }
        let want = (direct && self.is_final()).then_some(Lane::DirectReturn);
        check_direct_end("direct_return", self.direct_return.as_ref(), want, true)?;
        if self.tokens.is_none() {
            return Err(block_err("a stage record must carry tokens"));
        }
        Ok(())
    }

    fn validate_own_slice(&self, direct: bool) -> Result<(), StageError> {
        self.forbid_exchange_fields()?;
        if self.upstream.is_some() {
            return Err(block_err("stage 0 has no upstream"));
        }
        let downstream = self
            .downstream
            .as_ref()
            .ok_or_else(|| block_err("stage 0 must carry downstream"))?;
        check_downstream("downstream", downstream)?;
        let want = direct.then_some(Lane::DirectReturn);
        check_direct_end("direct_return", self.direct_return.as_ref(), want, false)?;
        if self.tokens.is_none() {
            return Err(block_err("the coordinator's own slice must carry tokens"));
        }
        Ok(())
    }

    fn validate_stage_exchange(&self, direct: bool) -> Result<(), StageError> {
        if self.upstream.is_some()
            || self.downstream.is_some()
            || self.direct_return.is_some()
            || self.tokens.is_some()
        {
            return Err(block_err(
                "a stage-exchange record carries no hop fields or tokens of the stage; \
                 first-hand observations go under coordinator_observed",
            ));
        }
        if self.coordinator_term.is_none() {
            return Err(block_err(
                "a stage-exchange record must carry coordinator_term",
            ));
        }
        let observed_flag = self
            .data_path_observed
            .ok_or_else(|| block_err("a stage-exchange record must carry data_path_observed"))?;
        let first_hop = self.stage_index == 1;
        let direct_final = direct && self.is_final();
        let can_observe = first_hop || direct_final;
        if observed_flag != can_observe {
            return Err(block_err(
                "data_path_observed must be true exactly for stage 1 and for the final stage \
                 under direct return",
            ));
        }
        match (&self.coordinator_observed, can_observe) {
            (None, false) => Ok(()),
            (Some(_), false) => Err(block_err("coordinator_observed without data_path_observed")),
            (None, true) => Err(block_err(
                "data_path_observed requires coordinator_observed",
            )),
            (Some(observed), true) => {
                match (&observed.downstream, first_hop) {
                    (Some(hop), true) => check_downstream("coordinator_observed.downstream", hop)?,
                    (None, true) => {
                        return Err(block_err(
                            "stage 1's exchange must carry the coordinator's downstream",
                        ))
                    }
                    (Some(_), false) => {
                        return Err(block_err(
                            "only stage 1's exchange carries the coordinator's downstream",
                        ))
                    }
                    (None, false) => {}
                }
                let want = direct_final.then_some(Lane::DirectReturn);
                check_direct_end(
                    "coordinator_observed.direct_return",
                    observed.direct_return.as_ref(),
                    want,
                    false,
                )
            }
        }
    }

    fn forbid_exchange_fields(&self) -> Result<(), StageError> {
        if self.coordinator_term.is_some()
            || self.data_path_observed.is_some()
            || self.coordinator_observed.is_some()
        {
            return Err(block_err(
                "coordinator_term / data_path_observed / coordinator_observed belong to \
                 stage-exchange records only",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The hop frame fold (design §4.2)
// ---------------------------------------------------------------------------

fn put_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).expect("fold fields are far below 4 GiB");
    hasher.update(len.to_be_bytes());
    hasher.update(bytes);
}

/// The running digest over one lane of one hop of one request.
///
/// ```text
/// h_0 = SHA-256( "skippy-stage-frames/v1" ‖ lp(salt) ‖ lp(coordinator_node_id)
///                ‖ lp(run_id) ‖ u64be(request_id) ‖ u32be(hop_index) ‖ lp(lane) )
/// h_i = SHA-256( h_{i-1} ‖ SHA-256(frame_i) )
/// ```
///
/// `lp(x)` is `u32be(len(x)) ‖ x`, so no two field sequences share a
/// preimage. `hop_index` is the upstream stage's index for the forward and
/// reply lanes, and the final stage's index for the direct return lane. The
/// salt is empty in v0 (design §4.4, Q-D3). `frame_i` is a frame's exact wire
/// bytes, before any parse or re-encode.
#[derive(Clone)]
pub struct FrameFold {
    state: [u8; 32],
    frames: u64,
}

impl FrameFold {
    pub fn new(
        salt: &[u8],
        coordinator_node_id: &str,
        run_id: &str,
        request_id: u64,
        hop_index: u32,
        lane: Lane,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(FRAME_FOLD_DOMAIN);
        put_len_prefixed(&mut hasher, salt);
        put_len_prefixed(&mut hasher, coordinator_node_id.as_bytes());
        put_len_prefixed(&mut hasher, run_id.as_bytes());
        hasher.update(request_id.to_be_bytes());
        hasher.update(hop_index.to_be_bytes());
        put_len_prefixed(&mut hasher, lane.wire().as_bytes());
        FrameFold {
            state: hasher.finalize().into(),
            frames: 0,
        }
    }

    pub fn push(&mut self, frame: &[u8]) {
        let frame_digest: [u8; 32] = Sha256::digest(frame).into();
        let mut hasher = Sha256::new();
        hasher.update(self.state);
        hasher.update(frame_digest);
        self.state = hasher.finalize().into();
        self.frames += 1;
    }

    pub fn finish(&self) -> LaneFold {
        LaneFold {
            frames: self.frames,
            digest: format!("sha256:{}", hex::encode(self.state)),
        }
    }
}

// ---------------------------------------------------------------------------
// Stage records and stage-exchange records
// ---------------------------------------------------------------------------

fn stage_action_id(block: &StageBlock) -> String {
    format!(
        "mesh-poc/split-stage/{}/{}/{}/{}/{}",
        block.side.wire(),
        block.coordinator_node_id,
        block.run_id,
        block.request_id,
        block.stage_index
    )
}

/// Seal a stage record (`side: stage`) or a coordinator's stage-exchange
/// record (`side: coordinator`, `stage_index >= 1`), chained onto
/// `chain_head` with `follows`, enveloped under `signing_key`.
///
/// `stage_record_id` is the stage's own record, cited `counterparty_half`
/// from a stage-exchange record once it has arrived. A stage record cites
/// nothing. The coordinator's own slice (`stage_index: 0`) is not a record of
/// its own: it rides on the main record ([`seal_split_main_record`]).
pub fn seal_stage_record(
    block: &StageBlock,
    stage_record_id: Option<&str>,
    chain_head: Option<&str>,
    signing_key: &ed25519_dalek::SigningKey,
) -> Result<Value, StageError> {
    block.validate()?;
    if block.side == Side::Coordinator && block.stage_index == 0 {
        return Err(block_err(
            "the coordinator's own slice rides on its main record, not a stage record",
        ));
    }
    let references = match (block.side, stage_record_id) {
        (Side::Stage, Some(_)) => return Err(block_err("a stage record cites nothing")),
        (Side::Stage, None) | (Side::Coordinator, None) => None,
        (Side::Coordinator, Some(id)) if is_lower_hex(id, 64) => Some(json!([{
            "type": REFERENCE_TYPE_CAPSULE,
            "digest_alg": REFERENCE_DIGEST_ALG,
            "digest": id,
            "citation_purpose": CITATION_PURPOSE_COUNTERPARTY_HALF,
        }])),
        (Side::Coordinator, Some(_)) => {
            return Err(block_err(
                "the cited stage record id must be 64 lowercase hex",
            ))
        }
    };
    let mut blocks = Map::new();
    blocks.insert("x-mesh-poc-v1".into(), json!({ "role": ROLE_STAGE }));
    blocks.insert(STAGE_BLOCK.into(), block.to_value());
    let chain = chain_head.map(|parent| ChainLink {
        parent_capsule_id: parent.to_string(),
        relation: CHAIN_RELATION_FOLLOWS.to_string(),
    });
    Ok(seal_local_record(
        stage_action_id(block),
        STAGE_RECORD_MODEL_ID,
        blocks,
        references,
        chain,
        signing_key,
    )?)
}

/// The stage block a record carries, if it is a split-stage record at all.
/// `Ok(None)` for every other record; `Err` for a record that claims to be
/// one and does not parse or validate.
pub fn stage_block_of(record: &Value) -> Result<Option<StageBlock>, StageError> {
    let compute_attestation = record
        .get("model_attestation")
        .and_then(|m| m.get("compute_attestation"));
    match compute_attestation.and_then(|ca| ca.get(STAGE_BLOCK)) {
        None => Ok(None),
        Some(block) => StageBlock::from_value(block).map(Some),
    }
}

// ---------------------------------------------------------------------------
// The coordinator receipt (moved from `mesh_coordinator_receipt_emitter.py`)
// ---------------------------------------------------------------------------

/// What the coordinator assigned one stage: which node, which layers, which
/// package. The stage strip's `✓` compares a stage record against this.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageAssignment {
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub package_id: String,
}

/// One hop of the coordinator's own claim of what it routed. Same shape as
/// the Python reference's `TopologyEntry`, plus the assignment. For a split,
/// `hop_id` is `stage-<k>`, `seq` is `k`, and `role` is `coordinator` for
/// stage 0 and `stage` for the rest.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TopologyEntry {
    pub seq: u32,
    pub hop_id: String,
    pub role: String,
    pub observation_point: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignment: Option<StageAssignment>,
}

/// A CPB typed digest reference to a stage's own record.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundleRef {
    #[serde(rename = "type")]
    pub ref_type: String,
    pub digest_alg: String,
    pub digest: String,
}

impl BundleRef {
    pub fn capsule(capsule_id: &str) -> Self {
        BundleRef {
            ref_type: REFERENCE_TYPE_CAPSULE.to_string(),
            digest_alg: REFERENCE_DIGEST_ALG.to_string(),
            digest: capsule_id.to_string(),
        }
    }

    fn check(&self) -> Result<(), StageError> {
        if self.ref_type.trim().is_empty() {
            return Err(receipt_err("bundle_ref.type must be a non-empty string"));
        }
        if self.digest_alg != REFERENCE_DIGEST_ALG {
            return Err(receipt_err("bundle_ref.digest_alg must be SHA-256"));
        }
        if !is_lower_hex(&self.digest, 64) {
            return Err(receipt_err("bundle_ref.digest must be 64 lowercase hex"));
        }
        Ok(())
    }
}

/// What actually came back for one hop.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageEntry {
    pub hop_id: String,
    pub bundle: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_ref: Option<BundleRef>,
    /// `conflict` only: every distinct record received under the stage key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_refs: Option<Vec<BundleRef>>,
}

/// The `x-mesh-coordinator-receipt-v1` block. `topology[]` ("I routed this")
/// and `stages[]` ("I hold proof of this") stay two arrays, never merged.
/// `coordinator_node_id`, `request_id` and `stage_seal_deadline_ms` ride as
/// the Python emitter's `extra` members.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorReceipt {
    pub v: u32,
    pub kind: String,
    pub run_id: String,
    pub topology: Vec<TopologyEntry>,
    pub stages: Vec<StageEntry>,
    pub coordinator_node_id: String,
    pub request_id: String,
    /// How long the coordinator waited for stage records before sealing, so
    /// `absent` can be read against it (design Q-D5).
    pub stage_seal_deadline_ms: u64,
}

/// The hop id of stage `k` in a split receipt.
pub fn hop_id(stage_index: u32) -> String {
    format!("stage-{stage_index}")
}

impl CoordinatorReceipt {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("a receipt is always JSON-serialisable")
    }

    pub fn from_value(value: &Value) -> Result<Self, StageError> {
        let receipt: CoordinatorReceipt = serde_json::from_value(value.clone())
            .map_err(|e| receipt_err(format!("does not parse: {e}")))?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn stage_count(&self) -> u32 {
        u32::try_from(self.topology.len()).unwrap_or(u32::MAX)
    }

    pub fn split_key(&self) -> SplitKey {
        SplitKey {
            coordinator_node_id: self.coordinator_node_id.clone(),
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
        }
    }

    pub fn entry(&self, stage_index: u32) -> Option<&StageEntry> {
        self.stages.get(usize::try_from(stage_index).ok()?)
    }

    pub fn assignment(&self, stage_index: u32) -> Option<&StageAssignment> {
        self.topology
            .get(usize::try_from(stage_index).ok()?)?
            .assignment
            .as_ref()
    }

    /// The Python reference's producer rules, plus the split-specific ones.
    pub fn validate(&self) -> Result<(), StageError> {
        if self.v != 1 || self.kind != COORDINATOR_RECEIPT_KIND {
            return Err(receipt_err("v must be 1 and kind mesh-coordinator-receipt"));
        }
        if self.run_id.is_empty() || self.coordinator_node_id.is_empty() {
            return Err(receipt_err(
                "run_id and coordinator_node_id must be non-empty",
            ));
        }
        if parse_request_id(&self.request_id).is_none() {
            return Err(receipt_err(
                "request_id must be a canonical decimal u64 string",
            ));
        }
        if self.topology.len() < 2 {
            return Err(receipt_err("a split has at least two stages"));
        }
        if self.stages.len() != self.topology.len() {
            return Err(receipt_err(
                "stages[] must have exactly one entry per topology[] hop",
            ));
        }
        for (k, (hop, stage)) in self.topology.iter().zip(&self.stages).enumerate() {
            let k = u32::try_from(k).map_err(|_| receipt_err("too many stages"))?;
            let want_role = if k == 0 { "coordinator" } else { "stage" };
            if hop.seq != k || hop.hop_id != hop_id(k) || hop.role != want_role {
                return Err(receipt_err(format!(
                    "topology[{k}] must be seq {k}, hop_id {}, role {want_role}",
                    hop_id(k)
                )));
            }
            let assignment = hop
                .assignment
                .as_ref()
                .ok_or_else(|| receipt_err(format!("topology[{k}] must carry its assignment")))?;
            if assignment.node_id.is_empty()
                || assignment.layer_end <= assignment.layer_start
                || !is_prefixed_sha256(&assignment.package_id)
            {
                return Err(receipt_err(format!(
                    "topology[{k}].assignment is malformed"
                )));
            }
            if k == 0 && assignment.node_id != self.coordinator_node_id {
                return Err(receipt_err("stage 0 is the coordinator's own slice"));
            }
            if stage.hop_id != hop.hop_id {
                return Err(receipt_err(format!(
                    "stages[{k}].hop_id must match topology[{k}]"
                )));
            }
            check_stage_entry(k, stage)?;
        }
        let mut cited = BTreeSet::new();
        for stage in &self.stages {
            let refs = stage
                .bundle_ref
                .iter()
                .chain(stage.bundle_refs.iter().flatten());
            for r in refs {
                if !cited.insert(r.digest.as_str()) {
                    return Err(receipt_err(format!(
                        "record {} is cited under more than one hop",
                        r.digest
                    )));
                }
            }
        }
        Ok(())
    }
}

fn check_stage_entry(k: u32, stage: &StageEntry) -> Result<(), StageError> {
    match (stage.bundle.as_str(), &stage.bundle_ref, &stage.bundle_refs) {
        ("not_requested", None, None) => Ok(()),
        (_, _, _) if k == 0 => Err(receipt_err(
            "stage 0 is the coordinator's own slice: its bundle is not_requested",
        )),
        ("absent", None, None) => Ok(()),
        ("present", Some(r), None) => r.check(),
        ("present", _, _) => Err(receipt_err(format!(
            "stages[{k}]: present requires exactly one bundle_ref"
        ))),
        ("absent" | "not_requested", _, _) => Err(receipt_err(format!(
            "stages[{k}]: {} must not carry a bundle_ref",
            stage.bundle
        ))),
        ("conflict", None, Some(refs)) => {
            let distinct: BTreeSet<&str> = refs.iter().map(|r| r.digest.as_str()).collect();
            if refs.len() < 2 || distinct.len() != refs.len() {
                return Err(receipt_err(format!(
                    "stages[{k}]: conflict cites two or more distinct records"
                )));
            }
            refs.iter().try_for_each(BundleRef::check)
        }
        ("conflict", _, _) => Err(receipt_err(format!(
            "stages[{k}]: conflict carries bundle_refs only"
        ))),
        (other, _, _) => Err(receipt_err(format!(
            "stages[{k}].bundle {other:?} is not one of {BUNDLE_STATES:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// The coordinator's main record
// ---------------------------------------------------------------------------

/// What turns an ordinary served exchange record into the coordinator's main
/// record of a split request.
pub struct SplitMainExtension {
    /// The coordinator's own slice: `side: coordinator`, `stage_index: 0`.
    pub own_slice: StageBlock,
    pub receipt: CoordinatorReceipt,
    /// The coordinator's stage-exchange record for each remote stage, as
    /// `(stage_index, capsule_id)`. Exactly one per stage 1..N-1.
    pub stage_exchange_records: Vec<(u32, String)>,
}

impl SplitMainExtension {
    pub fn validate(&self) -> Result<(), StageError> {
        let own = &self.own_slice;
        own.validate()?;
        if own.side != Side::Coordinator || own.stage_index != 0 {
            return Err(StageError::Main(
                "own_slice must be side coordinator, stage_index 0".into(),
            ));
        }
        self.receipt.validate()?;
        if own.split_key() != self.receipt.split_key() {
            return Err(StageError::Main(
                "own slice and receipt name different splits".into(),
            ));
        }
        if own.stage_count != self.receipt.stage_count() {
            return Err(StageError::Main(
                "own slice's stage_count must match the topology".into(),
            ));
        }
        let assigned = self
            .receipt
            .assignment(0)
            .expect("validated receipt has stage 0");
        if (
            assigned.layer_start,
            assigned.layer_end,
            assigned.package_id.as_str(),
        ) != (own.layer_start, own.layer_end, own.package_id.as_str())
        {
            return Err(StageError::Main(
                "own slice must match its own assignment".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for (k, id) in &self.stage_exchange_records {
            if *k == 0 || *k >= own.stage_count || !seen.insert(*k) {
                return Err(StageError::Main(format!(
                    "stage-exchange record for stage {k} is out of range or repeated"
                )));
            }
            if !is_lower_hex(id, 64) {
                return Err(StageError::Main(
                    "stage-exchange record ids must be 64 lowercase hex".into(),
                ));
            }
        }
        if seen.len() + 1 != self.receipt.topology.len() {
            return Err(StageError::Main(
                "the main record cites exactly one stage-exchange record per remote stage".into(),
            ));
        }
        Ok(())
    }

    fn references(&self) -> Value {
        let mut ordered = self.stage_exchange_records.clone();
        ordered.sort_by_key(|(k, _)| *k);
        Value::Array(
            ordered
                .iter()
                .map(|(_, id)| {
                    json!({
                        "type": REFERENCE_TYPE_CAPSULE,
                        "digest_alg": REFERENCE_DIGEST_ALG,
                        "digest": id,
                        "citation_purpose": CITATION_PURPOSE_SPLIT_STAGE,
                    })
                })
                .collect(),
        )
    }
}

/// Seal the coordinator's main record of a split request: `input` exactly as
/// [`crate::producer::capsule::seal`] would seal it (it must be a `served` record),
/// plus the own-slice block, the receipt and the `split_stage` references, all
/// committed under the one `capsule_id`. The caller attaches the envelope, as
/// for every exchange record.
pub fn seal_split_main_record(
    input: &CapsuleInput,
    split: &SplitMainExtension,
) -> Result<Value, StageError> {
    if input.mesh_poc.role != "served" {
        return Err(StageError::Main(
            "the coordinator's main record is a served record".into(),
        ));
    }
    split.validate()?;
    let mut body = seal_body(input)?;
    let compute_attestation = body
        .get_mut("model_attestation")
        .and_then(|m| m.get_mut("compute_attestation"))
        .and_then(Value::as_object_mut)
        .expect("seal_body always builds compute_attestation");
    compute_attestation.insert(STAGE_BLOCK.into(), split.own_slice.to_value());
    compute_attestation.insert(COORDINATOR_RECEIPT_BLOCK.into(), split.receipt.to_value());
    body.insert("references".into(), split.references());
    Ok(finish_seal(body)?)
}
