//! Split-inference stage records on the live plugin
//! (`docs/DESIGN-split-stage-records.md` §6).
//!
//! **Stage side.** One `skippy.stage.v1` event per request per stage: the
//! plugin seals the stage record and pushes it to the coordinator over
//! `record-push/1`. The stage never learns the end requester.
//!
//! **Coordinator side.** The stage-0 event names the split key, the
//! assignments and the `openai.exchange.v1` `exchange_id` of the same
//! request. [`SplitCollector`] holds that exchange's terminal envelope instead
//! of sealing it at once, gathers the stage records the receiver accepts, and
//! releases the split when every remote stage has arrived or the deadline
//! passes. [`plan_split`] then turns what arrived into the stage-exchange
//! records, the receipt and the stage records to carry to the requester. A
//! stage record that arrives after that is still received, and its ordinary
//! `counterparty_half` citing record is the follow-up; the sealed main record
//! is never rewritten.
//!
//! **No host emits `skippy.stage.v1` yet** (design §8). The channel is not
//! declared in the manifest, and the event shape here is the one the host
//! seam is asked for. The contract this module relies on, beyond §8: stage 0
//! emits its event before the `openai.exchange.v1` terminal event of the
//! same request. A terminal event that arrives first is sealed as an
//! ordinary exchange, and the late stage-0 event is refused and counted.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;

use crate::producer::stage::{
    hop_id, stage_block_of, BundleRef, CoordinatorObserved, CoordinatorReceipt, DirectReturn,
    ReturnMode, Side, SplitKey, StageAssignment, StageBlock, StageEntry, StageError, TopologyEntry,
    COORDINATOR_RECEIPT_KIND,
};
use serde::Deserialize;
use serde_json::Value;

/// The host channel the stage runtime would publish on (design §8, not yet
/// served by any host).
pub const SKIPPY_STAGE_CHANNEL: &str = "skippy.stage.v1";
/// How long the coordinator waits for stage records before it seals, in ms.
pub const ENV_STAGE_DEADLINE_MS: &str = "CAPSULES_SPLIT_STAGE_DEADLINE_MS";
const DEFAULT_STAGE_DEADLINE_MS: u64 = 2_000;
/// Splits held at once. A peer can push stage records for splits that never
/// complete here; past this, the oldest split opened by records alone is
/// dropped and counted. A split holding its terminal envelope is never
/// dropped for room; when nothing else can go, the newcomer is refused.
const MAX_PENDING: usize = 256;
/// Splits one sender may open with stage records before any stage-0 event.
const MAX_PENDING_PER_SENDER: usize = 8;
/// Distinct records one sender may supply for one stage of one split: a
/// second is kept (two signed records for one hop is the `conflict` the
/// receipt reports), a third is refused.
const MAX_RECORDS_PER_STAGE_SENDER: usize = 2;
/// Recently sealed split keys remembered, so a late stage record is routed to
/// the follow-up path instead of opening a new pending split.
const MAX_SEALED_MEMORY: usize = 1_024;
/// A pending split that never gets its stage-0 event or its terminal envelope
/// is dropped after this many deadlines.
const EXPIRY_DEADLINES: u64 = 10;

/// Milliseconds on a monotonic clock, for the collector's deadlines.
pub fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
}

pub fn stage_deadline_ms() -> u64 {
    crate::settings::var(ENV_STAGE_DEADLINE_MS)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(DEFAULT_STAGE_DEADLINE_MS)
}

/// One `skippy.stage.v1` event: the stage's block, and at stage 0 only, the
/// facts only the coordinator's host holds.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StageEvent {
    pub block: StageBlock,
    /// Stage 0 only: the same request's `openai.exchange.v1` exchange id.
    #[serde(default)]
    pub exchange_id: Option<String>,
    /// Stage 0 only: the term the coordinator assigned under.
    #[serde(default)]
    pub coordinator_term: Option<u64>,
    /// Stage 0 only: what each stage was assigned, stage 0 first.
    #[serde(default)]
    pub topology: Option<Vec<StageAssignment>>,
}

impl StageEvent {
    pub fn parse(bytes: &[u8]) -> Result<Self, StageError> {
        let event: StageEvent = serde_json::from_slice(bytes)
            .map_err(|e| StageError::Block(format!("skippy.stage.v1 event does not parse: {e}")))?;
        event.validate()?;
        Ok(event)
    }

    pub fn validate(&self) -> Result<(), StageError> {
        self.block.validate()?;
        let coordinator_facts = self.exchange_id.is_some()
            || self.coordinator_term.is_some()
            || self.topology.is_some();
        match (self.block.side, self.block.stage_index) {
            (Side::Stage, _) if coordinator_facts => Err(StageError::Block(
                "a stage's event carries no exchange_id, coordinator_term or topology".into(),
            )),
            (Side::Stage, _) => Ok(()),
            (Side::Coordinator, 0) => self.validate_stage_zero(),
            (Side::Coordinator, _) => Err(StageError::Block(
                "stage-exchange records are the plugin's own, never a host event".into(),
            )),
        }
    }

    fn validate_stage_zero(&self) -> Result<(), StageError> {
        let (Some(exchange_id), Some(_), Some(topology)) =
            (&self.exchange_id, self.coordinator_term, &self.topology)
        else {
            return Err(StageError::Block(
                "stage 0's event must carry exchange_id, coordinator_term and topology".into(),
            ));
        };
        if exchange_id.is_empty() {
            return Err(StageError::Block(
                "stage 0's exchange_id must be non-empty".into(),
            ));
        }
        let own = &self.block;
        if u32::try_from(topology.len()).ok() != Some(own.stage_count) {
            return Err(StageError::Block("topology must list every stage".into()));
        }
        let first = &topology[0];
        if first.node_id != own.coordinator_node_id
            || (
                first.layer_start,
                first.layer_end,
                first.package_id.as_str(),
            ) != (own.layer_start, own.layer_end, own.package_id.as_str())
        {
            return Err(StageError::Block(
                "topology[0] must be the coordinator's own slice as its block states it".into(),
            ));
        }
        Ok(())
    }
}

/// What the collector releases: a split ready to seal.
pub struct ReadySplit<E> {
    pub own: StageEvent,
    pub envelope: E,
    /// Stage records received, by stage index, then by `capsule_id`.
    pub arrived: BTreeMap<u32, BTreeMap<String, Value>>,
}

/// A held stage record and the mesh peer that pushed it.
struct Held {
    sender: String,
    record: Value,
}

struct Pending<E> {
    own: Option<StageEvent>,
    /// When the stage-0 event arrived: the deadline counts from here.
    opened_ms: Option<u64>,
    first_seen_ms: u64,
    arrived: BTreeMap<u32, BTreeMap<String, Held>>,
    envelope: Option<E>,
    /// The sender whose stage record opened this split before its stage-0
    /// event, for the per-sender cap.
    opened_by: Option<String>,
}

impl<E> Pending<E> {
    fn new(now_ms: u64, opened_by: Option<String>) -> Self {
        Pending {
            own: None,
            opened_ms: None,
            first_seen_ms: now_ms,
            arrived: BTreeMap::new(),
            envelope: None,
            opened_by,
        }
    }
}

/// The node the coordinator assigned stage `k` to, when stage 0 has named it.
fn assigned_node(own: &StageEvent, k: u32) -> Option<&str> {
    own.topology
        .as_ref()?
        .get(k as usize)
        .map(|a| a.node_id.as_str())
}

/// What happened to a stage record handed to the collector.
#[derive(Debug, PartialEq, Eq)]
pub enum Arrival {
    /// Held for a split still waiting on its stage records.
    Held,
    /// Its split was already sealed. The ordinary citing record is the
    /// follow-up.
    Late,
    /// Not a stage record of a split, or not one this node coordinates.
    NotStage,
    /// Refused and counted: not from the node assigned that stage, or past a
    /// holding cap.
    Refused,
}

struct Inner<E> {
    pending: BTreeMap<SplitKey, Pending<E>>,
    by_exchange: HashMap<String, SplitKey>,
    sealed: VecDeque<SplitKey>,
    dropped: u64,
    refused: u64,
}

/// The coordinator's in-memory waiting room for split requests. Generic over
/// the held terminal envelope so it is tested without a host.
pub struct SplitCollector<E> {
    inner: Mutex<Inner<E>>,
}

impl<E> Default for SplitCollector<E> {
    fn default() -> Self {
        SplitCollector {
            inner: Mutex::new(Inner {
                pending: BTreeMap::new(),
                by_exchange: HashMap::new(),
                sealed: VecDeque::new(),
                dropped: 0,
                refused: 0,
            }),
        }
    }
}

impl<E> SplitCollector<E> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<E>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Pending splits dropped for capacity or expiry, and stage-0 events
    /// refused because their exchange was already sealed.
    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }

    /// Stage records refused: from a node other than the one assigned that
    /// stage, or past a holding cap.
    pub fn refused(&self) -> u64 {
        self.lock().refused
    }

    /// A stage-0 event: open (or complete) the pending split for its key.
    /// Refused when the split was already sealed or already has one.
    pub fn on_stage_zero(&self, event: StageEvent, now_ms: u64) -> Result<(), StageError> {
        event.validate()?;
        if event.block.side != Side::Coordinator || event.block.stage_index != 0 {
            return Err(StageError::Block("not a stage-0 event".into()));
        }
        let key = event.block.split_key();
        let exchange_id = event.exchange_id.clone().expect("validated stage-0 event");
        let mut inner = self.lock();
        if inner.sealed.contains(&key) || inner.by_exchange.contains_key(&exchange_id) {
            inner.dropped += 1;
            return Err(StageError::Block(
                "this split already has its stage-0 event".into(),
            ));
        }
        if !make_room(&mut inner, &key) {
            inner.dropped += 1;
            return Err(StageError::Block("no room to hold another split".into()));
        }
        let pending = inner
            .pending
            .entry(key.clone())
            .or_insert_with(|| Pending::new(now_ms, None));
        if pending.own.is_some() {
            inner.dropped += 1;
            return Err(StageError::Block(
                "this split already has its stage-0 event".into(),
            ));
        }
        // Records that arrived first are kept only from the node stage 0 now
        // names for their stage.
        let before: usize = pending.arrived.values().map(BTreeMap::len).sum();
        let mut arrived = std::mem::take(&mut pending.arrived);
        for (k, held) in arrived.iter_mut() {
            let assigned = (*k >= 1).then(|| assigned_node(&event, *k)).flatten();
            held.retain(|_, h| Some(h.sender.as_str()) == assigned);
        }
        arrived.retain(|_, held| !held.is_empty());
        let after: usize = arrived.values().map(BTreeMap::len).sum();
        pending.arrived = arrived;
        pending.own = Some(event);
        pending.opened_ms = Some(now_ms);
        pending.opened_by = None;
        inner.refused += (before - after) as u64;
        inner.by_exchange.insert(exchange_id, key);
        Ok(())
    }

    /// The terminal envelope of `exchange_id`: held when a pending split waits
    /// for it, handed back otherwise (the caller seals it as usual).
    pub fn hold(&self, exchange_id: Option<&str>, envelope: E) -> Option<E> {
        let Some(exchange_id) = exchange_id else {
            return Some(envelope);
        };
        let mut inner = self.lock();
        let Some(key) = inner.by_exchange.get(exchange_id).cloned() else {
            return Some(envelope);
        };
        match inner.pending.get_mut(&key) {
            Some(pending) if pending.envelope.is_none() => {
                pending.envelope = Some(envelope);
                None
            }
            _ => Some(envelope),
        }
    }

    /// A stage record the receiver accepted, pushed by mesh peer `sender`. Once
    /// stage 0 has named the topology, only the node assigned that stage may
    /// supply its record.
    pub fn on_stage_record(&self, record: &Value, sender: &str, now_ms: u64) -> Arrival {
        let Ok(Some(block)) = stage_block_of(record) else {
            return Arrival::NotStage;
        };
        let Some(capsule_id) = record.get("capsule_id").and_then(Value::as_str) else {
            return Arrival::NotStage;
        };
        if block.side != Side::Stage {
            return Arrival::NotStage;
        }
        let key = block.split_key();
        let mut inner = self.lock();
        if inner.sealed.contains(&key) {
            return Arrival::Late;
        }
        if !inner.pending.contains_key(&key) {
            let opened_by_sender = inner
                .pending
                .values()
                .filter(|p| p.own.is_none() && p.opened_by.as_deref() == Some(sender))
                .count();
            if opened_by_sender >= MAX_PENDING_PER_SENDER || !make_room(&mut inner, &key) {
                inner.refused += 1;
                return Arrival::Refused;
            }
            inner
                .pending
                .insert(key.clone(), Pending::new(now_ms, Some(sender.to_string())));
        }
        let pending = inner.pending.get_mut(&key).expect("pending split exists");
        if let Some(own) = &pending.own {
            if block.stage_index >= own.block.stage_count {
                return Arrival::NotStage;
            }
            let assigned = (block.stage_index >= 1)
                .then(|| assigned_node(own, block.stage_index))
                .flatten();
            if assigned != Some(sender) {
                inner.refused += 1;
                return Arrival::Refused;
            }
        }
        let held = pending.arrived.entry(block.stage_index).or_default();
        if held.contains_key(capsule_id) {
            return Arrival::Held;
        }
        if held.values().filter(|h| h.sender == sender).count() >= MAX_RECORDS_PER_STAGE_SENDER {
            inner.refused += 1;
            return Arrival::Refused;
        }
        held.insert(
            capsule_id.to_string(),
            Held {
                sender: sender.to_string(),
                record: record.clone(),
            },
        );
        Arrival::Held
    }

    /// Release every split that has its stage-0 event and its envelope, and
    /// either every remote stage's record or a passed deadline. Drops splits
    /// that never became sealable within [`EXPIRY_DEADLINES`] deadlines.
    pub fn take_due(&self, now_ms: u64, deadline_ms: u64) -> Vec<ReadySplit<E>> {
        let mut inner = self.lock();
        let expiry = deadline_ms.saturating_mul(EXPIRY_DEADLINES);
        let mut due = Vec::new();
        let mut expired = Vec::new();
        for (key, pending) in &inner.pending {
            match (&pending.own, pending.opened_ms, &pending.envelope) {
                (Some(own), Some(opened), Some(_)) => {
                    let all_arrived =
                        (1..own.block.stage_count).all(|k| pending.arrived.contains_key(&k));
                    if all_arrived || now_ms.saturating_sub(opened) >= deadline_ms {
                        due.push(key.clone());
                    }
                }
                _ if now_ms.saturating_sub(pending.first_seen_ms) >= expiry => {
                    expired.push(key.clone())
                }
                _ => {}
            }
        }
        for key in expired {
            forget(&mut inner, &key);
            inner.dropped += 1;
        }
        let mut ready = Vec::new();
        for key in due {
            let pending = inner.pending.remove(&key).expect("due key is pending");
            let own = pending.own.expect("due split has its stage-0 event");
            inner.by_exchange.retain(|_, k| *k != key);
            remember_sealed(&mut inner, key);
            let arrived = pending
                .arrived
                .into_iter()
                .map(|(k, held)| (k, held.into_iter().map(|(id, h)| (id, h.record)).collect()))
                .collect();
            ready.push(ReadySplit {
                own,
                envelope: pending.envelope.expect("due split holds its envelope"),
                arrived,
            });
        }
        ready
    }
}

fn forget<E>(inner: &mut Inner<E>, key: &SplitKey) {
    inner.pending.remove(key);
    inner.by_exchange.retain(|_, k| k != key);
}

fn remember_sealed<E>(inner: &mut Inner<E>, key: SplitKey) {
    if inner.sealed.len() == MAX_SEALED_MEMORY {
        inner.sealed.pop_front();
    }
    inner.sealed.push_back(key);
}

/// Keep at most [`MAX_PENDING`] splits to admit `key`: drop the oldest split
/// opened by stage records alone, else the oldest still waiting for its
/// terminal envelope. A split that holds its envelope is never dropped (its
/// exchange would never seal). `false` when nothing can go.
fn make_room<E>(inner: &mut Inner<E>, key: &SplitKey) -> bool {
    if inner.pending.contains_key(key) || inner.pending.len() < MAX_PENDING {
        return true;
    }
    let oldest_where = |inner: &Inner<E>, pick: fn(&Pending<E>) -> bool| {
        inner
            .pending
            .iter()
            .filter(|(_, p)| pick(p))
            .min_by_key(|(_, p)| p.first_seen_ms)
            .map(|(k, _)| k.clone())
    };
    let victim = oldest_where(inner, |p| p.own.is_none())
        .or_else(|| oldest_where(inner, |p| p.envelope.is_none()));
    match victim {
        Some(victim) => {
            forget(inner, &victim);
            inner.dropped += 1;
            true
        }
        None => false,
    }
}

/// What the coordinator seals for a released split, before the main record.
#[derive(Debug)]
pub struct SplitPlan {
    pub own_slice: StageBlock,
    /// One stage-exchange block per remote stage, stage 1 first, with the
    /// stage record it cites when exactly one arrived.
    pub exchanges: Vec<(StageBlock, Option<String>)>,
    /// The receipt, `bundle_ref`s filled from what arrived.
    pub receipt: CoordinatorReceipt,
    /// The stage records the receipt cites, to carry to the requester.
    pub carried: Vec<Value>,
}

/// Turn a released split into its plan. Zero records for a stage is
/// `absent`, one is `present`, two or more distinct ones are `conflict`.
pub fn plan_split<E>(ready: &ReadySplit<E>, deadline_ms: u64) -> Result<SplitPlan, StageError> {
    let own = &ready.own.block;
    let topology = ready
        .own
        .topology
        .as_ref()
        .expect("validated stage-0 event");
    let term = ready.own.coordinator_term.expect("validated stage-0 event");
    let mut exchanges = Vec::new();
    let mut entries = vec![StageEntry {
        hop_id: hop_id(0),
        bundle: "not_requested".into(),
        bundle_ref: None,
        bundle_refs: None,
    }];
    let mut carried = Vec::new();
    for k in 1..own.stage_count {
        let assignment = &topology[k as usize];
        let arrived = ready.arrived.get(&k);
        let ids: Vec<&String> = arrived.map(|m| m.keys().collect()).unwrap_or_default();
        let (entry, cite) = match ids.as_slice() {
            [] => (
                StageEntry {
                    hop_id: hop_id(k),
                    bundle: "absent".into(),
                    bundle_ref: None,
                    bundle_refs: None,
                },
                None,
            ),
            [id] => (
                StageEntry {
                    hop_id: hop_id(k),
                    bundle: "present".into(),
                    bundle_ref: Some(BundleRef::capsule(id)),
                    bundle_refs: None,
                },
                Some((*id).clone()),
            ),
            many => (
                StageEntry {
                    hop_id: hop_id(k),
                    bundle: "conflict".into(),
                    bundle_ref: None,
                    bundle_refs: Some(many.iter().map(|id| BundleRef::capsule(id)).collect()),
                },
                None,
            ),
        };
        carried.extend(arrived.into_iter().flat_map(|m| m.values().cloned()));
        entries.push(entry);
        exchanges.push((exchange_block(own, assignment, k, term), cite));
    }
    let receipt = CoordinatorReceipt {
        v: 1,
        kind: COORDINATOR_RECEIPT_KIND.into(),
        run_id: own.run_id.clone(),
        topology: topology
            .iter()
            .enumerate()
            .map(|(k, assignment)| TopologyEntry {
                seq: k as u32,
                hop_id: hop_id(k as u32),
                role: if k == 0 { "coordinator" } else { "stage" }.into(),
                observation_point: None,
                assignment: Some(assignment.clone()),
            })
            .collect(),
        stages: entries,
        coordinator_node_id: own.coordinator_node_id.clone(),
        request_id: own.request_id.clone(),
        stage_seal_deadline_ms: deadline_ms,
    };
    receipt.validate()?;
    for (block, _) in &exchanges {
        block.validate()?;
    }
    Ok(SplitPlan {
        own_slice: own.clone(),
        exchanges,
        receipt,
        carried,
    })
}

/// The coordinator's stage-exchange block for stage `k`: what it assigned,
/// and only what it saw first-hand.
fn exchange_block(own: &StageBlock, assignment: &StageAssignment, k: u32, term: u64) -> StageBlock {
    let first = k == 1;
    let direct_final = own.return_mode == ReturnMode::Direct && k + 1 == own.stage_count;
    let observed = (first || direct_final).then(|| CoordinatorObserved {
        downstream: first.then(|| own.downstream.clone()).flatten(),
        direct_return: direct_final.then(|| DirectReturn {
            sent: None,
            received: own.direct_return.as_ref().and_then(|d| d.received.clone()),
        }),
    });
    StageBlock {
        v: 1,
        side: Side::Coordinator,
        coordinator_node_id: own.coordinator_node_id.clone(),
        run_id: own.run_id.clone(),
        request_id: own.request_id.clone(),
        topology_hash: own.topology_hash.clone(),
        stage_index: k,
        stage_count: own.stage_count,
        layer_start: assignment.layer_start,
        layer_end: assignment.layer_end,
        package_id: assignment.package_id.clone(),
        manifest_sha256: own.manifest_sha256.clone(),
        source_model_sha256: own.source_model_sha256.clone(),
        return_mode: own.return_mode,
        upstream: None,
        downstream: None,
        direct_return: None,
        tokens: None,
        terminal_state: own.terminal_state.clone(),
        coordinator_term: Some(term),
        data_path_observed: Some(first || direct_final),
        coordinator_observed: observed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::producer::keys::KeyPair;
    use crate::producer::stage::seal_stage_record;
    use serde_json::json;

    fn case(name: &str) -> Value {
        let cases: Value =
            serde_json::from_str(include_str!("../vectors/split-stage/hop-cases.json")).unwrap();
        cases
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap()
            .clone()
    }

    fn stage_zero(case: &Value, exchange_id: &str) -> StageEvent {
        let receipt = CoordinatorReceipt::from_value(&case["receipt"]).unwrap();
        StageEvent {
            block: StageBlock::from_value(&case["own"]).unwrap(),
            exchange_id: Some(exchange_id.into()),
            coordinator_term: Some(7),
            topology: Some(
                receipt
                    .topology
                    .iter()
                    .map(|t| t.assignment.clone().unwrap())
                    .collect(),
            ),
        }
    }

    fn stage_records(case: &Value) -> Vec<Value> {
        case["carried"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let key = KeyPair::generate();
                seal_stage_record(
                    &StageBlock::from_value(&c["block"]).unwrap(),
                    None,
                    None,
                    &key.signing_key,
                )
                .unwrap()
            })
            .collect()
    }

    /// The node `case`'s topology assigns `record`'s stage to: the one peer
    /// allowed to push it.
    fn sender(case: &Value, record: &Value) -> String {
        let k = stage_block_of(record).unwrap().unwrap().stage_index as usize;
        case["receipt"]["topology"][k]["assignment"]["node_id"]
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn a_stage_event_carries_no_coordinator_facts() {
        let c = case("relayed_all_agree");
        let mut event = StageEvent {
            block: StageBlock::from_value(&c["carried"][0]["block"]).unwrap(),
            exchange_id: None,
            coordinator_term: None,
            topology: None,
        };
        assert!(event.validate().is_ok());
        event.exchange_id = Some("x".into());
        assert!(event.validate().is_err());
    }

    #[test]
    fn stage_zero_must_name_its_exchange_and_match_its_own_assignment() {
        let c = case("relayed_all_agree");
        let good = stage_zero(&c, "exch-1");
        assert!(good.validate().is_ok());
        let mut missing = good.clone();
        missing.exchange_id = None;
        assert!(missing.validate().is_err());
        let mut short = good.clone();
        short.topology.as_mut().unwrap().pop();
        assert!(short.validate().is_err());
        let mut wrong = good.clone();
        wrong.topology.as_mut().unwrap()[0].layer_end = 99;
        assert!(wrong.validate().is_err());
        let bytes = serde_json::to_vec(&json!({"block": c["own"], "exchange_id": "e", "coordinator_term": 7,
            "topology": c["receipt"]["topology"].as_array().unwrap().iter().map(|t| t["assignment"].clone()).collect::<Vec<_>>(),
            "session_id": "1"})).unwrap();
        assert!(
            StageEvent::parse(&bytes).is_err(),
            "unknown members are refused"
        );
    }

    #[test]
    fn a_terminal_event_is_held_only_for_a_pending_split() {
        let collector = SplitCollector::<&str>::default();
        assert_eq!(collector.hold(Some("exch-1"), "env"), Some("env"));
        collector
            .on_stage_zero(stage_zero(&case("relayed_all_agree"), "exch-1"), 0)
            .unwrap();
        assert_eq!(collector.hold(None, "env"), Some("env"));
        assert_eq!(collector.hold(Some("exch-2"), "env"), Some("env"));
        assert_eq!(collector.hold(Some("exch-1"), "env"), None);
        assert_eq!(
            collector.hold(Some("exch-1"), "env-again"),
            Some("env-again")
        );
    }

    #[test]
    fn a_split_releases_when_every_stage_arrived_before_the_deadline() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .unwrap();
        collector.hold(Some("exch-1"), "env");
        let records = stage_records(&c);
        assert_eq!(
            collector.on_stage_record(&records[0], &sender(&c, &records[0]), 10),
            Arrival::Held
        );
        assert!(
            collector.take_due(20, 2_000).is_empty(),
            "stage 2 is still outstanding"
        );
        assert_eq!(
            collector.on_stage_record(&records[1], &sender(&c, &records[1]), 30),
            Arrival::Held
        );
        let ready = collector.take_due(40, 2_000);
        assert_eq!(ready.len(), 1);
        let plan = plan_split(&ready[0], 2_000).unwrap();
        assert!(plan.receipt.stages[1..]
            .iter()
            .all(|s| s.bundle == "present"));
        assert_eq!(plan.carried.len(), 2);
        assert_eq!(
            plan.exchanges[0].1.as_deref(),
            records[0]["capsule_id"].as_str()
        );
        assert_eq!(plan.receipt.stage_seal_deadline_ms, 2_000);
        // A record for the sealed split now is late: the follow-up path.
        assert_eq!(
            collector.on_stage_record(&records[1], &sender(&c, &records[1]), 50),
            Arrival::Late
        );
    }

    #[test]
    fn a_missing_stage_is_absent_once_the_deadline_passes_never_dropped() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 1_000)
            .unwrap();
        collector.hold(Some("exch-1"), "env");
        {
            let r = &stage_records(&c)[0];
            collector.on_stage_record(r, &sender(&c, r), 1_100)
        };
        assert!(collector.take_due(2_999, 2_000).is_empty());
        let ready = collector.take_due(3_000, 2_000);
        let plan = plan_split(&ready[0], 2_000).unwrap();
        assert_eq!(plan.receipt.stages[1].bundle, "present");
        assert_eq!(plan.receipt.stages[2].bundle, "absent");
        assert!(plan.receipt.stages[2].bundle_ref.is_none());
        assert_eq!(
            plan.exchanges.len(),
            2,
            "an absent stage still gets its stage-exchange record"
        );
        assert!(plan.exchanges[1].1.is_none());
    }

    #[test]
    fn two_records_under_one_key_are_a_conflict_and_neither_is_cited_as_present() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .unwrap();
        collector.hold(Some("exch-1"), "env");
        let records = stage_records(&c);
        let twin = stage_records(&c);
        collector.on_stage_record(&records[0], &sender(&c, &records[0]), 1);
        collector.on_stage_record(&records[1], &sender(&c, &records[1]), 1);
        collector.on_stage_record(&twin[1], &sender(&c, &twin[1]), 1);
        let plan = plan_split(&collector.take_due(2, 2_000)[0], 2_000).unwrap();
        assert_eq!(plan.receipt.stages[2].bundle, "conflict");
        assert_eq!(
            plan.receipt.stages[2].bundle_refs.as_ref().unwrap().len(),
            2
        );
        assert!(plan.exchanges[1].1.is_none());
        assert_eq!(plan.carried.len(), 3);
    }

    #[test]
    fn records_arriving_before_stage_zero_are_kept_for_it() {
        let c = case("direct_all_agree");
        let collector = SplitCollector::<&str>::default();
        let records = stage_records(&c);
        for r in &records {
            assert_eq!(
                collector.on_stage_record(r, &sender(&c, r), 0),
                Arrival::Held
            );
        }
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 5)
            .unwrap();
        collector.hold(Some("exch-1"), "env");
        let plan = plan_split(&collector.take_due(6, 2_000)[0], 2_000).unwrap();
        assert!(plan.receipt.stages[1..]
            .iter()
            .all(|s| s.bundle == "present"));
        // Final stage under direct return: the coordinator saw its own
        // receive end of the direct lane, and says so.
        let (last, _) = plan.exchanges.last().unwrap();
        assert_eq!(last.data_path_observed, Some(true));
        assert!(last
            .coordinator_observed
            .as_ref()
            .unwrap()
            .direct_return
            .is_some());
    }

    #[test]
    fn a_split_that_never_completes_expires_and_is_counted() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        {
            let r = &stage_records(&c)[0];
            collector.on_stage_record(r, &sender(&c, r), 0)
        };
        assert!(collector.take_due(19_999, 2_000).is_empty());
        assert_eq!(collector.dropped(), 0);
        assert!(collector.take_due(20_000, 2_000).is_empty());
        assert_eq!(collector.dropped(), 1);
    }

    #[test]
    fn a_second_stage_zero_for_one_split_is_refused() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .unwrap();
        assert!(collector
            .on_stage_zero(stage_zero(&c, "exch-2"), 0)
            .is_err());
        assert!(collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .is_err());
    }

    /// A stage record under a guessed split key from a node the coordinator
    /// did not assign that stage never fills `present` nor forces `conflict`:
    /// refused once stage 0 names the topology, and dropped at stage 0 when
    /// it came first. MUTANT: skip the sender check and stage 1 reads conflict.
    #[test]
    fn only_the_node_assigned_a_stage_may_supply_its_record() {
        let c = case("relayed_all_agree");
        let records = stage_records(&c);
        let forged = stage_records(&c);
        // Before stage 0: held, then dropped when stage 0 names stage 1's node.
        let collector = SplitCollector::<&str>::default();
        assert_eq!(
            collector.on_stage_record(&forged[0], "intruder", 0),
            Arrival::Held
        );
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 1)
            .unwrap();
        assert_eq!(collector.refused(), 1);
        // After stage 0: refused at once.
        assert_eq!(
            collector.on_stage_record(&forged[1], "intruder", 2),
            Arrival::Refused
        );
        assert_eq!(collector.refused(), 2);
        collector.hold(Some("exch-1"), "env");
        collector.on_stage_record(&records[0], &sender(&c, &records[0]), 3);
        collector.on_stage_record(&records[1], &sender(&c, &records[1]), 3);
        let plan = plan_split(&collector.take_due(4, 2_000)[0], 2_000).unwrap();
        assert!(plan.receipt.stages[1..]
            .iter()
            .all(|s| s.bundle == "present"));
        assert_eq!(plan.carried.len(), 2, "no forged record is carried");
    }

    #[test]
    fn one_sender_supplies_at_most_two_records_per_stage() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .unwrap();
        let from = sender(&c, &stage_records(&c)[0]);
        assert_eq!(
            collector.on_stage_record(&stage_records(&c)[0], &from, 1),
            Arrival::Held
        );
        assert_eq!(
            collector.on_stage_record(&stage_records(&c)[0], &from, 1),
            Arrival::Held
        );
        assert_eq!(
            collector.on_stage_record(&stage_records(&c)[0], &from, 1),
            Arrival::Refused
        );
    }

    fn junk_record(c: &Value, n: usize) -> Value {
        let mut block = StageBlock::from_value(&c["carried"][0]["block"]).unwrap();
        block.request_id = (900_000 + n).to_string();
        seal_stage_record(&block, None, None, &KeyPair::generate().signing_key).unwrap()
    }

    #[test]
    fn one_sender_opens_a_bounded_number_of_splits_before_stage_zero() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        for n in 0..MAX_PENDING_PER_SENDER {
            assert_eq!(
                collector.on_stage_record(&junk_record(&c, n), "flooder", 0),
                Arrival::Held
            );
        }
        assert_eq!(
            collector.on_stage_record(&junk_record(&c, 999), "flooder", 0),
            Arrival::Refused
        );
        assert_eq!(
            collector.on_stage_record(&junk_record(&c, 999), "someone-else", 0),
            Arrival::Held
        );
    }

    /// Many senders flooding fake split keys never evict a split that holds
    /// its terminal envelope: the real split still seals. MUTANT: evict the
    /// oldest regardless and the real split is gone.
    #[test]
    fn a_split_holding_its_envelope_is_never_evicted_for_room() {
        let c = case("relayed_all_agree");
        let collector = SplitCollector::<&str>::default();
        collector
            .on_stage_zero(stage_zero(&c, "exch-1"), 0)
            .unwrap();
        assert_eq!(collector.hold(Some("exch-1"), "env"), None);
        let mut n = 0;
        for s in 0..(2 * MAX_PENDING / MAX_PENDING_PER_SENDER) {
            for _ in 0..MAX_PENDING_PER_SENDER {
                collector.on_stage_record(&junk_record(&c, n), &format!("flooder-{s}"), 1);
                n += 1;
            }
        }
        let ready = collector.take_due(2_000, 2_000);
        assert_eq!(ready.len(), 1, "the real split seals");
        assert_eq!(ready[0].envelope, "env");
    }

    #[test]
    fn records_that_are_not_stage_side_are_not_collected() {
        let collector = SplitCollector::<&str>::default();
        assert_eq!(
            collector.on_stage_record(&json!({"capsule_id": "a"}), "peer", 0),
            Arrival::NotStage
        );
    }
}
