//! `node.availability`: latest-wins node availability state plus a bounded
//! capacity map.
//!
//! Producers today: `runtime::node_lifecycle_events` (starting/accepting/
//! draining/stopped), `runtime::local_split::topology_events` (degraded/
//! unavailable, `ResourcePressureChanged` with a success/failure outcome
//! only), `inference::skippy::stage::runtime_events`
//! (`AvailableStageSetChanged`, `RequestCapacityChanged`/
//! `LaneCapacityChanged` carrying a `lane_count` summary), and
//! `runtime::model_lifecycle::events` (`AvailableModelSetChanged`, scope
//! only). No producer attaches a model/stage COUNT, so
//! `available_model_count`/`available_stage_count` read the
//! `model_count`/`stage_count` numeric summaries and stay unset until a
//! producer supplies them.

use std::collections::BTreeMap;

use mesh_llm_runtime_event_contracts::{FactData, NodeAvailabilityEventKind, Outcome};

use super::{is_undelivered_terminal, summary_value, unsigned_summary};

/// Distinct capacity keys retained; a new key past this cap is rejected
/// and counted rather than evicting an established key.
pub const NODE_CAPACITY_KEY_BOUND: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeAvailabilityDomainState {
    pub state: Option<&'static str>,
    pub available_model_count: Option<u64>,
    pub available_stage_count: Option<u64>,
    /// `<capacity kind>.<summary key>` -> latest value, e.g.
    /// `lane_capacity.lane_count`. Namespaced by kind because producers
    /// reuse one summary key across several capacity kinds.
    pub capacity: BTreeMap<String, u64>,
    pub capacity_overflow: u64,
    /// Latest `ResourcePressureChanged` outcome: whether local capacity was
    /// available (`Success`) or not (any other outcome).
    pub local_capacity_available: Option<bool>,
}

impl NodeAvailabilityDomainState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

const fn state_label(kind: NodeAvailabilityEventKind) -> Option<&'static str> {
    match kind {
        NodeAvailabilityEventKind::NodeStarting => Some("starting"),
        NodeAvailabilityEventKind::NodeAcceptingRequests => Some("accepting_requests"),
        NodeAvailabilityEventKind::NodeDegraded => Some("degraded"),
        NodeAvailabilityEventKind::NodeUnavailable => Some("unavailable"),
        NodeAvailabilityEventKind::NodeDraining => Some("draining"),
        NodeAvailabilityEventKind::NodeStopped => Some("stopped"),
        NodeAvailabilityEventKind::AvailableModelSetChanged
        | NodeAvailabilityEventKind::AvailableStageSetChanged
        | NodeAvailabilityEventKind::RequestCapacityChanged
        | NodeAvailabilityEventKind::LaneCapacityChanged
        | NodeAvailabilityEventKind::SessionCapacityChanged
        | NodeAvailabilityEventKind::ResourcePressureChanged => None,
    }
}

const fn capacity_namespace(kind: NodeAvailabilityEventKind) -> Option<&'static str> {
    match kind {
        NodeAvailabilityEventKind::RequestCapacityChanged => Some("request_capacity"),
        NodeAvailabilityEventKind::LaneCapacityChanged => Some("lane_capacity"),
        NodeAvailabilityEventKind::SessionCapacityChanged => Some("session_capacity"),
        NodeAvailabilityEventKind::ResourcePressureChanged => Some("resource_pressure"),
        _ => None,
    }
}

fn record_capacity(state: &mut NodeAvailabilityDomainState, namespace: &str, data: &FactData) {
    for summary in data.numeric_summaries.as_slice() {
        let Some(value) = summary_value(summary) else {
            continue;
        };
        let key = format!("{namespace}.{}", summary.key.as_str());
        if state.capacity.len() >= NODE_CAPACITY_KEY_BOUND && !state.capacity.contains_key(&key) {
            state.capacity_overflow = state.capacity_overflow.saturating_add(1);
            continue;
        }
        state.capacity.insert(key, value);
    }
}

pub(super) fn apply_node_availability(
    state: &mut NodeAvailabilityDomainState,
    kind: NodeAvailabilityEventKind,
    data: &FactData,
) {
    // An engine-synthesized `terminal_not_delivered` `NodeStopped` means the
    // real terminal never arrived; it is not evidence the node stopped.
    if let Some(label) = state_label(kind)
        && !is_undelivered_terminal(data)
    {
        state.state = Some(label);
    }
    if let Some(namespace) = capacity_namespace(kind) {
        record_capacity(state, namespace, data);
    }
    match kind {
        NodeAvailabilityEventKind::AvailableModelSetChanged => {
            if let Some(count) = unsigned_summary(data, "model_count") {
                state.available_model_count = Some(count);
            }
        }
        NodeAvailabilityEventKind::AvailableStageSetChanged => {
            if let Some(count) = unsigned_summary(data, "stage_count") {
                state.available_stage_count = Some(count);
            }
        }
        NodeAvailabilityEventKind::ResourcePressureChanged => {
            if let Some(outcome) = data.outcome {
                state.local_capacity_available = Some(outcome == Outcome::Success);
            }
        }
        _ => {}
    }
}
