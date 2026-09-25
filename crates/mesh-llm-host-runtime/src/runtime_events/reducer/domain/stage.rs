//! `stages` category: latest topology state per stage.

use mesh_llm_runtime_event_contracts::{FactData, StageTopologyEventKind};

use super::bounded::touch;
use super::{DomainState, outcome_label};
use crate::runtime_events::config::LIFECYCLE_OPERATION_BOUND;

/// One tracked stage's reduced domain view: a `stages` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageDomainState {
    pub id: String,
    pub index: Option<u32>,
    pub state: Option<String>,
    pub last_outcome: Option<String>,
}

fn stage_identity(data: &FactData) -> Option<(String, u32)> {
    data.scope
        .stage
        .as_ref()
        .map(|stage| (stage.id.as_str().to_string(), stage.index))
}

fn stage_state_label(kind: StageTopologyEventKind) -> &'static str {
    use StageTopologyEventKind::{
        StageConnectionEstablished, StageConnectionLost, StageConnectionRecovered, StageDegraded,
        StageFailed, StageLoading, StageReady, StageStarting, StageStopped, StageStopping,
        StageUnavailable, TopologyAssembling, TopologyDegraded, TopologyReady, TopologyUnavailable,
    };
    match kind {
        StageStarting => "starting",
        StageLoading => "loading",
        StageReady | StageConnectionEstablished | StageConnectionRecovered => "ready",
        StageDegraded => "degraded",
        StageUnavailable => "unavailable",
        StageStopping => "stopping",
        StageStopped => "stopped",
        StageFailed | StageConnectionLost => "failed",
        TopologyAssembling => "assembling",
        TopologyReady => "topology_ready",
        TopologyDegraded => "topology_degraded",
        TopologyUnavailable => "topology_unavailable",
    }
}

pub(super) fn apply_stage_topology(
    state: &mut DomainState,
    kind: StageTopologyEventKind,
    data: &FactData,
) {
    let Some((id, index)) = stage_identity(data) else {
        return;
    };
    touch(
        &mut state.stages_order,
        &mut state.stages,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state
        .stages
        .entry(id.clone())
        .or_insert_with(|| StageDomainState {
            id,
            ..StageDomainState::default()
        });
    entry.index = Some(index);
    entry.state = Some(stage_state_label(kind).to_string());
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
}
