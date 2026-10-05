//! `stages` category: latest topology state per stage.

use mesh_llm_runtime_event_contracts::{FactData, StageTopologyEventKind};

use super::bounded::touch;
use super::{DomainState, outcome_label};
use crate::runtime_events::config::LIFECYCLE_OPERATION_BOUND;

/// One tracked stage's reduced domain view: a `stages` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageDomainState {
    pub topology_id: Option<String>,
    pub id: String,
    pub index: Option<u32>,
    pub state: Option<String>,
    pub last_outcome: Option<String>,
}

struct StageIdentityKey {
    topology_id: Option<String>,
    id: String,
    index: u32,
}

impl StageIdentityKey {
    /// Stage ids such as `stage-0` repeat across topologies, so the row key
    /// includes the topology when the fact carries one.
    fn row_key(&self) -> String {
        match &self.topology_id {
            Some(topology) => format!("{topology}/{}", self.id),
            None => self.id.clone(),
        }
    }
}

fn stage_identity(data: &FactData) -> Option<StageIdentityKey> {
    data.scope.stage.as_ref().map(|stage| StageIdentityKey {
        topology_id: data
            .scope
            .topology_id
            .as_ref()
            .map(|topology| topology.as_str().to_string()),
        id: stage.id.as_str().to_string(),
        index: stage.index,
    })
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
    let Some(identity) = stage_identity(data) else {
        return;
    };
    let key = identity.row_key();
    touch(
        &mut state.stages_order,
        &mut state.stages,
        &key,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state.stages.entry(key).or_insert_with(|| StageDomainState {
        topology_id: identity.topology_id,
        id: identity.id,
        ..StageDomainState::default()
    });
    entry.index = Some(identity.index);
    entry.state = Some(stage_state_label(kind).to_string());
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
}
