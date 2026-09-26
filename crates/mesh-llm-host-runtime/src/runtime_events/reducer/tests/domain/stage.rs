//! `stages` category.

use mesh_llm_runtime_event_contracts::{
    FamilyFact, RuntimeFact, StageTopologyEventKind, TopologyId,
};

use super::super::fixtures::{apply_all, input, scope as root, stage_fact};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

#[test]
fn stages_track_latest_topology_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            stage_fact(StageTopologyEventKind::StageReady, "stage-0", 0),
        ),
    ) else {
        panic!("stage_ready must apply");
    };
    let stage = snapshot
        .domain()
        .stages()
        .into_iter()
        .find(|stage| stage.id == "stage-0")
        .expect("stage must be tracked in state.stages");
    assert_eq!(stage.state.as_deref(), Some("ready"));
}

fn topology_stage_fact(topology_id: &str, stage_id: &str) -> RuntimeFact {
    let RuntimeFact::StageTopology(fact) =
        stage_fact(StageTopologyEventKind::StageReady, stage_id, 0)
    else {
        unreachable!("stage_fact builds a StageTopology fact");
    };
    let mut data = fact.data().clone();
    data.scope.topology_id = Some(TopologyId::new(topology_id).expect("valid topology id"));
    RuntimeFact::StageTopology(FamilyFact::with_data(*fact.kind(), data))
}

#[test]
fn same_stage_id_in_two_topologies_keeps_separate_rows() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        root(),
        0,
        vec![
            topology_stage_fact("topo-a", "stage-0"),
            topology_stage_fact("topo-b", "stage-0"),
        ],
    );
    let mut topologies: Vec<_> = snapshot
        .domain()
        .stages()
        .into_iter()
        .filter(|stage| stage.id == "stage-0")
        .map(|stage| stage.topology_id.clone())
        .collect();
    topologies.sort();
    assert_eq!(
        topologies,
        vec![Some("topo-a".to_owned()), Some("topo-b".to_owned())]
    );
}
