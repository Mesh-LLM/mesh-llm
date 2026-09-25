//! `stages` category.

use mesh_llm_runtime_event_contracts::StageTopologyEventKind;

use super::super::fixtures::{input, scope as root, stage_fact};
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
