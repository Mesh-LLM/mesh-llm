//! `node.availability`: node state, capacity map, and its key cap.

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, NodeAvailabilityEventKind, Outcome, ReasonCode, RuntimeFact,
};

use super::super::fixtures::{apply_each_on_fresh_root, summaries};
use crate::runtime_events::reducer::{NODE_CAPACITY_KEY_BOUND, ReducerSnapshot};

fn node(kind: NodeAvailabilityEventKind) -> RuntimeFact {
    RuntimeFact::NodeAvailability(FamilyFact::new(kind))
}

fn node_with(kind: NodeAvailabilityEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::NodeAvailability(FamilyFact::with_data(kind, data))
}

fn lane_capacity(lanes: u64) -> RuntimeFact {
    node_with(
        NodeAvailabilityEventKind::LaneCapacityChanged,
        FactData {
            numeric_summaries: summaries(&[("lane_count", lanes)]),
            ..FactData::default()
        },
    )
}

#[test]
fn a_fresh_reducer_reports_no_node_availability() {
    assert!(
        ReducerSnapshot::empty()
            .domain()
            .node_availability()
            .is_empty()
    );
}

#[test]
fn node_state_is_latest_wins_across_independent_roots() {
    let snapshot = apply_each_on_fresh_root(vec![
        node(NodeAvailabilityEventKind::NodeStarting),
        node(NodeAvailabilityEventKind::NodeAcceptingRequests),
        node(NodeAvailabilityEventKind::NodeDegraded),
    ]);

    assert_eq!(
        snapshot.domain().node_availability().state,
        Some("degraded")
    );
}

#[test]
fn a_synthesized_undelivered_node_stopped_does_not_claim_the_node_stopped() {
    let snapshot = apply_each_on_fresh_root(vec![
        node(NodeAvailabilityEventKind::NodeDraining),
        node_with(
            NodeAvailabilityEventKind::NodeStopped,
            FactData {
                outcome: Some(Outcome::Unknown),
                reason: Some(ReasonCode::TerminalNotDelivered),
                ..FactData::default()
            },
        ),
    ]);

    assert_eq!(
        snapshot.domain().node_availability().state,
        Some("draining")
    );
}

#[test]
fn capacity_is_latest_wins_and_namespaced_by_capacity_kind() {
    let snapshot = apply_each_on_fresh_root(vec![
        lane_capacity(4),
        node_with(
            NodeAvailabilityEventKind::RequestCapacityChanged,
            FactData {
                numeric_summaries: summaries(&[("lane_count", 9)]),
                ..FactData::default()
            },
        ),
        lane_capacity(2),
    ]);

    let capacity = &snapshot.domain().node_availability().capacity;
    assert_eq!(capacity.get("lane_capacity.lane_count"), Some(&2));
    assert_eq!(capacity.get("request_capacity.lane_count"), Some(&9));
}

#[test]
fn capacity_keys_past_the_cap_are_rejected_and_counted() {
    let overflow = 3;
    let facts = (0..NODE_CAPACITY_KEY_BOUND + overflow)
        .map(|index| {
            let key = format!("resource_{index}");
            node_with(
                NodeAvailabilityEventKind::SessionCapacityChanged,
                FactData {
                    numeric_summaries: summaries(&[(key.as_str(), 1)]),
                    ..FactData::default()
                },
            )
        })
        .collect();

    let snapshot = apply_each_on_fresh_root(facts);

    let availability = snapshot.domain().node_availability();
    assert_eq!(availability.capacity.len(), NODE_CAPACITY_KEY_BOUND);
    assert!(
        availability
            .capacity
            .contains_key("session_capacity.resource_0"),
        "established keys are kept, not evicted"
    );
    assert_eq!(availability.capacity_overflow, overflow as u64);
}

#[test]
fn resource_pressure_outcome_reports_local_capacity_availability() {
    let snapshot = apply_each_on_fresh_root(vec![node_with(
        NodeAvailabilityEventKind::ResourcePressureChanged,
        FactData {
            outcome: Some(Outcome::Failure),
            ..FactData::default()
        },
    )]);

    assert_eq!(
        snapshot
            .domain()
            .node_availability()
            .local_capacity_available,
        Some(false)
    );
}

#[test]
fn set_changes_read_model_and_stage_counts_from_their_summaries() {
    let snapshot = apply_each_on_fresh_root(vec![
        node_with(
            NodeAvailabilityEventKind::AvailableModelSetChanged,
            FactData {
                numeric_summaries: summaries(&[("model_count", 3)]),
                ..FactData::default()
            },
        ),
        node_with(
            NodeAvailabilityEventKind::AvailableStageSetChanged,
            FactData {
                numeric_summaries: summaries(&[("stage_count", 2)]),
                ..FactData::default()
            },
        ),
    ]);

    let availability = snapshot.domain().node_availability();
    assert_eq!(availability.available_model_count, Some(3));
    assert_eq!(availability.available_stage_count, Some(2));
}
