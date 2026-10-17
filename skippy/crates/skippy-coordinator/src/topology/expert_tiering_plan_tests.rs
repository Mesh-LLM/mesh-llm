//! Planner-level behaviour of routed-expert tiering: when spills happen, how
//! they compete with extra stages, and what the plan reports.

use super::tests::{GIB, input, latency_node, node};
use super::*;

/// Forty 1 GiB layers whose weight is three quarters routed experts.
fn moe_input(
    nodes: Vec<TopologyNode>,
    expert_offload: ExpertOffloadPolicy,
) -> TopologyPlanningInput {
    TopologyPlanningInput {
        layer_weight_bytes: vec![GIB; 40],
        layer_expert_weight_bytes: vec![3 * GIB / 4; 40],
        expert_count: 128,
        expert_used_count: 8,
        expert_offload,
        ..input(nodes)
    }
}

fn tiered_node(
    id: &str,
    gib: u64,
    host_expert_gib: u64,
    stream_bytes_per_second: Option<u64>,
) -> TopologyNode {
    TopologyNode {
        host_expert_bytes: Some(host_expert_gib * GIB),
        expert_stream_bytes_per_second: stream_bytes_per_second,
        ..node(id, gib)
    }
}

#[test]
fn expert_tiering_lets_one_node_hold_a_model_its_accelerator_cannot() {
    // 24 GiB of accelerator memory cannot hold 40 GiB of weights, but the
    // dense trunk is only 10 GiB and 32 GiB of host memory takes the rest.
    let plan = plan_topology(&moe_input(
        vec![tiered_node("solo", 24, 32, None)],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();

    assert_eq!(plan.stages.len(), 1);
    let placement = plan.stages[0]
        .expert_placement
        .clone()
        .expect("tiered placement");
    assert!(placement.host_layer_end > 0 && placement.host_layer_end <= 40);
    assert_eq!(placement.streamed_bytes, 0);
    assert_eq!(placement.estimated_stream_ms_per_token, 0);
    assert_eq!(plan.estimated_decode_expert_stream_ms_per_token, None);
}

#[test]
fn expert_tiering_stays_off_unless_the_policy_allows_it() {
    assert_eq!(
        plan_topology(&moe_input(
            vec![tiered_node("solo", 24, 32, None)],
            ExpertOffloadPolicy::Disabled,
        )),
        Err(TopologyPlanError::NoValidTopology {
            minimum_context: 65_536
        })
    );
}

#[test]
fn resident_placement_wins_whenever_it_fits() {
    let plan = plan_topology(&moe_input(
        vec![tiered_node("roomy", 80, 32, Some(GIB))],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();

    assert_eq!(plan.stages.len(), 1);
    assert_eq!(plan.stages[0].expert_placement, None);
}

#[test]
fn tiering_without_host_memory_needs_a_measured_stream_rate() {
    assert_eq!(
        plan_topology(&moe_input(
            vec![tiered_node("solo", 24, 0, None)],
            ExpertOffloadPolicy::Auto,
        )),
        Err(TopologyPlanError::NoValidTopology {
            minimum_context: 65_536
        })
    );

    let plan = plan_topology(&moe_input(
        vec![tiered_node("solo", 24, 0, Some(8 * GIB))],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();
    let placement = plan.stages[0]
        .expert_placement
        .clone()
        .expect("streamed placement");
    assert!(placement.streamed_bytes > 0);
    assert!(placement.estimated_stream_ms_per_token > 0);
    assert_eq!(
        plan.estimated_decode_expert_stream_ms_per_token,
        Some(placement.estimated_stream_ms_per_token)
    );
}

#[test]
fn a_resident_split_beats_tiering_even_against_a_slow_hop() {
    // Host-resident experts run on CPU kernels the estimate does not
    // price, so a resident two-stage split wins even when the second
    // node is 60 ms away and paging would be nearly free.
    let plan = plan_topology(&moe_input(
        vec![
            tiered_node("pager", 24, 0, Some(1024 * GIB)),
            latency_node("peer", 24, 60),
        ],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();

    assert_eq!(plan.stages.len(), 2);
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.expert_placement.is_none())
    );
}

#[test]
fn slow_expert_paging_loses_to_a_tiered_second_stage() {
    // Neither node can hold half the model resident, so tiering is
    // required. Paging every expert from a 1 GiB/s store on one node
    // costs far more than one extra 10 ms hop between two tiered stages
    // whose host memory holds the spilled experts.
    let plan = plan_topology(&moe_input(
        vec![
            tiered_node("a", 16, 16, Some(GIB)),
            TopologyNode {
                stage_transfer_latency_ms: Some(10),
                ..tiered_node("b", 16, 16, Some(GIB))
            },
        ],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();

    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.estimated_decode_expert_stream_ms_per_token, None);
}

#[test]
fn cheap_expert_paging_beats_a_tiered_second_stage() {
    // With a fast store, one node that pages its spilled experts costs
    // about 2 ms per token, less than the 60 ms hop to a second node.
    let plan = plan_topology(&moe_input(
        vec![
            tiered_node("pager", 16, 0, Some(1024 * GIB)),
            TopologyNode {
                stage_transfer_latency_ms: Some(60),
                ..tiered_node("peer", 16, 16, None)
            },
        ],
        ExpertOffloadPolicy::Auto,
    ))
    .unwrap();

    assert_eq!(plan.stages.len(), 1);
    assert_eq!(plan.stages[0].node_id, "pager");
    assert!(plan.estimated_decode_expert_stream_ms_per_token.is_some());
}

#[test]
fn tiered_plans_are_not_rebalanced() {
    let input = TopologyPlanningInput {
        nodes: vec![
            TopologyNode {
                decode_bytes_per_second: Some(100_000_000_000),
                ..tiered_node("a", 24, 32, None)
            },
            TopologyNode {
                decode_bytes_per_second: Some(100_000_000_000),
                ..tiered_node("b", 24, 32, None)
            },
        ],
        minimum_nodes: 2,
        auto_balance: true,
        ..moe_input(Vec::new(), ExpertOffloadPolicy::Auto)
    };
    let plan = plan_topology(&input).unwrap();
    assert_eq!(plan.stages.len(), 2);
    if plan
        .stages
        .iter()
        .any(|stage| stage.expert_placement.is_some())
    {
        assert_eq!(rebalance_topology(&input, &plan), None);
    }
}

#[test]
fn host_expert_bytes_for_stage_prices_the_spilled_prefix() {
    let experts = vec![10, 20, 30, 40];
    assert_eq!(host_expert_bytes_for_stage(&experts, 4, 1, 3), 50);
    assert_eq!(host_expert_bytes_for_stage(&experts, 4, 1, 1), 0);
    assert_eq!(host_expert_bytes_for_stage(&experts, 4, 2, 9), 70);
    assert_eq!(host_expert_bytes_for_stage(&experts, 5, 0, 4), 0);
}
