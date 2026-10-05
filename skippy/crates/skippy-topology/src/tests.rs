use super::*;
use serde::Deserialize;

fn nodes(count: u32) -> Vec<NodeSpec> {
    (0..count)
        .map(|index| NodeSpec {
            node_id: format!("node-{index}"),
            cached_slice_bytes: 0,
            vram_bytes: 0,
        })
        .collect()
}

fn compact_identity(value: &str) -> String {
    value.to_ascii_lowercase().replace(['_', '-', '/', ' '], "")
}

fn weighted_node(node_id: &str, vram_bytes: u64) -> NodeSpec {
    NodeSpec {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        vram_bytes,
    }
}

fn placement_signal(node_id: &str) -> NodePlacementSignal {
    NodePlacementSignal {
        node_id: node_id.to_string(),
        cached_slice_bytes: 0,
        missing_artifact_bytes: 0,
        rtt_ms: None,
        artifact_transfer_supported: false,
        availability_score: 0,
    }
}

fn edge(source: &str, target: &str, rtt_ms: u32) -> StageEdgeSignal {
    StageEdgeSignal {
        source_node_id: source.to_string(),
        target_node_id: target.to_string(),
        rtt_ms: Some(rtt_ms),
        large_frame_bytes_per_sec: None,
        direct_prediction_return_supported: true,
    }
}

fn stage_layout(plan: &TopologyPlan) -> Vec<(&str, u32, u32)> {
    plan.stages
        .iter()
        .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
        .collect()
}

fn role_layout(plan: &TopologyPlan) -> Vec<Vec<StageRole>> {
    plan.stages
        .iter()
        .map(|stage| stage.roles.clone())
        .collect()
}

#[test]
fn transport_aware_plan_orders_same_nodes_by_stage_edge_cost() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("node-a", 30),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[],
        &[
            edge("node-a", "node-b", 200),
            edge("node-b", "node-c", 200),
            edge("node-a", "node-c", 5),
            edge("node-c", "node-b", 5),
        ],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("node-a", 0, 3), ("node-c", 3, 6), ("node-b", 6, 9)]
    );
    assert!(plan.diagnostics.iter().any(|diagnostic| diagnostic.code
        == PlanReasonCode::NetworkPipelineCost
        && diagnostic.message.contains("node-a -> node-c")));
}

#[test]
fn transport_aware_plan_orders_two_stages_by_edge_cost() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 30), weighted_node("node-b", 30)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[],
        &[edge("node-a", "node-b", 200), edge("node-b", "node-a", 5)],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("node-b", 0, 3), ("node-a", 3, 6)]
    );
}

#[test]
fn transport_aware_plan_preserves_package_order_when_edges_tie() {
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("cold-a", 30),
            weighted_node("warm", 30),
            weighted_node("cold-b", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_transport(
        &request,
        &[warm],
        &[
            edge("warm", "cold-a", 10),
            edge("warm", "cold-b", 10),
            edge("cold-a", "warm", 10),
            edge("cold-a", "cold-b", 10),
            edge("cold-b", "warm", 10),
            edge("cold-b", "cold-a", 10),
        ],
    )
    .expect("plan");

    assert_eq!(
        stage_layout(&plan),
        vec![("warm", 0, 3), ("cold-a", 3, 6), ("cold-b", 6, 9)]
    );
}

#[test]
fn dense_attention_plan_allows_costed_kv_migration() {
    let request = TopologyPlanRequest {
        topology_id: "dense".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.stages.len(), 3);
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::AttentionKv)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::CostedKv)
    );
    assert!(plan.diagnostics.is_empty());
}

#[test]
fn split_topology_labels_driver_embedding_intermediate_and_readout_roles() {
    let request = TopologyPlanRequest {
        topology_id: "roles".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(9, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        role_layout(&plan),
        vec![
            vec![StageRole::Driver, StageRole::Embedding],
            vec![StageRole::Intermediate],
            vec![StageRole::Readout],
        ]
    );
}

#[test]
fn single_stage_topology_labels_combined_driver_embedding_and_readout() {
    let request = TopologyPlanRequest {
        topology_id: "single-roles".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(2, 10),
        nodes: nodes(1),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        role_layout(&plan),
        vec![vec![
            StageRole::Driver,
            StageRole::Embedding,
            StageRole::Readout,
        ]]
    );
}

#[test]
fn weighted_contiguous_plan_uses_node_vram_for_layer_spans() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 6), ("node-b", 6, 9), ("node-c", 9, 12)]
    );
}

#[test]
fn weighted_contiguous_plan_falls_back_to_even_without_weights() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 0), weighted_node("node-b", 0)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![("node-a", 0, 3), ("node-b", 3, 6)]
    );
}

#[test]
fn package_aware_plan_matches_weighted_without_package_signals() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(12, 10),
        nodes: vec![
            weighted_node("node-a", 60),
            weighted_node("node-b", 30),
            weighted_node("node-c", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let weighted_plan = plan_weighted_contiguous(&request).expect("weighted plan");
    let package_plan =
        plan_package_aware_contiguous_with_signals(&request, &[]).expect("package-aware plan");

    assert_eq!(stage_layout(&package_plan), stage_layout(&weighted_plan));
}

#[test]
fn package_aware_plan_prefers_cached_peer_for_equal_capacity() {
    let mut cold = placement_signal("cold");
    cold.missing_artifact_bytes = 32;
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    warm.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("cold", 40), weighted_node("warm", 40)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, warm]).expect("plan");

    assert_eq!(stage_layout(&plan), vec![("warm", 0, 4), ("cold", 4, 8)]);
    assert!(
        plan.stages[0]
            .reason_codes
            .contains(&PlanReasonCode::CacheLocalityPreferred)
    );
    assert!(
        plan.stages[1]
            .reason_codes
            .contains(&PlanReasonCode::ArtifactTransferPenalty)
    );
}

#[test]
fn package_aware_plan_reports_cold_start_artifact_totals() {
    let mut transfer_ready = placement_signal("transfer-ready");
    transfer_ready.missing_artifact_bytes = 32;
    transfer_ready.artifact_transfer_supported = true;
    let mut remote_fallback = placement_signal("remote-fallback");
    remote_fallback.missing_artifact_bytes = 16;
    remote_fallback.artifact_transfer_supported = false;
    let mut warm = placement_signal("warm");
    warm.cached_slice_bytes = 64;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(9, 10),
        nodes: vec![
            weighted_node("transfer-ready", 30),
            weighted_node("remote-fallback", 30),
            weighted_node("warm", 30),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(
        &request,
        &[transfer_ready, remote_fallback, warm],
    )
    .expect("plan");

    let diagnostic = plan
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == PlanReasonCode::ArtifactTransferPenalty)
        .expect("artifact diagnostic");
    assert!(diagnostic.message.contains("cached=30 bytes"));
    assert!(diagnostic.message.contains("missing=46 bytes"));
    assert!(
        diagnostic
            .message
            .contains("peer-transfer-eligible=30 bytes")
    );
    assert!(
        diagnostic
            .message
            .contains("remote-download-fallback=16 bytes")
    );
}

#[test]
fn weighted_plan_without_artifact_signals_stays_diagnostic_quiet() {
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(6, 10),
        nodes: vec![weighted_node("node-a", 30), weighted_node("node-b", 30)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_weighted_contiguous(&request).expect("plan");

    assert!(plan.diagnostics.is_empty());
}

#[test]
fn package_aware_plan_penalizes_missing_untransferable_artifacts() {
    let mut cold = placement_signal("cold-high-vram");
    cold.missing_artifact_bytes = 64;
    cold.artifact_transfer_supported = false;
    let mut ready = placement_signal("ready-lower-vram");
    ready.cached_slice_bytes = 16;
    ready.artifact_transfer_supported = true;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![
            weighted_node("cold-high-vram", 100),
            weighted_node("ready-lower-vram", 80),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cold, ready]).expect("plan");

    assert_eq!(plan.stages[0].node_id, "ready-lower-vram");
    assert!(
        plan.stages[1]
            .reason_codes
            .contains(&PlanReasonCode::ArtifactTransferPenalty)
    );
}

#[test]
fn package_aware_plan_treats_high_rtt_as_cost_not_exclusion() {
    let mut distant = placement_signal("distant");
    distant.rtt_ms = Some(250);
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(8, 10),
        nodes: vec![weighted_node("distant", 100), weighted_node("nearby", 80)],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[distant]).expect("plan");

    assert!(plan.stages.iter().any(|stage| stage.node_id == "distant"));
    let distant_stage = plan
        .stages
        .iter()
        .find(|stage| stage.node_id == "distant")
        .expect("distant stage");
    assert!(
        distant_stage
            .reason_codes
            .contains(&PlanReasonCode::NetworkPipelineCost)
    );
}

#[test]
fn package_aware_plan_can_promote_extra_cached_peer() {
    let mut cached_extra = placement_signal("cached-extra");
    cached_extra.cached_slice_bytes = 100;
    let request = TopologyPlanRequest {
        topology_id: "topology-a".into(),
        model_id: "model-a".into(),
        layers: dense_attention_layers(2, 10),
        nodes: vec![
            weighted_node("cold-a", 40),
            weighted_node("cold-b", 40),
            weighted_node("cached-extra", 40),
        ],
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_package_aware_contiguous_with_signals(&request, &[cached_extra]).expect("plan");

    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.stages[0].node_id, "cached-extra");
    assert!(
        plan.stages
            .iter()
            .any(|stage| stage.node_id == "cold-a" || stage.node_id == "cold-b")
    );
}

#[test]
fn falcon_h1_marks_every_stage_as_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(6, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 4), (4, 6)]
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::Mixed)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner)
    );
    assert_eq!(plan.diagnostics.len(), 3);
}

#[test]
fn qwen3next_mixed_layers_only_make_recurrent_ranges_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3next".to_string(),
        model_id: "qwen3next".to_string(),
        layers: qwen3next_layers(8, [2, 3, 6], 10),
        nodes: nodes(4),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.state_affinity)
            .collect::<Vec<_>>(),
        vec![
            StateAffinity::AttentionKv,
            StateAffinity::Recurrent,
            StateAffinity::AttentionKv,
            StateAffinity::Mixed
        ]
    );
    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| stage.migration_policy)
            .collect::<Vec<_>>(),
        vec![
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner,
            MigrationPolicy::CostedKv,
            MigrationPolicy::StickyRecurrentOwner
        ]
    );
    assert_eq!(plan.diagnostics.len(), 2);
}

#[test]
fn explicit_recurrent_transfer_policy_is_loud() {
    let request = TopologyPlanRequest {
        topology_id: "transfer".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: falcon_h1_layers(2, 10),
        nodes: nodes(1),
        family: None,
        policy: PlannerPolicy {
            allow_recurrent_state_transfer: true,
        },
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.stages[0].migration_policy,
        MigrationPolicy::RecurrentStateTransferAllowed
    );
    assert_eq!(plan.diagnostics[0].severity, DiagnosticSeverity::Warning);
}

#[test]
fn qwen3_family_uses_f32_wire_payloads() {
    let request = TopologyPlanRequest {
        topology_id: "qwen3-wire".to_string(),
        model_id: "qwen3".to_string(),
        layers: dense_attention_layers(28, 10),
        nodes: nodes(2),
        family: Some(qwen3_dense_capability(28, 1024)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("qwen3_dense"));
    assert_eq!(plan.boundaries.len(), 1);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 4096);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 4096);
}

#[test]
fn accepted_dense_families_emit_exact_state_mobility_reason() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3".to_string(),
        model_id: "gemma3".to_string(),
        layers: dense_attention_layers(26, 10),
        nodes: nodes(2),
        family: Some(gemma3_capability(26, 1152)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("gemma3"));
    assert!(plan.stages.iter().all(|stage| {
        stage
            .reason_codes
            .contains(&PlanReasonCode::ExactStateMobilityAccepted)
    }));
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityAccepted)
    );
}

#[test]
fn dense_family_uses_f32_without_split_constraints() {
    let request = TopologyPlanRequest {
        topology_id: "olmo".to_string(),
        model_id: "olmo".to_string(),
        layers: dense_attention_layers(32, 10),
        nodes: nodes(2),
        family: Some(olmo_capability(32, 4096)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.family_id.as_deref(), Some("olmo"));
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 16384);
}

#[test]
fn dense_family_f32_wire_bytes_are_recorded() {
    let families = [
        (gemma2_capability(26, 2304), 9216),
        (gemma3_capability(26, 1152), 4608),
        (glm4_capability(40, 4096), 16384),
    ];

    for (family, expected_f32_wire_bytes) in families {
        let request = TopologyPlanRequest {
            topology_id: family.family_id.clone(),
            model_id: family.family_id.clone(),
            layers: dense_attention_layers(family.layer_count, 10),
            nodes: nodes(2),
            family: Some(family),
            policy: PlannerPolicy::default(),
        };

        let plan = plan_even_contiguous(&request).expect("plan");
        assert_eq!(
            plan.boundaries[0].wire_payload_bytes_per_token,
            expected_f32_wire_bytes
        );
    }
}

#[test]
fn falcon_family_capability_marks_attention_layers_sticky() {
    let request = TopologyPlanRequest {
        topology_id: "falcon-family".to_string(),
        model_id: "falcon-h1".to_string(),
        layers: dense_attention_layers(24, 10),
        nodes: nodes(2),
        family: Some(falcon_h1_capability(24, 2048)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.state_affinity == StateAffinity::Mixed)
    );
    assert!(
        plan.stages
            .iter()
            .all(|stage| stage.migration_policy == MigrationPolicy::StickyRecurrentOwner)
    );
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::ExactStateMobilityRejected)
    );
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::RecurrentOwnerSticky)
    );
}

#[test]
fn gemma4_e4b_accepts_validated_boundary_with_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-valid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(2),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 21);
    assert_eq!(plan.boundaries[0].decision, BoundaryDecision::Accepted);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 10240);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 10240);
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::TokenSidebandRequired)
    );
}

#[test]
fn rwkv7_boundary_accounts_for_v_first_sideband() {
    let request = TopologyPlanRequest {
        topology_id: "rwkv7-sideband".to_string(),
        model_id: "rwkv7-191m".to_string(),
        layers: falcon_h1_layers(12, 4),
        nodes: nodes(3),
        family: Some(rwkv7_capability(12, 768)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(plan.boundaries[0].layer_boundary, 4);
    assert_eq!(plan.boundaries[0].raw_activation_bytes_per_token, 6144);
    assert_eq!(plan.boundaries[0].wire_payload_bytes_per_token, 6144);
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::ActivationSidebandRequired)
    );
    assert!(
        plan.boundaries[0]
            .reason_codes
            .contains(&PlanReasonCode::RecurrentOwnerSticky)
    );
}

#[test]
fn gemma4_e4b_rejects_shared_kv_consumer_boundaries() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-invalid".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_even_contiguous(&request).expect("plan");

    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (14, BoundaryDecision::Rejected),
            (28, BoundaryDecision::Rejected)
        ]
    );
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == PlanReasonCode::SharedKvRegionCut)
    );
}

#[test]
fn balanced_planner_avoids_rejected_gemma4_boundaries() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-balanced".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let two_stage = plan_balanced_accepted_contiguous(&request, 2).expect("two-stage plan");
    assert_eq!(two_stage.boundaries[0].layer_boundary, 21);
    assert_eq!(two_stage.boundaries[0].decision, BoundaryDecision::Accepted);

    let three_stage = plan_balanced_accepted_contiguous(&request, 3).expect("three-stage plan");
    assert_eq!(
        three_stage
            .boundaries
            .iter()
            .map(|boundary| boundary.layer_boundary)
            .collect::<Vec<_>>(),
        vec![13, 21]
    );
    assert!(
        three_stage
            .boundaries
            .iter()
            .all(|boundary| boundary.decision == BoundaryDecision::Accepted)
    );
}

#[test]
fn balanced_planner_uses_generic_boundaries_without_family_policy() {
    let request = TopologyPlanRequest {
        topology_id: "generic-balanced".to_string(),
        model_id: "unknown-family".to_string(),
        layers: dense_attention_layers(24, 10),
        nodes: nodes(3),
        family: None,
        policy: PlannerPolicy::default(),
    };

    let two_stage = plan_balanced_accepted_contiguous(&request, 2).expect("two-stage plan");
    assert_eq!(two_stage.boundaries[0].layer_boundary, 12);

    let three_stage = plan_balanced_accepted_contiguous(&request, 3).expect("three-stage plan");
    assert_eq!(
        three_stage
            .boundaries
            .iter()
            .map(|boundary| boundary.layer_boundary)
            .collect::<Vec<_>>(),
        vec![8, 16]
    );
}

#[test]
fn gemma3n_requires_altup_sideband_and_constrained_kv_boundary() {
    let request = TopologyPlanRequest {
        topology_id: "gemma3n".to_string(),
        model_id: "gemma3n".to_string(),
        layers: dense_attention_layers(30, 10),
        nodes: nodes(3),
        family: Some(gemma3n_capability(30, 2048)),
        policy: PlannerPolicy::default(),
    };

    let even_plan = plan_even_contiguous(&request).expect("even plan");
    assert_eq!(
        even_plan
            .boundaries
            .iter()
            .map(|boundary| {
                (
                    boundary.layer_boundary,
                    boundary.decision,
                    boundary.raw_activation_bytes_per_token,
                    boundary.wire_payload_bytes_per_token,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted, 32768, 32768),
            (20, BoundaryDecision::Rejected, 32768, 32768)
        ]
    );
    assert!(even_plan.boundaries.iter().all(|boundary| {
        boundary
            .reason_codes
            .contains(&PlanReasonCode::ActivationSidebandRequired)
    }));

    let accepted_plan = plan_contiguous_with_splits(&request, &[10, 18]).expect("accepted plan");
    assert_eq!(
        accepted_plan
            .boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (10, BoundaryDecision::Accepted),
            (18, BoundaryDecision::Accepted)
        ]
    );
}

#[test]
fn explicit_splits_return_reasoned_boundary_decisions() {
    let request = TopologyPlanRequest {
        topology_id: "gemma-explicit".to_string(),
        model_id: "gemma4-e4b".to_string(),
        layers: dense_attention_layers(42, 10),
        nodes: nodes(3),
        family: Some(gemma4_e4b_capability(42, 2560)),
        policy: PlannerPolicy::default(),
    };

    let plan = plan_contiguous_with_splits(&request, &[12, 24]).expect("plan");

    assert_eq!(
        plan.stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 12), (12, 24), (24, 42)]
    );
    assert_eq!(
        plan.boundaries
            .iter()
            .map(|boundary| (boundary.layer_boundary, boundary.decision))
            .collect::<Vec<_>>(),
        vec![
            (12, BoundaryDecision::Rejected),
            (24, BoundaryDecision::Rejected)
        ]
    );
}

#[test]
fn inferred_capabilities_do_not_override_model_metadata() {
    let llama = infer_family_capability(
        "/Volumes/External/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        16,
        8192,
    )
    .expect("known llama family");
    assert_eq!(llama.family_id, "llama");
    assert_eq!(llama.activation_width, 8192);
    assert_eq!(llama.exact_state_mobility, ExactStateMobility::Untested);

    let deepseek3 = infer_family_capability("unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL", 61, 7168)
        .expect("known deepseek family");
    assert_eq!(deepseek3.family_id, "deepseek3");
    assert_eq!(deepseek3.exact_state_mobility, ExactStateMobility::Untested);

    assert!(
        infer_family_capability("mradermacher/Maincoder-1B-GGUF:Q2_K", 24, 2048).is_none(),
        "an unrecognized name must not inherit a removed registry row"
    );

    let gemma4 = infer_family_capability("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M", 42, 2560)
        .expect("known Gemma4 E4B exception");
    assert_eq!(gemma4.family_id, "gemma4_e4b");
    assert!(!gemma4.split_constraints.is_empty());
    assert!(!gemma4.sidebands.is_empty());
}

#[test]
fn stage_runtime_test_catalog_has_unique_architectures() {
    for (index, expected) in TEST_LLAMA_ARCHITECTURE_CATALOG.iter().enumerate() {
        assert!(
            TEST_LLAMA_ARCHITECTURE_CATALOG[index + 1..]
                .iter()
                .all(|other| other.llama_architecture != expected.llama_architecture),
            "duplicate architecture in test-only coverage catalog: {}",
            expected.llama_architecture
        );
        assert!(!expected.family_id.is_empty());
    }
}

#[derive(Debug, Deserialize)]
struct ParityCandidateManifest {
    candidates: Vec<ParityCandidate>,
}

#[derive(Debug, Deserialize)]
struct ParityCandidate {
    llama_model: String,
    status: String,
}

// The parity roster is a workspace certification fixture, outside this
// publishable crate. Read it when these workspace tests run rather than
// embedding an external source file in the crate's compilation inputs.
fn parity_candidate_manifest() -> ParityCandidateManifest {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/llama-parity-candidates.json");
    let contents = std::fs::read_to_string(path).expect("read workspace parity candidate manifest");
    serde_json::from_str(&contents).expect("parity candidate manifest must parse")
}

#[test]
fn parity_candidate_manifest_covers_stage_runtime_architectures() {
    let manifest = parity_candidate_manifest();
    let candidates: Vec<String> = manifest
        .candidates
        .into_iter()
        .map(|candidate| compact_identity(&candidate.llama_model))
        .collect();

    for expected in TEST_LLAMA_ARCHITECTURE_CATALOG {
        assert!(
            candidates.contains(&compact_identity(expected.llama_architecture)),
            "missing parity candidate row for {}",
            expected.llama_architecture
        );
    }
}

#[test]
fn parity_candidate_manifest_uses_known_statuses() {
    let manifest = parity_candidate_manifest();

    for candidate in manifest.candidates {
        assert!(
            matches!(
                candidate.status.as_str(),
                "candidate"
                    | "candidate_stateful"
                    | "candidate_multimodal"
                    | "certified"
                    | "certified_package_only"
                    | "implementation_base"
                    | "needs_candidate"
                    | "needs_boundary_registration"
                    | "needs_runtime_slice_support"
                    | "no_public_gguf_candidate"
                    | "non_causal_aux"
                    | "package_or_remote_only"
            ),
            "{} has unknown status {}",
            candidate.llama_model,
            candidate.status
        );
    }
}

#[test]
fn qwen35_series_inference_covers_qwen36_release_names() {
    // Qwen3.6 and Qwen3.8 load as llama.cpp `qwen35`/`qwen35moe`; there is no
    // `qwen36` or `qwen38` arch.
    // Every quant and uploader must resolve to the recurrent series, not qwen3moe.
    for identity in [
        "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        "bartowski/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        "Qwen/Qwen3.6-35B-A3B-Instruct-GGUF:Q5_K_M",
        "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
        "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-Q1_0",
        "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-IQ2_XXS",
        "meshllm/Qwen3.8-2.4T-A95B-UD-Q1_0-layers",
        "qwen35moe",
        "qwen36moe",
        "qwen38moe",
    ] {
        let family = infer_family_capability(identity, 40, 2048)
            .unwrap_or_else(|| panic!("expected qwen35moe capability for {identity}"));
        assert_eq!(family.family_id, "qwen35moe", "wrong family for {identity}");
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 40 }],
            "qwen35moe must expose a recurrent range for {identity}"
        );
        assert_eq!(
            family.exact_state_mobility,
            ExactStateMobility::RejectedTooLarge,
            "qwen35moe full-state handoff must stay rejected for {identity}"
        );
    }

    for identity in [
        "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
        "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        "qwen35",
        "qwen36",
        "qwen38",
    ] {
        let family = infer_family_capability(identity, 32, 2560)
            .unwrap_or_else(|| panic!("expected qwen35 capability for {identity}"));
        assert_eq!(family.family_id, "qwen35", "wrong family for {identity}");
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 32 }],
            "qwen35 must expose a recurrent range for {identity}"
        );
    }
}

#[test]
fn qwen4exp_flash_next_has_its_own_fail_closed_hybrid_policy() {
    for identity in [
        "qwen4exp",
        "qwen4_exp",
        "Qwen/Qwen3.8-Flash-Next",
        "unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S",
    ] {
        let family = infer_family_capability(identity, 48, 2560)
            .unwrap_or_else(|| panic!("expected qwen4exp capability for {identity}"));
        assert_eq!(family.family_id, "qwen4exp", "wrong family for {identity}");
        assert_eq!(family.activation_width, 2560);
        assert_eq!(
            family.recurrent_ranges,
            vec![LayerRange { start: 0, end: 48 }],
            "QWEN4EXP state ownership stays sticky until per-layer certification"
        );
        assert_eq!(
            family.exact_state_mobility,
            ExactStateMobility::RejectedTooLarge,
            "QWEN4EXP must not advertise recurrent/indexer state mobility"
        );
        assert_eq!(family.sidebands.len(), 1);
        assert_eq!(family.sidebands[0].kind, SidebandKind::TokenIds);
        assert_eq!(family.sidebands[0].first_required_layer, 1);
    }

    let legacy = infer_family_capability("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", 32, 2560)
        .expect("legacy Qwen3.8 capability");
    assert_eq!(legacy.family_id, "qwen35");
}

#[test]
fn unknown_qwen3_point_releases_resolve_to_no_family() {
    // A Qwen3 point release we have no evidence for must not be guessed into
    // the non-recurrent `qwen3moe`/`qwen3_dense` families. Qwen3.5, 3.6 and
    // 3.8 are all hybrid, so a wrong guess advertises a non-recurrent policy
    // for a probably-recurrent model. No capability is the safe answer: it
    // surfaces the gap at onboarding instead of at runtime.
    for identity in [
        "Qwen/Qwen3.9-40B-A3B-GGUF:Q4_K_M",
        "unsloth/Qwen3.7-27B-GGUF:Q4_K_M",
        "qwen39",
        "qwen3.7",
        // A dotted release whose version is not a single digit must not be
        // narrowed to its first digit: `Qwen3.50` is not `Qwen3.5`.
        "Qwen/Qwen3.50-40B-A3B-GGUF:Q4_K_M",
        "Qwen/Qwen3.58-40B-GGUF:Q4_K_M",
        "qwen3.50",
    ] {
        assert!(
            infer_family_capability(identity, 40, 2048).is_none(),
            "{identity} must not resolve to a guessed family"
        );
    }
}

#[test]
fn qwen3_parameter_sizes_are_not_mistaken_for_qwen35_series() {
    // `Qwen3-5B` compacts to `qwen35b`: the digit is a parameter count, not a
    // series number, so these must stay on the non-recurrent Qwen3 families.
    for (identity, expected) in [
        ("Qwen/Qwen3-5B-GGUF:Q4_K_M", "qwen3_dense"),
        ("Qwen/Qwen3-6B-GGUF:Q4_K_M", "qwen3_dense"),
        ("Qwen/Qwen3-8B-GGUF:Q4_K_M", "qwen3_dense"),
        // Fractional sizes compact to `qwen30.6b`: the digit after the dot is
        // a size, not a point release.
        ("Qwen/Qwen3-0.6B-GGUF:Q8_0", "qwen3_dense"),
        // Multi-digit runs are parameter counts, not point releases.
        ("Qwen/Qwen3-235B-A22B-GGUF:Q4_K_M", "qwen3moe"),
        ("Qwen/Qwen3-0.6B:Q8_0", "qwen3_dense"),
        ("Qwen/Qwen3-35B-A3B-GGUF:Q4_K_M", "qwen3moe"),
        ("Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M", "qwen3moe"),
    ] {
        let family = infer_family_capability(identity, 40, 2048)
            .unwrap_or_else(|| panic!("expected capability for {identity}"));
        assert_eq!(family.family_id, expected, "wrong family for {identity}");
        assert!(
            family.recurrent_ranges.is_empty(),
            "{identity} must not be treated as recurrent"
        );
    }
}
