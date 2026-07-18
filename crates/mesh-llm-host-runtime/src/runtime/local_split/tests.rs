use super::coordinator::*;
use super::loading::*;
use super::test_support::*;
use super::*;
use crate::inference::election;
use crate::mesh::NodeRole;
use crate::plugin;
use crate::runtime::local::*;
use crate::runtime::local_package::*;
use crate::runtime::split_planning::{
    RuntimeSliceStagePlan, format_aggregate_split_capacity_error,
};
use crate::runtime::survey;
use skippy_protocol::{FlashAttentionType, LoadMode};
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

#[test]
fn runtime_local_targets_keep_duplicate_same_model_ports() {
    let (target_tx, _target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

    add_runtime_local_target(&target_tx, "Qwen", 41001);
    add_runtime_local_target(&target_tx, "Qwen", 41002);
    add_runtime_local_target(&target_tx, "Qwen", 41002);

    let targets = target_tx.borrow().candidates("Qwen");
    assert_eq!(
        targets,
        vec![
            election::InferenceTarget::Local(41002),
            election::InferenceTarget::Local(41001),
        ]
    );
}

#[test]
fn split_topology_planner_uses_all_eligible_participants() {
    let participants = vec![
        SplitParticipant::new(make_id(1), 16_000_000_000, None),
        SplitParticipant::new(make_id(2), 24_000_000_000, None),
        SplitParticipant::new(make_id(3), 32_000_000_000, None),
        SplitParticipant::new(make_id(4), 48_000_000_000, None),
    ];

    let stages = plan_runtime_slice_topology(
        "topology-test",
        "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
        &package(40),
        &participants,
    )
    .expect("topology plan");

    assert_eq!(stages.len(), 4);
    assert_eq!(stages[0].stage_index, 0);
    assert_eq!(stages[3].stage_index, 3);
    assert_eq!(
        stages
            .iter()
            .map(|stage| stage.stage_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(stages.first().unwrap().layer_start, 0);
    assert_eq!(stages.last().unwrap().layer_end, 40);
}

#[test]
fn split_topology_planner_prefers_cached_participant_in_runtime_path() {
    let cold = SplitParticipant::new(make_id(1), 24_000_000_000, None).with_package_signals(
        SplitParticipantPackageSignal {
            cached_slice_bytes: 0,
            missing_artifact_bytes: 40_000_000,
            availability_score: 0,
        },
        Some(80),
        true,
    );
    let warm = SplitParticipant::new(make_id(2), 24_000_000_000, None).with_package_signals(
        SplitParticipantPackageSignal {
            cached_slice_bytes: 40_000_000,
            missing_artifact_bytes: 0,
            availability_score: 40,
        },
        Some(5),
        true,
    );

    let stages = plan_runtime_slice_topology(
        "topology-test",
        "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
        &package(40),
        &[cold, warm],
    )
    .expect("package-aware topology plan");

    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].node_id, make_id(2));
    assert_eq!((stages[0].layer_start, stages[0].layer_end), (0, 20));
}

#[test]
fn split_inventory_package_signal_counts_cached_and_missing_ranges() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: 10,
        ready_ranges: vec![skippy::LayerRange {
            layer_start: 4,
            layer_end: 6,
        }],
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 4,
        }],
        missing_ranges: vec![skippy::LayerRange {
            layer_start: 6,
            layer_end: 10,
        }],
        preparing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: None,
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    };

    let signal = split_inventory_package_signal(&inventory, &package);

    assert_eq!(
        signal,
        SplitParticipantPackageSignal {
            cached_slice_bytes: 600,
            missing_artifact_bytes: 400,
            availability_score: 6,
        }
    );
    assert!(signal.can_stage_with(&package, true));
    assert!(!signal.can_stage_with(&package, false));
}

#[test]
fn split_package_signal_allows_hf_fallback_when_peer_transfer_is_disabled() {
    let mut package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    package.package_ref = "hf://meshllm/demo-layer-package@abc123".to_string();
    let signal = SplitParticipantPackageSignal {
        cached_slice_bytes: 200,
        missing_artifact_bytes: 800,
        availability_score: 2,
    };

    assert!(signal.can_stage_with(&package, false));
}

#[test]
fn split_participant_timeout_error_reports_blocker_summary() {
    let participants = vec![SplitParticipant::new(make_id(1), 2_000_000_000, None)];
    let excluded = vec![
        SplitParticipantExclusion {
            node_id: make_id(2),
            reason: SplitParticipantExclusionReason::MissingModelSource,
        },
        SplitParticipantExclusion {
            node_id: make_id(3),
            reason: SplitParticipantExclusionReason::MissingModelSource,
        },
        SplitParticipantExclusion {
            node_id: make_id(4),
            reason: SplitParticipantExclusionReason::MissingModelInterest,
        },
    ];

    let error = ensure_split_participant_timeout_has_quorum(
        "meshllm/Qwen3-layers",
        &participants,
        &excluded,
    )
    .expect_err("one participant should not satisfy split quorum")
    .to_string();

    assert!(error.contains("found 1 eligible"));
    assert!(error.contains("blockers [missing_model_source=2 nodes=["));
    assert!(error.contains("missing_model_interest=1 nodes=["));
    assert!(error.contains("next_step: Start the peer with a resolvable package source"));
}

#[test]
fn split_peer_preflight_requires_current_stage_protocol_generation() {
    let mut peer = split_test_peer(0x61, "Qwen3-Coder", false);
    peer.rtt_ms = Some(crate::mesh::MAX_SPLIT_RTT_MS);

    assert_eq!(
        split_peer_preflight_exclusion_reason(&peer, "Qwen3-Coder", "meshllm/Qwen3-Coder-layers"),
        Some(SplitParticipantExclusionReason::StageProtocolGeneration)
    );

    peer.stage_protocol_generation_supported = true;
    assert_eq!(
        split_peer_preflight_exclusion_reason(&peer, "Qwen3-Coder", "meshllm/Qwen3-Coder-layers"),
        None
    );
}

#[test]
fn split_peer_preflight_requires_measured_stage_path() {
    assert_eq!(
        split_peer_stage_path_exclusion_reason(mesh::SplitStagePathSnapshot::unknown()),
        Some(SplitParticipantExclusionReason::MissingStagePath)
    );
}

#[test]
fn split_peer_preflight_rejects_slow_stage_path() {
    assert_eq!(
        split_peer_stage_path_exclusion_reason(mesh::SplitStagePathSnapshot::direct(Some(
            crate::mesh::MAX_SPLIT_RTT_MS + 1,
        ))),
        Some(SplitParticipantExclusionReason::StagePathTooSlow)
    );
}

#[test]
fn split_peer_preflight_rejects_relay_only_stage_path() {
    assert_eq!(
        split_peer_stage_path_exclusion_reason(mesh::SplitStagePathSnapshot::relay(Some(
            crate::mesh::MAX_SPLIT_RTT_MS,
        ))),
        Some(SplitParticipantExclusionReason::StagePathRelayOnly)
    );
}

#[test]
fn split_peer_preflight_rejects_direct_stage_path_without_rtt() {
    assert_eq!(
        split_peer_stage_path_exclusion_reason(mesh::SplitStagePathSnapshot::direct(None)),
        Some(SplitParticipantExclusionReason::MissingStagePath)
    );
}

#[test]
fn split_peer_preflight_allows_fast_stage_path() {
    assert_eq!(
        split_peer_stage_path_exclusion_reason(mesh::SplitStagePathSnapshot::direct(Some(
            crate::mesh::MAX_SPLIT_RTT_MS,
        ))),
        None
    );
}

#[test]
fn split_peer_preflight_keeps_host_eligibility_separate_from_stage_path() {
    let mut peer = split_test_peer(0x66, "Qwen3-Coder", true);
    peer.rtt_ms = Some(crate::mesh::MAX_SPLIT_RTT_MS + 1);

    assert_eq!(
        split_peer_preflight_exclusion_reason(&peer, "Qwen3-Coder", "meshllm/Qwen3-Coder-layers"),
        None
    );
}

#[test]
fn split_peer_host_eligibility_classifies_client_by_role_before_capacity() {
    let mut peer = split_test_peer(0x62, "Qwen3-Coder", true);
    peer.role = NodeRole::Client;
    peer.vram_bytes = 24_000_000_000;

    assert_eq!(
        split_peer_stage_host_exclusion_reason(&peer),
        Some(SplitParticipantExclusionReason::Client)
    );
}

#[test]
fn split_peer_host_eligibility_classifies_non_client_zero_vram_as_capacity() {
    let mut peer = split_test_peer(0x63, "Qwen3-Coder", true);
    peer.role = NodeRole::Worker;
    peer.vram_bytes = 0;

    assert_eq!(
        split_peer_stage_host_exclusion_reason(&peer),
        Some(SplitParticipantExclusionReason::MissingVram)
    );
}

#[test]
fn split_package_signal_still_requires_transfer_for_missing_local_package() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let signal = SplitParticipantPackageSignal {
        cached_slice_bytes: 200,
        missing_artifact_bytes: 800,
        availability_score: 2,
    };

    assert!(!signal.can_stage_with(&package, false));
    assert!(signal.can_stage_with(&package, true));
}

#[test]
fn layer_package_stage_source_waits_for_exact_prepare_availability() {
    let load = stage_load_request(LoadMode::LayerPackage);
    let mut inventory = skippy::StageLayerInventory {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        layer_count: 36,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 36,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: Some(
            "/cache/models--meshllm--Qwen3-8B-Q4_K_M-layers/snapshots/main".to_string(),
        ),
        source_model_bytes: Some(4_900_000_000),
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    };

    assert!(!split_stage_source_is_ready(&inventory, &load));

    inventory
        .preparing_ranges
        .push(test_preparation_status_from_load(&load));

    assert!(split_stage_source_is_ready(&inventory, &load));
}

#[test]
fn runtime_slice_stage_source_accepts_inventory_availability() {
    let load = stage_load_request(LoadMode::RuntimeSlice);
    let inventory = skippy::StageLayerInventory {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        layer_count: 36,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 36,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: Some("/models/qwen.gguf".to_string()),
        source_model_bytes: Some(4_900_000_000),
        source_model_kind: skippy::SourceModelKind::PlainGguf,
    };

    assert!(split_stage_source_is_ready(&inventory, &load));
}

#[test]
fn split_inventory_package_signal_treats_unknown_inventory_as_missing_package() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: 0,
        ready_ranges: Vec::new(),
        available_ranges: Vec::new(),
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: None,
        source_model_kind: skippy::SourceModelKind::Unknown,
    };

    let signal = split_inventory_package_signal(&inventory, &package);

    assert_eq!(
        signal,
        SplitParticipantPackageSignal {
            cached_slice_bytes: 0,
            missing_artifact_bytes: 1_000,
            availability_score: 0,
        }
    );
}

#[test]
fn split_inventory_package_signal_result_classifies_empty_inventory() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: 0,
        ready_ranges: Vec::new(),
        available_ranges: Vec::new(),
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: None,
        source_model_bytes: None,
        source_model_kind: skippy::SourceModelKind::Unknown,
    };

    assert_eq!(
        split_inventory_package_signal_result(&inventory, &package, true),
        Err(SplitParticipantExclusionReason::StageInventoryEmpty)
    );
}

#[test]
fn split_inventory_package_signal_result_classifies_manifest_mismatch() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let mut inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: 10,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 10,
        }],
        missing_ranges: Vec::new(),
        preparing_ranges: Vec::new(),
        source_model_path: Some("/cache/layer-package".to_string()),
        source_model_bytes: Some(1_000),
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    };
    inventory.manifest_sha256 = "other-manifest".to_string();

    assert_eq!(
        split_inventory_package_signal_result(&inventory, &package, true),
        Err(SplitParticipantExclusionReason::PackageManifestMismatch)
    );
}

#[test]
fn split_inventory_package_signal_result_requires_transfer_for_partial_package() {
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };
    let inventory = skippy::StageLayerInventory {
        model_id: "model-a".to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        layer_count: 10,
        ready_ranges: Vec::new(),
        available_ranges: vec![skippy::LayerRange {
            layer_start: 0,
            layer_end: 4,
        }],
        missing_ranges: vec![skippy::LayerRange {
            layer_start: 4,
            layer_end: 10,
        }],
        preparing_ranges: Vec::new(),
        source_model_path: Some("/cache/layer-package".to_string()),
        source_model_bytes: Some(1_000),
        source_model_kind: skippy::SourceModelKind::LayerPackage,
    };

    assert_eq!(
        split_inventory_package_signal_result(&inventory, &package, false),
        Err(SplitParticipantExclusionReason::ArtifactTransferUnavailable)
    );
    assert!(split_inventory_package_signal_result(&inventory, &package, true).is_ok());
}

#[test]
fn split_startup_error_messages_include_specific_blocker_tokens() {
    let control = stage_control_unreachable_message("stage-1", make_id(2));
    let failed = stage_source_prepare_failed_message("stage-1", "package missing");
    let timeout = stage_source_prepare_timeout_message("stage-1", Duration::from_secs(30));

    assert!(control.contains("stage_control_unreachable"));
    assert!(control.contains(&make_id(2).fmt_short().to_string()));
    assert!(failed.contains("stage_source_prepare_failed"));
    assert!(failed.contains("package missing"));
    assert!(timeout.contains("stage_source_prepare_timeout"));
    assert!(timeout.contains("30s"));
}

#[test]
fn startup_runtime_plan_auto_splits_when_model_exceeds_local_capacity() {
    assert_eq!(
        startup_runtime_plan(false, 3_000_000_000, 4_800_000_000),
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity
        }
    );
}

#[test]
fn runtime_model_planning_bytes_uses_layer_package_source_model_bytes() {
    let dir = tempfile::tempdir().unwrap();
    write_test_layer_package(dir.path(), 4_800_000_000);

    let model_bytes = runtime_model_planning_bytes(dir.path()).unwrap();

    assert_eq!(model_bytes, 4_800_000_000);
    assert_eq!(
        startup_runtime_plan(false, 3_000_000_000, model_bytes),
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity
        }
    );
}

#[test]
fn startup_runtime_plan_keeps_local_when_model_fits_without_split_flag() {
    assert_eq!(
        startup_runtime_plan(false, 6_000_000_000, 4_800_000_000),
        StartupRuntimePlan::Local
    );
}

#[test]
fn startup_runtime_plan_respects_explicit_split_for_fitting_model() {
    assert_eq!(
        startup_runtime_plan(true, 6_000_000_000, 4_800_000_000),
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::Forced
        }
    );
}

#[test]
fn split_topology_planner_accepts_constrained_nodes_with_enough_aggregate_capacity() {
    let participants = vec![
        SplitParticipant::new(make_id(1), 3_000_000_000, None),
        SplitParticipant::new(make_id(2), 3_000_000_000, None),
    ];
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 4_800_000_000,
        layer_count: 48,
        ..package(48)
    };

    let stages = plan_runtime_slice_topology(
        "topology-test",
        "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        &package,
        &participants,
    )
    .expect("constrained nodes should form a split topology");

    assert_eq!(stages.len(), 2);
    assert_eq!(
        stages
            .iter()
            .map(|stage| (stage.layer_start, stage.layer_end))
            .collect::<Vec<_>>(),
        vec![(0, 24), (24, 48)]
    );
}

#[test]
fn split_topology_planner_rejects_insufficient_aggregate_capacity() {
    let participants = vec![
        SplitParticipant::new(make_id(1), 2_000_000_000, None),
        SplitParticipant::new(make_id(2), 2_000_000_000, None),
    ];
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 4_800_000_000,
        layer_count: 48,
        ..package(48)
    };

    let error = plan_runtime_slice_topology(
        "topology-test",
        "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        &package,
        &participants,
    )
    .expect_err("aggregate split capacity should be enforced")
    .to_string();

    assert!(error.contains("aggregate split capacity"));
    // Validation uses raw model weight (4.8GB) without the old 10%
    // headroom that was removed to avoid double-counting the topology
    // planner's own VRAM budget.
    assert!(error.contains("requires 4.8GB"));
    assert!(error.contains("has 4.0GB"));
    assert!(error.contains("short by 0.8GB"));
    assert!(error.contains("participants ["));
    assert!(error.contains(&format!("{}:2.0GB", make_id(1).fmt_short())));
    assert!(error.contains(&format!("{}:2.0GB", make_id(2).fmt_short())));
}

#[test]
fn split_topology_planner_rejects_stage_that_exceeds_participant_capacity() {
    // Node 2 has 150 bytes but the planner assigns it at least 2 layers
    // (200 bytes), which exceeds its capacity.  The previous version of
    // this test used 200 bytes for node 2 which passes now that the old
    // 10% headroom is no longer applied on top of the planner budget.
    let participants = vec![
        SplitParticipant::new(make_id(1), 900, None),
        SplitParticipant::new(make_id(2), 150, None),
    ];
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 1_000,
        layer_count: 10,
        ..package(10)
    };

    let error = plan_runtime_slice_topology(
        "topology-test",
        "tiny-capacity-test",
        &package,
        &participants,
    )
    .expect_err("per-stage split capacity should be enforced")
    .to_string();

    assert!(error.contains("stage-1"));
    assert!(error.contains("exceeds node capacity"));
}

#[test]
fn aggregate_split_capacity_error_reports_excluded_peers() {
    let participants = vec![SplitParticipant::new(make_id(1), 2_000_000_000, None)];
    let excluded = vec![
        SplitParticipantExclusion {
            node_id: make_id(2),
            reason: SplitParticipantExclusionReason::MissingModelInterest,
        },
        SplitParticipantExclusion {
            node_id: make_id(3),
            reason: SplitParticipantExclusionReason::MissingModelSource,
        },
    ];

    let error = format_aggregate_split_capacity_error(
        "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        5_280_000_000,
        2_000_000_000,
        &participants,
        &excluded,
    );

    assert!(error.contains("short by 3.3GB"));
    assert!(error.contains("excluded ["));
    assert!(error.contains(&format!(
        "{}:missing_model_interest",
        make_id(2).fmt_short()
    )));
    assert!(error.contains(&format!("{}:missing_model_source", make_id(3).fmt_short())));
}

#[test]
fn split_topology_planner_reports_exclusions_on_capacity_failure() {
    let participants = vec![
        SplitParticipant::new(make_id(1), 2_000_000_000, None),
        SplitParticipant::new(make_id(2), 2_000_000_000, None),
    ];
    let excluded = vec![SplitParticipantExclusion {
        node_id: make_id(3),
        reason: SplitParticipantExclusionReason::MissingModelInterest,
    }];
    let package = skippy::SkippyPackageIdentity {
        source_model_bytes: 4_800_000_000,
        layer_count: 48,
        ..package(48)
    };

    let error = plan_runtime_slice_topology_with_exclusions(
        "topology-test",
        "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        &package,
        &participants,
        &excluded,
    )
    .expect_err("aggregate split capacity should be enforced")
    .to_string();

    // Raw model weight (4.8GB) minus aggregate VRAM (4.0GB) = 0.8GB
    // shortfall, without the old 10% headroom.
    assert!(error.contains("short by 0.8GB"));
    assert!(error.contains("excluded ["));
    assert!(error.contains(&format!(
        "{}:missing_model_interest",
        make_id(3).fmt_short()
    )));
}

#[test]
fn stage_load_model_path_uses_local_path_outside_layer_packages() {
    let model_path = PathBuf::from("/models/runtime-slice.gguf");

    let layer_package = stage_load_model_path(
        LoadMode::LayerPackage,
        "hf://meshllm/demo-package",
        &model_path,
    );
    assert_eq!(layer_package, "hf://meshllm/demo-package");

    for mode in [LoadMode::RuntimeSlice, LoadMode::ArtifactSlice] {
        let path = stage_load_model_path(mode, "hf://meshllm/demo-package", &model_path);
        assert_eq!(path, "/models/runtime-slice.gguf");
    }
}

#[test]
fn skippy_stage_activation_width_rejects_i32_overflow() {
    let error = skippy_stage_activation_width(i32::MAX as u32 + 1, "overflow-model")
        .unwrap_err()
        .to_string();

    assert!(error.contains("exceeds skippy stage ABI limit"));
    assert!(error.contains("overflow-model"));
}

#[test]
fn split_participant_signature_includes_vram_for_stability() {
    let node_id = make_id(9);
    let first = vec![SplitParticipant::new(node_id, 16_000_000_000, None)];
    let second = vec![SplitParticipant::new(node_id, 24_000_000_000, None)];

    assert_ne!(
        split_participant_signature(&first),
        split_participant_signature(&second)
    );
}

#[test]
fn split_participant_signature_includes_package_signals_for_stability() {
    let node_id = make_id(9);
    let first = vec![SplitParticipant::new(node_id, 24_000_000_000, None)];
    let second = vec![
        SplitParticipant::new(node_id, 24_000_000_000, None).with_package_signals(
            SplitParticipantPackageSignal {
                cached_slice_bytes: 12_000_000,
                missing_artifact_bytes: 0,
                availability_score: 12,
            },
            Some(20),
            true,
        ),
    ];

    assert_ne!(
        split_participant_signature(&first),
        split_participant_signature(&second)
    );
}

#[test]
fn split_missing_active_stage_nodes_ignores_unused_lost_nodes() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );
    let connected_node_ids = vec![make_id(1), make_id(3)];

    assert_eq!(
        split_missing_active_stage_nodes(&active, &connected_node_ids),
        vec![make_id(2)]
    );
}

#[test]
fn split_unavailable_active_stage_nodes_includes_failed_stage_without_missing_peer() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );
    let statuses = vec![runtime_status_for_stage(
        &active,
        &active.stages[1],
        skippy::StageRuntimeState::Failed,
    )];

    assert_eq!(
        split_unavailable_active_stage_nodes(
            &active,
            &[make_id(1), make_id(2), make_id(3)],
            &statuses,
        ),
        vec![make_id(2)]
    );
}

#[test]
fn split_unavailable_active_stage_nodes_includes_stopping_stage_without_missing_peer() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );
    let statuses = vec![runtime_status_for_stage(
        &active,
        &active.stages[1],
        skippy::StageRuntimeState::Stopping,
    )];

    assert_eq!(
        split_unavailable_active_stage_nodes(
            &active,
            &[make_id(1), make_id(2), make_id(3)],
            &statuses,
        ),
        vec![make_id(2)]
    );
}

#[test]
fn split_active_stage_nodes_pending_eligibility_retains_connected_stage() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );

    assert_eq!(
        split_active_stage_nodes_pending_eligibility(
            &active,
            &[make_id(1), make_id(2)],
            &[participant(1)],
            &[],
        ),
        vec![make_id(2)]
    );
}

#[test]
fn split_recovery_candidate_participants_excludes_unavailable_stage_nodes() {
    let participants = vec![participant(1), participant(2), participant(3)];

    assert_eq!(
        split_recovery_candidate_participants(&participants, &[make_id(2)]),
        vec![participant(1), participant(3)]
    );
}

#[tokio::test]
async fn load_split_runtime_generation_stops_candidate_stages_after_partial_load_failure() {
    let node = mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
        .await
        .unwrap();
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<skippy::StageControlCommand>();
    node.set_stage_control_sender(control_tx).await;

    let requests = Arc::new(StdMutex::new(Vec::new()));
    let preparations = Arc::new(StdMutex::new(Vec::<skippy::StagePreparationStatus>::new()));
    let captured_requests = Arc::clone(&requests);
    let captured_preparations = Arc::clone(&preparations);
    tokio::spawn(async move {
        while let Some(command) = control_rx.recv().await {
            captured_requests
                .lock()
                .unwrap()
                .push(command.request.clone());
            let response = match &command.request {
                skippy::StageControlRequest::Prepare(prepare) => {
                    let status = test_preparation_status_from_load(&prepare.load);
                    captured_preparations.lock().unwrap().push(status.clone());
                    Ok(skippy::StageControlResponse::PrepareAccepted(
                        skippy::StagePrepareAcceptedResponse {
                            accepted: true,
                            status,
                            error: None,
                        },
                    ))
                }
                skippy::StageControlRequest::Inventory(inventory) => {
                    let mut response = test_inventory_from_request(inventory);
                    response.preparing_ranges = captured_preparations
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|status| {
                            status.model_id == inventory.model_id
                                && status.package_ref == inventory.package_ref
                                && status.manifest_sha256 == inventory.manifest_sha256
                        })
                        .cloned()
                        .collect();
                    Ok(skippy::StageControlResponse::Inventory(response))
                }
                skippy::StageControlRequest::Claim(claim) => Ok(
                    skippy::StageControlResponse::ClaimAccepted(skippy::StageCoordinatorClaimAck {
                        accepted: true,
                        claim: claim.clone(),
                        error: None,
                    }),
                ),
                skippy::StageControlRequest::Load(load) if load.stage_id == "stage-1" => {
                    Err(anyhow::anyhow!("injected stage load failure"))
                }
                skippy::StageControlRequest::Load(load) => Ok(skippy::StageControlResponse::Ready(
                    skippy::StageReadyResponse {
                        accepted: true,
                        status: test_stage_status_from_load(load, skippy::StageRuntimeState::Ready),
                        error: None,
                    },
                )),
                skippy::StageControlRequest::Stop(stop) => Ok(skippy::StageControlResponse::Ready(
                    skippy::StageReadyResponse {
                        accepted: true,
                        status: test_stage_status_from_stop(stop),
                        error: None,
                    },
                )),
                other => panic!("unexpected stage control request: {other:?}"),
            };
            let _ = command.resp.send(response);
        }
    });

    let mut package = package(40);
    package.package_ref = "hf://Mesh-LLM/test-split-package".to_string();
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("qwen.gguf");
    write_fake_gguf_model(&model_path);
    let local_id = node.id();
    let generation = SplitTopologyGeneration::new(
        "candidate-topology".into(),
        "candidate-run".into(),
        2,
        vec![SplitParticipant::new(local_id, 24_000_000_000, None)],
        vec![
            local_stage(local_id, 0, 0, 12),
            local_stage(local_id, 1, 12, 24),
            local_stage(local_id, 2, 24, 40),
        ],
    );
    let mesh_config = plugin::MeshConfig::default();

    let error = match Box::pin(load_split_runtime_generation(SplitGenerationLoadSpec {
        node: &node,
        mesh_config: &mesh_config,
        model_ref: "Qwen",
        model_path: &model_path,
        package: &package,
        generation: &generation,
        projector_path: None,
        ctx_size: 4096,
        pinned_gpu: None,
        slots: 1,
        cache_type_k_override: None,
        cache_type_v_override: None,
        n_batch_override: None,
        n_ubatch_override: None,
        flash_attention_override: FlashAttentionType::Auto,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            openai_frontend::GuardrailMode::Disabled,
        ),
        skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
        survey_telemetry: survey::SurveyTelemetry::disabled(),
    }))
    .await
    {
        Ok(_) => panic!("candidate split generation load unexpectedly succeeded"),
        Err(error) => error,
    };

    let error_chain = format!("{error:#}");
    assert!(
        error_chain.contains("injected stage load failure"),
        "unexpected error: {error_chain}"
    );

    let requests = requests.lock().unwrap();
    let claim_count = requests
        .iter()
        .filter(|request| matches!(request, skippy::StageControlRequest::Claim(_)))
        .count();
    assert_eq!(claim_count, generation.stages.len());
    let load_stage_ids = requests
        .iter()
        .filter_map(|request| match request {
            skippy::StageControlRequest::Load(load) => Some(load.stage_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(load_stage_ids, vec!["stage-2", "stage-1"]);

    let stop_requests = requests
        .iter()
        .filter_map(|request| match request {
            skippy::StageControlRequest::Stop(stop) => Some(stop),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(stop_requests.len(), 2);
    assert_eq!(stop_requests[0].stage_id, "stage-1");
    assert_eq!(stop_requests[1].stage_id, "stage-2");
    assert!(stop_requests.iter().all(|stop| {
        stop.topology_id == generation.topology_id
            && stop.run_id == generation.run_id
            && stop.shutdown_generation == generation.generation
    }));
}

#[test]
fn split_replan_decision_accepts_more_stage_capacity() {
    let participants = vec![SplitParticipant::new(make_id(1), 16_000_000_000, None)];
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        participants.clone(),
        vec![RuntimeSliceStagePlan {
            stage_id: "stage-0".into(),
            stage_index: 0,
            node_id: make_id(1),
            layer_start: 0,
            layer_end: 40,
            parameter_bytes: 40_000_000,
        }],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        participants,
        vec![
            RuntimeSliceStagePlan {
                stage_id: "stage-0".into(),
                stage_index: 0,
                node_id: make_id(1),
                layer_start: 0,
                layer_end: 16,
                parameter_bytes: 16_000_000,
            },
            RuntimeSliceStagePlan {
                stage_id: "stage-1".into(),
                stage_index: 1,
                node_id: make_id(2),
                layer_start: 16,
                layer_end: 40,
                parameter_bytes: 24_000_000,
            },
        ],
    );

    assert_eq!(
        split_replan_decision(&active, &candidate),
        SplitReplanDecision::Candidate
    );
    assert_eq!(
        split_replan_decision_with_reason(&active, &candidate),
        (SplitReplanDecision::Candidate, "candidate_has_more_stages")
    );
}

#[test]
fn split_replan_decision_keeps_equivalent_topology() {
    let stages = vec![RuntimeSliceStagePlan {
        stage_id: "stage-0".into(),
        stage_index: 0,
        node_id: make_id(1),
        layer_start: 0,
        layer_end: 40,
        parameter_bytes: 40_000_000,
    }];
    let participants = vec![SplitParticipant::new(make_id(1), 16_000_000_000, None)];
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        participants.clone(),
        stages.clone(),
    );
    let candidate =
        SplitTopologyGeneration::new("topology-b".into(), "run-b".into(), 2, participants, stages);

    assert_eq!(
        split_replan_decision(&active, &candidate),
        SplitReplanDecision::Keep
    );
    assert_eq!(
        split_replan_decision_with_reason(&active, &candidate),
        (SplitReplanDecision::Keep, "candidate_not_materially_better")
    );
}

#[test]
fn split_replan_decision_accepts_degraded_topology_when_active_stage_peer_is_lost() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(3)],
        vec![stage(1, 0, 0, 15), stage(3, 1, 15, 30)],
    );

    assert_eq!(
        split_replan_decision(&active, &candidate),
        SplitReplanDecision::Candidate
    );
}

#[test]
fn split_replan_decision_keeps_topology_when_only_unused_participant_is_lost() {
    let active_stages = vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)];
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        active_stages.clone(),
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(2)],
        active_stages,
    );

    assert_eq!(
        split_replan_decision(&active, &candidate),
        SplitReplanDecision::Keep
    );
}

#[test]
fn split_loss_recovery_uses_replacement_split_when_active_stage_peer_is_lost() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(3)],
        vec![stage(1, 0, 0, 15), stage(3, 1, 15, 30)],
    );

    assert_eq!(
        split_loss_recovery_decision(
            &active,
            &[make_id(1), make_id(3)],
            &[],
            Some(&candidate),
            true,
        ),
        SplitLossRecoveryDecision::ReplacementSplit
    );
}

#[test]
fn split_loss_recovery_uses_replacement_split_when_active_stage_has_failed() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(3)],
        vec![stage(1, 0, 0, 15), stage(3, 1, 15, 30)],
    );
    assert_eq!(
        split_loss_recovery_decision(
            &active,
            &[make_id(1), make_id(2), make_id(3)],
            &[make_id(2)],
            Some(&candidate),
            true,
        ),
        SplitLossRecoveryDecision::ReplacementSplit
    );
}

#[test]
fn split_loss_recovery_rejects_replacement_that_reuses_failed_stage_peer() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 10), stage(2, 1, 10, 20), stage(3, 2, 20, 30)],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(2), participant(3)],
        vec![stage(1, 0, 0, 15), stage(2, 1, 15, 30)],
    );

    assert_eq!(
        split_loss_recovery_decision(
            &active,
            &[make_id(1), make_id(2), make_id(3)],
            &[make_id(2)],
            Some(&candidate),
            true,
        ),
        SplitLossRecoveryDecision::LocalFallback
    );
    assert!(split_candidate_is_valid_replacement_split(&candidate));
    assert!(!split_candidate_is_valid_replacement_split_after_loss(
        &candidate,
        &[make_id(2)]
    ));
}

#[test]
fn split_loss_recovery_falls_back_to_local_when_replacement_split_is_unavailable() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );

    assert_eq!(
        split_loss_recovery_decision(&active, &[make_id(1)], &[], None, true),
        SplitLossRecoveryDecision::LocalFallback
    );
}

#[test]
fn split_loss_recovery_withdraws_when_split_and_local_paths_are_unavailable() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );

    assert_eq!(
        split_loss_recovery_decision(&active, &[make_id(1)], &[], None, false),
        SplitLossRecoveryDecision::Withdraw
    );
}

#[test]
fn split_withdraw_grace_defers_a_fresh_loss() {
    let first_seen = Instant::now();
    assert_eq!(
        split_withdraw_grace_action(Some(first_seen), first_seen, Duration::from_secs(75)),
        SplitWithdrawGraceAction::Defer
    );
    assert_eq!(
        split_withdraw_grace_action(
            Some(first_seen),
            first_seen + Duration::from_secs(30),
            Duration::from_secs(75),
        ),
        SplitWithdrawGraceAction::Defer
    );
}

#[test]
fn split_withdraw_grace_withdraws_after_grace_elapses() {
    let first_seen = Instant::now();
    assert_eq!(
        split_withdraw_grace_action(
            Some(first_seen),
            first_seen + Duration::from_secs(75),
            Duration::from_secs(75),
        ),
        SplitWithdrawGraceAction::Withdraw
    );
    assert_eq!(
        split_withdraw_grace_action(
            Some(first_seen),
            first_seen + Duration::from_secs(120),
            Duration::from_secs(75),
        ),
        SplitWithdrawGraceAction::Withdraw
    );
}

#[test]
fn split_withdraw_grace_defers_when_no_loss_recorded() {
    let now = Instant::now();
    assert_eq!(
        split_withdraw_grace_action(None, now, Duration::from_secs(75)),
        SplitWithdrawGraceAction::Defer
    );
}

#[test]
fn split_loss_recovery_rejects_single_participant_candidate_as_split_topology() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1)],
        vec![stage(1, 0, 0, 40)],
    );

    assert_eq!(
        split_loss_recovery_decision(&active, &[make_id(1)], &[], Some(&candidate), true),
        SplitLossRecoveryDecision::LocalFallback
    );
    assert!(!split_candidate_is_valid_replacement_split(&candidate));
}

#[test]
fn split_loss_recovery_ignores_unused_participant_loss() {
    let active_stages = vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)];
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2), participant(3)],
        active_stages.clone(),
    );
    let candidate = SplitTopologyGeneration::new(
        "topology-b".into(),
        "run-b".into(),
        2,
        vec![participant(1), participant(2)],
        active_stages,
    );

    assert_eq!(
        split_loss_recovery_decision(
            &active,
            &[make_id(1), make_id(2)],
            &[],
            Some(&candidate),
            false,
        ),
        SplitLossRecoveryDecision::NoActiveStageLoss
    );
}

#[test]
fn split_loss_recovery_ignores_connected_but_temporarily_ineligible_stage_peer() {
    let active = SplitTopologyGeneration::new(
        "topology-a".into(),
        "run-a".into(),
        1,
        vec![participant(1), participant(2)],
        vec![stage(1, 0, 0, 20), stage(2, 1, 20, 40)],
    );

    assert_eq!(
        split_loss_recovery_decision(&active, &[make_id(1), make_id(2)], &[], None, true,),
        SplitLossRecoveryDecision::NoActiveStageLoss
    );
}

#[test]
fn split_topology_minimum_rejects_single_stage_split_candidate() {
    assert!(split_participants_meet_minimum(&[
        participant(1),
        participant(2)
    ]));
    assert!(!split_participants_meet_minimum(&[participant(1)]));
    assert!(split_stages_meet_minimum(&[
        stage(1, 0, 0, 20),
        stage(2, 1, 20, 40)
    ]));
    assert!(!split_stages_meet_minimum(&[stage(1, 0, 0, 40)]));
}
