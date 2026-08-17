fn test_stage_status(
    node_id: EndpointId,
    stage_id: &str,
    stage_index: u32,
    bind_addr: &str,
    state: crate::inference::skippy::StageRuntimeState,
) -> StageRuntimeStatus {
    StageRuntimeStatus {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: Some("gguf:///model.gguf".to_string()),
        manifest_sha256: Some("direct-gguf:1:model.gguf".to_string()),
        source_model_path: Some("/model.gguf".to_string()),
        source_model_sha256: None,
        source_model_bytes: Some(1),
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stage_id.to_string(),
        stage_index,
        node_id: Some(node_id),
        layer_start: stage_index * 12,
        layer_end: (stage_index + 1) * 12,
        state,
        bind_addr: bind_addr.to_string(),
        activation_width: 896,
        wire_dtype: crate::inference::skippy::StageWireDType::F16,
        selected_device: None,
        ctx_size: 512,
        lane_count: 4,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        error: None,
        shutdown_generation: 1,
    }
}

fn test_stage_load_request() -> crate::inference::skippy::StageLoadRequest {
    crate::inference::skippy::StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 12,
        layer_end: 24,
        model_path: Some("/model.gguf".to_string()),
        source_model_bytes: Some(123_456_789),
        projector_path: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: 896,
        wire_dtype: crate::inference::skippy::StageWireDType::F16,
        ctx_size: 512,
        lane_count: 4,
        n_batch: Some(128),
        n_ubatch: Some(64),
        n_gpu_layers: -1,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        native_mtp_enabled: true,
        shutdown_generation: 7,
        coordinator_term: 11,
        coordinator_id: Some(make_test_endpoint_id(0x70)),
        lease_until_unix_ms: 999_999,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        upstream: None,
        downstream: Some(crate::inference::skippy::StagePeerDescriptor {
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            endpoint: "127.0.0.1:9002".to_string(),
            node_id: Some(make_test_endpoint_id(0x80)),
        }),
    }
}

fn test_preparation_status(
    state: crate::inference::skippy::StagePreparationState,
) -> crate::inference::skippy::StagePreparationStatus {
    crate::inference::skippy::StagePreparationStatus {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 12,
        layer_end: 24,
        state,
        bytes_done: Some(1024),
        bytes_total: Some(4096),
        bind_addr: Some("127.0.0.1:51234".to_string()),
        error: None,
        shutdown_generation: 7,
        coordinator_term: 11,
        coordinator_id: Some(make_test_endpoint_id(0x70)),
        lease_until_unix_ms: 999_999,
    }
}

#[test]
fn stage_control_inventory_request_round_trips_proto() {
    let requester = make_test_endpoint_id(0x81);
    let request = crate::inference::skippy::StageControlRequest::Inventory(
        crate::inference::skippy::StageInventoryRequest {
            model_id: "model-a".to_string(),
            package_ref: "gguf:///model.gguf".to_string(),
            manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        },
    );

    let decoded =
        stage_control_request_from_proto(stage_control_request_to_proto(requester, request))
            .unwrap();

    let crate::inference::skippy::StageControlRequest::Inventory(inventory) = decoded else {
        panic!("expected inventory request");
    };
    assert_eq!(inventory.model_id, "model-a");
    assert_eq!(inventory.package_ref, "gguf:///model.gguf");
    assert_eq!(inventory.manifest_sha256, "direct-gguf:1:model.gguf");
}

#[test]
fn stage_control_prepare_request_round_trips_proto() {
    let requester = make_test_endpoint_id(0x82);
    let coordinator_id = make_test_endpoint_id(0x83);
    let request = crate::inference::skippy::StageControlRequest::Prepare(
        crate::inference::skippy::StagePrepareRequest {
            load: test_stage_load_request(),
            coordinator_id: Some(coordinator_id),
        },
    );

    let decoded =
        stage_control_request_from_proto(stage_control_request_to_proto(requester, request))
            .unwrap();

    let crate::inference::skippy::StageControlRequest::Prepare(prepare) = decoded else {
        panic!("expected prepare request");
    };
    assert_eq!(prepare.coordinator_id, Some(coordinator_id));
    assert_eq!(prepare.load.stage_id, "stage-1");
    assert_eq!(prepare.load.layer_start, 12);
    assert_eq!(prepare.load.layer_end, 24);
    assert_eq!(prepare.load.model_path.as_deref(), Some("/model.gguf"));
    assert_eq!(
        prepare.load.load_mode,
        skippy_protocol::LoadMode::RuntimeSlice
    );
    assert_eq!(
        prepare.load.downstream.and_then(|peer| peer.node_id),
        Some(make_test_endpoint_id(0x80))
    );
}

#[test]
fn stage_control_status_update_request_round_trips_proto() {
    let requester = make_test_endpoint_id(0x84);
    let status = test_preparation_status(crate::inference::skippy::StagePreparationState::Loading);
    let request = crate::inference::skippy::StageControlRequest::StatusUpdate(status);

    let decoded =
        stage_control_request_from_proto(stage_control_request_to_proto(requester, request))
            .unwrap();

    let crate::inference::skippy::StageControlRequest::StatusUpdate(status) = decoded else {
        panic!("expected status update request");
    };
    assert_eq!(
        status.state,
        crate::inference::skippy::StagePreparationState::Loading
    );
    assert_eq!(status.bind_addr.as_deref(), Some("127.0.0.1:51234"));
    assert_eq!(status.bytes_done, Some(1024));
    assert_eq!(status.bytes_total, Some(4096));
}

#[test]
fn stage_control_inventory_response_round_trips_plain_gguf_source() {
    let response = crate::inference::skippy::StageControlResponse::Inventory(
        crate::inference::skippy::StageLayerInventory {
            model_id: "model-a".to_string(),
            package_ref: "gguf:///model.gguf".to_string(),
            manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
            layer_count: 32,
            ready_ranges: vec![crate::inference::skippy::LayerRange {
                layer_start: 0,
                layer_end: 16,
            }],
            available_ranges: vec![crate::inference::skippy::LayerRange {
                layer_start: 0,
                layer_end: 32,
            }],
            missing_ranges: Vec::new(),
            preparing_ranges: vec![test_preparation_status(
                crate::inference::skippy::StagePreparationState::Resolving,
            )],
            source_model_path: Some("/model.gguf".to_string()),
            source_model_bytes: Some(4_096),
            source_model_kind: crate::inference::skippy::SourceModelKind::PlainGguf,
        },
    );

    let decoded =
        stage_control_response_from_proto(stage_control_response_to_proto(response, true)).unwrap();

    let crate::inference::skippy::StageControlResponse::Inventory(inventory) = decoded else {
        panic!("expected inventory response");
    };
    assert_eq!(inventory.layer_count, 32);
    assert_eq!(
        inventory.source_model_kind,
        crate::inference::skippy::SourceModelKind::PlainGguf
    );
    assert_eq!(inventory.source_model_path.as_deref(), Some("/model.gguf"));
    assert_eq!(inventory.available_ranges[0].layer_start, 0);
    assert_eq!(inventory.available_ranges[0].layer_end, 32);
    assert_eq!(
        inventory.preparing_ranges[0].state,
        crate::inference::skippy::StagePreparationState::Resolving
    );
}

#[test]
fn stage_control_prepare_response_round_trips_failed_status() {
    let mut status =
        test_preparation_status(crate::inference::skippy::StagePreparationState::Failed);
    status.error = Some("source GGUF missing".to_string());
    let response = crate::inference::skippy::StageControlResponse::PrepareAccepted(
        crate::inference::skippy::StagePrepareAcceptedResponse {
            accepted: false,
            status,
            error: Some("source GGUF missing".to_string()),
        },
    );

    let decoded =
        stage_control_response_from_proto(stage_control_response_to_proto(response, true)).unwrap();

    let crate::inference::skippy::StageControlResponse::PrepareAccepted(accepted) = decoded else {
        panic!("expected prepare response");
    };
    assert!(!accepted.accepted);
    assert_eq!(
        accepted.status.state,
        crate::inference::skippy::StagePreparationState::Failed
    );
    assert_eq!(accepted.error.as_deref(), Some("source GGUF missing"));
    assert_eq!(
        accepted.status.error.as_deref(),
        Some("source GGUF missing")
    );
}

#[test]
fn stage_control_status_list_response_round_trips_all_statuses() {
    let first = stage_status_from_load(
        &test_stage_load_request(),
        crate::inference::skippy::StageRuntimeState::Ready,
    );
    let mut second = first.clone();
    second.stage_id = "stage-2".to_string();
    second.stage_index = 2;
    second.layer_start = 24;
    second.layer_end = 36;
    second.bind_addr = "127.0.0.1:51235".to_string();
    let response =
        crate::inference::skippy::StageControlResponse::Status(vec![first.clone(), second.clone()]);

    let decoded =
        stage_control_response_from_proto(stage_control_response_to_proto(response, true)).unwrap();

    let crate::inference::skippy::StageControlResponse::Status(statuses) = decoded else {
        panic!("expected status response");
    };
    assert_eq!(statuses.len(), 2);
    assert_eq!(statuses[0].stage_id, first.stage_id);
    assert_eq!(statuses[1].stage_id, second.stage_id);
    assert_eq!(statuses[1].bind_addr, "127.0.0.1:51235");
}

#[test]
fn empty_stage_control_status_list_response_round_trips_as_empty() {
    let response = crate::inference::skippy::StageControlResponse::Status(Vec::new());

    let decoded =
        stage_control_response_from_proto(stage_control_response_to_proto(response, true)).unwrap();

    let crate::inference::skippy::StageControlResponse::Status(statuses) = decoded else {
        panic!("expected status response");
    };
    assert!(statuses.is_empty());
}

#[test]
fn legacy_stage_control_status_response_still_decodes() {
    let status = stage_status_from_load(
        &test_stage_load_request(),
        crate::inference::skippy::StageRuntimeState::Ready,
    );
    let response = crate::inference::skippy::StageControlResponse::Status(vec![status.clone()]);

    let decoded =
        stage_control_response_from_proto(stage_control_response_to_proto(response, false))
            .unwrap();

    let crate::inference::skippy::StageControlResponse::Status(statuses) = decoded else {
        panic!("expected status response");
    };
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].stage_id, status.stage_id);
}

#[test]
fn stage_status_updates_materialized_topology_endpoint() {
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x31; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_topology(StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![StageAssignment {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            node_id,
            layer_start: 12,
            layer_end: 24,
            endpoint: StageEndpoint {
                bind_addr: "127.0.0.1:0".to_string(),
            },
        }],
    });

    state.record_status(test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));

    let topology = state.topologies.values().next().unwrap();
    assert_eq!(topology.stages[0].endpoint.bind_addr, "127.0.0.1:51234");
}

#[test]
fn public_stage_topologies_hide_worker_only_load_fragments() {
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x32; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_topology(StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![StageAssignment {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            node_id,
            layer_start: 12,
            layer_end: 24,
            endpoint: StageEndpoint {
                bind_addr: "127.0.0.1:0".to_string(),
            },
        }],
    });
    state.record_status(test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));

    assert!(state.visible_topologies().is_empty());
    assert_eq!(state.runtime_statuses().len(), 1);
}

#[test]
fn full_stage_topology_remains_visible_after_status_updates() {
    let host_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
    let worker_id = EndpointId::from(SecretKey::from_bytes(&[0x34; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_topology(StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: host_id,
                layer_start: 0,
                layer_end: 12,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:50000".to_string(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: worker_id,
                layer_start: 12,
                layer_end: 24,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:0".to_string(),
                },
            },
        ],
    });
    state.record_status(test_stage_status(
        worker_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));

    let visible = state.visible_topologies();
    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].stages[1].endpoint.bind_addr, "127.0.0.1:51234");
}

#[test]
fn active_stage_topology_replaces_previous_generation_for_model() {
    let host_id = EndpointId::from(SecretKey::from_bytes(&[0x36; 32]).public());
    let first_worker_id = EndpointId::from(SecretKey::from_bytes(&[0x37; 32]).public());
    let second_worker_id = EndpointId::from(SecretKey::from_bytes(&[0x38; 32]).public());
    let mut state = StageTopologyState::default();
    state.activate_topology(StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: host_id,
                layer_start: 0,
                layer_end: 12,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:50000".to_string(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: first_worker_id,
                layer_start: 12,
                layer_end: 24,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:0".to_string(),
                },
            },
        ],
    });
    state.record_status(test_stage_status(
        first_worker_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));

    state.activate_topology(StageTopologyInstance {
        topology_id: "topology-b".to_string(),
        run_id: "run-b".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: host_id,
                layer_start: 0,
                layer_end: 8,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:50001".to_string(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: second_worker_id,
                layer_start: 8,
                layer_end: 24,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:0".to_string(),
                },
            },
        ],
    });

    let visible = state.visible_topologies();
    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].topology_id, "topology-b");
    assert!(state.runtime_statuses().is_empty());
}

#[test]
fn stage_topology_withdraw_removes_active_topology_and_statuses() {
    let host_id = EndpointId::from(SecretKey::from_bytes(&[0x41; 32]).public());
    let worker_id = EndpointId::from(SecretKey::from_bytes(&[0x42; 32]).public());
    let mut state = StageTopologyState::default();
    state.activate_topology(StageTopologyInstance {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stages: vec![
            StageAssignment {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: host_id,
                layer_start: 0,
                layer_end: 12,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:50000".to_string(),
                },
            },
            StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: worker_id,
                layer_start: 12,
                layer_end: 24,
                endpoint: StageEndpoint {
                    bind_addr: "127.0.0.1:0".to_string(),
                },
            },
        ],
    });
    state.record_status(test_stage_status(
        worker_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));

    assert_eq!(state.visible_topologies().len(), 1);
    assert_eq!(state.runtime_statuses().len(), 1);
    assert!(state.withdraw_topology("topology-a", "run-a"));
    assert!(state.visible_topologies().is_empty());
    assert!(state.runtime_statuses().is_empty());
    assert!(!state.withdraw_topology("topology-a", "run-a"));
}

#[test]
fn empty_stage_status_snapshots_are_ignored() {
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x39; 32]).public());
    let mut state = StageTopologyState::default();
    let mut status = test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    );
    status.topology_id.clear();
    status.run_id.clear();
    status.stage_id.clear();

    state.record_status(status);

    assert!(state.runtime_statuses().is_empty());
}

#[test]
fn active_stage_refresh_marks_missing_stage_failed() {
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x35; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_status(test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));
    let cached = state.active_statuses().into_iter().next().unwrap();
    state.record_status(stage_runtime_status_from_snapshot(
        cached.node_id,
        stage_snapshot_from_runtime_status(
            &cached,
            crate::inference::skippy::StageRuntimeState::Failed,
            Some("stage status missing from runtime".to_string()),
        ),
    ));

    let status = state.runtime_statuses().into_iter().next().unwrap();
    assert_eq!(
        status.state,
        crate::inference::skippy::StageRuntimeState::Failed
    );
    assert_eq!(
        status.error.as_deref(),
        Some("stage status missing from runtime")
    );
}

#[test]
fn active_stage_missing_from_runtime_marks_cached_stage_failed() {
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x43; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_status(test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));
    let cached = state.active_statuses().into_iter().next().unwrap();

    state.record_status_refresh_failure(
        &cached,
        crate::mesh::stage_transport::StageStatusRefreshFailure::MissingFromRuntime,
    );

    let status = state.runtime_statuses().into_iter().next().unwrap();
    assert_eq!(
        status.state,
        crate::inference::skippy::StageRuntimeState::Failed
    );
    assert_eq!(
        status.error.as_deref(),
        Some("stage status missing from runtime")
    );
}

#[test]
fn active_stage_refresh_timeout_retains_cached_stage_status() {
    // A transient refresh failure (timeout / unreachable peer) must retain the
    // last-known Ready status rather than tearing down a healthy split on a
    // momentary blip.
    let node_id = EndpointId::from(SecretKey::from_bytes(&[0x43; 32]).public());
    let mut state = StageTopologyState::default();
    state.record_status(test_stage_status(
        node_id,
        "stage-1",
        1,
        "127.0.0.1:51234",
        crate::inference::skippy::StageRuntimeState::Ready,
    ));
    let cached = state.active_statuses().into_iter().next().unwrap();

    state.record_status_refresh_failure(
        &cached,
        crate::mesh::stage_transport::StageStatusRefreshFailure::Transient,
    );

    let status = state.runtime_statuses().into_iter().next().unwrap();
    assert_eq!(
        status.state,
        crate::inference::skippy::StageRuntimeState::Ready
    );
    assert_eq!(status.error, None);
}

#[test]
fn passive_streams_are_gated_when_trust_policy_enforces_ownership() {
    // With an enforcing trust policy, only gossip bypasses the quarantine
    // gate. A leaked invite token must not be a bearer credential for
    // inference: a caller rejected by the trust gate (UntrustedOwner) must
    // not be able to route requests via the passive paths.
    for policy in [TrustPolicy::RequireOwned, TrustPolicy::Allowlist] {
        assert!(
            stream_allowed_before_admission(STREAM_GOSSIP, policy),
            "gossip must always be allowed ({policy:?}) — it is the admission path"
        );
        assert!(
            !stream_allowed_before_admission(STREAM_TUNNEL_HTTP, policy),
            "HTTP tunnel must be gated under {policy:?} — otherwise a leaked token serves inference"
        );
        assert!(
            !stream_allowed_before_admission(STREAM_ROUTE_REQUEST, policy),
            "route request must be gated under {policy:?}"
        );
        assert!(
            !stream_allowed_before_admission(STREAM_TUNNEL, policy),
            "raw tunnel must stay gated under {policy:?}"
        );
    }

    // Non-enforcing policies keep passive paths open. PreferOwned is advisory:
    // it warns about unattributed peers but does not reject them.
    for policy in [TrustPolicy::Off, TrustPolicy::PreferOwned] {
        assert!(stream_allowed_before_admission(STREAM_TUNNEL_HTTP, policy));
        assert!(stream_allowed_before_admission(
            STREAM_ROUTE_REQUEST,
            policy
        ));
    }
}
