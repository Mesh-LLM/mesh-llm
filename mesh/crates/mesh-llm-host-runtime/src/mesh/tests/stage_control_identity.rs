// Stage-control command identity binding (#2217).
//
// The stage-control wire frame carries the authenticated QUIC requester
// (`requester_id`; `decode_stage_control_request` pins it to the peer's
// endpoint identity before conversion). The wire-to-domain conversion must
// keep that identity and refuse commands whose self-asserted coordinator
// identity does not match it: today an admitted mesh peer that is not the
// coordinator can submit a Claim or Load under a forged (or absent)
// coordinator identity with a higher term, and the command reaches the stage
// transition, where the term alone decides fencing and shutdown.

fn identity_test_claim(
    coordinator: EndpointId,
    term: u64,
) -> crate::inference::skippy::StageCoordinatorClaim {
    crate::inference::skippy::StageCoordinatorClaim {
        model_id: "model-a".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        coordinator_id: coordinator.to_string(),
        coordinator_term: term,
        participant_set_hash: "participants".to_string(),
        topology_hash: "topology".to_string(),
        lease_until_unix_ms: u64::MAX,
    }
}

fn identity_test_load_request() -> crate::inference::skippy::StageLoadRequest {
    crate::inference::skippy::StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        runtime_profile: Some(String::new()),
        backend: "skippy".to_string(),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 12,
        layer_end: 24,
        admission: crate::inference::skippy::test_stage_admission(12, 24),
        participant_set_hash: "participants".to_string(),
        topology_hash: "topology".to_string(),
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: Default::default(),
        topology_stages: vec![
            crate::inference::skippy::StageTopologyStageDescriptor {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: make_test_endpoint_id(0x60),
                layer_start: 0,
                layer_end: 12,
                bind_addr: "127.0.0.1:9000".to_string(),
            },
            crate::inference::skippy::StageTopologyStageDescriptor {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: make_test_endpoint_id(0x80),
                layer_start: 12,
                layer_end: 24,
                bind_addr: "127.0.0.1:0".to_string(),
            },
        ],
        model_path: Some("/model.gguf".to_string()),
        source_model_bytes: Some(123_456_789),
        source_model_sha256: None,
        split_certification: Some("certified".to_string()),
        local_source_required: false,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
        generation_signal_window: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        ctx_size: 512,
        lane_count: 4,
        continuous_batching: true,
        last_stage_decode_batch: None,
        n_batch: Some(128),
        n_ubatch: Some(64),
        n_gpu_layers: -1,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        runtime_settings: crate::inference::skippy::StageLoadRuntimeSettings {
            repack: true,
            op_offload: Some(false),
            no_host_buffer: true,
            check_tensors: true,
            direct_io: true,
            main_gpu: Some(2),
            split_mode: skippy_protocol::SplitMode::Row,
            kv_offload: Some(false),
            kv_unified: Some(true),
            swa_full: Some(false),
            cache_idle_slots: Some(5),
            activation_codec_policy: Default::default(),
        },
        native_mtp_enabled: true,
        dflash: None,
        shutdown_generation: 7,
        coordinator_term: 11,
        coordinator_id: Some(make_test_endpoint_id(0x70)),
        lease_until_unix_ms: 999_999,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        upstream: None,
        downstream: Some(crate::inference::skippy::StagePeerDescriptor {
            persistent_lane_marker: false,
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            endpoint: "127.0.0.1:9002".to_string(),
            node_id: Some(make_test_endpoint_id(0x80)),
        }),
    }
}

#[test]
fn admitted_non_coordinator_claim_with_forged_identity_and_higher_term_is_rejected() {
    let coordinator = make_test_endpoint_id(0x70);
    let attacker = make_test_endpoint_id(0xB0);

    // Wire frame: the authenticated QUIC requester is the attacker (this is
    // what `decode_stage_control_request` pins against the peer identity),
    // while the claim body forges the coordinator identity at a higher term.
    let frame = stage_control_request_to_proto(
        attacker,
        crate::inference::skippy::StageControlRequest::Claim(identity_test_claim(coordinator, 2)),
    )
    .expect("forged claim frame must encode");

    let error = stage_control_request_from_proto(frame)
        .expect_err(
            "#2217: a forged higher-term claim from a non-coordinator must not convert into an executable stage transition",
        )
        .to_string();
    assert!(
        error.contains("authenticated requester"),
        "forged claim rejection must cite the identity mismatch: {error}"
    );
}

#[test]
fn admitted_non_coordinator_load_with_forged_identity_and_higher_term_is_rejected() {
    let coordinator = make_test_endpoint_id(0x70);
    let attacker = make_test_endpoint_id(0xB1);

    let mut load = identity_test_load_request();
    load.coordinator_id = Some(coordinator);
    load.coordinator_term = 22;
    let frame = stage_control_request_to_proto(
        attacker,
        crate::inference::skippy::StageControlRequest::Load(load),
    )
    .expect("forged load frame must encode");

    let error = stage_control_request_from_proto(frame)
        .expect_err(
            "#2217: a forged higher-term load from a non-coordinator must not convert into an executable stage transition",
        )
        .to_string();
    assert!(
        error.contains("authenticated requester"),
        "forged load rejection must cite the identity mismatch: {error}"
    );
}

#[test]
fn legitimate_coordinator_commands_still_convert_for_their_authenticated_requester() {
    let coordinator = make_test_endpoint_id(0x70);

    let claim_frame = stage_control_request_to_proto(
        coordinator,
        crate::inference::skippy::StageControlRequest::Claim(identity_test_claim(coordinator, 1)),
    )
    .expect("legitimate claim frame must encode");
    assert!(
        matches!(
            stage_control_request_from_proto(claim_frame).unwrap(),
            crate::inference::skippy::StageControlRequest::Claim(_)
        ),
        "a claim issued by the claimed coordinator must still convert"
    );

    let mut load = identity_test_load_request();
    load.coordinator_id = Some(coordinator);
    let load_frame = stage_control_request_to_proto(
        coordinator,
        crate::inference::skippy::StageControlRequest::Load(load),
    )
    .expect("legitimate load frame must encode");
    assert!(
        matches!(
            stage_control_request_from_proto(load_frame).unwrap(),
            crate::inference::skippy::StageControlRequest::Load(_)
        ),
        "a load issued by the claimed coordinator must still convert"
    );
}

#[test]
fn unbound_legacy_load_still_converts_from_the_authenticated_requester() {
    let requester = make_test_endpoint_id(0x72);
    let mut load = identity_test_load_request();
    load.coordinator_id = None;
    load.coordinator_term = 0;

    let frame = stage_control_request_to_proto(
        requester,
        crate::inference::skippy::StageControlRequest::Load(load),
    )
    .expect("unbound load frame must encode");
    assert!(
        stage_control_request_from_proto(frame).is_ok(),
        "an unbound legacy load must keep converting at the wire boundary"
    );
}

// Regression guard for the #2217 authorization follow-up: the split
// coordinator probes a peer's stage inventory during eligibility planning
// and before that peer records the topology being planned (the target only
// registers the topology once it accepts a load). Requiring a registered
// topology for Inventory dropped the peer out of split membership as
// `stage_control_unreachable` before the coordinator could claim the lease.
fn identity_test_inventory_request() -> crate::inference::skippy::StageInventoryRequest {
    crate::inference::skippy::StageInventoryRequest {
        model_id: "model-a".to_string(),
        runtime_profile: Some(String::new()),
        package_ref: "gguf:///model.gguf".to_string(),
        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
        expected_source_model_sha256: None,
        local_source_required: true,
    }
}

#[tokio::test]
async fn pre_topology_inventory_query_is_admitted_for_authenticated_mesh_peer() -> Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;
    let requester = make_test_endpoint_id(0xC1);

    node.authorize_remote_stage_control_request(
        requester,
        &crate::inference::skippy::StageControlRequest::Inventory(
            identity_test_inventory_request(),
        ),
    )
    .await
    .map_err(|error| {
        anyhow::anyhow!(
            "pre-topology inventory probe must be admitted before the target records the planned topology: {error}"
        )
    })?;

    Ok(())
}

#[tokio::test]
async fn remote_status_query_outside_the_target_topology_is_rejected() -> Result<()> {
    let node = Node::new_for_tests(super::NodeRole::Worker).await?;
    let participant = make_test_endpoint_id(0x60);
    let outsider = make_test_endpoint_id(0xC2);

    node.stage_topologies
        .lock()
        .await
        .record_topology(crate::mesh::StageTopologyInstance {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: "gguf:///model.gguf".to_string(),
            manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
            admissions: Default::default(),
            stages: vec![crate::mesh::StageAssignment {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: participant,
                layer_start: 12,
                layer_end: 24,
                endpoint: crate::mesh::StageEndpoint {
                    bind_addr: "127.0.0.1:0".to_string(),
                },
            }],
        });

    let error = node
        .authorize_remote_stage_control_request(
            outsider,
            &crate::inference::skippy::StageControlRequest::Status(
                crate::inference::skippy::StageStatusFilter {
                    topology_id: Some("topology-a".to_string()),
                    run_id: Some("run-a".to_string()),
                    stage_id: None,
                },
            ),
        )
        .await
        .expect_err(
            "#2217: a status query from a peer outside the target topology must stay rejected",
        )
        .to_string();
    assert!(
        error.contains("not a participant"),
        "status rejection must cite the participant check: {error}"
    );

    let unknown = node
        .authorize_remote_stage_control_request(
            participant,
            &crate::inference::skippy::StageControlRequest::Status(
                crate::inference::skippy::StageStatusFilter {
                    topology_id: Some("topology-missing".to_string()),
                    run_id: Some("run-missing".to_string()),
                    stage_id: None,
                },
            ),
        )
        .await
        .expect_err("a status query for an unknown topology must stay rejected")
        .to_string();
    assert!(
        unknown.contains("unknown stage topology"),
        "unknown-topology rejection must name the missing topology: {unknown}"
    );

    Ok(())
}
