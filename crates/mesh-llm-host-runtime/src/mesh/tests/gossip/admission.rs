#[tokio::test]
pub(crate) async fn gossip_admission_advertisement_matrix() {
    use crate::runtime::activity_policy::{
        activity_advertisement_decision, ActivityPolicyState,
    };
    use mesh_llm_config::ActivityAdvertisement;

    let accepting = crate::proto::node::InferenceAdmissionState::Accepting;
    let paused = crate::proto::node::InferenceAdmissionState::RemotePaused;
    let cases = [
        (ActivityAdvertisement::None, false, None, false),
        (ActivityAdvertisement::None, true, None, false),
        (ActivityAdvertisement::AvailabilityOnly, false, None, false),
        (ActivityAdvertisement::AvailabilityOnly, true, None, true),
        (
            ActivityAdvertisement::CoarseState,
            false,
            Some(accepting),
            false,
        ),
        (
            ActivityAdvertisement::CoarseState,
            true,
            Some(paused),
            true,
        ),
    ];
    for (mode, blocking, expected_state, expected_withdrawal) in cases {
        let state = if blocking {
            ActivityPolicyState::RemotePaused
        } else {
            ActivityPolicyState::Accepting
        };
        let decision = activity_advertisement_decision(true, mode, state, false);
        assert_eq!(decision.admission_state, expected_state, "mode={mode:?}");
        assert_eq!(
            decision.withdraw_model_availability, expected_withdrawal,
            "mode={mode:?}"
        );
    }

    let private = activity_advertisement_decision(
        true,
        ActivityAdvertisement::PrivateCoarseState,
        ActivityPolicyState::AllPaused,
        false,
    );
    assert_eq!(
        private.admission_state,
        Some(crate::proto::node::InferenceAdmissionState::AllPaused)
    );
    assert!(private.withdraw_model_availability);

    let public = activity_advertisement_decision(
        true,
        ActivityAdvertisement::PrivateCoarseState,
        ActivityPolicyState::AllPaused,
        true,
    );
    assert_eq!(public.admission_state, None);
    assert!(public.withdraw_model_availability);

    let disabled = activity_advertisement_decision(
        false,
        ActivityAdvertisement::CoarseState,
        ActivityPolicyState::AllPaused,
        false,
    );
    assert_eq!(disabled.admission_state, None);
    assert!(!disabled.withdraw_model_availability);
}

#[tokio::test]
pub(crate) async fn legacy_peer_admission_and_known_empty_withdrawal() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x71);
    let addr = test_addr(0x71);

    let mut legacy_add = test_announcement(Some(100));
    legacy_add.addr = addr.clone();
    legacy_add.role = NodeRole::Host { http_port: 9337 };
    legacy_add.version = None;
    legacy_add.serving_models = vec!["legacy-model".to_string()];
    legacy_add.hosted_models = None;
    legacy_add.inference_admission_state = None;
    node.add_peer(peer_id, addr.clone(), &legacy_add, None).await;

    let state = node.state.lock().await;
    let peer = state
        .peers
        .get(&peer_id)
        .expect("legacy-style peer should be admitted");
    assert_eq!(peer.version, None);
    assert!(!peer.hosted_models_known);
    assert!(peer.routes_http_model("legacy-model"));
    drop(state);

    let mut legacy_withdrawal = test_announcement(Some(100));
    legacy_withdrawal.addr = addr.clone();
    legacy_withdrawal.role = NodeRole::Host { http_port: 9337 };
    legacy_withdrawal.version = None;
    legacy_withdrawal.serving_models = Vec::new();
    legacy_withdrawal.hosted_models = Some(Vec::new());
    legacy_withdrawal.inference_admission_state = None;
    node.add_peer(peer_id, addr, &legacy_withdrawal, None).await;

    let state = node.state.lock().await;
    let peer = state
        .peers
        .get(&peer_id)
        .expect("legacy-withdrawal update should be applied");
    assert_eq!(peer.hosted_models, Vec::<String>::new());
    assert!(peer.hosted_models_known);
    assert!(!peer.routes_http_model("legacy-model"));
}

#[test]
pub(crate) fn version_allowed_for_rebroadcast_handles_floor() {
    // At or above the floor — allowed.
    assert!(version_allowed_for_rebroadcast(Some("0.60.0")));
    assert!(version_allowed_for_rebroadcast(Some("0.60.2")));
    assert!(version_allowed_for_rebroadcast(Some("0.64.0")));
    assert!(version_allowed_for_rebroadcast(Some("0.65.1")));
    assert!(version_allowed_for_rebroadcast(Some("1.0.0")));
    // Below the floor — refused.
    assert!(!version_allowed_for_rebroadcast(Some("0.57.0")));
    assert!(!version_allowed_for_rebroadcast(Some("0.55.1")));
    assert!(!version_allowed_for_rebroadcast(Some("0.58.0")));
    assert!(!version_allowed_for_rebroadcast(Some("0.59.99")));
}

#[test]
pub(crate) fn version_allowed_for_rebroadcast_handles_metadata_and_prerelease() {
    // Build metadata is stripped.
    assert!(version_allowed_for_rebroadcast(Some(
        "0.65.1+skippy.20260504.kv.2"
    )));
    assert!(!version_allowed_for_rebroadcast(Some("0.57.0+anything")));
    // Pre-release tags are stripped — 0.63.0-rc5 still passes.
    assert!(version_allowed_for_rebroadcast(Some("0.63.0-rc5")));
    assert!(!version_allowed_for_rebroadcast(Some("0.58.0-beta")));
}

#[test]
pub(crate) fn version_allowed_for_rebroadcast_is_conservative_on_unknown() {
    // Unparseable / missing / empty — preserved (don't drop legacy nodes
    // that never advertised a version).
    assert!(version_allowed_for_rebroadcast(None));
    assert!(version_allowed_for_rebroadcast(Some("")));
    assert!(version_allowed_for_rebroadcast(Some("   ")));
    assert!(version_allowed_for_rebroadcast(Some("garbage")));
    assert!(version_allowed_for_rebroadcast(Some("0")));
    assert!(version_allowed_for_rebroadcast(Some("0.x")));
}

#[tokio::test]
pub(crate) async fn transitive_ingest_rejects_below_version_floor() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let old_addr = test_addr(0x57);
    let new_addr = test_addr(0x65);
    let old_id = old_addr.id;
    let new_id = new_addr.id;

    let mut old_ann = test_announcement(None);
    old_ann.addr = old_addr.clone();
    old_ann.role = NodeRole::Client;
    old_ann.version = Some("0.57.0".to_string());
    let mut new_ann = test_announcement(None);
    new_ann.addr = new_addr.clone();
    new_ann.role = NodeRole::Client;
    new_ann.version = Some("0.65.0".to_string());
    // Give the v0.65.0 client a demand signal so the idle-transitive-
    // client filter (a separate gate) doesn't drop it — this test
    // exercises the version floor specifically.
    new_ann.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

    let bridge = test_endpoint_id(0xBB);
    node.update_transitive_peer(old_id, &old_addr, &old_ann, bridge)
        .await;
    node.update_transitive_peer(new_id, &new_addr, &new_ann, bridge)
        .await;

    // Old peer must NOT be in local state — it was rejected at ingest.
    // New peer must be present.
    {
        let state = node.state.lock().await;
        assert!(
            !state.peers.contains_key(&old_id),
            "v0.57.0 peer must be rejected at ingest, not appear in local state"
        );
        assert!(
            state.peers.contains_key(&new_id),
            "v0.65.0 peer should be added to local state"
        );
    }

    // Outbound gossip must also exclude the old peer.
    let announcements = node.collect_announcements().await;
    assert!(
        !announcements.iter().any(|a| a.addr.id == old_id),
        "v0.57.0 peer must not appear in outbound gossip"
    );
    assert!(
        announcements.iter().any(|a| a.addr.id == new_id),
        "v0.65.0 peer should appear in outbound gossip"
    );
}

#[test]
pub(crate) fn peer_is_idle_transitive_client_basic_shapes() {
    // Empty idle client: no hostname, no direct measurement, no
    // interests → caught.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    assert!(peer_is_idle_transitive_client(&ann));

    // Real idle user with a hostname → kept.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.hostname = Some("Sams-MacBook-Pro.local".into());
    assert!(!peer_is_idle_transitive_client(&ann));

    // Hostname-less client that someone directly measured → kept.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.latency_source = Some(crate::proto::node::LatencySource::Direct);
    assert!(!peer_is_idle_transitive_client(&ann));

    // Estimated latency (propagated guess, not direct) — still caught;
    // only Direct counts as proof of contact.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.latency_source = Some(crate::proto::node::LatencySource::Estimated);
    assert!(peer_is_idle_transitive_client(&ann));

    // Client asking for a model → kept (demand signal).
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];
    assert!(!peer_is_idle_transitive_client(&ann));

    // Client somehow advertising serving → kept.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.serving_models = vec!["Qwen3-8B-Q4_K_M".to_string()];
    assert!(!peer_is_idle_transitive_client(&ann));

    // Client advertising hosted → kept.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Client;
    ann.hosted_models = Some(vec!["Qwen3-8B-Q4_K_M".to_string()]);
    assert!(!peer_is_idle_transitive_client(&ann));

    // Host → never caught regardless of other fields.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Host { http_port: 9337 };
    assert!(!peer_is_idle_transitive_client(&ann));

    // Worker → never caught.
    let mut ann = test_announcement(None);
    ann.role = NodeRole::Worker;
    assert!(!peer_is_idle_transitive_client(&ann));
}

#[tokio::test]
pub(crate) async fn transitive_ingest_drops_idle_clients_but_keeps_clients_with_demand() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let idle_addr = test_addr(0xC1);
    let demand_addr = test_addr(0xC2);
    let host_addr = test_addr(0xC3);
    let idle_id = idle_addr.id;
    let demand_id = demand_addr.id;
    let host_id = host_addr.id;

    // Idle client — should be dropped at transitive ingest.
    let mut idle = test_announcement(None);
    idle.addr = idle_addr.clone();
    idle.role = NodeRole::Client;
    idle.version = Some("0.65.1".to_string());

    // Client asking for a model — must be kept (demand signal).
    let mut with_demand = test_announcement(None);
    with_demand.addr = demand_addr.clone();
    with_demand.role = NodeRole::Client;
    with_demand.version = Some("0.65.1".to_string());
    with_demand.requested_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

    // Host — must be kept (real compute).
    let mut host = test_announcement(None);
    host.addr = host_addr.clone();
    host.role = NodeRole::Host { http_port: 9337 };
    host.version = Some("0.65.1".to_string());
    host.serving_models = vec!["Qwen3-8B-Q4_K_M".to_string()];

    let bridge = test_endpoint_id(0xBB);
    node.update_transitive_peer(idle_id, &idle_addr, &idle, bridge)
        .await;
    node.update_transitive_peer(demand_id, &demand_addr, &with_demand, bridge)
        .await;
    node.update_transitive_peer(host_id, &host_addr, &host, bridge)
        .await;

    let state = node.state.lock().await;
    assert!(
        !state.peers.contains_key(&idle_id),
        "idle transitive client must be rejected"
    );
    assert!(
        state.peers.contains_key(&demand_id),
        "client with requested_models must be kept (demand signal)"
    );
    assert!(
        state.peers.contains_key(&host_id),
        "host must be kept (real compute)"
    );
}

#[tokio::test]
pub(crate) async fn direct_add_peer_admits_idle_clients() {
    // Idle clients we actually directly contact are still admitted.
    // The predicate is for transitive ingest only — a direct connection
    // is proof of life and the peer is observable.
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let addr = test_addr(0xC4);
    let id = addr.id;

    let mut ann = test_announcement(None);
    ann.addr = addr.clone();
    ann.role = NodeRole::Client;
    ann.version = Some("0.65.1".to_string());
    // No requested, no serving, no hosted — pure idle client.

    node.add_peer(id, addr, &ann, None).await;

    let state = node.state.lock().await;
    assert!(
        state.peers.contains_key(&id),
        "direct idle client must be admitted (direct contact is proof of life)"
    );
}

#[tokio::test]
#[serial]
pub(crate) async fn direct_add_peer_rejects_below_version_floor() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let addr = test_addr(0x57);
    let id = addr.id;

    let mut ann = test_announcement(None);
    ann.addr = addr.clone();
    ann.role = NodeRole::Client;
    ann.version = Some("0.57.0".to_string());

    node.add_peer(id, addr, &ann, None).await;

    let state = node.state.lock().await;
    assert!(
        !state.peers.contains_key(&id),
        "direct add of v0.57.0 peer must be rejected (no local state entry)"
    );
}

#[tokio::test]
#[serial]
pub(crate) async fn below_version_floor_audit_is_direct_only() {
    let direct_addr = test_addr(0x57);
    let mut direct_ann = test_announcement(None);
    direct_ann.addr = direct_addr.clone();
    direct_ann.role = NodeRole::Client;
    direct_ann.version = Some("0.57.0".to_string());

    let (mut direct_audits, direct_capture) =
        crate::mesh::capture_mesh_operational_audits();
    let direct_node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    direct_node
        .add_peer(
            direct_addr.id,
            direct_addr,
            &direct_ann,
            Some(crate::protocol::NODE_PROTOCOL_GENERATION),
        )
        .await;

    let direct_subject_id = hex::encode(direct_ann.addr.id.as_bytes());
    let mut direct_records = Vec::new();
    while let Ok(record) = direct_audits.try_recv() {
        if record.code() == "gossip_incompatible_version_rejected"
            && record
                .context()
                .is_some_and(|context| context.fields()["subject_id"] == direct_subject_id)
        {
            direct_records.push(record);
        }
    }
    assert_eq!(direct_records.len(), 1);
    let direct_record = &direct_records[0];
    assert_eq!(direct_record.code(), "gossip_incompatible_version_rejected");
    let direct_context = direct_record.context().expect("direct peer context").fields();
    assert_eq!(
        direct_context["reason_code"],
        "protocol_version_unsupported"
    );
    assert_eq!(direct_context["outcome"], "rejected");
    assert!(direct_context.get("numeric_summaries").is_none());
    assert_eq!(direct_context["subject_kind"], "mesh_peer");
    assert_eq!(
        direct_context["subject_id"],
        direct_subject_id
    );
    drop(direct_capture);

    let transitive_addr = test_addr(0x58);
    let mut transitive_ann = test_announcement(None);
    transitive_ann.addr = transitive_addr.clone();
    transitive_ann.role = NodeRole::Host { http_port: 9337 };
    transitive_ann.version = Some("0.57.0".to_string());
    let bridge = test_endpoint_id(0xBB);

    let (mut transitive_audits, _transitive_capture) =
        crate::mesh::capture_mesh_operational_audits();
    let transitive_node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    transitive_node
        .update_transitive_peer(
            transitive_addr.id,
            &transitive_addr,
            &transitive_ann,
            bridge,
        )
        .await;

    let transitive_subject_id = hex::encode(transitive_ann.addr.id.as_bytes());
    let mut transitive_records = 0;
    while let Ok(record) = transitive_audits.try_recv() {
        if record.code() == "gossip_incompatible_version_rejected"
            && record
                .context()
                .is_some_and(|context| context.fields()["subject_id"] == transitive_subject_id)
        {
            transitive_records += 1;
        }
    }
    assert_eq!(transitive_records, 0);
}

#[tokio::test]
pub(crate) async fn rejected_direct_peer_does_not_apply_mesh_state_or_demand() {
    // Given: a direct announcement that passes requirement validation but is
    // rejected by the supported-version admission policy.
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let addr = test_addr(0x59);
    let mut ann = test_announcement(None);
    ann.addr = addr.clone();
    ann.version = Some("0.57.0".to_string());
    ann.mesh_id = Some("rejected-mesh".to_string());
    ann.model_demand.insert(
        "rejected-demand".to_string(),
        ModelDemand {
            last_active: now_secs(),
            request_count: 1,
        },
    );

    // When: the direct payload is applied through the production gossip path.
    node.apply_announced_peers(addr.id, &[(addr.clone(), ann)], None, None, true)
        .await
        .expect("policy rejection should not fail the gossip frame");

    // Then: admission rejection must happen before mesh-wide side effects.
    assert_eq!(node.mesh_id().await, None);
    assert!(node.get_demand().is_empty());
    assert!(!node.state.lock().await.peers.contains_key(&addr.id));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
pub(crate) async fn inbound_gossip_rejection_preserves_dead_peer_state() -> Result<()> {
    // Given: a host that still considers the direct sender dead, and a direct
    // gossip payload whose sender announcement fails admission.
    let host = Node::new_for_tests(NodeRole::Worker).await?;
    let policy = crate::MeshGenesisPolicy::new(
        "test-owner",
        current_time_unix_ms(),
        crate::MeshRequirements::default(),
    )
    .expect("test policy should be valid");
    let mesh_id = policy.policy_derived_mesh_id().expect("mesh id");
    let policy_hash = policy.canonical_hash_hex().expect("policy hash");
    host.set_active_mesh_policy_for_tests(policy).await;

    let sender_id = test_endpoint_id(0x58);
    host.state
        .lock()
        .await
        .dead_peers
        .insert(sender_id, std::time::Instant::now());

    let mut sender_announcement = test_announcement(None);
    sender_announcement.addr = EndpointAddr {
        id: sender_id,
        addrs: Default::default(),
    };
    sender_announcement.role = NodeRole::Client;
    sender_announcement.version = Some(crate::VERSION.to_string());
    sender_announcement.mesh_id = Some(mesh_id);
    sender_announcement.mesh_policy_hash = Some(policy_hash);
    let announcements = [(sender_announcement.addr.clone(), sender_announcement)];

    // When: the same production phase used by `handle_gossip_stream` processes
    // the rejected direct announcement.
    host.validate_and_capture_inbound_gossip(
        ControlProtocol::ProtoV1,
        &announcements,
        AnnouncedPeerContext::direct(sender_id, Some(NODE_PROTOCOL_GENERATION)),
    )
    .await
    .expect_err("rejected gossip must fail admission");

    // Then: admission rejection is recorded, but liveness recovery has not
    // cleared the sender from dead_peers.
    let state = host.state.lock().await;
    assert!(
        state.dead_peers.contains_key(&sender_id),
        "rejected inbound gossip must not clear dead-peer state"
    );
    assert!(
        state.requirement_rejected_peers.contains(&sender_id),
        "rejected inbound gossip should still be tracked as an admission rejection"
    );
    assert!(
        !state.peers.contains_key(&sender_id),
        "rejected inbound gossip must not admit the sender"
    );

    Ok(())
}
