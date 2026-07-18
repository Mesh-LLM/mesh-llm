use super::*;
use crate::crypto::OwnershipSummary;
use crate::mesh::connections::encode_endpoint_addr_token;
use crate::mesh::now_secs;
use iroh::SecretKey;
use mesh_llm_types::mesh::DEMAND_TTL_SECS;
use std::collections::HashMap;

pub(crate) fn test_endpoint_id(seed: u8) -> EndpointId {
    EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
}

pub(crate) fn test_addr(seed: u8) -> EndpointAddr {
    EndpointAddr {
        id: test_endpoint_id(seed),
        addrs: Default::default(),
    }
}

pub(crate) fn test_announcement(ts: Option<u64>) -> PeerAnnouncement {
    PeerAnnouncement {
        addr: test_addr(0x11),
        role: NodeRole::Worker,
        first_joined_mesh_ts: ts,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        mesh_policy_hash: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
        artifact_transfer_supported: true,
        stage_protocol_generation_supported: true,
        stage_status_list_supported: true,
        advertised_model_throughput: vec![],
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
    }
}

pub(crate) fn test_peer(ts: Option<u64>) -> PeerInfo {
    PeerInfo::from_announcement(
        test_endpoint_id(0x22),
        test_addr(0x22),
        &test_announcement(ts),
        OwnershipSummary::default(),
    )
}

#[test]
pub(crate) fn test_merge_none_to_some() {
    let mut existing = test_peer(None);
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_some_to_none_keeps_existing() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(None);

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_earlier_incoming_wins() {
    let mut existing = test_peer(Some(200));
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_later_incoming_loses() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(Some(200));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_equal_values_unchanged() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_meaningfully_changed_first_joined_mesh_ts() {
    let old_peer = test_peer(Some(100));
    let new_peer = test_peer(Some(200));

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_explicit_model_interests() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_stage_status_list_support() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.stage_status_list_supported = !old_peer.stage_status_list_supported;

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_stage_protocol_generation_support() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.stage_protocol_generation_supported = !old_peer.stage_protocol_generation_supported;

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_explicit_model_interests() {
    let mut existing = test_peer(Some(100));
    let mut ann = test_announcement(Some(100));
    ann.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(
        existing.explicit_model_interests,
        vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
    );
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_stage_status_list_support() {
    let mut existing = test_peer(Some(100));
    existing.stage_status_list_supported = false;
    let mut ann = test_announcement(Some(100));
    ann.stage_status_list_supported = true;

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert!(existing.stage_status_list_supported);
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_stage_protocol_generation_support() {
    let mut existing = test_peer(Some(100));
    existing.stage_protocol_generation_supported = false;
    let mut ann = test_announcement(Some(100));
    ann.stage_protocol_generation_supported = true;

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert!(existing.stage_protocol_generation_supported);
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_advertised_model_throughput() {
    let mut existing = test_peer(Some(100));
    let mut ann = test_announcement(Some(100));
    ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
        model_name: "qwen".to_string(),
        avg_tokens_per_second_milli: 35_000,
        throughput_samples: 4,
    }];

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(
        existing.advertised_model_throughput,
        ann.advertised_model_throughput
    );
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_stage_status_list_support() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x44);
    let addr = test_addr(0x44);
    let mut ann = test_announcement(Some(100));
    ann.stage_status_list_supported = false;

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.stage_status_list_supported = true;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(peer.stage_status_list_supported);
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_stage_protocol_generation_support() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x45);
    let addr = test_addr(0x45);
    let mut ann = test_announcement(Some(100));
    ann.stage_protocol_generation_supported = false;

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.stage_protocol_generation_supported = true;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(peer.stage_protocol_generation_supported);
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_advertised_model_throughput() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x46);
    let addr = test_addr(0x46);
    let mut ann = test_announcement(Some(100));
    ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
        model_name: "qwen".to_string(),
        avg_tokens_per_second_milli: 20_000,
        throughput_samples: 2,
    }];

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.advertised_model_throughput[0].avg_tokens_per_second_milli = 48_000;
    ann.advertised_model_throughput[0].throughput_samples = 9;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert_eq!(
        peer.advertised_model_throughput,
        ann.advertised_model_throughput
    );
}

#[tokio::test]
pub(crate) async fn test_collect_announcements_includes_self_explicit_model_interests() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.set_explicit_model_interests(vec![
        "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
        "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
    ])
    .await;

    let announcements = node.collect_announcements().await;
    let self_announcement = announcements
        .iter()
        .find(|announcement| announcement.addr.id == node.id())
        .expect("self announcement must be present");

    assert_eq!(
        self_announcement.explicit_model_interests,
        vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
    );
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

#[tokio::test]
pub(crate) async fn future_demand_timestamps_are_active_without_underflow() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let now = now_secs();
    node.merge_remote_demand(&HashMap::from([(
        "future-demand".to_string(),
        ModelDemand {
            last_active: now + DEMAND_TTL_SECS + 60,
            request_count: 3,
        },
    )]));

    let active = node.active_demand().await;
    assert!(active.contains_key("future-demand"));

    node.gc_demand().await;
    assert!(node.get_demand().contains_key("future-demand"));
}

/// Regression test for the `--auto` startup wedge: when a transitive
/// gossip payload includes peers that would be rejected at ingest
/// (version-floor or idle-transitive-client), `maybe_connect_discovered_peer`
/// must skip the dial. Otherwise each unreachable ghost address triggers
/// a 30 s `connect_to_peer` timeout sequentially in the dial loop,
/// wedging the surrounding gossip exchange (and the `attempt_run_auto_join`
/// that initiated it) for tens of minutes.
///
/// The function returns without panicking and without dialing within a
/// generous time bound — a real dial to a fake address would block on
/// the 30 s `PEER_CONNECT_AND_GOSSIP_TIMEOUT`. We assert the result is
/// reached well under that bound and that no connection entry was created.
#[tokio::test]
pub(crate) async fn maybe_connect_discovered_peer_skips_filtered_announcements() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let my_role = NodeRole::Worker;

    // Below-floor version — must be skipped without dialing.
    let old_addr = test_addr(0x57);
    let old_id = old_addr.id;
    let mut old_ann = test_announcement(None);
    old_ann.addr = old_addr.clone();
    old_ann.role = NodeRole::Client;
    old_ann.version = Some("0.57.0".to_string());

    // Idle transitive client (matching version, but no hostname / no
    // direct measurement / no model interests) — must also be skipped.
    let idle_addr = test_addr(0xC1);
    let idle_id = idle_addr.id;
    let mut idle_ann = test_announcement(None);
    idle_ann.addr = idle_addr.clone();
    idle_ann.role = NodeRole::Client;
    idle_ann.version = Some("0.65.1".to_string());

    // Both calls together must return well under the 30 s connect
    // timeout. If the dial-loop skip is missing, each call will block
    // on PEER_CONNECT_AND_GOSSIP_TIMEOUT (30 s) attempting to dial the
    // fake test address.
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        node.maybe_connect_discovered_peer(&my_role, old_addr, &old_ann, true, false)
            .await;
        node.maybe_connect_discovered_peer(&my_role, idle_addr, &idle_ann, true, false)
            .await;
    })
    .await
    .expect("filtered peers must be skipped quickly, not dialed");

    // No connection was attempted (no entry in state.connections), and
    // no peer was added (the filtered announcements never reach add_peer
    // or update_transitive_peer through this path).
    let state = node.state.lock().await;
    assert!(
        !state.connections.contains_key(&old_id),
        "below-floor peer must not be dialed"
    );
    assert!(
        !state.connections.contains_key(&idle_id),
        "idle transitive client must not be dialed"
    );
    assert!(
        !state.peers.contains_key(&old_id),
        "below-floor peer must not be added (this path is dial-only)"
    );
    assert!(
        !state.peers.contains_key(&idle_id),
        "idle transitive client must not be added (this path is dial-only)"
    );
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_returns_none_for_single_candidate() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let token = encode_endpoint_addr_token(&test_addr(0x42));

    let selected = node
        .join_first_responsive_candidate(&[(token, Some("single".to_string()))])
        .await
        .unwrap();

    assert!(selected.is_none());
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_candidate_collection_filters_unusable_tokens() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let valid_addr = test_addr(0x42);
    let dead_addr = test_addr(0x43);
    let self_token = encode_endpoint_addr_token(&node.endpoint_addr_for_advertisement());
    let dead_token = encode_endpoint_addr_token(&dead_addr);
    let valid_token = encode_endpoint_addr_token(&valid_addr);

    node.state
        .lock()
        .await
        .dead_peers
        .insert(dead_addr.id, std::time::Instant::now());

    let candidates = node
        .collect_join_probe_candidates(&[
            ("not-an-invite-token".to_string(), None),
            (self_token, None),
            (dead_token, None),
            (valid_token, Some("usable".to_string())),
        ])
        .await;

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].addr.id, valid_addr.id);
    assert_eq!(candidates[0].mesh_name.as_deref(), Some("usable"));
}
