use super::heartbeat::{
    HomeRelayStatusTransition, RELAY_DEGRADED_RTT_MS, RELAY_MISSING_GRACE_SECS,
    RELAY_ONLY_RECONNECT_SECS, RELAY_RECONNECT_COOLDOWN_SECS, RelayPathSnapshot,
    RelayPeerHealth, RelayPeerObservation, RelayReconnectController, RelayReconnectReason,
    SelectedPathKind, relay_reconnect_reason, should_remove_connection,
};
use super::*;
use crate::api;
use crate::network::affinity;
use crate::plugin;
use crate::proto::node::{GossipFrame, NodeRole, PeerAnnouncement, RouteTableRequest};
use serial_test::serial;
use skippy_protocol::proto::stage as skippy_stage_proto;
use std::collections::{HashMap, HashSet};
use tokio::sync::{mpsc, watch};

mod direct_path;

#[tokio::test]
async fn owner_control_stream_work_is_bounded_per_connection() {
    let permits = control_stream_semaphore();
    let mut held = Vec::new();
    for _ in 0..MAX_CONTROL_STREAM_WORK_PER_CONNECTION {
        held.push(
            permits
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore should remain open"),
        );
    }

    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(25),
            permits.clone().acquire_owned(),
        )
        .await
        .is_err(),
        "the thirty-third stream must wait for capacity"
    );

    held.pop();
    let _unblocked_permit = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        permits.acquire_owned(),
    )
    .await
    .expect("released capacity should unblock the next stream")
    .expect("semaphore should remain open");
}

#[test]
fn quic_bind_addr_uses_explicit_port_on_all_platforms() {
    assert_eq!(
        quic_bind_addr(QuicBindSelection {
            ip: None,
            port: Some(7000)
        }),
        Some(std::net::SocketAddr::from(([0, 0, 0, 0], 7000)))
    );
}

#[test]
fn quic_bind_addr_uses_explicit_ip_and_port() {
    assert_eq!(
        quic_bind_addr(QuicBindSelection {
            ip: Some("10.1.2.3".parse().unwrap()),
            port: Some(7000)
        }),
        Some("10.1.2.3:7000".parse().unwrap())
    );
}

#[test]
fn quic_bind_addr_uses_explicit_ip_with_ephemeral_port() {
    assert_eq!(
        quic_bind_addr(QuicBindSelection {
            ip: Some("10.1.2.3".parse().unwrap()),
            port: None
        }),
        Some("10.1.2.3:0".parse().unwrap())
    );
}

#[test]
#[cfg(target_os = "windows")]
fn quic_bind_addr_falls_back_to_localhost_ephemeral_on_windows() {
    assert_eq!(
        quic_bind_addr(QuicBindSelection::default()),
        Some(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
    );
}

#[test]
#[cfg(not(target_os = "windows"))]
fn quic_bind_addr_keeps_endpoint_default_on_non_windows() {
    assert_eq!(quic_bind_addr(QuicBindSelection::default()), None);
}

#[test]
fn split_stage_path_allows_fast_direct_path() {
    assert_eq!(
        SplitStagePathSnapshot::direct(Some(MAX_SPLIT_RTT_MS)).stage_path_rejection(),
        None
    );
}

#[test]
fn split_stage_path_rejects_missing_rtt() {
    assert_eq!(
        SplitStagePathSnapshot::direct(None).stage_path_rejection(),
        Some(SplitStagePathRejection::MissingStagePath)
    );
}

#[test]
fn split_stage_path_accepts_direct_path_with_peer_rtt_fallback() {
    assert_eq!(
        SplitStagePathSnapshot::direct(None)
            .with_direct_rtt_fallback(Some(MAX_SPLIT_RTT_MS))
            .stage_path_rejection(),
        None
    );
}

#[test]
fn split_stage_path_keeps_relay_rejection_with_peer_rtt_fallback() {
    assert_eq!(
        SplitStagePathSnapshot::relay(None)
            .with_direct_rtt_fallback(Some(1))
            .stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathRelayOnly)
    );
}

#[test]
fn split_stage_path_rejects_slow_peer_rtt_fallback() {
    assert_eq!(
        SplitStagePathSnapshot::direct(None)
            .with_direct_rtt_fallback(Some(MAX_SPLIT_RTT_MS + 1))
            .stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathTooSlow)
    );
}

#[test]
fn split_stage_path_rejects_relay_path() {
    assert_eq!(
        SplitStagePathSnapshot::relay(Some(1)).stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathRelayOnly)
    );
}

#[test]
fn split_stage_path_rejects_slow_direct_path() {
    assert_eq!(
        SplitStagePathSnapshot::direct(Some(MAX_SPLIT_RTT_MS + 1)).stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathTooSlow)
    );
}

#[test]
fn split_stage_path_rejects_unknown_path() {
    assert_eq!(
        SplitStagePathSnapshot::unknown().stage_path_rejection(),
        Some(SplitStagePathRejection::MissingStagePath)
    );
}

#[test]
fn split_stage_path_uses_direct_peer_path_fallback_for_unknown_stage_path() {
    let fallback = SelectedPathObservation {
        path_type: "direct",
        rtt_ms: Some(MAX_SPLIT_RTT_MS),
        observed_direct_remote_addr: None,
    };

    assert_eq!(
        SplitStagePathSnapshot::unknown()
            .with_peer_path_fallback(Some(fallback))
            .stage_path_rejection(),
        None
    );
}

#[test]
fn split_stage_path_keeps_relay_peer_path_fallback_rejected() {
    let fallback = SelectedPathObservation {
        path_type: "relay",
        rtt_ms: Some(1),
        observed_direct_remote_addr: None,
    };

    assert_eq!(
        SplitStagePathSnapshot::unknown()
            .with_peer_path_fallback(Some(fallback))
            .stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathRelayOnly)
    );
}

#[test]
fn split_stage_path_peer_fallback_does_not_convert_relay_rtt_to_direct() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(0x4a));
    peer.rtt_ms = Some(1);
    peer.selected_path = Some(SelectedPathObservation {
        path_type: "relay",
        rtt_ms: Some(1),
        observed_direct_remote_addr: None,
    });

    assert_eq!(
        SplitStagePathSnapshot::unknown()
            .with_peer_path_fallback(peer.split_stage_path_fallback())
            .stage_path_rejection(),
        Some(SplitStagePathRejection::StagePathRelayOnly)
    );
}

#[test]
fn split_stage_path_peer_fallback_uses_best_direct_rtt() {
    let mut peer = make_test_peer_info(make_test_endpoint_id(0x4b));
    peer.rtt_ms = Some(MAX_SPLIT_RTT_MS);
    peer.selected_path = Some(SelectedPathObservation {
        path_type: "direct",
        rtt_ms: None,
        observed_direct_remote_addr: None,
    });

    assert_eq!(
        SplitStagePathSnapshot::unknown()
            .with_peer_path_fallback(peer.split_stage_path_fallback())
            .stage_path_rejection(),
        None
    );
}

#[test]
fn endpoint_addr_filter_for_bind_ip_keeps_selected_ip_relay_and_public_candidates() {
    let mut addr = EndpointAddr {
        id: make_test_endpoint_id(0x42),
        addrs: Default::default(),
    };
    addr.addrs
        .insert(iroh::TransportAddr::Ip("10.1.2.3:47916".parse().unwrap()));
    addr.addrs
        .insert(iroh::TransportAddr::Ip("172.23.0.1:47916".parse().unwrap()));
    addr.addrs.insert(iroh::TransportAddr::Ip(
        "100.107.22.123:47916".parse().unwrap(),
    ));
    addr.addrs.insert(iroh::TransportAddr::Ip(
        "192.168.1.20:47916".parse().unwrap(),
    ));
    addr.addrs.insert(iroh::TransportAddr::Ip(
        "35.199.1.10:47916".parse().unwrap(),
    ));
    addr.addrs.insert(iroh::TransportAddr::Relay(
        "https://relay.example.com".parse().unwrap(),
    ));

    let filtered = filter_endpoint_addr_for_bind_ip(addr, Some("10.1.2.3".parse().unwrap()), true);
    let ip_addrs: HashSet<_> = filtered
        .addrs
        .iter()
        .filter_map(|addr| match addr {
            iroh::TransportAddr::Ip(socket) => Some(socket.to_string()),
            _ => None,
        })
        .collect();

    assert!(ip_addrs.contains("10.1.2.3:47916"));
    assert!(ip_addrs.contains("35.199.1.10:47916"));
    assert!(!ip_addrs.contains("172.23.0.1:47916"));
    assert!(!ip_addrs.contains("100.107.22.123:47916"));
    assert!(!ip_addrs.contains("192.168.1.20:47916"));
    assert!(
        filtered
            .addrs
            .iter()
            .any(|addr| matches!(addr, iroh::TransportAddr::Relay(_)))
    );
}

#[test]
fn endpoint_addr_filter_for_lan_only_bind_ip_strips_public_candidates() {
    let mut addr = EndpointAddr {
        id: make_test_endpoint_id(0x42),
        addrs: Default::default(),
    };
    addr.addrs
        .insert(iroh::TransportAddr::Ip("10.1.2.3:47916".parse().unwrap()));
    addr.addrs.insert(iroh::TransportAddr::Ip(
        "35.199.1.10:47916".parse().unwrap(),
    ));
    addr.addrs.insert(iroh::TransportAddr::Relay(
        "https://relay.example.com".parse().unwrap(),
    ));

    let filtered = filter_endpoint_addr_for_bind_ip(addr, Some("10.1.2.3".parse().unwrap()), false);
    let ip_addrs: HashSet<_> = filtered
        .addrs
        .iter()
        .filter_map(|addr| match addr {
            iroh::TransportAddr::Ip(socket) => Some(socket.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(ip_addrs, HashSet::from(["10.1.2.3:47916".to_string()]));
    assert!(
        filtered
            .addrs
            .iter()
            .any(|addr| matches!(addr, iroh::TransportAddr::Relay(_)))
    );
}

fn stage_load_request() -> crate::inference::skippy::StageLoadRequest {
    crate::inference::skippy::StageLoadRequest {
        topology_id: "topology-a".to_string(),
        run_id: "run-a".to_string(),
        model_id: "model-a".to_string(),
        backend: "skippy".to_string(),
        package_ref: "hf://meshllm/demo-package".to_string(),
        manifest_sha256: "manifest".to_string(),
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 4,
        layer_end: 8,
        model_path: Some("/models/demo.gguf".to_string()),
        source_model_bytes: Some(123_456_789),
        projector_path: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: 4096,
        wire_dtype: crate::inference::skippy::StageWireDType::F16,
        ctx_size: 8192,
        lane_count: 2,
        n_batch: Some(1024),
        n_ubatch: Some(512),
        n_gpu_layers: -1,
        mmap: Some(false),
        mlock: true,
        cache_type_k: "f16".to_string(),
        cache_type_v: "q8_0".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        native_mtp_enabled: true,
        shutdown_generation: 3,
        coordinator_term: 11,
        coordinator_id: None,
        lease_until_unix_ms: 999_999,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        upstream: None,
        downstream: None,
    }
}

async fn make_test_node(role: super::NodeRole) -> Result<Node> {
    make_test_node_with_requirements(role, crate::MeshRequirements::unrestricted()).await
}

async fn make_test_node_with_requirements(
    role: super::NodeRole,
    local_mesh_requirements: crate::MeshRequirements,
) -> Result<Node> {
    use iroh::endpoint::QuicTransportConfig;

    let transport_config = QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(128u32.into())
        .build();
    let endpoint_secret_key = SecretKey::generate();
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(endpoint_secret_key.clone())
        .alpns(vec![
            ALPN_V1.to_vec(),
            skippy_protocol::STAGE_ALPN_V2.to_vec(),
        ])
        .transport_config(transport_config)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;

    let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
    let (inflight_change_tx, _) = watch::channel(0u64);
    let (tunnel_tx, _tunnel_rx) = tokio::sync::mpsc::channel(8);
    let (tunnel_http_tx, _tunnel_http_rx) = tokio::sync::mpsc::channel(8);
    let (stage_transport_tx, _stage_transport_rx) = tokio::sync::mpsc::channel(8);
    let runtime_data_producer = crate::runtime_data::RuntimeDataCollector::new().producer(
        crate::runtime_data::RuntimeDataSource {
            scope: "routing",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        },
    );

    let node = Node {
        endpoint,
        endpoint_secret_key,
        public_addr: None,
        quic_bind: QuicBindSelection::default(),
        relay_policy: RelayPolicy::DefaultPublic,
        owner_keypair: None,
        local_mesh_requirements,
        state: Arc::new(Mutex::new(MeshState {
            peers: HashMap::new(),
            connections: HashMap::new(),
            pending_connections: HashMap::new(),
            next_pending_connection_attempt: 1,
            remote_tunnel_maps: HashMap::new(),
            dead_peers: HashMap::new(),
            peer_down_rejections: HashMap::new(),
            direct_path_request_last_at: HashMap::new(),
            seen_plugin_messages: HashMap::new(),
            seen_plugin_message_order: VecDeque::new(),
            policy_rejected_peers: HashMap::new(),
            requirement_rejected_peers: HashSet::new(),
            recent_mesh_rejections: VecDeque::new(),
        })),
        role: Arc::new(Mutex::new(role)),
        models: Arc::new(Mutex::new(Vec::new())),
        model_source: Arc::new(Mutex::new(None)),
        serving_models: Arc::new(Mutex::new(Vec::new())),
        served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
        model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
        hosted_models: Arc::new(Mutex::new(Vec::new())),
        llama_ready: Arc::new(Mutex::new(false)),
        available_models: Arc::new(Mutex::new(Vec::new())),
        requested_models: Arc::new(Mutex::new(Vec::new())),
        explicit_model_interests: Arc::new(Mutex::new(Vec::new())),
        model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
        requirement_mesh_state: Arc::new(Mutex::new(None)),
        mesh_id: Arc::new(Mutex::new(None)),
        mesh_policy_hash: Arc::new(Mutex::new(None)),
        signed_genesis_policy: Arc::new(Mutex::new(None)),
        bootstrap_token: Arc::new(Mutex::new(None)),
        join_targets: Arc::new(Mutex::new(Vec::new())),
        first_joined_mesh_ts: Arc::new(Mutex::new(None)),
        accepting: Arc::new((
            tokio::sync::Notify::new(),
            std::sync::atomic::AtomicBool::new(false),
        )),
        vram_bytes: 64 * 1024 * 1024 * 1024,
        peer_change_tx,
        peer_change_rx,
        inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        inflight_change_tx,
        routing_metrics: crate::network::metrics::RoutingMetrics::default(),
        routing_telemetry: Arc::new(std::sync::Mutex::new(None)),
        swarm_capture: Arc::new(std::sync::Mutex::new(None)),
        local_request_metrics: Arc::new(LocalRequestMetricsSampler::default()),
        runtime_data_producer,
        tunnel_tx,
        tunnel_http_tx,
        stage_transport_tx,
        stage_control_tx: Arc::new(Mutex::new(None)),
        stage_transport_bridges: Arc::new(Mutex::new(HashMap::new())),
        stage_transport_aliases: Arc::new(Mutex::new(HashMap::new())),
        stage_topologies: Arc::new(Mutex::new(StageTopologyState::default())),
        plugin_manager: Arc::new(Mutex::new(None)),
        display_name: Arc::new(Mutex::new(None)),
        owner_attestation: Arc::new(Mutex::new(None)),
        release_attestation: Arc::new(Mutex::new(None)),
        release_attestation_summary: Arc::new(Mutex::new(
            crate::ReleaseAttestationSummary::default(),
        )),
        owner_summary: Arc::new(Mutex::new(OwnershipSummary::default())),
        control_listener: Arc::new(Mutex::new(None)),
        trust_store: Arc::new(Mutex::new(TrustStore::default())),
        trust_policy: TrustPolicy::Off,
        enumerate_host: false,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
        gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
        config_state: Arc::new(tokio::sync::Mutex::new(
            crate::runtime::config_state::ConfigState::default(),
        )),
        config_revision_tx: {
            let (tx, _rx) = tokio::sync::watch::channel(0u64);
            Arc::new(tx)
        },
    };

    let accept_node = node.clone();
    tokio::spawn(async move {
        accept_node.accept_loop().await;
    });

    Ok(node)
}

#[tokio::test]
async fn set_serving_models_preserves_existing_known_descriptor_capabilities_when_adding_model()
-> Result<()> {
    let node = make_test_node(super::NodeRole::Worker).await?;
    let vision_model = "Qwen3VL-2B-Instruct-Q4_K_M".to_string();
    let text_model = "Qwen3-8B-Q4_K_M".to_string();

    node.set_serving_models(vec![vision_model.clone()]).await;
    node.upsert_served_model_descriptor(ServedModelDescriptor {
        identity: ServedModelIdentity {
            model_name: vision_model.clone(),
            is_primary: true,
            source_kind: ModelSourceKind::LocalGguf,
            local_file_name: Some(format!("{vision_model}.gguf")),
            ..Default::default()
        },
        capabilities_known: true,
        capabilities: crate::models::ModelCapabilities {
            multimodal: true,
            vision: crate::models::CapabilityLevel::Supported,
            ..Default::default()
        },
        topology: None,
        metadata: None,
    })
    .await;

    node.set_serving_models(vec![vision_model.clone(), text_model.clone()])
        .await;

    let descriptors = node.served_model_descriptors().await;
    let vision = descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == vision_model)
        .expect("existing vision descriptor should remain served");
    assert!(vision.identity.is_primary);
    assert!(vision.capabilities_known);
    assert_eq!(
        vision.capabilities.vision,
        crate::models::CapabilityLevel::Supported
    );
    assert!(vision.capabilities.multimodal);

    let text = descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == text_model)
        .expect("new text descriptor should be inferred");
    assert!(!text.identity.is_primary);
    assert!(!text.capabilities_known);
    assert_eq!(
        text.capabilities,
        crate::models::ModelCapabilities::default()
    );

    Ok(())
}

#[tokio::test]
async fn local_request_metrics_snapshot_tracks_accepted_and_completed_requests() {
    let node = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node should initialize");

    {
        let _request = node.begin_inflight_request();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let snapshot = node.local_request_metrics_snapshot();
    assert_eq!(snapshot.accepted_request_counts.len(), 24 * 60 * 60);
    assert_eq!(snapshot.accepted_request_counts.iter().sum::<u64>(), 1);
    assert_eq!(snapshot.latency_samples_ms.len(), 1);
}

#[derive(Default)]
struct TestRoutingTelemetrySink {
    inflight: std::sync::Mutex<Vec<u64>>,
    requests: std::sync::Mutex<
        Vec<(
            Option<String>,
            usize,
            crate::network::metrics::RequestOutcome,
        )>,
    >,
    attempts: std::sync::Mutex<
        Vec<(
            Option<String>,
            String,
            crate::network::metrics::AttemptOutcome,
        )>,
    >,
}

impl crate::network::metrics::RoutingTelemetrySink for TestRoutingTelemetrySink {
    fn observe_inflight_requests(&self, current: u64) {
        self.inflight.lock().unwrap().push(current);
    }

    fn record_model_request(
        &self,
        model: Option<&str>,
        attempts: usize,
        outcome: crate::network::metrics::RequestOutcome,
    ) {
        self.requests
            .lock()
            .unwrap()
            .push((model.map(str::to_string), attempts, outcome));
    }

    fn record_route_attempt(
        &self,
        model: Option<&str>,
        target: &crate::network::metrics::AttemptTarget,
        outcome: crate::network::metrics::AttemptOutcome,
    ) {
        let target_kind = match target {
            crate::network::metrics::AttemptTarget::Local(_) => "local",
            crate::network::metrics::AttemptTarget::Remote(_) => "remote",
            crate::network::metrics::AttemptTarget::Endpoint(_) => "endpoint",
        };
        self.attempts.lock().unwrap().push((
            model.map(str::to_string),
            target_kind.into(),
            outcome,
        ));
    }
}

#[tokio::test]
async fn routing_telemetry_sink_receives_request_pressure_and_attempt_events() {
    let node = make_test_node(super::NodeRole::Client)
        .await
        .expect("test node should initialize");
    let sink = Arc::new(TestRoutingTelemetrySink::default());
    node.set_routing_telemetry_sink(Some(sink.clone()));

    {
        let _request = node.begin_inflight_request();
        assert_eq!(sink.inflight.lock().unwrap().as_slice(), &[1]);
    }
    assert_eq!(sink.inflight.lock().unwrap().as_slice(), &[1, 0]);

    node.record_routed_request(
        Some("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
        2,
        crate::network::metrics::RequestOutcome::Success(
            crate::network::metrics::RequestService::Remote,
        ),
    );
    node.record_inference_attempt(
        Some("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
        &crate::inference::election::InferenceTarget::Remote(iroh::EndpointId::from(
            SecretKey::from_bytes(&[0x45; 32]).public(),
        )),
        std::time::Duration::from_millis(3),
        std::time::Duration::from_millis(5),
        crate::network::metrics::AttemptOutcome::Success,
        Some(16),
    );

    let requests = sink.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0],
        (
            Some("Qwen/Qwen3-8B-GGUF:Q4_K_M".into()),
            2,
            crate::network::metrics::RequestOutcome::Success(
                crate::network::metrics::RequestService::Remote
            )
        )
    );
    drop(requests);

    let attempts = sink.attempts.lock().unwrap();
    assert_eq!(
        attempts.as_slice(),
        &[(
            Some("Qwen/Qwen3-8B-GGUF:Q4_K_M".into()),
            "remote".into(),
            crate::network::metrics::AttemptOutcome::Success,
        )]
    );
}
