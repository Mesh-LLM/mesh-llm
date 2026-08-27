use super::{AffinityRouter, RequestId, handle_mesh_request};
use crate::inference::election;
use crate::mesh;
use crate::network::openai::ingress::api_proxy;
use crate::network::tunnel::Manager as TunnelManager;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{oneshot, watch};

async fn start_http_tunnel_test_node() -> (mesh::Node, mesh::TunnelChannels) {
    let relay_urls = Vec::new();
    let relay_auths = HashMap::new();
    mesh::Node::start(
        mesh::NodeRole::Client,
        mesh::RelayConfig {
            urls: &relay_urls,
            auths: &relay_auths,
            policy: mesh::RelayPolicy::Disabled,
        },
        mesh::QuicBindSelection {
            ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            port: Some(0),
        },
        Some(0.0),
        false,
        None,
        None,
        crate::MeshRequirements::unrestricted(),
    )
    .await
    .expect("start HTTP tunnel test node")
}

async fn spawn_tunnel_capture() -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind tunnel capture listener");
    let port = listener
        .local_addr()
        .expect("tunnel capture listener address")
        .port();
    let (capture_tx, capture_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let mut request = vec![0; 8192];
        let bytes_read = stream.read(&mut request).await.unwrap_or_default();
        request.truncate(bytes_read);
        let _ = capture_tx.send(request);
        let _ = stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await;
    });
    (port, capture_rx, handle)
}

async fn send_tunneled_request(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    path: &str,
) -> Vec<u8> {
    let body = r#"{"model":"test"}"#;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len(),
    );
    let (mut send, mut recv) = node
        .open_http_tunnel(peer_id)
        .await
        .expect("open HTTP tunnel");
    send.write_all(request.as_bytes())
        .await
        .expect("write tunneled lifecycle request");
    send.finish().expect("finish tunneled lifecycle request");
    recv.read_to_end(4 * 1024 * 1024)
        .await
        .expect("read tunneled lifecycle response")
}

#[tokio::test]
async fn paused_inbound_quic_http_rejects_before_reading_or_routing_request() {
    let (sender, sender_channels) = start_http_tunnel_test_node().await;
    let (mut receiver, receiver_channels) = start_http_tunnel_test_node().await;
    receiver.activity_policy_guard = crate::runtime::activity_policy::ActivityPolicyGuard::new(
        &mesh_llm_config::RuntimeActivityConfig {
            enabled: true,
            response: mesh_llm_config::ActivityResponse::PauseRemote,
            ..Default::default()
        },
    );
    receiver
        .activity_policy_guard
        .update_detector_state(mesh_llm_system::activity::HostActivity::Active);

    let (upstream_port, upstream_rx, upstream_handle) = spawn_tunnel_capture().await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "test".to_string(),
        vec![election::InferenceTarget::Local(upstream_port)],
    );
    let (_target_tx, target_rx) = watch::channel(targets);
    let tunnel_manager = TunnelManager::start(
        receiver.clone(),
        receiver_channels.rpc,
        receiver_channels.http,
        receiver_channels.stage,
    )
    .await
    .expect("start receiving tunnel manager");
    tunnel_manager.set_http_ingress(target_rx, AffinityRouter::new());
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect HTTP tunnel test nodes");

    let (mut send, mut recv) = sender
        .open_http_tunnel(receiver.id())
        .await
        .expect("open paused HTTP tunnel");
    send.write_all(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 8388608\r\n\r\n",
    )
    .await
    .expect("write request headers");

    let wire = tokio::time::timeout(Duration::from_secs(2), recv.read_to_end(1024 * 1024))
        .await
        .expect("paused host must respond without waiting for the declared body")
        .expect("read paused response");
    let response = String::from_utf8(wire).expect("HTTP response should be UTF-8");
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable"),
        "response: {response}"
    );
    assert!(response.contains("remote inference paused (host activity)"));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), upstream_rx)
            .await
            .is_err(),
        "paused request must not reach an inference target"
    );

    drop(send);
    upstream_handle.abort();
    drop(sender_channels);
}

#[tokio::test]
async fn inbound_quic_http_dispatches_without_local_api_listener() {
    let (sender, sender_channels) = start_http_tunnel_test_node().await;
    let (receiver, receiver_channels) = start_http_tunnel_test_node().await;
    let (upstream_port, upstream_rx, upstream_handle) = spawn_tunnel_capture().await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "test".to_string(),
        vec![election::InferenceTarget::Local(upstream_port)],
    );
    let (_target_tx, target_rx) = watch::channel(targets);
    let tunnel_manager = TunnelManager::start(
        receiver.clone(),
        receiver_channels.rpc,
        receiver_channels.http,
        receiver_channels.stage,
    )
    .await
    .expect("start receiving tunnel manager");
    tunnel_manager.set_http_ingress(target_rx, AffinityRouter::new());
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect HTTP tunnel test nodes");

    let wire = send_tunneled_request(&sender, receiver.id(), "/v1/chat/completions").await;
    let response = String::from_utf8(wire).expect("HTTP response should be UTF-8");
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "response: {response}"
    );
    let upstream_request = upstream_rx.await.expect("capture upstream request");
    assert!(
        String::from_utf8_lossy(&upstream_request).starts_with("POST /v1/chat/completions "),
        "upstream request: {}",
        String::from_utf8_lossy(&upstream_request)
    );

    upstream_handle.abort();
    drop(sender_channels);
}

async fn assert_passive_legacy_lifecycle_path_is_rejected(path: &str) {
    let (sender, sender_channels) = start_http_tunnel_test_node().await;
    let (receiver, receiver_channels) = start_http_tunnel_test_node().await;
    let (upstream_port, upstream_rx, upstream_handle) = spawn_tunnel_capture().await;
    let api_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind receiving API proxy listener");
    let api_port = api_listener
        .local_addr()
        .expect("receiving API proxy listener address")
        .port();
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "test".to_string(),
        vec![election::InferenceTarget::Local(upstream_port)],
    );
    let (_target_tx, target_rx) = watch::channel(targets);
    let api_proxy_handle = tokio::spawn(api_proxy(
        receiver.clone(),
        api_port,
        target_rx,
        Some(api_listener),
        false,
        AffinityRouter::new(),
    ));
    let tunnel_manager = TunnelManager::start(
        receiver.clone(),
        receiver_channels.rpc,
        receiver_channels.http,
        receiver_channels.stage,
    )
    .await
    .expect("start receiving tunnel manager");
    tunnel_manager.set_http_port(api_port);
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect HTTP tunnel test nodes");

    let before_models = receiver.models.lock().await.clone();
    let before_requested_models = receiver.requested_models.lock().await.clone();
    let before_runtime_intents = receiver.runtime_intents.lock().unwrap().len();
    let wire = send_tunneled_request(&sender, receiver.id(), path).await;
    let response = String::from_utf8(wire).expect("HTTP response should be UTF-8");
    assert!(
        response.starts_with("HTTP/1.1 410 Gone"),
        "response: {response}"
    );
    assert!(
        response.contains("legacy_route_gone"),
        "response: {response}"
    );
    assert_eq!(receiver.models.lock().await.clone(), before_models);
    assert_eq!(
        receiver.requested_models.lock().await.clone(),
        before_requested_models
    );
    assert_eq!(
        receiver.runtime_intents.lock().unwrap().len(),
        before_runtime_intents
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), upstream_rx)
            .await
            .is_err(),
        "legacy lifecycle paths must not reach an inference target"
    );
    api_proxy_handle.abort();
    upstream_handle.abort();
    drop(sender_channels);
}

#[tokio::test]
async fn passive_legacy_lifecycle_paths_are_rejected_before_mesh_routing() {
    for path in ["/mesh/load", "/mesh/drop?model=test"] {
        assert_passive_legacy_lifecycle_path_is_rejected(path).await;
    }
}

#[tokio::test]
#[serial_test::serial]
async fn passive_missing_model_error_persists_the_client_visible_response_artifact() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let root = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind passive test listener");
    let address = listener.local_addr().expect("passive listener address");
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept passive client");
        handle_mesh_request(node, stream.into(), true, AffinityRouter::new()).await;
    });

    let request_id = RequestId::new();
    let body = r#"{"model":"not-served"}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n{body}",
        request_id.as_uuid(),
        body.len(),
    );
    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect passive client");
    let caller_addr = client
        .local_addr()
        .expect("passive caller address")
        .to_string();
    client
        .write_all(request.as_bytes())
        .await
        .expect("write parsed request");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read passive error");
    server.await.expect("passive handler joins");
    assert!(
        String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 429 Too Many Requests"),
        "the passive no-model route returns its normal client-visible error"
    );

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    let active = state
        .service_for_test()
        .expect("logging service")
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("active request summary");
    assert_eq!(active.metadata.caller_addr(), Some(caller_addr.as_str()));
    assert_eq!(active.metadata.caller_path_type(), Some("local_http"));
    state.pump_persistence_for_test().await;
    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("durable request query")
        .expect("durable request summary");
    assert_eq!(durable.caller_addr.as_deref(), Some(caller_addr.as_str()));
    assert_eq!(durable.caller_path_type.as_deref(), Some("local_http"));
    assert!(durable.caller_endpoint_id.is_none());
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_id.as_uuid().to_string(),
            &mesh_llm_log_store::PageQuery {
                limit: 10,
                cursor: None,
                sort: mesh_llm_log_store::QuerySort::Ascending,
            },
        )
        .expect("response artifact query");
    let response = artifacts
        .items
        .iter()
        .find(|artifact| artifact.kind == "response")
        .expect("durable error response artifact");
    assert_eq!(response.media_kind.as_deref(), Some("application/json"));
    let body_start = wire
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP response header terminator")
        + 4;
    let content = state
        .query_facade()
        .expect("artifact reader")
        .read_artifact(&response.artifact_id)
        .expect("response artifact content");
    assert_eq!(content.bytes, wire[body_start..]);
    let response_json = String::from_utf8(content.bytes).expect("JSON response artifact");
    assert!(response_json.contains("not-served"));
    assert!(response_json.contains("rate_limit_exceeded"));
}

#[tokio::test]
#[serial_test::serial]
async fn passive_body_parse_error_persists_a_response_only_after_complete_headers() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let root = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind passive parser test listener");
    let address = listener.local_addr().expect("passive listener address");
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept passive client");
        handle_mesh_request(node, stream.into(), true, AffinityRouter::new()).await;
    });

    let request_id = RequestId::new();
    let request = format!(
        "POST /v1/tokenize HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Length: 1\r\n\r\n{{",
        request_id.as_uuid(),
    );
    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect passive client");
    client
        .write_all(request.as_bytes())
        .await
        .expect("write malformed body");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read passive error");
    server.await.expect("passive handler joins");
    assert!(String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 400 Bad Request"));

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    state.pump_persistence_for_test().await;
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_id.as_uuid().to_string(),
            &mesh_llm_log_store::PageQuery {
                limit: 10,
                cursor: None,
                sort: mesh_llm_log_store::QuerySort::Ascending,
            },
        )
        .expect("response artifact query");
    assert_eq!(
        artifacts.items.len(),
        1,
        "pre-admission parse failures must never fabricate a request artifact"
    );
    let response = &artifacts.items[0];
    assert_eq!(response.kind, "response");
    assert_eq!(response.media_kind.as_deref(), Some("application/json"));
    let body_start = wire
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP response header terminator")
        + 4;
    let content = state
        .query_facade()
        .expect("artifact reader")
        .read_artifact(&response.artifact_id)
        .expect("response artifact content");
    assert_eq!(content.bytes, wire[body_start..]);
}
