//! Bounded duplicate-public-alias reproduction against actual peer HTTP ingress.
use super::*;
const INTERNAL: &str = "Falcon-H1-1.5B-Instruct-Q4_K_M";
const PUBLIC: &str = "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M";
fn duplicate_descriptor(model: &str) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.to_owned(),
            source_kind: mesh::ModelSourceKind::Catalog,
            canonical_ref: Some(PUBLIC.to_owned()),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn duplicate_peer(peer_id: iroh::EndpointId, model: &str) -> mesh::PeerInfo {
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![model.to_string()],
        hosted_models: vec![model.to_string()],
        hosted_models_known: true,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        memory: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: std::collections::HashMap::new(),
        served_model_descriptors: vec![duplicate_descriptor(model)],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![],
        #[cfg(feature = "payments")]
        lightning_offers: Default::default(),
        cache_affinity: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        inference_admission_state: None,
        decode_batch_policy_supported: false,
    }
}

#[tokio::test]
async fn duplicate_public_alias_executes_on_selected_host() {
    duplicate_host_request(PUBLIC, true, false, false, "").await;
}

#[tokio::test]
async fn duplicate_internal_name_executes_on_selected_host() {
    duplicate_host_request(INTERNAL, true, false, false, "").await;
}

#[tokio::test]
async fn peer_missing_local_model_does_not_use_advertised_remote() {
    duplicate_host_request(INTERNAL, false, false, false, "").await;
}

#[tokio::test]
async fn peer_automatic_model_does_not_route() {
    duplicate_host_request("mesh", true, false, false, "").await;
}

#[tokio::test]
async fn peer_without_execution_handler_is_rejected() {
    duplicate_host_request(INTERNAL, false, true, false, "").await;
}

#[tokio::test]
async fn local_http_can_still_route_to_peer() {
    duplicate_host_request(PUBLIC, true, false, true, "").await;
}

#[tokio::test]
async fn peer_cannot_force_forwarding_with_headers() {
    duplicate_host_request(INTERNAL, true, false, false, "exclude-self").await;
}

async fn duplicate_host_request(
    model: &str,
    available: bool,
    unavailable_ingress: bool,
    local: bool,
    header: &str,
) {
    let (a, ac, _ad) = start_http_tunnel_test_node().await;
    let (b, bc, _bd) = start_http_tunnel_test_node().await;
    let (port_a, mut backend_a, task_a) = spawn_tunnel_capture().await;
    let (port_b, mut backend_b, task_b) = spawn_tunnel_capture().await;
    let mut managers = Vec::new();
    let mut senders = Vec::new();
    for (node, channels, port) in [(&a, ac, port_a), (&b, bc, port_b)] {
        node.set_hosted_models(vec![INTERNAL.to_owned()]).await;
        node.set_served_model_descriptors(vec![duplicate_descriptor(INTERNAL)])
            .await;
        let mut targets = election::ModelTargets::default();
        let candidates = duplicate_candidates(node.id(), a.id(), b.id(), port, available);
        targets.targets.insert(INTERNAL.to_owned(), candidates);
        let (tx, rx) = watch::channel(targets);
        let manager =
            TunnelManager::start(node.clone(), channels.rpc, channels.http, channels.stage)
                .await
                .unwrap();
        if !unavailable_ingress || node.id() != b.id() {
            manager.set_http_ingress(rx, AffinityRouter::new());
        }
        managers.push(manager);
        senders.push(tx);
        node.start_accepting();
    }
    a.connect_to_peer(b.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    a.insert_test_peer(duplicate_peer(b.id(), INTERNAL)).await;
    b.insert_test_peer(duplicate_peer(a.id(), INTERNAL)).await;
    assert_duplicate_names(&a).await;
    let raw = duplicate_request_bytes(model, header, b.id());
    let response = send_duplicate_request(&a, &b, raw.as_bytes(), local).await;
    let observed = backend_b.try_recv();
    let no_backend_a = backend_a.try_recv().is_err();
    a.close_endpoint().await;
    b.close_endpoint().await;
    task_a.abort();
    task_b.abort();
    assert!(
        no_backend_a,
        "selected peer forwarded inference back to caller"
    );
    if unavailable_ingress {
        assert_unavailable_ingress(&response, observed.ok());
    } else {
        assert_duplicate_result(model, available, header, observed.ok(), &response);
    }
}

fn assert_unavailable_ingress(response: &[u8], observed: Option<Vec<u8>>) {
    assert!(String::from_utf8_lossy(response).starts_with("HTTP/1.1 503"));
    assert!(observed.is_none());
}

async fn assert_duplicate_names(node: &mesh::Node) {
    let served = node.models_being_served().await;
    assert!(served.contains(&PUBLIC.to_owned()), "{served:?}");
    assert!(served.contains(&INTERNAL.to_owned()), "{served:?}");
}

async fn send_duplicate_request(
    a: &mesh::Node,
    b: &mesh::Node,
    raw: &[u8],
    local: bool,
) -> Vec<u8> {
    if local {
        return local_gateway_request(a, b, raw).await;
    }
    let (mut send, mut recv) = a.open_http_tunnel(b.id()).await.unwrap();
    send.write_all(raw).await.unwrap();
    send.finish().unwrap();
    tokio::time::timeout(Duration::from_secs(5), recv.read_to_end(16384))
        .await
        .unwrap()
        .unwrap()
}

fn duplicate_request_bytes(model: &str, header: &str, host: iroh::EndpointId) -> String {
    let body = serde_json::json!({"model":model,"messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true}).to_string();
    let extra = if header == "exclude-self" {
        format!("x-mesh-exclude: {}\r\n", hex::encode(host.as_bytes()))
    } else {
        String::new()
    };
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{extra}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn duplicate_candidates(
    node: iroh::EndpointId,
    a: iroh::EndpointId,
    b: iroh::EndpointId,
    port: u16,
    available: bool,
) -> Vec<election::InferenceTarget> {
    if node == b && !available {
        vec![election::InferenceTarget::Remote(a)]
    } else {
        vec![
            election::InferenceTarget::Local(port),
            election::InferenceTarget::Remote(if node == a { b } else { a }),
        ]
    }
}

fn assert_duplicate_result(
    model: &str,
    available: bool,
    header: &str,
    observed: Option<Vec<u8>>,
    response: &[u8],
) {
    let expects_backend = available && model != "mesh" && header.is_empty();
    if expects_backend {
        let raw = observed.expect("selected peer must execute locally");
        assert!(String::from_utf8_lossy(&raw).contains(INTERNAL));
    } else {
        assert!(observed.is_none(), "unexpected backend execution");
        let status = if model == "mesh" {
            "400"
        } else if !header.is_empty() {
            "409"
        } else {
            "503"
        };
        assert!(
            String::from_utf8_lossy(response).starts_with(&format!("HTTP/1.1 {status}")),
            "{}",
            String::from_utf8_lossy(response)
        );
    }
}

async fn local_gateway_request(a: &mesh::Node, b: &mesh::Node, raw: &[u8]) -> Vec<u8> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        INTERNAL.into(),
        vec![election::InferenceTarget::Remote(b.id())],
    );
    let (_tx, rx) = watch::channel(targets);
    let proxy = tokio::spawn(crate::network::openai::ingress::api_proxy(
        a.clone(),
        port,
        rx,
        Some(listener.into()),
        false,
        AffinityRouter::new(),
    ));
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    client.write_all(raw).await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), client.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    proxy.abort();
    response
}

#[tokio::test]
async fn peer_ingress_executes_registered_plugin_without_mesh_forwarding() {
    let (caller, _channels, _dir) = start_http_tunnel_test_node().await;
    let (host, channels, _host_dir) = start_http_tunnel_test_node().await;
    let (port, backend, task) = spawn_tunnel_capture().await;
    let manager = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "test-provider".into(),
            endpoint_id: "execution".into(),
            address: format!("http://127.0.0.1:{port}"),
            models: vec![PUBLIC.into()],
        }])
        .await;
    host.set_plugin_manager(manager).await;
    // A remote target with the same public model must not shadow the local plugin.
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        PUBLIC.into(),
        vec![election::InferenceTarget::Remote(caller.id())],
    );
    let (_tx, rx) = watch::channel(targets);
    let tunnel = TunnelManager::start(host.clone(), channels.rpc, channels.http, channels.stage)
        .await
        .unwrap();
    tunnel.set_http_ingress(rx, AffinityRouter::new());
    host.start_accepting();
    caller
        .connect_to_peer(host.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    let body = serde_json::json!({"model": PUBLIC, "messages": [{"role":"user", "content":"hello"}], "stream":true}).to_string();
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let (mut send, _recv) = caller.open_http_tunnel(host.id()).await.unwrap();
    send.write_all(raw.as_bytes()).await.unwrap();
    send.finish().unwrap();
    let captured = tokio::time::timeout(Duration::from_secs(3), backend).await;
    caller.close_endpoint().await;
    host.close_endpoint().await;
    task.abort();
    let captured = captured.expect("plugin execution timed out").unwrap();
    assert!(String::from_utf8_lossy(&captured).contains(PUBLIC));
}

#[tokio::test]
async fn peer_incompatible_local_workload_does_not_discover_capable_remote() {
    let (caller, mut caller_channels, _caller_directory) = start_http_tunnel_test_node().await;
    let (receiver, channels, _receiver_directory) = start_http_tunnel_test_node().await;
    let (local_port, mut local_requests, local_task) = spawn_tunnel_capture().await;
    let mut local_descriptor = duplicate_descriptor(INTERNAL);
    local_descriptor.metadata = Some(mesh::ServedModelMetadata {
        workload_class: Some(mesh::ModelWorkloadClass::CausalGeneration),
        ..Default::default()
    });
    receiver.set_hosted_models(vec![INTERNAL.into()]).await;
    receiver
        .set_served_model_descriptors(vec![local_descriptor])
        .await;
    let mut capable_peer = duplicate_peer(caller.id(), INTERNAL);
    capable_peer.served_model_descriptors[0].metadata = Some(mesh::ServedModelMetadata {
        workload_class: Some(mesh::ModelWorkloadClass::Embedding),
        ..Default::default()
    });
    caller.set_hosted_models(vec![INTERNAL.into()]).await;
    caller
        .set_served_model_descriptors(capable_peer.served_model_descriptors.clone())
        .await;
    receiver.insert_test_peer(capable_peer.clone()).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        INTERNAL.into(),
        vec![election::InferenceTarget::Local(local_port)],
    );
    // Prove this fixture would discover the capable peer on ordinary user ingress.
    let user_candidates = crate::network::openai::workload_routing::ingress_candidates(
        &receiver,
        INTERNAL,
        "/v1/embeddings",
        &targets,
    )
    .await;
    assert_eq!(
        user_candidates,
        vec![election::InferenceTarget::Remote(caller.id())]
    );
    let (_tx, rx) = watch::channel(targets);
    let manager = TunnelManager::start(
        receiver.clone(),
        channels.rpc,
        channels.http,
        channels.stage,
    )
    .await
    .unwrap();
    manager.set_http_ingress(rx, AffinityRouter::new());
    caller.start_accepting();
    receiver.start_accepting();
    caller
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    receiver.insert_test_peer(capable_peer).await;
    let body = format!(r#"{{"model":"{INTERNAL}","input":"hello"}}"#);
    let raw = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = send_duplicate_request(&caller, &receiver, raw.as_bytes(), false).await;
    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 503"));
    assert!(
        local_requests.try_recv().is_err(),
        "incompatible local backend must not execute"
    );
    assert!(
        caller_channels.http.try_recv().is_err(),
        "peer ingress must not open another HTTP tunnel"
    );
    // No HTTP tunnel consumer is installed on caller: any attempted second hop
    // would stall until send_duplicate_request's bounded timeout, failing this test.
    caller.close_endpoint().await;
    receiver.close_endpoint().await;
    local_task.abort();
}
