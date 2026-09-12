use super::routing::test_peer_serving_model;
use super::*;
use crate::mesh::ModelWorkloadClass;
use crate::models::{CapabilityLevel, ModelCapabilities};
use crate::network::openai::workload_routing;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL: &str = "shared-workload-model";

#[tokio::test]
async fn explicit_ingress_falls_back_to_capable_remote_when_local_workload_is_incompatible() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let remote = peer(&node, Some(ModelWorkloadClass::Embedding), false).await;
    for class in [None, Some(ModelWorkloadClass::CausalGeneration)] {
        node.set_served_model_descriptors(vec![descriptor(class, false)])
            .await;
        let mut targets = election::ModelTargets::default();
        targets
            .targets
            .insert(MODEL.into(), vec![election::InferenceTarget::Local(9337)]);
        let selected =
            workload_routing::ingress_candidates(&node, MODEL, "/v1/embeddings", &targets).await;
        assert_eq!(selected, vec![election::InferenceTarget::Remote(remote)]);
    }
}

#[tokio::test]
async fn stateless_user_metadata_does_not_disable_replica_reservation_spreading() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    for (path, class) in [
        ("/v1/embeddings", ModelWorkloadClass::Embedding),
        ("/v1/rerank", ModelWorkloadClass::Rerank),
        ("/v1/audio/speech", ModelWorkloadClass::SpeechSynthesis),
    ] {
        let first = peer(&node, Some(class), false).await;
        let second = peer(&node, Some(class), false).await;
        let affinity = AffinityRouter::new();
        let mut request = request(path, MODEL);
        let plan = build_mesh_request_plan(&node, &mut request, false, &affinity)
            .await
            .unwrap_or_else(|_| panic!("compatible replicas"));
        assert!(!plan.affinity_applied);
        assert_eq!(plan.equivalent_hosts, 2);
        assert!(plan.target_hosts.contains(&first) && plan.target_hosts.contains(&second));
        let (first_hosts, _first_reservation) = reserve_mesh_request_target(&plan, &affinity);
        let (second_hosts, _second_reservation) = reserve_mesh_request_target(&plan, &affinity);
        assert_ne!(
            first_hosts[0], second_hosts[0],
            "concurrent stateless requests must spread"
        );
    }
}

fn descriptor(class: Option<ModelWorkloadClass>, audio: bool) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: MODEL.into(),
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: class,
            ..Default::default()
        }),
        capabilities_known: true,
        capabilities: ModelCapabilities {
            audio: if audio {
                CapabilityLevel::Supported
            } else {
                CapabilityLevel::None
            },
            multimodal: audio,
            ..Default::default()
        },
        ..Default::default()
    }
}

async fn peer(
    node: &mesh::Node,
    class: Option<ModelWorkloadClass>,
    audio: bool,
) -> iroh::EndpointId {
    let id = iroh::SecretKey::generate().public();
    let mut peer = test_peer_serving_model(id, MODEL);
    peer.served_model_descriptors = vec![descriptor(class, audio)];
    node.insert_test_peer(peer).await;
    id
}

fn request(path: &str, model: &str) -> BufferedHttpRequest {
    let body = serde_json::json!({
        "model": model,
        "input": "workload routing regression",
        "user": "workload-session",
        "prompt_cache_key": "workload-cache",
    });
    let bytes = serde_json::to_vec(&body).expect("serialize body");
    BufferedHttpRequest {
        raw: format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", bytes.len(), body).into_bytes(),
        method: "POST".into(),
        path: path.into(),
        client_path: path.into(),
        request_id: RequestId::default(),
        body_json: Some(body),
        body_json_attempted: true,
        body_len_bytes: bytes.len(),
        body_bytes: Some(bytes),
        completion_tokens: None,
        stream: None,
        model_name: Some(model.into()),
        request_object_request_ids: vec![],
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    }
}

#[tokio::test]
async fn non_chat_targets_require_their_own_workload_advertisement() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let legacy = peer(&node, None, true).await;
    for (path, class, audio) in [
        ("/v1/embeddings", ModelWorkloadClass::Embedding, false),
        ("/v1/rerank", ModelWorkloadClass::Rerank, false),
        (
            "/v1/audio/speech",
            ModelWorkloadClass::SpeechSynthesis,
            false,
        ),
        (
            "/v1/audio/transcriptions",
            ModelWorkloadClass::CausalGeneration,
            true,
        ),
        (
            "/v1/audio/translations?trace=1",
            ModelWorkloadClass::CausalGeneration,
            true,
        ),
    ] {
        let capable = peer(&node, Some(class), audio).await;
        let candidates = vec![
            election::InferenceTarget::Remote(legacy),
            election::InferenceTarget::Remote(capable),
        ];
        assert_eq!(
            workload_routing::eligible_targets(&node, MODEL, path, &candidates).await,
            vec![election::InferenceTarget::Remote(capable)],
            "{path} must not inherit another peer's capability",
        );
    }
}

#[tokio::test]
async fn local_target_cannot_inherit_a_remote_workload_or_vice_versa() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let remote = peer(&node, Some(ModelWorkloadClass::Embedding), false).await;
    node.set_served_model_descriptors(vec![descriptor(None, false)])
        .await;
    let local = election::InferenceTarget::Local(9337);
    let remote_target = election::InferenceTarget::Remote(remote);
    let candidates = vec![local.clone(), remote_target.clone()];
    assert_eq!(
        workload_routing::eligible_targets(&node, MODEL, "/v1/embeddings", &candidates).await,
        vec![remote_target],
    );

    node.set_served_model_descriptors(vec![descriptor(Some(ModelWorkloadClass::Embedding), false)])
        .await;
    let mut legacy_peer = test_peer_serving_model(remote, MODEL);
    legacy_peer.served_model_descriptors = vec![descriptor(None, false)];
    node.insert_test_peer(legacy_peer).await;
    assert_eq!(
        workload_routing::eligible_targets(&node, MODEL, "/v1/embeddings", &candidates).await,
        vec![local],
    );
}

#[tokio::test]
async fn legacy_generation_and_control_routes_remain_eligible() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let legacy = peer(&node, None, true).await;
    let targets = vec![election::InferenceTarget::Remote(legacy)];
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/tokenize",
    ] {
        assert_eq!(
            workload_routing::eligible_targets(&node, MODEL, path, &targets).await,
            targets
        );
    }
    node.remove_test_peer(legacy).await;
    assert!(
        workload_routing::eligible_targets(&node, MODEL, "/v1/embeddings", &targets)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn passive_plan_excludes_legacy_hosts_before_affinity_and_reservation() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let legacy = peer(&node, None, false).await;
    let capable = peer(&node, Some(ModelWorkloadClass::Embedding), false).await;
    let affinity = AffinityRouter::new();
    let mut request = request("/v1/embeddings", MODEL);
    let prefix =
        crate::network::affinity::cache_prefix_hash(request.body_json.as_ref()).expect("prefix");
    affinity.remember_cache_lease_if_epoch(
        MODEL,
        prefix,
        &election::InferenceTarget::Remote(legacy),
        affinity.cache_lease_epoch(),
    );
    let plan = build_mesh_request_plan(&node, &mut request, false, &affinity)
        .await
        .unwrap_or_else(|_| panic!("capable peer must remain routable"));
    assert_eq!(plan.target_hosts, vec![capable]);
    assert_eq!(plan.equivalent_hosts, 1);
    let (hosts, _reservation) = reserve_mesh_request_target(&plan, &affinity);
    assert_eq!(hosts, vec![capable]);
}

#[tokio::test]
async fn passive_plan_rejects_unknown_workloads_instead_of_forwarding_them() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    peer(&node, None, true).await;
    let affinity = AffinityRouter::new();
    for path in ["/v1/embeddings", "/v1/audio/transcriptions"] {
        let mut request = request(path, MODEL);
        assert!(matches!(
            build_mesh_request_plan(&node, &mut request, false, &affinity).await,
            Err(MeshRequestFailure::UnsupportedWorkload),
        ));
    }
}

#[tokio::test]
async fn passive_auto_audio_uses_the_capable_descriptor_and_rewrites_multipart() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    peer(&node, None, false).await;
    let capable = peer(&node, Some(ModelWorkloadClass::CausalGeneration), true).await;
    // A local legacy descriptor deliberately precedes all current peer metadata.
    node.set_served_model_descriptors(vec![descriptor(None, false)])
        .await;
    let path = "/v1/audio/transcriptions";
    let mut request = request(path, "auto");
    let body = b"--fixture\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nauto\r\n--fixture\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\r\nRIFF\r\n--fixture--\r\n".to_vec();
    request.raw = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=fixture\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    request.raw.extend_from_slice(&body);
    request.body_len_bytes = body.len();
    request.body_bytes = Some(body);
    request.body_json = None;
    let plan = build_mesh_request_plan(&node, &mut request, false, &AffinityRouter::new())
        .await
        .unwrap_or_else(|_| panic!("current audio peer must be selected"));
    assert_eq!(plan.effective_model.as_deref(), Some(MODEL));
    assert_eq!(plan.target_hosts, vec![capable]);
    assert!(String::from_utf8_lossy(&request.raw).contains(&format!("\r\n\r\n{MODEL}\r\n")));
    assert!(String::from_utf8_lossy(&request.raw).contains("\r\n\r\nRIFF\r\n"));
}

#[tokio::test]
async fn passive_auto_model_cache_cannot_cross_workload_boundaries() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let capable = peer(&node, Some(ModelWorkloadClass::Embedding), false).await;
    let chat_id = iroh::SecretKey::generate().public();
    node.insert_test_peer(test_peer_serving_model(chat_id, "legacy-chat"))
        .await;
    let affinity = AffinityRouter::new();
    let mut request = request("/v1/embeddings", "auto");
    let key = crate::network::affinity::auto_model_session_key(request.body_json.as_ref())
        .expect("legacy cache key");
    assert_eq!(auto_session_key_for_request(&mut request, true), None);
    affinity.remember_auto_model(key, "legacy-chat");
    // No healthy compatible alternative: the availability fallback must still
    // stay inside the requested workload, never restore the cached chat model.
    affinity.record_target_outcome(
        Some(MODEL),
        &election::InferenceTarget::Remote(capable),
        TargetHealthOutcome::Unavailable,
    );
    let plan = build_mesh_request_plan(&node, &mut request, false, &affinity)
        .await
        .unwrap_or_else(|_| panic!("embedding fallback remains available"));
    assert_eq!(plan.effective_model.as_deref(), Some(MODEL));
    assert_eq!(plan.target_hosts, vec![capable]);
    assert_eq!(
        affinity.lookup_auto_model(key).as_deref(),
        Some("legacy-chat")
    );
}

#[tokio::test]
async fn audio_upload_capabilities_must_belong_to_one_descriptor_on_the_target() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let id = iroh::SecretKey::generate().public();
    let mut mixed = test_peer_serving_model(id, MODEL);
    mixed.served_model_descriptors = vec![
        descriptor(None, true),
        descriptor(Some(ModelWorkloadClass::CausalGeneration), false),
    ];
    node.insert_test_peer(mixed).await;
    let candidates = vec![election::InferenceTarget::Remote(id)];
    assert!(
        workload_routing::eligible_targets(&node, MODEL, "/v1/audio/transcriptions", &candidates)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn host_dispatch_rejects_local_legacy_target_without_capable_replicas() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_served_model_descriptors(vec![descriptor(None, false)])
        .await;
    let backend = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("backend");
    let backend_port = backend.local_addr().expect("address").port();
    let backend_task = tokio::spawn(async move {
        if let Ok(Ok((mut stream, _))) =
            tokio::time::timeout(Duration::from_secs(5), backend.accept()).await
        {
            let mut request_bytes = [0; 4096];
            let _ = stream.read(&mut request_bytes).await;
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                .await;
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("client listener");
    let (client, server) = tokio::join!(
        tokio::net::TcpStream::connect(listener.local_addr().expect("address")),
        listener.accept()
    );
    let mut client = client.expect("client");
    let (server, _) = server.expect("server");
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.into(),
        vec![election::InferenceTarget::Local(backend_port)],
    );
    let request = request("/v1/embeddings", MODEL);
    let affinity = AffinityRouter::new();
    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        route_model_request(
            node,
            server.into(),
            &targets,
            MODEL,
            &request,
            RouteModelRequestContext {
                required_tokens: None,
                affinity: &affinity,
                route_observer: OpenAiRouteObserver::default(),
            },
        ),
    )
    .await;
    backend_task.abort();
    let _ = backend_task.await;
    assert!(matches!(outcome, Ok(RouteDispatchOutcome::Responded(503))));
    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .await
        .expect("response");
    assert!(response.starts_with("HTTP/1.1 503"));
}
