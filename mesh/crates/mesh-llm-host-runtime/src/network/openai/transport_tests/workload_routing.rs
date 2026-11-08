use super::routing::test_peer_serving_model;
use super::*;
use crate::mesh::ModelWorkloadClass;
use crate::models::{CapabilityLevel, ModelCapabilities};
use crate::network::openai::workload_routing;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL: &str = "shared-workload-model";

#[tokio::test]
async fn document_ocr_requires_ocr_metadata_on_each_target() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    let legacy = peer(&node, None, false).await;
    let chat = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let ocr = peer(&node, Some(ModelWorkloadClass::DocumentOcr), false).await;
    let candidates = vec![
        election::InferenceTarget::Remote(legacy),
        election::InferenceTarget::Remote(chat),
        election::InferenceTarget::Remote(ocr),
    ];
    assert_eq!(
        workload_routing::eligible_targets(
            &node,
            MODEL,
            "/mesh/v1/documents/ocr?trace=1",
            &candidates
        )
        .await,
        vec![election::InferenceTarget::Remote(ocr)]
    );
}

#[tokio::test]
async fn ocr_plugin_does_not_authorize_a_native_port_with_same_model_name() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_served_model_descriptors(vec![descriptor(
        Some(ModelWorkloadClass::CausalGeneration),
        false,
    )])
    .await;
    let manager = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "onnx-ocr".into(),
            endpoint_id: "indra".into(),
            protocol: Some("document_ocr_v1".into()),
            address: "http://127.0.0.1:1".into(),
            models: vec![MODEL.into()],
        }])
        .await;
    node.set_plugin_manager(manager).await;
    let native = vec![election::InferenceTarget::Local(9337)];
    assert!(
        workload_routing::eligible_targets(&node, MODEL, "/mesh/v1/documents/ocr", &native)
            .await
            .is_empty()
    );
}

#[tokio::test]
/// An incompatible local copy must not shadow a valid same-model remote target.
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
/// Automatic routing draws its host over every placement: the local target
/// stays first and every capable peer follows, so a request that reached the
/// slow host can still land on a faster peer; the named-model contract stays
/// local-first.
async fn mesh_candidates_cover_local_and_remote_placements() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let remote = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    node.set_served_model_descriptors(vec![descriptor(
        Some(ModelWorkloadClass::CausalGeneration),
        false,
    )])
    .await;
    let mut targets = election::ModelTargets::default();
    targets
        .targets
        .insert(MODEL.into(), vec![election::InferenceTarget::Local(9337)]);
    let path = "/v1/chat/completions";
    assert_eq!(
        workload_routing::mesh_candidates(&node, MODEL, path, &targets).await,
        vec![
            election::InferenceTarget::Local(9337),
            election::InferenceTarget::Remote(remote)
        ]
    );
    assert_eq!(
        workload_routing::ingress_candidates(&node, MODEL, path, &targets).await,
        vec![election::InferenceTarget::Local(9337)]
    );
}

#[tokio::test]
/// The live automatic path: ingress resolves the directive (`mesh`, the
/// deprecated `auto`, or an absent model) to the concrete model and writes
/// that name back into the request before host selection runs, so the name no
/// longer says the request was automatic. Host selection must still draw over
/// this node and its peers — from the automatic-route flag — or every
/// automatic request takes the named-model local-first path (lab, 2026-11-08:
/// 0 of 100 automatic requests left the origin node). The flag does not
/// outrank `local_execution_only`: a request this node must execute itself
/// keeps the peer out of the draw.
async fn automatic_request_keeps_mesh_candidates_after_model_rewrite() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let remote = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    node.set_served_model_descriptors(vec![descriptor(
        Some(ModelWorkloadClass::CausalGeneration),
        false,
    )])
    .await;
    let mut targets = election::ModelTargets::default();
    targets
        .targets
        .insert(MODEL.into(), vec![election::InferenceTarget::Local(9337)]);
    let path = "/v1/chat/completions";
    let local = vec![election::InferenceTarget::Local(9337)];
    let mesh_wide = vec![
        election::InferenceTarget::Local(9337),
        election::InferenceTarget::Remote(remote),
    ];
    for directive in [Some("mesh"), Some("auto"), None] {
        let mut automatic = request_with_model(path, directive);
        crate::network::openai::ingress::maybe_enable_auto_route_hooks(&mut automatic, Some(MODEL));
        assert_eq!(
            automatic.model_name.as_deref(),
            Some(MODEL),
            "{directive:?}: directive rewritten to the resolved model"
        );
        assert!(
            automatic.automatic_route,
            "{directive:?}: flagged automatic"
        );
        assert_eq!(
            super::super::route_model::execution_candidates(
                &node, MODEL, &automatic, &targets, false
            )
            .await,
            mesh_wide,
            "{directive:?}: automatic request draws over local and peer"
        );
        assert_eq!(
            super::super::route_model::execution_candidates(
                &node, MODEL, &automatic, &targets, true
            )
            .await,
            local,
            "{directive:?}: local-only execution keeps the peer out"
        );
    }
    let mut named = request(path, MODEL);
    crate::network::openai::ingress::maybe_enable_auto_route_hooks(&mut named, Some(MODEL));
    assert!(!named.automatic_route);
    assert_eq!(
        super::super::route_model::execution_candidates(&node, MODEL, &named, &targets, false)
            .await,
        local
    );
}

#[test]
/// A peer advertises the model this node serves under its public id, so the
/// automatic candidate list would hold the same model twice: the identity
/// (local targets) and the public id (peer targets). The public id folds into
/// the identity; a public id with no local identity in the list, and a second
/// distinct identity, stay separate candidates.
fn public_alias_of_a_local_model_folds_into_its_identity() {
    use super::durable_artifacts::duplicate_alias_repro::duplicate_descriptor;
    let identity = "Falcon-H1-1.5B-Instruct-Q4_K_M".to_string();
    let public = "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string();
    let other = "Qwen2.5-3B-Instruct-Q6_K".to_string();
    let descriptors = vec![duplicate_descriptor(&identity)];
    let both = vec![identity.clone(), public.clone(), other.clone()];
    assert!(workload_routing::is_public_alias_of_local_model(
        &public,
        &both,
        &descriptors
    ));
    assert!(!workload_routing::is_public_alias_of_local_model(
        &identity,
        &both,
        &descriptors
    ));
    assert!(!workload_routing::is_public_alias_of_local_model(
        &other,
        &both,
        &descriptors
    ));
    let public_only = vec![public.clone(), other];
    assert!(!workload_routing::is_public_alias_of_local_model(
        &public,
        &public_only,
        &descriptors
    ));
}

#[tokio::test]
/// The public id folds into the local identity only while every peer behind it
/// is also reachable under that identity. A peer whose runtime name differs
/// (A on this node, B on the peer, both public id P) is reachable under P
/// alone, so P stays a candidate and that placement is not lost.
async fn public_alias_stays_when_a_peer_is_reachable_only_under_it() {
    use super::durable_artifacts::duplicate_alias_repro::{duplicate_descriptor, duplicate_peer};
    let local = "Falcon-H1-1.5B-Instruct-Q4_K_M".to_string();
    let peer_runtime = "falcon-h1-1.5b-instruct-q4_k_m.gguf".to_string();
    let public = "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string();
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_hosted_models(vec![local.clone()]).await;
    let descriptors = vec![duplicate_descriptor(&local)];
    node.set_served_model_descriptors(descriptors.clone()).await;
    let models = vec![local.clone(), public.clone()];

    let same_identity = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("peer");
    node.insert_test_peer(duplicate_peer(same_identity.id(), &local))
        .await;
    assert_eq!(
        workload_routing::folded_public_aliases(&node, &models, &descriptors).await,
        vec![public.clone()],
        "a peer with the same runtime identity is reachable under it"
    );

    let other_identity = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("peer");
    node.insert_test_peer(duplicate_peer(other_identity.id(), &peer_runtime))
        .await;
    assert!(
        workload_routing::folded_public_aliases(&node, &models, &descriptors)
            .await
            .is_empty(),
        "a peer reachable only under the public id keeps it listed"
    );
}

#[tokio::test]
/// A user tag on stateless work must not create generation-session affinity.
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

/// Build a descriptor with independently selectable workload and audio capability.
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

/// Create an isolated peer carrying the exact descriptors needed by a routing case.
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

/// Construct a parsed endpoint request without invoking a native model.
fn request(path: &str, model: &str) -> BufferedHttpRequest {
    request_with_model(path, Some(model))
}

/// Like [`request`]; `None` leaves the `model` field out of the body, the
/// way a client that relies on automatic routing may send it.
fn request_with_model(path: &str, model: Option<&str>) -> BufferedHttpRequest {
    let mut body = serde_json::json!({
        "input": "workload routing regression",
        "user": "workload-session",
        "prompt_cache_key": "workload-cache",
    });
    if let Some(model) = model {
        body["model"] = model.into();
    }
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
        model_name: model.map(Into::into),
        request_object_request_ids: vec![],
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
        exchange_observation_id: None,
        automatic_route: false,
    }
}

#[tokio::test]
/// Admission must use the selected target's own descriptor rather than fleet-wide metadata.
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
/// Never combine incompatible local and remote declarations into invented support.
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
/// Workload filtering must preserve existing generation and control-plane behavior.
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
/// Filter unsupported passive-client targets before assigning routing state.
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
/// An explicit future class is not an absent legacy descriptor.
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
/// A workload filter that empties a live host set is transient routing state, not a client error.
async fn passive_plan_reports_no_hosts_when_the_workload_filter_empties_the_host_set() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("node");
    // Fleet-wide admission sees this peer's advertisement, but the resolver
    // cannot route to it, so the per-host filter empties the resolved set. That
    // is the same shape as a peer that vanishes between discovery and filtering.
    let mut advertiser = test_peer_serving_model(iroh::SecretKey::generate().public(), MODEL);
    advertiser.admitted = false;
    advertiser.served_model_descriptors =
        vec![descriptor(Some(ModelWorkloadClass::Embedding), false)];
    node.insert_test_peer(advertiser).await;
    let host = peer(&node, None, false).await;
    assert_eq!(node.hosts_for_model(MODEL).await, vec![host]);

    let mut request = request("/v1/embeddings", MODEL);
    assert!(matches!(
        build_mesh_request_plan(&node, &mut request, false, &AffinityRouter::new()).await,
        Err(MeshRequestFailure::NoHostsAvailable),
    ));
}

#[tokio::test]
/// Passive audio routing selects a capable model and preserves the binary upload.
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
/// A cached automatic choice for one endpoint cannot authorize another workload.
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
/// Workload and audio support must coexist on one target descriptor.
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
/// Fail closed when neither the local target nor any replica can serve the endpoint.
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
                local_execution_only: false,
                automatic_retry: None,
                required_tokens: None,
                affinity: &affinity,
                served_by_header: None,
                route_observer: OpenAiRouteObserver::default(),
                peer_capsule_id: None,
                served_by_node_id: None,
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

#[tokio::test]
/// Capacity counts this node when it serves the model, and drops paused and
/// operator-blocked peers; a client node contributes no local placement.
async fn placements_count_local_host_and_skip_paused_and_blocked_peers() {
    use crate::proto::node::InferenceAdmissionState;
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    let models = vec![MODEL.to_string()];
    let placements = node.placements_for_models(&models).await;
    assert_eq!(placements[MODEL], mesh::ModelPlacements::default());
    assert_eq!(
        placements[MODEL].count(),
        1,
        "no placement still gets one neutral draw"
    );

    // Local only.
    node.set_hosted_models(vec![MODEL.to_string()]).await;
    let placements = node.placements_for_models(&models).await;
    assert!(placements[MODEL].local && placements[MODEL].remote.is_empty());
    assert_eq!(placements[MODEL].count(), 1);

    // Local + remote.
    let remote = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let placements = node.placements_for_models(&models).await;
    assert!(placements[MODEL].local);
    assert_eq!(placements[MODEL].remote, vec![remote]);
    assert_eq!(placements[MODEL].count(), 2);

    // A paused peer and a deprioritized peer: only the deprioritized one counts.
    let paused_id = iroh::SecretKey::generate().public();
    let mut paused = test_peer_serving_model(paused_id, MODEL);
    paused.inference_admission_state = Some(InferenceAdmissionState::RemotePaused);
    node.insert_test_peer(paused).await;
    let deprioritized_id = iroh::SecretKey::generate().public();
    let mut deprioritized = test_peer_serving_model(deprioritized_id, MODEL);
    deprioritized.inference_admission_state = Some(InferenceAdmissionState::AcceptingDeprioritized);
    node.insert_test_peer(deprioritized).await;
    let placements = node.placements_for_models(&models).await;
    assert!(!placements[MODEL].remote.contains(&paused_id));
    assert!(placements[MODEL].remote.contains(&deprioritized_id));
    assert_eq!(placements[MODEL].count(), 3);

    // Blocking the first remote removes it.
    node.peer_blocks
        .block(
            &remote,
            crate::network::peer_blocks::BlockLength::UntilUndone,
            crate::network::peer_blocks::Requester::Operator,
            None,
            crate::network::peer_blocks::now_ms(),
        )
        .expect("block");
    let placements = node.placements_for_models(&models).await;
    assert!(!placements[MODEL].remote.contains(&remote));
    assert_eq!(placements[MODEL].count(), 2);

    // Remote only, as seen from a client node: no local placement even though
    // the client lists the model.
    let client = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("client");
    client.set_hosted_models(vec![MODEL.to_string()]).await;
    let remote = peer(&client, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let placements = client.placements_for_models(&models).await;
    assert!(!placements[MODEL].local);
    assert_eq!(placements[MODEL].remote, vec![remote]);
}

#[tokio::test]
/// The model draw weight sums one neutral weight per unmeasured placement, so a
/// model on this host plus one peer weighs twice a model on a single peer.
async fn routing_candidates_weight_local_plus_remote_as_two_placements() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_hosted_models(vec![MODEL.to_string()]).await;
    peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let lone = "lone-model";
    let lone_id = iroh::SecretKey::generate().public();
    node.insert_test_peer(test_peer_serving_model(lone_id, lone))
        .await;
    let models = vec![MODEL.to_string(), lone.to_string()];
    let candidates =
        workload_routing::routing_candidates(&node, &models, "/v1/chat/completions", &[]).await;
    let by_name = |name: &str| {
        candidates
            .iter()
            .find(|candidate| candidate.name == name)
            .unwrap_or_else(|| panic!("candidate {name}"))
            .clone()
    };
    assert_eq!(by_name(MODEL).replicas, 2);
    assert_eq!(by_name(lone).replicas, 1);
    assert!((by_name(MODEL).capacity - 2.0 * by_name(lone).capacity).abs() < 1e-9);
}

/// Record `samples` successful attempts of `tokens` completion tokens over one
/// second each, i.e. `tokens` tok/s, against `target`.
fn record_throughput(
    node: &mesh::Node,
    target: &election::InferenceTarget,
    tokens: u64,
    samples: usize,
) {
    for _ in 0..samples {
        node.record_inference_attempt(
            Some(MODEL),
            target,
            Duration::ZERO,
            Duration::from_secs(1),
            crate::network::metrics::AttemptOutcome::Success,
            Some(tokens),
        );
    }
}

#[tokio::test]
/// The local placement weighs only this host's own serving samples. Samples
/// this node recorded while routing to a peer belong to that peer's
/// placement; they neither inflate nor dilute the local weight.
async fn local_placement_weight_uses_local_target_evidence_only() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("node");
    node.set_hosted_models(vec![MODEL.to_string()]).await;
    let remote = peer(&node, Some(ModelWorkloadClass::CausalGeneration), false).await;
    let models = vec![MODEL.to_string()];
    let placement = node.placements_for_models(&models).await[MODEL].clone();
    assert!(placement.local && placement.remote == vec![remote]);

    // Only the remote has samples: the local host stays at the neutral cold
    // weight instead of borrowing the peer's speed.
    record_throughput(&node, &election::InferenceTarget::Remote(remote), 80, 3);
    let capacity = workload_routing::model_capacity(&node, MODEL, &placement).await;
    let neutral = crate::network::router::placement_weight(None, 0);
    assert!(
        (capacity - (neutral + 80.0)).abs() < 1e-9,
        "remote-only samples: local must stay neutral, got {capacity}"
    );

    // Slow local host, fast peer, more peer samples than local ones: each
    // placement weighs its own measured speed. The model-wide aggregate
    // ((3·10 + 6·80) / 9 ≈ 56.7 per placement) would weigh the pair ~113.
    record_throughput(&node, &election::InferenceTarget::Local(9001), 10, 3);
    record_throughput(&node, &election::InferenceTarget::Remote(remote), 80, 3);
    let capacity = workload_routing::model_capacity(&node, MODEL, &placement).await;
    assert!(
        (capacity - (10.0 + 80.0)).abs() < 1e-9,
        "local 10 tok/s + remote 80 tok/s, got {capacity}"
    );

    // Local-only placement with a fast peer's samples on record for the same
    // model: the local weight is still this host's own 10 tok/s.
    let local_only = mesh::ModelPlacements {
        local: true,
        remote: Vec::new(),
    };
    let capacity = workload_routing::model_capacity(&node, MODEL, &local_only).await;
    assert!(
        (capacity - 10.0).abs() < 1e-9,
        "local-only weight must be this host's own speed, got {capacity}"
    );
}
