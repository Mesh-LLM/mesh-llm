use super::*;
use crate::network::openai::client_stream::ResponseSafetyBuffer;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use base64::Engine;
use mesh_llm_plugin::image_safety::{
    GENERATION_GUARD_CAPABILITY, IMAGE_OPERATION, REQUEST_GUARD_CAPABILITY, STATUS_OPERATION,
    TEXT_OPERATION,
};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Default)]
struct StrictGuard(Mutex<Vec<String>>);

impl PluginRpcBridge for StrictGuard {
    fn handle_request(
        &self,
        _: String,
        method: String,
        params: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        assert_eq!(method, "tools/call");
        let params: serde_json::Value = serde_json::from_str(&params).unwrap();
        let operation = params["name"].as_str().unwrap().to_owned();
        self.0.lock().unwrap().push(operation.clone());
        Box::pin(async move {
            let value = if operation == STATUS_OPERATION {
                serde_json::json!({"protocol_version":1,"ready":true,"screen_received_images":true,"model":"test-nemotron","policy_revision":"test-policy"})
            } else {
                assert!(matches!(
                    operation.as_str(),
                    TEXT_OPERATION | IMAGE_OPERATION
                ));
                serde_json::json!({"decision":if operation == IMAGE_OPERATION {"block"} else {"allow"},"model":"test-nemotron","policy_revision":"test-policy","reason_code":"fixture_policy"})
            };
            let result = rmcp::model::CallToolResult::structured(value);
            Ok(RpcResult {
                result_json: serde_json::to_string(&result).unwrap(),
            })
        })
    }

    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

async fn install_guard(node: &Node, capabilities: &[&str]) -> Arc<StrictGuard> {
    let guard = Arc::new(StrictGuard::default());
    let manager = PluginManager::for_test_bridge(&["strict-get-images"], guard.clone());
    manager.set_test_capability_providers(
        capabilities
            .iter()
            .map(|capability| crate::plugin::PluginCapabilityProvider {
                capability: (*capability).into(),
                plugin_name: "strict-get-images".into(),
                plugin_status: "running".into(),
                endpoint_id: None,
                available: true,
                detail: None,
            })
            .collect(),
    );
    node.set_plugin_manager(manager).await;
    guard
}

async fn request(method: &str, path: &str, model: &str) -> transport::BufferedHttpRequest {
    let body = serde_json::json!({"model":model,"input":"hello","prompt":"hello","text":"hello","expected_identity":{"model_id":model},"messages":[{"role":"user","content":"hello"}]}).to_string();
    request_with_body(method, path, "application/json", body.as_bytes()).await
}

async fn request_with_body(
    method: &str,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> transport::BufferedHttpRequest {
    let mut raw = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    ).into_bytes();
    raw.extend_from_slice(body);
    let (mut writer, mut reader) = tokio::io::duplex(raw.len() + 128);
    writer.write_all(&raw).await.unwrap();
    transport::read_http_request(&mut reader).await.unwrap()
}

async fn selected_admission(
    node: &Node,
    method: &str,
    path: &str,
    model: &str,
    target: &InferenceTarget,
) -> (Option<transport::RouteDispatchOutcome>, bool) {
    let request = request(method, path, model).await;
    let (mut stream, quarantine) = ResponseSafetyBuffer::wrap(ClientStream::null());
    let result = crate::network::openai::exchange_admission::admit_selected_route(
        node,
        &mut stream,
        &request,
        Some(model),
        "mesh",
        SelectedRouteTarget::Mesh(target),
        1,
    )
    .await;
    (
        result,
        quarantine
            .capture_guard()
            .load(std::sync::atomic::Ordering::Acquire),
    )
}

#[tokio::test]
async fn missing_image_guard_rejects_candidate_even_when_fallback_is_available() {
    use crate::network::openai::exchange_admission::{
        SelectedRouteAdmission, admit_selected_candidate,
    };

    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_served_model_descriptors(vec![ServedModelDescriptor {
        identity: crate::mesh::ServedModelIdentity {
            model_name: "image-model".into(),
            ..Default::default()
        },
        metadata: Some(crate::mesh::ServedModelMetadata {
            output_modalities: vec!["text".into(), "image".into()],
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let request = request("POST", "/v1/chat/completions", "image-model").await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (server, _) = listener.accept().await.unwrap();
    let (mut stream, quarantine) = ResponseSafetyBuffer::wrap(ClientStream::from(server));
    let target = InferenceTarget::Local(1);
    let admission = admit_selected_candidate(
        &node,
        &mut stream,
        &request,
        Some("image-model"),
        "mesh",
        SelectedRouteTarget::Mesh(&target),
        1,
        true,
    )
    .await;
    let outcome = match admission {
        SelectedRouteAdmission::Rejected(outcome) => outcome,
        SelectedRouteAdmission::Proceed | SelectedRouteAdmission::NextCandidate => {
            panic!("missing image guard must reject the request, not try another candidate")
        }
    };
    assert_eq!(
        outcome,
        transport::RouteDispatchOutcome::admission_rejected(503)
    );
    assert!(
        quarantine
            .capture_guard()
            .load(std::sync::atomic::Ordering::Acquire)
    );
    let _ = quarantine.finish(&node, outcome).await;
    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
    assert!(response.contains("\"code\":\"image_safety_unavailable\""));
    node.close_endpoint().await;
}

#[tokio::test]
async fn duplicate_model_is_rejected_even_with_a_selected_text_route() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_served_model_descriptors(vec![ServedModelDescriptor {
        identity: crate::mesh::ServedModelIdentity {
            model_name: "image-model".into(),
            ..Default::default()
        },
        metadata: Some(crate::mesh::ServedModelMetadata {
            output_modalities: vec!["text".into(), "image".into()],
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let manager = PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "text-provider".into(),
            endpoint_id: "inference".into(),
            protocol: Some("openai_compatible".into()),
            address: "http://127.0.0.1:1".into(),
            models: vec!["text-model".into()],
        }])
        .await;
    node.set_plugin_manager(manager).await;
    let local = InferenceTarget::Local(1);
    for target in [
        SelectedRouteTarget::Mesh(&local),
        SelectedRouteTarget::Url("http://127.0.0.1:1"),
    ] {
        assert!(!selected_outputs_images(&node, Some("text-model"), &target).await);
        for path in ["/v1/chat/completions", "/provider/custom", "/v1/embeddings"] {
            let body = br#"{"model":"image-model","model":"text-model","messages":[{"role":"user","content":"hello"}]}"#;
            let request = request_with_body("POST", path, "application/json", body).await;
            assert!(request.model_name.is_none());
            assert_eq!(request.effective_http_entity().unwrap().as_ref(), body);
            let (mut stream, quarantine) = ResponseSafetyBuffer::wrap(ClientStream::null());
            assert_eq!(
                admit(&node, &mut stream, &request, Some("text-model"), &target).await,
                Err(image_safety::ImageSafetyResponseError::Unavailable),
                "{path}"
            );
            assert!(
                quarantine
                    .capture_guard()
                    .load(std::sync::atomic::Ordering::Acquire)
            );
        }
    }
    node.close_endpoint().await;
}

#[tokio::test]
async fn non_json_nonimage_workloads_keep_the_existing_transport() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let local = InferenceTarget::Local(1);
    for (path, content_type, body) in [
        (
            "/v1/audio/transcriptions",
            "audio/wav",
            b"RIFF\0\xffWAVE".as_slice(),
        ),
        ("/v1/ocr", "image/jpeg", b"\xff\xd8\xff\xe0".as_slice()),
        (
            "/provider/custom",
            "application/octet-stream",
            b"binary\0payload".as_slice(),
        ),
    ] {
        let request = request_with_body("POST", path, content_type, body).await;
        let (mut stream, quarantine) = ResponseSafetyBuffer::wrap(ClientStream::null());
        assert_eq!(
            admit(
                &node,
                &mut stream,
                &request,
                Some("text-model"),
                &SelectedRouteTarget::Mesh(&local)
            )
            .await,
            Ok(())
        );
        assert!(
            !quarantine
                .capture_guard()
                .load(std::sync::atomic::Ordering::Acquire)
        );
        assert_eq!(request.effective_http_entity().unwrap().as_ref(), body);
    }
    node.close_endpoint().await;
}

#[tokio::test]
async fn selected_image_model_rejects_unsupported_paths_before_guard_or_backend_dispatch() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_served_model_descriptors(vec![ServedModelDescriptor {
        identity: crate::mesh::ServedModelIdentity {
            model_name: "image-model".into(),
            ..Default::default()
        },
        metadata: Some(crate::mesh::ServedModelMetadata {
            output_modalities: vec!["text".into(), "image".into()],
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let guard = install_guard(
        &node,
        &[REQUEST_GUARD_CAPABILITY, GENERATION_GUARD_CAPABILITY],
    )
    .await;
    for (method, path) in [
        ("POST", "/v1/images/generations"),
        ("POST", "/provider/custom"),
        ("POST", "/v1/chat/completions/"),
        ("GET", "/v1/chat/completions?prompt=hello"),
        ("PUT", "/v1/responses"),
        ("POST", "/v1/embeddings"),
    ] {
        let (outcome, private) = selected_admission(
            &node,
            method,
            path,
            "image-model",
            &InferenceTarget::Local(1),
        )
        .await;
        assert_eq!(
            outcome,
            Some(transport::RouteDispatchOutcome::RequiredHookFailed),
            "{method} {path}"
        );
        assert!(private, "{method} {path} must suppress raw capture");
    }
    assert!(
        guard
            .0
            .lock()
            .unwrap()
            .iter()
            .all(|name| name == STATUS_OPERATION)
    );
    node.close_endpoint().await;
}

#[tokio::test]
async fn strict_requester_rejects_unknown_image_paths_despite_text_only_advertisement() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .unwrap();
    let remote = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let peer = text_only_peer(&node, &remote).await;
    let guard = install_guard(&node, &[REQUEST_GUARD_CAPABILITY]).await;
    for (method, path) in [
        ("POST", "/v1/images/generations"),
        ("POST", "/provider/custom"),
        ("GET", "/v1/chat/completions?prompt=hello"),
    ] {
        let (outcome, private) = selected_admission(
            &node,
            method,
            path,
            "text-model",
            &InferenceTarget::Remote(peer),
        )
        .await;
        assert_eq!(
            outcome,
            Some(transport::RouteDispatchOutcome::RequiredHookFailed),
            "{method} {path}"
        );
        assert!(private, "{method} {path} must suppress raw capture");
    }
    assert!(
        guard
            .0
            .lock()
            .unwrap()
            .iter()
            .all(|name| name == STATUS_OPERATION)
    );
    node.close_endpoint().await;
    remote.close_endpoint().await;
}

#[tokio::test]
async fn strict_requester_preserves_nonimage_workloads_and_token_counting() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .unwrap();
    let remote = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let peer = text_only_peer(&node, &remote).await;
    let guard = install_guard(&node, &[REQUEST_GUARD_CAPABILITY]).await;
    for path in [
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/mesh/v1/documents/ocr",
        "/v1/tokenize",
        "/v1/messages/count_tokens",
    ] {
        let (outcome, private) = selected_admission(
            &node,
            "POST",
            path,
            "text-model",
            &InferenceTarget::Remote(peer),
        )
        .await;
        assert_eq!(outcome, None, "{path}");
        assert!(!private, "{path} keeps its existing transport");
    }
    assert!(guard.0.lock().unwrap().is_empty());
    node.close_endpoint().await;
    remote.close_endpoint().await;
}

async fn text_only_peer(node: &Node, remote: &Node) -> iroh::EndpointId {
    remote
        .set_served_model_descriptors(vec![ServedModelDescriptor {
            identity: crate::mesh::ServedModelIdentity {
                model_name: "text-model".into(),
                ..Default::default()
            },
            metadata: Some(crate::mesh::ServedModelMetadata {
                output_modalities: vec!["text".into()],
                ..Default::default()
            }),
            ..Default::default()
        }])
        .await;
    let mut announcement =
        remote.build_local_announcement(remote.snapshot_local_announcement_data().await);
    announcement.role = crate::mesh::NodeRole::Host { http_port: 9337 };
    announcement.serving_models = vec!["text-model".into()];
    let mut peer = crate::mesh::PeerInfo::from_announcement(
        remote.id(),
        remote.endpoint.addr(),
        &announcement,
        crate::crypto::OwnershipSummary::default(),
    );
    peer.admitted = true;
    node.insert_test_peer(peer).await;
    remote.id()
}

#[tokio::test]
async fn accepted_legacy_text_endpoint_streams_but_vanished_selected_endpoint_is_guarded() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let manager = PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "text-provider".into(),
            endpoint_id: "inference".into(),
            protocol: Some("openai_compatible".into()),
            address: "http://127.0.0.1:1".into(),
            models: vec!["legacy-text".into()],
        }])
        .await;
    node.set_plugin_manager(manager).await;
    assert!(
        !selected_outputs_images(
            &node,
            Some("legacy-text"),
            &SelectedRouteTarget::Url("http://127.0.0.1:1")
        )
        .await
    );
    assert!(
        selected_outputs_images(
            &node,
            Some("legacy-text"),
            &SelectedRouteTarget::Url("http://127.0.0.1:2")
        )
        .await
    );
    node.close_endpoint().await;
}

#[tokio::test]
async fn independent_client_blocks_images_even_when_peer_advertises_only_text() {
    let node = Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .unwrap();
    let remote = Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let peer = text_only_peer(&node, &remote).await;
    let guard = Arc::new(StrictGuard::default());
    let manager = PluginManager::for_test_bridge(&["strict-get-images"], guard.clone());
    manager.set_test_capability_providers(vec![crate::plugin::PluginCapabilityProvider {
        capability: REQUEST_GUARD_CAPABILITY.into(),
        plugin_name: "strict-get-images".into(),
        plugin_status: "running".into(),
        endpoint_id: None,
        available: true,
        detail: None,
    }]);
    node.set_plugin_manager(manager).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (server, _) = listener.accept().await.unwrap();
    let body = r#"{"model":"text-model","messages":[{"role":"user","content":"hello"}]}"#;
    client.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
    let mut server = ClientStream::from(server);
    let request = transport::read_http_request(&mut server).await.unwrap();
    let (mut server, quarantine) = ResponseSafetyBuffer::wrap(server);
    admit(
        &node,
        &mut server,
        &request,
        Some("text-model"),
        &SelectedRouteTarget::Mesh(&InferenceTarget::Remote(peer)),
    )
    .await
    .unwrap();
    assert!(
        quarantine
            .capture_guard()
            .load(std::sync::atomic::Ordering::Acquire)
    );

    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(2, 2)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let artifact = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    );
    let body = serde_json::json!({"choices":[{"index":0,"message":{"role":"assistant","content":format!("![fixture]({artifact})")},"finish_reason":"stop"}]}).to_string();
    server.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
    server.shutdown().await.unwrap();
    let outcome = quarantine
        .finish(&node, transport::RouteDispatchOutcome::Responded(200))
        .await;
    assert_eq!(
        outcome,
        transport::RouteDispatchOutcome::admission_rejected(403)
    );
    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
    assert!(!response.contains(&artifact));
    {
        let calls = guard.0.lock().unwrap();
        assert!(calls.iter().any(|name| name == TEXT_OPERATION));
        assert!(calls.iter().any(|name| name == IMAGE_OPERATION));
    }
    node.close_endpoint().await;
    remote.close_endpoint().await;
}
