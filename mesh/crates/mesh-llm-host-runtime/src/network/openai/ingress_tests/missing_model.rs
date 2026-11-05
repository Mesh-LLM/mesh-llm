//! Status lines for a request naming a model this node does not serve yet:
//! retryable 503 while it loads, the exclusion 409, the plugin route.
use super::tests::{empty_test_plugin_manager, plugin_only_request, spawn_answering_chat_backend};
use super::*;
use mesh_llm_events::logging::identifiers::RequestId;

/// Status line `route_missing_local_model` writes for a chat request naming
/// `model` on a node with no targets, no peer and no plugin manager.
async fn missing_local_model_status(
    node: &mesh::Node,
    model: &str,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
) -> String {
    missing_local_model_response(node, model, target, excluded)
        .await
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// The whole response behind `missing_local_model_status`.
async fn missing_local_model_response(
    node: &mesh::Node,
    model: &str,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
) -> String {
    missing_local_model_response_with_targets(
        node,
        &election::ModelTargets::default(),
        model,
        target,
        excluded,
    )
    .await
}

/// The same, with this node's election targets given.
async fn missing_local_model_response_with_targets(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    model: &str,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
) -> String {
    use tokio::io::AsyncReadExt;

    let affinity = affinity::AffinityRouter::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let (client_side, server_side) = tokio::join!(tokio::net::TcpStream::connect(addr), async {
        listener.accept().await.map(|(stream, _)| stream)
    });
    let mut client_side = client_side.expect("connect client side");
    let mut request = proxy::BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 0,
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
        exchange_observation_id: None,
    };
    let ctx = IngressRouteContext {
        node,
        targets,
        affinity: &affinity,
        plugin_manager: None,
        requested_by_node_id: None,
        exchange_channel: None,
    };
    let lifecycle = OpenAiLifecycleAttachment::unowned();
    route_missing_local_model(
        server_side.expect("accept server side").into(),
        &mut request,
        &ctx,
        model,
        target,
        excluded,
        None,
        lifecycle.route_observer(),
    )
    .await;

    let mut response = Vec::new();
    client_side
        .read_to_end(&mut response)
        .await
        .expect("read response");
    String::from_utf8_lossy(&response).into_owned()
}

/// Headers and the OpenAI error `code` of a raw response.
fn retry_after_and_error_code(response: &str) -> (Option<&str>, String) {
    let (head, body) = response.split_once("\r\n\r\n").expect("header end");
    let retry_after = head
        .lines()
        .find_map(|line| line.strip_prefix("Retry-After: "));
    let body: serde_json::Value = serde_json::from_str(body).expect("JSON error body");
    let code = body["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    (retry_after, code)
}

/// A model's first load has a serving assignment but no target yet. Naming it
/// meanwhile is retryable, not "not found".
#[tokio::test]
async fn route_missing_local_model_answers_503_while_the_first_load_runs() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_serving_models(vec!["loading-model".into()]).await;

    assert_eq!(
        missing_local_model_status(&node, "loading-model", None, &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "unknown-model", None, &[]).await,
        "HTTP/1.1 404 Not Found"
    );
    node.endpoint.close().await;
}

/// A plugin endpoint serving the same name keeps the plugin route it had
/// before the first-load 503 existed.
#[tokio::test]
async fn route_missing_local_model_prefers_a_plugin_serving_a_loading_name() {
    use tokio::io::AsyncReadExt;

    let model = "acme/plugin-model:Q4_K_M";
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    node.set_serving_models(vec![model.into()]).await;
    // Even while the startup models resolve.
    node.set_startup_resolution(mesh::StartupResolution::AnyName)
        .await;
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();
    let plugin_manager = empty_test_plugin_manager();
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: Some(&plugin_manager),
        requested_by_node_id: None,
        exchange_channel: None,
    };
    // Untargeted, and with `x-mesh-target` naming this node. The backend
    // answers one request, so each case gets its own.
    for target in [None, Some(node.id())] {
        let plugin_port = spawn_answering_chat_backend("from-plugin").await;
        plugin_manager
            .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
                plugin_name: "acme".to_string(),
                endpoint_id: "ep1".to_string(),
                protocol: Some("openai_compatible".to_string()),
                address: format!("http://127.0.0.1:{plugin_port}"),
                models: vec![model.to_string()],
            }])
            .await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (client_side, server_side) =
            tokio::join!(tokio::net::TcpStream::connect(addr), async {
                listener.accept().await.map(|(stream, _)| stream)
            });
        let mut client_side = client_side.expect("connect");
        let mut request = plugin_only_request(model);
        route_missing_local_model(
            server_side.expect("accept").into(),
            &mut request,
            &ctx,
            model,
            target,
            &[],
            None,
            OpenAiRouteObserver::default(),
        )
        .await;

        let mut response = Vec::new();
        client_side
            .read_to_end(&mut response)
            .await
            .expect("read response");
        let response_text = String::from_utf8_lossy(&response);
        assert!(
            response_text.contains("from-plugin"),
            "a plugin-served name must take the plugin route (target {target:?}), got: {response_text}"
        );
    }
    node.endpoint.close().await;
}

/// A client that excluded this node can't be served by its local load, so it
/// keeps the exclusion 409 rather than a 503 it would retry forever.
#[tokio::test]
async fn route_missing_local_model_keeps_the_exclusion_409_while_the_first_load_runs() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_serving_models(vec!["loading-model".into()]).await;

    assert_eq!(
        missing_local_model_status(&node, "loading-model", None, &[node.id()]).await,
        "HTTP/1.1 409 Conflict"
    );
    node.endpoint.close().await;
}

/// `x-mesh-target` naming this node takes `route_self_targeted_model`, which
/// must also answer the retryable 503 while the model loads here, and still
/// fail closed for a model this node doesn't serve at all.
#[tokio::test]
async fn route_self_targeted_model_answers_503_while_the_first_load_runs() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_serving_models(vec!["loading-model".into()]).await;

    assert_eq!(
        missing_local_model_status(&node, "loading-model", Some(node.id()), &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "unknown-model", Some(node.id()), &[]).await,
        "HTTP/1.1 409 Conflict"
    );
    node.endpoint.close().await;
}

/// A self-target contradicted by a self-exclude keeps its 409 while the model
/// loads: this node can never serve that client, so a 503 would loop.
#[tokio::test]
async fn route_self_targeted_model_keeps_the_exclusion_409_while_the_first_load_runs() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_serving_models(vec!["loading-model".into()]).await;

    assert_eq!(
        missing_local_model_status(&node, "loading-model", Some(node.id()), &[node.id()]).await,
        "HTTP/1.1 409 Conflict"
    );
    node.endpoint.close().await;
}

/// While a bare local GGUF resolves (it is hashed for its id), no name is
/// known yet, so any name may be one of them: retryable, untargeted or
/// self-targeted. Once they resolve, an unknown name fails as before.
#[tokio::test]
async fn route_missing_local_model_answers_503_while_the_startup_models_resolve() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_resolution(mesh::StartupResolution::AnyName)
        .await;

    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", None, &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", Some(node.id()), &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );

    node.set_startup_resolution(mesh::StartupResolution::Done)
        .await;
    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", None, &[]).await,
        "HTTP/1.1 404 Not Found"
    );
    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", Some(node.id()), &[]).await,
        "HTTP/1.1 409 Conflict"
    );
    node.endpoint.close().await;
}

/// When every startup model is named before it resolves, only those names wait
/// on it: another name still gets its 404 at once.
#[tokio::test]
async fn route_missing_local_model_answers_503_only_for_named_startup_models() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_resolution(mesh::StartupResolution::Names(vec![
        "Qwen3-8B-Q4_K_M".into(),
    ]))
    .await;

    assert_eq!(
        missing_local_model_status(&node, "Qwen3-8B-Q4_K_M", None, &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "Qwen3-8B-Q4_K_M", Some(node.id()), &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "Qwen3-8B-Typo", None, &[]).await,
        "HTTP/1.1 404 Not Found"
    );
    node.endpoint.close().await;
}

/// A client that excluded this node keeps its 409 while the startup models
/// resolve: this node can never serve it.
#[tokio::test]
async fn route_missing_local_model_keeps_the_exclusion_409_while_the_startup_models_resolve() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_resolution(mesh::StartupResolution::AnyName)
        .await;

    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", None, &[node.id()]).await,
        "HTTP/1.1 409 Conflict"
    );
    assert_eq!(
        missing_local_model_status(
            &node,
            "local-gguf/sha256-abc",
            Some(node.id()),
            &[node.id()]
        )
        .await,
        "HTTP/1.1 409 Conflict"
    );
    node.endpoint.close().await;
}

/// A 503 for a model still loading here tells the client when to retry and
/// why, untargeted and self-targeted, with the interval the docs state. A
/// name still unknown once the startup models resolve keeps its plain 404.
#[tokio::test]
async fn route_missing_local_model_tells_clients_when_to_retry_a_loading_model() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_resolution(mesh::StartupResolution::AnyName)
        .await;
    for target in [None, Some(node.id())] {
        let response =
            missing_local_model_response(&node, "local-gguf/sha256-abc", target, &[]).await;
        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
            "{response}"
        );
        assert_eq!(
            retry_after_and_error_code(&response),
            (Some("5"), "model_not_ready".to_owned()),
            "target {target:?}"
        );
    }

    node.set_startup_resolution(mesh::StartupResolution::Done)
        .await;
    let response = missing_local_model_response(&node, "local-gguf/sha256-abc", None, &[]).await;
    assert!(
        response.starts_with("HTTP/1.1 404 Not Found\r\n"),
        "{response}"
    );
    assert_eq!(
        retry_after_and_error_code(&response),
        (None, "model_not_found".to_owned())
    );
    node.endpoint.close().await;
}

/// `auto` with no model ready yet answers the same retryable 503.
#[tokio::test]
async fn auto_route_with_no_model_ready_tells_clients_when_to_retry() {
    use tokio::io::AsyncReadExt;

    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let (client_side, server_side) = tokio::join!(tokio::net::TcpStream::connect(addr), async {
        listener.accept().await.map(|(stream, _)| stream)
    });
    let mut client_side = client_side.expect("connect client side");
    send_auto_route_rejection(
        server_side.expect("accept server side").into(),
        AutoRouteRejection::NoModelReady,
        &node,
        &[],
        "/v1/chat/completions",
        OpenAiRouteObserver::default(),
    )
    .await;

    let mut response = Vec::new();
    client_side
        .read_to_end(&mut response)
        .await
        .expect("read response");
    let response = String::from_utf8_lossy(&response);
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
        "{response}"
    );
    assert_eq!(
        retry_after_and_error_code(&response),
        (Some("5"), "model_not_ready".to_owned())
    );
    node.endpoint.close().await;
}

/// A model whose local copies are all down (draining, reloading) gives a
/// client that excluded this node the exclusion 409, like the other loading
/// cases: the retryable 503 is only for clients this node could serve.
#[tokio::test]
async fn route_missing_local_model_keeps_the_exclusion_409_when_local_copies_are_down() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "draining-model".to_string(),
        vec![election::InferenceTarget::None],
    );

    let excluded = missing_local_model_response_with_targets(
        &node,
        &targets,
        "draining-model",
        None,
        &[node.id()],
    )
    .await;
    assert!(
        excluded.starts_with("HTTP/1.1 409 Conflict\r\n"),
        "{excluded}"
    );

    let retryable =
        missing_local_model_response_with_targets(&node, &targets, "draining-model", None, &[])
            .await;
    assert!(
        retryable.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
        "{retryable}"
    );
    assert_eq!(
        retry_after_and_error_code(&retryable),
        (Some("5"), "model_not_ready".to_owned())
    );
    node.endpoint.close().await;
}
