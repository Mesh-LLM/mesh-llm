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
    use tokio::io::AsyncReadExt;

    let targets = election::ModelTargets::default();
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
        targets: &targets,
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
    String::from_utf8_lossy(&response)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned()
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
    node.set_startup_models_resolving(true).await;
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

/// While the startup models resolve (a local GGUF is hashed for its id), no
/// name is known yet, so any name may be one of them: retryable, untargeted or
/// self-targeted. Once they resolve, an unknown name fails as before.
#[tokio::test]
async fn route_missing_local_model_answers_503_while_the_startup_models_resolve() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_models_resolving(true).await;

    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", None, &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );
    assert_eq!(
        missing_local_model_status(&node, "local-gguf/sha256-abc", Some(node.id()), &[]).await,
        "HTTP/1.1 503 Service Unavailable"
    );

    node.set_startup_models_resolving(false).await;
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

/// A client that excluded this node keeps its 409 while the startup models
/// resolve: this node can never serve it.
#[tokio::test]
async fn route_missing_local_model_keeps_the_exclusion_409_while_the_startup_models_resolve() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_startup_models_resolving(true).await;

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
