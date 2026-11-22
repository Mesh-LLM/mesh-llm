use super::tests::empty_test_plugin_manager;
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn send_buffered_request(
    raw_request: &str,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> String {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let targets = election::ModelTargets::default();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ingress listener");
    let address = listener.local_addr().expect("ingress addr");

    let client_task = {
        let raw = raw_request.to_owned();
        tokio::spawn(async move {
            let mut client = TcpStream::connect(address).await.expect("connect ingress");
            client.write_all(raw.as_bytes()).await.expect("write raw");
            let mut response = String::new();
            client
                .read_to_string(&mut response)
                .await
                .expect("read response");
            response
        })
    };

    let (mut stream, _) = listener.accept().await.expect("accept client");
    let request = proxy::read_http_request(&mut stream)
        .await
        .expect("parse buffered request");
    let affinity = affinity::AffinityRouter::new();
    handle_buffered_api_request(
        stream.into(),
        request,
        ProxyConnectionContext {
            route: IngressRouteContext {
                node: &node,
                targets: &targets,
                affinity: &affinity,
                plugin_manager,
                requested_by_node_id: None,
                automatic_retry: None,
                exchange_channel: None,
            },
        },
        None,
        crate::runtime::IngressType::LocalOpenAi,
    )
    .await;

    client_task.await.expect("client task join")
}

#[test]
fn openai_ingress_backend_path_gate_admits_known_routes_and_rejects_unknown() {
    for (method, path) in [
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions?stream=true"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/rerank"),
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/audio/transcriptions"),
        ("POST", "/v1/audio/translations"),
        ("POST", "/v1/tokenize"),
        ("POST", "/v1/systemone"),
        ("POST", "/systemone"),
        ("POST", "/v1/decisions"),
        ("POST", "/mesh/v1/documents/ocr"),
        ("POST", "/mesh/v1/images/similarity"),
        ("POST", "/mesh/v1/images/depth"),
        ("GET", "/health"),
        ("HEAD", "/health"),
        ("GET", "/healthz"),
        ("GET", "/readyz"),
    ] {
        assert!(
            workload_routing::is_openai_ingress_backend_path(method, path),
            "expected admitted OpenAI ingress route: {method} {path}"
        );
    }

    for (method, path) in [
        ("POST", "/api/plugins/mesh-dev/tools/env_probe"),
        ("GET", "/api/status"),
        ("POST", "/api/objects"),
        ("POST", "/health"),
        ("POST", "/v1/models"),
        ("GET", "/does-not-exist"),
    ] {
        assert!(
            !workload_routing::is_openai_ingress_backend_path(method, path),
            "expected rejected OpenAI ingress route: {method} {path}"
        );
    }
}

#[tokio::test]
async fn unknown_api_post_returns_404_instead_of_auto_routing_to_plugin() {
    let plugin_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake plugin listener");
    let plugin_addr = plugin_listener.local_addr().expect("plugin addr");
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_bg = Arc::clone(&hits);
    let plugin_task = tokio::spawn(async move {
        if let Ok((_sock, _)) = plugin_listener.accept().await {
            hits_bg.fetch_add(1, Ordering::SeqCst);
        }
    });

    let manager = empty_test_plugin_manager();
    manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: "mesh-dev".into(),
            endpoint_id: "mesh-dev".into(),
            protocol: Some("openai_compatible".into()),
            address: format!("http://{plugin_addr}"),
            models: vec!["dev/mem-probe".into()],
        }])
        .await;

    let body = "{}";
    let raw = format!(
        "POST /api/plugins/mesh-dev/tools/env_probe HTTP/1.1\r\nHost: localhost:9337\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_buffered_request(&raw, Some(&manager)).await;
    plugin_task.abort();

    assert!(
        response.starts_with("HTTP/1.1 404"),
        "unknown /api/... POST on OpenAI ingress must return 404, got:\n{response}"
    );
    assert!(
        response.contains("\"code\":\"not_found\""),
        "404 body should report code=not_found, got:\n{response}"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "unknown /api/... POST must never be forwarded to a plugin inference endpoint"
    );
}

#[tokio::test]
async fn unknown_get_path_and_health_post_return_404() {
    for raw in [
        "GET /does-not-exist HTTP/1.1\r\nHost: localhost:9337\r\n\r\n",
        "POST /health HTTP/1.1\r\nHost: localhost:9337\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
    ] {
        let response = send_buffered_request(raw, None).await;
        assert!(
            response.starts_with("HTTP/1.1 404") && response.contains("\"code\":\"not_found\""),
            "expected 404 not_found, got:\n{response}"
        );
    }
}

#[tokio::test]
async fn bootstrap_mesh_request_returns_404_for_unknown_paths() {
    for raw in [
        "POST /api/plugins/mesh-dev/tools/env_probe HTTP/1.1\r\nHost: localhost:9337\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        "GET /does-not-exist HTTP/1.1\r\nHost: localhost:9337\r\n\r\n",
    ] {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .expect("test node");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind bootstrap ingress listener");
        let address = listener.local_addr().expect("bootstrap ingress addr");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept client");
            proxy::handle_mesh_request(node, stream.into(), false, affinity::AffinityRouter::new())
                .await;
        });

        let mut client = TcpStream::connect(address).await.expect("connect ingress");
        client.write_all(raw.as_bytes()).await.expect("write raw");
        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .await
            .expect("read response");
        server.await.expect("server task join");

        assert!(
            response.starts_with("HTTP/1.1 404") && response.contains("\"code\":\"not_found\""),
            "bootstrap proxy expected 404 not_found, got:\n{response}"
        );
    }
}
