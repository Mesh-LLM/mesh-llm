//! Socket-level regressions for automatic model exhaustion handoff.
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn backend(
    status: u16,
    model: &'static str,
    calls: Arc<AtomicUsize>,
) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let n = stream.read(&mut buffer).await.unwrap();
            if n == 0 {
                return;
            }
            bytes.extend_from_slice(&buffer[..n]);
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n")
                && serde_json::from_slice::<serde_json::Value>(&bytes[end + 4..]).is_ok()
            {
                break;
            }
        }
        calls.fetch_add(1, Ordering::SeqCst);
        let end = bytes
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .unwrap();
        let request: serde_json::Value = serde_json::from_slice(&bytes[end + 4..]).unwrap();
        assert_eq!(request["model"], model);
        let body = serde_json::json!({"id":"test", "model":model, "choices":[]}).to_string();
        if status == 0 {
            std::future::pending::<()>().await;
        }
        let wire = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(wire.as_bytes()).await.unwrap();
    });
    (port, task)
}

async fn dispatch_fixture(status: u16, expected: u16, attempts: usize, automatic: bool) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let (a1, first) = backend(status, "model-a", calls.clone()).await;
    let (a2, second) = backend(status, "model-a", calls.clone()).await;
    let (b, third) = backend(200, "model-b", calls.clone()).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "model-a".into(),
        vec![
            election::InferenceTarget::Local(a1),
            election::InferenceTarget::Local(a2),
        ],
    );
    targets
        .targets
        .insert("model-b".into(), vec![election::InferenceTarget::Local(b)]);
    let affinity = affinity::AffinityRouter::new();
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        requested_by_node_id: None,
        automatic_retry: None,
        exchange_channel: None,
    };
    let body = r#"{"model":"model-a","messages":[{"role":"user","content":"hello"}]}"#;
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    client.write_all(wire.as_bytes()).await.unwrap();
    let (mut stream, _) = listener.accept().await.unwrap();
    let mut request = proxy::read_http_request(&mut stream).await.unwrap();
    let client = tokio::spawn(async move {
        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        response
    });
    let mut model = Some("model-a".into());
    let dispatch = async {
        if automatic {
            automatic_dispatch::dispatch(
                stream.into(),
                &mut request,
                &ctx,
                &mut model,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        } else {
            route_request(
                false,
                stream.into(),
                &mut request,
                &ctx,
                model.as_deref(),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        }
    };
    let outcome = crate::network::openai::response::TEST_RESPONSE_FIRST_BYTE_TIMEOUT
        .scope(std::time::Duration::from_millis(100), dispatch)
        .await;
    let response = tokio::time::timeout(std::time::Duration::from_secs(10), client)
        .await
        .unwrap()
        .unwrap();
    assert_dispatch_result(outcome, &response, model.as_deref(), expected);
    assert_eq!(calls.load(Ordering::SeqCst), attempts);
    first.abort();
    second.abort();
    third.abort();
    for task in [first, second, third] {
        if let Err(error) = task.await {
            assert!(error.is_cancelled(), "{error}");
        }
    }
    node.endpoint.close().await;
}

fn assert_dispatch_result(
    outcome: proxy::RouteDispatchOutcome,
    response: &str,
    model: Option<&str>,
    expected: u16,
) {
    assert_eq!(
        model_access_succeeded(outcome),
        expected == 200,
        "{outcome:?}"
    );
    assert!(
        response.starts_with(&format!("HTTP/1.1 {expected}")),
        "{response}"
    );
    assert_eq!(
        model,
        Some(if expected == 200 {
            "model-b"
        } else {
            "model-a"
        })
    );
}

#[tokio::test]
async fn automatic_exhaustion_spills_to_other_model_and_rewrites_body() {
    dispatch_fixture(429, 200, 3, true).await;
}

#[tokio::test]
async fn automatic_timeout_exhaustion_never_spills_to_another_model() {
    dispatch_fixture(0, 504, 2, true).await;
}

#[tokio::test]
async fn named_model_never_spills_to_another_model() {
    dispatch_fixture(429, 503, 2, false).await;
}

#[tokio::test]
async fn automatic_internal_server_error_stays_terminal() {
    dispatch_fixture(500, 500, 1, true).await;
}
