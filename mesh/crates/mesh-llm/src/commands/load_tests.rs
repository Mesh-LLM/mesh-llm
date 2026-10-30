use super::runtime::{LoadSizing, send_load_request};
use serde_json::json;
use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TEST_TOKEN: &str = "load-test-secret";
const TEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Returns the request head (lowercased) and body.
async fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut request = Vec::new();
    let mut chunk = [0u8; 1024];
    let header_end = loop {
        let read = stream.read(&mut chunk).await.expect("read request");
        assert!(read > 0, "client closed before sending a request");
        request.extend_from_slice(&chunk[..read]);
        if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let head = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
    let content_length = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .map(|value| value.trim().parse::<usize>().expect("content length"))
        .unwrap_or(0);
    while request.len() < header_end + content_length {
        let read = stream.read(&mut chunk).await.expect("read body");
        assert!(read > 0, "client closed mid-body");
        request.extend_from_slice(&chunk[..read]);
    }
    (
        head,
        request[header_end..header_end + content_length].to_vec(),
    )
}

async fn spawn_loading_node() -> (u16, tokio::task::JoinHandle<serde_json::Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let (head, body) = read_request(&mut stream).await;
        assert!(
            head.contains(&format!("authorization: bearer {TEST_TOKEN}")),
            "load must present the owner-control credential: {head}"
        );
        let reply = r#"{"loaded":"org/model","instance_id":"runtime-1"}"#;
        let response = format!(
            "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}",
            reply.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        serde_json::from_slice(&body).expect("load body is JSON")
    });
    (port, server)
}

#[tokio::test]
async fn load_sends_requested_context_size_and_lanes() {
    let (port, server) = spawn_loading_node().await;
    let sizing = LoadSizing::from_flags(NonZeroU32::new(8192), NonZeroUsize::new(1));

    send_load_request("org/model", port, sizing, TEST_TOKEN, TEST_TIMEOUT)
        .await
        .expect("load should succeed");

    assert_eq!(
        server.await.expect("node task"),
        json!({"model": "org/model", "ctx_size": 8192, "parallel": 1})
    );
}

#[tokio::test]
async fn load_without_sizing_flags_sends_only_the_model() {
    let (port, server) = spawn_loading_node().await;

    send_load_request(
        "org/model",
        port,
        LoadSizing::default(),
        TEST_TOKEN,
        TEST_TIMEOUT,
    )
    .await
    .expect("load should succeed");

    assert_eq!(
        server.await.expect("node task"),
        json!({"model": "org/model"})
    );
}

#[tokio::test]
async fn load_that_outlasts_the_client_timeout_reports_uncertain_status() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let node = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.expect("accept");
        std::future::pending::<()>().await;
    });

    let error = tokio::time::timeout(
        Duration::from_secs(10),
        send_load_request(
            "org/model",
            port,
            LoadSizing::default(),
            TEST_TOKEN,
            Duration::from_millis(200),
        ),
    )
    .await
    .expect("the client timeout must end the load request")
    .expect_err("an unanswered load should fail the client");
    node.abort();

    let text = format!("{error:#}");
    assert!(text.contains("may have been accepted"), "{text}");
    assert!(text.contains("may still be running"), "{text}");
    assert!(
        text.contains(&format!("mesh-llm status --port {port}")),
        "{text}"
    );
    assert!(!text.contains("Can't connect"), "{text}");
}

#[tokio::test]
async fn load_against_a_closed_port_reports_cannot_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);

    let error = send_load_request(
        "org/model",
        port,
        LoadSizing::default(),
        TEST_TOKEN,
        TEST_TIMEOUT,
    )
    .await
    .expect_err("a closed port should fail the client");
    let text = format!("{error:#}");
    assert!(text.contains("Can't connect"), "{text}");
}
