use clap::Parser;
use mesh_llm_cli::Cli;
use serde_json::json;
use std::future::Future;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn run_on_application_stack<F, Fut>(test: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()>,
{
    std::thread::Builder::new()
        .stack_size(crate::DEFAULT_THREAD_STACK_SIZE)
        .spawn(move || {
            let mut builder = tokio::runtime::Builder::new_current_thread();
            builder.enable_all();
            builder.build().expect("test runtime").block_on(test())
        })
        .expect("spawn application-stack test thread")
        .join()
        .expect("application-stack test should not panic");
}

fn parse_cli(args: &[&str]) -> Cli {
    let normalized = mesh_llm_cli::normalize_runtime_surface_args(args.iter().copied());
    Cli::try_parse_from(normalized.normalized).expect("load arguments should parse")
}

async fn read_request_body(stream: &mut TcpStream) -> Vec<u8> {
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
    request[header_end..header_end + content_length].to_vec()
}

async fn spawn_loading_node() -> (u16, tokio::task::JoinHandle<serde_json::Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let body = read_request_body(&mut stream).await;
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

#[test]
fn load_sends_requested_context_size_and_lanes() {
    run_on_application_stack(|| async {
        let (port, server) = spawn_loading_node().await;
        let port_arg = port.to_string();
        let cli = parse_cli(&[
            "mesh-llm",
            "load",
            "org/model",
            "--port",
            &port_arg,
            "--ctx-size",
            "8192",
            "--parallel",
            "1",
        ]);

        super::dispatch(&cli).await.expect("load should succeed");

        assert_eq!(
            server.await.expect("node task"),
            json!({"model": "org/model", "ctx_size": 8192, "parallel": 1})
        );
    });
}

#[test]
fn load_without_sizing_flags_sends_only_the_model() {
    run_on_application_stack(|| async {
        let (port, server) = spawn_loading_node().await;
        let port_arg = port.to_string();
        let cli = parse_cli(&["mesh-llm", "load", "org/model", "--port", &port_arg]);

        super::dispatch(&cli).await.expect("load should succeed");

        assert_eq!(
            server.await.expect("node task"),
            json!({"model": "org/model"})
        );
    });
}

#[tokio::test]
async fn load_that_outlasts_the_client_timeout_reports_still_loading() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local addr").port();
    let node = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.expect("accept");
        std::future::pending::<()>().await;
    });

    let error = tokio::time::timeout(
        Duration::from_secs(10),
        super::runtime::run_load_with_timeout(
            "org/model",
            port,
            super::runtime::LoadSizing::default(),
            Duration::from_millis(200),
        ),
    )
    .await
    .expect("the client timeout must end the load request")
    .expect_err("an unanswered load should fail the client");
    node.abort();

    let text = format!("{error:#}");
    assert!(text.contains("still loading"), "{text}");
    assert!(!text.contains("Can't connect"), "{text}");
}

#[test]
fn load_against_a_closed_port_reports_cannot_connect() {
    run_on_application_stack(|| async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port_arg = listener
            .local_addr()
            .expect("local addr")
            .port()
            .to_string();
        drop(listener);
        let cli = parse_cli(&["mesh-llm", "load", "org/model", "--port", &port_arg]);

        let error = super::dispatch(&cli)
            .await
            .expect_err("a closed port should fail the client");
        let text = format!("{error:#}");
        assert!(text.contains("Can't connect"), "{text}");
    });
}
