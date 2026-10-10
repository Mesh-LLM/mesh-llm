use super::*;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
    task::{JoinHandle, JoinSet},
};

pub(crate) enum Reply {
    Body(u16, Vec<u8>),
    Fragments(Vec<Vec<u8>>),
    Hold,
}
impl Reply {
    pub(crate) fn models() -> Self {
        Self::Body(200, br#"{"data":[{"id":"served-real-id"}]}"#.to_vec())
    }
    pub(crate) fn stream(tokens: u64) -> Self {
        Self::Fragments(vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n".to_vec(),
            format!("data: {{\"usage\":{{\"completion_tokens\":{tokens}}}}}\n\ndata: [DONE]\n\n")
                .into_bytes(),
        ])
    }
}
pub(crate) struct Fixture {
    pub(crate) port: u16,
    requests: mpsc::UnboundedReceiver<String>,
    task: JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    pub(crate) async fn new(replies: Vec<Reply>) -> Self {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let (send, requests) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            for reply in replies {
                let (socket, _) = listener.accept().await.unwrap();
                let send = send.clone();
                connections.spawn(async move {
                    serve(socket, reply, send).await;
                });
            }
            while connections.join_next().await.is_some() {}
        });
        Self {
            port,
            requests,
            task,
        }
    }
    pub(crate) async fn next(&mut self) -> String {
        tokio::time::timeout(Duration::from_secs(2), self.requests.recv())
            .await
            .unwrap()
            .unwrap()
    }
    pub(crate) fn empty(&mut self) -> bool {
        self.requests.try_recv().is_err()
    }
}
async fn serve(
    mut socket: tokio::net::TcpStream,
    reply: Reply,
    send: mpsc::UnboundedSender<String>,
) {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut byte = [0];
        if socket.read_exact(&mut byte).await.is_err() {
            return;
        }
        bytes.push(byte[0]);
        if bytes.ends_with(b"\r\n\r\n") {
            break bytes.len();
        }
        assert!(
            bytes.len() < 16 * 1024,
            "fixture request headers exceeded bound"
        );
    };
    let header = std::str::from_utf8(&bytes).unwrap();
    let length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length <= 64 * 1024);
    bytes.resize(header_end + length, 0);
    if socket.read_exact(&mut bytes[header_end..]).await.is_err() {
        return;
    }
    if send.send(String::from_utf8(bytes).unwrap()).is_err() {
        return;
    }
    match reply {
        Reply::Hold => std::future::pending::<()>().await,
        Reply::Body(status, body) => {
            let header = format!(
                "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            if socket.write_all(header.as_bytes()).await.is_err() {
                return;
            }
            let _ = socket.write_all(&body).await;
        }
        Reply::Fragments(chunks) => {
            if socket.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").await.is_err() { return; }
            for chunk in chunks {
                let framed = format!("{:x}\r\n", chunk.len());
                if socket.write_all(framed.as_bytes()).await.is_err()
                    || socket.write_all(&chunk).await.is_err()
                    || socket.write_all(b"\r\n").await.is_err()
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        }
    }
}
pub(crate) fn body(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}
async fn call(fixture: &Fixture) -> DynResult<Measurement> {
    tokio::time::timeout(
        Duration::from_secs(2),
        request(fixture.port, &json!({"model":"served-real-id"})),
    )
    .await
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn loopback_post_preserves_body_and_minimal_usage() {
    let mut fixture = Fixture::new(vec![Reply::stream(7)]).await;
    let result = call(&fixture).await.unwrap();
    let sent = fixture.next().await;
    assert!(sent.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    assert_eq!(body(&sent), json!({"model":"served-real-id"}));
    assert_eq!(result.completion_tokens, Some(7));
    assert!(result.ttft_ms.is_some());
    assert!(result.decode_tok_s.is_some());
    assert!(!result.malformed);
}
#[tokio::test(flavor = "current_thread")]
async fn fragmented_utf8_and_sse_are_reassembled() {
    let all = "data: {\"choices\":[{\"delta\":{\"content\":\"é\"}}]}\n\ndata: {\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n";
    let fragments = all.as_bytes().chunks(1).map(<[u8]>::to_vec).collect();
    let fixture = Fixture::new(vec![Reply::Fragments(fragments)]).await;
    let result = call(&fixture).await.unwrap();
    assert_eq!(result.completion_tokens, Some(3));
    assert!(result.ttft_ms.is_some());
}
#[tokio::test(flavor = "current_thread")]
async fn http_failure_has_no_success_metrics() {
    let fixture = Fixture::new(vec![Reply::Body(503, b"unavailable".to_vec())]).await;
    assert!(
        call(&fixture)
            .await
            .unwrap_err()
            .to_string()
            .contains("503")
    );
}
#[tokio::test(flavor = "current_thread")]
async fn redirect_is_refused_instead_of_followed() {
    let fixture = Fixture::new(vec![Reply::Body(302, b"remote redirect".to_vec())]).await;
    assert!(
        call(&fixture)
            .await
            .unwrap_err()
            .to_string()
            .contains("302")
    );
}
#[tokio::test(flavor = "current_thread")]
async fn server_error_clears_previous_usage() {
    let fixture = Fixture::new(vec![Reply::Fragments(vec![b"data: {\"usage\":{\"completion_tokens\":9}}\n\ndata: {\"error\":{\"message\":\"failed\"}}\n\n".to_vec()])]).await;
    let result = call(&fixture).await.unwrap();
    assert!(result.malformed);
    assert_eq!(result.completion_tokens, None);
    assert_eq!(result.decode_tok_s, None);
}
#[tokio::test(flavor = "current_thread")]
async fn missing_usage_marks_response_malformed() {
    let fixture = Fixture::new(vec![Reply::Fragments(vec![
        b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n".to_vec(),
    ])])
    .await;
    let result = call(&fixture).await.unwrap();
    assert!(result.malformed);
    assert_eq!(result.ttft_ms, None);
}
#[tokio::test(flavor = "current_thread")]
async fn zero_port_cannot_admit_network_request() {
    assert!(
        request(0, &json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("nonzero")
    );
}
