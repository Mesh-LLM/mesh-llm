//! Finite actual recording, upstream transport and shutdown tests.
use super::{forwarding::Forwarder, listener, request_projection::Endpoint};
use crate::process::Cancellation;
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::automation::tls_fixture;

struct Proxy {
    directory: tempfile::TempDir,
    ready: PathBuf,
    address: String,
    cancellation: Cancellation,
    worker: Option<JoinHandle<Result<(), String>>>,
}

impl Proxy {
    fn new(base: &str, ca: Option<&std::path::Path>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let ready = root.join("ready");
        let marker = ready.clone();
        let log = root.join("capture.jsonl");
        fs::write(&log, "stale non-JSON capture from a prior run\n").unwrap();
        let endpoint = Endpoint::parse(base).unwrap();
        let selected = Forwarder::fixture(ca);
        let cancellation = Cancellation::default();
        let token = cancellation.clone();
        let worker = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(listener::serve_selected(
                endpoint,
                &log,
                &marker,
                Duration::from_secs(30),
                token,
                selected,
            ))
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let address = loop {
            if let Ok(text) = fs::read_to_string(&ready) {
                break url::Url::parse(&text)
                    .unwrap()
                    .socket_addrs(|| None)
                    .unwrap()[0]
                    .to_string();
            }
            assert!(
                !worker.is_finished() && Instant::now() < deadline,
                "proxy did not become ready"
            );
            thread::sleep(Duration::from_millis(5));
        };
        Self {
            directory,
            ready,
            address,
            cancellation,
            worker: Some(worker),
        }
    }

    fn capture(&self) -> Vec<Value> {
        fs::read_to_string(self.directory.path().join("capture.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn stop(&mut self) {
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            assert_eq!(worker.join().unwrap(), Ok(()));
        }
        assert!(!self.ready.exists());
    }
}
impl Drop for Proxy {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            let joined = worker.join();
            if !thread::panicking() {
                assert_eq!(joined.unwrap(), Ok(()));
            }
        }
    }
}

fn client(address: &str, method: &str, path: &str, body: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut socket = TcpStream::connect(address)?;
    socket.set_read_timeout(Some(Duration::from_secs(8)))?;
    socket.set_write_timeout(Some(Duration::from_secs(2)))?;
    write!(
        socket,
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nAccept: application/json\r\nAuthorization: Bearer fixture-only\r\nConnection: close, X-Hop\r\nX-Hop: private-hop\r\n\r\n",
        body.len()
    )?;
    socket.write_all(body)?;
    let mut response = Vec::new();
    socket.read_to_end(&mut response)?;
    Ok(response)
}

fn raw_request(socket: &mut TcpStream) -> Vec<u8> {
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    let boundary = loop {
        let count = socket.read(&mut chunk).unwrap();
        assert!(count > 0 && bytes.len() < 65536);
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let head = String::from_utf8_lossy(&bytes[..boundary]);
    let length = head
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length < 65536);
    while bytes.len() < boundary + length {
        let count = socket.read(&mut chunk).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
    }
    bytes
}

#[test]
fn actual_http_forwarding_preserves_non_success_status_body_query_and_header_intent() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", upstream.local_addr().unwrap());
    let worker = thread::spawn(move || {
        upstream.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut socket = loop {
            match upstream.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        let request = raw_request(&mut socket);
        socket.write_all(b"HTTP/1.1 429 fixture\r\nContent-Length: 7\r\nRetry-After: 4\r\nSet-Cookie: first=1\r\nSet-Cookie: second=2\r\nConnection: close, X-Hop\r\nX-Hop: removed\r\n\r\nlimited").unwrap();
        request
    });
    let mut proxy = Proxy::new(&base, None);
    let body = br#"{"messages":[{"role":"user","content":"fixture"}]}"#;
    let response = client(&proxy.address, "POST", "/v1/chat/completions?trace=1", body).unwrap();
    let text = String::from_utf8(response).unwrap();
    assert!(
        text.starts_with("HTTP/1.1 429") && text.ends_with("limited"),
        "{text}"
    );
    assert!(text.to_ascii_lowercase().contains("retry-after: 4"));
    assert_eq!(text.to_ascii_lowercase().matches("set-cookie:").count(), 2);
    assert!(!text.to_ascii_lowercase().contains("x-hop:"));
    let request = String::from_utf8(worker.join().unwrap()).unwrap();
    assert!(request.starts_with("POST /v1/chat/completions?trace=1 HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer fixture-only")
    );
    assert!(!request.to_ascii_lowercase().contains("x-hop:"));
    let records = proxy.capture();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0]["body"],
        serde_json::from_slice::<Value>(body).unwrap()
    );
    assert_eq!(records[0]["path"], "/v1/chat/completions?trace=1");
    assert!(!records[0].to_string().contains("fixture-only"));
    proxy.stop();
}

#[test]
fn actual_tls_forwarding_accepts_trusted_certificate_and_preserves_chunked_sse_bytes() {
    let body = "data: {\"choices\":[]}\n\ndata: [DONE]\n\n";
    let server = tls_fixture::Server::new(vec![tls_fixture::Reply::stream(body.into(), false)]);
    let mut proxy = Proxy::new(&server.base, Some(&server.ca));
    let response = client(&proxy.address, "GET", "/v1/models", &[]).unwrap();
    assert!(response.starts_with(b"HTTP/1.1 200") && response.ends_with(body.as_bytes()));
    assert_eq!(proxy.capture()[0]["body"], Value::Null);
    assert!(
        server.requests.lock().unwrap()[0]
            .0
            .starts_with("GET /tenant/v1/models")
    );
    proxy.stop();
}

#[test]
fn actual_tls_forwarding_rejects_untrusted_certificate_instead_of_disabling_verification() {
    let server = tls_fixture::Server::new(vec![tls_fixture::Reply::json(200, &json!({"data":[]}))]);
    let mut proxy = Proxy::new(&server.base, None);
    let response = client(&proxy.address, "GET", "/models", &[]).unwrap();
    assert!(response.starts_with(b"HTTP/1.1 502"));
    assert!(server.requests.lock().unwrap().is_empty());
    proxy.stop();
}

#[test]
fn listener_shutdown_cancels_actual_held_tls_transfer_and_removes_readiness() {
    let server = tls_fixture::Server::new(vec![tls_fixture::Reply::incomplete()]);
    let mut proxy = Proxy::new(&server.base, Some(&server.ca));
    let address = proxy.address.clone();
    let request = thread::spawn(move || client(&address, "GET", "/models", &[]));
    let deadline = Instant::now() + Duration::from_secs(5);
    while !server
        .response_started
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let started = Instant::now();
    proxy.stop();
    assert!(started.elapsed() < Duration::from_secs(4));
    let _ = request.join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !server
        .held_connections_closed
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        assert!(
            Instant::now() < deadline,
            "cancelled forwarding retained upstream TLS connection"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(proxy.capture().len(), 1);
}

#[test]
fn unsupported_request_method_fails_before_any_upstream_connection_or_capture() {
    let mut proxy = Proxy::new("http://127.0.0.1:1/v1", None);
    let response = client(&proxy.address, "DELETE", "/v1/models", &[]).unwrap();
    assert!(response.starts_with(b"HTTP/1.1 405"));
    assert!(proxy.capture().is_empty());
    proxy.stop();
}
