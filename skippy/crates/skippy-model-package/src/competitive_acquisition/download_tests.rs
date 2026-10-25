#![cfg(unix)]
use super::*;
use std::{
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};
pub(super) struct Peer {
    pub(super) endpoint: String,
    stop: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<String>>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Peer {
    pub(super) fn new(responses: Vec<(String, Vec<u8>, usize)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let done = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let thread = thread::spawn(move || {
            // Native TLS client setup precedes the first request on macOS.
            // Keep that readiness allowance separate from the five-second request roster.
            let mut until = Instant::now() + Duration::from_secs(30);
            for (index, (revision, body, length)) in responses.into_iter().enumerate() {
                let mut socket = loop {
                    if done.load(Ordering::SeqCst) || Instant::now() >= until {
                        return;
                    }
                    match listener.accept() {
                        Ok((s, _)) => break s,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(2))
                        }
                        Err(_) => return,
                    }
                };
                if index == 0 {
                    until = Instant::now() + Duration::from_secs(5);
                }
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_millis(500)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0; 1024];
                loop {
                    match socket.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => bytes.extend_from_slice(&chunk[..n]),
                    };
                    if bytes.windows(4).any(|b| b == b"\r\n\r\n") {
                        break;
                    }
                    if bytes.len() > 8192 {
                        return;
                    }
                }
                seen.lock().unwrap().push(String::from_utf8(bytes).unwrap());
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nETag: \"fixture\"\r\nX-Repo-Commit: {revision}\r\nConnection: close\r\n\r\n"
                );
                if socket.write_all(header.as_bytes()).is_err() {
                    return;
                }
                let _ = socket.write_all(&body);
            }
        });
        Self {
            endpoint,
            stop,
            requests,
            thread: Some(thread),
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.thread.take() {
            worker.join().unwrap();
        }
    }
}
fn client(peer: &Peer) -> hf_hub::HFClient {
    hf_hub::HFClient::builder()
        .endpoint(&peer.endpoint)
        .token("fixture-token")
        .cache_enabled(false)
        .client(
            reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap(),
        )
        .retry_max_attempts(0)
        .build()
        .unwrap()
}
#[test]
fn actual_native_hf_stream_uses_immutable_revision_and_verifies_exact_bytes_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let revision = "a".repeat(40);
    let client_setup_started = Instant::now();
    let peer = Peer::new(vec![
        (revision.clone(), vec![], 4),
        (revision.clone(), vec![], 4),
        (revision.clone(), b"real".to_vec(), 4),
    ]);
    let api = client(&peer);
    let client_setup_elapsed = client_setup_started.elapsed();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let output = root.join("model.bin");
    let (hash, size) = runtime
        .block_on(acquire::one(
            &api.model("owner", "repo"),
            &revision,
            "model.bin",
            &output,
            Some(&contract::digest(b"real")),
            4,
            Instant::now() + Duration::from_secs(3),
        ))
        .unwrap_or_else(|error| {
            panic!("native stream failed: {error}; client setup {client_setup_elapsed:?}; observed requests {}", peer.requests.lock().unwrap().len())
        });
    assert_eq!(size, 4);
    assert_eq!(hash, contract::digest(b"real"));
    assert_eq!(std::fs::read(&output).unwrap(), b"real");
    let requests = peer.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .all(|s| s.contains(&format!("/owner/repo/resolve/{revision}/model.bin")))
    );
    assert!(requests[0].starts_with("HEAD "));
    assert!(requests[2].starts_with("GET "));
    drop(requests);
    drop(peer);
    temp.close().unwrap();
}
#[test]
fn native_hf_wrong_revision_pin_oversize_or_incomplete_body_refuse_fresh_output() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let revision = "a".repeat(40);
    for (mode, responses) in [
        ("revision", vec![("b".repeat(40), vec![], 4)]),
        (
            "pin",
            vec![
                (revision.clone(), vec![], 4),
                (revision.clone(), vec![], 4),
                (revision.clone(), b"wrong".to_vec(), 4),
            ],
        ),
        (
            "long",
            vec![
                (revision.clone(), vec![], 4),
                (revision.clone(), vec![], 4),
                (revision.clone(), b"12345678".to_vec(), 8),
            ],
        ),
        (
            "short",
            vec![
                (revision.clone(), vec![], 4),
                (revision.clone(), vec![], 4),
                (revision.clone(), b"x".to_vec(), 4),
            ],
        ),
    ] {
        let peer = Peer::new(responses);
        let api = client(&peer);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let output = root.join(mode);
        assert!(
            runtime
                .block_on(acquire::one(
                    &api.model("owner", "repo"),
                    &revision,
                    "model.bin",
                    &output,
                    Some(&contract::digest(b"real")),
                    4,
                    Instant::now() + Duration::from_secs(3)
                ))
                .is_err()
        );
        assert!(!output.exists());
        if mode == "revision" {
            assert_eq!(peer.requests.lock().unwrap().len(), 1);
        }
        drop(peer);
    }
    temp.close().unwrap();
}
