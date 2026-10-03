//! Minimal in-process HTTP endpoint used by the real-transport webhook tests.
//!
//! Extracted from `tests.rs` so the delivery test module stays within the
//! logging module-boundary line budget enforced by
//! `scripts/tests/test_logging_module_boundaries.py`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, watch};

#[derive(Clone, Copy)]
pub(super) enum LocalHttpReply {
    Status(u16),
    Stall,
}

pub(super) struct LocalFakeHttpServer {
    pub(super) endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    received: Arc<Notify>,
    shutdown_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl LocalFakeHttpServer {
    pub(super) async fn start(replies: impl IntoIterator<Item = LocalHttpReply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local fake webhook server");
        let endpoint = format!(
            "http://{}/webhook",
            listener.local_addr().expect("local server address")
        );
        let requests = Arc::new(Mutex::new(Vec::new()));
        let received = Arc::new(Notify::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(run_local_fake_http_server(
            listener,
            replies.into_iter().collect(),
            Arc::clone(&requests),
            Arc::clone(&received),
            shutdown_rx,
        ));
        Self {
            endpoint,
            requests,
            received,
            shutdown_tx,
            task,
        }
    }

    pub(super) async fn wait_for_requests(&self, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if self.requests.lock().expect("request lock").len() >= expected {
                    return;
                }
                self.received.notified().await;
            }
        })
        .await
        .expect("fake server received expected requests");
    }

    pub(super) fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("request lock").clone()
    }

    pub(super) async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        self.task.abort();
        let _ = self.task.await;
    }
}

async fn run_local_fake_http_server(
    listener: TcpListener,
    mut replies: VecDeque<LocalHttpReply>,
    requests: Arc<Mutex<Vec<String>>>,
    received: Arc<Notify>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    loop {
        let accepted = tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return;
                }
                continue;
            }
            accepted = listener.accept() => accepted,
        };
        let (mut stream, _) = accepted.expect("accept fake webhook request");
        let request = read_fake_http_request(&mut stream).await;
        requests.lock().expect("request lock").push(request);
        received.notify_one();
        match replies.pop_front().unwrap_or(LocalHttpReply::Status(500)) {
            LocalHttpReply::Status(status) => write_fake_http_response(&mut stream, status).await,
            LocalHttpReply::Stall => {
                let _ = shutdown_rx.changed().await;
                return;
            }
        }
    }
}

async fn read_fake_http_request(stream: &mut TcpStream) -> String {
    const MAX_REQUEST_BYTES: usize = 16 * 1024;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await.expect("read webhook request");
        assert!(read > 0, "webhook client closed before sending a request");
        bytes.extend_from_slice(&chunk[..read]);
        assert!(
            bytes.len() <= MAX_REQUEST_BYTES,
            "fake request exceeded bound"
        );
        let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = std::str::from_utf8(&bytes[..header_end]).expect("request headers utf-8");
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.strip_prefix("content-length: ")
                    .or_else(|| line.strip_prefix("Content-Length: "))
            })
            .and_then(|value| value.parse::<usize>().ok())
            .expect("webhook request content length");
        if bytes.len() >= header_end + 4 + content_length {
            return String::from_utf8(bytes).expect("request utf-8");
        }
    }
}

async fn write_fake_http_response(stream: &mut TcpStream, status: u16) {
    stream
        .write_all(
            format!("HTTP/1.1 {status} Test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await
        .expect("write fake webhook response");
}
