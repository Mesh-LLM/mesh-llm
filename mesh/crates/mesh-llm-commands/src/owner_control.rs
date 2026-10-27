//! CLI-side owner-control credential acquisition.
//!
//! Owner-control management routes require the host's owner-control
//! credential as a `Bearer` token. The host publishes that credential on the
//! loopback-only `GET /api/runtime/control/owner-control-token` endpoint, so
//! a CLI running on the host fetches it once and presents it on the
//! owner-control requests it makes.

use anyhow::{Context, Result, bail};
use serde_json::Value;

const OWNER_CONTROL_TOKEN_PATH: &str = "/api/runtime/control/owner-control-token";
const OWNER_CONTROL_TOKEN_TIMEOUT_SECS: u64 = 10;

/// Fetches the owner-control credential the host published for loopback
/// callers, ready to present as `Authorization: Bearer <token>`.
pub async fn owner_control_token(port: u16) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            OWNER_CONTROL_TOKEN_TIMEOUT_SECS,
        ))
        .build()?;
    let url = format!("http://127.0.0.1:{port}{OWNER_CONTROL_TOKEN_PATH}");
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("can't reach the mesh-llm console on port {port} to fetch the owner-control credential; is it running?")
        })?;
    let status = response.status();
    let body: Value = response.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        bail!(
            "owner-control credential request failed ({status}): {}",
            body.get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        );
    }
    body.get("token")
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| "owner-control credential response has no token")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A loopback stub that serves `replies` in order and records every
    /// header line of each request.
    pub(crate) struct Stub {
        pub(crate) port: u16,
        requests: Arc<Mutex<Vec<Vec<String>>>>,
    }

    pub(crate) fn serve_in_order(replies: Vec<(u16, String)>) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("addr").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::new(Mutex::new(replies));
        let requests_thread = requests.clone();
        std::thread::spawn(move || {
            let mut replies = captured.lock().expect("replies lock");
            for (status, body) in replies.drain(..) {
                let (mut stream, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let mut buf: Vec<u8> = Vec::new();
                while buf.windows(4).all(|w| w != b"\r\n\r\n") && buf.len() < 65536 {
                    let mut chunk = [0u8; 4096];
                    let n = match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    buf.extend_from_slice(&chunk[..n]);
                }
                let head = String::from_utf8_lossy(&buf)
                    .split("\r\n\r\n")
                    .next()
                    .unwrap_or("")
                    .lines()
                    .map(str::to_string)
                    .collect();
                requests_thread.lock().expect("requests lock").push(head);
                let response = format!(
                    "HTTP/1.1 {status} Test\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        Stub { port, requests }
    }

    impl Stub {
        /// All header lines (request line included) of each recorded request.
        pub(crate) fn request_heads(&self) -> Vec<Vec<String>> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    #[tokio::test]
    async fn owner_control_token_reads_published_credential() {
        let stub = serve_in_order(vec![(200, r#"{"token":"loopback-secret"}"#.to_string())]);
        let token = owner_control_token(stub.port)
            .await
            .expect("loopback token fetch");
        assert_eq!(token, "loopback-secret");
        let heads = stub.request_heads();
        assert!(
            heads.first().and_then(|head| head.first()).is_some_and(
                |line| line.starts_with("GET /api/runtime/control/owner-control-token")
            ),
            "token endpoint must be fetched over loopback: {heads:?}"
        );
    }

    #[tokio::test]
    async fn owner_control_token_surfaces_host_error() {
        let stub = serve_in_order(vec![(
            503,
            r#"{"error":"owner control credential is not configured"}"#.to_string(),
        )]);
        let err = owner_control_token(stub.port)
            .await
            .expect_err("unconfigured host must fail");
        assert!(
            err.to_string()
                .contains("owner control credential is not configured"),
            "host error must surface verbatim: {err}"
        );
    }
}
