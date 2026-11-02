//! CLI-side owner-control credential acquisition.
//!
//! Owner-control management routes require the host's owner-control
//! credential as a `Bearer` token. The CLI reads it from the owner-only local
//! token file instead of exposing the credential over loopback TCP.

use anyhow::{Context, Result, bail};
/// Read the local credential. Set `MESH_LLM_OWNER_CONTROL_TOKEN_FILE` when the
/// host uses a custom owner keystore path.
pub async fn owner_control_token(_port: u16) -> Result<String> {
    #[cfg(test)]
    if let Ok(token) = tests::TEST_TOKEN.try_with(Clone::clone) {
        return Ok(token);
    }
    let path = match std::env::var_os("MESH_LLM_OWNER_CONTROL_TOKEN_FILE") {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            mesh_llm_identity::default_keystore_path()?.with_file_name("owner-control-token-v2")
        }
    };
    read_token(&path)
}

fn read_token(path: &std::path::Path) -> Result<String> {
    let contents = mesh_llm_config::read_private_file(path)
        .with_context(|| format!("reading owner-control credential from {}", path.display()))?;
    let contents = zeroize::Zeroizing::new(contents);
    let token = std::str::from_utf8(&contents)
        .context("owner-control credential is not UTF-8")?
        .trim();
    if token.is_empty() {
        bail!("owner-control credential is empty");
    }
    Ok(token.to_owned())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    tokio::task_local! {
        pub(crate) static TEST_TOKEN: String;
    }

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

    #[test]
    fn reads_owner_only_credential_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("owner-control-token-v2");
        mesh_llm_config::create_private_file_if_absent(&path, b"test-token\n").unwrap();
        assert_eq!(read_token(&path).unwrap(), "test-token");
    }

    #[test]
    fn missing_credential_fails_without_network_fallback() {
        let directory = tempfile::tempdir().unwrap();
        assert!(read_token(&directory.path().join("missing")).is_err());
    }
}
