use super::super::{MeshApi, http::respond_error};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Whether the management caller behind a forwarded inference request passed
/// the trusted-local check (loopback peer + local `Host`/`Origin`).
///
/// The console forwards inference to the OpenAI port over a fresh loopback
/// socket, so without this the payment boundary in
/// `network::openai::response::paid` would see every forwarded caller as local
/// — including remote ones reaching a `--listen-all` console (Loupe #3051).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CallerTrust {
    TrustedLocal,
    Untrusted,
}

impl CallerTrust {
    pub(crate) fn from_trusted_local(trusted: bool) -> Self {
        if trusted {
            Self::TrustedLocal
        } else {
            Self::Untrusted
        }
    }
}

/// Upper bound on the upstream response head we buffer before relaying.
const MAX_UPSTREAM_HEAD_BYTES: usize = 64 * 1024;

/// Headers forced onto every response the console relays from the inference
/// port. The body is produced by whichever mesh peer served the request, so
/// it must never be able to execute on the management console's origin
/// (Loupe #2834): `sandbox` gives it an opaque origin even if it is HTML, and
/// `nosniff` stops browsers from guessing a script/HTML type.
const RELAY_SECURITY_HEADERS: &str =
    "X-Content-Type-Options: nosniff\r\nContent-Security-Policy: sandbox\r\n";

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    req: &str,
    caller: CallerTrust,
) -> anyhow::Result<()> {
    let is_openai_passthrough = path_only.starts_with("/v1/") || path_only == "/models";
    if method == "OPTIONS" && is_openai_passthrough {
        stream
            .write_all(
                b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type, authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\n\r\n",
            )
            .await?;
        return Ok(());
    }

    // A browser navigation is a GET, and the inference port's model-less
    // fallback would route an arbitrary `/v1/*` GET to whichever peer is first
    // in the target list. Only the model list is a legitimate GET here.
    let allowed_get =
        method == "GET" && crate::network::proxy::is_models_list_request(method, path_only);
    if method != "POST" && !allowed_get {
        return respond_error(stream, 405, "Method Not Allowed").await;
    }

    let upstream_path = if is_openai_passthrough {
        path_only
    } else if path_only.starts_with("/api/chat") {
        "/v1/chat/completions"
    } else if path_only.starts_with("/api/responses") {
        "/v1/responses"
    } else {
        return Ok(());
    };

    let port = state.inner.lock().await.api_port;

    let target = format!("127.0.0.1:{port}");
    match TcpStream::connect(&target).await {
        Ok(mut upstream) => {
            // A remote caller must stay remote across the loopback hop so the
            // payment gate charges (or refuses) it like any other remote peer.
            let _remote_origin = match caller {
                CallerTrust::TrustedLocal => None,
                CallerTrust::Untrusted => Some(crate::network::tunnel::RemoteBridge::register(
                    upstream.local_addr()?,
                )?),
            };
            let rewritten = if is_openai_passthrough {
                req.to_string()
            } else if path_only.starts_with("/api/chat") {
                req.replacen("/api/chat", upstream_path, 1)
            } else {
                req.replacen("/api/responses", upstream_path, 1)
            };
            upstream.write_all(rewritten.as_bytes()).await?;
            relay_upstream_response(stream, &mut upstream).await?;
        }
        _ => {
            respond_error(stream, 502, "Cannot reach LLM server").await?;
        }
    }
    Ok(())
}

/// Relay the upstream response, splicing [`RELAY_SECURITY_HEADERS`] into its
/// head, then stream the rest bidirectionally (request bodies may still be
/// in flight for streaming clients).
async fn relay_upstream_response(
    client: &mut TcpStream,
    upstream: &mut TcpStream,
) -> anyhow::Result<()> {
    let mut head = Vec::with_capacity(1024);
    loop {
        if let Some(end) = header_end(&head) {
            head.splice(end..end, RELAY_SECURITY_HEADERS.bytes());
            break;
        }
        if head.len() >= MAX_UPSTREAM_HEAD_BYTES {
            anyhow::bail!("upstream response head exceeded {MAX_UPSTREAM_HEAD_BYTES} bytes");
        }
        let mut chunk = [0u8; 4096];
        let n = upstream.read(&mut chunk).await?;
        if n == 0 {
            // Upstream closed without a complete head: pass through whatever
            // arrived so the client sees the same failure it would have.
            client.write_all(&head).await?;
            return Ok(());
        }
        head.extend_from_slice(&chunk[..n]);
    }
    client.write_all(&head).await?;
    tokio::io::copy_bidirectional(client, upstream).await?;
    Ok(())
}

/// Offset of the start of the blank line terminating the response head
/// (the position where an extra header line can be inserted).
fn header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_end_points_at_blank_line() {
        let head = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html>";
        let end = header_end(head).unwrap();
        assert_eq!(
            &head[..end],
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
        );
        assert_eq!(header_end(b"HTTP/1.1 200 OK\r\nX: y\r\n"), None);
    }

    #[tokio::test]
    async fn relay_forces_sandbox_and_nosniff_on_upstream_html() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = "<script>alert(1)</script>";
        let upstream_task = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes()).await.unwrap();
            let _ = s.shutdown().await;
        });
        let client_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        let mut browser = TcpStream::connect(client_addr).await.unwrap();
        let (mut client, _) = client_listener.accept().await.unwrap();
        let mut upstream = TcpStream::connect(addr).await.unwrap();
        browser.shutdown().await.unwrap();
        relay_upstream_response(&mut client, &mut upstream)
            .await
            .unwrap();
        drop(client);
        let mut out = Vec::new();
        browser.read_to_end(&mut out).await.unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(
            text.contains("X-Content-Type-Options: nosniff\r\n"),
            "{text}"
        );
        assert!(
            text.contains("Content-Security-Policy: sandbox\r\n"),
            "{text}"
        );
        assert!(text.ends_with(&format!("\r\n\r\n{body}")), "{text}");
        upstream_task.await.unwrap();
    }
}
