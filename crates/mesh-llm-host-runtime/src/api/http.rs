use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

/// The largest response header that a transparent management proxy will hold
/// before it knows the terminal HTTP status. This bounds memory for an
/// untrusted plugin response while still accommodating ordinary HTTP headers.
pub(super) const MAX_FORWARDED_RESPONSE_HEADER_BYTES: usize = 16 * 1024;

pub(super) fn http_body_text(raw: &[u8]) -> &str {
    let body_start = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
        .unwrap_or(raw.len());
    std::str::from_utf8(&raw[body_start..]).unwrap_or("")
}

/// Write one complete management response header, normalizing its request ID
/// to the scoped value. The status is recorded only after the header reaches
/// the socket so lifecycle state cannot claim a response that was never sent.
pub(super) async fn write_managed_response_head(
    stream: &mut TcpStream,
    head: Vec<u8>,
) -> anyhow::Result<()> {
    let (head, status) = managed_response_head(head)?;
    stream.write_all(&head).await?;
    super::management_lifecycle::record_response_status(status);
    Ok(())
}

/// Validate and decorate a complete, bounded HTTP/1 response head before it
/// is forwarded to a management caller. This is intentionally header-only:
/// opaque response bodies keep their original streaming/backpressure path.
pub(super) fn managed_response_head(mut head: Vec<u8>) -> anyhow::Result<(Vec<u8>, u16)> {
    if head.len() > MAX_FORWARDED_RESPONSE_HEADER_BYTES || !head.ends_with(b"\r\n\r\n") {
        anyhow::bail!("plugin response header is malformed or exceeds the bounded limit");
    }

    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Response::new(&mut headers);
    let parsed_len = match parsed.parse(&head)? {
        httparse::Status::Complete(length) if length == head.len() => length,
        httparse::Status::Complete(_) | httparse::Status::Partial => {
            anyhow::bail!("plugin response header is malformed")
        }
    };
    debug_assert_eq!(parsed_len, head.len());
    let status = parsed
        .code
        .ok_or_else(|| anyhow::anyhow!("plugin response status is missing"))?;
    if !(200..=599).contains(&status) {
        anyhow::bail!("plugin response must start with a terminal HTTP status");
    }
    if let Some(request_id) = super::management_lifecycle::response_request_id_header() {
        head = replace_response_header(head, "x-request-id", &request_id);
    }
    Ok((head, status))
}

/// Split a buffered opaque response at its first complete HTTP/1 header. A
/// plugin cannot make the management proxy retain an unbounded preamble.
pub(super) fn take_bounded_response_head(
    buffer: &mut Vec<u8>,
) -> anyhow::Result<Option<(Vec<u8>, Vec<u8>)>> {
    let Some(head_end) = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
    else {
        if buffer.len() > MAX_FORWARDED_RESPONSE_HEADER_BYTES {
            anyhow::bail!("plugin response header exceeds the bounded limit");
        }
        return Ok(None);
    };
    if head_end > MAX_FORWARDED_RESPONSE_HEADER_BYTES {
        anyhow::bail!("plugin response header exceeds the bounded limit");
    }
    let body = buffer.split_off(head_end);
    Ok(Some((std::mem::take(buffer), body)))
}

fn replace_response_header(head: Vec<u8>, name: &str, value: &str) -> Vec<u8> {
    let mut rewritten = Vec::with_capacity(head.len() + value.len() + 18);
    let mut lines = head.split_inclusive(|byte| *byte == b'\n');
    // Keep the status line exactly as the plugin produced it. Each following
    // non-empty line is a header; discard every case-insensitive occurrence of
    // the correlation header before appending the scope-owned value.
    if let Some(status_line) = lines.next() {
        rewritten.extend_from_slice(status_line);
    }
    for line in lines {
        let trimmed = line.strip_suffix(b"\r\n").unwrap_or(line);
        if trimmed.is_empty() {
            break;
        }
        let header_name = trimmed
            .splitn(2, |byte| *byte == b':')
            .next()
            .unwrap_or_default()
            .trim_ascii();
        if header_name.eq_ignore_ascii_case(name.as_bytes()) {
            continue;
        }
        rewritten.extend_from_slice(line);
    }
    rewritten.extend_from_slice(format!("{name}: {value}\r\n\r\n").as_bytes());
    rewritten
}

pub(super) async fn respond_error(
    stream: &mut TcpStream,
    code: u16,
    msg: &str,
) -> anyhow::Result<()> {
    let body = serde_json::to_string(&serde_json::json!({"error": msg}))
        .unwrap_or_else(|_| r#"{"error":"internal error"}"#.to_string());
    let status = match code {
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
    let request_id = super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{request_id}Content-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    super::management_lifecycle::record_response_status(code);
    Ok(())
}

pub(super) async fn respond_json<T: serde::Serialize>(
    stream: &mut TcpStream,
    code: u16,
    value: &T,
) -> anyhow::Result<()> {
    let json = serde_json::to_string(value)?;
    let status = match code {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let request_id = super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{request_id}Content-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    super::management_lifecycle::record_response_status(code);
    Ok(())
}

pub(super) async fn respond_runtime_error(stream: &mut TcpStream, msg: &str) -> anyhow::Result<()> {
    respond_error(stream, crate::api::classify_runtime_error(msg), msg).await
}

pub(super) async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    respond_bytes_cached(stream, code, status, content_type, "no-cache", body).await
}

pub(super) async fn respond_bytes_cached(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    cache_control: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    write_bytes_response(stream, code, status, content_type, cache_control, body).await
}

async fn write_bytes_response<W: AsyncWrite + Unpin>(
    stream: &mut W,
    code: u16,
    status: &str,
    content_type: &str,
    cache_control: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let request_id = super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\n{request_id}Content-Length: {}\r\nCache-Control: {cache_control}\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    super::management_lifecycle::record_response_status(code);
    stream.write_all(body).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll},
    };

    use mesh_llm_events::logging::identifiers::RequestId;
    use tokio::{
        io::{AsyncReadExt, AsyncWrite, AsyncWriteExt},
        net::TcpListener,
    };

    use super::{
        MAX_FORWARDED_RESPONSE_HEADER_BYTES, managed_response_head, take_bounded_response_head,
        write_bytes_response, write_managed_response_head,
    };

    #[derive(Default)]
    struct FailAfterResponseHead {
        writes: usize,
        head: Vec<u8>,
    }

    impl AsyncWrite for FailAfterResponseHead {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.writes += 1;
            if self.writes == 1 {
                self.head.extend_from_slice(buffer);
                Poll::Ready(Ok(buffer.len()))
            } else {
                Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "induced body write failure",
                )))
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    async fn assert_scoped_response_lifecycle(status: u16, expected_state: &str) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        let (mut server, _) = listener.accept().await.unwrap();
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_post",
        );

        super::super::management_lifecycle::scope(lifecycle, async {
            write_managed_response_head(
                &mut server,
                format!(
                    "HTTP/1.1 {status} Test\r\nX-Request-Id: upstream\r\nContent-Length: 0\r\n\r\n"
                )
                .into_bytes(),
            )
            .await
            .expect("write managed response header");
        })
        .await;
        server.shutdown().await.unwrap();

        let mut response = Vec::new();
        client.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with(&format!("HTTP/1.1 {status} Test\r\n")));
        assert!(response.contains(&format!("x-request-id: {}\r\n", request_id.as_uuid())));
        assert!(!response.contains("upstream"));
        assert_eq!(response.matches("x-request-id:").count(), 1);
        let entry = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal lifecycle entry");
        assert_eq!(entry.state, expected_state);
    }

    #[tokio::test]
    async fn managed_response_head_replaces_a_conflicting_request_id() {
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            service,
            request_id,
            "management_post",
        );
        let (_, status) = super::super::management_lifecycle::scope(lifecycle, async {
            let (head, status) = managed_response_head(
                b"HTTP/1.1 201 Created\r\nX-Request-Id: upstream-id\r\nx-request-id: second-upstream-id\r\nContent-Length: 0\r\n\r\n".to_vec(),
            )
            .expect("response head");
            let head = String::from_utf8(head).expect("UTF-8 response head");
            assert!(head.contains(&format!("x-request-id: {}", request_id.as_uuid())));
            assert!(!head.contains("upstream-id"));
            assert!(!head.contains("second-upstream-id"));
            assert_eq!(head.matches("x-request-id:").count(), 1);
            ((), status)
        })
        .await;

        assert_eq!(status, 201);
    }

    #[tokio::test]
    async fn managed_response_writer_records_exact_terminal_status_classes() {
        assert_scoped_response_lifecycle(201, "completed").await;
        assert_scoped_response_lifecycle(404, "rejected").await;
        assert_scoped_response_lifecycle(503, "failed").await;
    }

    #[tokio::test]
    async fn committed_response_head_keeps_status_when_body_write_fails() {
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_post",
        );
        let mut writer = FailAfterResponseHead::default();

        let result = super::super::management_lifecycle::scope(lifecycle, async {
            write_bytes_response(
                &mut writer,
                200,
                "OK",
                "application/octet-stream",
                "no-cache",
                b"body",
            )
            .await
        })
        .await;

        assert!(result.is_err());
        assert!(writer.head.starts_with(b"HTTP/1.1 200 OK\r\n"));
        let entry = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal lifecycle entry");
        assert_eq!(entry.state, "completed");
        let exact_status = service
            .bus_ref()
            .replay_window()
            .records
            .into_iter()
            .filter_map(|record| {
                let envelope =
                    serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
                let payload = envelope.get("payload")?.as_str()?;
                serde_json::from_str::<mesh_llm_events::logging::events::LifecycleEvent>(payload)
                    .ok()
            })
            .find_map(|event| match event {
                mesh_llm_events::logging::events::LifecycleEvent::Completed {
                    status_code, ..
                } => status_code,
                _ => None,
            });
        assert_eq!(exact_status, Some(200));
    }

    #[test]
    fn bounded_response_head_preserves_the_first_body_bytes_and_rejects_oversize() {
        let mut response = b"HTTP/1.1 404 Not Found\r\nContent-Length: 7\r\n\r\nmissing".to_vec();
        let (head, body) = take_bounded_response_head(&mut response)
            .expect("bounded response header")
            .expect("complete response header");
        assert_eq!(head, b"HTTP/1.1 404 Not Found\r\nContent-Length: 7\r\n\r\n");
        assert_eq!(body, b"missing");
        assert!(response.is_empty());

        let mut incomplete = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n".to_vec();
        assert!(
            take_bounded_response_head(&mut incomplete)
                .expect("incomplete header stays buffered")
                .is_none()
        );
        let mut oversized = vec![b'x'; MAX_FORWARDED_RESPONSE_HEADER_BYTES + 1];
        assert!(take_bounded_response_head(&mut oversized).is_err());
    }

    #[test]
    fn managed_response_head_rejects_non_terminal_or_incomplete_responses() {
        for head in [
            b"HTTP/1.1 100 Continue\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n".as_slice(),
        ] {
            assert!(managed_response_head(head.to_vec()).is_err());
        }
    }
}
