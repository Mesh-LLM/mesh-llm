//! Management HTTP framing, bounded forwarding and response serialization.
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

/// The largest response header that a transparent management proxy will hold
/// before it knows the terminal HTTP status. This bounds memory for an
/// untrusted plugin response while still accommodating ordinary HTTP headers.
pub const MAX_FORWARDED_RESPONSE_HEADER_BYTES: usize = 16 * 1024;

pub fn http_body_text(raw: &[u8]) -> &str {
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
pub async fn write_managed_response_head(
    stream: &mut TcpStream,
    head: Vec<u8>,
) -> anyhow::Result<()> {
    let (head, status) = managed_response_head(head)?;
    stream.write_all(&head).await?;
    crate::response_scope::record_response_status(status);
    Ok(())
}

/// Validate and decorate a complete, bounded HTTP/1 response head before it
/// is forwarded to a management caller. This is intentionally header-only:
/// opaque response bodies keep their original streaming/backpressure path.
pub fn managed_response_head(mut head: Vec<u8>) -> anyhow::Result<(Vec<u8>, u16)> {
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
    if let Some(request_id) = crate::response_scope::response_request_id_header() {
        head = replace_response_header(head, "x-request-id", &request_id);
    }
    Ok((head, status))
}

/// Split a buffered opaque response at its first complete HTTP/1 header. A
/// plugin cannot make the management proxy retain an unbounded preamble.
pub fn take_bounded_response_head(
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

pub async fn respond_error(stream: &mut TcpStream, code: u16, msg: &str) -> anyhow::Result<()> {
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
    let request_id = crate::response_scope::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{request_id}Content-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    crate::response_scope::record_response_status(code);
    Ok(())
}

pub async fn respond_json<T: serde::Serialize>(
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
    let request_id = crate::response_scope::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{request_id}Content-Length: {}\r\n\r\n{}",
        json.len(),
        json
    );
    stream.write_all(resp.as_bytes()).await?;
    crate::response_scope::record_response_status(code);
    Ok(())
}

pub async fn respond_runtime_error(stream: &mut TcpStream, msg: &str) -> anyhow::Result<()> {
    respond_error(stream, classify_runtime_error(msg), msg).await
}

pub async fn respond_bytes(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    respond_bytes_cached(stream, code, status, content_type, "no-cache", body).await
}

pub async fn respond_bytes_cached<W: AsyncWrite + Unpin>(
    stream: &mut W,
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
    let request_id = crate::response_scope::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\n{request_id}Content-Length: {}\r\nCache-Control: {cache_control}\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    crate::response_scope::record_response_status(code);
    stream.write_all(body).await?;
    Ok(())
}

pub fn classify_runtime_error(msg: &str) -> u16 {
    if msg.contains("not loaded") {
        404
    } else if msg.contains("already loaded") || msg.contains("multiple loaded instances") {
        409
    } else if msg.contains("fit locally")
        || msg.contains("runtime load only supports")
        || msg.contains("runtime capacity")
    {
        422
    } else {
        400
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    #[test]
    fn test_classify_runtime_error_codes() {
        assert_eq!(classify_runtime_error("model 'x' is not loaded"), 404);
        assert_eq!(classify_runtime_error("model 'x' is already loaded"), 409);
        assert_eq!(
            classify_runtime_error("runtime load only supports models that fit locally"),
            422
        );
        assert_eq!(
            classify_runtime_error("runtime capacity for model 'x' exceeds node pool"),
            422
        );
        assert_eq!(classify_runtime_error("bad request"), 400);
    }
}
