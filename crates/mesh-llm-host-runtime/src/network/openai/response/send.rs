use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub async fn send_json_ok(mut stream: TcpStream, data: &serde_json::Value) -> std::io::Result<()> {
    let body = data.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// RFC 7230 tchar set for header field names: ASCII alphanumeric plus
/// `!#$%&'*+-.^_`|~`. We additionally forbid `:` because it terminates
/// the field-name in the wire grammar. Used to reject caller-provided
/// header names that could carry CR/LF or other injection bytes.
pub(crate) fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Append a single `name: value` header line if `name` is a valid HTTP
/// header field name. CR/LF in `value` is stripped defensively. Used by
/// the `*_with_headers` writers below so a malformed header from a
/// future caller can't inject extra headers / smuggle a response.
pub(crate) fn append_safe_header(headers: &mut String, name: &str, value: &str) {
    if !is_valid_header_name(name) {
        tracing::warn!(
            "openai transport: dropping header with invalid name `{name}` (RFC 7230 tchar required)"
        );
        return;
    }
    let safe_value: String = value.chars().filter(|c| *c != '\r' && *c != '\n').collect();
    headers.push_str(name);
    headers.push_str(": ");
    headers.push_str(&safe_value);
    headers.push_str("\r\n");
}

/// Like `send_json_ok` but allows the caller to append arbitrary response
/// headers (e.g. `x-moa-*` observability headers).
///
/// Header names must satisfy the RFC 7230 tchar grammar (ASCII
/// alphanumeric + a small symbol set); invalid names are dropped with a
/// warning rather than written verbatim. Values are stripped of CR/LF.
pub async fn send_json_ok_with_headers(
    mut stream: TcpStream,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let body = data.to_string();
    let mut headers = String::from("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Send a JSON body with a non-200 status and the given extra headers.
///
/// The body is sent verbatim — caller controls the shape. Use for cases
/// where the in-band payload is already a structured error (e.g. MoA's
/// `error_response`) and we still want to attach observability headers
/// while signalling failure via the HTTP status line.
pub async fn send_json_with_status_and_headers(
    mut stream: TcpStream,
    code: u16,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let status = match code {
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    let body = data.to_string();
    let mut headers = format!("HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_400(mut stream: TcpStream, msg: &str) -> std::io::Result<()> {
    let body = openai_error_body(400, msg);
    let headers = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_error(mut stream: TcpStream, code: u16, msg: &str) -> std::io::Result<()> {
    let status = match code {
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Bad Request",
    };
    let body = openai_error_body(code, msg);
    let retry_after = if code == 429 {
        "Retry-After: 5\r\n"
    } else {
        ""
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\n\r\n{}",
        body.len(),
        String::from_utf8_lossy(&body)
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

pub async fn send_503(stream: TcpStream, reason: &str) -> std::io::Result<()> {
    tracing::warn!("503 → client: {reason}");
    send_503_inner(stream, reason).await
}

async fn send_503_inner(mut stream: TcpStream, reason: &str) -> std::io::Result<()> {
    let body = openai_error_body(503, reason);
    let resp = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        String::from_utf8_lossy(&body)
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn openai_error_body(status_code: u16, message: &str) -> Vec<u8> {
    let status =
        http::StatusCode::from_u16(status_code).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
    let kind = openai_error_kind_for_status(status_code);
    let error = openai_frontend::OpenAiError::from_kind(status, kind, message)
        .with_code(openai_error_code_for_status(status_code));
    serde_json::to_vec(&error.body()).expect("serializing JSON error response should not fail")
}

const fn openai_error_kind_for_status(status_code: u16) -> openai_frontend::OpenAiErrorKind {
    match status_code {
        401 => openai_frontend::OpenAiErrorKind::Authentication,
        403 => openai_frontend::OpenAiErrorKind::Permission,
        404 => openai_frontend::OpenAiErrorKind::NotFound,
        413 => openai_frontend::OpenAiErrorKind::PayloadTooLarge,
        429 => openai_frontend::OpenAiErrorKind::RateLimit,
        500 => openai_frontend::OpenAiErrorKind::Internal,
        502 => openai_frontend::OpenAiErrorKind::ServiceUnavailable,
        503 => openai_frontend::OpenAiErrorKind::ServiceUnavailable,
        504 => openai_frontend::OpenAiErrorKind::Timeout,
        _ => openai_frontend::OpenAiErrorKind::InvalidRequest,
    }
}

const fn openai_error_code_for_status(status_code: u16) -> &'static str {
    match status_code {
        400 => "bad_request",
        401 => "invalid_api_key",
        403 => "permission_denied",
        404 => "model_not_found",
        409 => "conflict",
        413 => "payload_too_large",
        422 => "unprocessable_content",
        429 => "rate_limit_exceeded",
        500 => "internal_server_error",
        502 => "service_unavailable",
        503 => "service_unavailable",
        504 => "timeout",
        _ => "invalid_request",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use tokio::net::TcpListener;

    #[test]
    fn is_valid_header_name_accepts_normal_observability_headers() {
        assert!(is_valid_header_name("x-moa-elapsed-ms"));
        assert!(is_valid_header_name("X-MoA-Workers"));
        assert!(is_valid_header_name("Content-Type"));
        assert!(is_valid_header_name("x-request-id"));
    }

    #[test]
    fn is_valid_header_name_rejects_injection_attempts() {
        // Regression for PR #566 review item #5c: header NAMES were not
        // sanitized, only values. A name carrying CR/LF or a colon would
        // smuggle extra headers / split the response.
        assert!(!is_valid_header_name("x-evil\r\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil: hijacked"));
        assert!(!is_valid_header_name("x evil")); // space inside name
        assert!(!is_valid_header_name(""));
    }

    #[test]
    fn append_safe_header_drops_invalid_name() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-evil\r\nSet-Cookie", "bad");
        assert!(buf.is_empty(), "invalid name must be dropped, got {buf:?}");
    }

    #[test]
    fn append_safe_header_strips_crlf_from_value() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-ok", "ok\r\nSet-Cookie: hijack");
        assert!(
            buf.starts_with("x-ok: okSet-Cookie: hijack\r\n"),
            "value CRLF must be stripped; got {buf:?}"
        );
        assert_eq!(buf.matches("\r\n").count(), 1);
    }

    #[tokio::test]
    async fn test_send_error_429_includes_retry_after() {
        let response = capture_proxy_error_response(|stream| async move {
            super::send_error(stream, 429, "model not available").await
        })
        .await;
        let body = response_json_body(&response);

        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(response.contains("Retry-After: 5\r\n"));
        assert_eq!(body["error"]["message"], "model not available");
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn test_send_503_uses_openai_error_shape() {
        let response = capture_proxy_error_response(|stream| async move {
            super::send_503(stream, "skippy ABI call failed: Unsupported").await
        })
        .await;
        let body = response_json_body(&response);

        assert!(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
        assert_eq!(
            body["error"]["message"],
            "skippy ABI call failed: Unsupported"
        );
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "service_unavailable");
    }
    async fn capture_proxy_error_response<F, Fut>(send: F) -> String
    where
        F: FnOnce(tokio::net::TcpStream) -> Fut + Send + 'static,
        Fut: Future<Output = std::io::Result<()>> + Send + 'static,
    {
        use tokio::io::AsyncReadExt;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            send(stream).await.unwrap();
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server.await.unwrap();
        String::from_utf8(output).unwrap()
    }

    fn response_json_body(response: &str) -> serde_json::Value {
        let body_start = response
            .find("\r\n\r\n")
            .map(|index| index + 4)
            .expect("response contains header terminator");
        serde_json::from_str(&response[body_start..]).unwrap()
    }
}
