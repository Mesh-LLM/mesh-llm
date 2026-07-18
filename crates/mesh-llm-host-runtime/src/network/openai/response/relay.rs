use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_completion_tokens_from_json_body,
    retryable_quality_result,
};
use super::probe::{ParsedResponseHeaders, ResponseProbe, read_response_chunk};
use anyhow::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const MAX_ERROR_RESPONSE_BYTES: usize = 256 * 1024;

fn reason_phrase(status_code: u16) -> &'static str {
    match status_code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub(in crate::network::openai::response) fn remap_error_http_response(
    status_code: u16,
    header_end: usize,
    full_response: &[u8],
) -> Option<Vec<u8>> {
    if status_code < 400 || header_end > full_response.len() {
        return None;
    }
    let mapped_body =
        openai_frontend::map_upstream_error_body(status_code, &full_response[header_end..])?;
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_code,
        reason_phrase(status_code),
        mapped_body.len()
    );
    let mut response = header.into_bytes();
    response.extend_from_slice(&mapped_body);
    Some(response)
}

fn oversized_error_http_response(status_code: u16) -> Vec<u8> {
    let body = serde_json::json!({
        "error": {
            "message": "upstream error response exceeded proxy limit",
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": "upstream_error_too_large",
        }
    })
    .to_string();
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status_code,
        reason_phrase(status_code),
        body.len(),
        body
    )
    .into_bytes()
}

pub(in crate::network::openai::response) async fn relay_error_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
) -> Result<RouteAttemptResult> {
    let status_code = probe.status_code;
    let header_end = probe.header_end;
    let mut buffered = probe.buffered;
    let mut limited = reader.take((MAX_ERROR_RESPONSE_BYTES + 1) as u64);
    if let Err(err) = limited.read_to_end(&mut buffered).await {
        tracing::debug!("error response relay read ended before EOF: {err}");
    }
    let outgoing = if buffered.len().saturating_sub(header_end) > MAX_ERROR_RESPONSE_BYTES {
        tracing::warn!(
            "upstream error body exceeded {} bytes for status {}",
            MAX_ERROR_RESPONSE_BYTES,
            status_code
        );
        oversized_error_http_response(status_code)
    } else {
        remap_error_http_response(status_code, header_end, &buffered).unwrap_or(buffered)
    };
    tcp_stream.write_all(&outgoing).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code,
        completion_tokens: None,
    })
}

pub(in crate::network::openai::response) async fn relay_success_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    parsed: ParsedResponseHeaders,
    retry_policy: ResponseRetryPolicy,
) -> Result<RouteAttemptResult> {
    if let Some(content_length) = parsed.content_length {
        const MAX_SUCCESS_METRICS_BODY_BYTES: usize = 1024 * 1024;
        if content_length <= MAX_SUCCESS_METRICS_BODY_BYTES {
            let mut buffered = probe.buffered;
            while buffered.len() < parsed.header_end + content_length {
                read_response_chunk(reader, &mut buffered).await?;
            }
            if let Some(result) =
                retryable_quality_result(&buffered[parsed.header_end..], retry_policy)
            {
                return Ok(result);
            }
            let completion_tokens =
                parse_completion_tokens_from_json_body(&buffered[parsed.header_end..]);
            tcp_stream.write_all(&buffered).await?;
            let _ = tcp_stream.shutdown().await;
            return Ok(RouteAttemptResult::Delivered {
                status_code: probe.status_code,
                completion_tokens,
            });
        }
    }

    tcp_stream.write_all(&probe.buffered).await?;
    if let Err(err) = tokio::io::copy(reader, &mut *tcp_stream).await {
        tracing::debug!("response relay ended after headers were committed: {err}");
    }
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_error_http_response_rewrites_llama_error_body() {
        let upstream = b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 52\r\n\r\n{\"type\":\"not_found_error\",\"message\":\"model missing\"}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        let remapped = remap_error_http_response(404, header_end, upstream)
            .expect("llama error should be remapped");
        let remapped_text = String::from_utf8(remapped).unwrap();

        assert!(remapped_text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(remapped_text.contains("\r\nContent-Type: application/json\r\n"));
        assert!(remapped_text.contains("\"type\":\"invalid_request_error\""));
        assert!(remapped_text.contains("\"code\":\"model_not_found\""));
        assert!(remapped_text.contains("\"message\":\"model missing\""));
    }

    #[test]
    fn test_remap_error_http_response_keeps_openai_error_passthrough() {
        let upstream = b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 110\r\n\r\n{\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_value\"}}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        assert!(remap_error_http_response(400, header_end, upstream).is_none());
    }
}
