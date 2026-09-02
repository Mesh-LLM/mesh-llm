use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_token_usage_from_json_body,
    retryable_quality_result,
};
use super::probe::{
    ParsedResponseHeaders, ResponseProbe, read_response_chunk, try_parse_response_headers,
};
use crate::logging::{ArtifactUnavailableReason, OpenAiRouteObserver};
use crate::network::openai::client_stream::ClientStream;
use anyhow::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

const MAX_ERROR_RESPONSE_BYTES: usize = 256 * 1024;

fn http_body(response: &[u8]) -> &[u8] {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map_or(&[][..], |header_end| &response[header_end + 4..])
}

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
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    route_observer: OpenAiRouteObserver<'_>,
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
    let media_kind = try_parse_response_headers(&outgoing)
        .ok()
        .flatten()
        .and_then(|headers| headers.content_type);
    route_observer.capture_response_body(http_body(&outgoing), media_kind.as_deref());
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code,
        usage: None,
    })
}

pub(in crate::network::openai::response) async fn relay_success_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    parsed: ParsedResponseHeaders,
    retry_policy: ResponseRetryPolicy,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if let Some(content_length) = parsed.content_length {
        const MAX_SUCCESS_METRICS_BODY_BYTES: usize = 1024 * 1024;
        if content_length <= MAX_SUCCESS_METRICS_BODY_BYTES {
            let mut buffered = probe.buffered;
            let body_end = parsed
                .header_end
                .checked_add(content_length)
                .ok_or_else(|| anyhow::anyhow!("upstream Content-Length overflow"))?;
            while buffered.len() < body_end {
                read_response_chunk(reader, &mut buffered).await?;
            }
            let body = &buffered[parsed.header_end..body_end];
            if let Some(result) = retryable_quality_result(body, retry_policy) {
                return Ok(result);
            }
            let usage = parse_token_usage_from_json_body(body);
            // Reads may include bytes beyond the declared HTTP body. Only the
            // declared response is client-visible and capturable.
            tcp_stream.write_all(&buffered[..body_end]).await?;
            route_observer.capture_response_body(body, parsed.content_type.as_deref());
            let _ = tcp_stream.shutdown().await;
            return Ok(RouteAttemptResult::Delivered {
                status_code: probe.status_code,
                usage,
            });
        }
    }

    tcp_stream.write_all(&probe.buffered).await?;
    route_observer.capture_response_unavailable(ArtifactUnavailableReason::ResponseBodyNotBounded);
    if let Err(err) = tokio::io::copy(reader, &mut *tcp_stream).await {
        tracing::debug!("response relay ended after headers were committed: {err}");
    }
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        usage: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{OpenAiArtifactCapture, OpenAiRouteObserver};
    use mesh_llm_events::logging::identifiers::RequestId;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    type CaptureRecord = (String, Vec<u8>, Option<String>);

    #[derive(Default)]
    struct Captures(Mutex<Vec<CaptureRecord>>);

    impl OpenAiArtifactCapture for Captures {
        fn capture_body(
            &self,
            _request_id: RequestId,
            kind: &'static str,
            content: &[u8],
            media_kind: Option<&str>,
        ) {
            self.0.lock().unwrap().push((
                kind.to_string(),
                content.to_vec(),
                media_kind.map(str::to_owned),
            ));
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            _reason: ArtifactUnavailableReason,
        ) {
        }
    }

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

    #[tokio::test]
    async fn relay_success_captures_client_visible_non_stream_body() {
        let body = br#"{"id":"chatcmpl-safe","usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let captured = Arc::new(Captures::default());
        let captures: Arc<dyn OpenAiArtifactCapture> = captured.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let observer = OpenAiRouteObserver::capture_test_observer(RequestId::new(), &captures);
            relay_success_response(
                &mut client,
                &mut upstream_reader,
                ResponseProbe {
                    buffered: header.clone().into_bytes(),
                    header_end: header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ParsedResponseHeaders {
                    header_end: header.len(),
                    status_code: 200,
                    content_length: Some(body.len()),
                    content_type: Some("application/json".to_owned()),
                },
                ResponseRetryPolicy::next_target_available(false),
                observer,
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut client_response = Vec::new();
        socket.read_to_end(&mut client_response).await.unwrap();
        task.await.unwrap();

        let captures = captured.0.lock().unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].0, "response");
        assert_eq!(captures[0].1, body);
        assert_eq!(captures[0].2.as_deref(), Some("application/json"));
        assert!(client_response.ends_with(body));
    }

    #[tokio::test]
    async fn relay_success_excludes_bytes_read_past_declared_content_length() {
        let body = br#"{"id":"chatcmpl-safe"}"#;
        let overread = b"NEXT-RESPONSE-MUST-NOT-BE-CAPTURED";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut buffered = header.clone().into_bytes();
        buffered.extend_from_slice(body);
        buffered.extend_from_slice(overread);
        let captured = Arc::new(Captures::default());
        let captures: Arc<dyn OpenAiArtifactCapture> = captured.clone();
        let task_header = header.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let observer = OpenAiRouteObserver::capture_test_observer(RequestId::new(), &captures);
            let mut empty_reader = tokio::io::empty();
            relay_success_response(
                &mut client,
                &mut empty_reader,
                ResponseProbe {
                    buffered,
                    header_end: task_header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ParsedResponseHeaders {
                    header_end: task_header.len(),
                    status_code: 200,
                    content_length: Some(body.len()),
                    content_type: Some("application/json; charset=utf-8".to_owned()),
                },
                ResponseRetryPolicy::next_target_available(false),
                observer,
            )
            .await
            .unwrap();
        });
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut client_response = Vec::new();
        socket.read_to_end(&mut client_response).await.unwrap();
        task.await.unwrap();

        assert_eq!(client_response, [header.as_bytes(), body].concat());
        let captures = captured.0.lock().unwrap();
        assert_eq!(captures[0].1, body);
        assert!(
            !captures[0]
                .1
                .windows(overread.len())
                .any(|part| part == overread)
        );
    }
}
