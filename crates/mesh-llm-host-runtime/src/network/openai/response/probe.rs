use super::common::is_retryable_context_overflow_response;
use anyhow::{Context, Result, anyhow, bail};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::network::openai::request_parse::{MAX_HEADER_BYTES, MAX_HEADERS};

const MAX_RESPONSE_BODY_PREVIEW_BYTES: usize = 4 * 1024;

pub(in crate::network::openai::response) struct ParsedResponseHeaders {
    pub(in crate::network::openai::response) header_end: usize,
    pub(in crate::network::openai::response) status_code: u16,
    pub(in crate::network::openai::response) content_length: Option<usize>,
    pub(in crate::network::openai::response) content_type: Option<String>,
}

#[derive(Clone, Copy)]
pub(in crate::network::openai::response) struct ResponseBodyReadLimits {
    pub(in crate::network::openai::response) max_body_bytes: usize,
    pub(in crate::network::openai::response) idle_timeout: Duration,
}

#[derive(Clone)]
pub(in crate::network::openai::response) struct ResponseProbe {
    pub(in crate::network::openai::response) buffered: Vec<u8>,
    pub(in crate::network::openai::response) header_end: usize,
    pub(in crate::network::openai::response) status_code: u16,
    pub(in crate::network::openai::response) retryable_context_overflow: bool,
}

/// Remote first-byte timeout: 5 minutes. This covers the full round trip
/// through the QUIC tunnel including remote prefill. Concurrent requests
/// on a loaded host can legitimately take minutes. A truly dead QUIC
/// connection will reset/error much faster than this (QUIC idle timeout,
/// connection loss detection). The old 60s default caused spurious 503s
/// when the remote host was alive but busy.
fn response_first_byte_timeout() -> Duration {
    Duration::from_secs(5 * 60)
}

pub(in crate::network::openai::response) fn response_is_event_stream(
    headers: &ParsedResponseHeaders,
) -> bool {
    headers
        .content_type
        .as_deref()
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or(value)
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false)
}

pub(in crate::network::openai::response) fn try_parse_response_headers(
    buf: &[u8],
) -> Result<Option<ParsedResponseHeaders>> {
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut response = httparse::Response::new(&mut headers_buf);
    match response.parse(buf) {
        Ok(httparse::Status::Complete(header_end)) => {
            let mut content_length = None;
            let mut content_type = None;
            for header in response.headers.iter() {
                if header.name.eq_ignore_ascii_case("content-length") {
                    let value = std::str::from_utf8(header.value)
                        .context("invalid response Content-Length encoding")?;
                    content_length =
                        Some(value.trim().parse::<usize>().with_context(|| {
                            format!("invalid response Content-Length: {value}")
                        })?);
                } else if header.name.eq_ignore_ascii_case("content-type") {
                    content_type = Some(
                        std::str::from_utf8(header.value)
                            .context("invalid response Content-Type encoding")?
                            .trim()
                            .to_string(),
                    );
                }
            }
            Ok(Some(ParsedResponseHeaders {
                header_end,
                status_code: response.code.unwrap_or(0),
                content_length,
                content_type,
            }))
        }
        Ok(httparse::Status::Partial) => Ok(None),
        Err(err) => Err(anyhow!("HTTP response parse error: {err}")),
    }
}

/// Read the next chunk of HTTP response data without any timeout.
/// Used for continuation reads after the first byte has already arrived.
pub(in crate::network::openai::response) async fn read_response_chunk<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
) -> Result<usize> {
    let mut chunk = [0u8; 8192];
    let read_result = reader.read(&mut chunk).await?;
    if read_result == 0 {
        bail!("unexpected EOF while reading HTTP response");
    }
    buf.extend_from_slice(&chunk[..read_result]);
    Ok(read_result)
}

pub(in crate::network::openai::response) async fn read_transformed_response_body<
    R: AsyncRead + Unpin,
>(
    reader: &mut R,
    buffered: &mut Vec<u8>,
    header_end: usize,
    content_length: Option<usize>,
    limits: ResponseBodyReadLimits,
) -> Result<usize> {
    if header_end > buffered.len() {
        bail!("invalid HTTP response header boundary");
    }
    let buffered_body_bytes = buffered.len() - header_end;
    if buffered_body_bytes > limits.max_body_bytes {
        bail!(
            "upstream success response body exceeds {} bytes",
            limits.max_body_bytes
        );
    }

    let expected_end = content_length
        .map(|content_length| {
            if content_length > limits.max_body_bytes {
                bail!(
                    "upstream success response Content-Length exceeds {} bytes",
                    limits.max_body_bytes
                );
            }
            header_end
                .checked_add(content_length)
                .ok_or_else(|| anyhow!("upstream success response Content-Length overflow"))
        })
        .transpose()?;

    loop {
        if let Some(expected_end) = expected_end
            && buffered.len() >= expected_end
        {
            return Ok(expected_end);
        }

        let mut chunk = [0u8; 8192];
        let read_result = tokio::time::timeout(limits.idle_timeout, reader.read(&mut chunk))
            .await
            .context("upstream success response body idle timeout")??;
        if read_result == 0 {
            return expected_end.map_or_else(
                || Ok(buffered.len()),
                |_| Err(anyhow!("unexpected EOF while reading HTTP response body")),
            );
        }
        let next_body_bytes = buffered
            .len()
            .saturating_sub(header_end)
            .saturating_add(read_result);
        if next_body_bytes > limits.max_body_bytes {
            bail!(
                "upstream success response body exceeds {} bytes",
                limits.max_body_bytes
            );
        }
        buffered.extend_from_slice(&chunk[..read_result]);
    }
}

pub(in crate::network::openai::response) async fn probe_http_response<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<ResponseProbe> {
    probe_http_response_with_timeout(reader, response_first_byte_timeout()).await
}

/// Like `probe_http_response` but with a much longer timeout suitable for
/// the local OpenAI surface. Prefill on a busy or slow machine can
/// legitimately take minutes (large prompts, concurrent slot contention,
/// slower hardware). We still bound the wait to catch a truly wedged local
/// runtime path.
pub(in crate::network::openai::response) async fn probe_http_response_local<
    R: AsyncRead + Unpin,
>(
    reader: &mut R,
) -> Result<ResponseProbe> {
    probe_http_response_with_timeout(reader, local_response_first_byte_timeout()).await
}

/// Local OpenAI surface timeout: 10 minutes. This is a safety net for a wedged
/// local runtime path, not a latency budget. Normal prefill even on slow
/// hardware with large prompts and concurrent slots completes well within this
/// window.
fn local_response_first_byte_timeout() -> Duration {
    Duration::from_secs(10 * 60)
}

pub(in crate::network::openai::response) async fn probe_http_response_with_timeout<
    R: AsyncRead + Unpin,
>(
    reader: &mut R,
    timeout: Duration,
) -> Result<ResponseProbe> {
    let started = Instant::now();
    let mut buffered = Vec::with_capacity(8192);
    let parsed = loop {
        if let Some(parsed) = try_parse_response_headers(&buffered)? {
            break parsed;
        }
        let first_read = buffered.is_empty();
        if first_read {
            let mut chunk = [0u8; 8192];
            let read_result = tokio::time::timeout(timeout, reader.read(&mut chunk))
                .await
                .map_err(|_| {
                    anyhow!(
                        "upstream sent no response within {:.3}s",
                        timeout.as_secs_f64()
                    )
                })??;
            if read_result == 0 {
                bail!("unexpected EOF while reading HTTP response");
            }
            buffered.extend_from_slice(&chunk[..read_result]);
        } else {
            read_response_chunk(reader, &mut buffered).await?;
        }
        if buffered.len() > MAX_HEADER_BYTES {
            bail!("HTTP response headers exceed {MAX_HEADER_BYTES} bytes");
        }
    };

    let preview_len = if parsed.status_code == 400 {
        parsed
            .content_length
            .map(|value| value.min(MAX_RESPONSE_BODY_PREVIEW_BYTES))
            .unwrap_or(0)
    } else {
        0
    };
    while buffered.len() < parsed.header_end + preview_len {
        read_response_chunk(reader, &mut buffered).await?;
    }

    let retryable_context_overflow = parsed.status_code == 400
        && preview_len > 0
        && is_retryable_context_overflow_response(
            &buffered[parsed.header_end..parsed.header_end + preview_len],
        );
    tracing::debug!(
        status_code = parsed.status_code,
        header_bytes = parsed.header_end,
        probe_ms = started.elapsed().as_millis(),
        "openai transport: upstream response probe complete"
    );

    Ok(ResponseProbe {
        buffered,
        header_end: parsed.header_end,
        status_code: parsed.status_code,
        retryable_context_overflow,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::openai::response::common::is_timeout_error;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn transformed_response_rejects_oversized_content_length_before_reading() {
        let (_writer, mut reader) = tokio::io::duplex(64);
        let mut buffered = b"HTTP/1.1 200 OK\r\n\r\n".to_vec();
        let header_end = buffered.len();

        let error = read_transformed_response_body(
            &mut reader,
            &mut buffered,
            header_end,
            Some(9),
            ResponseBodyReadLimits {
                max_body_bytes: 8,
                idle_timeout: Duration::from_secs(1),
            },
        )
        .await
        .expect_err("oversized declared body must be rejected");

        assert!(error.to_string().contains("Content-Length exceeds 8 bytes"));
    }

    #[tokio::test]
    async fn transformed_response_rejects_oversized_unframed_body() {
        let (mut writer, mut reader) = tokio::io::duplex(64);
        writer.write_all(b"123456789").await.unwrap();
        drop(writer);
        let mut buffered = b"HTTP/1.1 200 OK\r\n\r\n".to_vec();
        let header_end = buffered.len();

        let error = read_transformed_response_body(
            &mut reader,
            &mut buffered,
            header_end,
            None,
            ResponseBodyReadLimits {
                max_body_bytes: 8,
                idle_timeout: Duration::from_secs(1),
            },
        )
        .await
        .expect_err("oversized unframed body must be rejected");

        assert!(error.to_string().contains("body exceeds 8 bytes"));
    }

    #[tokio::test]
    async fn transformed_response_body_read_has_idle_timeout() {
        let (_writer, mut reader) = tokio::io::duplex(64);
        let mut buffered = b"HTTP/1.1 200 OK\r\n\r\n".to_vec();
        let header_end = buffered.len();

        let error = read_transformed_response_body(
            &mut reader,
            &mut buffered,
            header_end,
            None,
            ResponseBodyReadLimits {
                max_body_bytes: 8,
                idle_timeout: Duration::from_millis(10),
            },
        )
        .await
        .expect_err("idle body read must time out");

        assert!(is_timeout_error(&error), "unexpected error: {error:#}");
    }
    #[tokio::test(start_paused = true)]
    async fn test_probe_http_response_local_tolerates_slow_first_byte() {
        use tokio::io::AsyncWriteExt;

        let (client, mut server) = tokio::io::duplex(4096);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let _ = server
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                .await;
        });

        let mut reader = client;
        let result = super::probe_http_response_local(&mut reader).await;
        assert!(
            result.is_ok(),
            "probe_http_response_local should NOT timeout for slow local responses"
        );
        assert_eq!(result.unwrap().status_code, 200);
    }
}
