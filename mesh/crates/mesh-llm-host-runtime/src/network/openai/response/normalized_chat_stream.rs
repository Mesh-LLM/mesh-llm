//! Client-facing chat SSE normalization. Billing watermarks remain on the peer transport.
use super::normalized_stream_completion::finish_normalized_chat_stream;
use super::*;
use crate::network::openai::response::stream_resume;
use serde_json::Value;

fn observe_output_progress(observer: OpenAiRouteObserver<'_>, errored: bool, first: &mut bool) {
    if errored {
        return;
    }
    if *first {
        observer.stream_chunk();
    } else {
        observer.stream_first_token();
        *first = true;
    }
}

fn frame_data(frame: &str) -> Option<String> {
    let lines = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn hold_usage_frame(data: &str, pending: &mut Option<String>) -> bool {
    let usage_only = serde_json::from_str::<serde_json::Value>(data).is_ok_and(|value| {
        value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
            && value.get("usage").is_some_and(serde_json::Value::is_object)
            && value.get("error").is_none()
    });
    // A usage-only frame between deltas is a billing watermark, not the
    // terminal usage frame OpenAI clients expect. Keep only a trailing one.
    // Without final backend totals, this remains the last partial watermark.
    if usage_only {
        *pending = Some(data.to_owned());
    } else if !data.is_empty() && !is_finish_frame(data) {
        *pending = None;
    }
    usage_only
}

fn is_finish_frame(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data).is_ok_and(|value| {
        value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|choices| {
                !choices.is_empty()
                    && choices.iter().all(|choice| {
                        choice
                            .get("finish_reason")
                            .is_some_and(serde_json::Value::is_string)
                    })
            })
    })
}

async fn flush_pending_usage(
    client: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    pending: &mut Option<String>,
) -> Result<()> {
    if let Some(data) = pending.take() {
        write_captured_sse_event(client, capture, None, &data).await?;
    }
    Ok(())
}

type StreamIdentity = (Option<Value>, Option<Value>, Option<Value>);

/// Whether a continuation attempt can write into the caller's open stream:
/// only a successful event stream can, never an error or a plain body.
fn begin_continuation_relay(probe: &ResponseProbe) -> bool {
    let streams = try_parse_response_headers(&probe.buffered)
        .ok()
        .flatten()
        .is_some_and(|parsed| response_is_event_stream(&parsed));
    let relayable = (200..300).contains(&probe.status_code) && streams;
    if relayable {
        stream_resume::finish_continuation();
    }
    relayable
}

/// Note the first chunk's identity and count the chunks that carried content.
fn track_stream_identity(
    normalized: &str,
    identity: &mut Option<StreamIdentity>,
    content_chunks: &mut u32,
) {
    let Ok(chunk) = serde_json::from_str::<Value>(normalized) else {
        return;
    };
    if identity.is_none() && chunk.get("choices").is_some() {
        *identity = Some((
            chunk.get("id").cloned(),
            chunk.get("created").cloned(),
            chunk.get("model").cloned(),
        ));
    }
    let has_content = chunk
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .is_some_and(|content| !content.is_empty());
    *content_chunks += u32::from(has_content);
}

/// The frame as the caller gets it: unchanged, or rewritten into the stream
/// it continues; `None` drops it.
fn continue_chunk(
    normalized: String,
    continuation: Option<&stream_resume::StreamContinuation>,
) -> Option<String> {
    match continuation {
        Some(continuation) => stream_resume::rewrite_continuation_chunk(&normalized, continuation),
        None => Some(normalized),
    }
}

/// Read the next upstream bytes. Once output flows and another target can
/// resume the stream, a host silent past the idle limit, or a broken read,
/// counts as an early end (0): an abruptly killed peer is otherwise only
/// noticed at the transport's idle timeout, tens of seconds later. Before the
/// first content the host may still be prefilling, so no limit applies.
async fn read_resumable<R: AsyncRead + Unpin>(
    reader: &mut R,
    chunk: &mut [u8],
    content_chunks: u32,
) -> Result<usize> {
    let resumable = stream_resume::allowed();
    let idle_limit = stream_resume::idle_limit().filter(|_| content_chunks > 0 && resumable);
    let read = match idle_limit {
        Some(limit) => match tokio::time::timeout(limit, reader.read(chunk)).await {
            Ok(read) => read,
            Err(_) => {
                tracing::info!(
                    idle_ms = limit.as_millis(),
                    "openai committed stream went silent past its idle limit"
                );
                Ok(0)
            }
        },
        None => reader.read(chunk).await,
    };
    match read {
        Ok(n) => Ok(n),
        Err(_) if resumable => Ok(0),
        Err(error) => Err(error.into()),
    }
}

/// With another target left and content already sent, hand what the caller
/// received to the route loop, which continues the stream elsewhere.
fn hold_for_resume(
    assembly: &StreamedChatAssembly,
    identity: Option<StreamIdentity>,
    content_chunks: u32,
) -> bool {
    if !stream_resume::allowed() || !assembly.resumable() || assembly.content().is_empty() {
        return false;
    }
    let (id, created, model) = identity.unwrap_or_default();
    stream_resume::record_delivered(stream_resume::DeliveredStream {
        content: assembly.content().to_string(),
        content_chunks,
        id,
        created,
        model,
    });
    true
}

/// Send the response head and start the capture, unless this attempt
/// continues a stream the caller already has open.
async fn open_client_stream(
    tcp_stream: &mut ClientStream,
    header: &str,
    continues_open_stream: bool,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<Option<crate::logging::OpenAiStreamArtifactCapture>> {
    if continues_open_stream {
        return Ok(None);
    }
    tcp_stream.write_all(header.as_bytes()).await?;
    let capture = route_observer.begin_stream_response_capture();
    route_observer.stream_started(None);
    Ok(capture)
}

/// Relay a streaming chat-completions upstream response, normalizing tool-call ids.
pub(in crate::network::openai::response) async fn relay_normalized_chat_completion_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    request_id: RequestId,
) -> Result<RouteAttemptResult> {
    if let Some(result) =
        crate::network::openai::response::common::retry_before_relay(&probe, retry_policy)
    {
        return Ok(result);
    }

    // A continuation writes into the caller's already open stream, so it can
    // neither relay an error response nor a non-stream body into it.
    let continuation = stream_resume::continuation();
    if continuation.is_some() && !begin_continuation_relay(&probe) {
        return Ok(RouteAttemptResult::RetryableUnavailable);
    }

    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
        return relay_error_response(
            tcp_stream,
            reader,
            probe,
            served_by,
            route_observer,
            request_id,
        )
        .await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    if !response_is_event_stream(&parsed) {
        return relay_success_response(
            tcp_stream,
            reader,
            probe,
            parsed,
            retry_policy,
            served_by,
            route_observer,
            request_id,
        )
        .await;
    }

    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ChatStreamNormalizationState::default();
    let mut assembly = StreamedChatAssembly::default();
    let mut observed_usage = None;
    let mut pending_usage = None;
    let mut observed_cache_cost = None;
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n",
    );
    append_capsule_nonce_headers(
        &mut header,
        parsed.client_nonce.as_deref(),
        parsed.nonce_origin.as_deref(),
    );
    parsed.served_model.append_to(&mut header);
    append_mesh_served_by_header(&mut header, served_by);
    append_mesh_request_id_header(&mut header, request_id);
    header.push_str("Connection: close\r\n\r\n");
    let mut response_capture =
        open_client_stream(tcp_stream, &header, continuation.is_some(), route_observer).await?;
    let mut identity: Option<StreamIdentity> = None;
    let mut content_chunks = 0u32;

    let mut done_seen = false;
    let mut first_chunk_seen = false;
    let mut upstream_error_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let Some(data) = frame_data(frame) else {
                continue;
            };
            if data == "[DONE]" {
                done_seen = true;
                flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
                write_captured_sse_event(tcp_stream, &mut response_capture, None, "[DONE]").await?;
                break;
            }

            if !upstream_error_seen && sse_data_frame_is_openai_error(&data) {
                // The upstream backend frames failures as OpenAI error bodies
                // inside a 200 stream. Relay the frame untouched, but do not
                // let it count as stream progress or terminal success.
                upstream_error_seen = true;
                flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
            }
            if let Some(usage) = parse_token_usage_from_json_body(data.as_bytes()) {
                observed_usage = Some(usage);
            }
            observed_cache_cost =
                observed_cache_cost.or_else(|| parse_cache_cost_from_json_body(data.as_bytes()));
            let normalized = state.normalize_data(&data);
            assembly.ingest_chunk(&normalized);
            track_stream_identity(&normalized, &mut identity, &mut content_chunks);
            let Some(normalized) = continue_chunk(normalized, continuation.as_ref()) else {
                continue;
            };
            if hold_usage_frame(&normalized, &mut pending_usage) {
                continue;
            }
            write_captured_sse_event(tcp_stream, &mut response_capture, None, &normalized).await?;
            observe_output_progress(route_observer, upstream_error_seen, &mut first_chunk_seen);
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = read_resumable(reader, &mut chunk, content_chunks).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    // Truncated with another target left: keep the caller's stream open and
    // hand what it received to the route loop, which continues it elsewhere.
    if !done_seen && !upstream_error_seen && hold_for_resume(&assembly, identity, content_chunks) {
        route_observer.stream_error("upstream_stream_incomplete");
        return Ok(RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause: CommittedStreamFailureCause::Truncated,
        });
    }

    flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
    finish_normalized_chat_stream(
        tcp_stream,
        route_observer,
        (done_seen, upstream_error_seen),
        response_capture,
        &assembly,
        observed_usage,
        observed_cache_cost,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frames(watermarks: bool) -> String {
        let mut events = vec![
            json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{"}}]},"finish_reason":null}]}),
        ];
        if watermarks {
            events.push(json!({"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":1,"total_tokens":1}}));
        }
        events.push(json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"command\":\"ls\"}"}}]},"finish_reason":null}]}));
        if watermarks {
            events.push(json!({"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":4,"total_tokens":4}}));
        }
        events.push(json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}));
        events.push(json!({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":8}}}));
        events
            .into_iter()
            .map(|v| format!("data: {v}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n"
    }

    async fn relay(body: String, request_id: RequestId) -> (String, Result<RouteAttemptResult>) {
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut client = ClientStream::from(socket);
            let mut reader = std::io::Cursor::new(body.into_bytes());
            relay_normalized_chat_completion_stream(
                &mut client,
                &mut reader,
                ResponseProbe {
                    buffered: header.to_vec(),
                    header_end: header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                request_id,
            )
            .await
        });
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut bytes = Vec::new();
        socket.read_to_end(&mut bytes).await.unwrap();
        (String::from_utf8(bytes).unwrap(), task.await.unwrap())
    }

    #[tokio::test]
    async fn billing_watermarks_do_not_interrupt_tool_arguments() {
        // One request id for both runs: the wire comparison below is
        // byte-for-byte, and every response head echoes the canonical id.
        let request_id = RequestId::new();
        let (paid, paid_result) = relay(frames(true), request_id).await;
        let (free, free_result) = relay(frames(false), request_id).await;
        assert_eq!(paid, free);
        assert_eq!(paid.matches("\"usage\"").count(), 1);
        let RouteAttemptResult::Delivered {
            usage,
            output_digests,
            ..
        } = paid_result.unwrap()
        else {
            panic!("paid stream not delivered")
        };
        let RouteAttemptResult::Delivered {
            usage: free_usage,
            output_digests: free_digests,
            ..
        } = free_result.unwrap()
        else {
            panic!("free stream not delivered")
        };
        assert_eq!(usage, free_usage);
        assert_eq!(usage.unwrap().completion_tokens, Some(4));
        assert_eq!(output_digests, free_digests);
        assert!(
            paid.find("\"finish_reason\":\"tool_calls\"").unwrap()
                < paid.find("\"usage\"").unwrap()
        );
        assert!(paid.find("\"usage\"").unwrap() < paid.find("[DONE]").unwrap());
    }

    #[tokio::test]
    async fn usage_precedes_upstream_error_without_hiding_error() {
        let usage = json!({"choices":[],"usage":{"completion_tokens":2}});
        let error = json!({"error":{"message":"generation failed","type":"backend_error"}});
        let (wire, _) = relay(
            format!("data: {usage}\n\ndata: {error}\n\ndata: [DONE]\n\n"),
            RequestId::new(),
        )
        .await;
        assert!(
            wire.find("\"completion_tokens\":2").unwrap() < wire.find("generation failed").unwrap()
        );
    }

    #[tokio::test]
    async fn eof_flushes_usage_but_remains_truncated() {
        let body = frames(true).replace("data: [DONE]\n\n", "");
        let (wire, result) = relay(body, RequestId::new()).await;
        assert!(result.is_err());
        assert_eq!(wire.matches("\"usage\"").count(), 1);
        assert!(!wire.contains("[DONE]"));
    }

    #[tokio::test]
    async fn totals_before_finish_are_preserved_after_finish() {
        let body = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
        let (wire, result) = relay(body.to_string(), RequestId::new()).await;
        assert!(result.is_ok());
        assert_eq!(wire.matches("\"usage\"").count(), 1);
        assert!(wire.find("tool_calls").unwrap() < wire.find("\"usage\"").unwrap());
    }

    #[tokio::test]
    async fn empty_data_frames_preserve_trailing_usage() {
        let body =
            frames(false).replace("data: [DONE]\n\n", "data:\n\ndata:   \n\ndata: [DONE]\n\n");
        let (wire, result) = relay(body, RequestId::new()).await;
        assert!(matches!(
            result.unwrap(),
            RouteAttemptResult::Delivered { .. }
        ));
        assert_eq!(wire.matches("\"usage\"").count(), 1);
        assert!(wire.contains("\"prompt_tokens\":12"));
        assert!(wire.find("\"usage\"").unwrap() < wire.find("[DONE]").unwrap());
    }

    #[test]
    fn only_usage_only_frames_are_deferred() {
        let mut pending = None;
        assert!(hold_usage_frame(
            r#"{"choices":[],"usage":{"completion_tokens":1}}"#,
            &mut pending
        ));
        assert!(hold_usage_frame(
            r#"{"choices":[],"usage":{"completion_tokens":2}}"#,
            &mut pending
        ));
        assert!(pending.as_ref().unwrap().contains(":2"));
        for data in [
            r#"{"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":2}}"#,
            r#"{"choices":[],"usage":null}"#,
            "not-json",
        ] {
            assert!(!hold_usage_frame(data, &mut pending));
            assert!(pending.is_none());
        }
    }
}
