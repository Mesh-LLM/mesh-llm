use super::common::{ResponseRetryPolicy, RouteAttemptResult, parse_token_usage_from_json_body};
use super::probe::{ResponseProbe, response_is_event_stream, try_parse_response_headers};
use super::relay::{relay_error_response, relay_success_response};
use crate::logging::{OpenAiRouteObserver, OpenAiStreamArtifactCapture};
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::response_adapter;
use crate::network::openai::tool_call_ids::ChatStreamNormalizationState;
use anyhow::{Context, Result, anyhow};
use mesh_llm_events::logging::events::TokenUsage;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

async fn write_captured_sse_event(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    event: Option<&str>,
    data: &str,
) -> std::io::Result<()> {
    if let Some(capture) = capture {
        capture.push(&response_adapter::sse_frame(event, data));
    }
    response_adapter::write_chunked_sse_event(tcp_stream, event, data).await
}

struct ResponsesStreamRelayState {
    created_at: i64,
    response_id: String,
    item_id: String,
    model: String,
    output_text: String,
    usage: Option<serde_json::Value>,
    observed_usage: Option<TokenUsage>,
    sequence_number: i32,
    created_emitted: bool,
    output_item_emitted: bool,
}

impl ResponsesStreamRelayState {
    fn new() -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or(0);
        Self {
            created_at,
            response_id: format!("resp_{created_at}"),
            item_id: format!("msg_{created_at}"),
            model: String::new(),
            output_text: String::new(),
            usage: None,
            observed_usage: None,
            sequence_number: 0,
            created_emitted: false,
            output_item_emitted: false,
        }
    }

    fn next_sequence_number(&mut self) -> i32 {
        self.sequence_number = self.sequence_number.saturating_add(1);
        self.sequence_number
    }
}

pub(in crate::network::openai::response) async fn relay_normalized_chat_completion_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
        return relay_error_response(tcp_stream, reader, probe, route_observer).await;
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
            route_observer,
        )
        .await;
    }

    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ChatStreamNormalizationState::default();
    let mut observed_usage = None;
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    tcp_stream.write_all(header.as_bytes()).await?;
    let mut response_capture = route_observer.begin_stream_response_capture();
    route_observer.stream_started(None);

    let mut done_seen = false;
    let mut first_chunk_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let data_lines = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n");
            if data == "[DONE]" {
                done_seen = true;
                write_captured_sse_event(tcp_stream, &mut response_capture, None, "[DONE]").await?;
                break;
            }

            if let Some(usage) = parse_token_usage_from_json_body(data.as_bytes()) {
                observed_usage = Some(usage);
            }
            let normalized = state.normalize_data(&data);
            write_captured_sse_event(tcp_stream, &mut response_capture, None, &normalized).await?;
            if first_chunk_seen {
                route_observer.stream_chunk();
            } else {
                route_observer.stream_first_token();
                first_chunk_seen = true;
            }
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    if !done_seen {
        route_observer.stream_error("upstream_stream_incomplete");
        return Err(anyhow!("upstream chat stream ended before [DONE]"));
    }
    route_observer.complete_stream_response_capture(response_capture);
    route_observer.stream_completed(observed_usage);
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage: observed_usage,
    })
}

pub(in crate::network::openai::response) async fn relay_translated_responses_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    fn should_parse_stream_chunk(data: &str, model_missing: bool, usage_missing: bool) -> bool {
        model_missing
            || usage_missing
            || data.contains("\"delta\"")
            || data.contains("\"content\"")
            || data.contains("\"logprobs\"")
            || data.contains("\"usage\"")
    }

    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
        return relay_error_response(tcp_stream, reader, probe, route_observer).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ResponsesStreamRelayState::new();
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    tcp_stream.write_all(header.as_bytes()).await?;
    let mut response_capture = route_observer.begin_stream_response_capture();
    route_observer.stream_started(None);

    let mut done_seen = false;
    let mut first_chunk_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let data_lines = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n");
            if data == "[DONE]" {
                done_seen = true;
                break;
            }

            if !should_parse_stream_chunk(&data, state.model.is_empty(), state.usage.is_none()) {
                continue;
            }

            process_translated_responses_frame(
                tcp_stream,
                &mut response_capture,
                &mut state,
                &data,
            )
            .await?;
            if first_chunk_seen {
                route_observer.stream_chunk();
            } else {
                route_observer.stream_first_token();
                first_chunk_seen = true;
            }
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        // Normalize CRLF so frame parsing works for both LF and CRLF upstreams
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    if !done_seen {
        route_observer.stream_error("upstream_stream_incomplete");
        return Err(anyhow!("upstream Responses stream ended before [DONE]"));
    }
    finish_translated_responses_stream(tcp_stream, &mut response_capture, &mut state).await?;
    write_captured_sse_event(tcp_stream, &mut response_capture, Some("done"), "[DONE]").await?;
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    route_observer.complete_stream_response_capture(response_capture);
    route_observer.stream_completed(state.observed_usage);
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage: state.observed_usage,
    })
}

async fn process_translated_responses_frame(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    data: &str,
) -> Result<()> {
    let chunk = openai_frontend::parse_chat_stream_chunk(data)
        .context("parse typed upstream chat stream chunk")?;
    update_translated_responses_model(state, &chunk);
    emit_translated_response_created(tcp_stream, capture, state).await?;
    emit_translated_reasoning_delta(tcp_stream, capture, state, &chunk).await?;
    emit_translated_output_delta(tcp_stream, capture, state, &chunk).await?;
    update_translated_responses_usage(state, &chunk);
    Ok(())
}

fn update_translated_responses_model(
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) {
    if let Some(chunk_model) = chunk.model.as_deref().filter(|_| state.model.is_empty()) {
        state.model = chunk_model.to_string();
    }
}

async fn emit_translated_response_created(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.created_emitted || state.model.is_empty() {
        return Ok(());
    }
    let sequence_number = state.next_sequence_number();
    let created = serde_json::to_string(
        &response_adapter::responses_stream_created_event_with_sequence(
            &state.model,
            state.created_at,
            sequence_number,
        ),
    )
    .context("serialize response.created stream event")?;
    write_captured_sse_event(tcp_stream, capture, Some("response.created"), &created).await?;
    state.created_emitted = true;
    Ok(())
}

async fn emit_translated_reasoning_delta(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) -> Result<()> {
    let Some(delta) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.as_ref())
        .and_then(|delta| delta.reasoning_content.as_deref())
    else {
        return Ok(());
    };
    let sequence_number = state.next_sequence_number();
    let event = serde_json::to_string(
        &response_adapter::responses_stream_reasoning_delta_event_with_sequence(
            &state.item_id,
            delta,
            sequence_number,
        ),
    )
    .context("serialize response.reasoning_text.delta event")?;
    write_captured_sse_event(
        tcp_stream,
        capture,
        Some("response.reasoning_text.delta"),
        &event,
    )
    .await?;
    Ok(())
}

async fn emit_translated_output_delta(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) -> Result<()> {
    let Some(delta) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.as_ref())
        .and_then(|delta| delta.content.as_deref())
    else {
        return Ok(());
    };
    emit_translated_output_item_prelude(tcp_stream, capture, state).await?;
    let logprobs = chunk
        .choices
        .first()
        .and_then(|choice| choice.logprobs.clone());
    state.output_text.push_str(delta);
    let sequence_number = state.next_sequence_number();
    let event = serde_json::to_string(
        &response_adapter::responses_stream_delta_event_with_logprobs_and_sequence(
            &state.item_id,
            delta,
            logprobs,
            sequence_number,
        ),
    )
    .context("serialize response.output_text.delta event")?;
    write_captured_sse_event(
        tcp_stream,
        capture,
        Some("response.output_text.delta"),
        &event,
    )
    .await?;
    Ok(())
}

async fn emit_translated_output_item_prelude(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.output_item_emitted {
        return Ok(());
    }
    let item_added_sequence_number = state.next_sequence_number();
    let item_added =
        serde_json::to_string(&response_adapter::responses_stream_output_item_added_event(
            &state.item_id,
            item_added_sequence_number,
        ))
        .context("serialize response.output_item.added event")?;
    write_captured_sse_event(
        tcp_stream,
        capture,
        Some("response.output_item.added"),
        &item_added,
    )
    .await?;
    let part_added_sequence_number = state.next_sequence_number();
    let part_added = serde_json::to_string(
        &response_adapter::responses_stream_content_part_added_event(
            &state.item_id,
            part_added_sequence_number,
        ),
    )
    .context("serialize response.content_part.added event")?;
    write_captured_sse_event(
        tcp_stream,
        capture,
        Some("response.content_part.added"),
        &part_added,
    )
    .await?;
    state.output_item_emitted = true;
    Ok(())
}

fn update_translated_responses_usage(
    state: &mut ResponsesStreamRelayState,
    chunk: &openai_frontend::responses::ChatCompletionStreamChunk,
) {
    if let Some(usage) = chunk.usage.as_ref() {
        state.usage = Some(response_adapter::stream_usage_to_responses_usage(usage));
        if let Some(authoritative) = TokenUsage::from_counts(
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        ) {
            state.observed_usage = Some(
                authoritative.with_cached_prompt_tokens(
                    usage
                        .prompt_tokens_details
                        .as_ref()
                        .and_then(|details| details.cached_tokens),
                ),
            );
        }
    }
}

async fn finish_translated_responses_stream(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    emit_translated_fallback_created(tcp_stream, capture, state).await?;
    emit_translated_output_item_prelude(tcp_stream, capture, state).await?;
    let text_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        capture,
        Some("response.output_text.done"),
        serde_json::to_string(
            &response_adapter::responses_stream_text_done_event_with_sequence(
                &state.item_id,
                &state.output_text,
                text_done_sequence_number,
            ),
        )
        .context("serialize response.output_text.done event")?,
    )
    .await?;
    let content_part_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        capture,
        Some("response.content_part.done"),
        serde_json::to_string(&response_adapter::responses_stream_content_part_done_event(
            &state.item_id,
            &state.output_text,
            content_part_done_sequence_number,
        ))
        .context("serialize response.content_part.done event")?,
    )
    .await?;
    let output_item_done_sequence_number = state.next_sequence_number();
    emit_translated_stream_done_event(
        tcp_stream,
        capture,
        Some("response.output_item.done"),
        serde_json::to_string(&response_adapter::responses_stream_output_item_done_event(
            &state.item_id,
            &state.output_text,
            output_item_done_sequence_number,
        ))
        .context("serialize response.output_item.done event")?,
    )
    .await?;
    let completed_sequence_number = state.next_sequence_number();
    let completed = serde_json::to_string(
        &response_adapter::responses_stream_completed_event_with_sequence(
            &state.response_id,
            state.created_at,
            &state.model,
            &state.item_id,
            &state.output_text,
            state.usage.clone(),
            completed_sequence_number,
        ),
    )
    .context("serialize response.completed event")?;
    write_captured_sse_event(tcp_stream, capture, Some("response.completed"), &completed).await?;
    Ok(())
}

async fn emit_translated_fallback_created(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
) -> Result<()> {
    if state.created_emitted {
        return Ok(());
    }
    let sequence_number = state.next_sequence_number();
    let created = serde_json::to_string(
        &response_adapter::responses_stream_created_event_with_sequence(
            &state.model,
            state.created_at,
            sequence_number,
        ),
    )
    .context("serialize response.created stream event")?;
    write_captured_sse_event(tcp_stream, capture, Some("response.created"), &created).await?;
    state.created_emitted = true;
    Ok(())
}

async fn emit_translated_stream_done_event(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    event_name: Option<&str>,
    payload: String,
) -> Result<()> {
    write_captured_sse_event(tcp_stream, capture, event_name, &payload).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{ArtifactUnavailableReason, OpenAiArtifactCapture};
    use mesh_llm_events::logging::identifiers::RequestId;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    #[derive(Debug, Eq, PartialEq)]
    enum CapturedArtifact {
        Body(Vec<u8>, Option<String>),
        Unavailable(ArtifactUnavailableReason),
    }

    #[derive(Default)]
    struct Captures(Mutex<Vec<CapturedArtifact>>);

    impl OpenAiArtifactCapture for Captures {
        fn capture_body(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            content: &[u8],
            media_kind: Option<&str>,
        ) {
            self.0.lock().unwrap().push(CapturedArtifact::Body(
                content.to_vec(),
                media_kind.map(str::to_owned),
            ));
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            reason: ArtifactUnavailableReason,
        ) {
            self.0
                .lock()
                .unwrap()
                .push(CapturedArtifact::Unavailable(reason));
        }
    }

    #[tokio::test]
    async fn relay_translated_responses_stream_emits_one_delta_per_upstream_chunk() {
        use tokio::io::AsyncWriteExt;

        // ── upstream side: a writer we can push chat.completion.chunk frames into
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);

        // ── client-side TCP stream to capture relay output
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec(),
                header_end: b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".len(),
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_translated_responses_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                OpenAiRouteObserver::default(),
            )
            .await
            .expect("relay")
        });

        // ── push three separate delta chunks plus a finish chunk
        for delta in ["Hello", " world", "!"] {
            let chunk = format!(
                r#"{{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{{"index":0,"delta":{{"content":"{delta}"}},"finish_reason":null}}]}}"#
            );
            let framed = format!("data: {}\n\n", chunk);
            upstream_writer.write_all(framed.as_bytes()).await.unwrap();
            // tiny gap so the relay actually services the chunk
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let finish = r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":13,"total_tokens":18}}"#;
        upstream_writer
            .write_all(format!("data: {}\n\n", finish).as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        // ── read everything the relay wrote
        let mut client = ClientStream::connect(addr).await.unwrap();
        use tokio::io::AsyncReadExt;
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");

        assert!(output.starts_with(b"HTTP/1.1 200 OK\r\n"));

        assert_eq!(
            route_result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: Some(TokenUsage {
                    prompt_tokens: Some(5),
                    cached_prompt_tokens: None,
                    completion_tokens: Some(13),
                    total_tokens: Some(18),
                }),
            }
        );

        let body = String::from_utf8_lossy(&output);
        let delta_count = body
            .matches("\"type\":\"response.output_text.delta\"")
            .count();
        assert!(
            delta_count >= 3,
            "expected ≥3 delta events, one per upstream chunk; got {delta_count}.\nBody:\n{body}"
        );
        assert!(
            body.contains("\"type\":\"response.completed\""),
            "missing completed event:\n{body}"
        );
    }

    #[tokio::test]
    async fn relay_normalized_chat_completion_stream_returns_completion_tokens() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let capture = Arc::new(Captures::default());
        let observer_capture: Arc<dyn OpenAiArtifactCapture> = capture.clone();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                OpenAiRouteObserver::capture_test_observer(RequestId::new(), &observer_capture),
            )
            .await
            .expect("relay")
        });

        let usage_chunk = r#"{"id":"chatcmpl-y","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":7,"total_tokens":9}}"#;
        upstream_writer
            .write_all(format!("data: {usage_chunk}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");

        assert!(output.starts_with(b"HTTP/1.1 200 OK\r\n"));

        assert_eq!(
            route_result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: Some(TokenUsage {
                    prompt_tokens: Some(2),
                    cached_prompt_tokens: None,
                    completion_tokens: Some(7),
                    total_tokens: Some(9),
                }),
            }
        );
        assert!(String::from_utf8_lossy(&output).contains("hello"));
        let captured = capture.0.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let CapturedArtifact::Body(body, media_kind) = &captured[0] else {
            panic!("completed bounded stream should retain a response body")
        };
        let body = String::from_utf8_lossy(body);
        assert!(body.contains("hello"));
        assert!(body.contains("data: [DONE]"));
        assert_eq!(media_kind.as_deref(), Some("text/event-stream"));
    }

    #[tokio::test]
    async fn incomplete_normalized_stream_is_an_error_without_synthesized_done() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            super::super::dispatch::relay_attempted_response(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                super::super::dispatch::RelayAttemptContext {
                    request_id: RequestId::new(),
                    disconnect_message: "test client disconnected",
                    commit_message: "test stream relay failed",
                    route_observer: OpenAiRouteObserver::default(),
                },
                ResponseRetryPolicy::next_target_available(false),
                crate::network::openai::request_normalize::ResponseAdapter::OpenAiChatCompletionsStream,
            )
            .await
        });

        upstream_writer
            .write_all(b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":7,\"total_tokens\":9}}\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");
        let body = String::from_utf8_lossy(&output);

        assert_eq!(
            route_result,
            RouteAttemptResult::CommittedStreamFailure { status_code: 200 }
        );
        assert!(!body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn incomplete_translated_stream_is_an_error_without_completed_tail() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_translated_responses_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                OpenAiRouteObserver::default(),
            )
            .await
        });

        upstream_writer
            .write_all(b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":7,\"total_tokens\":9}}\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");
        let body = String::from_utf8_lossy(&output);

        assert!(route_result.is_err());
        assert!(!body.contains("response.completed"));
        assert!(!body.contains("data: [DONE]"));
    }
}
