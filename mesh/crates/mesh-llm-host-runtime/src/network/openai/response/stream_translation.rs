use super::cache_cost::{CacheCostObservation, parse_cache_cost_from_json_body};
use super::common::{
    CommittedStreamFailureCause, ResponseRetryPolicy, RouteAttemptResult,
    parse_token_usage_from_json_body,
};
use super::probe::{
    ResponseProbe, append_capsule_nonce_headers, append_mesh_request_id_header,
    append_mesh_served_by_header, response_is_event_stream, try_parse_response_headers,
};
use super::relay::{relay_error_response, relay_success_response};
use crate::logging::{OpenAiRouteObserver, OpenAiStreamArtifactCapture};
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::response::common::sse_data_frame_is_openai_error;
use crate::network::openai::response_adapter;
use crate::plugin::openai_exchange::ExchangeOutputDigests;
use anyhow::{Context, Result, anyhow};
#[cfg(test)]
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;
use skippy_inference_api::chat::identifiers::ChatStreamNormalizationState;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
#[path = "normalized_chat_stream.rs"]
mod normalized_chat_stream;
#[path = "normalized_stream_completion.rs"]
mod normalized_stream_completion;
#[cfg(test)]
#[path = "responses_function_calls.rs"]
mod responses_function_calls;

/// One tool call's deltas folded across `chat.completion.chunk` frames,
/// keyed by the `index` OpenAI streaming clients use to tell concurrent tool
/// calls apart.
#[derive(Debug, Default)]
struct AssembledStreamToolCall {
    id: Option<String>,
    kind: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Folds a raw chat-completions SSE stream's `choices[0].delta` chunks into
/// the single served message a non-streamed response would have carried, so
/// [`ExchangeOutputDigests`] can be computed over it with the SAME
/// construction used for a buffered (non-streamed) response — this is what
/// "the digest covers the assembled response the host actually returned"
/// means for a streamed delivery: not a per-chunk partial, but the full
/// content/tool_calls/reasoning folded across every chunk actually sent to
/// the client.
#[derive(Debug, Default)]
pub(super) struct StreamedChatAssembly {
    content: String,
    saw_content: bool,
    reasoning_content: String,
    saw_reasoning: bool,
    tool_calls: std::collections::BTreeMap<u64, AssembledStreamToolCall>,
    /// Set when a chunk carried more than one choice, or a choice whose
    /// `index` is not 0 — a stream that asked for `n > 1` choices. The
    /// assembly folds `choices[0]` only (there is one served message to fold
    /// into), so a multi-choice stream would digest choice 0 and silently
    /// omit every later choice: a *wrong* digest, which is worse than an
    /// absent one. No digest is published for such a stream at all.
    multi_choice: bool,
}

impl StreamedChatAssembly {
    /// Fold one (already client-facing, i.e. post-normalization) SSE data
    /// frame's `choices[0].delta` into the running assembly. A frame that
    /// isn't a recognizable chat-completion chunk, or carries no delta (an
    /// error frame, a bare `[DONE]`), is silently skipped — best-effort over
    /// whatever the client actually saw, never a guess.
    pub(super) fn ingest_chunk(&mut self, data: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            return;
        };
        let Some(choices) = value.get("choices").and_then(|choices| choices.as_array()) else {
            return;
        };
        if choices.len() > 1
            || choices.first().is_some_and(|choice| {
                choice
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|index| index != 0)
            })
        {
            self.multi_choice = true;
        }
        let Some(delta) = choices.first().and_then(|choice| choice.get("delta")) else {
            return;
        };
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            self.content.push_str(content);
            self.saw_content = true;
        }
        if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
            self.reasoning_content.push_str(reasoning);
            self.saw_reasoning = true;
        }
        let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) else {
            return;
        };
        for (position, tool_call) in tool_calls.iter().enumerate() {
            let index = tool_call
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(position as u64);
            let entry = self.tool_calls.entry(index).or_default();
            if let Some(id) = tool_call.get("id").and_then(|v| v.as_str()) {
                entry.id = Some(id.to_string());
            }
            if let Some(kind) = tool_call.get("type").and_then(|v| v.as_str()) {
                entry.kind = Some(kind.to_string());
            }
            let Some(function) = tool_call.get("function") else {
                continue;
            };
            if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                entry.name = Some(name.to_string());
            }
            if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str()) {
                entry.arguments.push_str(arguments);
            }
        }
    }

    /// Render the accumulated deltas as the single served message a
    /// non-streamed response would have carried, in the same
    /// `choices[].message.*` shape [`ExchangeOutputDigests::from_response_value`]
    /// expects. `None` when the stream carried no content, tool_calls, or
    /// reasoning to assemble at all — never a fabricated empty message.
    fn assembled_response(&self) -> Option<serde_json::Value> {
        if !self.saw_content && !self.saw_reasoning && self.tool_calls.is_empty() {
            return None;
        }
        let mut message = serde_json::Map::new();
        message.insert("role".into(), serde_json::json!("assistant"));
        message.insert("content".into(), serde_json::json!(self.content));
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<serde_json::Value> = self
                .tool_calls
                .values()
                .map(|call| {
                    serde_json::json!({
                        "id": call.id,
                        "type": call.kind,
                        "function": {"name": call.name, "arguments": call.arguments},
                    })
                })
                .collect();
            message.insert("tool_calls".into(), serde_json::Value::Array(tool_calls));
        }
        if self.saw_reasoning {
            message.insert(
                "reasoning_content".into(),
                serde_json::json!(self.reasoning_content),
            );
        }
        Some(serde_json::json!({
            "choices": [{"index": 0, "message": serde_json::Value::Object(message)}]
        }))
    }

    /// Compute the response/tool_calls/reasoning digests over the assembled
    /// result, or an all-`None` bundle when nothing was assembled — or when
    /// the stream carried more than one choice, which this fold cannot
    /// honestly cover (see [`Self::multi_choice`]).
    pub(super) fn output_digests(&self) -> ExchangeOutputDigests {
        if self.multi_choice {
            return ExchangeOutputDigests::default();
        }
        self.assembled_response()
            .map(|value| ExchangeOutputDigests::from_response_value(&value))
            .unwrap_or_default()
    }
}

pub(super) async fn write_captured_sse_event(
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

pub(super) use normalized_chat_stream::relay_normalized_chat_completion_stream;

fn normalized_stream_is_truncated(done_seen: bool, upstream_error_seen: bool) -> bool {
    !done_seen && !upstream_error_seen
}

/// Relay a streaming chat-completions upstream response translated into Responses-API SSE.
// Keep this wrapper's transport arguments aligned with the sibling protocol
// relays; the final argument is the ingress-owned Responses state.
#[allow(clippy::too_many_arguments)]
pub(in crate::network::openai::response) async fn relay_translated_responses_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    request_id: RequestId,
    responses_turn: Option<std::sync::Arc<skippy_inference_api::responses::ResponsesTurn>>,
) -> Result<RouteAttemptResult> {
    let turn = responses_turn.context("Responses stream requires a prepared request")?;
    Box::pin(super::shared_responses_stream::relay(
        tcp_stream,
        reader,
        probe,
        retry_policy,
        served_by,
        route_observer,
        request_id,
        turn,
    ))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{ArtifactUnavailableReason, OpenAiArtifactCapture};
    use crate::network::openai::response::common::sse_data_frame_is_openai_error;
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

    fn test_turn() -> std::sync::Arc<skippy_inference_api::responses::ResponsesTurn> {
        test_turn_with(skippy_inference_api::responses::ResponseContinuation::disabled())
    }

    fn test_turn_with(
        continuation: skippy_inference_api::responses::ResponseContinuation,
    ) -> std::sync::Arc<skippy_inference_api::responses::ResponsesTurn> {
        use skippy_inference_api::responses::ResponsesTurn;
        let mut request = serde_json::json!({
            "model":"qwen", "input":"test", "stream_options":{"include_obfuscation":false},
            "tools":[{"type":"function","name":"read"},{"type":"function","name":"write"},
                     {"type":"function","name":"list"},{"type":"function","name":"now"}]
        });
        std::sync::Arc::new(ResponsesTurn::prepare(&mut request, continuation).unwrap())
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
                buffered: b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_vec(),
                header_end: b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".len(),
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_translated_responses_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
                Some(test_turn()),
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
                cache_cost: None,
                // The completed stream digests the message assembled from
                // the three deltas, as a non-streamed body would be digested.
                output_digests: ExchangeOutputDigests::from_response_value(&serde_json::json!({
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "Hello world!"
                    }}]
                })),
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
        let header =
            b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
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
                None,
                OpenAiRouteObserver::capture_test_observer(RequestId::new(), &observer_capture),
                RequestId::new(),
            )
            .await
            .expect("relay")
        });

        let usage_chunk = r#"{"id":"chatcmpl-y","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":7,"total_tokens":9},"timings":{"prompt_ms":514.0,"queue_wait_ms":1.0,"cache_restore_ms":2.0,"suffix_prefill_n":256}}"#;
        upstream_writer
            .write_all(format!("data: {usage_chunk}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_ms\":99.0,\"queue_wait_ms\":99.0,\"cache_restore_ms\":99.0,\"suffix_prefill_n\":1}}\n\ndata: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");

        assert!(output.starts_with(b"HTTP/1.1 200 OK\r\n"));

        // Destructure rather than compare the whole `Delivered` literal: this
        // relay now also assembles real `output_digests` across the streamed
        // chunks, so assert status/usage/cache_cost as before AND that the
        // assembled response digest is present (the stream carried real
        // content: "hello") while tool_calls/reasoning stay absent (the
        // stream carried neither).
        let RouteAttemptResult::Delivered {
            status_code,
            usage,
            cache_cost,
            output_digests,
        } = route_result
        else {
            panic!("expected Delivered, got {route_result:?}");
        };
        assert_eq!(status_code, 200);
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: Some(2),
                cached_prompt_tokens: None,
                completion_tokens: Some(7),
                total_tokens: Some(9),
            })
        );
        assert_eq!(
            cache_cost,
            Some(CacheCostObservation {
                queue_delay_micros: 1_000,
                restore_micros: 2_000,
                prefill_micros_per_token: Some(2_000),
            })
        );
        assert_eq!(
            output_digests.response.map(hex::encode),
            crate::plugin::openai_exchange::request_body_digest(
                &serde_json::json!({"choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "hello"
                }}]}),
                None
            ),
            "the assembled response over the streamed \"hello\" delta must digest the same as the equivalent non-streamed body"
        );
        assert!(output_digests.tool_calls.is_none());
        assert!(output_digests.reasoning.is_none());
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
    async fn relay_normalized_chat_completion_stream_echoes_capsule_nonce_headers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nx-capsule-client-nonce: nonce-under-test\r\nx-capsule-nonce-origin: frontend\r\n\r\n";
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
            .expect("relay")
        });

        upstream_writer
            .write_all(b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server_task.await.expect("server task");

        let output_text = String::from_utf8_lossy(&output);
        assert!(
            output_text.contains("x-capsule-client-nonce: nonce-under-test\r\n"),
            "public-proxy SSE response must echo the client nonce header: {output_text}"
        );
        assert!(
            output_text.contains("x-capsule-nonce-origin: frontend\r\n"),
            "public-proxy SSE response must echo the nonce origin marker: {output_text}"
        );
    }

    #[tokio::test]
    async fn incomplete_normalized_stream_is_an_error_without_synthesized_done() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
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
                    served_by: None,
                    route_observer: OpenAiRouteObserver::default(),
                    responses_turn: None,
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
            RouteAttemptResult::CommittedStreamFailure {
                status_code: 200,
                cause: CommittedStreamFailureCause::Truncated,
            }
        );
        assert!(!body.contains("data: [DONE]"));
    }

    /// Drive the Responses-API translated stream over raw upstream SSE bytes.
    pub(super) async fn relay_translated_over(
        upstream: &'static [u8],
    ) -> (Result<RouteAttemptResult>, String) {
        relay_translated_with_header(
            upstream,
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        )
        .await
    }

    async fn relay_translated_with_header(
        upstream: &[u8],
        header: &'static [u8],
    ) -> (Result<RouteAttemptResult>, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        use skippy_inference_api::responses::{CallerScope, ResponseContinuation, ResponseStore};
        let store = std::sync::Arc::new(ResponseStore::default());
        let turn = test_turn_with(ResponseContinuation::new(
            Some(store.clone()),
            Some(CallerScope::LocalHost),
        ));
        let response_id = turn.id().to_string();
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
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
                Some(turn),
            )
            .await
        });
        upstream_writer.write_all(upstream).await.unwrap();
        upstream_writer.shutdown().await.unwrap();
        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let result = server_task.await.expect("server task");
        let wire = String::from_utf8_lossy(&output).to_string();
        if wire.contains("event: response.failed") || wire.contains("event: response.incomplete") {
            assert!(
                store.get(&CallerScope::LocalHost, &response_id).is_err(),
                "failed and incomplete turns must not be retained"
            );
        }
        (result, wire)
    }

    #[tokio::test]
    async fn damaged_http_framing_cannot_complete_a_translated_response() {
        let payload =
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n";
        let truncated_chunk = format!("{:x}\r\n{payload}\r\n", payload.len());
        for (body, header) in [
            (
                payload.as_bytes(),
                &b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"[..],
            ),
            (
                truncated_chunk.as_bytes(),
                &b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"[..],
            ),
            (
                payload.as_bytes(),
                &b"HTTP/1.1 200 OK\r\nContent-Length: 10000\r\n\r\n"[..],
            ),
        ] {
            let (result, wire) = relay_translated_with_header(body, header).await;
            assert!(
                matches!(
                    result,
                    Ok(RouteAttemptResult::CommittedStreamFailure {
                        cause: CommittedStreamFailureCause::Truncated,
                        ..
                    })
                ),
                "{result:?}"
            );
            assert!(wire.contains("response.failed"));
            assert!(!wire.contains("response.completed"));
        }
    }

    /// Streamed /v1/responses requests use this translated path. A completed
    /// stream must carry the response digest, over the same assembled message
    /// the serving node digests, or the requester's terminal event carries none.
    #[tokio::test]
    async fn completed_translated_stream_carries_the_assembled_response_digest() {
        let (result, body) = relay_translated_over(
            b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n",
        )
        .await;
        let Ok(RouteAttemptResult::Delivered { output_digests, .. }) = result else {
            panic!("expected Delivered, got {result:?}");
        };
        assert_eq!(
            output_digests.response.map(hex::encode),
            crate::plugin::openai_exchange::request_body_digest(
                &serde_json::json!({"choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "hello"
                }}]}),
                None
            ),
            "the translated stream digests the same served message as the chat-completions stream"
        );
        assert!(output_digests.tool_calls.is_none());
        assert!(output_digests.reasoning.is_none());
        assert!(body.contains("response.completed"));
    }
    /// A stream that ends in an error frame is a failure, and carries no
    /// digest of the partial it relayed.
    #[tokio::test]
    async fn translated_stream_ending_in_an_error_frame_carries_no_digest() {
        let (result, _) = relay_translated_over(
            b"data: {\"id\":\"chatcmpl-y\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n\
data: {\"error\":{\"message\":\"deadline\",\"type\":\"server_error\",\"param\":null,\"code\":\"timeout\"}}\n\n\
data: [DONE]\n\n",
        )
        .await;
        // A failed stream is reported as a committed failure, which carries
        // no digest at all.
        assert!(
            matches!(
                result,
                Ok(RouteAttemptResult::CommittedStreamFailure {
                    status_code: 200,
                    cause: CommittedStreamFailureCause::BackendError,
                })
            ),
            "expected a committed backend failure, got {result:?}"
        );
    }

    #[tokio::test]
    async fn incomplete_translated_stream_is_an_error_without_completed_tail() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
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
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
                Some(test_turn()),
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

        assert!(matches!(
            route_result,
            Ok(RouteAttemptResult::CommittedStreamFailure {
                cause: CommittedStreamFailureCause::Truncated,
                ..
            })
        ));
        assert!(body.contains("response.failed"));
        assert!(!body.contains("response.completed"));
    }

    #[tokio::test]
    async fn upstream_error_frame_is_relayed_but_not_recorded_as_completed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_vec(),
                header_end: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
            .expect("relay")
        });

        let error_frame = r#"{"error":{"message":"cache runtime operation req-1 exceeded its deadline","type":"server_error","param":null,"code":"timeout"}}"#;
        upstream_writer
            .write_all(format!("data: {error_frame}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");
        let body = String::from_utf8_lossy(&output);

        // The error frame is relayed to the client untouched...
        assert!(body.contains("exceeded its deadline"));
        assert!(body.contains("data: [DONE]"));

        // ...but the attempt is a failed inference, not a success.
        assert_eq!(
            route_result,
            RouteAttemptResult::CommittedStreamFailure {
                status_code: 200,
                cause: CommittedStreamFailureCause::BackendError,
            }
        );
    }

    #[tokio::test]
    async fn upstream_error_frame_without_done_is_terminal_not_incomplete() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_vec(),
                header_end: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
            .expect("relay")
        });

        let error_frame = r#"{"error":{"message":"cache runtime operation req-1 exceeded its deadline","type":"server_error","param":null,"code":"timeout"}}"#;
        upstream_writer
            .write_all(format!("data: {error_frame}\n\n").as_bytes())
            .await
            .unwrap();
        // The upstream dies without ever sending [DONE].
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");
        let body = String::from_utf8_lossy(&output);

        // The error frame still reaches the client...
        assert!(body.contains("exceeded its deadline"));
        // ...and the embedded error is terminal: a backend failure, not
        // misreported as an incomplete-stream truncation.
        assert_eq!(
            route_result,
            RouteAttemptResult::CommittedStreamFailure {
                status_code: 200,
                cause: CommittedStreamFailureCause::BackendError,
            }
        );
    }

    #[tokio::test]
    async fn translated_upstream_error_frame_without_done_is_terminal_not_incomplete() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
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
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
                Some(test_turn()),
            )
            .await
            .expect("relay")
        });

        let error_frame = r#"{"error":{"message":"cache runtime operation req-2 exceeded its deadline","type":"server_error","param":null,"code":"timeout"}}"#;
        upstream_writer
            .write_all(format!("data: {error_frame}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        let route_result = server_task.await.expect("server task");
        let body = String::from_utf8_lossy(&output);

        assert!(body.contains("exceeded its deadline"));
        assert_eq!(
            route_result,
            RouteAttemptResult::CommittedStreamFailure {
                status_code: 200,
                cause: CommittedStreamFailureCause::BackendError,
            }
        );
    }

    /// A streamed tool call's `function.arguments` arrives fragmented across
    /// many `delta.tool_calls` chunks (the normal OpenAI streaming shape for
    /// a long argument string). The assembled digest must fold every
    /// fragment, not just the most recent one -- mutating `ingest_chunk` to
    /// overwrite rather than append must turn this test red.
    #[test]
    fn assembled_tool_call_arguments_fold_every_streamed_fragment() {
        let mut assembly = StreamedChatAssembly::default();
        assembly.ingest_chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}"#,
        );
        assembly.ingest_chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"berlin\"}"}}]}}]}"#,
        );

        let assembled = assembly
            .assembled_response()
            .expect("tool call deltas were ingested");
        let arguments =
            assembled["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .expect("assembled arguments string");
        assert_eq!(
            arguments, r#"{"city":"berlin"}"#,
            "arguments must be the fold of every fragment, not just the last chunk"
        );

        // The digest over the folded result must differ from a digest of the
        // last fragment alone -- pins the fold, not just the presence of a digest.
        let digests = assembly.output_digests();
        let last_fragment_only = ExchangeOutputDigests::from_response_value(&serde_json::json!({
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "\"berlin\"}"}}]
            }}]
        }));
        assert_ne!(digests.tool_calls, last_fragment_only.tool_calls);
    }

    /// A stream the client asked for with `n > 1` delivers several choices per
    /// chunk. The fold covers `choices[0]`, so digesting it would publish a
    /// response digest that silently omits every later choice — wrong, not
    /// merely incomplete. Such a stream publishes no digest at all, and the
    /// single-choice stream the host normally serves still does.
    #[test]
    fn multi_choice_streams_publish_no_digest() {
        let single = r#"{"choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        let mut assembly = StreamedChatAssembly::default();
        assembly.ingest_chunk(single);
        assert!(
            assembly.output_digests().response.is_some(),
            "the normal single-choice stream still digests"
        );

        // Two choices in one chunk.
        let mut assembly = StreamedChatAssembly::default();
        assembly.ingest_chunk(single);
        assembly.ingest_chunk(
            r#"{"choices":[{"index":0,"delta":{"content":" a"}},{"index":1,"delta":{"content":" b"}}]}"#,
        );
        assert_eq!(
            assembly.output_digests(),
            ExchangeOutputDigests::default(),
            "a chunk carrying two choices must not yield a digest over choice 0 alone"
        );

        // One choice per chunk, but the chunk is choice 1 — the fold covers
        // choice 0, so this is still a stream it cannot honestly digest.
        let mut assembly = StreamedChatAssembly::default();
        assembly.ingest_chunk(r#"{"choices":[{"index":1,"delta":{"content":"only choice 1"}}]}"#);
        assert_eq!(
            assembly.output_digests(),
            ExchangeOutputDigests::default(),
            "a non-zero choice index means the fold is not looking at the whole response"
        );
    }

    #[test]
    fn openai_error_frames_are_detected_across_shapes() {
        assert!(sse_data_frame_is_openai_error(
            r#"{"error":{"message":"boom","type":"server_error","code":"timeout"}}"#
        ));
        assert!(sse_data_frame_is_openai_error(
            r#"{"error":"plain string"}"#
        ));
        assert!(!sse_data_frame_is_openai_error(
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#
        ));
        assert!(!sse_data_frame_is_openai_error("[DONE]"));
        assert!(!sse_data_frame_is_openai_error("not json"));
    }

    /// Drive the plain chat-completions stream over the same raw upstream SSE.
    async fn relay_chat_stream_over(upstream: &'static [u8]) -> Result<RouteAttemptResult> {
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
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
        });
        upstream_writer.write_all(upstream).await.unwrap();
        upstream_writer.shutdown().await.unwrap();
        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server_task.await.expect("server task")
    }

    /// Streamed, the /v1/responses path and the
    /// /v1/chat/completions path carry the SAME response digest for the same
    /// served answer, so neither reads as a disagreement with the serving
    /// node's event.
    #[tokio::test]
    async fn a_streamed_responses_request_and_a_streamed_chat_request_carry_one_digest() {
        const UPSTREAM: &[u8] = b"data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Bl\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ue\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        let Ok(RouteAttemptResult::Delivered {
            output_digests: chat,
            ..
        }) = relay_chat_stream_over(UPSTREAM).await
        else {
            panic!("chat stream not delivered");
        };
        let (result, _) = relay_translated_over(UPSTREAM).await;
        let Ok(RouteAttemptResult::Delivered {
            output_digests: responses,
            ..
        }) = result
        else {
            panic!("responses stream not delivered");
        };
        assert!(chat.response.is_some());
        assert_eq!(responses.response, chat.response);
    }

    // issue #2352, papercut 4: the normalized chat-completions SSE head must
    // echo the canonical request id.
    #[tokio::test]
    async fn relay_normalized_chat_completion_stream_echoes_the_canonical_request_id() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let request_id = RequestId::new();
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                request_id,
            )
            .await
            .expect("relay")
        });

        let chunk = r#"{"id":"chatcmpl-rid","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        upstream_writer
            .write_all(format!("data: {chunk}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server_task.await.expect("server task");

        let head = String::from_utf8_lossy(&output);
        assert!(
            head.contains(&format!("x-request-id: {}", request_id.as_uuid())),
            "the normalized chat SSE head must echo the canonical request id: {head}"
        );
    }

    // issue #2352, papercut 4: the Responses-API SSE head must echo the
    // canonical request id.
    #[tokio::test]
    async fn relay_translated_responses_stream_echoes_the_canonical_request_id() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let request_id = RequestId::new();
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header =
            b"HTTP/1.1 201 Created\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_translated_responses_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                request_id,
                Some(test_turn()),
            )
            .await
            .expect("relay")
        });

        let finish = r#"{"id":"chatcmpl-rid","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        upstream_writer
            .write_all(format!("data: {finish}\n\n").as_bytes())
            .await
            .unwrap();
        upstream_writer
            .write_all(b"data: [DONE]\n\n")
            .await
            .unwrap();
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server_task.await.expect("server task");

        let head = String::from_utf8_lossy(&output);
        assert!(
            head.contains(&format!("x-request-id: {}", request_id.as_uuid())),
            "the Responses-API SSE head must echo the canonical request id: {head}"
        );
    }
}
