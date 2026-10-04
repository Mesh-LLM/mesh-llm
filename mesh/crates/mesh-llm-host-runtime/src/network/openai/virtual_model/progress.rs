//! Progress drip for streaming virtual models.
//!
//! The host reaches a virtual model through a single request/response RPC: the
//! plugin's answer is buffered and adapted afterwards. A streaming caller
//! would therefore receive *nothing* — not even response headers — until the
//! turn completes, which for a mixture-of-agents committee is seconds of an
//! open socket and a stalled spinner. When the route declares progress lines
//! ([`VirtualModelManifest::progress_lines`]) the host commits the response
//! head up front and drips those lines into the caller's reasoning channel
//! while the turn runs; the buffered body then follows as a continuation of
//! the same stream.
//!
//! Trade-offs, inherited from the in-process MoA gateway this restores:
//!
//! * HTTP headers must precede the body, so the plugin's own headers (the
//!   result-derived observability ones, for example `x-moa-*`) only reach the
//!   caller on the non-streaming path.
//! * A plugin-side rejection of a *streaming* request is delivered in-band —
//!   HTTP 200 followed by an error event and the stream sentinel — because the
//!   head is committed before the plugin turn starts.
//! * A progress write that fails means the caller is gone; the turn is dropped
//!   rather than awaited.

use super::stream_adapters;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::plugin::VirtualModelRoute;
use mesh_llm_plugin::VirtualModelResponse;
use serde_json::{Value, json};
use std::future::Future;
use std::time::Duration;

/// Time between progress events while the turn is still running. One second
/// feels alive without flooding the wire.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(1000);

/// How the buffered body continues a stream whose head the progress phase
/// already committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProgressContinuation {
    /// `created_at` the early `response.created` event carried; the body
    /// reuses it so one response keeps one timestamp.
    pub(super) created_at: i64,
    /// Next `sequence_number` to use — strictly greater than the last one the
    /// progress phase emitted.
    pub(super) next_sequence_number: i32,
}

/// Drive a virtual-model turn to completion, committing a progress head first
/// when the route asks for one.
pub(super) async fn drive<F>(plan: Option<ProgressPlan>, stream: ClientStream, work: F) -> Driven
where
    F: Future<Output = anyhow::Result<VirtualModelResponse>>,
{
    match plan {
        Some(plan) => plan.run(stream, work).await,
        None => match work.await {
            Ok(response) => Driven::Buffered { stream, response },
            Err(error) => Driven::WorkFailed { stream, error },
        },
    }
}

/// Result of driving a virtual-model turn to completion.
pub(super) enum Driven {
    /// Nothing was written: the caller writes the head and the body itself.
    Buffered {
        stream: ClientStream,
        response: VirtualModelResponse,
    },
    /// The head and the progress events are on the wire; only a body may
    /// follow.
    Committed {
        stream: ClientStream,
        response: VirtualModelResponse,
        continuation: ProgressContinuation,
    },
    /// The turn failed before anything was written, so the caller can still
    /// choose the HTTP status.
    WorkFailed {
        stream: ClientStream,
        error: anyhow::Error,
    },
    /// The turn failed after the head was committed; the error was delivered
    /// in-band and only the outcome is left to record.
    FailedAfterCommit { reason: &'static str },
    /// A write failed; the caller is gone and there is nothing left to write.
    ClientGone,
}

/// Everything the progress phase needs for one request.
pub(super) struct ProgressPlan {
    adapter: proxy::ResponseAdapter,
    model_id: String,
    completion_id: String,
    created_at: i64,
    lines: Vec<String>,
}

impl ProgressPlan {
    /// `None` when the route declares no progress lines, the caller is not
    /// streaming, or the adapter has no progress channel.
    pub(super) fn for_route(
        route: &VirtualModelRoute,
        adapter: proxy::ResponseAdapter,
        requests_stream: bool,
    ) -> Option<Self> {
        if !requests_stream || !route.supports_streaming || route.progress_lines.is_empty() {
            return None;
        }
        if !matches!(
            adapter,
            proxy::ResponseAdapter::OpenAiChatCompletionsStream
                | proxy::ResponseAdapter::OpenAiResponsesStream
                | proxy::ResponseAdapter::AnthropicMessagesStream
                | proxy::ResponseAdapter::None
        ) {
            return None;
        }
        Some(Self {
            adapter,
            model_id: route.model_id.clone(),
            completion_id: format!("chatcmpl-moa-{}", short_hex_nanos()),
            created_at: unix_secs(),
            lines: route.progress_lines.clone(),
        })
    }

    pub(super) async fn run<F>(self, stream: ClientStream, work: F) -> Driven
    where
        F: Future<Output = anyhow::Result<VirtualModelResponse>>,
    {
        let mut stream = stream;
        if stream_adapters::write_sse_headers(&mut stream, &[])
            .await
            .is_err()
        {
            return Driven::ClientGone;
        }
        // Responses-API clients reject a stream that opens with a delta, so
        // the envelope goes out before the first drip.
        if self.adapter == proxy::ResponseAdapter::OpenAiResponsesStream
            && self.write_created(&mut stream).await.is_err()
        {
            return Driven::ClientGone;
        }

        tokio::pin!(work);
        let mut ticker = tokio::time::interval(PROGRESS_INTERVAL);
        // Skip stacked ticks: a stalled write must not dump a burst of lines.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // We want the first line one interval in, not at t=0.
        ticker.tick().await;

        let mut step = 0usize;
        let mut sequence_number = 1i32;
        loop {
            tokio::select! {
                biased;
                result = &mut work => {
                    let continuation = ProgressContinuation {
                        created_at: self.created_at,
                        next_sequence_number: sequence_number,
                    };
                    return match result {
                        Ok(mut response) if (200..300).contains(&response.status_code) => {
                            align_committed_body(&mut response.body, &self.completion_id, self.created_at);
                            Driven::Committed {
                                stream,
                                response,
                                continuation,
                            }
                        }
                        Ok(response) => {
                            let message = failure_message(&response);
                            self.write_failure_tail(&mut stream, &message).await;
                            Driven::FailedAfterCommit {
                                reason: "virtual_model_failed_after_commit",
                            }
                        }
                        Err(error) => {
                            let message =
                                format!("virtual model '{}' failed: {error}", self.model_id);
                            self.write_failure_tail(&mut stream, &message).await;
                            Driven::FailedAfterCommit {
                                reason: "virtual_model_failed_after_commit",
                            }
                        }
                    };
                }
                _ = ticker.tick() => {}
            }
            let line = self.progress_line(step);
            step += 1;
            if self
                .write_progress(&mut stream, &line, &mut sequence_number)
                .await
                .is_err()
            {
                // Dropping the pinned turn here cancels the host's wait rather
                // than burning a plugin turn on a dead request.
                return Driven::ClientGone;
            }
        }
    }

    /// The line for `step`: the declared lines play once in order, then the
    /// last one repeats for as long as the turn runs.
    fn progress_line(&self, step: usize) -> String {
        let index = step.min(self.lines.len().saturating_sub(1));
        format!("{}\n", self.lines[index])
    }

    fn item_id(&self) -> String {
        progress_item_id(self.created_at)
    }

    async fn write_created(&self, stream: &mut ClientStream) -> std::io::Result<()> {
        let mut created =
            skippy_inference_api::responses_stream_created_event(&self.model_id, self.created_at);
        if let Some(response) = created.get_mut("response").and_then(Value::as_object_mut) {
            response.insert("id".into(), Value::String(self.completion_id.clone()));
        }
        stream_adapters::write_sse_data(stream, &created.to_string()).await
    }

    async fn write_progress(
        &self,
        stream: &mut ClientStream,
        line: &str,
        sequence_number: &mut i32,
    ) -> std::io::Result<()> {
        match progress_frame(
            self.adapter,
            line,
            &self.model_id,
            &self.completion_id,
            &self.item_id(),
            *sequence_number,
        ) {
            ProgressFrame::Chat(chunk) => {
                stream_adapters::write_sse_data(stream, &chunk.to_string()).await
            }
            ProgressFrame::Named(name, event) => {
                if self.adapter == proxy::ResponseAdapter::OpenAiResponsesStream {
                    *sequence_number = sequence_number.saturating_add(1);
                }
                stream_adapters::write_named_sse_data(stream, name, &event.to_string()).await
            }
        }
    }

    /// We already committed 200 OK, so a failed turn is reported in-band: the
    /// stream ends with an error event and its sentinel instead of truncating.
    async fn write_failure_tail(&self, stream: &mut ClientStream, message: &str) {
        let result = match self.adapter {
            proxy::ResponseAdapter::AnthropicMessagesStream => {
                self.write_anthropic_failure_tail(stream, message).await
            }
            proxy::ResponseAdapter::OpenAiResponsesStream => {
                let failed = json!({
                    "type": "response.failed",
                    "response": {
                        "id": self.completion_id,
                        "error": {"message": message},
                    },
                });
                let sent = stream_adapters::write_named_sse_data(
                    stream,
                    "response.failed",
                    &failed.to_string(),
                )
                .await;
                match sent {
                    Ok(()) => write_done_and_close(stream).await,
                    Err(error) => Err(error),
                }
            }
            _ => {
                let chunk = json!({
                    "id": self.completion_id,
                    "object": "chat.completion.chunk",
                    "model": self.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": format!("[error: {message}]")},
                        "finish_reason": "error",
                    }],
                });
                let sent = stream_adapters::write_sse_data(stream, &chunk.to_string()).await;
                match sent {
                    Ok(()) => write_done_and_close(stream).await,
                    Err(error) => Err(error),
                }
            }
        };
        if let Err(error) = result {
            tracing::warn!("virtual-model progress: failure tail write failed: {error}");
        }
    }

    async fn write_anthropic_failure_tail(
        &self,
        stream: &mut ClientStream,
        message: &str,
    ) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        let event = skippy_inference_api::anthropic::translate_stream_error_body(
            &json!({"error": {"message": message}}),
        );
        let data = serde_json::to_string(&event).map_err(std::io::Error::other)?;
        crate::network::openai::response_adapter::write_chunked_sse_event(
            stream,
            Some("error"),
            &data,
        )
        .await?;
        stream.write_all(b"0\r\n\r\n").await?;
        stream.shutdown().await
    }
}

/// One progress event, already shaped for the caller's protocol.
#[derive(Debug, PartialEq)]
enum ProgressFrame {
    /// A named `event:` frame (Responses deltas, Anthropic pings).
    Named(&'static str, Value),
    /// A chat-completions chunk carrying the line as `reasoning_content`.
    Chat(Value),
}

fn progress_frame(
    adapter: proxy::ResponseAdapter,
    line: &str,
    model_id: &str,
    completion_id: &str,
    item_id: &str,
    sequence_number: i32,
) -> ProgressFrame {
    if adapter == proxy::ResponseAdapter::AnthropicMessagesStream {
        // Anthropic has no reasoning channel on the wire; a ping keeps the
        // connection visible without polluting the answer.
        return ProgressFrame::Named("ping", json!({"type": "ping"}));
    }
    if adapter == proxy::ResponseAdapter::OpenAiResponsesStream {
        return ProgressFrame::Named(
            "response.reasoning_text.delta",
            json!({
                "type": "response.reasoning_text.delta",
                "sequence_number": sequence_number,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": line,
            }),
        );
    }
    ProgressFrame::Chat(json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {"reasoning_content": line},
            "finish_reason": null,
        }],
    }))
}

/// The output-item id a committed turn's progress deltas and content share.
///
/// The `/v1/responses` replay derives its item id from the body's `created`
/// timestamp, so the progress phase — which runs before the body exists —
/// derives the same id from the timestamp it baked into `response.created`;
/// [`align_committed_body`] then pins the body to that timestamp.
pub(super) fn progress_item_id(created_at: i64) -> String {
    format!("msg_{created_at}")
}

/// Pin a committed body to the identity the progress phase already published,
/// so a chunk-aggregating client sees one completion and one item rather than
/// progress and content belonging to different answers.
pub(super) fn align_committed_body(body: &mut Value, completion_id: &str, created_at: i64) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.insert("id".into(), Value::String(completion_id.to_string()));
    object.insert("created".into(), json!(created_at));
}

async fn write_done_and_close(stream: &mut ClientStream) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    stream.write_all(b"data: [DONE]\n\n0\r\n\r\n").await?;
    stream.shutdown().await
}

fn failure_message(response: &VirtualModelResponse) -> String {
    response
        .body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("virtual model returned HTTP {}", response.status_code))
}

fn unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

fn short_hex_nanos() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::VirtualModelRoute;

    fn route(progress_lines: Vec<String>) -> VirtualModelRoute {
        VirtualModelRoute {
            plugin_name: "mesh-moa".into(),
            model_id: "mesh".into(),
            handler: "chat".into(),
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            supports_tools: true,
            supports_streaming: true,
            requires_candidates: true,
            progress_lines,
        }
    }

    fn plan(lines: Vec<&str>, adapter: proxy::ResponseAdapter) -> ProgressPlan {
        ProgressPlan::for_route(
            &route(lines.into_iter().map(str::to_string).collect()),
            adapter,
            true,
        )
        .expect("a streaming route with progress lines plans a drip")
    }

    #[test]
    fn a_route_without_progress_lines_does_not_drip() {
        assert!(
            ProgressPlan::for_route(
                &route(Vec::new()),
                proxy::ResponseAdapter::OpenAiChatCompletionsStream,
                true
            )
            .is_none()
        );
        assert!(
            ProgressPlan::for_route(
                &route(vec!["line".into()]),
                proxy::ResponseAdapter::OpenAiResponsesJson,
                true
            )
            .is_none(),
            "a non-streaming adapter has no progress channel"
        );
        assert!(
            ProgressPlan::for_route(
                &route(vec!["line".into()]),
                proxy::ResponseAdapter::OpenAiChatCompletionsStream,
                false
            )
            .is_none(),
            "a non-streaming request is buffered"
        );
    }

    #[test]
    fn declared_lines_play_once_then_the_last_repeats() {
        let plan = plan(
            vec!["one", "two", "three"],
            proxy::ResponseAdapter::OpenAiChatCompletionsStream,
        );

        let lines = (0..6)
            .map(|step| plan.progress_line(step))
            .collect::<Vec<_>>();

        assert_eq!(
            lines,
            vec!["one\n", "two\n", "three\n", "three\n", "three\n", "three\n"]
        );
    }

    #[test]
    fn chat_progress_chunk_shares_the_completion_id_and_uses_the_reasoning_channel() {
        let frame = progress_frame(
            proxy::ResponseAdapter::OpenAiChatCompletionsStream,
            "Routing through mesh…\n",
            "mesh",
            "chatcmpl-moa-abc",
            "msg_7",
            1,
        );

        let ProgressFrame::Chat(chunk) = frame else {
            panic!("chat-completions drips a chat chunk");
        };
        assert_eq!(chunk["id"], "chatcmpl-moa-abc");
        assert_eq!(chunk["model"], "mesh");
        assert_eq!(
            chunk["choices"][0]["delta"]["reasoning_content"],
            "Routing through mesh…\n"
        );
        assert!(chunk["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn responses_progress_uses_the_reasoning_delta_channel_and_its_sequence() {
        let frame = progress_frame(
            proxy::ResponseAdapter::OpenAiResponsesStream,
            "Comparing responses…\n",
            "mesh",
            "chatcmpl-moa-abc",
            "msg_7",
            3,
        );

        let ProgressFrame::Named(name, event) = frame else {
            panic!("responses drips a named event");
        };
        assert_eq!(name, "response.reasoning_text.delta");
        assert_eq!(event["type"], "response.reasoning_text.delta");
        assert_eq!(event["sequence_number"], 3);
        assert_eq!(event["item_id"], "msg_7");
        assert_eq!(event["delta"], "Comparing responses…\n");
    }

    #[test]
    fn anthropic_progress_is_a_ping_and_carries_no_answer_text() {
        let frame = progress_frame(
            proxy::ResponseAdapter::AnthropicMessagesStream,
            "Routing through mesh…\n",
            "mesh",
            "chatcmpl-moa-abc",
            "msg_7",
            1,
        );

        assert_eq!(frame, ProgressFrame::Named("ping", json!({"type": "ping"})));
    }

    #[test]
    fn committed_body_keeps_the_progress_identity() {
        let mut body = json!({
            "id": "chatcmpl-plugin",
            "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
        });

        align_committed_body(&mut body, "chatcmpl-moa-abc", 42);

        assert_eq!(body["id"], "chatcmpl-moa-abc");
        assert_eq!(body["created"], 42);
        assert_eq!(progress_item_id(42), "msg_42");
    }

    #[test]
    fn failure_message_prefers_the_body_error() {
        let response = VirtualModelResponse {
            status_code: 502,
            body: json!({"error": {"message": "no concrete models available in the mesh"}}),
            headers: Vec::new(),
            event_stream: true,
        };

        assert_eq!(
            failure_message(&response),
            "no concrete models available in the mesh"
        );
    }
}
