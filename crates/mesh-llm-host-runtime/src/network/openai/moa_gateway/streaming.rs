use crate::network::openai::transport as proxy;
use mesh_mixture_of_agents as moa;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

/// Carries Responses-API streaming state from the progress phase to
/// the body writer so the full stream maintains monotonic
/// sequence_number and a stable created_at across both phases.
#[derive(Debug, Clone, Copy)]
pub(super) struct ProgressContinuation {
    /// `created_at` baked into the early `response.created` event;
    /// must be reused for the final `response.completed` so clients
    /// see one consistent timestamp for the response.
    pub(super) created_at: i64,
    /// Next `sequence_number` to use — strictly greater than the last
    /// `sequence_number` emitted by the progress phase.
    pub(super) next_sequence_number: i32,
}

/// Write the MoA response on the chosen transport (JSON or SSE), logging
/// (but not propagating) any I/O error.
///
/// Detect whether a MoA response body is signalling failure.
///
/// Two signals, either of which means "failure":
///
///   * Top-level `error` object — OpenAI-shape error envelope produced
///     by `moa::error_response`.
///   * `choices[0].finish_reason == "error"` — same convention applied
///     by the crate's response builder for in-band failure signalling.
///
/// Previously the HTTP-status decision was based on `TurnKind == Failed`,
/// but the tool-result reducer path can produce an error_response with
/// `TurnKind::ToolResult` when every reducer candidate fails. Tying the
/// status to the body's failure signal instead means *all* error-shaped
/// MoA responses get a non-200 status, regardless of which sub-flow
/// produced them.
pub(super) fn is_moa_failure_body(body: &serde_json::Value) -> bool {
    if body.get("error").is_some() {
        return true;
    }
    body.pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        == Some("error")
}

/// When the response body signals MoA failure (top-level `error` field or
/// `choices[0].finish_reason == "error"`) we send an HTTP 502 (Bad
/// Gateway), not HTTP 200. Unsophisticated clients that only check the
/// HTTP status need that status to actually reflect failure.
pub(super) async fn write_moa_response(
    tcp_stream: TcpStream,
    moa_result: &moa::TurnResult,
    extra_headers: &[(&str, String)],
    was_streaming: bool,
    response_adapter: proxy::ResponseAdapter,
) {
    let body = &moa_result.response_body;
    let is_failure = is_moa_failure_body(body);
    // Streaming + failure: respond as non-streaming HTTP 502 with the
    // structured error body. Failure path doesn't go through SSE in any
    // adapter mode — callers want a clean connection-level error.
    let (mode, result) = if was_streaming && !is_failure {
        match response_adapter {
            proxy::ResponseAdapter::OpenAiResponsesStream => (
                "SSE-responses",
                send_moa_as_responses_sse(
                    tcp_stream,
                    body,
                    extra_headers,
                    final_text_stream_mode_for_result(moa_result),
                )
                .await,
            ),
            // None, OpenAiChatCompletionsStream, OpenAiResponsesJson all
            // get the chat.completion.chunk SSE shape — the JSON-mode
            // adapter caller will never set was_streaming=true.
            _ => (
                "SSE-chat",
                send_moa_as_sse(
                    tcp_stream,
                    body,
                    extra_headers,
                    final_text_stream_mode_for_result(moa_result),
                )
                .await,
            ),
        }
    } else if is_failure {
        (
            "JSON-502",
            proxy::send_json_with_status_and_headers(tcp_stream, 502, body, extra_headers).await,
        )
    } else if response_adapter == proxy::ResponseAdapter::OpenAiResponsesJson {
        // Non-streaming Responses-API request: emit a Responses-shape
        // JSON body instead of the chat.completion shape.
        (
            "JSON-responses",
            proxy::send_json_ok_with_headers(
                tcp_stream,
                &chat_completion_to_responses_json(body),
                extra_headers,
            )
            .await,
        )
    } else {
        (
            "JSON",
            proxy::send_json_ok_with_headers(tcp_stream, body, extra_headers).await,
        )
    };
    if let Err(e) = result {
        tracing::warn!("MoA: response write failed ({mode}): {e}");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MoaFinalTextStreamMode {
    OneShot,
    ChunkedCommittedText,
}

pub(super) fn final_text_stream_mode_for_result(
    result: &moa::TurnResult,
) -> MoaFinalTextStreamMode {
    if result.reducer_used {
        MoaFinalTextStreamMode::OneShot
    } else {
        MoaFinalTextStreamMode::ChunkedCommittedText
    }
}

/// Send the MoA response as a one-shot SSE stream so SSE-only clients
/// (like Goose) can consume it. Emits one delta chunk with the full
/// content, then a `finish_reason: stop` chunk, then `[DONE]`.
///
/// `extra_headers` are emitted alongside the standard SSE response headers
/// (used to attach `x-moa-*` observability headers).
async fn send_moa_as_sse(
    stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    text_stream_mode: MoaFinalTextStreamMode,
) -> std::io::Result<()> {
    send_moa_as_sse_inner(stream, response, extra_headers, false, text_stream_mode).await
}

/// Write the standard SSE response header block, with optional
/// per-response extra headers (used for `x-moa-*` observability).
pub(super) async fn write_sse_response_headers(
    stream: &mut TcpStream,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Transfer-Encoding: chunked\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n",
    );
    for (name, value) in extra_headers {
        crate::network::openai::transport::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await
}

pub(super) async fn send_moa_as_sse_inner(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
    text_stream_mode: MoaFinalTextStreamMode,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_response_headers(&mut stream, extra_headers).await?;
    }

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-mesh");
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME);
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);

    let tool_calls = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .cloned();

    // Caller (`write_moa_response`) routes failure-shaped bodies to a
    // non-streaming 502 JSON response, so this function only ever sees a
    // successful turn. The only choice the SSE adapter still has to make
    // is `tool_calls` vs `stop`.
    let finish_reason: &str = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    debug_assert!(
        !is_moa_failure_body(response),
        "send_moa_as_sse received a failure body; should have routed to 502"
    );

    // Tool-call payloads are structured JSON — they must remain
    // atomic so harness parsers (Goose, OpenCode) see a single
    // well-formed tool_call object. Only the assistant *text* path
    // benefits from pseudo-streaming.
    if let Some(ref tcs) = tool_calls {
        let delta = serde_json::json!({
            "role": "assistant",
            "tool_calls": tcs.iter().enumerate().map(|(i, tc)| {
                serde_json::json!({
                    "index": i,
                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0"),
                    "type": "function",
                    "function": tc.get("function").cloned().unwrap_or(serde_json::json!({})),
                })
            }).collect::<Vec<_>>()
        });
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null,
            }]
        });
        let data = format!("data: {}\n\n", chunk);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
    } else {
        // Text path: stream committed non-reducer answers in chunks.
        // Reducer output remains one-shot because issue #618 explicitly
        // scoped reducer streaming out of the first MoA streaming pass.
        // First chunk carries `role: "assistant"`; continuation chunks
        // carry only `content` (matches OpenAI streaming convention).
        let pieces = content_pieces_for_streaming(&content, text_stream_mode);
        let chunk_delay = MOA_STREAM_CHUNK_DELAY;
        let inter_chunk_delay = if pieces.len() > 1 {
            Some(chunk_delay)
        } else {
            None
        };
        for (idx, piece) in pieces.iter().enumerate() {
            let delta = if idx == 0 {
                serde_json::json!({ "role": "assistant", "content": piece })
            } else {
                serde_json::json!({ "content": piece })
            };
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": null,
                }]
            });
            let data = format!("data: {}\n\n", chunk);
            let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
            stream.write_all(framed.as_bytes()).await?;
            stream.flush().await?;
            if let Some(delay) = inter_chunk_delay
                && idx + 1 < pieces.len()
            {
                tokio::time::sleep(delay).await;
            }
        }
    }

    let stop = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }]
    });
    let data = format!("data: {}\n\n", stop);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Strip `<think>...</think>` tags and orphan `</think>` from content.
/// Thin wrapper over the canonical implementation in moa::worker.
fn strip_think_from_content(text: &str) -> String {
    moa::strip_thinking(text)
}

/// Number of chunks to split MoA winner content into when emitting
/// pseudo-streaming SSE. Tuned for "feels live" — ~25 chunks over a
/// buffered response of any reasonable length lets the chat UI paint
/// progressively instead of jumping from spinner to wall-of-text.
const MOA_STREAM_CHUNKS: usize = 25;

/// Minimum content length (bytes) before pseudo-streaming kicks in.
/// Below this, the one-shot delta is fine and chunking just adds
/// scheduler noise. Checked against `content.len()` which is byte
/// length; the threshold is loose so the byte/char distinction
/// doesn't matter for non-ASCII (200 bytes ≥ 50 multi-byte chars,
/// well above the noise floor).
const MOA_STREAM_MIN_BYTES: usize = 200;

/// Delay between pseudo-stream chunks. Total animation budget for a
/// 25-chunk response is ~500ms, which feels live without artificially
/// slowing down agents that just want to read the whole reply.
const MOA_STREAM_CHUNK_DELAY: std::time::Duration = std::time::Duration::from_millis(20);

/// Split `content` into roughly `target_chunks` pieces along whitespace
/// or UTF-8 char boundaries. The returned slices, concatenated in order,
/// always reconstruct the original input exactly (no characters lost,
/// no separators inserted). Returns a single-element vector when
/// chunking is not worth the overhead (short content, target ≤ 1, or
/// content too short to split meaningfully).
fn chunk_content_for_streaming(content: &str, target_chunks: usize) -> Vec<&str> {
    if target_chunks <= 1
        || content.len() < MOA_STREAM_MIN_BYTES
        || content.chars().count() < target_chunks * 2
    {
        return vec![content];
    }

    // Walk char boundaries to compute desired cut points by char index,
    // then snap forward to the next whitespace boundary so we don't
    // split mid-word. If no whitespace exists (CJK, code blob, long
    // hash), fall through to the char-boundary cut.
    let total_chars = content.chars().count();
    let chars_per_chunk = total_chars / target_chunks;
    if chars_per_chunk == 0 {
        return vec![content];
    }

    let mut chunks = Vec::with_capacity(target_chunks);
    let mut cut_start = 0usize;
    let mut chars_since_last = 0usize;

    for (byte_idx, ch) in content.char_indices() {
        chars_since_last += 1;
        // Once we've passed the per-chunk char target, try to snap
        // forward to the next whitespace char so we cut on a word
        // boundary. If we're already on whitespace, cut here.
        if chars_since_last >= chars_per_chunk && ch.is_whitespace() {
            // Cut *after* the whitespace so the leading-space
            // boundary lives with the preceding chunk (matches how
            // word-by-word streaming reads).
            let cut_end = byte_idx + ch.len_utf8();
            if cut_end > cut_start {
                chunks.push(&content[cut_start..cut_end]);
                cut_start = cut_end;
                chars_since_last = 0;
            }
            if chunks.len() + 1 >= target_chunks {
                break;
            }
        }
    }

    if cut_start < content.len() {
        chunks.push(&content[cut_start..]);
    }

    // If we ended up with one chunk (no whitespace found), fall back
    // to a strict char-count split. Common for CJK or code-only output.
    if chunks.len() == 1 && total_chars >= target_chunks * 2 {
        chunks.clear();
        let mut cut_start = 0usize;
        let mut chars_since_last = 0usize;
        for (byte_idx, ch) in content.char_indices() {
            chars_since_last += 1;
            if chars_since_last >= chars_per_chunk {
                let cut_end = byte_idx + ch.len_utf8();
                chunks.push(&content[cut_start..cut_end]);
                cut_start = cut_end;
                chars_since_last = 0;
                if chunks.len() + 1 >= target_chunks {
                    break;
                }
            }
        }
        if cut_start < content.len() {
            chunks.push(&content[cut_start..]);
        }
    }

    chunks
}

fn content_pieces_for_streaming(
    content: &str,
    text_stream_mode: MoaFinalTextStreamMode,
) -> Vec<&str> {
    match text_stream_mode {
        MoaFinalTextStreamMode::OneShot => vec![content],
        MoaFinalTextStreamMode::ChunkedCommittedText => {
            chunk_content_for_streaming(content, MOA_STREAM_CHUNKS)
        }
    }
}

/// Emit the MoA response as an OpenAI Responses-API SSE stream so callers
/// that hit `/v1/responses` with `stream:true` get event shapes their parser
/// understands.
///
/// We synthesize the minimum set the standard Responses-API stream emits:
/// `response.created`, one or more `response.output_text.delta` events,
/// `response.output_text.done`, and `response.completed`. The text chunking
/// mode is chosen from the completed MoA turn: committed non-reducer answers
/// can be split for issue #618's visible streaming path, while reducer output
/// remains one-shot until reducer streaming is implemented deliberately.
async fn send_moa_as_responses_sse(
    stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    text_stream_mode: MoaFinalTextStreamMode,
) -> std::io::Result<()> {
    send_moa_as_responses_sse_inner(
        stream,
        response,
        extra_headers,
        false,
        text_stream_mode,
        None,
    )
    .await
}

pub(super) async fn send_moa_as_responses_sse_inner(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
    text_stream_mode: MoaFinalTextStreamMode,
    continuation: Option<ProgressContinuation>,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_response_headers(&mut stream, extra_headers).await?;
    }

    let response_id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("resp_moa")
        .to_string();
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME)
        .to_string();
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);
    // MoA's body is chat-shape; the Responses-API completed event
    // expects input_tokens / output_tokens. Translate before emitting
    // so downstream consumers (chat UI, billing) see the right keys.
    let usage = response
        .get("usage")
        .map(openai_frontend::responses::chat_usage_to_responses_usage);
    let item_id = format!("msg_moa_{}", short_id_from_response(response));

    // On the progress path, reuse the timestamp the early
    // `response.created` event already put on the wire, and start
    // sequence_number from where progress left off. Otherwise this
    // is a standalone Responses stream; compute a fresh created_at
    // and start the sequence counter at the conventional zero.
    let (created_at, mut sequence_number) = match continuation {
        Some(c) => (c.created_at, c.next_sequence_number),
        None => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            (now, 0)
        }
    };

    use openai_frontend::responses as resp;

    // `response.created` must come before any delta events. When the
    // progress path is driving us (continuation is Some), it already
    // emitted `response.created` up front with the correct id and
    // sequence_number=0 — emitting again would produce two `created`
    // events for the same stream with mismatched timestamps and a
    // duplicate sequence_number.
    if continuation.is_none() {
        let mut created =
            resp::responses_stream_created_event_with_sequence(&model, created_at, sequence_number);
        sequence_number = sequence_number.saturating_add(1);
        if let Some(obj) = created
            .get_mut("response")
            .and_then(serde_json::Value::as_object_mut)
        {
            obj.insert(
                "id".to_string(),
                serde_json::Value::String(response_id.clone()),
            );
        }
        let data = format!("data: {created}\n\n");
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
        stream.flush().await?;
    }

    let pieces = content_pieces_for_streaming(&content, text_stream_mode);
    let chunk_delay = MOA_STREAM_CHUNK_DELAY;
    let inter_chunk_delay = if pieces.len() > 1 {
        Some(chunk_delay)
    } else {
        None
    };
    for (idx, piece) in pieces.iter().enumerate() {
        let delta_event = resp::responses_stream_delta_event_with_logprobs_and_sequence(
            &item_id,
            piece,
            None,
            sequence_number,
        );
        sequence_number = sequence_number.saturating_add(1);
        let data = format!("data: {}\n\n", delta_event);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
        stream.flush().await?;
        if let Some(delay) = inter_chunk_delay
            && idx + 1 < pieces.len()
        {
            tokio::time::sleep(delay).await;
        }
    }

    let text_done =
        resp::responses_stream_text_done_event_with_sequence(&item_id, &content, sequence_number);
    sequence_number = sequence_number.saturating_add(1);
    let completed = resp::responses_stream_completed_event_with_sequence(
        &response_id,
        created_at,
        &model,
        &item_id,
        &content,
        usage,
        sequence_number,
    );
    let tail = [text_done, completed];
    for event in &tail {
        let data = format!("data: {}\n\n", event);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
    }

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Convert a chat.completion JSON body to a Responses-API JSON body.
/// Used for non-streaming `/v1/responses` requests against MoA.
fn chat_completion_to_responses_json(chat: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(chat).unwrap_or_default();
    match crate::network::openai::response_adapter::translate_chat_completion_to_responses(&bytes) {
        Ok(translated) => serde_json::from_slice(&translated).unwrap_or_else(|_| chat.clone()),
        Err(e) => {
            tracing::warn!("MoA: chat-to-responses JSON translate failed: {e}");
            chat.clone()
        }
    }
}

fn short_id_from_response(response: &serde_json::Value) -> String {
    response
        .get("id")
        .and_then(|v| v.as_str())
        .and_then(|id| id.rsplit('-').next())
        .unwrap_or("x")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_think_handles_simple_block() {
        assert_eq!(
            strip_think_from_content("<think>reasoning</think>answer"),
            "answer"
        );
    }

    #[test]
    fn strip_think_handles_orphan_close_tag() {
        // Orphan `</think>` is removed but prefix content is preserved.
        assert_eq!(
            strip_think_from_content("stuff</think>answer"),
            "stuffanswer"
        );
    }

    #[test]
    fn strip_think_handles_unclosed_block() {
        assert_eq!(
            strip_think_from_content("answer prefix<think>never closed"),
            "answer prefix"
        );
    }

    #[test]
    fn is_moa_failure_body_detects_top_level_error() {
        // Regression for PR #566 review (item #7): the HTTP status was
        // gated on `TurnKind == Failed`, but reducer-failure tool-result
        // turns produce an error_response with `TurnKind::ToolResult`.
        // The body still carries the canonical failure signals, so
        // status now follows the body.
        let body = serde_json::json!({
            "error": { "message": "reducer failed", "type": "moa_failure" },
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn is_moa_failure_body_detects_finish_reason_error() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn final_text_stream_mode_chunks_only_non_reducer_results() {
        assert_eq!(
            final_text_stream_mode_for_result(&moa_turn_result_for_stream_mode(false)),
            MoaFinalTextStreamMode::ChunkedCommittedText
        );
        assert_eq!(
            final_text_stream_mode_for_result(&moa_turn_result_for_stream_mode(true)),
            MoaFinalTextStreamMode::OneShot
        );
    }

    fn moa_turn_result_for_stream_mode(reducer_used: bool) -> moa::TurnResult {
        moa::TurnResult {
            response_body: fixture_chat_completion("answer"),
            worker_summaries: Vec::new(),
            reducer_used,
            reducer_attempts: u32::from(reducer_used),
            turn_kind: if reducer_used {
                moa::TurnKind::Fanout
            } else {
                moa::TurnKind::EarlyExit
            },
            elapsed_ms: 0,
        }
    }

    #[test]
    fn is_moa_failure_body_returns_false_for_success() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "hello" } }],
        });
        assert!(!is_moa_failure_body(&body));
    }
    #[test]
    fn is_moa_failure_body_returns_false_for_tool_calls() {
        let body = serde_json::json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                },
            }],
        });
        assert!(!is_moa_failure_body(&body));
    }

    // ── Streaming + failure routing ────────────────────────────────────
    //
    // The actual write path (`write_moa_response`) writes to a real
    // `TcpStream`, so we test the *decision* it makes by extracting the
    // failure detection into `is_moa_failure_body` and proving the
    // routing logic with the same booleans the writer uses.
    //
    // The contract is:
    //   was_streaming=false, is_failure=false  -> JSON 200
    //   was_streaming=false, is_failure=true   -> JSON 502
    //   was_streaming=true,  is_failure=false  -> SSE
    //   was_streaming=true,  is_failure=true   -> JSON 502 (NOT SSE 200)
    // The last row is the PR #612 review finding: streaming MoA failures
    // must surface as a real 502 at the HTTP layer instead of streaming
    // a 200 SSE carrying an in-band error.

    fn route_decision(was_streaming: bool, is_failure: bool) -> &'static str {
        if was_streaming && !is_failure {
            "sse"
        } else if is_failure {
            "json-502"
        } else {
            "json-200"
        }
    }

    #[test]
    fn streaming_success_routes_to_sse() {
        assert_eq!(route_decision(true, false), "sse");
    }

    #[test]
    fn streaming_failure_routes_to_json_502_not_sse() {
        // Regression for PR #612 review: streaming failures previously
        // went out as `SSE 200` + in-band `finish_reason: "error"`.
        // Now they collapse to a non-streaming JSON 502, matching the
        // OpenAI API and the non-streaming MoA failure path.
        assert_eq!(route_decision(true, true), "json-502");
    }

    #[test]
    fn non_streaming_success_routes_to_json_200() {
        assert_eq!(route_decision(false, false), "json-200");
    }

    #[test]
    fn non_streaming_failure_routes_to_json_502() {
        assert_eq!(route_decision(false, true), "json-502");
    }

    // ── Responses-API adapter ───────────────────────────────────────
    //
    // When the request came in via /v1/responses, MoA's response must
    // be rendered in the Responses-API shape, not chat.completion. The
    // chat UI's streaming parser ignores chat.completion.chunk events,
    // which is what caused the "streaming response" spinner with no
    // visible text on the public mesh.

    fn fixture_chat_completion(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        })
    }

    #[test]
    fn chat_completion_to_responses_json_returns_response_object() {
        // Non-streaming /v1/responses with model=mesh: the body that
        // reaches the client must be Responses-shape, not chat-shape.
        let chat = fixture_chat_completion("hello world");
        let responses = chat_completion_to_responses_json(&chat);
        assert_eq!(
            responses.get("object").and_then(|v| v.as_str()),
            Some("response"),
            "got: {}",
            serde_json::to_string(&responses).unwrap_or_default()
        );
        // The text must survive translation.
        let text = serde_json::to_string(&responses).unwrap_or_default();
        assert!(
            text.contains("hello world"),
            "response body must carry the original content; got {text}"
        );
    }

    #[test]
    fn chat_completion_to_responses_json_passes_through_on_malformed() {
        // Defensive: if the translator can't make sense of the body
        // we return the chat body unchanged rather than blowing up.
        let bogus = serde_json::json!({ "not": "a chat completion" });
        let out = chat_completion_to_responses_json(&bogus);
        // The translator may either succeed (producing an empty
        // response) or fall back to the input; both behaviours are
        // acceptable, what matters is no panic and a JSON value.
        assert!(out.is_object());
    }

    /// Run `send_moa_as_responses_sse` against a real TCP loopback
    /// pair and return the raw bytes the client received as a string.
    /// Includes HTTP/1.1 headers and the chunked-transfer framing
    /// around each SSE event. Callers in this module match by
    /// `.contains(...)`, which is robust to framing without needing
    /// to parse it.
    async fn capture_responses_sse_body(response: serde_json::Value) -> String {
        capture_responses_sse_body_with_mode(response, MoaFinalTextStreamMode::ChunkedCommittedText)
            .await
    }

    async fn capture_responses_sse_body_with_mode(
        response: serde_json::Value,
        text_stream_mode: MoaFinalTextStreamMode,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");

        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            send_moa_as_responses_sse(socket, &response, &[], text_stream_mode)
                .await
                .expect("sse write");
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn responses_sse_uses_same_response_id_for_created_and_completed() {
        // Regression: created and completed events used different
        // `response.id` values (one auto-generated, one from the chat
        // body), breaking clients that correlate by id.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-correlation",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }]
        });

        let raw =
            capture_responses_sse_body_with_mode(response, MoaFinalTextStreamMode::OneShot).await;

        // Extract every `data: { ... }` JSON blob and look at
        // (event.type, event.response.id).
        let mut ids = Vec::<(String, String)>::new();
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if event_type == "response.created" || event_type == "response.completed" {
                let id = v
                    .pointer("/response/id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                ids.push((event_type.to_string(), id));
            }
        }

        assert_eq!(ids.len(), 2, "need created + completed; got {ids:?}");
        assert_eq!(ids[0].1, "chatcmpl-moa-correlation");
        assert_eq!(
            ids[0].1, ids[1].1,
            "created and completed must share response.id: {ids:?}"
        );
    }

    #[tokio::test]
    async fn responses_sse_emits_responses_shape_usage_not_chat_shape() {
        // Regression: MoA was forwarding the chat-completion `usage`
        // object (prompt_tokens/completion_tokens) straight into the
        // Responses-API completed event, which expects
        // input_tokens/output_tokens. Downstream consumers that read
        // `response.usage.input_tokens` saw `undefined`.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "total_tokens": 24
            }
        });

        let raw =
            capture_responses_sse_body_with_mode(response, MoaFinalTextStreamMode::OneShot).await;

        // The completed event carries the response object including
        // usage. We assert by string match so we're robust to
        // serializer ordering.
        assert!(
            raw.contains("\"input_tokens\":11"),
            "expected input_tokens=11 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"output_tokens\":13"),
            "expected output_tokens=13 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"total_tokens\":24"),
            "expected total_tokens=24 in SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"prompt_tokens\":"),
            "chat-shape prompt_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"completion_tokens\":"),
            "chat-shape completion_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
    }
    // ── chunk_content_for_streaming ────────────────────────────────

    #[test]
    fn chunk_helper_empty_input_returns_single_empty_chunk() {
        // Empty input still returns a one-element vec (`vec![""]`), not
        // an empty slice — the SSE writer expects to always emit at
        // least one delta event so it can attach role/finish metadata.
        assert_eq!(chunk_content_for_streaming("", 25), vec![""]);
    }

    #[test]
    fn chunk_helper_short_input_returns_single_chunk() {
        // Below MOA_STREAM_MIN_BYTES — chunking overhead not worth it.
        let s = "hello world this is short";
        let out = chunk_content_for_streaming(s, 25);
        assert_eq!(out, vec![s]);
    }

    #[test]
    fn chunk_helper_target_one_returns_single_chunk() {
        let s = "x".repeat(500);
        let out = chunk_content_for_streaming(&s, 1);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn chunk_helper_long_text_splits_on_word_boundaries() {
        // 400+ chars of normal English prose.
        let s = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(out.len() > 1, "expected multiple chunks; got {}", out.len());
        assert!(
            out.len() <= 11,
            "expected at most ~10 chunks; got {}",
            out.len()
        );
        // Reconstruction is exact: no bytes lost or added.
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
        // Word boundaries: each non-final chunk ends in whitespace.
        for chunk in &out[..out.len() - 1] {
            assert!(
                chunk
                    .chars()
                    .last()
                    .map(|c| c.is_whitespace())
                    .unwrap_or(false),
                "non-final chunk should end on whitespace: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn chunk_helper_preserves_utf8_boundaries_for_cjk() {
        // No whitespace, multi-byte chars. Should still split cleanly
        // along char boundaries (no panic, exact reconstruction).
        let s = "中文测试内容".repeat(60); // 360 chars, all 3-byte UTF-8
        assert!(s.len() >= MOA_STREAM_MIN_BYTES);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(out.len() > 1, "CJK should still chunk; got {}", out.len());
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
        // Each chunk is valid UTF-8 (trivially, since &str by construction).
        for chunk in &out {
            assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
        }
    }

    #[test]
    fn chunk_helper_handles_text_with_no_whitespace_fallback() {
        // A long URL/hash — no whitespace to snap to. Helper should
        // fall through to char-boundary splitting.
        let s = "a".repeat(600);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(
            out.len() > 1,
            "expected fallback chunking; got {}",
            out.len()
        );
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
    }

    async fn capture_chat_sse_body(response: serde_json::Value) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            send_moa_as_sse(
                socket,
                &response,
                &[],
                MoaFinalTextStreamMode::ChunkedCommittedText,
            )
            .await
            .expect("sse");
        });
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn count_delta_events_with_content(raw: &str) -> usize {
        let mut count = 0;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            if v.pointer("/choices/0/delta/content")
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
                .is_some()
            {
                count += 1;
            }
        }
        count
    }

    #[tokio::test]
    async fn chat_sse_emits_multiple_deltas_for_long_content() {
        // ≥ MOA_STREAM_MIN_BYTES of word-spaced English → must split.
        let long_content = "Hello world. ".repeat(40);
        let response = serde_json::json!({
            "id": "chatcmpl-moa-chunky",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": long_content },
                "finish_reason": "stop"
            }]
        });
        // The real MOA_STREAM_CHUNK_DELAY (20ms) × ~25 chunks adds
        // ~500ms to test runtime — acceptable since this is the only
        // chunked-delay test on the chat path.
        let raw = capture_chat_sse_body(response).await;
        let n = count_delta_events_with_content(&raw);
        assert!(
            n > 1,
            "expected multiple content delta events; got {n}\nraw: {raw}"
        );
    }

    #[tokio::test]
    async fn chat_sse_tool_calls_remain_atomic() {
        // Tool-call payloads must NOT be chunked — harness parsers
        // (Goose, OpenCode) need a single well-formed tool_call object.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-tool",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\":\"/x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let raw = capture_chat_sse_body(response).await;
        // Count delta events with tool_calls.
        let mut tool_deltas = 0;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            if v.pointer("/choices/0/delta/tool_calls").is_some() {
                tool_deltas += 1;
            }
        }
        assert_eq!(
            tool_deltas, 1,
            "tool_calls must arrive as exactly one atomic delta; got {tool_deltas}\nraw: {raw}"
        );
    }

    #[tokio::test]
    async fn responses_sse_emits_multiple_deltas_for_long_content() {
        let long_content = "Hello world. ".repeat(40);
        let response = serde_json::json!({
            "id": "chatcmpl-moa-resp-chunky",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": long_content },
                "finish_reason": "stop"
            }]
        });
        // ~500ms test runtime acceptable (MOA_STREAM_CHUNK_DELAY × N).
        let raw = capture_responses_sse_body(response).await;
        // Count response.output_text.delta events.
        let delta_count = count_responses_output_text_deltas(&raw);
        assert!(
            delta_count >= 5,
            "expected at least 5 output_text.delta events; got {delta_count}\nraw: {raw}"
        );
    }

    fn count_responses_output_text_deltas(raw: &str) -> usize {
        raw.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| payload.trim() != "[DONE]")
            .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
            .filter(|v| {
                v.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta")
            })
            .count()
    }

    #[tokio::test]
    async fn responses_sse_keeps_reducer_output_one_delta_for_long_content() {
        let long_content = "Reduced answer. ".repeat(40);
        let response = serde_json::json!({
            "id": "chatcmpl-moa-resp-reducer",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": long_content },
                "finish_reason": "stop"
            }]
        });

        let raw =
            capture_responses_sse_body_with_mode(response, MoaFinalTextStreamMode::OneShot).await;
        let delta_count = count_responses_output_text_deltas(&raw);
        assert_eq!(
            delta_count, 1,
            "reducer output is intentionally not pseudo-streamed; raw: {raw}"
        );
    }
}
