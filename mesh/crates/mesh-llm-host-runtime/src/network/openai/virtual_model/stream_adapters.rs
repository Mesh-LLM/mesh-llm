//! Buffered virtual-model bodies → the caller's stream adapter.
//!
//! A virtual model answers with one complete body, so these writers translate
//! it into whichever stream the caller asked for. [`super::progress`] may
//! already have written the response head — and, for the Responses adapter, an
//! opening `response.created` plus progress events — which
//! `header_already_sent` (or a `Some` continuation) reports.

use super::progress::ProgressContinuation;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

pub(super) async fn write_sse_headers(
    stream: &mut ClientStream,
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
        proxy::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await
}

async fn write_sse_event(
    stream: &mut ClientStream,
    event: &serde_json::Value,
) -> std::io::Result<()> {
    write_sse_data(stream, &event.to_string()).await
}

pub(super) async fn write_sse_data(stream: &mut ClientStream, data: &str) -> std::io::Result<()> {
    let payload = format!("data: {data}\n\n");
    let framed = format!("{:x}\r\n{}\r\n", payload.len(), payload);
    stream.write_all(framed.as_bytes()).await
}

/// Write one named SSE frame — `event: <name>` plus the data payload.
pub(super) async fn write_named_sse_data(
    stream: &mut ClientStream,
    name: &str,
    data: &str,
) -> std::io::Result<()> {
    let payload = format!("event: {name}\ndata: {data}\n\n");
    let framed = format!("{:x}\r\n{}\r\n", payload.len(), payload);
    stream.write_all(framed.as_bytes()).await
}

/// Write one `/v1/responses` SSE frame. Responses streams name their events,
/// matching the framing the host uses for a served stream, so the frame name
/// comes from the event's own `type`.
async fn write_named_sse_event(
    stream: &mut ClientStream,
    event: &serde_json::Value,
) -> std::io::Result<()> {
    let Some(name) = event.get("type").and_then(serde_json::Value::as_str) else {
        return write_sse_event(stream, event).await;
    };
    write_named_sse_data(stream, name, &event.to_string()).await
}

/// Close a chunked SSE stream: a `[DONE]` sentinel for the OpenAI-shaped
/// adapters, then the terminating chunk.
async fn finish_sse(stream: &mut ClientStream) -> std::io::Result<()> {
    write_sse_data(stream, "[DONE]").await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await
}

pub(super) async fn send_chat_sse(
    mut stream: ClientStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_headers(&mut stream, extra_headers).await?;
    }
    let id = response
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("chatcmpl-virtual");
    let model = response
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("virtual-model");
    for chunk in chat_sse_chunks(response, id, model) {
        write_sse_event(&mut stream, &chunk).await?;
    }
    finish_sse(&mut stream).await
}

/// Convert one buffered completion into OpenAI-compatible chat stream chunks.
/// Text and tool calls can coexist in a single assistant message, every tool
/// call carries its stream position, and the terminal chunk preserves both
/// the backend finish reason and usage accounting.
fn chat_sse_chunks(response: &Value, id: &str, model: &str) -> Vec<Value> {
    let message = response
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut delta = serde_json::Map::new();
    delta.insert("role".into(), json!("assistant"));
    if let Some(content) = message.get("content") {
        delta.insert("content".into(), content.clone());
    }
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(index, call)| {
                    let mut call = call.clone();
                    if let Some(object) = call.as_object_mut() {
                        object.insert("index".into(), json!(index));
                    }
                    call
                })
                .collect::<Vec<_>>()
        });
    if let Some(tool_calls) = tool_calls.as_ref() {
        delta.insert("tool_calls".into(), json!(tool_calls));
    }

    let default_finish = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    let finish_reason = response
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .unwrap_or(default_finish);
    let first = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    });
    let mut terminal = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    });
    if let Some(usage) = response.get("usage") {
        terminal["usage"] = usage.clone();
    }
    vec![first, terminal]
}

pub(super) async fn send_responses_sse(
    mut stream: ClientStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    continuation: Option<ProgressContinuation>,
) -> std::io::Result<()> {
    if continuation.is_none() {
        write_sse_headers(&mut stream, extra_headers).await?;
    }
    for event in responses_stream_events(response, continuation) {
        write_named_sse_event(&mut stream, &event).await?;
    }
    finish_sse(&mut stream).await
}

/// Serve an Anthropic Messages stream from a buffered chat-shaped body.
///
/// Anthropic clients see `message_start` / content block / `message_delta` /
/// `message_stop` events; there is no `[DONE]` sentinel, the terminating
/// chunk ends the response.
pub(super) async fn send_anthropic_messages_sse(
    mut stream: ClientStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_headers(&mut stream, extra_headers).await?;
    }
    let events = skippy_inference_api::anthropic::completion_events(response)
        .map_err(std::io::Error::other)?;
    for event in events {
        let data = serde_json::to_string(&event).map_err(std::io::Error::other)?;
        crate::network::openai::response_adapter::write_chunked_sse_event(
            &mut stream,
            Some(event.event_name()),
            &data,
        )
        .await?;
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await
}

/// The `/v1/responses` event sequence for a buffered body.
///
/// With a `continuation` the head and the progress events are already on the
/// wire: the replay's own `response.created` is dropped and the sequence
/// resumes after the progress phase, so a strict Responses client still sees
/// one created event and a gap-free, increasing `sequence_number`.
fn responses_stream_events(
    response: &serde_json::Value,
    continuation: Option<ProgressContinuation>,
) -> Vec<Value> {
    let responses = chat_completion_to_responses_json(response);
    let mut events = skippy_inference_api::responses_stream_events_for_response(&responses);
    let Some(continuation) = continuation else {
        return events;
    };
    events.retain(|event| {
        event.get("type").and_then(serde_json::Value::as_str) != Some("response.created")
    });
    let mut next = continuation.next_sequence_number;
    for event in events.iter_mut() {
        if let Some(object) = event.as_object_mut() {
            object.insert("sequence_number".into(), json!(next));
        }
        next = next.saturating_add(1);
    }
    events
}

pub(super) fn chat_completion_to_responses_json(chat: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(chat).unwrap_or_default();
    match super::super::response_adapter::translate_chat_completion_to_responses(&bytes) {
        Ok(translated) => serde_json::from_slice(&translated).unwrap_or_else(|_| chat.clone()),
        Err(error) => {
            tracing::warn!("virtual-model response translation failed: {error}");
            chat.clone()
        }
    }
}

/// Serve a buffered chat-shaped body as an Anthropic Messages response.
///
/// A failed turn that carries no top-level `error` object is normalized into
/// one first: the Anthropic envelope has no way to express a chat completion
/// whose `finish_reason` is `error`, and translating it as a success would
/// hand the caller an empty message.
pub(super) fn chat_completion_to_messages_json(
    chat: &serde_json::Value,
    failed: bool,
) -> serde_json::Value {
    let source = if failed && chat.get("error").is_none() {
        json!({"error": {"message": "virtual model turn failed"}})
    } else {
        chat.clone()
    };
    match skippy_inference_api::anthropic::translate_chat_value(&source) {
        Ok(translated) => translated,
        Err(error) => {
            tracing::warn!("virtual-model Messages translation failed: {error}");
            source
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_completion() -> Value {
        json!({
            "id": "chatcmpl-moa-deadbeef",
            "object": "chat.completion",
            "created": 1_600_000_000,
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "committee answer"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })
    }

    #[test]
    fn messages_json_uses_the_anthropic_envelope() {
        let translated = chat_completion_to_messages_json(&chat_completion(), false);

        assert_eq!(translated["type"], "message");
        assert_eq!(translated["role"], "assistant");
        assert_eq!(translated["content"][0]["type"], "text");
        assert_eq!(translated["content"][0]["text"], "committee answer");
        assert_eq!(translated["stop_reason"], "end_turn");
    }

    #[test]
    fn messages_json_normalizes_a_failure_without_an_error_object() {
        let mut failed = chat_completion();
        failed["choices"][0]["finish_reason"] = json!("error");
        failed["choices"][0]["message"]["content"] = json!("");

        let translated = chat_completion_to_messages_json(&failed, true);

        assert_eq!(translated["type"], "error");
        assert!(translated["error"]["message"].as_str().is_some());
    }

    #[test]
    fn messages_json_keeps_a_plugin_error_envelope() {
        let rejected = json!({"error": {
            "message": "no concrete model satisfies the request capabilities",
            "type": "virtual_model_error"
        }});

        let translated = chat_completion_to_messages_json(&rejected, true);

        assert_eq!(translated["type"], "error");
        assert_eq!(
            translated["error"]["message"],
            "no concrete model satisfies the request capabilities"
        );
    }

    #[test]
    fn responses_stream_starts_with_created_without_a_continuation() {
        let events = responses_stream_events(&chat_completion(), None);

        assert_eq!(events[0]["type"], "response.created");
        assert_eq!(events[0]["sequence_number"], 0);
        assert!(
            events
                .iter()
                .filter_map(|event| event["sequence_number"].as_i64())
                .collect::<Vec<_>>()
                .windows(2)
                .all(|pair| pair[1] > pair[0]),
            "sequence numbers must increase: {events:?}"
        );
    }

    #[test]
    fn responses_stream_continuation_resumes_after_the_progress_phase() {
        let events = responses_stream_events(
            &chat_completion(),
            Some(ProgressContinuation {
                created_at: 1_600_000_000,
                next_sequence_number: 4,
            }),
        );

        assert!(
            events
                .iter()
                .all(|event| event["type"] != "response.created"),
            "the progress phase already sent response.created: {events:?}"
        );
        assert_eq!(events[0]["sequence_number"], 4);
        let numbers = events
            .iter()
            .map(|event| event["sequence_number"].as_i64().expect("sequence_number"))
            .collect::<Vec<_>>();
        assert_eq!(
            numbers,
            (4..4 + numbers.len() as i64).collect::<Vec<_>>(),
            "the replay must continue the progress sequence without gaps"
        );
    }

    #[test]
    fn responses_stream_replay_shares_the_progress_item_id() {
        // The progress deltas and the replayed content must reference one item
        // id, both derived from the `created` timestamp the committed phase
        // baked into `response.created`.
        let mut body = chat_completion();
        super::super::progress::align_committed_body(&mut body, "chatcmpl-moa-deadbeef", 42);

        let events = responses_stream_events(
            &body,
            Some(ProgressContinuation {
                created_at: 42,
                next_sequence_number: 1,
            }),
        );

        let item_id = events
            .iter()
            .find(|event| event["type"] == "response.output_item.added")
            .and_then(|event| event["item"]["id"].as_str())
            .expect("the replay names the output item");
        assert_eq!(
            item_id,
            super::super::progress::progress_item_id(42),
            "progress deltas must land on the item the content lands on"
        );
    }

    #[test]
    fn chat_stream_preserves_text_indexed_tools_usage_and_length_finish() {
        let response = json!({
            "id": "chatcmpl-mixed",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will check that.",
                    "tool_calls": [
                        {"id": "call-a", "type": "function", "function": {"name": "first", "arguments": "{}"}},
                        {"id": "call-b", "type": "function", "function": {"name": "second", "arguments": "{}"}}
                    ]
                },
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        });

        let chunks = chat_sse_chunks(&response, "chatcmpl-mixed", "mesh");

        assert_eq!(
            chunks[0]["choices"][0]["delta"]["content"],
            "I will check that."
        );
        assert_eq!(
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            chunks[0]["choices"][0]["delta"]["tool_calls"][1]["index"],
            1
        );
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "length");
        assert_eq!(chunks[1]["usage"], response["usage"]);
    }
}
