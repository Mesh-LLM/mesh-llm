//! Function calls in a translated Responses stream: chat-completion
//! `delta.tool_calls` fragments become `function_call` output items with
//! their own output positions, argument deltas, and terminal handling.

use super::{
    ResponsesStreamRelayState, emit_translated_stream_done_event, write_captured_sse_event,
};
use crate::logging::OpenAiStreamArtifactCapture;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::response_adapter;
use anyhow::Result;

/// One upstream tool call as the Responses stream carries it: a
/// `function_call` output item, announced once its name is known, whose
/// arguments stream as `response.function_call_arguments.delta` events.
#[derive(Debug)]
pub(super) struct TranslatedFunctionCall {
    pub(super) item_id: String,
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) arguments: String,
    /// Bytes of `arguments` already sent as argument deltas.
    pub(super) streamed: usize,
    /// Output position, assigned when `response.output_item.added` is sent.
    pub(super) output_index: Option<i32>,
}

impl TranslatedFunctionCall {
    // The same ids as the non-streamed translation's function_call items.
    pub(super) fn new(created_at: i64, index: u64) -> Self {
        Self {
            item_id: format!("fc_{created_at}_{index}"),
            call_id: format!("call_{created_at}_{index}"),
            name: String::new(),
            arguments: String::new(),
            streamed: 0,
            output_index: None,
        }
    }

    pub(super) fn item(&self, status: &str, arguments: &str) -> serde_json::Value {
        serde_json::json!({
            "id": self.item_id,
            "type": "function_call",
            "status": status,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": arguments,
        })
    }
}

/// The message-item builders place the message at output 0. Once function
/// calls can come first, move each event to where the message actually is.
pub(super) fn at_output_index(
    mut event: serde_json::Value,
    output_index: i32,
) -> serde_json::Value {
    if let Some(object) = event
        .as_object_mut()
        .filter(|object| object.contains_key("output_index"))
    {
        object.insert("output_index".into(), serde_json::json!(output_index));
    }
    event
}

pub(super) fn note_translated_finish_reason(
    state: &mut ResponsesStreamRelayState,
    chunk: &skippy_inference_api::responses::ChatCompletionStreamChunk,
) {
    if chunk
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
        == Some("length")
    {
        state.cut_by_length = true;
    }
}

/// Folds the chunk's `delta.tool_calls` into function calls and streams them:
/// a call is announced once its name is known, and its arguments follow as
/// they arrive, including any that came before the name.
pub(super) async fn emit_translated_function_call_deltas(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    chunk: &skippy_inference_api::responses::ChatCompletionStreamChunk,
) -> Result<()> {
    let Some(tool_calls) = chunk
        .choices
        .first()
        .and_then(|choice| choice.delta.as_ref())
        .and_then(|delta| delta.tool_calls.as_ref())
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(());
    };
    for (position, tool_call) in tool_calls.iter().enumerate() {
        let index = tool_call
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(position as u64);
        let created_at = state.created_at;
        let call = state
            .function_calls
            .entry(index)
            .or_insert_with(|| TranslatedFunctionCall::new(created_at, index));
        // Some backends repeat `"id": ""` or `"name": ""` in later fragments;
        // those must not blank what an earlier fragment set.
        if let Some(id) = tool_call
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|id| !id.is_empty())
        {
            call.call_id = id.to_string();
        }
        let function = tool_call.get("function");
        if let Some(name) = function
            .and_then(|function| function.get("name"))
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
        {
            call.name = name.to_string();
        }
        if let Some(arguments) = function
            .and_then(|function| function.get("arguments"))
            .and_then(serde_json::Value::as_str)
        {
            call.arguments.push_str(arguments);
        }
        if !call.name.is_empty() {
            announce_and_stream_function_call(tcp_stream, capture, state, index).await?;
        }
    }
    Ok(())
}

/// Sends `response.output_item.added` for the call if it has not been sent
/// yet, then whatever of its arguments has not been streamed.
pub(super) async fn announce_and_stream_function_call(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    index: u64,
) -> Result<()> {
    if state
        .function_calls
        .get(&index)
        .is_some_and(|call| call.output_index.is_none())
    {
        let output_index = state.claim_output_index();
        let sequence_number = state.next_sequence_number();
        let Some(call) = state.function_calls.get_mut(&index) else {
            return Ok(());
        };
        call.output_index = Some(output_index);
        let added = serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": call.item("in_progress", ""),
        });
        write_captured_sse_event(
            tcp_stream,
            capture,
            Some("response.output_item.added"),
            &added.to_string(),
        )
        .await?;
    }
    let Some(call) = state.function_calls.get(&index) else {
        return Ok(());
    };
    if call.streamed == call.arguments.len() {
        return Ok(());
    }
    let delta = call.arguments[call.streamed..].to_string();
    let streamed = call.arguments.len();
    let item_id = call.item_id.clone();
    let output_index = call.output_index.unwrap_or_default();
    let sequence_number = state.next_sequence_number();
    let event = serde_json::json!({
        "type": "response.function_call_arguments.delta",
        "sequence_number": sequence_number,
        "item_id": item_id,
        "output_index": output_index,
        "delta": delta,
    });
    write_captured_sse_event(
        tcp_stream,
        capture,
        Some("response.function_call_arguments.delta"),
        &event.to_string(),
    )
    .await?;
    if let Some(call) = state.function_calls.get_mut(&index) {
        call.streamed = streamed;
    }
    Ok(())
}

/// `response.completed` listing every item in output order, or
/// `response.incomplete` when the token limit cut the function calls.
pub(super) fn translated_terminal_event(
    state: &mut ResponsesStreamRelayState,
    finished: Vec<(i32, serde_json::Value)>,
    calls_cut: bool,
) -> (&'static str, serde_json::Value) {
    let sequence_number = state.next_sequence_number();
    let mut event = response_adapter::responses_stream_completed_event_with_sequence(
        &state.response_id,
        state.created_at,
        &state.model,
        &state.item_id,
        &state.output_text,
        state.usage.clone(),
        sequence_number,
    );
    if !finished.is_empty() {
        // The builder's output holds only the message.
        let mut output = finished;
        if let Some(message_index) = state.message_output_index
            && let Some(message) = event.pointer("/response/output/0").cloned()
        {
            output.push((message_index, message));
        }
        output.sort_by_key(|(output_index, _)| *output_index);
        if let Some(slot) = event.pointer_mut("/response/output") {
            *slot = serde_json::Value::Array(output.into_iter().map(|(_, item)| item).collect());
        }
    }
    if !calls_cut {
        return ("response.completed", event);
    }
    event["type"] = serde_json::json!("response.incomplete");
    if let Some(response) = event
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        response.insert("status".into(), serde_json::json!("incomplete"));
        // The builder filled completed_at for a completed response; only a
        // completed status carries one.
        response.insert("completed_at".into(), serde_json::Value::Null);
        response.insert(
            "incomplete_details".into(),
            serde_json::json!({ "reason": "max_output_tokens" }),
        );
    }
    ("response.incomplete", event)
}

/// Closes every announced function call, in output order, and returns its
/// item with that position for the terminal event. Calls cut by the token
/// limit stay open and come back `incomplete`.
pub(super) async fn finish_translated_function_calls(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    state: &mut ResponsesStreamRelayState,
    calls_cut: bool,
) -> Result<Vec<(i32, serde_json::Value)>> {
    let status = if calls_cut { "incomplete" } else { "completed" };
    let mut calls = state
        .function_calls
        .values()
        .filter_map(|call| {
            let output_index = call.output_index?;
            // A complete call without arguments closes with an empty JSON
            // object, which clients and backends parse; a cut call keeps the
            // fragment it got.
            let arguments = if call.arguments.is_empty() && !calls_cut {
                "{}".to_string()
            } else {
                call.arguments.clone()
            };
            Some((
                output_index,
                call.item_id.clone(),
                arguments.clone(),
                call.item(status, &arguments),
            ))
        })
        .collect::<Vec<_>>();
    calls.sort_by_key(|(output_index, ..)| *output_index);
    if calls_cut {
        return Ok(calls
            .into_iter()
            .map(|(output_index, _, _, item)| (output_index, item))
            .collect());
    }
    let mut finished = Vec::with_capacity(calls.len());
    for (output_index, item_id, arguments, item) in calls {
        let arguments_done = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": state.next_sequence_number(),
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments,
        });
        emit_translated_stream_done_event(
            tcp_stream,
            capture,
            Some("response.function_call_arguments.done"),
            arguments_done.to_string(),
        )
        .await?;
        let item_done = serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": state.next_sequence_number(),
            "output_index": output_index,
            "item": item,
        });
        emit_translated_stream_done_event(
            tcp_stream,
            capture,
            Some("response.output_item.done"),
            item_done.to_string(),
        )
        .await?;
        finished.push((output_index, item));
    }
    Ok(finished)
}

#[cfg(test)]
mod tests {
    use super::super::tests::relay_translated_over;
    use crate::network::openai::response::common::RouteAttemptResult;

    /// The JSON events of a relayed SSE body, in order.
    fn sse_json_events(body: &str) -> Vec<serde_json::Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }

    fn events_of<'a>(events: &'a [serde_json::Value], kind: &str) -> Vec<&'a serde_json::Value> {
        events
            .iter()
            .filter(|event| event["type"] == kind)
            .collect()
    }

    /// A tool call the upstream streams as chat chunks reaches a streamed
    /// /v1/responses client as a function_call item with its arguments, the
    /// way OpenAI streams it, so Responses clients such as Codex can call
    /// tools through the mesh. Before, only reasoning and an empty message
    /// came through. A later fragment repeating an empty id and name, as some
    /// backends send, leaves the call's own untouched.
    #[tokio::test]
    async fn translated_stream_carries_a_streamed_tool_call_as_a_function_call() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"","type":"function","function":{"name":"","arguments":"{\"path\":"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        let added = events_of(&events, "response.output_item.added");
        assert_eq!(added.len(), 1, "one item, the function call:\n{body}");
        assert_eq!(added[0]["output_index"], 0);
        assert_eq!(added[0]["item"]["type"], "function_call");
        assert_eq!(added[0]["item"]["status"], "in_progress");
        assert_eq!(added[0]["item"]["call_id"], "call_abc");
        assert_eq!(added[0]["item"]["name"], "read");
        let item_id = added[0]["item"]["id"].as_str().unwrap();

        let streamed = events_of(&events, "response.function_call_arguments.delta")
            .iter()
            .map(|event| {
                assert_eq!(event["item_id"], item_id);
                event["delta"].as_str().unwrap().to_string()
            })
            .collect::<String>();
        assert_eq!(streamed, r#"{"path":"a.txt"}"#);
        let done = events_of(&events, "response.function_call_arguments.done");
        assert_eq!(done.len(), 1);
        assert_eq!(done[0]["arguments"], r#"{"path":"a.txt"}"#);

        let item_done = events_of(&events, "response.output_item.done");
        assert_eq!(item_done.len(), 1);
        assert_eq!(item_done[0]["item"]["status"], "completed");
        assert_eq!(item_done[0]["item"]["arguments"], r#"{"path":"a.txt"}"#);

        let completed = events_of(&events, "response.completed");
        let output = completed[0]["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "no empty message next to the call");
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["call_id"], "call_abc");
        assert_eq!(output[0]["name"], "read");
        assert_eq!(output[0]["arguments"], r#"{"path":"a.txt"}"#);
        assert!(events_of(&events, "response.output_text.delta").is_empty());

        let sequence = events
            .iter()
            .filter_map(|event| event["sequence_number"].as_i64())
            .collect::<Vec<_>>();
        assert!(sequence.windows(2).all(|pair| pair[0] < pair[1]));
    }

    /// Text that comes before the calls keeps the message first, and
    /// concurrent calls keep their own items, in the order they appeared.
    #[tokio::test]
    async fn translated_stream_orders_text_before_concurrent_function_calls() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"role":"assistant","content":"Reading both."},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"read","arguments":"{\"path\":\"b\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        let added = events_of(&events, "response.output_item.added")
            .iter()
            .map(|event| {
                (
                    event["output_index"].as_i64().unwrap(),
                    event["item"]["type"].as_str().unwrap().to_string(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            added,
            [
                (0, "message".to_string()),
                (1, "function_call".to_string()),
                (2, "function_call".to_string()),
            ]
        );

        let completed = events_of(&events, "response.completed");
        let response = &completed[0]["response"];
        assert_eq!(response["output_text"], "Reading both.");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[1]["call_id"], "call_a");
        assert_eq!(output[2]["call_id"], "call_b");
        assert_eq!(output[2]["arguments"], r#"{"path":"b"}"#);
    }

    /// A reasoning model that thinks and then calls a tool, without text:
    /// its reasoning streams under the message item, so that message is
    /// announced first, keeps the position the reasoning named and stays in
    /// the output before the call.
    #[tokio::test]
    async fn translated_stream_keeps_reasoning_on_its_message_before_a_function_call() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Need the file."},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_r","type":"function","function":{"name":"read","arguments":"{\"path\":\"r\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        let reasoning = events_of(&events, "response.reasoning_text.delta");
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0]["output_index"], 0);
        let message_id = reasoning[0]["item_id"].as_str().unwrap();

        let added = events_of(&events, "response.output_item.added")
            .iter()
            .map(|event| {
                (
                    event["output_index"].as_i64().unwrap(),
                    event["item"]["type"].as_str().unwrap().to_string(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            added,
            [(0, "message".to_string()), (1, "function_call".to_string())],
            "items are announced in output order: {body}"
        );
        let first_added = events
            .iter()
            .position(|event| event["type"] == "response.output_item.added")
            .unwrap();
        let first_reasoning = events
            .iter()
            .position(|event| event["type"] == "response.reasoning_text.delta")
            .unwrap();
        assert!(first_added < first_reasoning, "{body}");

        let done = events_of(&events, "response.output_item.done")
            .iter()
            .map(|event| {
                (
                    event["output_index"].as_i64().unwrap(),
                    event["item"]["id"].as_str().unwrap().to_string(),
                    event["item"]["type"].as_str().unwrap().to_string(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(done.len(), 2, "{body}");
        assert_eq!(done[0], (0, message_id.to_string(), "message".to_string()));
        assert_eq!(done[1].0, 1);
        assert_eq!(done[1].2, "function_call");

        let completed = events_of(&events, "response.completed");
        let output = completed[0]["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["id"], message_id);
        assert_eq!(output[1]["call_id"], "call_r");
    }

    /// Two calls whose argument fragments interleave, with indices that are
    /// not contiguous, each keep their own item and arguments.
    #[tokio::test]
    async fn translated_stream_keeps_interleaved_function_calls_apart() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"read","arguments":"{\"path\":"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"call_y","type":"function","function":{"name":"list","arguments":"{\"dir\":"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"function":{"arguments":"\"y\"}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        let added = events_of(&events, "response.output_item.added");
        assert_eq!(added.len(), 2);
        for (position, (call_id, arguments)) in
            [("call_x", r#"{"path":"x"}"#), ("call_y", r#"{"dir":"y"}"#)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(added[position]["output_index"], position);
            assert_eq!(added[position]["item"]["call_id"], call_id);
            let item_id = &added[position]["item"]["id"];
            let streamed = events_of(&events, "response.function_call_arguments.delta")
                .iter()
                .filter(|event| &event["item_id"] == item_id)
                .map(|event| event["delta"].as_str().unwrap().to_string())
                .collect::<String>();
            assert_eq!(streamed, arguments, "{body}");
        }

        let completed = events_of(&events, "response.completed");
        let output = completed[0]["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["arguments"], r#"{"path":"x"}"#);
        assert_eq!(output[1]["name"], "list");
        assert_eq!(output[1]["arguments"], r#"{"dir":"y"}"#);
    }

    /// The token limit cuts a call in the middle of its arguments: the call
    /// is never closed, so a client cannot run it with truncated arguments,
    /// and the response ends `incomplete` for `max_output_tokens`.
    #[tokio::test]
    async fn translated_stream_never_closes_a_function_call_cut_by_the_token_limit() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_cut","type":"function","function":{"name":"write","arguments":"{\"path\":\"a\",\"text\":\"hel"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        assert_eq!(events_of(&events, "response.output_item.added").len(), 1);
        assert!(events_of(&events, "response.function_call_arguments.done").is_empty());
        assert!(events_of(&events, "response.output_item.done").is_empty());
        assert!(events_of(&events, "response.completed").is_empty());
        let incomplete = events_of(&events, "response.incomplete");
        assert_eq!(incomplete.len(), 1, "{body}");
        let response = &incomplete[0]["response"];
        assert_eq!(response["status"], "incomplete");
        assert!(response["completed_at"].is_null(), "{body}");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["status"], "incomplete");
        assert_eq!(output[0]["arguments"], r#"{"path":"a","text":"hel"#);
    }

    /// A call without arguments closes with an empty JSON object, which
    /// clients parse and send back, instead of an empty string.
    #[tokio::test]
    async fn translated_stream_closes_a_call_without_arguments_with_an_empty_object() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_now","type":"function","function":{"name":"now","arguments":""}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        let done = events_of(&events, "response.function_call_arguments.done");
        assert_eq!(done.len(), 1, "{body}");
        assert_eq!(done[0]["arguments"], "{}");
        let item_done = events_of(&events, "response.output_item.done");
        assert_eq!(item_done[0]["item"]["arguments"], "{}");
        let completed = events_of(&events, "response.completed");
        assert_eq!(completed[0]["response"]["output"][0]["arguments"], "{}");
    }

    /// An upstream error in the middle of a call's arguments relays the
    /// error and never closes the call.
    #[tokio::test]
    async fn translated_stream_never_closes_a_function_call_cut_by_an_error() {
        let (result, body) = relay_translated_over(
            concat!(
                r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_err","type":"function","function":{"name":"read","arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"error":{"message":"deadline","type":"server_error","param":null,"code":"timeout"}}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes(),
        )
        .await;
        assert!(matches!(result, Ok(RouteAttemptResult::Delivered { .. })));
        let events = sse_json_events(&body);

        assert_eq!(events_of(&events, "response.output_item.added").len(), 1);
        assert!(events_of(&events, "response.output_item.done").is_empty());
        assert!(events_of(&events, "response.completed").is_empty());
        assert!(
            events
                .iter()
                .any(|event| event["error"]["message"] == "deadline"),
            "the upstream error reaches the client: {body}"
        );
    }
}
