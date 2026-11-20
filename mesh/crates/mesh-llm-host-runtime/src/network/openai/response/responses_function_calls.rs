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

        let reasoning = events_of(&events, "response.reasoning.delta");
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
            [
                (0, "reasoning".to_string()),
                (1, "function_call".to_string())
            ],
            "items are announced in output order: {body}"
        );
        let first_added = events
            .iter()
            .position(|event| event["type"] == "response.output_item.added")
            .unwrap();
        let first_reasoning = events
            .iter()
            .position(|event| event["type"] == "response.reasoning.delta")
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
        assert_eq!(
            done[0],
            (0, message_id.to_string(), "reasoning".to_string())
        );
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

    /// A token cutoff closes the item lifecycle with explicit incomplete
    /// status, preserving partial arguments without reporting completion.
    #[tokio::test]
    async fn translated_stream_marks_a_function_call_cut_by_the_token_limit_incomplete() {
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
        let done = events_of(&events, "response.function_call_arguments.done");
        assert_eq!(done.len(), 1);
        assert_eq!(done[0]["arguments"], r#"{"path":"a","text":"hel"#);
        let item_done = events_of(&events, "response.output_item.done");
        assert_eq!(item_done.len(), 1);
        assert_eq!(item_done[0]["item"]["status"], "incomplete");
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
        assert!(matches!(
            result,
            Ok(RouteAttemptResult::CommittedStreamFailure { .. })
        ));
        let events = sse_json_events(&body);

        assert!(events_of(&events, "response.output_item.added").is_empty());
        assert!(events_of(&events, "response.function_call_arguments.delta").is_empty());
        assert!(events_of(&events, "response.function_call_arguments.done").is_empty());
        assert_eq!(events_of(&events, "response.failed").len(), 1);
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
