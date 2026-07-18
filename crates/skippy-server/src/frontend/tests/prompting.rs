use super::support::*;
use super::*;

#[test]
fn chat_runtime_feature_guard_allows_tool_calls() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto"
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "parallel_tool_calls": true
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        ensure_chat_runtime_features_supported(&request).unwrap();
    }
}

#[test]
fn plain_chat_does_not_require_chat_output_parser() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();

    assert!(!chat_output_parser_required(
        &request,
        &ChatTemplateOptions::default(),
    ));
}

#[test]
fn hidden_reasoning_format_requires_chat_output_parser_for_plain_chat() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Hidden),
        ..ChatTemplateOptions::default()
    };

    assert!(chat_output_parser_required(&request, &template_options));
}

#[test]
fn reasoning_format_none_leaves_plain_chat_unparsed() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::None),
        ..ChatTemplateOptions::default()
    };

    assert!(!chat_output_parser_required(&request, &template_options));
}

#[test]
fn tools_require_chat_output_parser() {
    assert!(chat_output_parser_required(
        &tool_request(),
        &ChatTemplateOptions::default(),
    ));
}

#[test]
fn parses_llama_message_tool_calls() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");

    assert_eq!(parsed.content, None);
    assert_eq!(parsed.tool_calls[0]["id"], "call_123");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn parses_llama_message_reasoning_content() {
    let parsed = parsed_chat_message_from_json(
        r#"{"role":"assistant","reasoning_content":"Checked facts first.","content":"Final answer."}"#,
        &ChatCompletionRequest {
            tools: None,
            ..tool_request()
        },
    )
    .expect("parsed message");

    assert_eq!(parsed.content.as_deref(), Some("Final answer."));
    assert_eq!(
        parsed.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
    assert_eq!(parsed.tool_calls, None);
}

#[test]
fn chat_response_from_parsed_message_separates_reasoning_content() {
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 7,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats {
            drafted_tokens: 7,
            accepted_tokens: 5,
            rejected_tokens: 2,
            verification_count: 7,
            proposal_compute_us: 100,
            verification_compute_us: 200,
            ..NativeMtpStats::default()
        },
        native_mtp_decode_telemetry: None,
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "Checked facts first.</think>Final answer.".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };

    let response = chat_response_from_generated_text("qwen".to_string(), &output, Some(parsed));

    let message = &response.choices[0].message;
    assert_eq!(message.content.as_deref(), Some("Final answer."));
    let timings = response.timings.as_ref().expect("native MTP timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(7)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(5)));
    assert_eq!(
        message.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
    assert_eq!(message.tool_calls, None);
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));

    let completion = completion_response_from_generated_text("qwen".to_string(), &output);
    let timings = completion
        .timings
        .as_ref()
        .expect("completion native MTP timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(7)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(5)));
    assert_eq!(timings.get("predicted_per_second"), Some(&json!(70.0)));
}

#[test]
fn generated_text_timings_are_present_without_native_mtp() {
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 8,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats::default(),
        native_mtp_decode_telemetry: None,
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "Paris".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };

    let timings = output.timings().expect("standard timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(0)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(0)));
    assert_eq!(timings.get("prompt_per_second"), Some(&json!(200.0)));
    assert_eq!(timings.get("predicted_per_second"), Some(&json!(80.0)));
}

#[test]
fn generated_text_timings_prefer_composite_proposal_totals() {
    let mut counters = NativeMtpDecodeCounters::default();
    let options = NativeMtpDecodeOptions {
        max_draft_tokens: 1,
        min_draft_tokens: 0,
        reject_cooldown_tokens: 0,
        suppress_cooldown_drafts: false,
        suppress_cooldown_draft_limit: 0,
        ngram_hybrid: true,
        ngram_size: 2,
        ngram_initial_extension_tokens: 2,
        ngram_max_proposal_tokens: 4,
        ngram_tail_backoff_proposals: 2,
        verify_window_min_tokens: 1,
        verify_window_max_tokens: 4,
    };
    let proposal = CompositeProposalProvider::from_options(options).propose(
        &[],
        &[0, 0, 2, 3, 9, 1, 7, 8, 2, 3],
        4,
    );
    counters.observe_hybrid_proposal(&proposal, 4);
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 8,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats::default(),
        native_mtp_decode_telemetry: Some(NativeMtpDecodeTelemetry::new(options, counters)),
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "ok".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };

    let timings = output.timings().expect("timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(4)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(4)));
}

#[test]
fn hidden_reasoning_visibility_removes_reasoning_content() {
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Hidden),
        ..ChatTemplateOptions::default()
    };

    let visible = apply_reasoning_visibility(Some(parsed), &template_options)
        .expect("parsed message should remain");

    assert_eq!(visible.content.as_deref(), Some("Final answer."));
    assert_eq!(visible.reasoning_content, None);
}

#[test]
fn auto_reasoning_visibility_keeps_reasoning_content() {
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Auto),
        ..ChatTemplateOptions::default()
    };

    let visible = apply_reasoning_visibility(Some(parsed), &template_options)
        .expect("parsed message should remain");

    assert_eq!(
        visible.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
}

#[test]
fn generation_event_to_chat_chunk_emits_reasoning_delta() {
    let chunk = generation_event_to_chat_chunk(
        Ok(GenerationStreamEvent::ReasoningDelta(
            "Checking the premise.".to_string(),
        )),
        "qwen",
    )
    .unwrap();

    let delta = &chunk.choices[0].delta;
    assert_eq!(delta.content, None);
    assert_eq!(
        delta.reasoning_content.as_deref(),
        Some("Checking the premise.")
    );
    assert_eq!(delta.tool_calls, None);
}

#[test]
fn llama_message_tool_parser_rejects_unknown_tool() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"shell","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());
}

#[test]
fn tool_choice_limits_allowed_tool_name() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "search"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .unwrap();

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());

    request.tool_choice = Some(json!("search"));
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request,
    )
    .expect("selected tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "search");
}

#[test]
fn parallel_tool_calls_false_keeps_first_call() {
    let mut request = tool_request();
    request.parallel_tool_calls = Some(false);
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}},
            {"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Melbourne\"}"}}
        ]}"#,
        &request,
    )
    .expect("tool calls");

    assert_eq!(parsed.tool_calls.as_array().unwrap().len(), 1);
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn emulated_output_final_parses_tool_call() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output(
        "Let me check.\nTOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Sydney\"}}",
        &request,
        false,
    )
    .expect("emulated parse");

    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
    let calls = parsed.tool_calls.expect("tool calls");
    assert_eq!(calls[0]["function"]["name"], "lookup");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Sydney");
}

#[test]
fn emulated_output_partial_withholds_tool_calls_and_incomplete_line() {
    let request = tool_request();
    // Streaming: the TOOL_CALL line is not yet newline-terminated.
    let parsed = parse_emulated_chat_output(
        "Let me check.\nTOOL_CALL {\"name\": \"lookup\", \"argumen",
        &request,
        true,
    )
    .expect("emulated parse");

    // Tool calls are withheld while streaming.
    assert!(parsed.tool_calls.is_none());
    // Only the completed prose line is exposed; the partial marker line is held.
    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
}

#[test]
fn emulated_output_partial_streams_single_line_prose() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output("The capital is Canber", &request, true)
        .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The capital is Canber"));
}

#[test]
fn emulated_output_partial_withholds_possible_marker_suffix() {
    let request = tool_request();
    let parsed =
        parse_emulated_chat_output("Let me check. TOOL_", &request, true).expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
}

#[test]
fn emulated_output_partial_ignores_marker_inside_completed_thinking() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output(
        "<think>TOOL_CALL {\"name\":\"lookup\"}</think>The answer is Syd",
        &request,
        true,
    )
    .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The answer is Syd"));
}

#[test]
fn emulated_output_respects_allowed_tool_names() {
    let request = tool_request();
    // "search" is not an allowed tool for this request (only "lookup" is).
    let parsed = parse_emulated_chat_output(
        "TOOL_CALL {\"name\": \"search\", \"arguments\": {}}",
        &request,
        false,
    )
    .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
}

#[test]
fn emulated_output_parallel_false_keeps_first_call() {
    let mut request = tool_request();
    request.parallel_tool_calls = Some(false);
    let parsed = parse_emulated_chat_output(
        "TOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Sydney\"}}\n\
         TOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Melbourne\"}}",
        &request,
        false,
    )
    .expect("emulated parse");

    let calls = parsed.tool_calls.expect("tool calls");
    assert_eq!(calls.as_array().unwrap().len(), 1);
}

#[test]
fn emulated_output_plain_prose_has_no_tool_calls() {
    let request = tool_request();
    let parsed =
        parse_emulated_chat_output("The capital is Canberra.", &request, false).expect("parse");
    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The capital is Canberra."));
}

#[test]
fn tool_call_stream_delta_adds_indexes() {
    let delta = tool_calls_stream_delta(json!([
        {"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{}"}},
        {"id":"call_b","type":"function","function":{"name":"lookup","arguments":"{}"}}
    ]));

    assert_eq!(delta[0]["index"], 0);
    assert_eq!(delta[1]["index"], 1);
}

#[test]
fn chat_message_generation_value_preserves_tool_history() {
    let message: openai_frontend::ChatMessage = serde_json::from_value(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"city\":\"Sydney\"}"}
        }]
    }))
    .unwrap();
    let mut media = Vec::new();

    let value = chat_message_generation_value(&message, "<__media__>", &mut media).unwrap();

    assert_eq!(value["content"], Value::Null);
    assert_eq!(value["tool_calls"][0]["id"], "call_123");
    assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
}
