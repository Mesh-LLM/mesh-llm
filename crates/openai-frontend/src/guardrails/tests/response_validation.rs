use super::*;

#[test]
fn strips_thinking_blocks_before_rescue_attempts() {
    assert_eq!(
        strip_thinking_blocks(
            "<think>private plan</think>```json\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Sydney\"}}\n```"
        ),
        "```json\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Sydney\"}}\n```"
    );
    assert_eq!(
        strip_thinking_blocks("[THINK]hidden[/THINK]Visible answer"),
        "Visible answer"
    );
}

#[test]
fn rescues_plain_json_tool_call_text() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"{"name":"lookup","arguments":{"city":"Sydney"}}"#,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonExact);
    assert_eq!(classified.visible_content, None);
    assert_eq!(tool_call_name(&classified), Some("lookup"));
    assert_eq!(
        tool_call_arguments(&classified),
        Some(r#"{"city":"Sydney"}"#)
    );
}

#[test]
fn rescues_json_tool_call_array_text() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"[{"type":"function","function":{"name":"lookup","arguments":{"city":"Sydney"}}}]"#,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(tool_call_name(&classified), Some("lookup"));
}

#[test]
fn rescues_fenced_json_tool_call_text() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        "```json\n{\"name\":\"lookup\",\"arguments\":{\"city\":\"Sydney\"}}\n```",
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonFenced);
}

#[test]
fn rescues_brace_balanced_json_substring_only_for_allowed_tools() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        "I'll call this now: {\"name\":\"lookup\",\"arguments\":{\"city\":\"Sydney\"}}",
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(tool_call_name(&classified), Some("lookup"));
}

#[test]
fn arbitrary_json_without_allowed_tool_name_is_not_rescued() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", r#"{"payload":{"city":"Sydney"}}"#);

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::MalformedToolText
    );
    assert!(classified.tool_calls.is_none());
}

#[test]
fn rescues_bracket_args_tool_syntax() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", "lookup[ARGS]{\"city\":\"Sydney\"}");

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonSubstring);
    assert_eq!(tool_call_name(&classified), Some("lookup"));
}

#[test]
fn rescues_qwen_xml_tool_syntax() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        "<function=lookup><parameter=city>Sydney</parameter></function>",
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonSubstring);
    assert_eq!(tool_call_name(&classified), Some("lookup"));
    assert_eq!(
        tool_call_arguments(&classified),
        Some(r#"{"city":"Sydney"}"#)
    );
}

#[test]
fn rescues_granite_tool_call_syntax() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"Sydney\"}}</tool_call>",
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(tool_call_name(&classified), Some("lookup"));
}

#[test]
fn rescue_strips_hidden_reasoning_from_client_visible_content() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        "<think>private reasoning</think>Hello there",
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::ValidText);
    assert_eq!(classified.visible_content.as_deref(), Some("Hello there"));
}

#[test]
fn unknown_tool_text_classifies_for_retry() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"{"name":"other_tool","arguments":{"city":"Sydney"}}"#,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::UnknownTool);
    assert!(classified.tool_calls.is_none());
}

#[test]
fn malformed_arguments_classify_without_panicking() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"{"name":"lookup","arguments":"not-json"}"#,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::InvalidToolArguments
    );
}

#[test]
fn existing_valid_tool_calls_are_classified() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{\"city\":\"Sydney\"}"
            }
        }]),
        None,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidToolCalls
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::None);
}

#[test]
fn synthetic_respond_classifies_and_extracts_message() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "type": "function",
            "function": {
                "name": MESH_RESPOND_TOOL_NAME,
                "arguments": "{\"message\":\"Hello there\"}"
            }
        }]),
        None,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidSyntheticRespond
    );
    assert_eq!(classified.synthetic_text.as_deref(), Some("Hello there"));
}

#[test]
fn synthetic_structured_classifies_when_allowed() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "json"}],
            "response_format": supported_json_schema_response_format()
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "type": "function",
            "function": {
                "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                "arguments": "{\"answer\":42}"
            }
        }]),
        None,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidSyntheticStructured
    );
    assert_eq!(classified.structured_payload, Some(json!({"answer": 42})));
}

#[test]
fn direct_structured_json_object_classifies_when_allowed() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "json"}],
            "response_format": supported_json_schema_response_format()
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", r#"{"answer":42}"#);

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidSyntheticStructured
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonExact);
    assert_eq!(classified.structured_payload, Some(json!({"answer": 42})));
}

#[test]
fn fenced_direct_structured_json_object_classifies_when_allowed() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "json"}],
            "response_format": supported_json_schema_response_format()
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", "```json\n{\"answer\":42}\n```");

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ValidSyntheticStructured
    );
    assert_eq!(classified.parser_stage, GuardrailParserStage::JsonFenced);
    assert_eq!(classified.structured_payload, Some(json!({"answer": 42})));
}

#[test]
fn invalid_structured_payload_classifies_without_leaking_arguments() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "json"}],
            "response_format": supported_json_schema_response_format()
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "type": "function",
            "function": {
                "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                "arguments": "bad-json"
            }
        }]),
        None,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::InvalidStructuredPayload
    );
    assert!(classified.structured_payload.is_none());
}

#[test]
fn mixed_terminal_and_tool_is_detected() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{\"city\":\"Sydney\"}"
            }
        }]),
        Some("Done"),
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::MixedTerminalAndTool
    );
}

#[test]
fn empty_output_is_classified() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_text_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", "<think>only hidden</think>");

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::EmptyOutput);
}

#[test]
fn text_after_tool_result_is_valid_not_malformed() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [
                {"role": "user", "content": "What's 5+5?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"a\":5,\"b\":5,\"op\":\"add\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "10"}
            ],
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "tool_choice": "auto"
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", "The answer is 10!");

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::ValidText);
    assert_eq!(
        classified.visible_content.as_deref(),
        Some("The answer is 10!")
    );
}

#[test]
fn tool_result_request_bypasses_guarded_mode() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [
                {"role": "user", "content": "What's 5+5?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"a\":5,\"b\":5,\"op\":\"add\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "10"}
            ],
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "tool_choice": "auto"
        }),
    );

    assert!(
        matches!(
            prepared.outcome,
            super::state::GuardrailRequestOutcome::PassThrough {
                reason: super::telemetry::GuardrailTelemetryBypassReason::AfterToolResult
            }
        ),
        "expected PassThrough with AfterToolResult reason, got {:?}",
        prepared.outcome
    );
    assert!(prepared.state.last_message_is_tool_result);
}

#[tokio::test]
async fn tool_result_enforce_mode_returns_text_successfully() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(response_with_content(
        "Qwen3-8B-Q4_K_M",
        "The answer is 10!",
    ))]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [
            {"role": "user", "content": "What's 5+5?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"a\":5,\"b\":5,\"op\":\"add\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "10"}
        ],
        "tools": [{"type": "function", "function": {"name": "calculator"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("The answer is 10!")
    );
    assert!(response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.choices[0].finish_reason,
        Some(crate::common::FinishReason::Stop)
    );
    // Verify original request passed through without retries
    assert_eq!(backend.chat_requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_tool_arguments_retry_once_then_succeed() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(response_with_content(
            "Qwen3-8B-Q4_K_M",
            r#"{"name":"lookup","arguments":"bad-json"}"#,
        )),
        Ok(response_with_tool_calls_with_usage(
            "Qwen3-8B-Q4_K_M",
            json!([{"id":"call_ok","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
            None,
            Usage::new(7, 3),
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 1,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "prompt_cache_key": "cache-1"
    }))
    .unwrap();

    let response = guarded.chat_completion(request.clone()).await.unwrap();

    assert_eq!(tool_call_name_from_response(&response), Some("lookup"));
    assert_eq!(response.usage, Usage::new(7, 3));
    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].prompt_cache_key.as_deref(), Some("cache-1"));
    assert_eq!(requests[1].prompt_cache_key, None);
    assert!(
        request.messages[0]
            .content
            .as_ref()
            .is_some_and(|content| content == &MessageContent::Text("weather".to_string()))
    );
    let retry_text = crate::chat::message_content_to_text(
        requests[1].messages[0]
            .content
            .as_ref()
            .expect("retry content exists"),
    )
    .expect("retry text exists");
    assert!(retry_text.contains("invalid JSON tool arguments"));
    assert!(retry_text.contains("Do not add extra text."));
}

#[tokio::test]
async fn retry_exhaustion_returns_openai_error() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(response_with_content(
            "Qwen3-8B-Q4_K_M",
            r#"{"name":"lookup","arguments":"bad-json"}"#,
        )),
        Ok(response_with_content(
            "Qwen3-8B-Q4_K_M",
            r#"{"name":"lookup","arguments":"still-bad"}"#,
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 1,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();

    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(body.error.message, GUARDRAIL_VALIDATION_FAILED_MESSAGE);
}

#[tokio::test]
async fn pass_last_text_exhaustion_returns_safe_final_text() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(
        response_with_tool_calls_with_usage(
            "Qwen3-8B-Q4_K_M",
            json!([{"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
            Some("Fallback assistant text"),
            Usage::new(5, 4),
        ),
    )]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 0,
            retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Fallback assistant text")
    );
    assert!(response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.choices[0].finish_reason,
        Some(crate::common::FinishReason::Stop)
    );
    assert_eq!(response.usage, Usage::new(5, 4));
}

#[tokio::test]
async fn pass_last_text_rejects_mixed_synthetic_and_real_exhausted_output() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(
        response_with_tool_calls_with_usage(
            "Qwen3-8B-Q4_K_M",
            json!([
                {"type":"function","function":{"name":"_mesh_respond","arguments":"{\"message\":\"done\"}"}},
                {"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}
            ]),
            None,
            Usage::new(5, 4),
        ),
    )]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 0,
            retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();

    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(body.error.message, GUARDRAIL_VALIDATION_FAILED_MESSAGE);
}

#[tokio::test]
async fn pass_last_text_rejects_sentinel_leaking_text_without_safe_fallback() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(
        response_with_content_with_usage(
            "Qwen3-8B-Q4_K_M",
            "I will call _mesh_respond next",
            Usage::new(4, 3),
        ),
    )]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 0,
            retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();

    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(body.error.message, GUARDRAIL_VALIDATION_FAILED_MESSAGE);
}

#[tokio::test]
async fn mesh_respond_stripped_to_assistant_text() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"_mesh_respond({"message":"Hello there"})"#,
    ))]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Hello there")
    );
    assert!(response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.choices[0].finish_reason,
        Some(crate::common::FinishReason::Stop)
    );
}

#[tokio::test]
async fn mixed_mesh_respond_plus_real_tool_calls_retry_exhaustion_handling() {
    let invalid = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([
            {"type":"function","function":{"name":"_mesh_respond","arguments":"{\"message\":\"done\"}"}},
            {"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}
        ]),
        None,
    );
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(invalid.clone()),
        Ok(invalid),
    ]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 1,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(backend.chat_requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn final_visible_usage_equals_final_attempt_usage_only() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(response_with_content_with_usage(
            "Qwen3-8B-Q4_K_M",
            r#"{"name":"lookup","arguments":"bad-json"}"#,
            Usage::new(40, 10),
        )),
        Ok(response_with_tool_calls_with_usage(
            "Qwen3-8B-Q4_K_M",
            json!([{"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
            None,
            Usage::new(3, 2),
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_tool_retries: 1,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(response.usage, Usage::new(3, 2));
}

#[tokio::test]
async fn no_mesh_tool_survives_responses_function_call_conversion() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":"_mesh_emit_structured","arguments":"{\"answer\":42}"}}]),
        None,
    ))]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();
    let translated = translate_chat_completion_to_responses(
        serde_json::to_string(&response).unwrap().as_bytes(),
    )
    .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&translated).unwrap();

    assert_eq!(parsed["output_text"], "{\"answer\":42}");
    assert!(
        parsed["output"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["type"] != "function_call")
    );
}

#[tokio::test]
async fn structured_response_format_rewrites_to_synthetic_tool() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let _ = guarded.chat_completion(request.clone()).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert!(seen.response_format.is_none());
    assert_eq!(
        request.response_format,
        Some(supported_json_schema_response_format())
    );
    let structured_tool = seen
        .tools
        .as_ref()
        .and_then(|tools| tools.as_array())
        .and_then(|entries| {
            entries.iter().find(|entry| {
                entry["function"]["name"].as_str() == Some(MESH_EMIT_STRUCTURED_TOOL_NAME)
            })
        })
        .cloned()
        .expect("synthetic structured tool injected");
    assert_eq!(
        structured_tool["function"]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "answer": {"type": "integer"}
            },
            "required": ["answer"],
            "additionalProperties": false
        })
    );
}

#[tokio::test]
async fn valid_structured_payload_becomes_json_assistant_text() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{
            "type":"function",
            "function": {
                "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                "arguments": "{\"answer\":42}"
            }
        }]),
        None,
    ))]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":42}")
    );
    assert!(response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.choices[0].finish_reason,
        Some(crate::common::FinishReason::Stop)
    );
}

#[tokio::test]
async fn direct_structured_payload_becomes_json_assistant_text() {
    let backend = Arc::new(SequencedBackend::new(vec![Ok(response_with_content(
        "Qwen3-8B-Q4_K_M",
        "```json\n{\"answer\":42}\n```",
    ))]));
    let guarded = GuardedOpenAiBackend::new(
        backend,
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":42}")
    );
    assert!(response.choices[0].message.tool_calls.is_none());
    assert_eq!(
        response.choices[0].finish_reason,
        Some(crate::common::FinishReason::Stop)
    );
}

#[tokio::test]
async fn invalid_structured_payload_retries_then_exhaustion_error() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(response_with_tool_calls(
            "Qwen3-8B-Q4_K_M",
            json!([{
                "type":"function",
                "function": {
                    "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                    "arguments": "{\"answer\":\"wrong\"}"
                }
            }]),
            None,
        )),
        Ok(response_with_tool_calls(
            "Qwen3-8B-Q4_K_M",
            json!([{
                "type":"function",
                "function": {
                    "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                    "arguments": "{\"answer\":\"still wrong\"}"
                }
            }]),
            None,
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            max_structured_retries: 1,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format(),
        "prompt_cache_key": "structured-cache"
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(backend.chat_requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn unsupported_schema_feature_behavior_is_explicit_and_asserted() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}]
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_CODE)
    );
    assert_eq!(
        error.body().error.message,
        GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_MESSAGE
    );
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn metrics_only_failed_validation_returns_original_backend_response() {
    let original = response_with_content(
        "Qwen3-8B-Q4_K_M",
        r#"{"name":"lookup","arguments":"bad-json"}"#,
    );
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let backend = Arc::new(SequencedBackend::new(vec![Ok(original.clone())]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::MetricsOnly,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let passthrough_request = request.clone();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        backend.chat_requests.lock().unwrap().as_slice(),
        &[passthrough_request]
    );
    assert_eq!(response, original);
    assert!(telemetry.outcomes.lock().unwrap().iter().any(|record| {
        record.outcome == GuardrailTelemetryOutcome::MetricsOnlyFailure.as_str()
            && record.parser_stage == Some(GuardrailTelemetryParserStage::JsonExact.as_str())
    }));
}

#[tokio::test]
async fn metrics_only_eligible_tool_request_does_not_rewrite_or_sanitize() {
    let original = response_with_content("Qwen3-8B-Q4_K_M", "plain assistant text");
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let backend = Arc::new(SequencedBackend::new(vec![Ok(original.clone())]));
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::MetricsOnly,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();
    let passthrough_request = request.clone();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        backend.chat_requests.lock().unwrap().as_slice(),
        &[passthrough_request]
    );
    assert_eq!(response, original);
    assert!(telemetry.decisions.lock().unwrap().iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Eligible.as_str()
            && record.contract == Some(GuardrailTelemetryContract::Tools.as_str())
            && record.bypass_reason.is_none()
    }));
    assert!(telemetry.outcomes.lock().unwrap().iter().any(|record| {
        record.outcome == GuardrailTelemetryOutcome::MetricsOnlyFailure.as_str()
            && record.parser_stage == Some(GuardrailTelemetryParserStage::None.as_str())
    }));
}
