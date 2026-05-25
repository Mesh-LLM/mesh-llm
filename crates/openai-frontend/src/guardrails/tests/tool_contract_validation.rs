use super::*;

#[test]
fn tool_choice_none_accepts_plain_text_despite_tools() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "none"
        }),
    );
    let response = response_with_content("Qwen3-8B-Q4_K_M", "No tool is needed.");

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::ValidText);
    assert_eq!(
        classified.visible_content.as_deref(),
        Some("No tool is needed.")
    );
}

#[test]
fn tool_choice_none_rejects_tool_calls() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(
        &engine,
        json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "none"
        }),
    );
    let response = response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
        None,
    );

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::ToolCallsNotAllowed
    );
    assert_eq!(tool_call_name(&classified), Some("lookup"));
}

#[test]
fn forced_tool_choice_rejects_different_allowed_tool() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(&engine, request_forced_lookup_tool_choice());
    let response = search_tool_response();

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(classified.category, GuardrailResponseCategory::UnknownTool);
    assert_eq!(tool_call_name(&classified), Some("search"));
}

#[test]
fn parallel_tool_calls_false_rejects_multiple_real_tool_calls() {
    let engine = GuardrailEngine::new(enforce_policy());
    let prepared = prepared_tool_request(&engine, request_parallel_tool_calls_false());
    let response = multiple_tool_calls_response();

    let classified = engine.classify_response(&prepared, &response);

    assert_eq!(
        classified.category,
        GuardrailResponseCategory::TooManyToolCalls
    );
    assert_eq!(classified_tool_call_count(&classified), Some(2));
}

#[tokio::test]
async fn tool_choice_none_retries_tool_call_then_accepts_text() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(lookup_tool_response()),
        Ok(response_with_content(
            "Qwen3-8B-Q4_K_M",
            "No tool is needed.",
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(backend.clone(), one_retry_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "answer without tools"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "none"
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("No tool is needed.")
    );
    assert!(response.choices[0].message.tool_calls.is_none());

    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tool_choice.as_ref(), Some(&json!("none")));
    assert!(!request_tool_names(&requests[0]).contains(&MESH_RESPOND_TOOL_NAME));
    assert!(!request_tool_names(&requests[1]).contains(&MESH_RESPOND_TOOL_NAME));
    assert!(retry_text(&requests[1]).contains("do not make a tool call"));
}

#[tokio::test]
async fn tool_choice_none_unknown_tool_retry_is_text_only() {
    assert_tool_choice_none_retry_text_only(search_tool_response()).await;
}

#[tokio::test]
async fn tool_choice_none_malformed_tool_retry_is_text_only() {
    assert_tool_choice_none_retry_text_only(response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"arguments":"{\"city\":\"Sydney\"}"}}]),
        None,
    ))
    .await;
}

#[tokio::test]
async fn tool_choice_none_mixed_tool_response_retry_is_text_only() {
    assert_tool_choice_none_retry_text_only(response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
        Some("I will call the tool."),
    ))
    .await;
}

#[tokio::test]
async fn structured_output_with_tool_choice_none_still_enforces_schema() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(response_with_content("Qwen3-8B-Q4_K_M", "plain text")),
        Ok(synthetic_structured_response()),
    ]));
    let guarded = GuardedOpenAiBackend::new(backend.clone(), one_retry_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format(),
        "tool_choice": "none"
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":42}")
    );
    assert!(response.choices[0].message.tool_calls.is_none());

    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tool_choice, None);
    assert_eq!(requests[1].tool_choice, None);
    assert!(request_tool_names(&requests[0]).contains(&MESH_EMIT_STRUCTURED_TOOL_NAME));
    assert!(retry_text(&requests[1]).contains("structured-output tool call"));
}

#[tokio::test]
async fn forced_tool_mismatch_retries_with_original_forced_choice() {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(search_tool_response()),
        Ok(lookup_tool_response()),
    ]));
    let guarded = GuardedOpenAiBackend::new(backend.clone(), one_retry_policy());
    let request: ChatCompletionRequest =
        serde_json::from_value(request_forced_lookup_tool_choice()).unwrap();
    let expected_tool_choice = request.tool_choice.clone();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(tool_call_name_from_response(&response), Some("lookup"));
    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tool_choice, expected_tool_choice);
    assert_eq!(requests[1].tool_choice, expected_tool_choice);
    assert!(!request_tool_names(&requests[1]).contains(&MESH_RESPOND_TOOL_NAME));
    assert!(retry_text(&requests[1]).contains("tool name that was not allowed"));
}

#[tokio::test]
async fn parallel_tool_calls_false_retries_then_exhausts_multiple_calls() {
    let invalid = multiple_tool_calls_response();
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(invalid.clone()),
        Ok(invalid),
    ]));
    let guarded = GuardedOpenAiBackend::new(backend.clone(), one_retry_policy());
    let request: ChatCompletionRequest =
        serde_json::from_value(request_parallel_tool_calls_false()).unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].parallel_tool_calls, Some(false));
    assert_eq!(requests[1].parallel_tool_calls, Some(false));
    assert!(retry_text(&requests[1]).contains("more tool calls than allowed"));
}

fn one_retry_policy() -> GuardrailPolicy {
    GuardrailPolicy {
        mode: GuardrailMode::Enforce,
        apply_to_all_models: true,
        max_tool_retries: 1,
        ..GuardrailPolicy::default()
    }
}

fn request_forced_lookup_tool_choice() -> serde_json::Value {
    json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "search"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    })
}

fn request_parallel_tool_calls_false() -> serde_json::Value {
    json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "search"}}
        ],
        "parallel_tool_calls": false
    })
}

fn lookup_tool_response() -> ChatCompletionResponse {
    response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]),
        None,
    )
}

fn search_tool_response() -> ChatCompletionResponse {
    response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":"search","arguments":"{\"query\":\"Sydney weather\"}"}}]),
        None,
    )
}

fn multiple_tool_calls_response() -> ChatCompletionResponse {
    response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([
            {"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}},
            {"type":"function","function":{"name":"search","arguments":"{\"query\":\"Sydney weather\"}"}}
        ]),
        None,
    )
}

fn synthetic_structured_response() -> ChatCompletionResponse {
    response_with_tool_calls(
        "Qwen3-8B-Q4_K_M",
        json!([{"type":"function","function":{"name":MESH_EMIT_STRUCTURED_TOOL_NAME,"arguments":"{\"answer\":42}"}}]),
        None,
    )
}

async fn assert_tool_choice_none_retry_text_only(first_response: ChatCompletionResponse) {
    let backend = Arc::new(SequencedBackend::new(vec![
        Ok(first_response),
        Ok(response_with_content(
            "Qwen3-8B-Q4_K_M",
            "No tool is needed.",
        )),
    ]));
    let guarded = GuardedOpenAiBackend::new(backend.clone(), one_retry_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "answer without tools"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "none"
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("No tool is needed.")
    );
    let requests = backend.chat_requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let retry = retry_text(&requests[1]);
    assert!(retry.contains("do not make a tool call"));
    assert!(!retry.contains("exactly one valid tool call"));
}

fn classified_tool_call_count(classified: &ClassifiedGuardrailResponse) -> Option<usize> {
    classified
        .tool_calls
        .as_ref()
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
}

fn request_tool_names(request: &ChatCompletionRequest) -> Vec<&str> {
    request
        .tools
        .as_ref()
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("function"))
        .filter_map(|function| function.get("name"))
        .filter_map(serde_json::Value::as_str)
        .collect()
}

fn retry_text(request: &ChatCompletionRequest) -> String {
    crate::chat::message_content_to_text(
        request.messages[0]
            .content
            .as_ref()
            .expect("retry content exists"),
    )
    .expect("retry text exists")
}
