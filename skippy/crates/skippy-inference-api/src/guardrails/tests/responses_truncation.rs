use super::*;

fn tiny_enabled_compaction() -> CompactionConfig {
    CompactionConfig {
        enabled: true,
        context_limit_tokens: Some(1),
        trigger_ratio_percent: 1,
        target_ratio_percent: 1,
        allow_reasoning_drop: false,
    }
}

fn responses_body(stream: bool) -> serde_json::Value {
    json!({
        "model": "tiny",
        "stream": stream,
        "input": [
            {"type": "message", "role": "user", "content": "run the lookup"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "full tool result that must survive"},
            {"type": "message", "role": "user", "content": "Your previous reply was not valid. Return valid output. Do not add extra text."}
        ]
    })
}

fn normalized_responses_request(stream: bool) -> GenerationRequest {
    let mut body = responses_body(stream);
    crate::responses::normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
    serde_json::from_value::<ChatCompletionRequest>(body)
        .unwrap()
        .into()
}

fn content_contains(message: &crate::generation::GenerationMessage, needle: &str) -> bool {
    matches!(
        message.content.as_ref(),
        Some(MessageContent::Text(content)) if content.contains(needle)
    )
}

#[tokio::test]
async fn normalized_responses_preserve_full_history_with_enabled_compaction() {
    let backend = Arc::new(RecordingBackend::default());
    let compacting = CompactingInferenceBackend::new(backend.clone(), tiny_enabled_compaction());
    let unary = normalized_responses_request(false);
    let streaming = normalized_responses_request(true);
    let expected_unary = unary.messages.clone();
    let expected_streaming = streaming.messages.clone();

    assert_eq!(unary.extra.get(MESH_COMPACT_FIELD), Some(&json!(false)));
    compacting.chat_completion(unary).await.unwrap();
    let stream = compacting
        .chat_completion_stream(streaming, InferenceRequestContext::new())
        .await
        .unwrap();
    drop(stream);

    assert_eq!(
        backend.seen_chat.lock().unwrap().as_ref().unwrap().messages,
        expected_unary
    );
    assert_eq!(
        backend
            .seen_chat_stream
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .messages,
        expected_streaming
    );
    for messages in [&expected_unary, &expected_streaming] {
        assert!(messages.iter().any(|message| message.role == "tool"));
        assert!(
            messages
                .iter()
                .any(|message| content_contains(message, "Your previous reply"))
        );
        assert!(
            messages.iter().all(|message| !content_contains(
                message,
                "Context was compacted before this turn"
            ))
        );
    }
}

#[tokio::test]
async fn ordinary_chat_still_compacts_at_the_same_threshold() {
    let backend = Arc::new(RecordingBackend::default());
    let compacting = CompactingInferenceBackend::new(backend.clone(), tiny_enabled_compaction());
    let request: GenerationRequest = serde_json::from_value::<ChatCompletionRequest>(json!({
        "model": "tiny",
        "messages": [
            {"role": "tool", "tool_call_id": "call_1", "content": "large stale result"},
            {"role": "user", "content": "Your previous reply was not valid. Return valid output. Do not add extra text."},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap()
    .into();

    compacting.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
    assert!(
        seen.messages
            .iter()
            .all(|message| !content_contains(message, "Your previous reply"))
    );
}
