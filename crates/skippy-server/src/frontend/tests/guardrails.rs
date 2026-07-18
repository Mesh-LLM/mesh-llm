use super::support::*;
use super::*;

#[test]
fn chat_runtime_feature_guard_allows_structured_output_for_guardrails() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}}
        }
    }))
    .unwrap();

    ensure_chat_runtime_features_supported(&request).unwrap();
}

#[derive(Default)]
struct StructuredGuardrailRecordingBackend {
    seen: Mutex<Option<ChatCompletionRequest>>,
}

#[async_trait]
impl OpenAiBackend for StructuredGuardrailRecordingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("test")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        ensure_chat_runtime_features_supported(&request)
            .expect("guarded wrapper should downgrade backend-facing structured requests");
        *self.seen.lock().unwrap() = Some(request);
        Ok(ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 123,
            model: "test".to_string(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(json!([{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "_mesh_emit_structured",
                            "arguments": "{\"answer\":\"ok\"}"
                        }
                    }])),
                },
                logprobs: None,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::new(1, 1),
            timings: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        unreachable!("streaming is not used in this test")
    }

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        unreachable!("completions are not used in this test")
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        unreachable!("completions are not used in this test")
    }
}

#[tokio::test]
async fn guarded_structured_output_is_not_rejected_by_runtime_feature_guard() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        }
        .into(),
        compaction: None,
    };
    let guarded = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":\"ok\"}")
    );

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert!(
        seen.response_format.is_none(),
        "guarded backend should clear backend-facing response_format"
    );
}

#[test]
fn standalone_guardrail_modes_have_expected_policies() {
    let metrics =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Metrics)
            .status();
    assert_eq!(metrics.mode, "metrics");
    assert_eq!(metrics.retry_exhaustion, "pass_last_text");
    assert_eq!(metrics.small_model_policy, "all");

    let enforce =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Enforce)
            .status();
    assert_eq!(enforce.mode, "enforce");
    assert_eq!(enforce.retry_exhaustion, "error");
    assert_eq!(enforce.small_model_policy, "all");

    let disabled =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Disabled)
            .status();
    assert_eq!(disabled.mode, "disabled");
    assert_eq!(disabled.small_model_policy, "small_models_only");
}

#[tokio::test]
async fn compaction_wraps_skippy_backend_even_when_guardrails_are_disabled() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy::default().into(),
        compaction: Some(CompactionConfig::default()),
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ],
        "mesh_compact": true
    }))
    .unwrap();

    let _ = wrapped.chat_completion(request).await;

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn disabled_skippy_guardrail_wrapper_can_be_enabled_live() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let policy: openai_frontend::GuardrailPolicyHandle = GuardrailPolicy::default().into();
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: policy.clone(),
        compaction: None,
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request = tool_request();

    wrapped.chat_completion(request.clone()).await.unwrap();
    assert_eq!(
        backend.seen.lock().unwrap().clone().unwrap().tools,
        request.tools
    );

    policy.update(GuardrailPolicy {
        mode: GuardrailMode::Enforce,
        apply_to_all_models: true,
        ..GuardrailPolicy::default()
    });
    let _ = wrapped.chat_completion(request).await;

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    let tool_names = seen
        .tools
        .as_ref()
        .and_then(|tools| tools.as_array())
        .unwrap()
        .iter()
        .filter_map(|tool| tool.get("function"))
        .filter_map(|function| function.get("name"))
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"_mesh_respond"));
    assert_eq!(guardrails.status().mode, "enforce");
}

#[tokio::test]
async fn compaction_wraps_skippy_backend_with_runtime_context_limit() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy::default().into(),
        compaction: Some(CompactionConfig {
            enabled: true,
            ..CompactionConfig::default()
        }),
    };
    let wrapped = guardrails.wrap_backend_with_context_limit(backend.clone(), Some(1));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap();

    wrapped.chat_completion(request).await.unwrap();

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn compaction_and_guardrails_can_stack() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        }
        .into(),
        compaction: Some(CompactionConfig::default()),
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        },
        "mesh_compact": true
    }))
    .unwrap();

    let response = wrapped.chat_completion(request).await.unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":\"ok\"}")
    );

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert!(seen.response_format.is_none());
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}
