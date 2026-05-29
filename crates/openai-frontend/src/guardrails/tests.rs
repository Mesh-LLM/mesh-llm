use super::*;
use std::{collections::VecDeque, sync::Mutex};

use futures_util::stream;
use serde_json::json;

use crate::{
    Usage, chat::MessageContent, common::ReasoningEffort,
    responses::translate_chat_completion_to_responses,
};

use super::{
    errors::{
        GUARDRAIL_RESERVED_TOOL_NAME_CODE, GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE,
        GUARDRAIL_UNSUPPORTED_COMBINATION_CODE, GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE,
        GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_CODE, GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_MESSAGE,
        GUARDRAIL_VALIDATION_FAILED_CODE, GUARDRAIL_VALIDATION_FAILED_MESSAGE,
        reserved_tool_name_error, unsupported_combination_error, validation_failed_error,
    },
    policy::RetryExhaustionMode,
    request_contract::{
        MeshGuardrailsOverride, ParallelToolCalls, RawResponseFormat, RawToolChoice, RawToolSpec,
    },
    rescue::{
        ClassifiedGuardrailResponse, GuardrailParserStage, GuardrailResponseCategory,
        strip_thinking_blocks,
    },
    telemetry::{
        GuardrailTelemetryBypassReason, GuardrailTelemetryContract, GuardrailTelemetryDecision,
        GuardrailTelemetryOutcome, GuardrailTelemetryParserStage,
    },
    tools::{MESH_EMIT_STRUCTURED_TOOL_NAME, MESH_RESPOND_TOOL_NAME},
};

#[derive(Default)]
struct RecordingBackend {
    seen_chat: Mutex<Option<ChatCompletionRequest>>,
    seen_chat_stream: Mutex<Option<ChatCompletionRequest>>,
    seen_completion: Mutex<Option<CompletionRequest>>,
    seen_completion_stream: Mutex<Option<CompletionRequest>>,
}

struct SequencedBackend {
    chat_requests: Mutex<Vec<ChatCompletionRequest>>,
    chat_responses: Mutex<VecDeque<OpenAiResult<ChatCompletionResponse>>>,
}

impl SequencedBackend {
    fn new(chat_responses: Vec<OpenAiResult<ChatCompletionResponse>>) -> Self {
        Self {
            chat_requests: Mutex::new(Vec::new()),
            chat_responses: Mutex::new(VecDeque::from(chat_responses)),
        }
    }
}

#[derive(Default)]
struct RecordingTelemetrySink {
    decisions: Mutex<Vec<RecordedDecision>>,
    outcomes: Mutex<Vec<RecordedOutcome>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedDecision {
    mode: GuardrailMode,
    contract: Option<&'static str>,
    decision: &'static str,
    bypass_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordedOutcome {
    mode: GuardrailMode,
    contract: Option<&'static str>,
    outcome: &'static str,
    parser_stage: Option<&'static str>,
    attempt_bucket: Option<&'static str>,
}

impl GuardrailTelemetrySink for RecordingTelemetrySink {
    fn record_decision(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        decision: &'static str,
        bypass_reason: Option<&'static str>,
    ) {
        self.decisions.lock().unwrap().push(RecordedDecision {
            mode,
            contract,
            decision,
            bypass_reason,
        });
    }

    fn record_outcome(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        outcome: &'static str,
        parser_stage: Option<&'static str>,
        attempt_bucket: Option<&'static str>,
    ) {
        self.outcomes.lock().unwrap().push(RecordedOutcome {
            mode,
            contract,
            outcome,
            parser_stage,
            attempt_bucket,
        });
    }
}

#[async_trait]
impl OpenAiBackend for RecordingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("guarded-model")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        *self.seen_chat.lock().unwrap() = Some(request.clone());
        Ok(recording_backend_chat_response(&request))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        *self.seen_chat_stream.lock().unwrap() = Some(request);
        Ok(Box::pin(stream::empty()))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        *self.seen_completion.lock().unwrap() = Some(request.clone());
        Ok(CompletionResponse::new(
            request.model,
            "ok",
            Usage::new(0, 0),
        ))
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        *self.seen_completion_stream.lock().unwrap() = Some(request);
        Ok(Box::pin(stream::empty()))
    }
}

#[async_trait]
impl OpenAiBackend for SequencedBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("guarded-model")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_requests.lock().unwrap().push(request.clone());
        self.chat_responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("expected queued chat response")
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Ok(Box::pin(stream::empty()))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        Ok(CompletionResponse::new(
            request.model,
            "ok",
            Usage::new(0, 0),
        ))
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        Ok(Box::pin(stream::empty()))
    }
}

#[tokio::test]
async fn disabled_mode_delegates_chat_completion() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(backend.clone(), GuardrailPolicy::default());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let models = guarded.models().await.unwrap();
    let _ = guarded.chat_completion(request.clone()).await.unwrap();
    let _ = guarded
        .chat_completion_stream(request.clone(), OpenAiRequestContext::new())
        .await
        .unwrap();

    let completion_request: CompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "prompt": "hello"
    }))
    .unwrap();
    let _ = guarded
        .completion(completion_request.clone())
        .await
        .unwrap();
    let _ = guarded
        .completion_stream(completion_request.clone(), OpenAiRequestContext::new())
        .await
        .unwrap();

    assert_eq!(models, vec![ModelObject::new("guarded-model")]);
    assert_eq!(
        backend.seen_chat.lock().unwrap().clone(),
        Some(request.clone())
    );
    assert_eq!(
        backend.seen_chat_stream.lock().unwrap().clone(),
        Some(request.clone())
    );
    assert_eq!(
        backend.seen_completion.lock().unwrap().clone(),
        Some(completion_request.clone())
    );
    assert_eq!(
        backend.seen_completion_stream.lock().unwrap().clone(),
        Some(completion_request)
    );
}

#[tokio::test]
async fn policy_handle_enables_same_guarded_backend_without_reconstruction() {
    let backend = Arc::new(RecordingBackend::default());
    let policy_handle = GuardrailPolicyHandle::default();
    let guarded = GuardedOpenAiBackend::with_policy_handle(backend.clone(), policy_handle.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    guarded.chat_completion(request.clone()).await.unwrap();
    assert_eq!(
        backend.seen_chat.lock().unwrap().clone(),
        Some(request.clone())
    );

    policy_handle.update(enforce_policy());
    guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
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
    assert!(tool_names.contains(&MESH_RESPOND_TOOL_NAME));
}

#[tokio::test]
async fn compacting_backend_applies_forced_mesh_compact_override() {
    let backend = Arc::new(RecordingBackend::default());
    let compacting = CompactingOpenAiBackend::new(backend.clone(), CompactionConfig::default());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny",
        "messages": [
            {"role": "tool", "content": "large stale result"},
            {"role": "user", "content": "continue"}
        ],
        "mesh_compact": true
    }))
    .unwrap();

    compacting.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn compacting_backend_applies_forced_mesh_compact_override_to_chat_stream() {
    let backend = Arc::new(RecordingBackend::default());
    let compacting = CompactingOpenAiBackend::new(backend.clone(), CompactionConfig::default());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny",
        "messages": [
            {"role": "tool", "content": "large stale result"},
            {"role": "user", "content": "continue"}
        ],
        "mesh_compact": true
    }))
    .unwrap();

    let stream = compacting
        .chat_completion_stream(request, OpenAiRequestContext::new())
        .await
        .unwrap();
    drop(stream);

    let seen = backend.seen_chat_stream.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn compacting_backend_leaves_completion_requests_untouched() {
    let backend = Arc::new(RecordingBackend::default());
    let compacting = CompactingOpenAiBackend::new(backend.clone(), CompactionConfig::default());
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "tiny",
        "prompt": "hello"
    }))
    .unwrap();

    compacting.completion(request.clone()).await.unwrap();
    let stream = compacting
        .completion_stream(request.clone(), OpenAiRequestContext::new())
        .await
        .unwrap();
    drop(stream);

    assert_eq!(
        backend.seen_completion.lock().unwrap().clone(),
        Some(request.clone())
    );
    assert_eq!(
        backend.seen_completion_stream.lock().unwrap().clone(),
        Some(request)
    );
}

#[test]
fn public_policy_defaults_are_conservative() {
    let policy = GuardrailPolicy::default();
    let _public_mode = crate::GuardrailMode::Disabled;
    let _public_streaming = crate::StreamingGuardrailMode::PassThrough;

    assert_eq!(policy.mode, GuardrailMode::Disabled);
    assert_eq!(policy.streaming_mode, StreamingGuardrailMode::PassThrough);
    assert_eq!(policy.max_tool_retries, 1);
    assert_eq!(policy.max_structured_retries, 2);
    assert_eq!(policy.retry_exhaustion_mode, RetryExhaustionMode::Error);
    assert!(policy.small_models_only());
    assert_eq!(policy.small_param_threshold_b, 9.0);
    assert_eq!(policy.reserved_tool_prefix, "_mesh_");

    let reserved = reserved_tool_name_error().body();
    let unsupported = unsupported_combination_error().body();
    let validation = validation_failed_error().body();
    assert_eq!(
        reserved.error.code.as_deref(),
        Some(GUARDRAIL_RESERVED_TOOL_NAME_CODE)
    );
    assert_eq!(reserved.error.message, GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE);
    assert_eq!(
        unsupported.error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
    );
    assert_eq!(
        unsupported.error.message,
        GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE
    );
    assert_eq!(
        validation.error.code.as_deref(),
        Some(GUARDRAIL_VALIDATION_FAILED_CODE)
    );
    assert_eq!(
        validation.error.message,
        GUARDRAIL_VALIDATION_FAILED_MESSAGE
    );
}

#[test]
fn engine_prepares_request_without_backend() {
    let engine = GuardrailEngine::new(GuardrailPolicy::default());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "parallel_tool_calls": false,
        "response_format": supported_json_schema_response_format(),
        "mesh_guardrails": true
    }))
    .unwrap();

    let prepared = engine.prepare_request(&request);
    let state = prepared.state;

    assert_eq!(state.model, "guarded-model");
    assert_eq!(state.mode, GuardrailMode::Disabled);
    assert!(!state.requested_stream);
    assert_eq!(
        state.mesh_guardrails_override,
        MeshGuardrailsOverride::Enabled
    );
    assert!(matches!(
        state.request_contract.tools,
        RawToolSpec::Entries(ref tools) if tools[0].name.as_deref() == Some("lookup")
    ));
    assert_eq!(
        state.request_contract.tool_choice,
        RawToolChoice::ForcedName("lookup".to_string())
    );
    assert_eq!(
        state.request_contract.parallel_tool_calls,
        ParallelToolCalls::Disabled
    );
    assert!(matches!(
        state.request_contract.response_format,
        RawResponseFormat::Structured(_)
    ));
}

#[test]
fn request_contract_parses_raw_openai_fields() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}, {"type": "function"}],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "response_format": {"type": "text"},
        "mesh_guardrails": false
    }))
    .unwrap();
    let contract = request_contract::from_request(&request);
    assert!(matches!(
        contract.tools,
        RawToolSpec::Entries(ref tools)
            if tools.len() == 2 && tools[0].name.as_deref() == Some("lookup") && tools[1].name.is_none()
    ));
    assert_eq!(contract.tool_choice, RawToolChoice::Auto);
    assert_eq!(contract.parallel_tool_calls, ParallelToolCalls::Enabled);
    assert_eq!(contract.response_format, RawResponseFormat::Text);
    assert_eq!(contract.mesh_guardrails, MeshGuardrailsOverride::Disabled);

    let absent: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    let absent_contract = request_contract::from_request(&absent);
    assert_eq!(absent_contract.tools, RawToolSpec::Absent);
    assert_eq!(absent_contract.tool_choice, RawToolChoice::Absent);
    assert_eq!(
        absent_contract.parallel_tool_calls,
        ParallelToolCalls::Absent
    );
    assert_eq!(absent_contract.response_format, RawResponseFormat::Absent);
    assert_eq!(
        absent_contract.mesh_guardrails,
        MeshGuardrailsOverride::Unset
    );

    let forced: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {"type": "json_object"}
    }))
    .unwrap();
    let forced_contract = request_contract::from_request(&forced);
    assert_eq!(
        forced_contract.tool_choice,
        RawToolChoice::ForcedName("lookup".to_string())
    );
    assert!(matches!(
        forced_contract.response_format,
        RawResponseFormat::Structured(_)
    ));

    let malformed: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "guarded-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": "bad",
        "tool_choice": 7,
        "response_format": [],
        "mesh_guardrails": "yes"
    }))
    .unwrap();
    let malformed_contract = request_contract::from_request(&malformed);
    assert_eq!(malformed_contract.tools, RawToolSpec::InvalidType);
    assert_eq!(malformed_contract.tool_choice, RawToolChoice::InvalidType);
    assert_eq!(
        malformed_contract.response_format,
        RawResponseFormat::InvalidType
    );
    assert_eq!(
        malformed_contract.mesh_guardrails,
        MeshGuardrailsOverride::InvalidType
    );

    let message_text = request.messages[0].content.clone();
    assert_eq!(
        message_text,
        Some(MessageContent::Text("hello".to_string()))
    );
}

#[tokio::test]
async fn auto_tool_request_injects_mesh_respond() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let original = request.clone();
    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    let seen_tools = seen.tools.unwrap();
    let tools = seen_tools.as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["function"]["name"], "lookup");
    assert_eq!(tools[1]["function"]["name"], MESH_RESPOND_TOOL_NAME);
    assert_eq!(
        tools[1]["function"]["parameters"]["properties"]["message"]["type"],
        "string"
    );
    assert_eq!(original.tools.unwrap().as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn guarded_tool_request_suppresses_implicit_thinking() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(backend.clone(), enforce_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(
        seen.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.enabled),
        Some(false)
    );
    assert_eq!(seen.reasoning_effort, Some(ReasoningEffort::None));
}

#[tokio::test]
async fn guarded_structured_request_suppresses_implicit_thinking() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(backend.clone(), enforce_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(
        seen.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.enabled),
        Some(false)
    );
    assert_eq!(seen.reasoning_effort, Some(ReasoningEffort::None));
}

#[tokio::test]
async fn guarded_request_preserves_explicit_reasoning_effort() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(backend.clone(), enforce_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "reasoning_effort": "low"
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.reasoning, None);
    assert_eq!(seen.reasoning_effort, Some(ReasoningEffort::Low));
}

#[tokio::test]
async fn guarded_request_preserves_explicit_provider_thinking_flag() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(backend.clone(), enforce_policy());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "chat_template_kwargs": {"enable_thinking": true}
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.reasoning, None);
    assert_eq!(seen.reasoning_effort, None);
    assert_eq!(
        seen.extra
            .get("chat_template_kwargs")
            .and_then(serde_json::Value::as_object)
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[tokio::test]
async fn absent_tool_choice_injects_mesh_respond() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "llama-3-7b-instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    let tools = seen.tools.unwrap();
    let tools = tools.as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[1]["function"]["name"], MESH_RESPOND_TOOL_NAME);
}

#[tokio::test]
async fn structured_only_request_injects_mesh_emit_structured() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "json"}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    let tools = seen.tools.unwrap();
    let tools = tools.as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], MESH_EMIT_STRUCTURED_TOOL_NAME);
}

#[tokio::test]
async fn forced_user_tool_request_does_not_inject_mesh_respond() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    let tools = seen.tools.unwrap();
    let tools = tools.as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "lookup");
}

#[tokio::test]
async fn reserved_tool_name_is_rejected_in_enforce_mode() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "_mesh_respond"}}],
        "tool_choice": "auto"
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();
    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_RESERVED_TOOL_NAME_CODE)
    );
    assert_eq!(body.error.message, GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE);
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn forced_reserved_tool_name_is_rejected_in_enforce_mode() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": {"type": "function", "function": {"name": "_mesh_emit_structured"}}
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();
    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_RESERVED_TOOL_NAME_CODE)
    );
    assert_eq!(body.error.message, GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE);
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn metrics_only_records_reserved_tool_collision_and_passes_through() {
    let backend = Arc::new(RecordingBackend::default());
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::MetricsOnly,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "_mesh_respond"}}],
        "tool_choice": "auto"
    }))
    .unwrap();
    let original = request.clone();

    let _ = guarded.chat_completion(request).await.unwrap();

    assert_eq!(backend.seen_chat.lock().unwrap().clone(), Some(original));
    let decisions = telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Unsupported.as_str()
            && record.bypass_reason
                == Some(GuardrailTelemetryBypassReason::ReservedCollision.as_str())
    }));
}

#[tokio::test]
async fn unsupported_structured_with_real_tools_is_rejected_in_enforce_mode() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_schema"}
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();
    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
    );
    assert_eq!(
        body.error.message,
        GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE
    );
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn real_tools_plus_structured_output_returns_unsupported() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": supported_json_schema_response_format()
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
    );
    assert_eq!(
        error.body().error.message,
        GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE
    );
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn forced_tool_plus_structured_is_rejected_in_enforce_mode() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "response_format": {"type": "json_schema"}
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();
    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
    );
    assert_eq!(
        body.error.message,
        GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE
    );
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn parallel_tool_calls_false_with_structured_is_rejected_in_enforce_mode() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "parallel_tool_calls": false,
        "response_format": {"type": "json_schema"}
    }))
    .unwrap();

    let error = guarded.chat_completion(request).await.unwrap_err();
    let body = error.body();
    assert_eq!(
        body.error.code.as_deref(),
        Some(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
    );
    assert_eq!(
        body.error.message,
        GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE
    );
    assert!(backend.seen_chat.lock().unwrap().is_none());
}

#[tokio::test]
async fn metrics_only_records_unsupported_combination_and_passes_through() {
    let backend = Arc::new(RecordingBackend::default());
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::MetricsOnly,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_schema"}
    }))
    .unwrap();
    let original = request.clone();

    let _ = guarded.chat_completion(request).await.unwrap();

    assert_eq!(backend.seen_chat.lock().unwrap().clone(), Some(original));
    let decisions = telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Unsupported.as_str()
            && record.bypass_reason
                == Some(GuardrailTelemetryBypassReason::MixedToolsStructured.as_str())
    }));
}

#[tokio::test]
async fn streaming_requests_bypass_guardrails() {
    let backend = Arc::new(RecordingBackend::default());
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true,
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let original = request.clone();

    let _ = guarded.chat_completion(request).await.unwrap();

    assert_eq!(backend.seen_chat.lock().unwrap().clone(), Some(original));
    let decisions = telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Bypassed.as_str()
            && record.bypass_reason == Some(GuardrailTelemetryBypassReason::Streaming.as_str())
    }));
}

#[tokio::test]
async fn no_tools_and_text_response_format_passes_through() {
    let backend = Arc::new(RecordingBackend::default());
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "text"}
    }))
    .unwrap();
    let original = request.clone();

    let _ = guarded.chat_completion(request).await.unwrap();

    assert_eq!(backend.seen_chat.lock().unwrap().clone(), Some(original));
    let decisions = telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Bypassed.as_str()
            && record.bypass_reason == Some(GuardrailTelemetryBypassReason::NoContract.as_str())
    }));
}

#[tokio::test]
async fn small_model_threshold_controls_small_model_eligibility() {
    let guarded_backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        guarded_backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            small_param_threshold_b: 8.0,
            ..GuardrailPolicy::default()
        },
    );
    let guarded_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let _ = guarded.chat_completion(guarded_request).await.unwrap();
    let guarded_seen = guarded_backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(guarded_seen.tools.unwrap().as_array().unwrap().len(), 2);

    let bypass_backend = Arc::new(RecordingBackend::default());
    let bypass_telemetry = Arc::new(RecordingTelemetrySink::default());
    let bypass = GuardedOpenAiBackend::new(
        bypass_backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            small_param_threshold_b: 7.0,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(bypass_telemetry.clone());
    let bypass_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let bypass_original = bypass_request.clone();
    let _ = bypass.chat_completion(bypass_request).await.unwrap();
    assert_eq!(
        bypass_backend.seen_chat.lock().unwrap().clone(),
        Some(bypass_original)
    );
    let decisions = bypass_telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Bypassed.as_str()
            && record.bypass_reason == Some(GuardrailTelemetryBypassReason::NoContract.as_str())
    }));
}

#[tokio::test]
async fn small_model_only_policy_bypasses_large_model() {
    let small_backend = Arc::new(RecordingBackend::default());
    let small_guarded = GuardedOpenAiBackend::new(
        small_backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let small_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let _ = small_guarded.chat_completion(small_request).await.unwrap();
    let small_seen = small_backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(small_seen.tools.unwrap().as_array().unwrap().len(), 2);

    let large_backend = Arc::new(RecordingBackend::default());
    let large_telemetry = Arc::new(RecordingTelemetrySink::default());
    let large_guarded = GuardedOpenAiBackend::new(
        large_backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(large_telemetry.clone());
    let large_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-70B-Instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let large_original = large_request.clone();
    let _ = large_guarded.chat_completion(large_request).await.unwrap();
    assert_eq!(
        large_backend.seen_chat.lock().unwrap().clone(),
        Some(large_original)
    );
    let decisions = large_telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Bypassed.as_str()
            && record.bypass_reason == Some(GuardrailTelemetryBypassReason::NoContract.as_str())
    }));
}

#[tokio::test]
async fn all_model_policy_guards_large_models_too() {
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
        "model": "Qwen3-70B-Instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.tools.unwrap().as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn mesh_guardrails_false_bypasses_request() {
    let backend = Arc::new(RecordingBackend::default());
    let telemetry = Arc::new(RecordingTelemetrySink::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        },
    )
    .with_telemetry(telemetry.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-8B-Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "mesh_guardrails": false,
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();
    let original = request.clone();

    let _ = guarded.chat_completion(request).await.unwrap();

    assert_eq!(backend.seen_chat.lock().unwrap().clone(), Some(original));
    let decisions = telemetry.decisions.lock().unwrap().clone();
    assert!(decisions.iter().any(|record| {
        record.decision == GuardrailTelemetryDecision::Bypassed.as_str()
            && record.bypass_reason == Some(GuardrailTelemetryBypassReason::Disabled.as_str())
    }));
}

#[tokio::test]
async fn mesh_guardrails_true_opts_large_model_into_guardrails() {
    let backend = Arc::new(RecordingBackend::default());
    let guarded = GuardedOpenAiBackend::new(
        backend.clone(),
        GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            ..GuardrailPolicy::default()
        },
    );
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "Qwen3-70B-Instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "mesh_guardrails": true,
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    }))
    .unwrap();

    let _ = guarded.chat_completion(request).await.unwrap();

    let seen = backend.seen_chat.lock().unwrap().clone().unwrap();
    assert_eq!(seen.tools.unwrap().as_array().unwrap().len(), 2);
}

#[test]
fn telemetry_response_records_use_bounded_enums_only() {
    assert_eq!(GuardrailTelemetryDecision::Eligible.as_str(), "eligible");
    assert_eq!(
        GuardrailTelemetryBypassReason::MixedToolsStructured.as_str(),
        "mixed_tools_structured"
    );
    assert_eq!(
        GuardrailTelemetryOutcome::MetricsOnlyFailure.as_str(),
        "metrics_only_failure"
    );
    assert_eq!(
        GuardrailTelemetryParserStage::JsonFenced.as_str(),
        "json_fenced"
    );
    assert_eq!(telemetry_attempt_bucket(3).as_str(), "3_plus");
}

mod response_validation;
mod tool_contract_validation;

fn enforce_policy() -> GuardrailPolicy {
    GuardrailPolicy {
        mode: GuardrailMode::Enforce,
        apply_to_all_models: true,
        ..GuardrailPolicy::default()
    }
}

fn recording_backend_chat_response(request: &ChatCompletionRequest) -> ChatCompletionResponse {
    let tool_names = request
        .tools
        .as_ref()
        .and_then(|tools| tools.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.get("function"))
                .filter_map(|function| function.get("name"))
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if tool_names.contains(&MESH_EMIT_STRUCTURED_TOOL_NAME) {
        return response_with_tool_calls(
            &request.model,
            json!([{
                "type": "function",
                "function": {
                    "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
                    "arguments": "{\"answer\":42}"
                }
            }]),
            None,
        );
    }

    if let Some(name) = tool_names
        .iter()
        .copied()
        .find(|name| *name != MESH_RESPOND_TOOL_NAME)
    {
        return response_with_tool_calls(
            &request.model,
            json!([{
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": "{\"ok\":true}"
                }
            }]),
            None,
        );
    }

    if tool_names.contains(&MESH_RESPOND_TOOL_NAME) {
        return response_with_tool_calls(
            &request.model,
            json!([{
                "type": "function",
                "function": {
                    "name": MESH_RESPOND_TOOL_NAME,
                    "arguments": "{\"message\":\"ok\"}"
                }
            }]),
            None,
        );
    }

    ChatCompletionResponse::new(&request.model, "ok", Usage::new(0, 0))
}

fn prepared_tool_request(
    engine: &GuardrailEngine,
    payload: serde_json::Value,
) -> super::state::PreparedGuardrailRequest {
    let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
    engine.prepare_request(&request)
}

fn prepared_text_request(
    engine: &GuardrailEngine,
    payload: serde_json::Value,
) -> super::state::PreparedGuardrailRequest {
    let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
    engine.prepare_request(&request)
}

fn response_with_content(model: &str, content: &str) -> ChatCompletionResponse {
    response_with_content_with_usage(model, content, Usage::new(3, 2))
}

fn response_with_content_with_usage(
    model: &str,
    content: &str,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "chatcmpl_test".to_string(),
        object: "chat.completion",
        created: 123,
        model: model.to_string(),
        choices: vec![crate::chat::ChatCompletionChoice {
            index: 0,
            message: crate::chat::AssistantMessage {
                role: "assistant",
                content: Some(content.to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: Some(crate::common::FinishReason::Stop),
        }],
        usage,
    }
}

fn response_with_tool_calls(
    model: &str,
    tool_calls: serde_json::Value,
    content: Option<&str>,
) -> ChatCompletionResponse {
    response_with_tool_calls_with_usage(model, tool_calls, content, Usage::new(3, 2))
}

fn response_with_tool_calls_with_usage(
    model: &str,
    tool_calls: serde_json::Value,
    content: Option<&str>,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "chatcmpl_test".to_string(),
        object: "chat.completion",
        created: 123,
        model: model.to_string(),
        choices: vec![crate::chat::ChatCompletionChoice {
            index: 0,
            message: crate::chat::AssistantMessage {
                role: "assistant",
                content: content.map(ToString::to_string),
                reasoning_content: None,
                tool_calls: Some(tool_calls),
            },
            logprobs: None,
            finish_reason: Some(crate::common::FinishReason::ToolCalls),
        }],
        usage,
    }
}

fn tool_call_name_from_response(response: &ChatCompletionResponse) -> Option<&str> {
    response
        .choices
        .first()?
        .message
        .tool_calls
        .as_ref()?
        .as_array()?
        .first()?
        .get("function")?
        .get("name")?
        .as_str()
}

fn tool_call_name(classified: &ClassifiedGuardrailResponse) -> Option<&str> {
    classified
        .tool_calls
        .as_ref()?
        .as_array()?
        .first()?
        .get("function")?
        .get("name")?
        .as_str()
}

fn tool_call_arguments(classified: &ClassifiedGuardrailResponse) -> Option<&str> {
    classified
        .tool_calls
        .as_ref()?
        .as_array()?
        .first()?
        .get("function")?
        .get("arguments")?
        .as_str()
}

fn supported_json_schema_response_format() -> serde_json::Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "integer"}
                },
                "required": ["answer"],
                "additionalProperties": false
            }
        }
    })
}
