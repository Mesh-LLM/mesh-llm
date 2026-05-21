use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    backend::{
        ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
    },
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionRequest, CompletionResponse},
    models::ModelObject,
};

mod compact;
mod engine;
mod errors;
mod policy;
mod request_contract;
mod rescue;
mod retry;
mod state;
mod structured;
mod telemetry;
mod tools;

pub use compact::CompactingOpenAiBackend;
pub use mesh_llm_guardrails::{
    CompactionConfig, CompactionDecision, CompactionOverride, CompactionReport, MESH_COMPACT_FIELD,
    MESH_RESPOND_TOOL_NAME,
};
pub use policy::{GuardrailMode, GuardrailPolicy, GuardrailPolicyHandle, StreamingGuardrailMode};
pub use telemetry::GuardrailTelemetrySink;

use self::{
    engine::GuardrailEngine,
    errors::guardrail_error_catalog,
    state::GuardrailRequestOutcome,
    telemetry::{
        GuardrailTelemetryAttemptBucket, GuardrailTelemetryBypassReason,
        GuardrailTelemetryContract, GuardrailTelemetryDecision, GuardrailTelemetryOutcome,
        GuardrailTelemetryParserStage,
    },
};

#[derive(Clone)]
pub struct GuardedOpenAiBackend {
    backend: Arc<dyn OpenAiBackend>,
    policy: GuardrailPolicyHandle,
    telemetry: Option<Arc<dyn GuardrailTelemetrySink>>,
}

impl GuardedOpenAiBackend {
    pub fn new(backend: Arc<dyn OpenAiBackend>, policy: GuardrailPolicy) -> Self {
        Self::with_policy_handle(backend, GuardrailPolicyHandle::new(policy))
    }

    pub fn with_policy_handle(
        backend: Arc<dyn OpenAiBackend>,
        policy: GuardrailPolicyHandle,
    ) -> Self {
        Self {
            backend,
            policy,
            telemetry: None,
        }
    }

    pub fn with_telemetry(mut self, telemetry: Arc<dyn GuardrailTelemetrySink>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    async fn guarded_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let _guardrail_error_catalog = guardrail_error_catalog();
        let policy = self.policy.snapshot();
        let engine = GuardrailEngine::new(policy.clone());
        let prepared = engine.prepare_request(&request);
        self.record_decision(&prepared);

        match &prepared.outcome {
            GuardrailRequestOutcome::PassThrough { .. } => {
                self.record_outcome(
                    prepared.state.mode,
                    telemetry_contract(&prepared.state.request_contract),
                    GuardrailTelemetryOutcome::PassThrough,
                    Some(GuardrailTelemetryParserStage::None),
                    None,
                );
                self.backend.chat_completion(request).await
            }
            GuardrailRequestOutcome::Reject { kind } => Err(errors::guardrail_error(*kind)),
            GuardrailRequestOutcome::Guarded { backend_request } => {
                let max_attempts = retry::max_attempts(&prepared, &policy);
                let mut attempt_index = 0_u8;
                let mut attempt_request = (**backend_request).clone();

                loop {
                    let response = self
                        .backend
                        .chat_completion(attempt_request.clone())
                        .await?;
                    let classified = engine.classify_response(&prepared, &response);
                    let parser_stage = telemetry_parser_stage(classified.parser_stage);
                    let contract = telemetry_contract(&prepared.state.request_contract);
                    let attempt_bucket = telemetry_attempt_bucket(attempt_index.saturating_add(1));

                    if let Some(sanitized) =
                        retry::sanitize_success_response(&policy, &response, &classified)
                    {
                        let outcome = if matches!(parser_stage, GuardrailTelemetryParserStage::None)
                        {
                            GuardrailTelemetryOutcome::Valid
                        } else {
                            GuardrailTelemetryOutcome::Rescued
                        };
                        self.record_outcome(
                            prepared.state.mode,
                            contract,
                            outcome,
                            Some(parser_stage),
                            Some(attempt_bucket),
                        );
                        return Ok(sanitized);
                    }

                    if matches!(policy.mode, GuardrailMode::MetricsOnly) {
                        self.record_outcome(
                            prepared.state.mode,
                            contract,
                            GuardrailTelemetryOutcome::MetricsOnlyFailure,
                            Some(parser_stage),
                            Some(attempt_bucket),
                        );
                        return Ok(response);
                    }

                    attempt_index = attempt_index.saturating_add(1);
                    if attempt_index >= max_attempts || !retry::should_retry(&classified) {
                        self.record_outcome(
                            prepared.state.mode,
                            contract,
                            GuardrailTelemetryOutcome::Failed,
                            Some(parser_stage),
                            Some(telemetry_attempt_bucket(attempt_index)),
                        );
                        return retry::exhaustion_result(&policy, response, &classified);
                    }

                    self.record_outcome(
                        prepared.state.mode,
                        contract,
                        GuardrailTelemetryOutcome::Retried,
                        Some(parser_stage),
                        Some(telemetry_attempt_bucket(attempt_index)),
                    );

                    attempt_request =
                        retry::build_retry_request(&prepared, attempt_index, &classified);
                }
            }
        }
    }

    fn record_decision(&self, prepared: &state::PreparedGuardrailRequest) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.record_decision(
                prepared.state.mode,
                telemetry_contract(&prepared.state.request_contract),
                telemetry_decision(&prepared.outcome).as_str(),
                telemetry_bypass_reason(&prepared.outcome)
                    .map(GuardrailTelemetryBypassReason::as_str),
            );
        }
    }

    fn record_outcome(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        outcome: GuardrailTelemetryOutcome,
        parser_stage: Option<GuardrailTelemetryParserStage>,
        attempt_bucket: Option<GuardrailTelemetryAttemptBucket>,
    ) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.record_outcome(
                mode,
                contract,
                outcome.as_str(),
                parser_stage.map(GuardrailTelemetryParserStage::as_str),
                attempt_bucket.map(GuardrailTelemetryAttemptBucket::as_str),
            );
        }
    }
}

fn telemetry_parser_stage(
    parser_stage: rescue::GuardrailParserStage,
) -> GuardrailTelemetryParserStage {
    match parser_stage {
        rescue::GuardrailParserStage::None => GuardrailTelemetryParserStage::None,
        rescue::GuardrailParserStage::JsonExact => GuardrailTelemetryParserStage::JsonExact,
        rescue::GuardrailParserStage::JsonFenced => GuardrailTelemetryParserStage::JsonFenced,
        rescue::GuardrailParserStage::JsonSubstring => GuardrailTelemetryParserStage::JsonSubstring,
    }
}

fn telemetry_decision(outcome: &state::GuardrailRequestOutcome) -> GuardrailTelemetryDecision {
    match outcome {
        state::GuardrailRequestOutcome::Guarded { .. } => GuardrailTelemetryDecision::Eligible,
        state::GuardrailRequestOutcome::Reject { .. } => GuardrailTelemetryDecision::Rejected,
        state::GuardrailRequestOutcome::PassThrough { reason } => match reason {
            GuardrailTelemetryBypassReason::Disabled
            | GuardrailTelemetryBypassReason::Streaming
            | GuardrailTelemetryBypassReason::NoContract => GuardrailTelemetryDecision::Bypassed,
            GuardrailTelemetryBypassReason::UnsupportedSurface
            | GuardrailTelemetryBypassReason::ReservedCollision
            | GuardrailTelemetryBypassReason::MixedToolsStructured => {
                GuardrailTelemetryDecision::Unsupported
            }
        },
    }
}

fn telemetry_bypass_reason(
    outcome: &state::GuardrailRequestOutcome,
) -> Option<GuardrailTelemetryBypassReason> {
    match outcome {
        state::GuardrailRequestOutcome::PassThrough { reason } => Some(*reason),
        state::GuardrailRequestOutcome::Reject { kind } => Some(match kind {
            errors::GuardrailErrorKind::ReservedToolName => {
                GuardrailTelemetryBypassReason::ReservedCollision
            }
            errors::GuardrailErrorKind::UnsupportedCombination => {
                GuardrailTelemetryBypassReason::MixedToolsStructured
            }
            errors::GuardrailErrorKind::UnsupportedSchemaFeature => {
                GuardrailTelemetryBypassReason::UnsupportedSurface
            }
            errors::GuardrailErrorKind::ValidationFailed => {
                GuardrailTelemetryBypassReason::NoContract
            }
        }),
        state::GuardrailRequestOutcome::Guarded { .. } => None,
    }
}

fn telemetry_contract(
    contract: &request_contract::GuardrailRequestContract,
) -> Option<&'static str> {
    if contract.requests_structured_output() {
        Some(GuardrailTelemetryContract::Structured.as_str())
    } else if contract.has_real_tools() {
        Some(GuardrailTelemetryContract::Tools.as_str())
    } else {
        None
    }
}

fn telemetry_attempt_bucket(attempts: u8) -> GuardrailTelemetryAttemptBucket {
    match attempts {
        0 | 1 => GuardrailTelemetryAttemptBucket::One,
        2 => GuardrailTelemetryAttemptBucket::Two,
        _ => GuardrailTelemetryAttemptBucket::ThreePlus,
    }
}

#[async_trait]
impl OpenAiBackend for GuardedOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.guarded_chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, sync::Mutex};

    use futures_util::stream;
    use serde_json::json;

    use crate::{chat::MessageContent, responses::translate_chat_completion_to_responses, Usage};

    use super::{
        errors::{
            reserved_tool_name_error, unsupported_combination_error, validation_failed_error,
            GUARDRAIL_RESERVED_TOOL_NAME_CODE, GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE,
            GUARDRAIL_UNSUPPORTED_COMBINATION_CODE, GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE,
            GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_CODE,
            GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_MESSAGE, GUARDRAIL_VALIDATION_FAILED_CODE,
            GUARDRAIL_VALIDATION_FAILED_MESSAGE,
        },
        policy::RetryExhaustionMode,
        request_contract::{
            MeshGuardrailsOverride, ParallelToolCalls, RawResponseFormat, RawToolChoice,
            RawToolSpec,
        },
        rescue::{
            strip_thinking_blocks, ClassifiedGuardrailResponse, GuardrailParserStage,
            GuardrailResponseCategory,
        },
        telemetry::{
            GuardrailTelemetryBypassReason, GuardrailTelemetryDecision, GuardrailTelemetryOutcome,
            GuardrailTelemetryParserStage,
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
        let guarded =
            GuardedOpenAiBackend::with_policy_handle(backend.clone(), policy_handle.clone());
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
        let response =
            response_with_content("Qwen3-8B-Q4_K_M", "lookup[ARGS]{\"city\":\"Sydney\"}");

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
        assert!(request.messages[0]
            .content
            .as_ref()
            .is_some_and(|content| content == &MessageContent::Text("weather".to_string())));
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
        assert!(parsed["output"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["type"] != "function_call"));
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
            backend,
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

        let response = guarded.chat_completion(request).await.unwrap();

        assert_eq!(response, original);
        assert!(telemetry.outcomes.lock().unwrap().iter().any(|record| {
            record.outcome == GuardrailTelemetryOutcome::MetricsOnlyFailure.as_str()
                && record.parser_stage == Some(GuardrailTelemetryParserStage::JsonExact.as_str())
        }));
    }

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
}
