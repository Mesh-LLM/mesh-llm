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
pub use policy::{
    GuardrailMode, GuardrailPolicy, GuardrailPolicyHandle, RetryExhaustionMode,
    StreamingGuardrailMode,
};
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
                if matches!(policy.mode, GuardrailMode::MetricsOnly) {
                    return self
                        .metrics_only_chat_completion(request, &engine, &prepared)
                        .await;
                }

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

    async fn metrics_only_chat_completion(
        &self,
        request: ChatCompletionRequest,
        engine: &GuardrailEngine,
        prepared: &state::PreparedGuardrailRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let response = self.backend.chat_completion(request).await?;
        let classified = engine.classify_response(prepared, &response);
        let parser_stage = telemetry_parser_stage(classified.parser_stage);
        self.record_outcome(
            prepared.state.mode,
            telemetry_contract(&prepared.state.request_contract),
            metrics_only_outcome(&classified, parser_stage),
            Some(parser_stage),
            Some(GuardrailTelemetryAttemptBucket::One),
        );
        Ok(response)
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

fn metrics_only_outcome(
    classified: &rescue::ClassifiedGuardrailResponse,
    parser_stage: GuardrailTelemetryParserStage,
) -> GuardrailTelemetryOutcome {
    match classified.category {
        rescue::GuardrailResponseCategory::ValidText
        | rescue::GuardrailResponseCategory::ValidToolCalls
        | rescue::GuardrailResponseCategory::ValidSyntheticRespond
        | rescue::GuardrailResponseCategory::ValidSyntheticStructured => {
            if matches!(parser_stage, GuardrailTelemetryParserStage::None) {
                GuardrailTelemetryOutcome::Valid
            } else {
                GuardrailTelemetryOutcome::Rescued
            }
        }
        rescue::GuardrailResponseCategory::MalformedToolText
        | rescue::GuardrailResponseCategory::UnknownTool
        | rescue::GuardrailResponseCategory::InvalidToolArguments
        | rescue::GuardrailResponseCategory::InvalidStructuredPayload
        | rescue::GuardrailResponseCategory::MixedTerminalAndTool
        | rescue::GuardrailResponseCategory::ToolCallsNotAllowed
        | rescue::GuardrailResponseCategory::TooManyToolCalls
        | rescue::GuardrailResponseCategory::EmptyOutput => {
            GuardrailTelemetryOutcome::MetricsOnlyFailure
        }
    }
}

fn telemetry_decision(outcome: &state::GuardrailRequestOutcome) -> GuardrailTelemetryDecision {
    match outcome {
        state::GuardrailRequestOutcome::Guarded { .. } => GuardrailTelemetryDecision::Eligible,
        state::GuardrailRequestOutcome::Reject { .. } => GuardrailTelemetryDecision::Rejected,
        state::GuardrailRequestOutcome::PassThrough { reason } => match reason {
            GuardrailTelemetryBypassReason::Disabled
            | GuardrailTelemetryBypassReason::Streaming
            | GuardrailTelemetryBypassReason::NoContract
            | GuardrailTelemetryBypassReason::AfterToolResult => {
                GuardrailTelemetryDecision::Bypassed
            }
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
mod tests;
