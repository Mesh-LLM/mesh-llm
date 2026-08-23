use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

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
mod retry;
mod state;
mod structured;
mod telemetry;
mod tools;
mod validation;

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

    async fn collaborative_chat_completion(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let Some(raw_contract) = request
            .extra
            .remove(mesh_llm_guardrails::MESH_COLLABORATION_FIELD)
        else {
            return self.guarded_chat_completion(request, context).await;
        };
        if request.stream {
            return Err(crate::errors::OpenAiError::unsupported(
                "mesh_collaboration does not support streaming",
            ));
        }
        let contract = mesh_llm_guardrails::CollaborationContract::parse(
            &raw_contract,
            request.tools.as_ref(),
        )
        .map_err(|error| crate::errors::OpenAiError::invalid_request(error.to_string()))?;
        let response = self
            .backend
            .chat_completion_with_context(request, context)
            .await?;
        finalize_collaboration_response(response, &contract)
    }

    async fn guarded_chat_completion(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
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
                    None,
                );
                self.backend
                    .chat_completion_with_context(request, context)
                    .await
            }
            GuardrailRequestOutcome::Reject { kind } => Err(errors::guardrail_error(*kind)),
            GuardrailRequestOutcome::Guarded { backend_request } => {
                if matches!(policy.mode, GuardrailMode::MetricsOnly) {
                    return self
                        .metrics_only_chat_completion(request, &engine, &prepared, context)
                        .await;
                }

                let max_attempts = retry::max_attempts(&prepared, &policy);
                let mut attempt_index = 0_u8;
                let mut attempt_request = (**backend_request).clone();

                loop {
                    let response = self
                        .backend
                        .chat_completion_with_context(attempt_request.clone(), context.clone())
                        .await?;
                    let classified = engine.classify_response(&prepared, &response);
                    let contract = telemetry_contract(&prepared.state.request_contract);
                    let attempt_bucket = telemetry_attempt_bucket(attempt_index.saturating_add(1));

                    if let Some(sanitized) =
                        retry::sanitize_success_response(&policy, &response, &classified)
                    {
                        self.record_outcome(
                            prepared.state.mode,
                            contract,
                            GuardrailTelemetryOutcome::Valid,
                            Some(attempt_bucket),
                        );
                        return Ok(sanitized);
                    }

                    if matches!(policy.mode, GuardrailMode::MetricsOnly) {
                        self.record_outcome(
                            prepared.state.mode,
                            contract,
                            GuardrailTelemetryOutcome::MetricsOnlyFailure,
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
                            Some(telemetry_attempt_bucket(attempt_index)),
                        );
                        return retry::exhaustion_result(&policy, response, &classified);
                    }

                    self.record_outcome(
                        prepared.state.mode,
                        contract,
                        GuardrailTelemetryOutcome::Retried,
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
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let response = self
            .backend
            .chat_completion_with_context(request, context)
            .await?;
        let classified = engine.classify_response(prepared, &response);
        self.record_outcome(
            prepared.state.mode,
            telemetry_contract(&prepared.state.request_contract),
            metrics_only_outcome(&classified),
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
        attempt_bucket: Option<GuardrailTelemetryAttemptBucket>,
    ) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.record_outcome(
                mode,
                contract,
                outcome.as_str(),
                attempt_bucket.map(GuardrailTelemetryAttemptBucket::as_str),
            );
        }
    }
}

fn metrics_only_outcome(
    classified: &validation::ClassifiedGuardrailResponse,
) -> GuardrailTelemetryOutcome {
    match classified.category {
        validation::GuardrailResponseCategory::ValidText
        | validation::GuardrailResponseCategory::ValidToolCalls
        | validation::GuardrailResponseCategory::ValidSyntheticRespond
        | validation::GuardrailResponseCategory::ValidSyntheticStructured => {
            GuardrailTelemetryOutcome::Valid
        }
        validation::GuardrailResponseCategory::MalformedToolText
        | validation::GuardrailResponseCategory::UnknownTool
        | validation::GuardrailResponseCategory::InvalidToolArguments
        | validation::GuardrailResponseCategory::InvalidStructuredPayload
        | validation::GuardrailResponseCategory::MixedTerminalAndTool
        | validation::GuardrailResponseCategory::ToolCallsNotAllowed
        | validation::GuardrailResponseCategory::TooManyToolCalls
        | validation::GuardrailResponseCategory::EmptyOutput => {
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

fn finalize_collaboration_response(
    mut response: ChatCompletionResponse,
    contract: &mesh_llm_guardrails::CollaborationContract,
) -> OpenAiResult<ChatCompletionResponse> {
    let choice = response
        .choices
        .first_mut()
        .ok_or_else(|| crate::errors::OpenAiError::backend("backend returned no choices"))?;
    if let Some(tool_calls) = choice.message.tool_calls.as_ref()
        && !mesh_llm_guardrails::collaboration_calls_output_tool(tool_calls, contract)
    {
        return Ok(response);
    }
    let arguments = contract
        .final_arguments(
            choice.message.content.as_deref(),
            choice.message.tool_calls.as_ref(),
        )
        .map_err(|error| crate::errors::OpenAiError::backend(error.to_string()))?;
    let tool = &contract.report_required().tool;
    choice.message.content = None;
    choice.message.tool_calls = Some(json!([{
        "id": "call_mesh_collaboration",
        "type": "function",
        "function": {
            "name": tool,
            "arguments": Value::Object(arguments).to_string(),
        }
    }]));
    choice.finish_reason = Some(crate::common::FinishReason::ToolCalls);
    Ok(response)
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
        self.collaborative_chat_completion(request, OpenAiRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.collaborative_chat_completion(request, context).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        if request
            .extra
            .contains_key(mesh_llm_guardrails::MESH_COLLABORATION_FIELD)
        {
            return Err(crate::errors::OpenAiError::unsupported(
                "mesh_collaboration does not support streaming",
            ));
        }
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        self.backend.completion_with_context(request, context).await
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
