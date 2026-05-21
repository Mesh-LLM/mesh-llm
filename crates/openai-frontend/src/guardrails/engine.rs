use crate::chat::ChatCompletionRequest;

use super::{
    errors::GuardrailErrorKind,
    policy::GuardrailPolicy,
    request_contract::{
        GuardrailRequestContract, MeshGuardrailsOverride, ParallelToolCalls, RawToolChoice,
    },
    rescue::ClassifiedGuardrailResponse,
    state::{GuardrailRequestOutcome, GuardrailRequestState, PreparedGuardrailRequest},
    telemetry::GuardrailTelemetryBypassReason,
    tools::{
        append_mesh_emit_structured_tool, append_mesh_respond_tool, model_param_size_b,
        request_uses_reserved_tool_name,
    },
};

#[derive(Debug, Clone)]
pub(crate) struct GuardrailEngine {
    policy: GuardrailPolicy,
}

impl GuardrailEngine {
    pub(crate) fn new(policy: GuardrailPolicy) -> Self {
        Self { policy }
    }

    pub(crate) fn prepare_request(
        &self,
        request: &ChatCompletionRequest,
    ) -> PreparedGuardrailRequest {
        let request_contract = super::request_contract::from_request(request);
        let state = GuardrailRequestState::from_request(
            request,
            self.policy.mode,
            self.policy.streaming_mode,
            request_contract,
        );

        let outcome = if matches!(state.mode, super::policy::GuardrailMode::Disabled) {
            GuardrailRequestOutcome::PassThrough {
                reason: GuardrailTelemetryBypassReason::Disabled,
            }
        } else if state.requested_stream {
            GuardrailRequestOutcome::PassThrough {
                reason: GuardrailTelemetryBypassReason::Streaming,
            }
        } else if matches!(
            state.mesh_guardrails_override,
            MeshGuardrailsOverride::Disabled
        ) {
            GuardrailRequestOutcome::PassThrough {
                reason: GuardrailTelemetryBypassReason::Disabled,
            }
        } else if !self.guardrails_apply_to_request(request, &state.request_contract)
            || (!state.request_contract.has_real_tools()
                && !state.request_contract.requests_structured_output())
        {
            GuardrailRequestOutcome::PassThrough {
                reason: GuardrailTelemetryBypassReason::NoContract,
            }
        } else if request_uses_reserved_tool_name(
            &state.request_contract,
            &self.policy.reserved_tool_prefix,
        ) {
            self.reject_or_record_only(GuardrailErrorKind::ReservedToolName)
        } else if self.uses_unsupported_v1_combination(&state.request_contract) {
            self.reject_or_record_only(GuardrailErrorKind::UnsupportedCombination)
        } else if state.request_contract.requests_structured_output()
            && !state.request_contract.has_supported_structured_output()
        {
            self.reject_or_record_only(GuardrailErrorKind::UnsupportedSchemaFeature)
        } else {
            let mut backend_request = request.clone();
            if state.request_contract.has_real_tools()
                && matches!(
                    state.request_contract.tool_choice,
                    RawToolChoice::Absent | RawToolChoice::Auto
                )
            {
                append_mesh_respond_tool(&mut backend_request);
            } else if state.request_contract.requests_structured_output()
                && !state.request_contract.has_real_tools()
                && state.request_contract.forced_tool_name().is_none()
            {
                append_mesh_emit_structured_tool(
                    &mut backend_request,
                    state
                        .request_contract
                        .structured_output_spec()
                        .expect("supported structured output checked before request rewrite"),
                );
                backend_request.response_format = None;
            }
            GuardrailRequestOutcome::Guarded {
                backend_request: Box::new(backend_request),
            }
        };

        PreparedGuardrailRequest { state, outcome }
    }

    pub(crate) fn classify_response(
        &self,
        prepared: &PreparedGuardrailRequest,
        response: &crate::chat::ChatCompletionResponse,
    ) -> ClassifiedGuardrailResponse {
        super::rescue::classify_response(prepared, response)
    }

    fn guardrails_apply_to_request(
        &self,
        request: &ChatCompletionRequest,
        contract: &GuardrailRequestContract,
    ) -> bool {
        matches!(contract.mesh_guardrails, MeshGuardrailsOverride::Enabled)
            || self.policy.apply_to_all_models
            || model_param_size_b(&request.model)
                .is_some_and(|param_size_b| param_size_b <= self.policy.small_param_threshold_b)
    }

    fn uses_unsupported_v1_combination(&self, contract: &GuardrailRequestContract) -> bool {
        contract.requests_structured_output()
            && (contract.has_real_tools()
                || contract.forced_tool_name().is_some()
                || matches!(contract.parallel_tool_calls, ParallelToolCalls::Disabled))
    }

    fn reject_or_record_only(&self, kind: GuardrailErrorKind) -> GuardrailRequestOutcome {
        match self.policy.mode {
            super::policy::GuardrailMode::MetricsOnly => GuardrailRequestOutcome::PassThrough {
                reason: match kind {
                    GuardrailErrorKind::ReservedToolName => {
                        GuardrailTelemetryBypassReason::ReservedCollision
                    }
                    GuardrailErrorKind::UnsupportedCombination => {
                        GuardrailTelemetryBypassReason::MixedToolsStructured
                    }
                    GuardrailErrorKind::UnsupportedSchemaFeature => {
                        GuardrailTelemetryBypassReason::UnsupportedSurface
                    }
                    GuardrailErrorKind::ValidationFailed => {
                        GuardrailTelemetryBypassReason::NoContract
                    }
                },
            },
            super::policy::GuardrailMode::Disabled => GuardrailRequestOutcome::PassThrough {
                reason: GuardrailTelemetryBypassReason::Disabled,
            },
            super::policy::GuardrailMode::Enforce => GuardrailRequestOutcome::Reject { kind },
        }
    }
}
