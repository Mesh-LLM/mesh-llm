use crate::chat::ChatCompletionRequest;

use super::{
    errors::GuardrailErrorKind,
    policy::{GuardrailMode, StreamingGuardrailMode},
    request_contract::{GuardrailRequestContract, MeshGuardrailsOverride},
    telemetry::GuardrailTelemetryBypassReason,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailRequestState {
    pub model: String,
    pub mode: GuardrailMode,
    pub streaming_mode: StreamingGuardrailMode,
    pub requested_stream: bool,
    pub request_contract: GuardrailRequestContract,
    pub mesh_guardrails_override: MeshGuardrailsOverride,
}

impl GuardrailRequestState {
    pub fn from_request(
        request: &ChatCompletionRequest,
        mode: GuardrailMode,
        streaming_mode: StreamingGuardrailMode,
        request_contract: GuardrailRequestContract,
    ) -> Self {
        Self {
            model: request.model.clone(),
            mode,
            streaming_mode,
            requested_stream: request.stream,
            mesh_guardrails_override: request_contract.mesh_guardrails,
            request_contract,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedGuardrailRequest {
    pub state: GuardrailRequestState,
    pub outcome: GuardrailRequestOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GuardrailRequestOutcome {
    PassThrough {
        reason: GuardrailTelemetryBypassReason,
    },
    Reject {
        kind: GuardrailErrorKind,
    },
    Guarded {
        backend_request: Box<ChatCompletionRequest>,
    },
}
