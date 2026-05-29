use crate::chat::ChatCompletionRequest;

pub use mesh_llm_guardrails::{
    GuardrailRequestContract, MESH_GUARDRAILS_FIELD, MeshGuardrailsOverride, ParallelToolCalls,
    RawToolChoice,
};

#[cfg(test)]
pub(crate) use mesh_llm_guardrails::{RawResponseFormat, RawToolSpec};

pub fn from_request(request: &ChatCompletionRequest) -> GuardrailRequestContract {
    GuardrailRequestContract::from_parts(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
        request.response_format.as_ref(),
        request.extra.get(MESH_GUARDRAILS_FIELD),
    )
}
