use serde_json::Value;

use crate::chat::ChatCompletionRequest;

use super::structured::StructuredOutputSpec;

pub(crate) use mesh_llm_guardrails::{
    MESH_EMIT_STRUCTURED_TOOL_NAME, MESH_RESPOND_TOOL_NAME, is_reserved_tool_name,
    mesh_emit_structured_tool_definition, mesh_respond_tool_definition, model_param_size_b,
    request_uses_reserved_tool_name,
};

pub(crate) fn append_mesh_respond_tool(request: &mut ChatCompletionRequest) {
    let mut tools = match request.tools.take() {
        Some(Value::Array(entries)) => entries,
        Some(other) => {
            request.tools = Some(other);
            return;
        }
        None => Vec::new(),
    };
    tools.push(mesh_respond_tool_definition());
    request.tools = Some(Value::Array(tools));
}

pub(crate) fn append_mesh_emit_structured_tool(
    request: &mut ChatCompletionRequest,
    structured_output: &StructuredOutputSpec,
) {
    let mut tools = match request.tools.take() {
        Some(Value::Array(entries)) => entries,
        Some(other) => {
            request.tools = Some(other);
            return;
        }
        None => Vec::new(),
    };
    tools.push(mesh_emit_structured_tool_definition(structured_output));
    request.tools = Some(Value::Array(tools));
}
