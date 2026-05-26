use crate::errors::OpenAiError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailErrorKind {
    ReservedToolName,
    UnsupportedCombination,
    UnsupportedSchemaFeature,
    ValidationFailed,
}

pub const GUARDRAIL_RESERVED_TOOL_NAME_CODE: &str = "guardrail_reserved_tool_name";
pub const GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE: &str = "tool name uses reserved _mesh_ namespace";
pub const GUARDRAIL_UNSUPPORTED_COMBINATION_CODE: &str = "guardrail_unsupported_combination";
pub const GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE: &str =
    "guardrails do not support combining real tools with strict structured output in v1";
pub const GUARDRAIL_VALIDATION_FAILED_CODE: &str = "guardrail_validation_failed";
pub const GUARDRAIL_VALIDATION_FAILED_MESSAGE: &str =
    "model output did not satisfy requested tool/structured contract after retries";
pub const GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_CODE: &str = "guardrail_unsupported_schema_feature";
pub const GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_MESSAGE: &str = "guardrails support only json_object or a json_schema subset using type/properties/required/additionalProperties/items in v1";

pub fn reserved_tool_name_error() -> OpenAiError {
    OpenAiError::invalid_request(GUARDRAIL_RESERVED_TOOL_NAME_MESSAGE)
        .with_code(GUARDRAIL_RESERVED_TOOL_NAME_CODE)
}

pub fn unsupported_combination_error() -> OpenAiError {
    OpenAiError::invalid_request(GUARDRAIL_UNSUPPORTED_COMBINATION_MESSAGE)
        .with_code(GUARDRAIL_UNSUPPORTED_COMBINATION_CODE)
}

pub fn validation_failed_error() -> OpenAiError {
    OpenAiError::invalid_request(GUARDRAIL_VALIDATION_FAILED_MESSAGE)
        .with_code(GUARDRAIL_VALIDATION_FAILED_CODE)
}

pub fn unsupported_schema_feature_error() -> OpenAiError {
    OpenAiError::invalid_request(GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_MESSAGE)
        .with_code(GUARDRAIL_UNSUPPORTED_SCHEMA_FEATURE_CODE)
}

pub(crate) fn guardrail_error(kind: GuardrailErrorKind) -> OpenAiError {
    match kind {
        GuardrailErrorKind::ReservedToolName => reserved_tool_name_error(),
        GuardrailErrorKind::UnsupportedCombination => unsupported_combination_error(),
        GuardrailErrorKind::UnsupportedSchemaFeature => unsupported_schema_feature_error(),
        GuardrailErrorKind::ValidationFailed => validation_failed_error(),
    }
}

pub(crate) fn guardrail_error_catalog() -> [OpenAiError; 4] {
    [
        guardrail_error(GuardrailErrorKind::ReservedToolName),
        guardrail_error(GuardrailErrorKind::UnsupportedCombination),
        guardrail_error(GuardrailErrorKind::UnsupportedSchemaFeature),
        guardrail_error(GuardrailErrorKind::ValidationFailed),
    ]
}
