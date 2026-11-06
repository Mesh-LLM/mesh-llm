//! Validation shared by owned generation inputs and their wire adapters.

use super::content::{GenerationMessage, messages_to_plain_prompt};
use crate::errors::InferenceError;
use serde_json::Value;

/// Only the fields validation reads, borrowed without copying payloads.
pub(crate) struct RequestValidation<'a> {
    pub(crate) model: &'a str,
    pub(crate) messages: &'a [GenerationMessage],
    pub(crate) max_tokens: Option<u32>,
    pub(crate) max_completion_tokens: Option<u32>,
    pub(crate) n: Option<u32>,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u32>,
    pub(crate) tools: Option<&'a Value>,
    pub(crate) response_format: Option<&'a Value>,
}

impl RequestValidation<'_> {
    pub(crate) fn validate(self) -> Result<(), InferenceError> {
        if self.model.trim().is_empty() {
            return Err(InferenceError::invalid_request("model is required"));
        }
        if self.messages.is_empty() {
            return Err(InferenceError::invalid_request("messages is required"));
        }
        if matches!(self.max_tokens, Some(0)) || matches!(self.max_completion_tokens, Some(0)) {
            return Err(InferenceError::invalid_request(
                "max_tokens must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n == 0) {
            return Err(InferenceError::invalid_request(
                "n must be greater than zero",
            ));
        }
        if self.n.is_some_and(|n| n > 1) {
            return Err(InferenceError::unsupported(
                "n > 1 is parsed but multiple choices are not yet implemented",
            ));
        }
        if self.top_logprobs.is_some() && !self.logprobs.unwrap_or(false) {
            return Err(InferenceError::invalid_request(
                "top_logprobs requires logprobs=true",
            ));
        }
        if let Some(tools) = self.tools {
            validate_tools_value(tools)
                .map_err(|message| InferenceError::invalid_request(message).with_param("tools"))?;
        }
        if self
            .response_format
            .is_some_and(invalid_response_format_value)
        {
            return Err(InferenceError::invalid_request(
                "response_format must be an object with a type field",
            ));
        }
        let prompt_is_empty = messages_to_plain_prompt(self.messages).trim().is_empty();
        let media_can_supply_prompt = crate::hooks::first_chat_media(self.messages).is_some();
        if prompt_is_empty && !media_can_supply_prompt {
            return Err(InferenceError::invalid_request(
                "messages produced an empty prompt",
            ));
        }
        Ok(())
    }
}

fn invalid_response_format_value(value: &Value) -> bool {
    !value
        .as_object()
        .and_then(|object| object.get("type"))
        .is_some_and(Value::is_string)
}

fn validate_tools_value(value: &Value) -> Result<(), String> {
    let Some(tools) = value.as_array() else {
        return Err("tools must be an array".to_string());
    };
    for (index, tool) in tools.iter().enumerate() {
        let Some(tool) = tool.as_object() else {
            return Err(format!("tools[{index}] must be an object"));
        };
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!("tools[{index}].type must be `function`"));
        }
        let Some(function) = tool.get("function").and_then(Value::as_object) else {
            return Err(format!("tools[{index}].function must be an object"));
        };
        if function
            .get("name")
            .and_then(Value::as_str)
            .is_none_or(|name| name.trim().is_empty())
        {
            return Err(format!(
                "tools[{index}].function.name must be a non-empty string"
            ));
        }
        if function
            .get("parameters")
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(format!(
                "tools[{index}].function.parameters must be an object"
            ));
        }
    }
    Ok(())
}
