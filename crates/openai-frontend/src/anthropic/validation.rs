//! Reject unsupported protocol semantics before routing or generation.
use super::protocol::{
    AnthropicContentBlock, AnthropicMessageContent, AnthropicMessagesRequest, AnthropicSystemPrompt,
};
use crate::OpenAiError;
use serde_json::Value;
use std::collections::BTreeMap;

fn reject_fields(
    fields: &BTreeMap<String, Value>,
    allowed: &[&str],
    location: &str,
) -> Result<(), OpenAiError> {
    for key in fields.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(OpenAiError::invalid_request(format!(
                "{location}.{key} is not supported"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_request(request: &AnthropicMessagesRequest) -> Result<(), OpenAiError> {
    if request.max_tokens == 0 {
        return Err(OpenAiError::invalid_request("max_tokens must be positive"));
    }
    // These are existing mesh extensions, retained through the shared chat path.
    reject_fields(
        &request.extra,
        &[
            "mesh_hooks",
            "mesh_guardrails",
            "mesh_agent_session",
            "mesh_agent_session_source",
            "prompt_cache_key",
            "prompt_cache_retention",
            "output_config",
        ],
        "request",
    )?;
    if let Some(config) = request.extra.get("output_config") {
        let object = config
            .as_object()
            .ok_or_else(|| OpenAiError::invalid_request("output_config must be an object"))?;
        for key in object.keys() {
            if key != "effort" && key != "format" {
                return Err(OpenAiError::invalid_request(format!(
                    "output_config.{key} is not supported"
                )));
            }
        }
    }
    if let Some(metadata) = &request.metadata {
        reject_fields(&metadata.extra, &[], "metadata")?;
    }
    if let Some(AnthropicSystemPrompt::Blocks(blocks)) = &request.system {
        for block in blocks {
            if block.kind != "text" || block.text.is_none() {
                return Err(OpenAiError::invalid_request(
                    "system supports text blocks only",
                ));
            }
            reject_fields(&block.extra, &[], "system")?;
        }
    }
    if let Some(tools) = &request.tools {
        for tool in tools {
            reject_fields(&tool.extra, &[], "tools")?;
        }
    }
    if let Some(choice) = &request.tool_choice {
        reject_fields(&choice.extra, &["disable_parallel_tool_use"], "tool_choice")?;
        if choice
            .extra
            .get("disable_parallel_tool_use")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(OpenAiError::invalid_request(
                "disable_parallel_tool_use must be a boolean",
            ));
        }
    }
    for message in &request.messages {
        if let AnthropicMessageContent::Blocks(blocks) = &message.content {
            for block in blocks {
                match block {
                    AnthropicContentBlock::Text(text) => {
                        reject_fields(&text.extra, &[], "text")?;
                        if text.text.is_none() {
                            return Err(OpenAiError::invalid_request("text block requires text"));
                        }
                    }
                    AnthropicContentBlock::ToolUse(tool) => {
                        reject_fields(&tool.extra, &[], "tool_use")?;
                        if message.role != "assistant"
                            || tool.id.is_empty()
                            || tool.name.is_empty()
                            || !tool.input.is_object()
                        {
                            return Err(OpenAiError::invalid_request(
                                "tool_use requires assistant role, id, name and object input",
                            ));
                        }
                    }
                    AnthropicContentBlock::ToolResult(result) => {
                        reject_fields(&result.extra, &[], "tool_result")?;
                        if message.role != "user" || result.tool_use_id.is_empty() {
                            return Err(OpenAiError::invalid_request(
                                "tool_result requires user role and tool_use_id",
                            ));
                        }
                    }
                    AnthropicContentBlock::Other(_) if message.role != "user" => {
                        return Err(OpenAiError::invalid_request(
                            "unsupported assistant content block",
                        ));
                    }
                    AnthropicContentBlock::Other(_) => {}
                }
            }
        }
    }
    Ok(())
}
