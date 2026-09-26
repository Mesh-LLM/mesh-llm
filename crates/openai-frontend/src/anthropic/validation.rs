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
            "thinking",
            "context_management",
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
    if let Some(thinking) = request.extra.get("thinking") {
        validate_thinking(thinking)?;
    }
    if let Some(context_management) = request.extra.get("context_management") {
        validate_context_management(context_management)?;
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
            validate_cache_control_fields(&block.extra, "system")?;
        }
    }
    if let Some(tools) = &request.tools {
        for (index, tool) in tools.iter().enumerate() {
            validate_cache_control_fields(&tool.extra, "tools")?;
            if tool
                .input_schema
                .as_ref()
                .is_none_or(|schema| !schema.is_object())
            {
                return Err(OpenAiError::invalid_request(format!(
                    "tools[{index}].input_schema must be an object"
                )));
            }
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
                        validate_cache_control_fields(&text.extra, "text")?;
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
                    AnthropicContentBlock::Other(value) if message.role == "assistant" => {
                        validate_assistant_opaque_block(value)?;
                    }
                    AnthropicContentBlock::Other(_) => {}
                }
            }
        }
    }
    Ok(())
}

fn validate_cache_control_fields(
    fields: &BTreeMap<String, Value>,
    location: &str,
) -> Result<(), OpenAiError> {
    reject_fields(fields, &["cache_control"], location)?;
    if let Some(cache_control) = fields.get("cache_control") {
        validate_cache_control(cache_control, location)?;
    }
    Ok(())
}

fn validate_cache_control(value: &Value, location: &str) -> Result<(), OpenAiError> {
    let object = value.as_object().ok_or_else(|| {
        OpenAiError::invalid_request(format!("{location}.cache_control must be an object"))
    })?;
    for key in object.keys() {
        if key != "type" && key != "ttl" {
            return Err(OpenAiError::invalid_request(format!(
                "{location}.cache_control.{key} is not supported"
            )));
        }
    }
    if object.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return Err(OpenAiError::invalid_request(format!(
            "{location}.cache_control.type must be `ephemeral`"
        )));
    }
    if let Some(ttl) = object.get("ttl").and_then(Value::as_str)
        && ttl != "5m"
        && ttl != "1h"
    {
        return Err(OpenAiError::invalid_request(format!(
            "{location}.cache_control.ttl must be `5m` or `1h`"
        )));
    }
    if object.get("ttl").is_some_and(|ttl| !ttl.is_string()) {
        return Err(OpenAiError::invalid_request(format!(
            "{location}.cache_control.ttl must be a string"
        )));
    }
    Ok(())
}

fn validate_thinking(value: &Value) -> Result<(), OpenAiError> {
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("thinking must be an object"))?;
    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        OpenAiError::invalid_request("thinking.type must be `adaptive`, `enabled`, or `disabled`")
    })?;
    match kind {
        "adaptive" => {
            reject_json_fields(object, &["type", "display"], "thinking")?;
            if let Some(display) = object.get("display").and_then(Value::as_str)
                && display != "omitted"
                && display != "summarized"
            {
                return Err(OpenAiError::invalid_request(
                    "thinking.display must be `omitted` or `summarized`",
                ));
            }
            if object
                .get("display")
                .is_some_and(|display| !display.is_string())
            {
                return Err(OpenAiError::invalid_request(
                    "thinking.display must be a string",
                ));
            }
        }
        "enabled" => {
            reject_json_fields(object, &["type", "budget_tokens"], "thinking")?;
            let budget = object
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    OpenAiError::invalid_request(
                        "thinking.budget_tokens must be a positive integer",
                    )
                })?;
            if budget == 0 || budget > u32::MAX as u64 {
                return Err(OpenAiError::invalid_request(
                    "thinking.budget_tokens must be a positive integer",
                ));
            }
        }
        "disabled" => reject_json_fields(object, &["type"], "thinking")?,
        _ => {
            return Err(OpenAiError::invalid_request(
                "thinking.type must be `adaptive`, `enabled`, or `disabled`",
            ));
        }
    }
    Ok(())
}

fn validate_context_management(value: &Value) -> Result<(), OpenAiError> {
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("context_management must be an object"))?;
    reject_json_fields(object, &["edits"], "context_management")?;
    let edits = object
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| OpenAiError::invalid_request("context_management.edits must be an array"))?;
    for (index, edit) in edits.iter().enumerate() {
        let edit = edit.as_object().ok_or_else(|| {
            OpenAiError::invalid_request(format!(
                "context_management.edits[{index}] must be an object"
            ))
        })?;
        reject_json_fields(
            edit,
            &["type", "keep"],
            &format!("context_management.edits[{index}]"),
        )?;
        if edit.get("type").and_then(Value::as_str) != Some("clear_thinking_20251015")
            || edit.get("keep").and_then(Value::as_str) != Some("all")
        {
            return Err(OpenAiError::invalid_request(format!(
                "context_management.edits[{index}] must be clear_thinking_20251015 with keep `all`"
            )));
        }
    }
    Ok(())
}

fn validate_assistant_opaque_block(value: &Value) -> Result<(), OpenAiError> {
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("unsupported assistant content block"))?;
    match object.get("type").and_then(Value::as_str) {
        Some("thinking") => {
            reject_json_fields(object, &["type", "thinking", "signature"], "thinking block")?;
            if !object.get("thinking").is_some_and(Value::is_string)
                || !object.get("signature").is_some_and(Value::is_string)
            {
                return Err(OpenAiError::invalid_request(
                    "thinking block requires thinking and signature strings",
                ));
            }
        }
        Some("redacted_thinking") => {
            reject_json_fields(object, &["type", "data"], "redacted_thinking block")?;
            if !object.get("data").is_some_and(Value::is_string) {
                return Err(OpenAiError::invalid_request(
                    "redacted_thinking block requires a data string",
                ));
            }
        }
        _ => {
            return Err(OpenAiError::invalid_request(
                "unsupported assistant content block",
            ));
        }
    }
    Ok(())
}

fn reject_json_fields(
    fields: &serde_json::Map<String, Value>,
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
