//! Responses function tools are flat on the wire; shared chat tools are nested.
use crate::OpenAiError;
use serde_json::{Map, Value};

pub(super) fn normalize(request: &mut Map<String, Value>) -> Result<bool, OpenAiError> {
    let mut changed = false;
    if let Some(tools) = request.get_mut("tools") {
        let tools = tools.as_array_mut().ok_or_else(|| {
            OpenAiError::invalid_request("tools must be an array").with_param("tools")
        })?;
        for tool in tools {
            changed |= normalize_function(tool, "tools")?;
        }
    }
    if let Some(choice) = request.get_mut("tool_choice")
        && choice.is_object()
    {
        changed |= normalize_function(choice, "tool_choice")?;
    }
    Ok(changed)
}

fn normalize_function(value: &mut Value, param: &str) -> Result<bool, OpenAiError> {
    let object = value.as_object_mut().ok_or_else(|| {
        OpenAiError::invalid_request("function tool must be an object").with_param(param)
    })?;
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(
            OpenAiError::unsupported("only function tools are supported").with_param(param),
        );
    }
    // Retain the already-supported nested Chat shape for existing clients.
    if object.contains_key("function") {
        if object.contains_key("name") {
            return Err(
                OpenAiError::invalid_request("ambiguous flat and nested function tool")
                    .with_param(param),
            );
        }
        return Ok(false);
    }
    // Validate the trimmed name and store it trimmed, so whitespace padding
    // cannot pass validation and leak into the shared Chat request.
    let name = match object.get("name").and_then(Value::as_str).map(str::trim) {
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => {
            return Err(
                OpenAiError::invalid_request(missing_function_name(param)).with_param(param)
            );
        }
    };
    let mut function = std::mem::take(object);
    function.remove("type");
    function.insert("name".into(), Value::String(name));
    object.insert("type".into(), Value::String("function".into()));
    object.insert("function".into(), Value::Object(function));
    Ok(true)
}

fn missing_function_name(param: &str) -> &'static str {
    if param == "tool_choice" {
        "tool_choice must reference a function by name"
    } else {
        "function name must be a non-empty string"
    }
}
