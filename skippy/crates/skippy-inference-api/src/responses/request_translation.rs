//! Translation from the Responses request shape to the shared Chat
//! Completions generation boundary.

use serde_json::{Map, Value};
use skippy_guardrails::MESH_COMPACT_FIELD;

use super::content_items::translate_responses_content_item;
use super::conversation::{is_responses_tool_item, translate_responses_conversation_items};
use super::{request_validation, store, stream, tools, turn};
use crate::errors::InferenceError;

pub(super) fn alias_chat_max_tokens(object: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    for alias in ["max_completion_tokens", "max_output_tokens"] {
        let Some(value) = object.remove(alias) else {
            continue;
        };
        changed = true;
        object.entry("max_tokens".to_string()).or_insert(value);
    }
    changed
}

fn map_response_role(role: &str) -> String {
    match role {
        "developer" => "system".to_string(),
        other => other.to_string(),
    }
}

fn collapse_blocks_if_text_only(blocks: Vec<Value>) -> Value {
    if blocks.len() == 1
        && let Some(text) = blocks[0].get("text").and_then(Value::as_str)
    {
        return Value::String(text.to_string());
    }
    Value::Array(blocks)
}

pub(super) fn translate_responses_message_content(
    content: &Value,
) -> Result<Value, InferenceError> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(items) => {
            let blocks = items
                .iter()
                .map(translate_responses_content_item)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(collapse_blocks_if_text_only(blocks))
        }
        Value::Object(_) => Ok(collapse_blocks_if_text_only(vec![
            translate_responses_content_item(content)?,
        ])),
        _ => Err(InferenceError::unsupported(
            "unsupported /v1/responses input content shape",
        )),
    }
}

pub(super) fn translate_responses_input_message(
    message: &Value,
) -> Result<Map<String, Value>, InferenceError> {
    let Some(object) = message.as_object() else {
        return Err(InferenceError::unsupported(
            "unsupported /v1/responses message shape",
        ));
    };

    let role = map_response_role(object.get("role").and_then(Value::as_str).unwrap_or("user"));
    let content_value = object
        .get("content")
        .map(translate_responses_message_content)
        .transpose()?
        .unwrap_or_else(|| Value::String(String::new()));

    let mut translated = Map::from_iter([
        ("role".to_string(), Value::String(role)),
        ("content".to_string(), content_value),
    ]);
    if let Some(phase) = object.get("phase") {
        let valid = object.get("role").and_then(Value::as_str) == Some("assistant")
            && matches!(phase.as_str(), Some("commentary" | "final_answer"));
        if !valid {
            return Err(InferenceError::invalid_request(
                "phase is only valid on assistant messages and must be \"commentary\" or \"final_answer\"",
            )
            .with_param("input"));
        }
        translated.insert("phase".into(), phase.clone());
    }
    Ok(translated)
}

pub(super) fn translate_responses_input_to_messages(
    input: &Value,
) -> Result<Vec<Value>, InferenceError> {
    match input {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => Ok(vec![serde_json::json!({
            "role": "user",
            "content": text,
        })]),
        Value::Array(items) if items.iter().any(is_responses_tool_item) => {
            super::lineage::validate(items)?;
            translate_responses_conversation_items(items)
        }
        Value::Array(items) => {
            let looks_like_messages = items.iter().all(|item| {
                item.as_object()
                    .map(|object| object.contains_key("role") || object.contains_key("content"))
                    .unwrap_or(false)
            });
            if looks_like_messages {
                items
                    .iter()
                    .map(translate_responses_input_message)
                    .map(|result| result.map(Value::Object))
                    .collect()
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        Value::Object(_) if is_responses_tool_item(input) => {
            super::lineage::validate(std::slice::from_ref(input))?;
            translate_responses_conversation_items(std::slice::from_ref(input))
        }
        Value::Object(object) => {
            if object.contains_key("role") || object.contains_key("content") {
                Ok(vec![Value::Object(translate_responses_input_message(
                    input,
                )?)])
            } else {
                let content = translate_responses_message_content(input)?;
                Ok(vec![serde_json::json!({
                    "role": "user",
                    "content": content,
                })])
            }
        }
        _ => Err(InferenceError::unsupported(
            "unsupported /v1/responses input shape",
        )),
    }
}

pub(super) fn translate_openai_responses_input(
    object: &mut Map<String, Value>,
    policy_enforced: bool,
) -> Result<bool, InferenceError> {
    let stripped_obfuscation = object
        .get("stream_options")
        .and_then(Value::as_object)
        .is_some_and(|options| options.contains_key("include_obfuscation"));
    stream::obfuscation::ObfuscationPolicy::capture_and_strip(object)?;
    let normalized_tools = tools::normalize(object, policy_enforced)?;
    let mut changed = stripped_obfuscation || normalized_tools;
    let mut messages = Vec::new();
    let mut state_cache_key = None;

    if let Some(instructions_value) = object.remove("instructions") {
        if let Some(instructions) = instructions_value.as_str().map(str::trim)
            && !instructions.is_empty()
        {
            messages.push(serde_json::json!({
                "role": "system",
                "content": instructions,
            }));
        }
        changed = true;
    }

    if let Some(input) = turn::take_effective_input(object) {
        messages.extend(translate_responses_input_to_messages(&input)?);
        changed = true;
    }

    if !messages.is_empty() {
        object.insert("messages".to_string(), Value::Array(messages));
    }

    // Only a prepared `ResponsesTurn` can restore a previous response; a
    // reference that reaches translation unresolved fails closed.
    if let Some(id) = object
        .get("previous_response_id")
        .filter(|id| !id.is_null())
    {
        return Err(store::previous_response_not_found(
            id.as_str().unwrap_or_default(),
        ));
    }
    request_validation::validate_request_fields(object)?;
    if object.insert(MESH_COMPACT_FIELD.into(), Value::Bool(false)) != Some(Value::Bool(false)) {
        changed = true;
    }
    if let Some(value) = object.get("conversation") {
        state_cache_key = responses_conversation_id(value);
    }
    if let Some(cache_key) = state_cache_key {
        object
            .entry("prompt_cache_key".to_string())
            .or_insert(Value::String(cache_key));
    }
    if let Some(format) = object.get("text").and_then(|text| text.get("format"))
        && let Some(response_format) = chat_response_format(format)?
    {
        object
            .entry("response_format".to_string())
            .or_insert(response_format);
    }
    let mut wants_logprobs = object
        .get("top_logprobs")
        .is_some_and(|value| !value.is_null());
    if let Some(include) = object.get("include").cloned() {
        let include = include.as_array().ok_or_else(|| {
            InferenceError::invalid_request("include must be an array").with_param("include")
        })?;
        for value in include {
            match value.as_str() {
                Some("message.output_text.logprobs") => {
                    wants_logprobs = true;
                }
                Some("reasoning.encrypted_content") => {
                    return Err(InferenceError::unsupported(
                        "encrypted reasoning content is not supported",
                    )
                    .with_param("include"));
                }
                _ => {
                    return Err(InferenceError::invalid_request(
                        "include contains an unknown value",
                    )
                    .with_param("include"));
                }
            }
        }
    }
    if wants_logprobs
        && object.insert("logprobs".into(), Value::Bool(true)) != Some(Value::Bool(true))
    {
        changed = true;
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        // Responses always report usage, so ask the chat backend for it.
        let options = object
            .entry("stream_options".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if options.is_null() {
            *options = Value::Object(Map::new());
        }
        if let Some(options) = options.as_object_mut() {
            options.insert("include_usage".to_string(), Value::Bool(true));
        }
        changed = true;
    }

    for key in [
        "background",
        "conversation",
        "include",
        "max_tool_calls",
        "output",
        "output_text",
        "previous_response_id",
        "store",
        "text",
        "truncation",
    ] {
        if object.remove(key).is_some() {
            changed = true;
        }
    }

    Ok(changed)
}

pub(super) fn responses_conversation_id(value: &Value) -> Option<String> {
    if let Some(id) = value.as_str() {
        return Some(id.to_string());
    }
    let object = value.as_object()?;
    for key in ["id", "conversation_id"] {
        if let Some(id) = object.get(key).and_then(Value::as_str) {
            return Some(id.to_string());
        }
    }
    None
}

/// The Chat Completions `response_format` for a Responses `text.format`.
fn chat_response_format(format: &Value) -> Result<Option<Value>, InferenceError> {
    let invalid = |message: &str| {
        InferenceError::invalid_request(message.to_string()).with_param("text.format")
    };
    match format.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(serde_json::json!({"type": "json_object"}))),
        Some("json_schema") => {
            let name = format
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("json_schema format requires a name"))?;
            let schema = format
                .get("schema")
                .ok_or_else(|| invalid("json_schema format requires a schema"))?;
            let mut json_schema = serde_json::json!({"name": name, "schema": schema});
            for key in ["description", "strict"] {
                if let Some(value) = format.get(key) {
                    json_schema[key] = value.clone();
                }
            }
            Ok(Some(
                serde_json::json!({"type": "json_schema", "json_schema": json_schema}),
            ))
        }
        Some(other) => Err(InferenceError::unsupported(format!(
            "unsupported text.format type {other:?}"
        ))
        .with_param("text.format")),
    }
}
