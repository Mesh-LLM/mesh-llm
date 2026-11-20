//! Copies of Skippy-private request helpers that Mesh Responses needs.
//! Skippy stays unchanged; keep these in sync by behaviour, not by import.

use serde_json::{Map, Value};
use skippy_inference_api::{
    AgentSessionIdentity, AgentSessionSource, InferenceError, common::THINKING_BOOLEAN_ALIASES,
    generation::GenerationRequest,
};

const INTERNAL_AGENT_SESSION_ID: &str = "mesh_internal_agent_session_id";
const INTERNAL_AGENT_SESSION_SOURCE: &str = "mesh_internal_agent_session_source";
const INTERNAL_AGENT_SESSION_TRUSTED: &str = "mesh_internal_agent_session_trusted";

pub(crate) fn set_agent_session(
    request: &mut GenerationRequest,
    identity: Option<AgentSessionIdentity>,
) {
    let extra = &mut request.extra;
    extra.remove(INTERNAL_AGENT_SESSION_ID);
    extra.remove(INTERNAL_AGENT_SESSION_SOURCE);
    extra.remove(INTERNAL_AGENT_SESSION_TRUSTED);
    if let Some(identity) = identity {
        let label = match identity.source() {
            AgentSessionSource::TrustedHeader(header) => header.as_str(),
            AgentSessionSource::ResponsesConversation => "responses.conversation",
        };
        extra.insert(
            INTERNAL_AGENT_SESSION_SOURCE.to_owned(),
            Value::String(label.to_owned()),
        );
        extra.insert(
            INTERNAL_AGENT_SESSION_ID.to_owned(),
            Value::String(identity.id().to_owned()),
        );
    }
}

pub(crate) fn normalize_reasoning_compat_fields(
    object: &mut Map<String, Value>,
) -> Result<bool, InferenceError> {
    let mut enable_thinking = reasoning_object_override(object.get("reasoning"))?;

    if let Some(effort) = optional_string_field(object, "reasoning_effort")? {
        enable_thinking = Some(match effort {
            "none" => false,
            "minimal" | "low" | "medium" | "high" | "xhigh" | "max" => true,
            _ => {
                return Err(InferenceError::invalid_request(
                    "reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max",
                ));
            }
        });
    }

    for field in THINKING_BOOLEAN_ALIASES {
        if let Some(enabled) = optional_bool_field(object, field)? {
            enable_thinking = Some(enabled);
        }
    }
    if optional_u32_field(object, "thinking_budget")? == Some(0) {
        enable_thinking = Some(false);
    }

    if chat_template_kwargs_override(object)? {
        return Ok(false);
    }

    let Some(enabled) = enable_thinking else {
        return Ok(false);
    };
    let kwargs = ensure_chat_template_kwargs_object(object)?;
    kwargs.insert("enable_thinking".to_string(), Value::Bool(enabled));
    Ok(true)
}

fn reasoning_object_override(value: Option<&Value>) -> Result<Option<bool>, InferenceError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| InferenceError::invalid_request("reasoning must be an object"))?;
    let enabled = optional_bool_field(object, "enabled")?;
    let effort = optional_string_field(object, "effort")?;
    let effort_is_valid = match effort {
        Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max") | None => true,
        Some(_) => false,
    };
    if !effort_is_valid {
        return Err(InferenceError::invalid_request(
            "reasoning.effort must be one of none, minimal, low, medium, high, xhigh, max",
        ));
    }
    let max_tokens = optional_u32_field(object, "max_tokens")?;

    if enabled == Some(false) || effort == Some("none") || max_tokens == Some(0) {
        Ok(Some(false))
    } else if enabled == Some(true) || effort.is_some() || max_tokens.is_some() {
        Ok(Some(true))
    } else {
        Ok(None)
    }
}

fn chat_template_kwargs_override(object: &Map<String, Value>) -> Result<bool, InferenceError> {
    let Some(value) = object.get("chat_template_kwargs") else {
        return Ok(false);
    };
    if value.is_null() {
        return Ok(false);
    }
    let kwargs = value
        .as_object()
        .ok_or_else(|| InferenceError::invalid_request("chat_template_kwargs must be an object"))?;
    for field in THINKING_BOOLEAN_ALIASES {
        if optional_bool_field(kwargs, field)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_chat_template_kwargs_object(
    object: &mut Map<String, Value>,
) -> Result<&mut Map<String, Value>, InferenceError> {
    if !object.contains_key("chat_template_kwargs") {
        object.insert(
            "chat_template_kwargs".to_string(),
            Value::Object(Map::new()),
        );
    }
    object
        .get_mut("chat_template_kwargs")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| InferenceError::invalid_request("chat_template_kwargs must be an object"))
}

fn optional_bool_field(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<bool>, InferenceError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                InferenceError::invalid_request(format!("{field} must be a boolean"))
            })
        })
        .transpose()
}

fn optional_string_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, InferenceError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| InferenceError::invalid_request(format!("{field} must be a string")))
        })
        .transpose()
}

fn optional_u32_field(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<u32>, InferenceError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|_| InferenceError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}
