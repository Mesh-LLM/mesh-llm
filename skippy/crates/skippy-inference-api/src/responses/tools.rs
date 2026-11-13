//! Responses function-tool validation, backend normalization, and output policy.
//!
//! The ingress captures [`ToolPolicy`] before the Responses request is
//! translated to Chat Completions. The backend still receives every declared
//! tool schema; only an `allowed_tools` choice is reduced to its selection mode.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use super::resource::ResponseStatus;
use crate::InferenceError;

pub(super) const MAX_TOOL_CALLS: usize = 128;
pub(super) const MAX_TOOL_NAME_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolMode {
    Auto,
    None,
    Required,
}

impl ToolMode {
    fn parse(value: &str, param: &'static str) -> Result<Self, InferenceError> {
        match value {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "required" => Ok(Self::Required),
            _ => Err(InferenceError::invalid_request(
                "tool_choice mode must be \"auto\", \"none\", or \"required\"",
            )
            .with_param(param)),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Required => "required",
        }
    }
}

/// The exact tool-call authorization owned by the original Responses ingress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ToolPolicy {
    mode: ToolMode,
    defined: BTreeSet<String>,
    allowed: BTreeSet<String>,
    max_calls: Option<u64>,
}

impl ToolPolicy {
    pub(super) fn capture(request: &Map<String, Value>) -> Result<Self, InferenceError> {
        let defined = declared_tool_names(request.get("tools"))?;
        let (mode, allowed) = choice_policy(request.get("tool_choice"), &defined)?;
        let max_calls = max_tool_calls(request.get("max_tool_calls"))?;
        if mode == ToolMode::Required && allowed.is_empty() {
            return Err(InferenceError::invalid_request(
                "required tool_choice needs at least one declared tool",
            )
            .with_param("tool_choice"));
        }
        Ok(Self {
            mode,
            defined,
            allowed,
            max_calls,
        })
    }

    pub(super) fn validate_calls<'a>(
        &self,
        status: ResponseStatus,
        had_call_fragments: bool,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ToolOutputViolation> {
        let names = names.into_iter().collect::<Vec<_>>();
        if names.len() > MAX_TOOL_CALLS {
            return Err(ToolOutputViolation::new(
                "tool_call_limit_exceeded",
                "upstream response exceeded the tool-call limit",
            ));
        }
        if self
            .max_calls
            .is_some_and(|max_calls| names.len() as u64 > max_calls)
        {
            return Err(ToolOutputViolation::new(
                "tool_call_limit_exceeded",
                "upstream response exceeded max_tool_calls",
            ));
        }
        if self.mode == ToolMode::None && had_call_fragments {
            return Err(ToolOutputViolation::new(
                "tool_choice_violation",
                "upstream response called a tool while tool_choice was none",
            ));
        }
        for name in &names {
            validate_output_name(name)?;
            if !self.defined.contains(*name) || !self.allowed.contains(*name) {
                return Err(ToolOutputViolation::new(
                    "tool_choice_violation",
                    "upstream response called a tool that was not allowed by the request",
                ));
            }
        }
        if status == ResponseStatus::Completed
            && self.mode == ToolMode::Required
            && names.is_empty()
        {
            return Err(ToolOutputViolation::new(
                "tool_choice_violation",
                "upstream completed without a required tool call",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolOutputViolation {
    pub(super) code: &'static str,
    pub(super) message: &'static str,
}

impl ToolOutputViolation {
    pub(super) const fn new(code: &'static str, message: &'static str) -> Self {
        Self { code, message }
    }
}

pub(super) fn normalize(request: &mut Map<String, Value>) -> Result<bool, InferenceError> {
    // Revalidate at this public normalization boundary as well as when a turn
    // captures its policy. Some compatibility callers normalize without a turn.
    let policy = ToolPolicy::capture(request)?;
    if request
        .get("max_tool_calls")
        .is_some_and(|value| !value.is_null())
    {
        return Err(InferenceError::unsupported(
            "max_tool_calls requires a prepared Responses turn",
        )
        .with_param("max_tool_calls"));
    }
    let mut changed = false;
    if let Some(tools) = request.get_mut("tools")
        && !tools.is_null()
    {
        for tool in tools.as_array_mut().expect("capture validated tools") {
            changed |= normalize_function(tool, "tools")?;
        }
    }
    if request
        .get("tool_choice")
        .and_then(Value::as_object)
        .and_then(|choice| choice.get("type"))
        .and_then(Value::as_str)
        == Some("allowed_tools")
    {
        request.insert(
            "tool_choice".into(),
            Value::String(policy.mode.as_str().into()),
        );
        changed = true;
    } else if let Some(choice) = request.get_mut("tool_choice")
        && choice.is_object()
    {
        changed |= normalize_function(choice, "tool_choice")?;
    }
    Ok(changed)
}

fn max_tool_calls(value: Option<&Value>) -> Result<Option<u64>, InferenceError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_u64()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or_else(|| {
            InferenceError::invalid_request("max_tool_calls must be a positive integer")
                .with_param("max_tool_calls")
        })
}

fn declared_tool_names(value: Option<&Value>) -> Result<BTreeSet<String>, InferenceError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(BTreeSet::new());
    };
    let tools = value.as_array().ok_or_else(|| {
        InferenceError::invalid_request("tools must be an array").with_param("tools")
    })?;
    if tools.len() > MAX_TOOL_CALLS {
        return Err(
            InferenceError::invalid_request("tools may contain at most 128 entries")
                .with_param("tools"),
        );
    }
    let mut names = BTreeSet::new();
    for tool in tools {
        let object = function_object(tool, "tools")?;
        let name = required_name(object, "tools")?;
        if !names.insert(name.to_string()) {
            return Err(InferenceError::invalid_request(
                "tools must not contain duplicate function names",
            )
            .with_param("tools"));
        }
    }
    Ok(names)
}

fn choice_policy(
    choice: Option<&Value>,
    defined: &BTreeSet<String>,
) -> Result<(ToolMode, BTreeSet<String>), InferenceError> {
    let Some(choice) = choice.filter(|choice| !choice.is_null()) else {
        return Ok((ToolMode::Auto, defined.clone()));
    };
    if let Some(mode) = choice.as_str() {
        return Ok((ToolMode::parse(mode, "tool_choice")?, defined.clone()));
    }
    let object = choice.as_object().ok_or_else(|| {
        InferenceError::invalid_request("tool_choice must be a string or object")
            .with_param("tool_choice")
    })?;
    match object.get("type").and_then(Value::as_str) {
        Some("function") => {
            let function = function_object(choice, "tool_choice")?;
            let name = required_name(function, "tool_choice")?;
            require_defined(name, defined)?;
            Ok((ToolMode::Required, BTreeSet::from([name.to_string()])))
        }
        Some("allowed_tools") => allowed_choice(object, defined),
        _ => Err(
            InferenceError::unsupported("only function tools are supported")
                .with_param("tool_choice"),
        ),
    }
}

fn allowed_choice(
    choice: &Map<String, Value>,
    defined: &BTreeSet<String>,
) -> Result<(ToolMode, BTreeSet<String>), InferenceError> {
    let tools = choice
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            InferenceError::invalid_request("allowed_tools.tools must be a non-empty array")
                .with_param("tool_choice.tools")
        })?;
    if tools.is_empty() || tools.len() > MAX_TOOL_CALLS {
        return Err(InferenceError::invalid_request(
            "allowed_tools.tools must contain between 1 and 128 entries",
        )
        .with_param("tool_choice.tools"));
    }
    let mut allowed = BTreeSet::new();
    for tool in tools {
        let function = function_object(tool, "tool_choice.tools")?;
        let name = required_name(function, "tool_choice.tools")?;
        require_defined(name, defined)?;
        if !allowed.insert(name.to_string()) {
            return Err(InferenceError::invalid_request(
                "allowed_tools.tools must not contain duplicates",
            )
            .with_param("tool_choice.tools"));
        }
    }
    let mode = choice
        .get("mode")
        .map(|mode| {
            mode.as_str()
                .ok_or_else(|| {
                    InferenceError::invalid_request("allowed_tools.mode must be a string")
                        .with_param("tool_choice.mode")
                })
                .and_then(|mode| ToolMode::parse(mode, "tool_choice.mode"))
        })
        .transpose()?
        .unwrap_or(ToolMode::Auto);
    Ok((mode, allowed))
}

fn require_defined(name: &str, defined: &BTreeSet<String>) -> Result<(), InferenceError> {
    if defined.contains(name) {
        Ok(())
    } else {
        Err(InferenceError::invalid_request(
            "tool_choice references a function that is not present in tools",
        )
        .with_param("tool_choice"))
    }
}

fn function_object<'a>(
    value: &'a Value,
    param: &'static str,
) -> Result<&'a Map<String, Value>, InferenceError> {
    let object = value.as_object().ok_or_else(|| {
        InferenceError::invalid_request("function tool must be an object").with_param(param)
    })?;
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(
            InferenceError::unsupported("only function tools are supported").with_param(param),
        );
    }
    match (object.get("function"), object.get("name")) {
        (Some(function), None) => function.as_object().ok_or_else(|| {
            InferenceError::invalid_request("function must be an object").with_param(param)
        }),
        (Some(_), Some(_)) => Err(InferenceError::invalid_request(
            "ambiguous flat and nested function tool",
        )
        .with_param(param)),
        (None, _) => Ok(object),
    }
}

fn required_name<'a>(
    function: &'a Map<String, Value>,
    param: &'static str,
) -> Result<&'a str, InferenceError> {
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            InferenceError::invalid_request(missing_function_name(param)).with_param(param)
        })?;
    validate_name(name)
        .map_err(|message| InferenceError::invalid_request(message).with_param(param))?;
    Ok(name)
}

fn validate_name(name: &str) -> Result<(), &'static str> {
    if name.is_empty() || name.len() > MAX_TOOL_NAME_BYTES {
        return Err("function name must contain between 1 and 64 bytes");
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("function name may contain only letters, digits, underscores, and hyphens");
    }
    Ok(())
}

fn validate_output_name(name: &str) -> Result<(), ToolOutputViolation> {
    validate_name(name).map_err(|_| {
        ToolOutputViolation::new(
            "upstream_tool_call_invalid",
            "upstream response contained an invalid function name",
        )
    })
}

fn normalize_function(value: &mut Value, param: &'static str) -> Result<bool, InferenceError> {
    let object = value
        .as_object_mut()
        .expect("capture validated function object");
    // Retain the legacy whitespace-padding normalization, applying the same
    // canonical name used by policy capture to both supported request shapes.
    if let Some(function) = object.get_mut("function") {
        let function = function
            .as_object_mut()
            .expect("capture validated function");
        let name = required_name(function, param)?.to_owned();
        let changed = function["name"].as_str() != Some(name.as_str());
        function.insert("name".into(), Value::String(name));
        return Ok(changed);
    }
    let name = required_name(object, param)?.to_owned();
    let mut function = std::mem::take(object);
    function.remove("type");
    function.insert("name".into(), Value::String(name));
    for field in ["description", "parameters", "strict"] {
        if function.get(field).is_some_and(Value::is_null) {
            function.remove(field);
        }
    }
    object.insert("type".into(), Value::String("function".into()));
    object.insert("function".into(), Value::Object(function));
    let _ = param;
    Ok(true)
}

fn missing_function_name(param: &str) -> &'static str {
    if param.starts_with("tool_choice") {
        "tool_choice must reference a function by name"
    } else {
        "function name must be a non-empty string"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nullable_resource_fields_can_be_reused_as_request_tools() {
        let mut request = json!({"tools": [{
            "type": "function", "name": "weather", "description": null,
            "parameters": null, "strict": null
        }]});
        normalize(request.as_object_mut().unwrap()).unwrap();
        assert_eq!(
            request["tools"][0],
            json!({
                "type": "function", "function": {"name": "weather"}
            })
        );
    }

    #[test]
    fn allowed_tools_keeps_all_schemas_and_normalizes_only_the_mode() {
        let mut request = json!({
            "tools": [
                {"type":"function","name":"safe","parameters":{"type":"object"}},
                {"type":"function","name":"admin","description":"retained"}
            ],
            "tool_choice": {"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ]}
        });
        let policy = ToolPolicy::capture(request.as_object().unwrap()).unwrap();
        normalize(request.as_object_mut().unwrap()).unwrap();
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["tools"].as_array().unwrap().len(), 2);
        assert_eq!(request["tools"][1]["function"]["description"], "retained");
        assert!(
            policy
                .validate_calls(ResponseStatus::Completed, true, ["safe"])
                .is_ok()
        );
        assert!(
            policy
                .validate_calls(ResponseStatus::Completed, true, ["admin"])
                .is_err()
        );
    }

    #[test]
    fn identifiers_references_and_duplicates_are_validated_uniformly() {
        for mut request in [
            json!({"tools":[{"type":"function","name":"bad name"}]}),
            json!({"tools":[{"type":"function","function":{"name":"bad space"}}]}),
            json!({"tools":[{"type":"function","name":"same"},{"type":"function","name":"same"}]}),
            json!({"tools":[{"type":"function","name":"safe"}],"tool_choice":{"type":"allowed_tools","tools":[{"type":"function","name":"missing"}]}}),
        ] {
            assert!(normalize(request.as_object_mut().unwrap()).is_err());
        }
    }

    #[test]
    fn required_is_checked_only_for_completed_outputs() {
        let policy = ToolPolicy::capture(
            json!({"tools":[{"type":"function","name":"safe"}],"tool_choice":{"type":"allowed_tools","tools":[{"type":"function","name":"safe"}],"mode":"required"}})
                .as_object()
                .unwrap(),
        )
        .unwrap();
        assert!(
            policy
                .validate_calls(ResponseStatus::Incomplete("max_output_tokens"), false, [])
                .is_ok()
        );
        assert!(
            policy
                .validate_calls(ResponseStatus::Completed, false, [])
                .is_err()
        );
    }

    #[test]
    fn max_tool_calls_accepts_null_and_the_complete_u64_range() {
        for value in [
            json!(null),
            json!(1),
            json!(128),
            json!(129),
            json!(u64::MAX),
        ] {
            let request = json!({"max_tool_calls": value});
            ToolPolicy::capture(request.as_object().unwrap()).unwrap();
        }
    }

    #[test]
    fn max_tool_calls_rejects_non_positive_or_non_u64_json_values() {
        for value in [
            json!(0),
            json!(-1),
            json!(1.0),
            json!(1.5),
            json!(true),
            json!("1"),
            json!([]),
            json!({"value": 1}),
        ] {
            let request = json!({"max_tool_calls": value});
            let error = ToolPolicy::capture(request.as_object().unwrap()).unwrap_err();
            assert_eq!(error.body().error.param.as_deref(), Some("max_tool_calls"));
            assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
        }
        let too_large: Value =
            serde_json::from_str(r#"{"max_tool_calls":18446744073709551616}"#).unwrap();
        assert!(ToolPolicy::capture(too_large.as_object().unwrap()).is_err());
    }

    #[test]
    fn requested_cap_counts_repeated_names_and_does_not_clamp_large_values() {
        let request = json!({
            "tools": [{"type": "function", "name": "same"}],
            "max_tool_calls": 2,
        });
        let policy = ToolPolicy::capture(request.as_object().unwrap()).unwrap();
        assert!(
            policy
                .validate_calls(ResponseStatus::Completed, true, ["same", "same"])
                .is_ok()
        );
        let violation = policy
            .validate_calls(ResponseStatus::Completed, true, ["same", "same", "same"])
            .unwrap_err();
        assert_eq!(violation.code, "tool_call_limit_exceeded");

        let request = json!({
            "tools": [{"type": "function", "name": "same"}],
            "max_tool_calls": u64::MAX,
        });
        let policy = ToolPolicy::capture(request.as_object().unwrap()).unwrap();
        let violation = policy
            .validate_calls(
                ResponseStatus::Completed,
                true,
                std::iter::repeat_n("same", MAX_TOOL_CALLS + 1),
            )
            .unwrap_err();
        assert_eq!(violation.code, "tool_call_limit_exceeded");
    }
}
