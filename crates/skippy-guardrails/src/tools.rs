use serde_json::{Map, Value, json};
use std::fmt;

use crate::{request_contract::GuardrailRequestContract, structured::StructuredOutputSpec};

pub const MESH_RESPOND_TOOL_NAME: &str = "_mesh_respond";
pub const MESH_EMIT_STRUCTURED_TOOL_NAME: &str = "_mesh_emit_structured";

pub fn request_uses_reserved_tool_name(
    request: &GuardrailRequestContract,
    reserved_prefix: &str,
) -> bool {
    request
        .tool_names()
        .any(|name| is_reserved_tool_name(name, reserved_prefix))
        || request
            .forced_tool_name()
            .is_some_and(|name| is_reserved_tool_name(name, reserved_prefix))
}

pub fn is_reserved_tool_name(name: &str, reserved_prefix: &str) -> bool {
    name.starts_with(reserved_prefix)
}

pub fn model_param_size_b(name: &str) -> Option<f32> {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        if c == b'0' {
            continue;
        }

        let mut end = i + 1;
        while let Some(&next) = bytes.get(end) {
            if next.is_ascii_digit() || next == b'.' {
                end += 1;
                continue;
            }
            break;
        }

        let Some(&unit) = bytes.get(end) else {
            continue;
        };
        if unit != b'b' && unit != b'B' {
            continue;
        }
        if let Some(&after) = bytes.get(end + 1)
            && after.is_ascii_digit()
        {
            continue;
        }

        let number = std::str::from_utf8(&bytes[i..end])
            .ok()?
            .parse::<f32>()
            .ok()?;
        if number > 0.0 {
            return Some(number);
        }
    }
    None
}

pub fn mesh_respond_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": MESH_RESPOND_TOOL_NAME,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }
        }
    })
}

pub fn mesh_emit_structured_tool_definition(structured_output: &StructuredOutputSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
            "parameters": structured_output.tool_parameters()
        }
    })
}

pub fn extract_tool_name_and_arguments(value: &Value) -> Option<(&str, &Value)> {
    let object = value.as_object()?;
    let nested_function = object.get("function").and_then(Value::as_object);
    let name = nested_function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| object.get("name").and_then(Value::as_str))
        .or_else(|| object.get("function").and_then(Value::as_str))
        .or_else(|| object.get("tool").and_then(Value::as_str))?;
    let arguments = nested_function
        .and_then(|function| function.get("arguments"))
        .or_else(|| object.get("arguments"))?;
    Some((name, arguments))
}

pub fn normalize_tool_arguments(arguments: &Value) -> Option<Map<String, Value>> {
    match arguments {
        Value::Object(arguments) => Some(arguments.clone()),
        Value::String(arguments) => serde_json::from_str::<Value>(arguments)
            .ok()?
            .as_object()
            .cloned(),
        Value::Null => None,
        _ => Some(Map::new()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolArgumentSchemaError {
    MissingRequired {
        tool_name: String,
        fields: Vec<String>,
    },
}

impl fmt::Display for ToolArgumentSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequired { tool_name, fields } => {
                write!(
                    f,
                    "tool {tool_name:?} missing required argument(s): {}",
                    fields.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for ToolArgumentSchemaError {}

pub fn sanitize_tool_arguments_for_tool(
    tool_name: &str,
    arguments: &Value,
    tools: Option<&Value>,
) -> Result<Value, ToolArgumentSchemaError> {
    let mut arguments = normalize_tool_arguments(arguments)
        .map(Value::Object)
        .unwrap_or_else(|| json!({}));

    let Some(parameters) = tool_parameters(tool_name, tools) else {
        return Ok(arguments);
    };

    sanitize_object_for_schema(&mut arguments, parameters);
    ensure_required_arguments(tool_name, &arguments, parameters)?;
    Ok(arguments)
}

fn tool_parameters<'a>(tool_name: &str, tools: Option<&'a Value>) -> Option<&'a Value> {
    tools?
        .as_array()?
        .iter()
        .find(|tool| {
            tool.pointer("/function/name")
                .and_then(Value::as_str)
                .is_some_and(|name| name == tool_name)
        })?
        .pointer("/function/parameters")
}

fn sanitize_object_for_schema(arguments: &mut Value, schema: &Value) {
    let Some(arguments) = arguments.as_object_mut() else {
        return;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let allow_additional = matches!(schema.get("additionalProperties"), Some(Value::Bool(true)))
        || schema
            .get("additionalProperties")
            .is_some_and(Value::is_object);

    arguments.retain(|key, value| {
        let Some(property_schema) = properties.get(key) else {
            return allow_additional;
        };
        argument_value_matches_schema(value, property_schema)
    });
}

fn argument_value_matches_schema(value: &Value, schema: &Value) -> bool {
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|allowed| allowed == value)
    {
        return false;
    }

    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    let types: Vec<&str> = match schema_type {
        Value::String(t) => vec![t.as_str()],
        Value::Array(types) => types.iter().filter_map(Value::as_str).collect(),
        _ => return true,
    };
    types
        .iter()
        .any(|schema_type| value_matches_type(value, schema_type))
}

fn value_matches_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn ensure_required_arguments(
    tool_name: &str,
    arguments: &Value,
    schema: &Value,
) -> Result<(), ToolArgumentSchemaError> {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return Ok(());
    };
    let Some(arguments) = arguments.as_object() else {
        return Ok(());
    };
    let missing: Vec<String> = required
        .iter()
        .filter_map(Value::as_str)
        .filter(|field| !arguments.contains_key(*field))
        .map(str::to_string)
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(ToolArgumentSchemaError::MissingRequired {
            tool_name: tool_name.to_string(),
            fields: missing,
        })
    }
}

pub fn tool_arguments_wire_string(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => serde_json::from_str::<Value>(value)
            .ok()
            .filter(Value::is_object)
            .map_or_else(|| "{}".to_string(), |_| value.clone()),
        Value::Object(_) => serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
        Value::Null => "{}".to_string(),
        _ => "{}".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::*;

    #[test]
    fn normalize_tool_arguments_handles_null_string_and_primitive_inputs() {
        let object = json!({"path": "README.md"});
        assert_eq!(
            normalize_tool_arguments(&object).unwrap()["path"],
            "README.md"
        );

        let string = Value::String("{\"path\":\"README.md\"}".to_string());
        assert_eq!(
            normalize_tool_arguments(&string).unwrap()["path"],
            "README.md"
        );

        assert_eq!(normalize_tool_arguments(&Value::Null), None);
        assert_eq!(normalize_tool_arguments(&Value::from(42)), Some(Map::new()));
    }

    #[test]
    fn tool_arguments_wire_string_always_returns_object_json() {
        assert_eq!(tool_arguments_wire_string(&Value::Null), "{}");
        assert_eq!(tool_arguments_wire_string(&Value::from(42)), "{}");
        assert_eq!(
            tool_arguments_wire_string(&Value::String("not json".into())),
            "{}"
        );
        assert_eq!(
            tool_arguments_wire_string(&json!({"path": "README.md"})),
            "{\"path\":\"README.md\"}"
        );
    }

    #[test]
    fn schema_sanitizer_removes_unknown_and_invalid_arguments() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "exec",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "host": {"type": "string", "enum": ["gateway"]}
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "exec",
            &json!({
                "command": "echo ok",
                "host": "sandbox",
                "extra": true
            }),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"command": "echo ok"}));
    }

    #[test]
    fn schema_sanitizer_rejects_missing_required_after_cleanup() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }]);

        let err = sanitize_tool_arguments_for_tool(
            "read_file",
            &json!({"path": 42, "other": "x"}),
            Some(&tools),
        )
        .unwrap_err();

        assert_eq!(
            err,
            ToolArgumentSchemaError::MissingRequired {
                tool_name: "read_file".into(),
                fields: vec!["path".into()]
            }
        );
    }

    #[test]
    fn schema_sanitizer_preserves_additional_properties_when_schema_allows_them() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "kv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fixed": {"type": "string"}
                    },
                    "additionalProperties": true
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "kv",
            &json!({"fixed": "a", "dynamic": 1}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"fixed": "a", "dynamic": 1}));
    }
}
