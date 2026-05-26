use serde_json::{Map, Value, json};

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
        if let Some(&after) = bytes.get(end + 1) {
            if after.is_ascii_digit() {
                continue;
            }
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
}
