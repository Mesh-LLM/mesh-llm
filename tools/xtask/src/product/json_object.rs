use crate::ci_plan::document::Json;

pub(super) fn type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Number(_) => "number",
        Json::String(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

pub(super) fn is_dict(value: &Json) -> bool {
    matches!(value, Json::Object(_))
}

pub(super) fn item<'a>(value: &'a Json, key: &str) -> Result<&'a Json, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing object field {key}"))
}

pub(super) fn get<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    value.get(key)
}

pub(super) fn require_hashable(value: &Json) -> Result<(), String> {
    match value {
        Json::Array(_) | Json::Object(_) => Err("backend must be a scalar value".into()),
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => Ok(()),
    }
}

pub(super) fn display(value: &Json) -> String {
    match value {
        Json::String(text) => text.clone(),
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::Array(_) | Json::Object(_) => {
            value.to_value().to_string()
        }
    }
}

pub(super) fn equal(left: &Json, right: &Json) -> bool {
    left.to_value() == right.to_value()
}

pub(super) fn dumps(value: &Json) -> String {
    crate::prepared_input::json_bytes::dumps_indented(value)
}
