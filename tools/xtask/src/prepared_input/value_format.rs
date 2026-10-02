use crate::ci_plan::document::Json;

pub(crate) fn repr(value: Option<&Json>) -> String {
    value.map_or_else(
        || "missing".to_owned(),
        |value| value.to_value().to_string(),
    )
}

pub(crate) fn display(value: Option<&Json>) -> String {
    match value {
        Some(Json::String(text)) => text.clone(),
        other => repr(other),
    }
}

pub(crate) fn float_repr(value: f64) -> String {
    serde_json::Number::from_f64(value)
        .map_or_else(|| value.to_string(), |number| number.to_string())
}
