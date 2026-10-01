use crate::ci_plan::document::Json;
use crate::prepared_input::python_json;

/// Failure text: `str(error)` of the exception the legacy tool caught.
pub(crate) type Outcome<T> = Result<T, String>;

pub(crate) fn type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Number(number) if number.is_f64() => "float",
        Json::Number(_) => "int",
        Json::String(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

/// `value[key]` with a string key.
pub(crate) fn item<'a>(value: &'a Json, key: &str) -> Outcome<&'a Json> {
    value
        .get(key)
        .ok_or_else(|| format!("missing object field {key}"))
}

/// `value[key]` where the key is itself a parsed value (a dict lookup).
pub(crate) fn item_by<'a>(value: &'a Json, key: &Json) -> Outcome<&'a Json> {
    match key {
        Json::String(text) => item(value, text),
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::Array(_) | Json::Object(_) => {
            Err("object key must be a string".into())
        }
    }
}

/// `key in mapping` for a dict of string keys.
pub(crate) fn contains_key(mapping: &Json, key: &Json) -> Outcome<bool> {
    match key {
        Json::String(text) => Ok(mapping.get(text).is_some()),
        Json::Array(_) | Json::Object(_) => Err(unhashable(key)),
        _ => Ok(false),
    }
}

pub(crate) fn unhashable(key: &Json) -> String {
    format!("object key must be a string, got {}", type_name(key))
}

/// Python `==` between two parsed values.
pub(crate) fn eq(left: &Json, right: &Json) -> bool {
    python_json::equal(left, right)
}

/// Python `value == "text"`.
pub(crate) fn is_str(value: &Json, text: &str) -> bool {
    value.as_str() == Some(text)
}

/// `type(value) is int and value == 1`.
pub(crate) fn is_int_one(value: &Json) -> bool {
    value.as_int() == Some(1)
}

/// `isinstance(value, dict) and set(value) == set(names.split())`.
pub(crate) fn has_exact_fields(value: &Json, names: &str) -> bool {
    let Some(entries) = value.as_object() else {
        return false;
    };
    let expected: Vec<&str> = names.split(' ').collect();
    entries.len() == expected.len()
        && entries
            .iter()
            .all(|(key, _)| expected.contains(&key.as_str()))
}

pub(crate) fn object(pairs: &[(&str, Json)]) -> Json {
    Json::Object(
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
    )
}

pub(crate) fn string(text: &str) -> Json {
    Json::String(text.to_owned())
}

/// `require(condition, message)` raising the caller's exception type.
pub(crate) fn require(condition: bool, message: impl FnOnce() -> String) -> Outcome<()> {
    if condition { Ok(()) } else { Err(message()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Json {
        Json::parse(text.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn migration_ci_operations_subscript_errors_match_python() {
        let value = parsed(r#"{"a": 1, "s": "x", "l": [], "n": null}"#);
        assert_eq!(item(&value, "b"), Err("missing object field b".to_owned()));
        let nested = |key: &str| item(item(&value, key).expect("present"), "k");
        assert_eq!(nested("s"), Err("missing object field k".to_owned()));
        assert_eq!(nested("l"), Err("missing object field k".to_owned()));
        assert_eq!(nested("n"), Err("missing object field k".to_owned()));
        assert_eq!(nested("a"), Err("missing object field k".to_owned()));
        assert_eq!(
            contains_key(&value, &parsed("[]")),
            Err("object key must be a string, got array".to_owned())
        );
        assert!(has_exact_fields(&value, "n l s a"));
    }
}
