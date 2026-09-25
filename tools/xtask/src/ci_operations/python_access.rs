//! Python's subscript, membership and error wording over `json.loads`
//! values, so a malformed catalog reports the same `KeyError`/`TypeError`
//! text the legacy tool printed after its `runner image identity: ` prefix.

use crate::ci_plan::document::Json;
use crate::prepared_input::python_json;
use crate::repository::python_text;

/// Failure text: `str(error)` of the exception the legacy tool caught.
pub(crate) type Outcome<T> = Result<T, String>;

pub(crate) fn type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Number(number) if number.is_f64() => "float",
        Json::Number(_) => "int",
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// `value[key]` with a string key.
pub(crate) fn item<'a>(value: &'a Json, key: &str) -> Outcome<&'a Json> {
    match value {
        Json::Object(entries) => entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, child)| child)
            .ok_or_else(|| python_text::repr(key)),
        Json::String(_) => Err("string indices must be integers, not 'str'".to_owned()),
        Json::Array(_) => Err("list indices must be integers or slices, not str".to_owned()),
        other => Err(format!(
            "'{}' object is not subscriptable",
            type_name(other)
        )),
    }
}

/// `value[key]` where the key is itself a parsed value (a dict lookup).
pub(crate) fn item_by<'a>(value: &'a Json, key: &Json) -> Outcome<&'a Json> {
    match key {
        Json::String(text) => item(value, text),
        _ if !matches!(value, Json::Object(_)) => item(value, ""),
        Json::Array(_) | Json::Object(_) => Err(unhashable(key)),
        other => Err(python_value_repr(other)),
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
    format!("unhashable type: '{}'", type_name(key))
}

fn python_value_repr(value: &Json) -> String {
    crate::prepared_input::python_value::repr(Some(value))
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
        assert_eq!(item(&value, "b"), Err("'b'".to_owned()));
        let nested = |key: &str| item(item(&value, key).expect("present"), "k");
        assert_eq!(
            nested("s"),
            Err("string indices must be integers, not 'str'".to_owned())
        );
        assert_eq!(
            nested("l"),
            Err("list indices must be integers or slices, not str".to_owned())
        );
        assert_eq!(
            nested("n"),
            Err("'NoneType' object is not subscriptable".to_owned())
        );
        assert_eq!(
            nested("a"),
            Err("'int' object is not subscriptable".to_owned())
        );
        assert_eq!(
            contains_key(&value, &parsed("[]")),
            Err("unhashable type: 'list'".to_owned())
        );
        assert!(has_exact_fields(&value, "n l s a"));
    }
}
