//! The Python object operations `release-notes-regroup.py` applies to a
//! `json.load`ed plan (`dict.get`, `obj[key]`, iteration, hashing and
//! truthiness), each failing with the exception CPython would raise.

use crate::ci_operations::ci_metrics_value::{Value, display, repr};
use crate::prepared_input::python_value::float_repr;
use crate::release::link_gh::{truthy, type_name};
use crate::release::python_failure::Uncaught;
use crate::repository::python_text;

/// `obj.get(key)`: `None` when the key is absent.
pub(crate) fn get<'v>(value: &'v Value, key: &str) -> Result<Option<&'v Value>, Uncaught> {
    match value {
        Value::Object(_) => Ok(value.get(key)),
        other => Err(Uncaught::new(
            "AttributeError",
            format!("'{}' object has no attribute 'get'", type_name(other)),
        )),
    }
}

/// `obj.get(key)` with a present-but-falsy value kept, `None` → `Null`.
pub(crate) fn get_or_null(value: &Value, key: &str) -> Result<Value, Uncaught> {
    Ok(get(value, key)?.cloned().unwrap_or(Value::Null))
}

/// `bool(obj.get(key))`.
pub(crate) fn get_truthy(value: &Value, key: &str) -> Result<bool, Uncaught> {
    Ok(get(value, key)?.is_some_and(truthy))
}

/// `obj[key]` with a string key.
pub(crate) fn item<'v>(value: &'v Value, key: &str) -> Result<&'v Value, Uncaught> {
    let message = match value {
        Value::Object(_) => {
            return value
                .get(key)
                .ok_or_else(|| Uncaught::new("KeyError", python_text::repr(key)));
        }
        Value::Array(_) => "list indices must be integers or slices, not str".to_owned(),
        Value::Str(_) => "string indices must be integers, not 'str'".to_owned(),
        other => format!("'{}' object is not subscriptable", type_name(other)),
    };
    Err(Uncaught::new("TypeError", message))
}

/// `key in obj` for a string key, as `"prs" in section`.
pub(crate) fn contains(value: &Value, key: &str) -> Result<bool, Uncaught> {
    match value {
        Value::Object(_) => Ok(value.get(key).is_some()),
        Value::Str(text) => Ok(text.contains(key)),
        Value::Array(items) => Ok(items
            .iter()
            .any(|item| matches!(item, Value::Str(text) if text == key))),
        other => Err(Uncaught::new(
            "TypeError",
            format!(
                "argument of type '{}' is not a container or iterable",
                type_name(other)
            ),
        )),
    }
}

/// `iter(obj)`: list items, dict keys, or string characters.
pub(crate) fn iterate(value: &Value) -> Result<Vec<Value>, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Object(entries) => Ok(entries.iter().map(|(key, _)| Value::text(key)).collect()),
        Value::Str(text) => Ok(text.chars().map(|ch| Value::Str(ch.to_string())).collect()),
        other => Err(Uncaught::new(
            "TypeError",
            format!("'{}' object is not iterable", type_name(other)),
        )),
    }
}

/// `len(obj)`.
pub(crate) fn length(value: &Value) -> Result<usize, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.len()),
        Value::Object(entries) => Ok(entries.len()),
        Value::Str(text) => Ok(text.chars().count()),
        other => Err(Uncaught::new(
            "TypeError",
            format!("object of type '{}' has no len()", type_name(other)),
        )),
    }
}

/// The equivalence class of `hash`/`==` a set or dict lookup uses:
/// `1 == 1.0 == True`, and the one shared `NaN` equals itself.
pub(crate) fn key(value: &Value) -> Result<String, Uncaught> {
    Ok(match value {
        Value::Null => "none".to_owned(),
        Value::Bool(flag) => format!("int:{}", u8::from(*flag)),
        Value::Int(int) => format!("int:{int}"),
        Value::BigInt(text) => format!("int:{text}"),
        Value::Float(float) if float.is_finite() && float.fract() == 0.0 => {
            format!("int:{float:.0}")
        }
        Value::Float(float) => format!("float:{}", float_repr(*float)),
        Value::Str(text) => format!("str:{text}"),
        other => {
            return Err(Uncaught::new(
                "TypeError",
                format!("unhashable type: '{}'", type_name(other)),
            ));
        }
    })
}

/// The dict key of the pull request `int(digits)`.
pub(crate) fn pr_key(canonical: &str) -> String {
    format!("int:{canonical}")
}

/// `str(obj)` and `repr(obj)`, re-exported for the renderer's messages.
pub(crate) fn text(value: &Value) -> String {
    display(value)
}

pub(crate) fn quoted(value: &Value) -> String {
    repr(value)
}

/// `json.load(open(path))`: the decoded plan or the exception it raises.
pub(crate) fn load_json(path: &str) -> Result<Value, Uncaught> {
    let bytes =
        std::fs::read(path).map_err(|error| Uncaught::os(std::path::Path::new(path), &error))?;
    crate::ci_operations::ci_metrics_value::parse(&bytes).map_err(|message| {
        let class = if message.starts_with("'utf-8' codec") {
            "UnicodeDecodeError"
        } else if message == "maximum recursion depth exceeded" {
            "RecursionError"
        } else {
            "json.decoder.JSONDecodeError"
        };
        Uncaught::new(class, message)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_release_regroup_keys_follow_python_equality() {
        let one = [Value::Int(1), Value::Bool(true), Value::Float(1.0)];
        for value in &one {
            assert_eq!(key(value).ok().as_deref(), Some("int:1"));
        }
        assert_ne!(key(&Value::Float(1.5)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::text("1")).ok(), key(&Value::Int(1)).ok());
        assert!(key(&Value::Array(Vec::new())).is_err());
        assert_eq!(pr_key("7"), "int:7");
    }
}
