use crate::ci_operations::ci_metrics_value::{Value, display, repr};
use crate::prepared_input::python_value::float_repr;
use crate::release::link_gh::truthy;
use crate::release::python_failure::Uncaught;

/// `obj.get(key)`: `None` when the key is absent.
pub(crate) fn get<'v>(value: &'v Value, key: &str) -> Result<Option<&'v Value>, Uncaught> {
    match value {
        Value::Object(_) => Ok(value.get(key)),
        _ => Err(Uncaught::new("plan", "plan entry must be an object".into())),
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
    get(value, key)?.ok_or_else(|| Uncaught::new("plan", format!("missing plan field {key}")))
}

/// `key in obj` for a string key, as `"prs" in section`.
pub(crate) fn contains(value: &Value, key: &str) -> Result<bool, Uncaught> {
    Ok(get(value, key)?.is_some())
}

/// `iter(obj)`: list items, dict keys, or string characters.
pub(crate) fn iterate(value: &Value) -> Result<Vec<Value>, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        _ => Err(Uncaught::new(
            "plan",
            "plan collection must be an array".into(),
        )),
    }
}

/// `len(obj)`.
pub(crate) fn length(value: &Value) -> Result<usize, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.len()),
        _ => Err(Uncaught::new(
            "plan",
            "plan collection must be an array".into(),
        )),
    }
}

/// The equivalence class of `hash`/`==` a set or dict lookup uses:
/// `1 == 1.0 == True`, and the one shared `NaN` equals itself.
pub(crate) fn key(value: &Value) -> Result<String, Uncaught> {
    Ok(match value {
        Value::Null => "none".to_owned(),
        Value::Bool(flag) => format!("bool:{flag}"),
        Value::Int(int) => format!("int:{int}"),
        Value::BigInt(text) => format!("int:{text}"),
        Value::Float(float) => format!("float:{}", float_repr(*float)),
        Value::Str(text) => format!("str:{text}"),
        _ => {
            return Err(Uncaught::new(
                "plan",
                "pull-request identity must be a scalar".into(),
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
    crate::ci_operations::ci_metrics_value::parse(&bytes)
        .map_err(|message| Uncaught::new("plan", message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_release_regroup_keys_follow_python_equality() {
        assert_eq!(key(&Value::Int(1)).ok().as_deref(), Some("int:1"));
        assert_ne!(key(&Value::Bool(true)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::Float(1.0)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::Float(1.5)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::text("1")).ok(), key(&Value::Int(1)).ok());
        assert!(key(&Value::Array(Vec::new())).is_err());
        assert_eq!(pr_key("7"), "int:7");
    }
}
