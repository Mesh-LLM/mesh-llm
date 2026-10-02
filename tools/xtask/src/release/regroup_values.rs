use crate::ci_operations::ci_metrics_value::{Value, display, repr};
use crate::prepared_input::value_format::float_repr;
use crate::release::command_failure::Uncaught;

/// Optional field lookup on a plan object.
pub(crate) fn get<'v>(value: &'v Value, key: &str) -> Result<Option<&'v Value>, Uncaught> {
    match value {
        Value::Object(_) => Ok(value.get(key)),
        _ => Err(Uncaught::new("plan", "plan entry must be an object".into())),
    }
}

/// Missing fields default to null.
pub(crate) fn get_or_null(value: &Value, key: &str) -> Result<Value, Uncaught> {
    Ok(get(value, key)?.cloned().unwrap_or(Value::Null))
}

/// Whether a plan field is populated.
pub(crate) fn populated_text(value: &Value, key: &str) -> Result<bool, Uncaught> {
    match get(value, key)? {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Str(text)) => Ok(!text.is_empty()),
        Some(_) => Err(Uncaught::new(
            "plan",
            format!("plan {key} must be a string or null"),
        )),
    }
}

pub(crate) fn internal_present(value: &Value) -> Result<bool, Uncaught> {
    match value {
        Value::Null => Ok(false),
        Value::Object(entries) => Ok(!entries.is_empty()),
        _ => Err(Uncaught::new(
            "plan",
            "internal plan must be an object or null".into(),
        )),
    }
}

/// Required plan field lookup.
pub(crate) fn item<'v>(value: &'v Value, key: &str) -> Result<&'v Value, Uncaught> {
    get(value, key)?.ok_or_else(|| Uncaught::new("plan", format!("missing plan field {key}")))
}

/// Whether the plan object has a field.
pub(crate) fn contains(value: &Value, key: &str) -> Result<bool, Uncaught> {
    Ok(get(value, key)?.is_some())
}

/// Array items for a plan collection.
pub(crate) fn iterate(value: &Value) -> Result<Vec<Value>, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        _ => Err(Uncaught::new(
            "plan",
            "plan collection must be an array".into(),
        )),
    }
}

/// Number of items in a plan collection.
pub(crate) fn length(value: &Value) -> Result<usize, Uncaught> {
    match value {
        Value::Array(items) => Ok(items.len()),
        _ => Err(Uncaught::new(
            "plan",
            "plan collection must be an array".into(),
        )),
    }
}

/// Type-distinct scalar identity for pull-request membership.
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

/// Identity of a canonical pull-request number.
pub(crate) fn pr_key(canonical: &str) -> String {
    format!("int:{canonical}")
}

/// Value display for renderer messages.
pub(crate) fn text(value: &Value) -> String {
    display(value)
}

pub(crate) fn quoted(value: &Value) -> String {
    repr(value)
}

/// Read and decode a plan file.
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
    fn pull_request_keys_distinguish_scalar_types() {
        assert_eq!(key(&Value::Int(1)).ok().as_deref(), Some("int:1"));
        assert_ne!(key(&Value::Bool(true)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::Float(1.0)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::Float(1.5)).ok(), key(&Value::Int(1)).ok());
        assert_ne!(key(&Value::text("1")).ok(), key(&Value::Int(1)).ok());
        assert!(key(&Value::Array(Vec::new())).is_err());
        assert_eq!(pr_key("7"), "int:7");
    }
    #[test]
    fn optional_plan_metadata_uses_its_declared_types() {
        assert!(!internal_present(&Value::Null).unwrap());
        assert!(!internal_present(&Value::Object(vec![])).unwrap());
        assert!(internal_present(&Value::Bool(false)).is_err());
        let plan = |value| Value::Object(vec![("version".into(), value)]);
        assert!(!populated_text(&plan(Value::Null), "version").unwrap());
        assert!(!populated_text(&plan(Value::text("")), "version").unwrap());
        assert!(populated_text(&plan(Value::text("1.2.3")), "version").unwrap());
        assert!(populated_text(&plan(Value::Bool(true)), "version").is_err());
    }
}
