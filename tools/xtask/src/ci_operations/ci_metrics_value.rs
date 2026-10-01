//! The Python objects `json.load` gives `collect-ci-metrics.py`: ordered
//! dicts (a repeated key keeps its first position and last value), ints,
//! floats including `NaN`/`Infinity`, and their `str`, `==` and
//! `json.dumps(..., indent=2[, sort_keys=True])` behavior.

use crate::ci_plan::document::Json;
use crate::ci_plan::plan_bytes::write_string;
use crate::prepared_input::python_value::float_repr;
use crate::repository::python_text;

#[derive(Debug, Clone)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Int(i128),
    /// An integer beyond `i128`, as its decimal text.
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

pub(crate) fn parse(raw: &[u8]) -> Result<Value, String> {
    let json = Json::parse(raw).map_err(|error| error.to_string())?;
    Ok(convert(json))
}

fn convert(json: Json) -> Value {
    match json {
        Json::Null => Value::Null,
        Json::Bool(flag) => Value::Bool(flag),
        Json::Number(number) => number
            .as_i64()
            .map(|int| Value::Int(i128::from(int)))
            .or_else(|| number.as_u64().map(|int| Value::Int(i128::from(int))))
            .unwrap_or_else(|| Value::Float(number.as_f64().unwrap_or(f64::NAN))),
        Json::String(text) => Value::Str(text),
        Json::Array(items) => Value::Array(items.into_iter().map(convert).collect()),
        Json::Object(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(key, item)| (key, convert(item)))
                .collect(),
        ),
    }
}

impl Value {
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub(crate) fn text(text: &str) -> Self {
        Value::Str(text.to_owned())
    }

    pub(crate) fn opt_text(text: Option<&str>) -> Self {
        text.map_or(Value::Null, Value::text)
    }

    pub(crate) fn opt_float(value: Option<f64>) -> Self {
        value.map_or(Value::Null, Value::Float)
    }

    pub(crate) fn count(value: usize) -> Self {
        Value::Int(i128::try_from(value).unwrap_or(i128::MAX))
    }
}

/// Builds an object from `(key, value)` pairs.
pub(crate) fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

/// Python `str(value)`.
pub(crate) fn display(value: &Value) -> String {
    match value {
        Value::Str(text) => text.clone(),
        other => repr(other),
    }
}

/// Python `repr(value)`.
pub(crate) fn repr(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(true) => "true".to_owned(),
        Value::Bool(false) => "false".to_owned(),
        Value::Int(int) => int.to_string(),
        Value::BigInt(text) => text.clone(),
        Value::Float(float) => float_repr(*float),
        Value::Str(text) => python_text::repr(text),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(entries) => {
            let entries: Vec<String> = entries
                .iter()
                .map(|(key, item)| format!("{}: {}", python_text::repr(key), repr(item)))
                .collect();
            format!("{{{}}}", entries.join(", "))
        }
    }
}

/// Python `==` (`1 == 1.0 == True`). `NaN` equals itself, as `json`'s
/// shared `NaN` constant is one object and dict lookups check identity.
pub(crate) fn equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Str(a), Value::Str(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, item)| right.get(key).is_some_and(|other| equal(item, other)))
        }
        (Value::BigInt(a), Value::BigInt(b)) => a == b,
        (Value::BigInt(_), _) | (_, Value::BigInt(_)) => false,
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::Int(left), Value::Int(right)) => left == right,
        (Value::Float(left), Value::Float(right)) => left == right,
        _ => false,
    }
}

/// `json.dumps(value, indent=2, sort_keys=sort_keys)`.
pub(crate) fn dumps(value: &Value, sort_keys: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0, sort_keys);
    out
}

fn write_value(out: &mut String, value: &Value, depth: usize, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Int(int) => out.push_str(&int.to_string()),
        Value::BigInt(text) => out.push_str(text),
        Value::Float(float) if float.is_nan() => out.push_str("NaN"),
        Value::Float(float) if float.is_infinite() => {
            out.push_str(if *float > 0.0 {
                "Infinity"
            } else {
                "-Infinity"
            });
        }
        Value::Float(float) => out.push_str(&float_repr(*float)),
        Value::Str(text) => write_string(out, text),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                separator(out, index, depth + 1);
                write_value(out, item, depth + 1, sort_keys);
            }
            close(out, depth, ']');
        }
        Value::Object(entries) => {
            let mut ordered: Vec<&(String, Value)> = entries.iter().collect();
            if sort_keys {
                ordered.sort_by(|(a, _), (b, _)| a.cmp(b));
            }
            out.push('{');
            for (index, (key, item)) in ordered.into_iter().enumerate() {
                separator(out, index, depth + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1, sort_keys);
            }
            close(out, depth, '}');
        }
    }
}

fn separator(out: &mut String, index: usize, depth: usize) {
    if index > 0 {
        out.push(',');
    }
    out.push('\n');
    out.push_str(&"  ".repeat(depth));
}

fn close(out: &mut String, depth: usize, bracket: char) {
    out.push('\n');
    out.push_str(&"  ".repeat(depth));
    out.push(bracket);
}
