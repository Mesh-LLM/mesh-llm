//! Python object behavior over `json.loads` results as the composer uses
//! them: `type()` names, `value[key]`, `str()`/`repr()`, `==`, hashing for
//! a dict lookup, and `json.dumps(value, indent=2, sort_keys=True)`.
//!
//! Values come from `loads_exact`, so an integer beyond 64 bits, an
//! overflowing float and the constants `NaN`/`Infinity`/`-Infinity` arrive
//! as `{EXACT_NUMBER: "<source text>"}` and are treated as numbers here.

use crate::ci_operations::python_json_decode::EXACT_NUMBER;
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::float_repr;
use crate::repository::python_text;
use std::fmt::Write;

/// A Python number decoded from JSON.
enum Number {
    Int(String),
    Float(f64),
}

fn number(value: &Json) -> Option<Number> {
    match value {
        Json::Number(number) if number.is_f64() => number.as_f64().map(Number::Float),
        Json::Number(number) => Some(Number::Int(number.to_string())),
        Json::Object(entries) => match entries.as_slice() {
            [(key, Json::String(text))] if key == EXACT_NUMBER => Some(exact(text)),
            _ => None,
        },
        _ => None,
    }
}

fn exact(text: &str) -> Number {
    match text {
        "NaN" => Number::Float(f64::NAN),
        "Infinity" => Number::Float(f64::INFINITY),
        "-Infinity" => Number::Float(f64::NEG_INFINITY),
        _ if text.contains(['.', 'e', 'E']) => {
            Number::Float(text.parse::<f64>().unwrap_or(f64::NAN))
        }
        _ => Number::Int(text.to_owned()),
    }
}

pub(super) fn type_name(value: &Json) -> &'static str {
    match (value, number(value)) {
        (_, Some(Number::Int(_))) => "int",
        (_, Some(Number::Float(_))) => "float",
        (Json::Null, _) => "NoneType",
        (Json::Bool(_), _) => "bool",
        (Json::String(_), _) => "str",
        (Json::Array(_), _) => "list",
        _ => "dict",
    }
}

pub(super) fn is_dict(value: &Json) -> bool {
    matches!(value, Json::Object(_)) && number(value).is_none()
}

/// `value[key]` with a string key, or the uncaught exception's last line.
pub(super) fn item<'a>(value: &'a Json, key: &str) -> Result<&'a Json, String> {
    match value {
        Json::Object(entries) if is_dict(value) => entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, child)| child)
            .ok_or_else(|| format!("KeyError: {}", python_text::repr(key))),
        Json::String(_) => Err("TypeError: string indices must be integers, not 'str'".to_owned()),
        Json::Array(_) => {
            Err("TypeError: list indices must be integers or slices, not str".to_owned())
        }
        other => Err(format!(
            "TypeError: '{}' object is not subscriptable",
            type_name(other)
        )),
    }
}

/// `dict.get(key)` on a value already known to be a dict.
pub(super) fn get<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    value.get(key)
}

/// `value.removeprefix("v")`.
pub(super) fn remove_v_prefix(value: &Json) -> Result<String, String> {
    match value {
        Json::String(text) => Ok(text.strip_prefix('v').unwrap_or(text).to_owned()),
        other => Err(format!(
            "AttributeError: '{}' object has no attribute 'removeprefix'",
            type_name(other)
        )),
    }
}

/// `hash(value)` succeeds: lists and dicts are unhashable.
pub(super) fn require_hashable(value: &Json) -> Result<(), String> {
    match value {
        Json::Array(_) => Err("TypeError: unhashable type: 'list'".to_owned()),
        _ if is_dict(value) => Err("TypeError: unhashable type: 'dict'".to_owned()),
        _ => Ok(()),
    }
}

/// `str(value)`.
pub(super) fn display(value: &Json) -> String {
    match value {
        Json::String(text) => text.clone(),
        other => repr(other),
    }
}

fn number_repr(number: &Number) -> String {
    match number {
        Number::Int(text) => text.clone(),
        Number::Float(float) => float_repr(*float),
    }
}

/// `repr(value)`.
pub(super) fn repr(value: &Json) -> String {
    if let Some(found) = number(value) {
        return number_repr(&found);
    }
    match value {
        Json::Null => "None".to_owned(),
        Json::Bool(flag) => if *flag { "True" } else { "False" }.to_owned(),
        Json::String(text) => python_text::repr(text),
        Json::Array(items) => {
            let rendered: Vec<String> = items.iter().map(repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        Json::Object(entries) => {
            let rendered: Vec<String> = entries
                .iter()
                .map(|(key, item)| format!("{}: {}", python_text::repr(key), repr(item)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        Json::Number(_) => String::new(),
    }
}

/// Python `==`: `True == 1 == 1.0`, dicts ignore order, `nan != nan`.
pub(super) fn equal(left: &Json, right: &Json) -> bool {
    let numeric = |value: &Json| match value {
        Json::Bool(flag) => Some(Number::Int(u8::from(*flag).to_string())),
        _ => number(value),
    };
    if let (Some(a), Some(b)) = (numeric(left), numeric(right)) {
        return match (a, b) {
            (Number::Int(a), Number::Int(b)) => a == b,
            (Number::Float(a), Number::Float(b)) => a == b,
            (Number::Int(int), Number::Float(float)) | (Number::Float(float), Number::Int(int)) => {
                int.parse::<f64>().is_ok_and(|int| int == float)
            }
        };
    }
    match (left, right) {
        (Json::Null, Json::Null) => true,
        (Json::String(a), Json::String(b)) => a == b,
        (Json::Array(a), Json::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| equal(x, y))
        }
        (Json::Object(a), Json::Object(_)) if is_dict(left) && is_dict(right) => {
            right.as_object().is_some_and(|b| b.len() == a.len())
                && a.iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| equal(value, other)))
        }
        _ => false,
    }
}

/// `json.dumps(value, indent=2, sort_keys=True)` (ASCII-escaped); the
/// caller appends the trailing newline.
pub(super) fn dumps(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out
}

fn write_value(out: &mut String, value: &Json, depth: usize) {
    if let Some(found) = number(value) {
        out.push_str(&match found {
            Number::Float(float) if float.is_nan() => "NaN".to_owned(),
            Number::Float(float) if float.is_infinite() => {
                if float > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
            }
            other => number_repr(&other),
        });
        return;
    }
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Json::String(text) => write_string(out, text),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                separate(out, index, depth + 1);
                write_value(out, item, depth + 1);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(depth));
            out.push(']');
        }
        Json::Object(entries) => {
            let mut sorted: Vec<&(String, Json)> = entries.iter().collect();
            sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                separate(out, index, depth + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(depth));
            out.push('}');
        }
        Json::Number(_) => {}
    }
}

fn separate(out: &mut String, index: usize, depth: usize) {
    if index > 0 {
        out.push(',');
    }
    out.push('\n');
    out.push_str(&"  ".repeat(depth));
}

/// `json.encoder.py_encode_basestring_ascii`.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0_u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _infallible = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}
