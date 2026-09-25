//! The `json` module behavior prepared-input manifests depend on: reading a
//! UTF-8 JSON file, Python `==` between loaded values, and
//! `json.dumps(value, indent=2, sort_keys=True)` bytes.

use crate::ci_plan::document::Json;
use crate::ci_plan::plan_bytes::write_string;
use std::path::Path;

/// Reads and parses a JSON file. Errors carry serde_json's wording, not
/// Python's `JSONDecodeError`; callers keep the legacy status.
pub(crate) fn load(path: &Path) -> Result<Json, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
    Json::parse(text.as_bytes()).map_err(|error| error.to_string())
}

/// Python `==` between two `json.loads` results: numbers compare by value
/// (`1 == 1.0`, `True == 1`), objects ignore key order.
pub(crate) fn equal(left: &Json, right: &Json) -> bool {
    match (left, right) {
        (Json::Null, Json::Null) => true,
        (Json::String(a), Json::String(b)) => a == b,
        (Json::Array(a), Json::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| equal(x, y))
        }
        (Json::Object(a), Json::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| equal(value, other)))
        }
        _ => match (numeric(left), numeric(right)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
    }
}

/// Numeric value of a JSON number or bool, as Python compares them.
fn numeric(value: &Json) -> Option<f64> {
    match value {
        Json::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Json::Number(number) => number.as_f64(),
        _ => None,
    }
}

/// `json.dumps(value, indent=2, sort_keys=True)`; the caller appends the
/// trailing newline, like the legacy writers.
pub(crate) fn dumps_indented(value: &Json) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out
}

fn write_value(out: &mut String, value: &Json, depth: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Json::Number(number) => out.push_str(&number.to_string()),
        Json::String(text) => write_string(out, text),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                separator(out, index, depth + 1);
                write_value(out, item, depth + 1);
            }
            close(out, depth, ']');
        }
        Json::Object(entries) => {
            let mut sorted: Vec<&(String, Json)> = entries.iter().collect();
            sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                separator(out, index, depth + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Json {
        Json::parse(text.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn migration_prepared_inputs_dumps_matches_python_indent_sort_keys() {
        let value = parsed(r#"{"b": [1, {}], "a": {"é": []}}"#);
        assert_eq!(
            dumps_indented(&value),
            "{\n  \"a\": {\n    \"\\u00e9\": []\n  },\n  \"b\": [\n    1,\n    {}\n  ]\n}"
        );
    }

    #[test]
    fn migration_prepared_inputs_equality_follows_python() {
        assert!(equal(
            &parsed(r#"{"a": 1, "b": true}"#),
            &parsed(r#"{"b": 1.0, "a": 1.0}"#)
        ));
        assert!(!equal(&parsed(r#"{"a": 1}"#), &parsed(r#"{"a": "1"}"#)));
        assert!(!equal(
            &parsed(r#"{"a": 1}"#),
            &parsed(r#"{"a": 1, "b": 2}"#)
        ));
    }
}
