//! Prepared-input JSON loading, structural comparison, and stable output bytes.

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

pub(crate) fn equal(left: &Json, right: &Json) -> bool {
    left.to_value() == right.to_value()
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
    fn structural_comparison_ignores_object_key_order() {
        assert!(equal(
            &parsed(r#"{"a": 1, "b": true}"#),
            &parsed(r#"{"b": true, "a": 1}"#)
        ));
    }

    #[test]
    fn structural_comparison_rejects_different_json_types() {
        assert!(!equal(&parsed("true"), &parsed("1")));
        assert!(!equal(&parsed("1"), &parsed("1.0")));
        assert!(!equal(&parsed(r#"{"a": 1}"#), &parsed(r#"{"a": "1"}"#)));
        assert!(!equal(
            &parsed(r#"{"a": 1}"#),
            &parsed(r#"{"a": 1, "b": 2}"#)
        ));
    }

    #[test]
    fn structural_comparison_preserves_integer_precision() {
        assert!(!equal(
            &parsed("9007199254740992"),
            &parsed("9007199254740993")
        ));
    }
}
