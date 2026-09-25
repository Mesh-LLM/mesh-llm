//! The planner's stdout bytes: Python `json.dumps(plan, separators=(",",
//! ":"), sort_keys=True)` with the default `ensure_ascii`. `jq -c .` of these
//! bytes is what the protected caller hashes.

use serde_json::Value;
use std::fmt::Write;

pub(crate) fn render(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            let mut sorted = entries.iter().collect::<Vec<_>>();
            sorted.sort_by_key(|(key, _)| *key);
            out.push('{');
            for (index, (key, item)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(out, key);
                out.push(':');
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// `ensure_ascii` escaping: named escapes for `\" \\ \n \r \t \b \f`,
/// `\uXXXX` for other controls and every non-ASCII UTF-16 code unit.
pub(crate) fn write_string(out: &mut String, text: &str) {
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
            ' '..='~' | '\u{7f}' => out.push(ch),
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

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn migration_ci_plan_bytes_match_python_json_dumps() {
        let value = json!({"b": ["\u{1}\u{8}\t\n\u{c}\r\u{7f}\"\\é😀"], "a": {"z": 1, "y": true}});
        assert_eq!(
            render(&value),
            r#"{"a":{"y":true,"z":1},"b":["\u0001\b\t\n\f\r"#.to_owned()
                + "\u{7f}"
                + r#"\"\\\u00e9\ud83d\ude00"]}"#
        );
    }
}
