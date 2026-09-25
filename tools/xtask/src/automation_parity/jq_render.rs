//! How `jq` renders values in the protected action: `-c` compact output
//! (document key order, raw UTF-8, escaped controls), `-r` raw strings and
//! object constructors with a fixed key order.

use serde_json::Value;
use std::fmt::Write;

/// `jq -c` output: keys in document order, raw UTF-8, escaped controls.
pub(super) fn compact(value: &Value) -> String {
    let mut out = String::new();
    write_compact(&mut out, value);
    out
}

fn write_compact(out: &mut String, value: &Value) {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => out.push_str(&value.to_string()),
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            let parts = items.iter().map(compact).collect::<Vec<_>>();
            let _ = write!(out, "[{}]", parts.join(","));
        }
        Value::Object(entries) => {
            let parts = entries
                .iter()
                .map(|(key, item)| {
                    let mut part = String::new();
                    write_string(&mut part, key);
                    part.push(':');
                    write_compact(&mut part, item);
                    part
                })
                .collect::<Vec<_>>();
            let _ = write!(out, "{{{}}}", parts.join(","));
        }
    }
}

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
            '\u{0}'..='\u{1f}' | '\u{7f}' => {
                let _ = write!(out, "\\u{:04x}", u32::from(ch));
            }
            _ => out.push(ch),
        }
    }
    out.push('"');
}

/// `jq -r`: strings raw, everything else compact.
pub(super) fn raw(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => compact(value),
    }
}

/// An object literal in the given key order, as a jq object constructor.
pub(super) fn object(pairs: &[(&str, String)]) -> String {
    let parts = pairs
        .iter()
        .map(|(key, value)| format!("{}:{value}", compact(&Value::from(*key))))
        .collect::<Vec<_>>();
    format!("{{{}}}", parts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migration_ci_shadow_compact_matches_jq_escaping_and_order() {
        let value: Value =
            serde_json::from_str(r#"{"b":1,"a":"x\u007f\u001f\u00e9\n"}"#).unwrap_or(Value::Null);
        assert_eq!(compact(&value), "{\"a\":\"x\\u007f\\u001fé\\n\",\"b\":1}");
        assert_eq!(raw(&json!("text")), "text");
        assert_eq!(
            object(&[("z", "1".to_owned()), ("a", "2".to_owned())]),
            "{\"z\":1,\"a\":2}"
        );
    }
}
