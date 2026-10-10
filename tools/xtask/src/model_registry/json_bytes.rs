//! Stable order-preserving registry bytes consumed by generated-file checks.
//! Layout, ASCII escaping, and decimal spelling are output contracts.

use crate::ci_plan::document::Json;
use serde_json::Number;
use std::fmt::Write;

/// Layout and escaping for a consumed JSON document.
#[derive(Clone, Copy)]
pub(crate) struct Style {
    pub(crate) indent: Option<usize>,
    pub(crate) item_separator: &'static str,
    pub(crate) key_separator: &'static str,
    pub(crate) ensure_ascii: bool,
}

/// Two-space indentation with literal Unicode.
pub(super) const PRETTY: Style = Style {
    indent: Some(2),
    item_separator: ",",
    key_separator: ": ",
    ensure_ascii: false,
};

/// Single-line JSON with spaced separators and literal Unicode.
pub(crate) const INLINE: Style = Style {
    indent: None,
    item_separator: ", ",
    key_separator: ": ",
    ensure_ascii: false,
};

/// Single-line JSON with ASCII escaping.
pub(crate) const ASCII: Style = Style {
    ensure_ascii: true,
    ..INLINE
};

/// Compact ASCII-escaped step-output JSON.
pub(super) const ASCII_COMPACT: Style = Style {
    item_separator: ",",
    key_separator: ":",
    ..ASCII
};

pub(crate) fn dumps(value: &Json, style: Style) -> String {
    let mut out = String::new();
    write_value(&mut out, value, style, 0);
    out
}

fn newline(out: &mut String, style: Style, depth: usize) {
    if let Some(indent) = style.indent {
        out.push('\n');
        out.push_str(&" ".repeat(indent * depth));
    }
}

fn write_value(out: &mut String, value: &Json, style: Style, depth: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Json::Number(number) => out.push_str(&number_text(number)),
        Json::String(text) => write_string(out, text, style.ensure_ascii),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                separate(out, style, depth, index);
                write_value(out, item, style, depth + 1);
            }
            newline(out, style, depth);
            out.push(']');
        }
        Json::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                separate(out, style, depth, index);
                write_string(out, key, style.ensure_ascii);
                out.push_str(style.key_separator);
                write_value(out, item, style, depth + 1);
            }
            newline(out, style, depth);
            out.push('}');
        }
    }
}

fn separate(out: &mut String, style: Style, depth: usize, index: usize) {
    if index > 0 {
        out.push_str(style.item_separator);
    }
    newline(out, style, depth + 1);
}

/// Preserve the registry's decimal output spelling.
fn number_text(number: &Number) -> String {
    match number.as_f64().filter(|_| number.is_f64()) {
        Some(float) => decimal_text(float),
        None => number.to_string(),
    }
}

/// Shortest round-trip digits, positional notation for
/// decimal exponents in `-4..16`, otherwise `d.ddde+XX`.
fn decimal_text(value: f64) -> String {
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent = exponent.parse::<i32>().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let body = match usize::try_from(point) {
            Ok(point) if point >= digits.len() => {
                format!("{digits}{}.0", "0".repeat(point - digits.len()))
            }
            Ok(point) if point > 0 => format!("{}.{}", &digits[..point], &digits[point..]),
            _ => format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize)),
        };
        return format!("{sign}{body}");
    }
    let exponent_sign = if exponent < 0 { '-' } else { '+' };
    format!(
        "{sign}{mantissa}e{exponent_sign}{:02}",
        exponent.unsigned_abs()
    )
}

/// JSON string escapes; `ensure_ascii` adds `\uXXXX` for every
/// non-ASCII UTF-16 unit, including DEL.
fn write_string(out: &mut String, text: &str, ensure_ascii: bool) {
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
            _ if ch < ' ' || ensure_ascii => {
                let mut units = [0_u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _infallible = write!(out, "\\u{unit:04x}");
                }
            }
            _ => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        Json::parse(text.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn registry_decimal_bytes_preserve_notation() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1.5e300, "1.5e+300"),
            (-2.5e-7, "-2.5e-07"),
            (123_456_789_012_345_680.0, "1.2345678901234568e+17"),
        ];
        for (value, expected) in cases {
            assert_eq!(decimal_text(value), expected, "{value}");
        }
    }

    #[test]
    fn registry_bytes_preserve_indentation() {
        let value = parse(r#"{"a":[1,{"b":[]}],"c":{},"d":"é\u2028\u007f","e":2.0}"#);
        assert_eq!(
            dumps(&value, PRETTY),
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": []\n    }\n  ],\n  \"c\": {},\n  \"d\": \"é\u{2028}\u{7f}\",\n  \"e\": 2.0\n}"
        );
    }

    #[test]
    fn step_output_bytes_preserve_escaping() {
        let value = parse(r#"{"k":["é","\u007f\u0001"]}"#);
        assert_eq!(dumps(&value, ASCII), r#"{"k": ["\u00e9", "\u007f\u0001"]}"#);
        assert_eq!(
            dumps(&value, ASCII_COMPACT),
            r#"{"k":["\u00e9","\u007f\u0001"]}"#
        );
        assert_eq!(dumps(&value, INLINE), "{\"k\": [\"é\", \"\u{7f}\\u0001\"]}");
    }
}
