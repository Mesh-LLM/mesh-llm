//! Python `repr` and `str` of parsed JSON values, for diagnostics that
//! interpolate `{value!r}` or `{value}` from a manifest.

use crate::ci_plan::document::Json;
use crate::repository::python_text;
use serde_json::Number;

/// Python `repr(value)` of `json.loads` output; `None` for a missing key.
pub(crate) fn repr(value: Option<&Json>) -> String {
    let Some(value) = value else {
        return "None".to_owned();
    };
    match value {
        Json::Null => "None".to_owned(),
        Json::Bool(true) => "True".to_owned(),
        Json::Bool(false) => "False".to_owned(),
        Json::Number(number) => number_repr(number),
        Json::String(text) => python_text::repr(text),
        Json::Array(items) => {
            let rendered: Vec<String> = items.iter().map(|item| repr(Some(item))).collect();
            format!("[{}]", rendered.join(", "))
        }
        Json::Object(entries) => {
            let rendered: Vec<String> = entries
                .iter()
                .map(|(key, item)| format!("{}: {}", python_text::repr(key), repr(Some(item))))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

/// Python `str(value)`: strings print bare, everything else as `repr`.
pub(crate) fn display(value: Option<&Json>) -> String {
    match value {
        Some(Json::String(text)) => text.clone(),
        other => repr(other),
    }
}

fn number_repr(number: &Number) -> String {
    if let Some(int) = number.as_i64() {
        return int.to_string();
    }
    if let Some(int) = number.as_u64() {
        return int.to_string();
    }
    number
        .as_f64()
        .map_or_else(|| number.to_string(), float_repr)
}

/// Python `float.__repr__`: shortest round-trip digits, positional between
/// 1e-4 and 1e16, otherwise `d.ddde+XX` with a signed two-digit exponent.
fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if value != 0.0 && !(-4..16).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let positional = value.to_string();
    if positional.contains('.') {
        positional
    } else {
        format!("{positional}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> Json {
        Json::parse(text.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn migration_prepared_inputs_repr_matches_python() {
        let value = parsed(r#"["a'b", true, null, 1.5, 1e20, 1e-7, 3.0, -2, {"k": "v"}]"#);
        assert_eq!(
            repr(Some(&value)),
            r#"["a'b", True, None, 1.5, 1e+20, 1e-07, 3.0, -2, {'k': 'v'}]"#
        );
        assert_eq!(repr(None), "None");
        assert_eq!(display(Some(&parsed("\"x\""))), "x");
        assert_eq!(display(Some(&parsed("7"))), "7");
    }
}
