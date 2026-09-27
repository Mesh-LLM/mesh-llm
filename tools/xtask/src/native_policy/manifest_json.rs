//! `json.loads(path.read_text(encoding="utf-8"))` and the dict subscripts
//! `select-native-runtime.py` applies to a runtime manifest. Any failure is
//! an exception the legacy script leaves uncaught, reported as the final
//! line of its traceback.

use crate::artifact::os_error_line;
use crate::ci_operations::python_access::type_name;
use crate::ci_operations::python_json_decode::{DecodeError, EXACT_NUMBER, Hooks, loads_exact};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value;
use crate::repository::python_text::repr;
use std::path::Path;

/// The last traceback line of an uncaught exception (`Class: message`).
pub(super) struct Raised(pub(super) String);

/// Marks `NaN`/`Infinity`/`-Infinity`, which Python loads as floats.
const CONSTANT: &str = "\u{0}constant:";

/// `object_pairs_hook` equivalent of a plain `dict`: a repeated key keeps
/// its first position and its last value.
fn dict(pairs: Vec<(String, Json)>) -> Result<Json, String> {
    let mut entries: Vec<(String, Json)> = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        match entries.iter_mut().find(|(name, _)| *name == key) {
            Some(entry) => entry.1 = value,
            None => entries.push((key, value)),
        }
    }
    Ok(Json::Object(entries))
}

fn constant(name: &str) -> Result<Json, String> {
    Ok(Json::String(format!("{CONSTANT}{name}")))
}

pub(super) fn load_manifest(path: &Path, shown: &str) -> Result<Json, Raised> {
    let raw = std::fs::read(path).map_err(|error| Raised(os_error_line(&error, shown)))?;
    let hooks = Hooks {
        pairs: dict,
        constant,
    };
    if std::str::from_utf8(&raw).is_err() {
        return Err(match loads_exact(&raw, &hooks) {
            Err(DecodeError::Value(message)) => Raised(format!("UnicodeDecodeError: {message}")),
            _ => Raised("UnicodeDecodeError".to_owned()),
        });
    }
    let text = String::from_utf8_lossy(&raw)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if text.starts_with('\u{feff}') {
        return Err(decode_error(
            "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)",
        ));
    }
    loads_exact(text.as_bytes(), &hooks).map_err(|error| match error {
        DecodeError::Value(message) => decode_error(&message),
        DecodeError::Recursion => Raised(format!(
            "RecursionError: maximum recursion depth exceeded while decoding a JSON {} from a unicode string",
            deepest_container(&text)
        )),
    })
}

fn decode_error(message: &str) -> Raised {
    Raised(format!("json.decoder.JSONDecodeError: {message}"))
}

/// The kind of the container that first nests past CPython's scanner limit.
fn deepest_container(text: &str) -> &'static str {
    const LIMIT: usize = 9998;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            escaped = !escaped && ch == '\\';
            in_string = escaped || ch != '"';
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' | '{' => {
                depth += 1;
                if depth > LIMIT {
                    return if ch == '[' { "array" } else { "object" };
                }
            }
            ']' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    "array"
}

/// The Python type name of a loaded value, constants included.
fn kind(value: &Json) -> &'static str {
    match value {
        Json::String(text) if text.starts_with(CONSTANT) => "float",
        Json::Object(entries) if is_exact_number(entries) => "int",
        other => type_name(other),
    }
}

fn is_exact_number(entries: &[(String, Json)]) -> bool {
    matches!(entries, [(key, Json::String(_))] if key == EXACT_NUMBER)
}

/// `value[key]` with a string key.
pub(super) fn subscript<'a>(value: &'a Json, key: &str) -> Result<&'a Json, Raised> {
    match value {
        Json::Object(entries) if !is_exact_number(entries) => entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, child)| child)
            .ok_or_else(|| Raised(format!("KeyError: {}", repr(key)))),
        Json::String(text) if !text.starts_with(CONSTANT) => Err(Raised(
            "TypeError: string indices must be integers, not 'str'".to_owned(),
        )),
        Json::Array(_) => Err(Raised(
            "TypeError: list indices must be integers or slices, not str".to_owned(),
        )),
        other => Err(Raised(format!(
            "TypeError: '{}' object is not subscriptable",
            kind(other)
        ))),
    }
}

/// `str(value.get(outer, {}).get(inner, ""))` for a dict `value`.
pub(super) fn python_str(value: &Json, outer: &str, inner: &str) -> Result<String, Raised> {
    let nested = value.get(outer);
    let Some(nested) = nested else {
        return Ok(String::new());
    };
    match nested {
        Json::Object(entries) if !is_exact_number(entries) => {
            Ok(nested.get(inner).map(display).unwrap_or_default())
        }
        other => Err(Raised(format!(
            "AttributeError: '{}' object has no attribute 'get'",
            kind(other)
        ))),
    }
}

/// Python `str(value)`.
fn display(value: &Json) -> String {
    match value {
        Json::String(text) => match text.strip_prefix(CONSTANT) {
            Some("NaN") => "nan".to_owned(),
            Some("Infinity") => "inf".to_owned(),
            Some(_) => "-inf".to_owned(),
            None => text.clone(),
        },
        Json::Object(entries) if is_exact_number(entries) => match &entries[0].1 {
            Json::String(source) if source.contains(['.', 'e', 'E']) => {
                if source.starts_with('-') {
                    "-inf"
                } else {
                    "inf"
                }
                .to_owned()
            }
            Json::String(source) => source.clone(),
            _ => String::new(),
        },
        other => python_value::display(Some(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_native_policy_deepest_container_tracks_strings() {
        let nested = format!("{}{}", "[".repeat(9998), "{\"[\": 1}");
        assert_eq!(deepest_container(&nested), "object");
        assert_eq!(deepest_container(&"[".repeat(9999)), "array");
    }

    #[test]
    fn migration_native_policy_str_matches_python() {
        let value = Json::parse(br#"{"cuda": {"toolkit_major": 12.0}}"#).expect("json");
        assert_eq!(
            python_str(&value, "cuda", "toolkit_major").ok().as_deref(),
            Some("12.0")
        );
        assert_eq!(python_str(&value, "rocm", "x").ok().as_deref(), Some(""));
    }
}
