//! `json.loads(path.read_text(encoding="utf-8"))` with the last line of the
//! traceback the legacy composer would print when it fails: the `OSError`
//! subclass, `UnicodeDecodeError`, `JSONDecodeError`, the integer digit
//! limit `ValueError`, or `RecursionError`. Numbers `serde_json` cannot hold
//! exactly (and `NaN`/`Infinity`) are kept as source text.

use crate::artifact::zip_extract::os_error_line;
use crate::ci_operations::python_json_decode::{
    DecodeError, EXACT_NUMBER, Hooks, loads_exact, utf8_error,
};
use crate::ci_plan::document::Json;
use std::path::Path;

/// The decoder recurses per container; CPython allows 9998 levels.
const DECODE_STACK_BYTES: usize = 64 * 1024 * 1024;
const RECURSION_DEPTH: usize = 9999;

/// Reads and decodes `path`; `shown` is `str(path)` for `OSError` text.
pub(super) fn load(path: &Path, shown: &str) -> Result<Json, String> {
    let raw = std::fs::read(path).map_err(|error| os_error_line(&error, shown))?;
    if let Err(error) = std::str::from_utf8(&raw) {
        return Err(format!("UnicodeDecodeError: {}", utf8_error(&raw, &error)));
    }
    if raw.starts_with(b"\xef\xbb\xbf") {
        return Err(
            "json.decoder.JSONDecodeError: Unexpected UTF-8 BOM (decode using \
                    utf-8-sig): line 1 column 1 (char 0)"
                .to_owned(),
        );
    }
    let worker = std::thread::Builder::new()
        .stack_size(DECODE_STACK_BYTES)
        .spawn(move || decode(&raw))
        .map_err(|error| format!("OSError: {error}"))?;
    worker
        .join()
        .unwrap_or_else(|_| Err("RecursionError: maximum recursion depth exceeded".to_owned()))
}

fn decode(raw: &[u8]) -> Result<Json, String> {
    let hooks = Hooks { pairs, constant };
    loads_exact(raw, &hooks).map_err(|error| match error {
        DecodeError::Value(text) if text.starts_with("Exceeds the limit") => {
            format!("ValueError: {text}")
        }
        DecodeError::Value(text) => format!("json.decoder.JSONDecodeError: {text}"),
        DecodeError::Recursion => format!(
            "RecursionError: maximum recursion depth exceeded while decoding a JSON {} \
             from a unicode string",
            deepest_container(raw)
        ),
    })
}

/// A `dict`: a repeated key keeps its first position and its last value.
fn pairs(items: Vec<(String, Json)>) -> Result<Json, String> {
    let mut merged: Vec<(String, Json)> = Vec::with_capacity(items.len());
    for (key, value) in items {
        match merged.iter_mut().find(|(seen, _)| *seen == key) {
            Some(slot) => slot.1 = value,
            None => merged.push((key, value)),
        }
    }
    Ok(Json::Object(merged))
}

/// `NaN`, `Infinity` and `-Infinity` load as floats.
fn constant(name: &str) -> Result<Json, String> {
    Ok(Json::Object(vec![(
        EXACT_NUMBER.to_owned(),
        Json::String(name.to_owned()),
    )]))
}

/// Whether the container that exceeds the depth limit is an object or an
/// array, found by scanning brackets outside strings.
fn deepest_container(raw: &[u8]) -> &'static str {
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in raw {
        match (in_string, byte) {
            (true, _) if escaped => escaped = false,
            (true, b'\\') => escaped = true,
            (true, b'"') | (false, b'"') => in_string = !in_string,
            (false, b'[' | b'{') => {
                depth += 1;
                if depth == RECURSION_DEPTH {
                    return if *byte == b'{' { "object" } else { "array" };
                }
            }
            (false, b']' | b'}') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    "object"
}
