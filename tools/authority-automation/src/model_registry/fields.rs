//! Field checks shared by registry validation, with the legacy generator's
//! exact diagnostics (`_object`, `_string`, `_string_list`, `_exact_keys`).

use ci_controller::plan::document::Json;
use std::fmt;

/// A registry or manifest contract violation; the text is the diagnostic.
#[derive(Debug, PartialEq, Eq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

pub type ModelResult<T> = Result<T, ModelError>;

pub fn fail<T>(message: impl Into<String>) -> ModelResult<T> {
    Err(ModelError(message.into()))
}

/// The object entries of `value`, or `<field> must be an object`.
pub fn object<'a>(value: Option<&'a Json>, field: &str) -> ModelResult<&'a [(String, Json)]> {
    match value.and_then(Json::as_object) {
        Some(entries) => Ok(entries),
        None => fail(format!("{field} must be an object")),
    }
}

/// A non-empty, single-line string without NUL, tab or `|`.
pub fn string<'a>(value: Option<&'a Json>, field: &str) -> ModelResult<&'a str> {
    let text = match value.and_then(Json::as_str) {
        Some(text) if !text.is_empty() => text,
        _ => return fail(format!("{field} must be a non-empty string")),
    };
    if text.contains(['\0', '\r', '\n', '\t', '|']) {
        return fail(format!("{field} must be a single-line value"));
    }
    Ok(text)
}

/// A non-empty list of distinct [`string`] values.
pub fn string_list<'a>(value: Option<&'a Json>, field: &str) -> ModelResult<Vec<&'a str>> {
    let items = match value.and_then(Json::as_array) {
        Some(items) if !items.is_empty() => items,
        _ => return fail(format!("{field} must be a non-empty array")),
    };
    let result = items
        .iter()
        .enumerate()
        .map(|(index, item)| string(Some(item), &format!("{field}[{index}]")))
        .collect::<ModelResult<Vec<_>>>()?;
    let distinct = result.iter().collect::<std::collections::BTreeSet<_>>();
    if distinct.len() != result.len() {
        return fail(format!("{field} must not contain duplicates"));
    }
    Ok(result)
}

/// Rejects keys outside `allowed`, naming them in sorted order.
pub fn exact_keys(entries: &[(String, Json)], allowed: &[&str], field: &str) -> ModelResult<()> {
    let mut unknown = entries
        .iter()
        .map(|(key, _)| key.as_str())
        .filter(|key| !allowed.contains(key))
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    fail(format!(
        "{field} contains unknown fields: {}",
        unknown.join(", ")
    ))
}

/// `dict.get` over object entries.
pub fn get<'a>(entries: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

pub fn has(entries: &[(String, Json)], key: &str) -> bool {
    entries.iter().any(|(name, _)| name == key)
}

/// `^[a-z0-9][a-z0-9._-]*$`.
pub fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars.all(|ch| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '_' | '-')
        })
}

/// Lowercase hex of a length in `lengths`.
pub fn is_lower_hex(text: &str, lengths: std::ops::RangeInclusive<usize>) -> bool {
    lengths.contains(&text.len())
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// `PurePosixPath` escape checks: absolute or any `..` component.
pub fn escapes_root(path: &str) -> bool {
    path.starts_with('/') || path.split('/').any(|part| part == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> Json {
        Json::parse(text.as_bytes()).expect("valid JSON")
    }

    #[test]
    fn migration_models_string_rejects_injection_characters() {
        for bad in ["a|b", "a\\nb", "a\\tb", "a\\rb", "a\\u0000b"] {
            let value = json(&format!("\"{bad}\""));
            assert_eq!(
                string(Some(&value), "f"),
                fail("f must be a single-line value"),
                "{bad}"
            );
        }
        assert_eq!(
            string(Some(&json("\"\"")), "f"),
            fail("f must be a non-empty string")
        );
        assert_eq!(string(None, "f"), fail("f must be a non-empty string"));
    }

    #[test]
    fn migration_models_identifiers_and_hex_follow_legacy_patterns() {
        assert!(is_identifier("family-qwen3.5_x"));
        assert!(!is_identifier("-x") && !is_identifier("Upper") && !is_identifier(""));
        assert!(is_lower_hex(&"a".repeat(40), 40..=64));
        assert!(!is_lower_hex(&"A".repeat(40), 40..=64));
        assert!(!is_lower_hex(&"a".repeat(65), 40..=64));
    }

    #[test]
    fn migration_models_path_escape_matches_pure_posix_path() {
        assert!(escapes_root("/etc") && escapes_root("a/../b") && escapes_root(".."));
        assert!(!escapes_root("a/..b") && !escapes_root("./a") && !escapes_root("a//b"));
    }

    #[test]
    fn migration_models_unknown_keys_are_sorted() {
        let value = json(r#"{"z":1,"id":2,"a":3}"#);
        let entries = value.as_object().expect("object");
        assert_eq!(
            exact_keys(entries, &["id"], "row"),
            fail("row contains unknown fields: a, z")
        );
    }
}
