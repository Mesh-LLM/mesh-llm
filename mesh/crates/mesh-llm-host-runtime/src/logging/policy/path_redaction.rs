//! Home-directory redaction for paths and free-form log text.

/// Sanitize a file path for logging: replace the private home directory prefix
/// with `~`, keeping the platform separator that follows it.
///
/// The home directory comes from `dirs::home_dir()`, as everywhere else in this
/// crate. Reading `HOME` directly redacted nothing on Windows, where that
/// variable is not set and the home lives behind the known-folder API.
pub fn sanitize_path(path: &std::path::Path) -> String {
    if let Some(home) = dirs::home_dir() {
        return match path.strip_prefix(&home) {
            Ok(rest) if rest.as_os_str().is_empty() => "~".to_string(),
            Ok(rest) => format!("~{}{}", std::path::MAIN_SEPARATOR, rest.display()),
            // A sibling such as `<home>-backup` only shares a string prefix and
            // must stay intact, so it is returned unchanged rather than mangled.
            Err(_) => path.to_string_lossy().to_string(),
        };
    }

    // Fallback: just show the last 3 components.
    let parts: Vec<_> = path.components().collect();
    if parts.len() <= 3 {
        return path.to_string_lossy().to_string();
    }
    format!(
        "{}/.../{}",
        parts[parts.len() - 3].as_os_str().to_string_lossy(),
        parts.last().unwrap().as_os_str().to_string_lossy()
    )
}

/// Remove home-directory occurrences followed by a separator or text delimiter.
/// Sibling components such as `alice-backup` and `alice.txt` remain intact.
/// This is a log-text heuristic, not a parser for quoted or escaped paths.
pub fn sanitize_paths_in_text(text: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        sanitize_paths_with_home(text, &home.to_string_lossy())
            .replace("/private/var/", "/var/")
            .replace("/private/tmp/", "/tmp/")
    } else {
        text.to_string()
    }
}

/// Redact decoded JSON string values at the audit persistence boundary.
/// Non-JSON details retain the free-form text fallback.
pub fn sanitize_paths_in_json_text(text: &str) -> String {
    dirs::home_dir().map_or_else(
        || text.to_string(),
        |home| sanitize_json_paths_with_home(text, &home.to_string_lossy()),
    )
}

fn sanitize_json_paths_with_home(text: &str, home: &str) -> String {
    if home.is_empty() {
        return text.to_string();
    }
    if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(text) {
        sanitize_json_paths(&mut value, home);
        value.to_string()
    } else {
        sanitize_paths_with_home(text, home)
            .replace("/private/var/", "/var/")
            .replace("/private/tmp/", "/tmp/")
    }
}

fn sanitize_json_paths(value: &mut serde_json::Value, home: &str) {
    match value {
        serde_json::Value::String(text) => {
            *text = sanitize_paths_with_home(text, home)
                .replace("/private/var/", "/var/")
                .replace("/private/tmp/", "/tmp/");
        }
        serde_json::Value::Array(values) => {
            for value in values {
                sanitize_json_paths(value, home);
            }
        }
        serde_json::Value::Object(fields) => {
            for value in fields.values_mut() {
                sanitize_json_paths(value, home);
            }
        }
        _ => {}
    }
}

fn sanitize_paths_with_home(text: &str, home: &str) -> String {
    if home.is_empty() {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut copied = 0;
    for (start, _) in text.match_indices(home) {
        let end = start + home.len();
        if ends_home(&text[end..]) {
            result.push_str(&text[copied..start]);
            result.push('~');
            copied = end;
        }
    }
    result.push_str(&text[copied..]);
    result
}

fn ends_home(suffix: &str) -> bool {
    let mut chars = suffix.chars();
    match chars.next() {
        None | Some('/' | '\\') => true,
        Some(c) if is_text_delimiter(c) => true,
        // A sentence-final period is a delimiter, but `.backup` is a sibling.
        Some('.') => chars.next().is_none_or(is_text_delimiter),
        _ => false,
    }
}

fn is_text_delimiter(c: char) -> bool {
    // Whitespace and these punctuation marks are interpreted as prose, even
    // though some filesystems permit them within a component.
    c.is_whitespace()
        || matches!(
            c,
            '\"' | '\'' | '`' | ')' | ']' | '}' | ',' | ';' | ':' | '!' | '?' | '&'
        )
}

#[cfg(test)]
#[path = "path_redaction_tests.rs"]
mod tests;
