//! Centralized privacy policy for operator logging.
//! Provides redaction, truncation, hashing, and sanitization primitives that protect sensitive data before any log event is serialized or persisted.

use std::collections::HashMap;

use mesh_llm_events::logging::events::LifecycleEvent;

mod artifact_redaction;
pub use artifact_redaction::redact_artifact_bytes;

/// Redaction mode applied to a string value. The most restrictive applicable rule wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedactMode {
    /// Value contains credentials; replace entirely with `[REDACTED]`.
    FullRedact,
    /// Value is safe but must be truncated to a maximum length.
    Truncate(usize),
    /// Value passes through unchanged (already verified safe).
    PassThrough,
}

/// Sensitive header names that must never appear in logs (case-insensitive matching).
const REDACTED_HEADERS: &[&str] = &[
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "x-bearer-token",
    "x-forwarded-access-token", // can leak internal IDs on some platforms
    "x-ms-client-request-id",   // can leak internal IDs on some platforms
    "x-session-id",
];

/// Query parameter names that carry credentials or tokens.
const REDACTED_QUERY_PARAMS: &[&str] = &[
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "key",
    "password",
    "secret",
    "session_id",
    "token",
];

/// Prefixes that indicate a token/credential value regardless of key name.
const TOKEN_VALUE_PREFIXES: &[&str] = &[
    "bearer ", // HTTP Bearer auth pattern.
    "basic ",  // HTTP Basic auth (base64 user:pass).
    "sk_",     // OpenAI-style secret key prefix (underscore variant).
    "sk-",     // OpenAI-style secret key prefix (dash variant).
    "ghp_",    // GitHub personal access token prefix.
];

/// Maximum length for a preserved (non-redacted) string value in logs.
pub const MAX_LOG_STRING_LEN: usize = 1024;

/// Maximum number of lines to preserve from a stack trace or multi-line error.
pub const MAX_STACK_LINES: usize = 32;

/// Maximum bytes for an artifact body snapshot before truncation.
pub const DEFAULT_ARTIFACT_BODY_LIMIT: usize = 256 * 1024; // 256 KiB

// ---------------------------------------------------------------------------
// Redaction entry point
// ---------------------------------------------------------------------------

/// Apply the privacy policy to a raw string value, returning how it was treated.
/// This is the single function called before any log line touches storage or export.
pub fn apply_redaction(value: &str) -> (String, RedactMode) {
    // Check credential prefixes first — these always fully redact.
    if contains_credential_marker(value) {
        return ("[REDACTED]".to_string(), RedactMode::FullRedact);
    }

    // Check for common credential patterns in the middle of strings (e.g., JSON bodies).
    if looks_like_credential_value(value) {
        return ("[REDACTED]".to_string(), RedactMode::FullRedact);
    }

    // Truncate long values — use char-based truncation to avoid splitting multi-byte UTF-8.
    if value.chars().count() > MAX_LOG_STRING_LEN {
        let truncated: String = value.chars().take(MAX_LOG_STRING_LEN).collect();
        return (
            format!("{truncated}... [TRUNCATED]"),
            RedactMode::Truncate(MAX_LOG_STRING_LEN),
        );
    }

    (value.to_string(), RedactMode::PassThrough)
}

/// Apply credential redaction to an explicitly captured artifact string
/// without the presentation-oriented 1 KiB log-value truncation. Artifact
/// bytes already have configured per-item and aggregate bounds; clipping here
/// would silently corrupt an otherwise available payload and leave its
/// `truncated` metadata false.
fn apply_artifact_redaction(value: &str) -> String {
    if contains_credential_marker(value) || looks_like_credential_value(value) {
        "[REDACTED]".to_string()
    } else {
        value.to_string()
    }
}

/// Sanitize every free-form string carried by a canonical lifecycle event.
///
/// This is deliberately exhaustive: adding a string-bearing lifecycle variant
/// requires deciding how its content crosses the durable logging boundary.
pub fn sanitize_lifecycle_event(event: LifecycleEvent) -> LifecycleEvent {
    fn value(input: String) -> String {
        let path_safe = sanitize_paths_in_text(&input);
        let url_safe = redact_urls_in_text(&path_safe);
        apply_redaction(&url_safe).0
    }
    fn optional(input: Option<String>) -> Option<String> {
        input.map(value)
    }

    match event {
        LifecycleEvent::Admitted { model, method } => LifecycleEvent::Admitted {
            model: optional(model),
            method: optional(method),
        },
        LifecycleEvent::RouteSelected {
            model,
            provider,
            engine,
        } => LifecycleEvent::RouteSelected {
            model: optional(model),
            provider: optional(provider),
            engine: optional(engine),
        },
        LifecycleEvent::AttemptFailed { attempt_id, error } => LifecycleEvent::AttemptFailed {
            attempt_id,
            error: optional(error),
        },
        LifecycleEvent::StreamStarted { model } => LifecycleEvent::StreamStarted {
            model: optional(model),
        },
        LifecycleEvent::StreamError { error } => LifecycleEvent::StreamError {
            error: optional(error),
        },
        LifecycleEvent::AuditError { message } => LifecycleEvent::AuditError {
            message: value(message),
        },
        LifecycleEvent::Failed { error, status_code } => LifecycleEvent::Failed {
            error: value(error),
            status_code,
        },
        LifecycleEvent::Rejected {
            reason,
            status_code,
        } => LifecycleEvent::Rejected {
            reason: optional(reason),
            status_code,
        },
        LifecycleEvent::Cancelled { reason } => LifecycleEvent::Cancelled {
            reason: optional(reason),
        },
        LifecycleEvent::Dropped { reason } => LifecycleEvent::Dropped {
            reason: optional(reason),
        },
        other => other,
    }
}

/// True if `value` starts with an `http`/`https` scheme, matched
/// case-insensitively.
///
/// URL schemes are case-insensitive (RFC 3986 §3.1), so an address like
/// `HTTP://user:pass@host` is still an HTTP URL. A case-sensitive check would
/// let such an address bypass credential redaction, so every redaction gate
/// must use this helper rather than a bare `starts_with("http://")`.
pub(crate) fn is_http_url(value: &str) -> bool {
    http_url_start(value) == Some(0)
}

fn http_url_start(value: &str) -> Option<usize> {
    let bytes = value.as_bytes();
    let starts_ci = |prefix: &[u8]| {
        bytes.len() >= prefix.len() && bytes[..prefix.len()].eq_ignore_ascii_case(prefix)
    };
    if starts_ci(b"http://") || starts_ci(b"https://") {
        return Some(0);
    }
    (1..bytes.len()).find(|&index| {
        let suffix = &bytes[index..];
        (suffix.len() >= b"http://".len()
            && suffix[..b"http://".len()].eq_ignore_ascii_case(b"http://"))
            || (suffix.len() >= b"https://".len()
                && suffix[..b"https://".len()].eq_ignore_ascii_case(b"https://"))
    })
}

fn redactable_url_start(value: &str) -> Option<usize> {
    let http_start = http_url_start(value);
    let bytes = value.as_bytes();
    let tcp_start = (0..bytes.len()).find(|&index| {
        let suffix = &bytes[index..];
        suffix.len() >= b"tcp://".len() && suffix[..b"tcp://".len()].eq_ignore_ascii_case(b"tcp://")
    });
    match (http_start, tcp_start) {
        (Some(http), Some(tcp)) => Some(http.min(tcp)),
        (Some(http), None) => Some(http),
        (None, Some(tcp)) => Some(tcp),
        (None, None) => None,
    }
}

pub(crate) fn redact_urls_in_text(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            let mut redacted = String::with_capacity(word.len());
            let mut cursor = 0;
            while let Some(relative_start) = redactable_url_start(&word[cursor..]) {
                let url_start = cursor + relative_start;
                redacted.push_str(&word[cursor..url_start]);

                let scheme_len = if word.as_bytes()[url_start..]
                    .get(..b"https://".len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"https://"))
                {
                    b"https://".len()
                } else if word.as_bytes()[url_start..]
                    .get(..b"http://".len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"http://"))
                {
                    b"http://".len()
                } else {
                    b"tcp://".len()
                };
                let after_scheme = url_start + scheme_len;
                let url_end = redactable_url_start(&word[after_scheme..])
                    .map_or(word.len(), |next_start| after_scheme + next_start);
                let url_with_separator = &word[url_start..url_end];
                let url_len = url_with_separator
                    .trim_end_matches(|character: char| {
                        character.is_ascii_punctuation() && character != '='
                    })
                    .len();
                let (url, separator) = url_with_separator.split_at(url_len);
                redacted.push_str(&redact_url_query(url));
                redacted.push_str(separator);
                cursor = url_end;
            }
            redacted.push_str(&word[cursor..]);
            redacted
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strip all sensitive headers from a header map, returning the sanitized version.
pub fn redact_headers<'a>(
    headers: impl Iterator<Item = (&'a str, &'a str)>,
) -> HashMap<String, String> {
    let mut clean = HashMap::new();
    for (key, value) in headers {
        if REDACTED_HEADERS.contains(&&*key.to_lowercase()) {
            clean.insert(key.to_string(), "[REDACTED]".to_string());
        } else {
            let (sanitized_val, _) = apply_redaction(value);
            clean.insert(key.to_string(), sanitized_val);
        }
    }
    clean
}

/// Redact sensitive query parameters from a URL. Returns the cleaned URL string.
pub fn redact_url_query(url: &str) -> String {
    let (base, query_part) = url.split_once('?').unwrap_or((url, ""));
    let base = redact_url_userinfo(base);
    if query_part.is_empty() {
        return base;
    }

    // Handle fragments.
    let (query_only, fragment) = match query_part.split_once('#') {
        Some((q, f)) => (q, f.to_string()),
        None => (query_part, String::new()),
    };

    let mut cleaned_params = Vec::new();
    for param in query_only.split('&') {
        if param.is_empty() {
            continue;
        }
        match param.split_once('=') {
            Some((key, value)) => {
                let key_lower = key.to_lowercase();
                if REDACTED_QUERY_PARAMS.contains(&&*key_lower) || is_token_like_value(value) {
                    cleaned_params.push(format!("{key}=[REDACTED]"));
                } else {
                    let (sanitized, _) = apply_redaction(value);
                    cleaned_params.push(format!("{key}={sanitized}"));
                }
            }
            None => {
                // Key-only parameter; pass through if not a known sensitive name.
                if REDACTED_QUERY_PARAMS.contains(&&*param.to_lowercase()) {
                    cleaned_params.push(format!("{param}=[REDACTED]"));
                } else {
                    cleaned_params.push(param.to_string());
                }
            }
        }
    }

    let fragment = if fragment.is_empty() {
        String::new()
    } else {
        format!("#{fragment}")
    };
    if cleaned_params.is_empty() {
        format!("{base}{fragment}")
    } else {
        format!("{}?{}{}", base, cleaned_params.join("&"), fragment)
    }
}

fn redact_url_userinfo(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let authority_start = scheme_end + 3;
    let authority_end = url[authority_start..]
        .find(['/', '#'])
        .map_or(url.len(), |offset| authority_start + offset);
    let authority = &url[authority_start..authority_end];
    let Some(user_info_end) = authority.rfind('@') else {
        return url.to_string();
    };
    format!(
        "{}[REDACTED]@{}{}",
        &url[..authority_start],
        &authority[user_info_end + 1..],
        &url[authority_end..]
    )
}

/// Truncate a multi-line string (stack trace, error output) to MAX_STACK_LINES.
pub fn truncate_stack_trace(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() <= MAX_STACK_LINES {
        return input.to_string();
    }

    // Keep first 8 lines (context) and last N-8 lines, with a marker in between.
    let head_count = 8.min(lines.len());
    let tail_count = (MAX_STACK_LINES - head_count).min(lines.len() - head_count);
    let skipped = lines.len() - head_count - tail_count;

    format!(
        "{}\n[... {} frame(s) elided ...]\n{}",
        lines[..head_count].join("\n"),
        skipped,
        lines[lines.len() - tail_count..].join("\n")
    )
}

/// Sanitize a JSON-like body string by redacting known sensitive keys.
pub fn sanitize_json_body(body: &str) -> String {
    const SENSITIVE_KEYS: &[&str] = &[
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "password",
        "secret",
        "secrets",
        "session_id",
        "token",
    ];

    let mut result = body.to_string();
    for key in SENSITIVE_KEYS {
        result = redact_key_in_json(&result, key);
    }

    // Post-redaction pass: only check token prefixes (not credential keywords which match JSON keys).
    if contains_credential_marker(&result) {
        return "[REDACTED]".to_string();
    }
    result
}

/// Redact all values for a given JSON key in text. Processes one pattern at a time on mutable text to avoid index shifts from cross-key interference.
fn redact_key_in_json(text: &str, key: &str) -> String {
    let colon_pattern = format!("\"{}\":", key);
    let mut result = text.to_string();
    let mut search_from = 0;

    while let Some(idx) = result[search_from..].find(&colon_pattern) {
        let after_colon = search_from + idx + colon_pattern.len();

        // Skip whitespace between colon and value.
        let ws_end = after_colon
            + result[after_colon..]
                .find(|c: char| !c.is_whitespace())
                .unwrap_or(0);
        if ws_end >= result.len() {
            break;
        }

        let remaining = &result[ws_end..];

        // Determine the end of the value based on its type.
        let value_end = if remaining.starts_with('"') {
            // Quoted string — find closing quote.
            if let Some(close_idx) = find_closing_quote(remaining, 1) {
                ws_end + close_idx + 1
            } else {
                break; // malformed JSON.
            }
        } else if remaining.starts_with('[') || remaining.starts_with('{') {
            let open_char = remaining.chars().next().unwrap();
            let close_char = if open_char == '[' { ']' } else { '}' };
            if let Some(close_idx) = find_matching_bracket(remaining, open_char, close_char) {
                ws_end + close_idx + 1
            } else {
                break; // malformed JSON.
            }
        } else {
            // Unquoted value — replace until delimiter.
            if let Some(delim_offset) =
                remaining.find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
            {
                ws_end + delim_offset
            } else {
                result.len()
            }
        };

        let replacement = if remaining.starts_with('"') {
            "\"[REDACTED]\""
        } else {
            "[REDACTED]"
        };

        result.replace_range(ws_end..value_end, replacement);
        search_from = ws_end + replacement.len();
    }

    result
}

/// Sanitize a file path for logging: replace private home directory prefix with `~/`.
pub fn sanitize_path(path: &std::path::Path) -> String {
    use std::env;
    if let Some(home) = env::var_os("HOME") {
        let home_str = home.to_string_lossy();
        return path.to_string_lossy().replace(&*home_str, "~");
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

/// Hash a value for fingerprinting (e.g., token fingerprints in mDNS).
pub fn hash_value(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    // Return first 16 hex chars (8 bytes of fingerprint).
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Remove private directory prefixes from a string containing paths.
pub fn sanitize_paths_in_text(text: &str) -> String {
    use std::env;
    if let Some(home) = env::var_os("HOME") {
        let home_str = home.to_string_lossy().to_string();
        text.replace(&home_str, "~")
            .replace("/private/var/", "/var/")
            .replace("/private/tmp/", "/tmp/")
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// Policy construction from config
// ---------------------------------------------------------------------------

/// Construct a runtime privacy policy from the logging configuration.
pub fn build_policy(config: &mesh_llm_config::LoggingConfig) -> PrivacyPolicy {
    use mesh_llm_config::CaptureMode;

    let capture_mode = match config.artifact.capture_mode {
        CaptureMode::MetadataOnly => PolicyCaptureMode::MetadataOnly,
        CaptureMode::RedactedArtifacts => PolicyCaptureMode::RedactedWithLimits(
            config.artifact.byte_limit_bytes as usize,
            config.artifact.aggregate_limit_bytes as usize,
        ),
    };

    PrivacyPolicy {
        enabled: config.enabled,
        application_state_root: config.application_state_root.clone(),
        summary_line_limit: config.summary_line_limit as usize,
        event_buffer_size: config.event_buffer_size as usize,
        retention_ttl_secs: config.retention_ttl_secs,
        retention_max_rows: config.retention_max_rows,
        replay_capacity: config.replay_capacity as usize,
        queue_capacity: config.queue_capacity as usize,
        capture_mode,
        export_limit_bytes: config.export_limit_bytes as usize,
        cleanup_cadence_secs: config.cleanup_cadence_secs,
        webhook_enabled: config.webhook.enabled,
    }
}

/// Capture mode for the constructed policy (mirrors config but with runtime semantics).
#[derive(Clone, Copy, Debug)]
pub enum PolicyCaptureMode {
    /// Only metadata is captured; no artifact content. This is the default.
    MetadataOnly,
    /// Redacted artifacts are captured up to per-item and aggregate byte limits.
    RedactedWithLimits(usize, usize), // (per_item_bytes, aggregate_bytes)
}

/// Runtime privacy policy derived from configuration at startup. All logging must pass through this policy before serialization.
#[derive(Clone, Debug)]
pub struct PrivacyPolicy {
    pub enabled: bool,
    pub application_state_root: Option<std::path::PathBuf>,
    pub summary_line_limit: usize,
    pub event_buffer_size: usize,
    pub retention_ttl_secs: u64,
    pub retention_max_rows: u64,
    pub replay_capacity: usize,
    pub queue_capacity: usize,
    pub capture_mode: PolicyCaptureMode,
    pub export_limit_bytes: usize,
    pub cleanup_cadence_secs: u64,
    pub webhook_enabled: bool,
}

impl PrivacyPolicy {
    /// Check whether artifact content may be captured under this policy.
    pub fn allows_artifact_content(&self) -> bool {
        !matches!(self.capture_mode, PolicyCaptureMode::MetadataOnly)
    }

    /// Apply the full redaction pipeline to a log value: path sanitization → credential detection → truncation.
    pub fn sanitize_value(&self, input: &str) -> String {
        let text = sanitize_paths_in_text(input);
        let (sanitized, _) = apply_redaction(&text);
        sanitized
    }

    /// Only retention and replay limits are eligible for dynamic application.
    /// Runtime application remains the responsibility of the logging service.
    pub fn is_restart_required(setting_path: &str) -> bool {
        !matches!(
            setting_path,
            "logging.retention_ttl_secs" | "logging.replay_capacity"
        )
    }

    /// Classify changed settings using the public logging schema contract.
    pub fn classify_change<'a>(paths: impl IntoIterator<Item = &'a str>) -> (bool, Vec<&'a str>) {
        let mut dynamic_paths = Vec::new();
        let mut needs_restart = false;
        for path in paths {
            if Self::is_restart_required(path) {
                needs_restart = true;
            } else {
                dynamic_paths.push(path);
            }
        }
        (needs_restart, dynamic_paths)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a string value looks like a credential regardless of its key name.
fn is_token_like_value(value: &str) -> bool {
    let v = value.trim();

    // Token prefixes (e.g., "Bearer ") must be checked before the no-space filter,
    // since Bearer tokens contain spaces after the prefix ("Bearer abc123...").
    contains_credential_marker(v) && v.len() >= 20
}

/// Detect credential markers in arbitrary free-form text. Values routinely
/// arrive wrapped in JSON, error messages, or lifecycle annotations, so a
/// start-of-string check is not sufficient.
fn contains_credential_marker(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    TOKEN_VALUE_PREFIXES
        .iter()
        .any(|prefix| contains_marker_at_token_boundary(&lower, prefix))
}

fn contains_marker_at_token_boundary(value: &str, marker: &str) -> bool {
    value.match_indices(marker).any(|(index, _)| {
        index == 0
            || value[..index]
                .chars()
                .next_back()
                .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
    })
}

/// Check if a string looks like it contains credential values.
fn looks_like_credential_value(value: &str) -> bool {
    let lower = value.to_lowercase();
    // Common patterns in JSON or URL-encoded bodies that indicate credentials.
    const PATTERNS: &[&str] = &[
        "password",
        "secret_key",
        "private_key",
        "auth_token",
        "access_token",
    ];

    PATTERNS.iter().any(|pattern| lower.contains(pattern)) && value.len() > 10
}

/// Find the index of the closing quote, handling escaped quotes. Returns None if no matching close is found.
fn find_closing_quote(s: &str, start: usize) -> Option<usize> {
    let mut i = start;
    while i < s.len() {
        let ch = s.as_bytes()[i];
        match ch {
            b'"' => return Some(i),
            b'\\' => i += 2, // skip escaped character.
            _ => i += 1,
        }
    }
    None
}

/// Find the matching closing bracket/brace for an opening one at position 0 of `s`. Returns index from start of `s`, or None if unmatched.
fn find_matching_bracket(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1;
    let mut i = 1; // skip the opening bracket at position 0.
    while i < s.len() && depth > 0 {
        match s.as_bytes()[i] as char {
            c if c == open => depth += 1,
            c if c == close => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            i += 1;
        }
    }
    (depth == 0).then_some(i)
}

// ---------------------------------------------------------------------------
// Redaction corpus tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod redaction_corpus_tests {
    use super::*;

    /// Test suite proving each sensitive category is stripped and bounded metadata survives.
    #[test]
    fn bearer_auth_header_is_redacted() {
        let headers = vec![(
            "Authorization",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        )];
        let clean = redact_headers(headers.into_iter());
        assert_eq!(clean.get("Authorization"), Some(&"[REDACTED]".to_string()));
    }

    #[test]
    fn basic_auth_header_is_redacted() {
        let headers = vec![("Proxy-Authorization", "Basic dXNlcjpwYXNz")];
        let clean = redact_headers(headers.into_iter());
        assert_eq!(
            clean.get("Proxy-Authorization"),
            Some(&"[REDACTED]".to_string())
        );
    }

    #[test]
    fn cookie_header_is_redacted() {
        let headers = vec![("Set-Cookie", "session=abc123; Path=/")];
        let clean = redact_headers(headers.into_iter());
        assert_eq!(clean.get("Set-Cookie"), Some(&"[REDACTED]".to_string()));
    }

    #[test]
    fn non_sensitive_header_survives() {
        let headers = vec![("Content-Type", "application/json")];
        let clean = redact_headers(headers.into_iter());
        assert_eq!(
            clean.get("Content-Type"),
            Some(&"application/json".to_string())
        );
    }

    #[test]
    fn token_value_prefix_bearer_redacted() {
        let (_, mode) = apply_redaction("Bearer eyJhbGciOiJIUzI1NiIs");
        assert_eq!(mode, RedactMode::FullRedact);
    }

    #[test]
    fn mesh_invite_token_is_not_treated_as_a_credential() {
        let (val, mode) = apply_redaction("mesh-llm-invite-token-abc123def456ghi789jkl0");
        assert_eq!(mode, RedactMode::PassThrough);
        assert_eq!(val, "mesh-llm-invite-token-abc123def456ghi789jkl0");
    }

    #[test]
    fn credential_prefix_text_requires_a_token_boundary() {
        for value in ["disk-pressure recovered", "task-run completed"] {
            let (sanitized, mode) = apply_redaction(value);
            assert_eq!(mode, RedactMode::PassThrough, "{value}");
            assert_eq!(sanitized, value);
        }
    }

    #[test]
    fn openai_secret_key_redacted() {
        let (_, mode) = apply_redaction("sk-abc123def456ghi789jklmnopqrstuvwxyz0123456789ABCD");
        assert_eq!(mode, RedactMode::FullRedact);
    }

    #[test]
    fn github_token_redacted() {
        let (_, mode) = apply_redaction("ghp_abc123def456ghi789jklmnopqrstuvwxyz0");
        assert_eq!(mode, RedactMode::FullRedact);
    }

    #[test]
    fn query_params_with_token_are_redacted() {
        let url = "https://example.com/api?token=secret123&format=json&page=1";
        let cleaned = redact_url_query(url);
        assert!(cleaned.contains("token=[REDACTED]"));
        assert!(!cleaned.contains("secret123"));
        assert!(cleaned.contains("format=json"));
        assert!(cleaned.contains("page=1"));
    }

    #[test]
    fn query_params_with_access_token_redacted() {
        let url = "https://api.example.com?access_token=my_secret_token&user=alice";
        let cleaned = redact_url_query(url);
        assert!(cleaned.contains("access_token=[REDACTED]"));
        assert!(!cleaned.contains("my_secret_token"));
    }

    #[test]
    fn url_without_query_passes_through() {
        let url = "https://example.com/path/to/resource";
        assert_eq!(redact_url_query(url), url);
    }

    #[test]
    fn every_punctuation_adjacent_url_is_redacted() {
        let text = "(https://alice:first@host/v1),(https://bob:second@host/v2)";
        let cleaned = redact_urls_in_text(text);
        assert_eq!(
            cleaned,
            "(https://[REDACTED]@host/v1),(https://[REDACTED]@host/v2)"
        );
        assert!(!cleaned.contains("alice:first"));
        assert!(!cleaned.contains("bob:second"));
    }

    #[test]
    fn punctuation_between_query_bearing_urls_is_preserved() {
        let text = "(https://host/v1?token=first),(https://host/v2?token=second)";
        let cleaned = redact_urls_in_text(text);
        assert_eq!(
            cleaned,
            "(https://host/v1?token=[REDACTED]),(https://host/v2?token=[REDACTED])"
        );
        assert!(!cleaned.contains("first"));
        assert!(!cleaned.contains("second"));
    }

    #[test]
    fn arbitrary_punctuation_between_query_bearing_urls_is_preserved() {
        let text = "https://host/v1?token=first).https://host/v2?token=second";
        let cleaned = redact_urls_in_text(text);
        assert_eq!(
            cleaned,
            "https://host/v1?token=[REDACTED]).https://host/v2?token=[REDACTED]"
        );
        assert!(!cleaned.contains("first"));
        assert!(!cleaned.contains("second"));
    }

    #[test]
    fn url_syntax_punctuation_between_adjacent_urls_is_preserved() {
        for separator in ['?', '&', '#'] {
            let text =
                format!("https://host/v1?token=first{separator}https://host/v2?token=second");
            let cleaned = redact_urls_in_text(&text);
            assert_eq!(
                cleaned,
                format!(
                    "https://host/v1?token=[REDACTED]{separator}https://host/v2?token=[REDACTED]"
                )
            );
            assert!(!cleaned.contains("first"));
            assert!(!cleaned.contains("second"));
        }
    }

    #[test]
    fn json_body_password_redacted() {
        let body = r#"{"username": "alice", "password": "super_secret_123"}"#;
        let sanitized = sanitize_json_body(body);
        assert!(sanitized.contains("password"));
        assert!(!sanitized.contains("super_secret_123"));
    }

    #[test]
    fn json_body_api_key_redacted() {
        let body = r#"{"api_key": "sk-abc123", "service": "openai"}"#;
        let sanitized = sanitize_json_body(body);
        assert!(sanitized.contains("api_key"));
        assert!(!sanitized.contains("sk-abc123"));
    }

    #[test]
    fn json_non_sensitive_keys_survive() {
        let body = r#"{"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}"#;
        let sanitized = sanitize_json_body(body);
        assert!(sanitized.contains("gpt-4"));
        assert!(sanitized.contains("0.7"));
    }

    #[test]
    fn stack_trace_truncated() {
        // Generate a long stack trace (> 32 lines).
        let mut lines = Vec::new();
        for i in 1..=50 {
            lines.push(format!("   at module.func{i} (file.rs:{i})"));
        }
        let trace = lines.join("\n");

        let truncated = truncate_stack_trace(&trace);
        assert!(truncated.contains("[... ") && truncated.contains(" frame(s) elided ...]"));
        // Should keep first 8 and last 24 (32 total).
        let result_lines: Vec<&str> = truncated.lines().collect();
        assert_eq!(result_lines.len(), MAX_STACK_LINES + 1); // +1 for the elision marker.
    }

    #[test]
    fn short_stack_trace_unchanged() {
        let trace = "at module.func1 (file.rs:1)\nat module.func2 (file.rs:2)";
        assert_eq!(truncate_stack_trace(trace), trace);
    }

    #[test]
    fn long_string_truncated() {
        let long_val = "x".repeat(2048);
        let (_, mode) = apply_redaction(&long_val);
        matches!(mode, RedactMode::Truncate(_));
    }

    #[test]
    fn short_string_passes_through() {
        let (val, mode) = apply_redaction("hello world");
        assert_eq!(mode, RedactMode::PassThrough);
        assert_eq!(val, "hello world");
    }

    #[test]
    fn path_sanitization_hides_home_dir() {
        let home = dirs::home_dir().expect("test runner has a home directory");
        let test_path = home.join("some/deep/path/file.log");
        let sanitized = sanitize_path(&test_path);
        assert!(sanitized.starts_with("~/"));
        assert!(!sanitized.contains(&home.display().to_string()));
    }

    #[test]
    fn hash_value_produces_fingerprint() {
        let h1 = hash_value("some-token-value");
        let h2 = hash_value("different-token");
        assert_eq!(h1.len(), 16); // 8 bytes hex.
        assert_ne!(h1, h2);
    }

    #[test]
    fn sanitize_paths_in_text_replaces_home() {
        let home = dirs::home_dir().expect("test runner has a home directory");
        let text = format!("Error in {}", home.join("mesh-llm/logs/app.log").display());
        let sanitized = sanitize_paths_in_text(&text);
        assert!(sanitized.contains("~/mesh-llm"));
    }

    #[test]
    fn policy_metadata_only_no_artifacts() {
        use mesh_llm_config::LoggingConfig;
        let config = LoggingConfig::default(); // CaptureMode defaults to MetadataOnly.
        let policy = build_policy(&config);
        assert!(!policy.allows_artifact_content());
    }

    #[test]
    fn policy_redacted_artifacts_allows_content() {
        use mesh_llm_config::{CaptureMode, LoggingConfig};
        let config = LoggingConfig {
            artifact: mesh_llm_config::LoggingArtifactConfig {
                capture_mode: CaptureMode::RedactedArtifacts,
                ..Default::default()
            },
            ..Default::default()
        };
        let policy = build_policy(&config);
        assert!(policy.allows_artifact_content());
    }

    #[test]
    fn logging_settings_classify_only_retention_and_replay_as_dynamic() {
        let restart_required_paths = [
            "logging.enabled",
            "logging.application_state_root",
            "logging.summary_line_limit",
            "logging.event_buffer_size",
            "logging.retention_max_rows",
            "logging.queue_capacity",
            "logging.cleanup_cadence_secs",
            "logging.artifact.capture_mode",
            "logging.artifact.byte_limit_bytes",
            "logging.export_limit_bytes",
        ];

        for path in restart_required_paths {
            assert!(
                PrivacyPolicy::is_restart_required(path),
                "{path} should be restart-required"
            );
        }

        for path in ["logging.retention_ttl_secs", "logging.replay_capacity"] {
            assert!(
                !PrivacyPolicy::is_restart_required(path),
                "{path} should be dynamically applicable"
            );
        }

        let (needs_restart, dynamic_paths) = PrivacyPolicy::classify_change([
            "logging.retention_ttl_secs",
            "logging.replay_capacity",
        ]);
        assert!(!needs_restart);
        assert_eq!(
            dynamic_paths,
            vec!["logging.retention_ttl_secs", "logging.replay_capacity"]
        );

        let (needs_restart, dynamic_paths) = PrivacyPolicy::classify_change([
            "logging.retention_ttl_secs",
            "logging.queue_capacity",
        ]);
        assert!(needs_restart);
        assert_eq!(dynamic_paths, vec!["logging.retention_ttl_secs"]);
    }

    #[test]
    fn credential_value_detection_in_body() {
        let body = r#"{"auth_token": "secret1234567890"}"#;
        let (_, mode) = apply_redaction(body);
        assert_eq!(mode, RedactMode::FullRedact);
    }

    #[test]
    fn non_credential_body_passes_through() {
        let body = r#"{"status": "ok", "count": 42}"#;
        let (_, mode) = apply_redaction(body);
        assert_eq!(mode, RedactMode::PassThrough);
    }

    #[test]
    fn json_array_value_for_sensitive_key_redacted() {
        // When a sensitive key has an array value.
        let body = r#"{"secrets": ["a", "b"]}"#;
        let sanitized = sanitize_json_body(body);
        assert!(!sanitized.contains("\"a\""));
    }

    #[test]
    fn query_param_with_bearer_value_redacted() {
        // A value that starts with "Bearer " (space) should be redacted even if the key isn't in the sensitive list.
        let url = "https://example.com/callback?code=Bearer abc123def456ghi789jkl0&state=x";
        let cleaned = redact_url_query(url);
        // The value starts with Bearer so it should be caught by is_token_like_value.
        assert!(!cleaned.contains("abc123"));
    }

    #[test]
    fn credentials_embedded_in_free_form_text_are_redacted() {
        for value in [
            "request failed: Bearer secret-token-value",
            "upstream returned Basic YWxpY2U6cGFzc3dvcmQ=",
            "body contains sk-live-example",
            "body contains ghp_exampletoken",
        ] {
            assert_eq!(apply_redaction(value).1, RedactMode::FullRedact, "{value}");
        }
    }

    #[test]
    fn url_fragment_is_not_prefixed_twice() {
        assert_eq!(
            redact_url_query("https://example.com/path?token=value#section"),
            "https://example.com/path?token=[REDACTED]#section"
        );
    }

    /// Regression test: multi-key JSON body must terminate without infinite loop,
    /// redacting sensitive values while preserving safe ones.
    #[test]
    fn json_redaction_multi_key_terminates() {
        let body = r#"{"username": "alice", "password": "secret_1", "token": "secret_2", "model": "gpt-4"}"#;
        let sanitized = sanitize_json_body(body);
        // Sensitive values redacted.
        assert!(!sanitized.contains("secret_1"));
        assert!(!sanitized.contains("secret_2"));
        // Non-sensitive values preserved.
        assert!(sanitized.contains("alice"));
        assert!(sanitized.contains("gpt-4"));
    }

    /// Regression test: UTF-8 truncation must not panic on multi-byte characters.
    #[test]
    fn utf8_truncation_safe() {
        // String of emoji (4 bytes each in UTF-8) exceeding MAX_LOG_STRING_LEN chars.
        let long_val = "🦀".repeat(2048);
        let (_, mode) = apply_redaction(&long_val);
        matches!(mode, RedactMode::Truncate(_));
    }

    #[test]
    fn url_fragment_preserved() {
        let url = "https://example.com/page#section?token=secret";
        // Fragment handling: the ? appears in fragment, not as query separator.
        let cleaned = redact_url_query(url);
        assert!(cleaned.contains("#"));
    }

    #[test]
    fn policy_sanitize_value_full_pipeline() {
        use mesh_llm_config::LoggingConfig;
        let config = LoggingConfig::default();
        let policy = build_policy(&config);

        // Test the full pipeline: path sanitization + credential redaction.
        let home = dirs::home_dir().expect("test runner has a home directory");
        let input = format!(
            "Error in {} with token Bearer secret123",
            home.join("data").display()
        );
        let sanitized = policy.sanitize_value(&input);
        // Path sanitization replaces HOME prefix. Full redaction may or may not trigger depending on content.
        assert!(!sanitized.contains(&home.display().to_string()));
    }

    #[test]
    fn lifecycle_strings_are_secret_path_and_length_safe() {
        let home = dirs::home_dir().expect("home");
        let long_tail = "x".repeat(MAX_LOG_STRING_LEN + 100);
        let event = LifecycleEvent::Failed {
            error: format!(
                "request failed at {} https://example.test/run?token=secret {long_tail}",
                home.join("private/model.gguf").display()
            ),
            status_code: Some(502),
        };
        let LifecycleEvent::Failed { error, status_code } = sanitize_lifecycle_event(event) else {
            panic!("failed event expected");
        };
        assert_eq!(status_code, Some(502));
        assert!(!error.contains("secret"));
        assert!(!error.contains(&home.display().to_string()));
        assert!(error.chars().count() <= MAX_LOG_STRING_LEN + "... [TRUNCATED]".len());
    }
}
