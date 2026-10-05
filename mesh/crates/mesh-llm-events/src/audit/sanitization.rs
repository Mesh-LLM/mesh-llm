//! Audit-event privacy and size-boundary enforcement.

use super::AuditEvent;

mod detail;

pub use detail::SanitizedAuditDetailJson;

/// Free-form audit scalar that has passed the shared privacy and size policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SanitizedAuditScalar(String);

impl SanitizedAuditScalar {
    pub fn sanitize(value: &str) -> Self {
        Self(sanitize_audit_text(value, true))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests;

const SECRET_PATTERNS: &[&str] = &[
    "token",
    "password",
    "secret",
    "key",
    "credential",
    "auth",
    "bearer",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "certificate",
];

pub(super) const MAX_AUDIT_TEXT_LEN: usize = 1024;
pub(super) const MAX_AUDIT_DETAIL_RAW_BYTES: usize = 256 * 1024;
pub(super) const MAX_AUDIT_METADATA_NODES: usize = 256;
pub(super) const MAX_AUDIT_METADATA_DEPTH: usize = 8;
const AUDIT_TRUNCATION_SUFFIX: &str = "... [TRUNCATED]";

/// Redact sensitive fields and bound untrusted audit metadata before emission.
pub(super) fn redact_secrets(event: AuditEvent) -> AuditEvent {
    let mut redacted = event;
    redacted.metadata = detail::sanitize_metadata(std::mem::take(&mut redacted.metadata));
    redacted.action = redact_audit_text(&redacted.action);
    redacted.resource = redacted.resource.as_deref().map(redact_audit_text);
    redacted.actor = redacted.actor.as_deref().map(redact_audit_text);
    redacted.error = redacted.error.as_deref().map(redact_audit_text);
    redacted
}

pub(super) fn sanitize_audit_key(key: &str) -> String {
    if is_invite_token_key(key) {
        return redact_audit_text(key);
    }
    if is_sensitive_key(key) {
        return "[REDACTED]".to_owned();
    }
    redact_audit_text(key)
}

pub(super) fn is_invite_token_key(key: &str) -> bool {
    let compact = key
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    compact.ends_with("invitetoken")
}

pub(super) fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_lowercase();
    SECRET_PATTERNS.iter().any(|pattern| key.contains(pattern))
}

fn redact_audit_text(value: &str) -> String {
    sanitize_audit_text(value, true)
}

pub(super) fn sanitize_audit_text(value: &str, redact_secret_values: bool) -> String {
    let mut sanitized = value.to_string();
    for variable in ["HOME", "USERPROFILE"] {
        if let Ok(home) = std::env::var(variable)
            && !home.is_empty()
        {
            sanitized = sanitized.replace(&home, "~");
        }
    }
    sanitized.retain(|character| !character.is_control());

    if redact_secret_values && contains_secret_value(&sanitized) {
        return "[REDACTED]".to_owned();
    }
    if sanitized.chars().count() > MAX_AUDIT_TEXT_LEN {
        let prefix_len = MAX_AUDIT_TEXT_LEN - AUDIT_TRUNCATION_SUFFIX.chars().count();
        let prefix = sanitized.chars().take(prefix_len).collect::<String>();
        return format!("{prefix}{AUDIT_TRUNCATION_SUFFIX}");
    }
    sanitized
}

fn contains_secret_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "bearer ",
        "basic ",
        "sk-",
        "sk_",
        "ghp_",
        "password=",
        "password:",
        "secret=",
        "secret:",
        "api_key=",
        "api_key:",
        "access_token=",
        "access_token:",
        "auth_token=",
        "auth_token:",
        "refresh_token=",
        "refresh_token:",
        "private_key=",
        "private_key:",
    ];
    MARKERS
        .iter()
        .any(|marker| contains_marker_at_token_boundary(&lower, marker))
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
