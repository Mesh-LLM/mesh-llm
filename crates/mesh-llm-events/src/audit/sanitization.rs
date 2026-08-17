//! Audit-event privacy and size-boundary enforcement.

use super::AuditEvent;
use serde_json::Value;

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

const MAX_AUDIT_TEXT_LEN: usize = 1024;
const MAX_AUDIT_METADATA_NODES: usize = 256;
const MAX_AUDIT_METADATA_DEPTH: usize = 8;

/// Redact sensitive fields and bound untrusted audit metadata before emission.
pub(super) fn redact_secrets(event: AuditEvent) -> AuditEvent {
    let mut redacted = event;

    // Bound the complete metadata tree as well as each individual string. This
    // keeps one hostile event from bypassing rotation with a wide/deep value.
    let metadata = std::mem::take(&mut redacted.metadata);
    let mut remaining_nodes = MAX_AUDIT_METADATA_NODES;
    for (key, mut value) in metadata {
        if remaining_nodes == 0 {
            break;
        }
        remaining_nodes -= 1;
        redact_json_value(Some(&key), &mut value, 0, &mut remaining_nodes);
        redacted.metadata.insert(key, value);
    }

    redacted.action = redact_audit_text(&redacted.action);
    redacted.resource = redacted.resource.as_deref().map(redact_audit_text);
    redacted.actor = redacted.actor.as_deref().map(redact_audit_text);
    redacted.error = redacted.error.as_deref().map(redact_audit_text);

    redacted
}

fn redact_json_value(
    key: Option<&str>,
    value: &mut Value,
    depth: usize,
    remaining_nodes: &mut usize,
) {
    if let Value::String(token) = value
        && key.is_some_and(is_invite_token_key)
    {
        *token = sanitize_audit_text(token, false);
        return;
    }
    if key.is_some_and(is_sensitive_key) {
        *value = Value::String("[REDACTED]".to_string());
        return;
    }
    if depth >= MAX_AUDIT_METADATA_DEPTH && matches!(value, Value::Object(_) | Value::Array(_)) {
        *value = Value::String("[TRUNCATED]".to_string());
        return;
    }
    match value {
        Value::Object(object) => {
            let original = std::mem::take(object);
            for (nested_key, mut nested_value) in original {
                if *remaining_nodes == 0 {
                    break;
                }
                *remaining_nodes -= 1;
                redact_json_value(
                    Some(&nested_key),
                    &mut nested_value,
                    depth + 1,
                    remaining_nodes,
                );
                object.insert(nested_key, nested_value);
            }
        }
        Value::Array(values) => {
            let original = std::mem::take(values);
            for mut nested_value in original {
                if *remaining_nodes == 0 {
                    break;
                }
                *remaining_nodes -= 1;
                redact_json_value(None, &mut nested_value, depth + 1, remaining_nodes);
                values.push(nested_value);
            }
        }
        Value::String(string) => {
            *string = sanitize_audit_text(string, true);
        }
        _ => {}
    }
}

fn is_invite_token_key(key: &str) -> bool {
    let compact = key
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    compact.ends_with("invitetoken")
}

fn is_sensitive_key(key: &str) -> bool {
    let key = key.to_lowercase();
    SECRET_PATTERNS.iter().any(|pattern| key.contains(pattern))
}

fn redact_audit_text(value: &str) -> String {
    sanitize_audit_text(value, true)
}

fn sanitize_audit_text(value: &str, redact_secret_values: bool) -> String {
    let mut sanitized = value.to_string();
    for variable in ["HOME", "USERPROFILE"] {
        if let Ok(home) = std::env::var(variable)
            && !home.is_empty()
        {
            sanitized = sanitized.replace(&home, "~");
        }
    }

    if redact_secret_values && contains_secret_value(&sanitized) {
        return "[REDACTED]".to_string();
    }

    if sanitized.chars().count() > MAX_AUDIT_TEXT_LEN {
        let prefix: String = sanitized.chars().take(MAX_AUDIT_TEXT_LEN).collect();
        return format!("{prefix}... [TRUNCATED]");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditCategory, AuditOutcome};

    fn event() -> AuditEvent {
        AuditEvent::new(AuditCategory::System, "event", AuditOutcome::Success)
    }

    #[test]
    fn nested_metadata_is_recursively_redacted() {
        let event = event().with_metadata(
            "changes",
            serde_json::json!({
                "old_value": {"api_token": "secret-one"},
                "new_value": [{"PASSWORD": "secret-two"}, "Bearer secret-three"]
            }),
        );
        let serialized = redact_secrets(event).to_json_line().unwrap();
        assert!(!serialized.contains("secret-one"));
        assert!(!serialized.contains("secret-two"));
        assert!(!serialized.contains("secret-three"));
    }

    #[test]
    fn invite_token_metadata_is_preserved() {
        let redacted = redact_secrets(event().with_metadata(
            "invite_token",
            serde_json::json!("mesh-llm-invite-token-abc123"),
        ));

        assert_eq!(
            redacted.metadata["invite_token"],
            "mesh-llm-invite-token-abc123"
        );
    }

    #[test]
    fn invite_token_metadata_is_path_sanitized_and_bounded() {
        let home = ["HOME", "USERPROFILE"]
            .into_iter()
            .find_map(|variable| std::env::var(variable).ok())
            .filter(|home| !home.is_empty())
            .expect("HOME or USERPROFILE should be available to the test");
        let redacted = redact_secrets(event().with_metadata(
            "invite_token",
            serde_json::json!(format!("{home}/{}", "x".repeat(MAX_AUDIT_TEXT_LEN + 1))),
        ));
        let token = redacted.metadata["invite_token"]
            .as_str()
            .expect("invite token should remain a string");

        assert!(!token.contains(&home));
        assert!(token.starts_with("~/"));
        assert!(token.ends_with("... [TRUNCATED]"));
        assert!(token.chars().count() <= MAX_AUDIT_TEXT_LEN + "... [TRUNCATED]".len());
    }

    #[test]
    fn metadata_strings_are_recursively_secret_path_and_length_safe() {
        let home = ["HOME", "USERPROFILE"]
            .into_iter()
            .find_map(|variable| std::env::var(variable).ok())
            .filter(|home| !home.is_empty())
            .expect("HOME or USERPROFILE should be available to the test");
        let redacted = redact_secrets(
            event()
                .with_metadata("refresh", serde_json::json!("refresh_token=secret-value"))
                .with_metadata(
                    "nested",
                    serde_json::json!({
                        "note": format!("{} {}", home, "x".repeat(MAX_AUDIT_TEXT_LEN + 1)),
                        "credential": "private_key=secret-key"
                    }),
                ),
        );

        assert_eq!(redacted.metadata["refresh"], "[REDACTED]");
        assert_eq!(redacted.metadata["nested"]["credential"], "[REDACTED]");
        let note = redacted.metadata["nested"]["note"]
            .as_str()
            .expect("note should remain a string");
        assert!(!note.contains(&home));
        assert!(note.ends_with("... [TRUNCATED]"));
        assert!(note.chars().count() <= MAX_AUDIT_TEXT_LEN + "... [TRUNCATED]".len());
    }

    #[test]
    fn metadata_collections_are_bounded_before_serialization() {
        let values = (0..1_000)
            .map(|index| serde_json::json!({"index": index, "note": "x".repeat(2_048)}))
            .collect::<Vec<_>>();
        let redacted = redact_secrets(event().with_metadata("items", Value::Array(values)));
        let items = redacted.metadata["items"]
            .as_array()
            .expect("items should remain an array");

        assert!(items.len() <= MAX_AUDIT_METADATA_NODES);
        assert!(redacted.to_json_line().unwrap().len() < 300_000);
    }

    #[test]
    fn invite_token_exception_does_not_bypass_nested_sanitization() {
        let redacted = redact_secrets(event().with_metadata(
            "invite_token",
            serde_json::json!({"password": "must-not-survive"}),
        ));

        assert_eq!(redacted.metadata["invite_token"], "[REDACTED]");
    }

    #[test]
    fn free_form_audit_fields_are_redacted_and_bounded() {
        let event = AuditEvent::new(
            AuditCategory::System,
            "restart failed: Bearer audit-secret".to_string(),
            AuditOutcome::Failure,
        )
        .with_resource("model sk-live-example")
        .with_actor("ghp_exampletoken")
        .with_error("password=example-password")
        .with_metadata("note", serde_json::json!("Basic YWxpY2U6cGFzcw=="));
        let redacted = redact_secrets(event);

        assert_eq!(redacted.action, "[REDACTED]");
        assert_eq!(redacted.resource.as_deref(), Some("[REDACTED]"));
        assert_eq!(redacted.actor.as_deref(), Some("[REDACTED]"));
        assert_eq!(redacted.error.as_deref(), Some("[REDACTED]"));
        assert_eq!(redacted.metadata["note"], "[REDACTED]");

        let long_action = AuditEvent::new(
            AuditCategory::System,
            "x".repeat(MAX_AUDIT_TEXT_LEN + 1),
            AuditOutcome::Success,
        );
        assert!(
            redact_secrets(long_action)
                .action
                .ends_with("... [TRUNCATED]")
        );
    }
}
