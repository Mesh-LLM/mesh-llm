use super::*;
use crate::audit::{AuditCategory, AuditOutcome};
use serde_json::Value;
use std::collections::BTreeMap;

fn event() -> AuditEvent {
    AuditEvent::new(AuditCategory::System, "event", AuditOutcome::Success)
}

fn home_directory() -> String {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .find_map(|variable| std::env::var(variable).ok())
        .filter(|home| !home.is_empty())
        .expect("HOME or USERPROFILE should be available to the test")
}

#[test]
fn audit_scalar_sanitizes_private_paths_controls_credentials_and_bounds() {
    let home = home_directory();
    let private_path = SanitizedAuditScalar::sanitize(&format!("{home}/private\naudit"));
    let credential = SanitizedAuditScalar::sanitize("Bearer audit-secret");
    let oversized = SanitizedAuditScalar::sanitize(&"x".repeat(MAX_AUDIT_TEXT_LEN + 1));

    assert_eq!(private_path.as_str(), "~/privateaudit");
    assert_eq!(credential.as_str(), "[REDACTED]");
    assert!(oversized.as_str().ends_with("... [TRUNCATED]"));
    assert!(oversized.as_str().chars().count() <= MAX_AUDIT_TEXT_LEN);
}

#[test]
fn audit_scalar_preserves_safe_and_fixed_values() {
    for value in ["runtime", "runtime_ready", "[REDACTED]", "[TRUNCATED]"] {
        assert_eq!(SanitizedAuditScalar::sanitize(value).as_str(), value);
    }
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
    let home = home_directory();
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
    assert!(token.chars().count() <= MAX_AUDIT_TEXT_LEN);
}

#[test]
fn metadata_strings_are_recursively_secret_path_and_length_safe() {
    let home = home_directory();
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
    assert_eq!(redacted.metadata["nested"]["[REDACTED]"], "[REDACTED]");
    let note = redacted.metadata["nested"]["note"]
        .as_str()
        .expect("note should remain a string");
    assert!(!note.contains(&home));
    assert!(note.ends_with("... [TRUNCATED]"));
    assert!(note.chars().count() <= MAX_AUDIT_TEXT_LEN);
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

#[test]
fn sensitive_key_contract_matches_top_level_audit_metadata() {
    // Given: equivalent detail JSON and top-level audit metadata.
    let raw = r#"{"[REDACTED]":"canonical","credential":"credential-value","invite_token":"mesh-llm-invite-token-abc123","password":"password-value"}"#;
    let metadata: BTreeMap<String, Value> =
        serde_json::from_str(raw).expect("fixture is a JSON object");
    let mut audit_event = event();
    audit_event.metadata = metadata;

    // When: both audit sanitization paths process the same keys.
    let detail = SanitizedAuditDetailJson::sanitize(raw);
    let redacted_event = redact_secrets(audit_event);

    // Then: their normalized key and value behavior is identical.
    let detail_value: Value =
        serde_json::from_str(detail.as_str()).expect("sanitized detail is valid JSON");
    let event_value =
        serde_json::to_value(redacted_event.metadata).expect("sanitized metadata is serializable");
    assert_eq!(event_value, detail_value);
    assert!(!event_value.to_string().contains("credential"));
    assert!(!event_value.to_string().contains("password"));
}
