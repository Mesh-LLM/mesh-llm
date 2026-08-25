use super::*;

fn parse_detail(raw: &str) -> Value {
    let sanitized = SanitizedAuditDetailJson::sanitize(raw);
    serde_json::from_str(sanitized.as_str()).expect("sanitized detail is valid JSON")
}

fn home_directory() -> String {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .find_map(|variable| std::env::var(variable).ok())
        .filter(|home| !home.is_empty())
        .expect("HOME or USERPROFILE should be available to the test")
}

#[test]
fn audit_detail_sanitization_preserves_safe_json_structure() {
    let raw = r#"{"safe":{"outcome":"completed","count":3},"nested":{"password":"detail-secret"}}"#;
    let sanitized = SanitizedAuditDetailJson::sanitize(raw);
    let value: Value =
        serde_json::from_str(sanitized.as_str()).expect("sanitized detail is valid JSON");
    assert_eq!(value["safe"]["outcome"], "completed");
    assert_eq!(value["safe"]["count"], 3);
    assert!(!sanitized.as_str().contains("detail-secret"));
}

#[test]
fn audit_detail_sanitization_replaces_malformed_sensitive_input() {
    let raw = r#"{"note":"Bearer malformed-credential""#;
    let sanitized = SanitizedAuditDetailJson::sanitize(raw);
    assert_eq!(sanitized.as_str(), r#""[REDACTED]""#);
    assert!(!sanitized.as_str().contains("malformed-credential"));
}

#[test]
fn audit_detail_sanitization_truncates_oversized_raw_input_before_parsing() {
    let raw = serde_json::json!({
        "sentinel": "OVERSIZED-RAW-SENTINEL",
        "padding": "x".repeat(256 * 1024),
    })
    .to_string();
    let sanitized = SanitizedAuditDetailJson::sanitize(&raw);
    assert_eq!(sanitized.as_str(), r#""[TRUNCATED]""#);
    assert!(!sanitized.as_str().contains("OVERSIZED-RAW-SENTINEL"));
}

#[test]
fn audit_detail_sanitization_bounds_object_keys() {
    let oversized_key = "k".repeat(MAX_AUDIT_TEXT_LEN + 1);
    let raw = serde_json::json!({ oversized_key: "safe-value" }).to_string();
    let value = parse_detail(&raw);
    let object = value
        .as_object()
        .expect("sanitized detail remains an object");
    let (key, stored_value) = object.iter().next().expect("bounded entry remains present");
    assert!(key.chars().count() <= MAX_AUDIT_TEXT_LEN);
    assert_eq!(stored_value, "safe-value");
}

#[test]
fn audit_detail_sanitization_removes_private_and_credential_key_text() {
    let home = home_directory();
    let raw = serde_json::json!({
        format!("{home}/private/audit\npath"): "safe-path-value",
        "Bearer object-key-credential": "sensitive-field-value",
    })
    .to_string();
    let value = parse_detail(&raw);
    let keys = value
        .as_object()
        .expect("sanitized detail remains an object")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert!(keys.iter().all(|key| !key.contains(&home)));
    assert!(keys.iter().all(|key| !key.contains("Bearer")));
    assert!(
        keys.iter()
            .all(|key| !key.contains("object-key-credential"))
    );
    assert!(keys.iter().all(|key| !key.contains('\n')));
}

#[test]
fn audit_detail_sanitization_preserves_values_at_deterministic_collision_keys() {
    let home = home_directory();
    let private_key = format!("{home}/collision");
    let canonical_key = "~/collision".to_string();
    let raw = serde_json::json!({
        private_key: "private-path-value",
        canonical_key: "canonical-value",
    })
    .to_string();
    let value = parse_detail(&raw);
    assert_eq!(value["~/collision"], "canonical-value");
    assert_eq!(value["~/collision#2"], "private-path-value");
}

#[test]
fn sensitive_key_contract_redacts_nested_bare_key_names() {
    // Given: nested detail containing every bare sensitive-key spelling.
    let raw = serde_json::json!({
        "nested": {
            "password": "password-value",
            "credential": "credential-value",
            "api_key": "api-key-value",
            "private_key": "private-key-value",
            "authorization": "authorization-value",
        }
    })
    .to_string();

    // When: the durable detail sanitizer processes the document.
    let sanitized = SanitizedAuditDetailJson::sanitize(&raw);

    // Then: neither sensitive names nor their values persist.
    for sensitive in [
        "password",
        "credential",
        "api_key",
        "private_key",
        "authorization",
    ] {
        assert!(!sanitized.as_str().contains(sensitive));
    }
    let value: Value =
        serde_json::from_str(sanitized.as_str()).expect("sanitized detail is valid JSON");
    assert_eq!(value["nested"].as_object().map(Map::len), Some(5));
}

#[test]
fn sensitive_key_contract_preserves_invite_token_exception() {
    // Given: a string value under the invite-token exception key.
    let raw = r#"{"invite_token":"mesh-llm-invite-token-abc123"}"#;

    // When: the durable detail sanitizer processes the document.
    let sanitized = SanitizedAuditDetailJson::sanitize(raw);

    // Then: the key and bounded string value remain available.
    assert_eq!(sanitized.as_str(), raw);
}

#[test]
fn sensitive_key_contract_reserves_canonical_collision_chain() {
    // Given: canonical keys that reserve the first two collision positions.
    let raw = r#"{"[REDACTED]":"canonical","[REDACTED]#2":"reserved","authorization":"auth-value","password":"password-value"}"#;

    // When: sensitive key names normalize in sorted original-key order.
    let value = parse_detail(raw);

    // Then: every entry remains and later collisions use unreserved suffixes.
    assert_eq!(value["[REDACTED]"], "canonical");
    assert_eq!(value["[REDACTED]#2"], "reserved");
    assert_eq!(value["[REDACTED]#3"], "[REDACTED]");
    assert_eq!(value["[REDACTED]#4"], "[REDACTED]");
    assert_eq!(value.as_object().map(Map::len), Some(4));
}

#[test]
fn sensitive_key_contract_is_byte_idempotent() {
    // Given: detail with sensitive-key and canonical-key collisions.
    let raw =
        r#"{"[REDACTED]":"canonical","credential":"credential-value","password":"password-value"}"#;

    // When: the sanitizer processes its own output.
    let first = SanitizedAuditDetailJson::sanitize(raw);
    let second = SanitizedAuditDetailJson::sanitize(first.as_str());

    // Then: the serialized bytes do not change.
    assert_eq!(second.as_str(), first.as_str());
}
