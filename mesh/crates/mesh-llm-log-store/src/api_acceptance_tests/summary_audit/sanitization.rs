use super::*;

fn home_directory() -> String {
    home_directory_from(|variable| std::env::var(variable).ok())
        .expect("HOME or USERPROFILE should be available to the test")
}

fn home_directory_from(mut variable_value: impl FnMut(&str) -> Option<String>) -> Option<String> {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .find_map(|variable| variable_value(variable).filter(|value| !value.is_empty()))
}

#[test]
fn home_directory_ignores_empty_home_before_userprofile() {
    let selected = home_directory_from(|variable| match variable {
        "HOME" => Some(String::new()),
        "USERPROFILE" => Some(r"C:\Users\operator".to_owned()),
        _ => None,
    });

    assert_eq!(selected.as_deref(), Some(r"C:\Users\operator"));
}

#[test]
fn insert_audit_entry_sanitizes_detail_before_sqlite_persistence() {
    // Given: valid structured detail containing every sensitive value class.
    let (store, clock, _tmp) = open_store();
    let home = home_directory();
    let detail = serde_json::json!({
        "safe": {"outcome": "completed", "count": 3},
        "nested": {"password": "store-secret-value"},
        "note": "request failed with Bearer store-credential-value",
        "path": format!("{home}/private/audit.json"),
    })
    .to_string();

    // When: the public repository method persists the detail.
    store
        .insert_audit_entry(
            "audit-sanitized-detail",
            None,
            &clock.now(),
            "runtime",
            "runtime_ready",
            Some(&detail),
        )
        .expect("insert audit detail");
    let raw_detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE entry_id = 'audit-sanitized-detail'",
            [],
            |row| row.get(0),
        )
        .expect("read raw audit detail");

    // Then: the raw column retains safe structure without sensitive originals.
    assert!(!raw_detail.contains("store-secret-value"));
    assert!(!raw_detail.contains("Bearer"));
    assert!(!raw_detail.contains("store-credential-value"));
    assert!(!raw_detail.contains(&home));
    let stored: serde_json::Value =
        serde_json::from_str(&raw_detail).expect("stored detail remains valid JSON");
    assert_eq!(stored["safe"]["outcome"], "completed");
    assert_eq!(stored["safe"]["count"], 3);
}

#[test]
fn insert_audit_entry_sanitizes_actor_and_action_before_sqlite_persistence() {
    let (store, clock, _tmp) = open_store();
    let home = home_directory();
    let source = format!("{home}/runtime\nsource");
    let code = format!("Bearer {}", "x".repeat(1_100));

    store
        .insert_audit_entry(
            "audit-sanitized-scalars",
            None,
            &clock.now(),
            &source,
            &code,
            None,
        )
        .expect("insert audit scalars");
    store
        .insert_audit_entry(
            "audit-bounded-scalars",
            None,
            &clock.now(),
            "runtime",
            &"x".repeat(1_100),
            None,
        )
        .expect("insert oversized audit scalar");
    let stored: (String, String) = store
        .conn()
        .query_row(
            "SELECT actor, action FROM audit_entries WHERE entry_id = 'audit-sanitized-scalars'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("read raw audit scalars");

    assert_eq!(stored.0, "~/runtimesource");
    assert_eq!(stored.1, "[REDACTED]");
    assert!(!stored.0.contains(&home));
    assert!(stored.0.chars().count() <= 1_024);
    assert!(stored.1.chars().count() <= 1_024);
    let bounded: String = store
        .conn()
        .query_row(
            "SELECT action FROM audit_entries WHERE entry_id = 'audit-bounded-scalars'",
            [],
            |row| row.get(0),
        )
        .expect("read bounded audit scalar");
    assert!(bounded.ends_with("... [TRUNCATED]"));
    assert!(bounded.chars().count() <= 1_024);
}

#[test]
fn insert_audit_entry_replaces_malformed_sensitive_detail_before_sqlite_persistence() {
    // Given: malformed JSON containing a credential marker and value.
    let (store, clock, _tmp) = open_store();
    let malformed = r#"{"note":"Bearer malformed-store-credential""#;

    // When: the public repository method persists the detail.
    store
        .insert_audit_entry(
            "audit-malformed-detail",
            None,
            &clock.now(),
            "runtime",
            "runtime_ready",
            Some(malformed),
        )
        .expect("insert malformed audit detail");
    let raw_detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE entry_id = 'audit-malformed-detail'",
            [],
            |row| row.get(0),
        )
        .expect("read raw malformed audit detail");

    // Then: SQLite contains only the fixed valid-JSON replacement.
    assert_eq!(raw_detail, r#""[REDACTED]""#);
    assert!(!raw_detail.contains("Bearer"));
    assert!(!raw_detail.contains("malformed-store-credential"));
}

#[test]
fn insert_audit_entry_bounds_raw_detail_and_sanitizes_colliding_keys_before_persistence() {
    // Given: oversized raw detail plus valid detail with unsafe and colliding oversized keys.
    let (store, clock, _tmp) = open_store();
    let home = home_directory();
    let oversized_raw = serde_json::json!({
        "sentinel": "RAW-SQLITE-OVERSIZED-SENTINEL",
        "padding": "x".repeat(256 * 1024),
    })
    .to_string();
    let collision_prefix = "k".repeat(1_024);
    let unsafe_keys = serde_json::json!({
        format!("{home}/private/audit\nkey"): "path-value",
        "Bearer raw-store-key-private": "sensitive-field-value",
        format!("{collision_prefix}a"): "collision-a",
        format!("{collision_prefix}b"): "collision-b",
    })
    .to_string();

    // When: the public repository method persists both raw detail documents.
    store
        .insert_audit_entry(
            "audit-oversized-raw-detail",
            None,
            &clock.now(),
            "runtime",
            "runtime_ready",
            Some(&oversized_raw),
        )
        .expect("insert oversized raw detail");
    store
        .insert_audit_entry(
            "audit-unsafe-key-detail",
            None,
            &clock.now(),
            "runtime",
            "runtime_ready",
            Some(&unsafe_keys),
        )
        .expect("insert unsafe key detail");
    let oversized_stored: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE entry_id = 'audit-oversized-raw-detail'",
            [],
            |row| row.get(0),
        )
        .expect("read oversized raw detail");
    let keys_stored: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE entry_id = 'audit-unsafe-key-detail'",
            [],
            |row| row.get(0),
        )
        .expect("read unsafe key detail");

    // Then: SQLite retains no oversized sentinel or unsafe key text and keeps both safe values.
    assert_eq!(oversized_stored, r#""[TRUNCATED]""#);
    assert!(!oversized_stored.contains("RAW-SQLITE-OVERSIZED-SENTINEL"));
    assert!(!keys_stored.contains(&home));
    assert!(!keys_stored.contains("Bearer"));
    assert!(!keys_stored.contains("raw-store-key-private"));
    let stored: serde_json::Value =
        serde_json::from_str(&keys_stored).expect("stored detail remains valid JSON");
    let object = stored.as_object().expect("stored detail remains an object");
    assert!(object.keys().all(|key| key.chars().count() <= 1_024));
    assert!(object.values().any(|value| value == "collision-a"));
    assert!(object.values().any(|value| value == "collision-b"));
}
