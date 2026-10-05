use super::*;

#[tokio::test]
async fn audit_list_sanitizes_raw_actor_and_action_before_rest_serialization() {
    // Given: a raw SQLite row that bypasses the typed write sanitizer.
    let (_temp, state) = runtime();
    let raw_actor = "Bearer rest-raw-actor-secret";
    let raw_action = format!("REST-RAW-ACTION-SENTINEL-{}", "x".repeat(1_100));
    state
        .store()
        .expect("store")
        .conn()
        .execute(
            "INSERT INTO audit_entries \
             (entry_id, request_id, occurred_at, actor, action, detail_json) \
             VALUES (?1, NULL, ?2, ?3, ?4, ?5)",
            (
                "00000000-0000-4000-8000-000000000099",
                "2026-08-22T12:00:00.000000000Z",
                raw_actor,
                raw_action.as_str(),
                r#"{"severity":"error"}"#,
            ),
        )
        .expect("insert raw audit row");

    // When: the REST route reads and serializes the durable row.
    let page = list_audits(&state, "/api/logs/audit?limit=1")
        .await
        .expect("list raw audit row");
    let json = serde_json::to_value(page).expect("serialize raw audit page");
    let row = &json["items"][0];

    // Then: the API exposes only canonicalized scalar values.
    assert_eq!(row["source"], "[REDACTED]");
    assert_eq!(row["severity"], "error");
    let code = row["code"].as_str().expect("audit code string");
    assert_eq!(code.chars().count(), 1_024);
    assert!(code.ends_with("... [TRUNCATED]"));
    let wire = json.to_string();
    assert!(!wire.contains(raw_actor));
    assert!(!wire.contains(&raw_action));
}
