use std::sync::Arc;

use mesh_llm_log_store::{AuditEntryFilters, AuditEntrySeverity, LogStore, RealClock};

#[test]
fn audit_entry_reads_sanitize_raw_actor_and_action_scalars() {
    // Given: rows written outside the typed repository with malformed and oversized scalars.
    let temp = tempfile::tempdir().expect("create temporary log store");
    let store = LogStore::open(temp.path(), Arc::new(RealClock)).expect("open log store");
    let home = std::env::var("HOME").expect("HOME should be available to the test");
    let malformed_actor = format!("{home}/runtime\nactor");
    let secret_action = "Bearer raw-sql-action-secret";
    let oversized_actor = "a".repeat(1_100);
    let oversized_action = "z".repeat(1_100);
    store
        .conn()
        .execute(
            "INSERT INTO audit_entries \
             (entry_id, request_id, occurred_at, actor, action, detail_json) \
             VALUES (?1, NULL, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                "raw-audit-malformed",
                "2026-08-22T12:00:01.000000000Z",
                malformed_actor,
                secret_action,
                r#"{"severity":"warning"}"#,
            ],
        )
        .expect("insert malformed raw audit row");
    store
        .conn()
        .execute(
            "INSERT INTO audit_entries \
             (entry_id, request_id, occurred_at, actor, action, detail_json) \
             VALUES (?1, NULL, ?2, ?3, ?4, NULL)",
            rusqlite::params![
                "raw-audit-oversized",
                "2026-08-22T12:00:00.000000000Z",
                oversized_actor,
                oversized_action,
            ],
        )
        .expect("insert oversized raw audit row");

    // When: the shared durable read projection maps both rows.
    let page = store
        .list_audit_entries(Some(2), None, AuditEntryFilters::default())
        .expect("list raw audit rows");

    // Then: ordering and detail projection survive while both scalar columns are canonicalized.
    assert_eq!(page.items[0].entry_id, "raw-audit-malformed");
    assert_eq!(page.items[0].source, "~/runtimeactor");
    assert_eq!(page.items[0].code, "[REDACTED]");
    assert_eq!(page.items[0].severity, Some(AuditEntrySeverity::Warning));
    assert_eq!(page.items[1].entry_id, "raw-audit-oversized");
    assert_eq!(page.items[1].source.chars().count(), 1_024);
    assert!(page.items[1].source.ends_with("... [TRUNCATED]"));
    assert_eq!(page.items[1].code.chars().count(), 1_024);
    assert!(page.items[1].code.ends_with("... [TRUNCATED]"));
}
