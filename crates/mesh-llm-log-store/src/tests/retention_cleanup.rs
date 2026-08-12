use crate::error::LogStoreError;

use super::store_setup::open_store;

#[test]
fn cascade_cleanup_removes_by_cutoff() {
    let (store, _, _tmp) = open_store();
    for index in 0..5_u32 {
        let month = index + 1;
        let occurred_at = format!("2025-{month:02}-15T00:00:00Z");
        let request_id = format!("cleanup-summ-{index:04}");
        store
            .conn()
            .execute(
                "INSERT INTO summaries (request_id, state, created_at) VALUES (?, 'active', ?)",
                rusqlite::params![request_id, occurred_at],
            )
            .unwrap();
        for (table, id_column, id, extra_columns, extra_values) in [
            (
                "lifecycle_events",
                "event_id",
                format!("ev-{index:04}"),
                ", payload_json",
                ", '{\"type\":\"admitted\"}'",
            ),
            (
                "artifact_pointers",
                "artifact_id",
                format!("art-{index:04}"),
                ", kind",
                ", 'log'",
            ),
            (
                "proxy_records",
                "attempt_id",
                format!("proxy-{index:04}"),
                ", target",
                ", 'http://example.com'",
            ),
        ] {
            store.conn().execute(
                &format!("INSERT INTO {table} ({id_column}, request_id, occurred_at{extra_columns}) VALUES (?, ?, ?{extra_values})"),
                rusqlite::params![id, format!("cleanup-summ-{index:04}"), format!("2025-{month:02}-15T00:00:00Z")],
            ).unwrap();
        }
        store.conn().execute(
            "INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action) VALUES (?, ?, ?, 'system', 'create')",
            rusqlite::params![format!("audit-{index:04}"), format!("cleanup-summ-{index:04}"), format!("2025-{month:02}-15T00:00:00Z")],
        ).unwrap();
        store.conn().execute(
            "INSERT INTO webhook_deliveries (delivery_id, request_id, occurred_at, target_url, attempt_number) VALUES (?, ?, ?, 'https://hooks.example', 1)",
            rusqlite::params![format!("wh-{index:04}"), format!("cleanup-summ-{index:04}"), format!("2025-{month:02}-15T00:00:00Z")],
        ).unwrap();
    }

    store
        .cascade_cleanup_before("2025-03-01T00:00:00Z")
        .unwrap();
    for table in [
        "lifecycle_events",
        "artifact_pointers",
        "proxy_records",
        "summaries",
        "audit_entries",
        "webhook_deliveries",
    ] {
        assert_eq!(store.count_table(table).unwrap(), 3, "{table}");
    }
}

#[test]
fn cascade_cleanup_uses_independent_audit_and_webhook_cutoffs() {
    let (store, _, _tmp) = open_store();
    for month in 1..=3 {
        let occurred_at = format!("2025-{month:02}-15T00:00:00Z");
        store
            .insert_audit_entry(
                &format!("audit-{month}"),
                None,
                &occurred_at,
                "system",
                "test",
                None,
            )
            .unwrap();
        store
            .insert_webhook_delivery(
                &format!("webhook-{month}"),
                None,
                &occurred_at,
                "https://hooks.example",
                1,
                None,
                None,
                None,
            )
            .unwrap();
    }
    store
        .cascade_cleanup_with_retention_cutoffs(
            "2025-01-01T00:00:00Z",
            "2025-03-01T00:00:00Z",
            "2025-02-01T00:00:00Z",
        )
        .unwrap();
    assert_eq!(store.count_table("audit_entries").unwrap(), 1);
    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 2);
}

#[test]
fn cascade_cleanup_counts_artifact_pointers_already_queued_for_deletion() {
    let (store, _, _tmp) = open_store();
    store.conn().execute("INSERT INTO summaries (request_id, state, created_at) VALUES ('req-queued', 'active', '2025-03-01T00:00:00Z')", []).unwrap();
    store.conn().execute("INSERT INTO artifact_pointers (artifact_id, request_id, occurred_at, kind) VALUES ('art-queued', 'req-queued', '2025-01-01T00:00:00Z', 'log')", []).unwrap();
    store.conn().execute("INSERT INTO pending_artifact_deletions (artifact_id, request_id) VALUES ('art-queued', 'req-queued')", []).unwrap();
    let (deleted, pending) = store
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(pending.len(), 1);
    assert_eq!(store.count_table("artifact_pointers").unwrap(), 0);
}

#[test]
fn foreign_keys_enforced() {
    let (store, _, _tmp) = open_store();
    assert!(matches!(
        store.insert_lifecycle_event("nonexistent-request", "evt-orph", r#"{"type":"admitted"}"#, "2025-01-01T00:00:00Z"),
        Err(LogStoreError::ForeignKeyViolation { entity }) if entity == "lifecycle_event"
    ));
}
