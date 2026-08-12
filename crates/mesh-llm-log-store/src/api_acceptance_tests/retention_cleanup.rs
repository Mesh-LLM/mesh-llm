use super::*;

// ════════════════════════════════════
//  CASCADE CLEANUP TESTS
// ════════════════════════════════════

#[test]
fn cascade_cleanup_removes_by_cutoff() {
    let (store, _, _tmp) = open_store();

    // Create summaries + events for months Jan(1)..May(5).
    for i in 0..5u32 {
        let month = 1 + i;
        store
            .conn()
            .execute(
                "INSERT INTO summaries (request_id, state, created_at, terminal_at)\n\
                 VALUES (?, 'completed', ?, ?)",
                rusqlite::params![
                    format!("cleanup-summ-{:04}", i),
                    format!("2025-{:02}-15T00:00:00Z", month),
                    format!("2025-{:02}-15T00:00:00Z", month)
                ],
            )
            .unwrap();

        store.conn().execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
            rusqlite::params![format!("ev-{:04}", i), format!("cleanup-summ-{:04}", i),
                format!("2025-{:02}-15T00:00:00Z", month), r#"{"type":"admitted"}"#],
        ).unwrap();

        store.conn().execute(
            "INSERT INTO artifact_pointers (artifact_id, request_id, occurred_at, kind) VALUES (?, ?, ?, 'log')",
            rusqlite::params![format!("art-{:04}", i), format!("cleanup-summ-{:04}", i),
                format!("2025-{:02}-15T00:00:00Z", month)],
        ).unwrap();

        store.conn().execute(
            "INSERT INTO proxy_records (attempt_id, request_id, occurred_at, target) VALUES (?, ?, ?, 'http://example.com')",
            rusqlite::params![format!("proxy-{:04}", i), format!("cleanup-summ-{:04}", i),
                format!("2025-{:02}-15T00:00:00Z", month)],
        ).unwrap();

        // Audit entries with request_id reference (SET NULL on summary delete).
        store.conn().execute(
            "INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action) VALUES (?, ?, ?, 'system', 'create')",
            rusqlite::params![format!("audit-{:04}", i), format!("cleanup-summ-{:04}", i),
                format!("2025-{:02}-15T00:00:00Z", month)],
        ).unwrap();

        // Webhook deliveries with request_id reference (SET NULL on summary delete).
        store.conn().execute(
            "INSERT INTO webhook_deliveries (delivery_id, request_id, terminal_outcome, occurred_at, target_url, attempt_number) VALUES (?, ?, 'completed', ?, 'https://hooks.example', 1)",
            rusqlite::params![format!("wh-{:04}", i), format!("cleanup-summ-{:04}", i),
                format!("2025-{:02}-15T00:00:00Z", month)],
        ).unwrap();
    }

    // Cleanup everything before March (Jan and Feb entries — indices 0,1).
    store
        .cascade_cleanup_before("2025-03-01T00:00:00Z")
        .unwrap();

    let ev_count = store.count_table("lifecycle_events").unwrap();
    assert_eq!(ev_count, 3, "only Mar/Apr/May events remain");

    let art_count = store.count_table("artifact_pointers").unwrap();
    assert_eq!(art_count, 3, "only Mar/Apr/May artifacts remain");

    let proxy_count = store.count_table("proxy_records").unwrap();
    assert_eq!(proxy_count, 3, "only Mar/Apr/May proxy records remain");

    // Terminal Jan/Feb summaries cascade with their owned detail; Mar/Apr/May survive.
    let summ_count = store.count_table("summaries").unwrap();
    assert_eq!(summ_count, 3, "only Mar/Apr/May summaries remain");

    let audit_count = store.count_table("audit_entries").unwrap();
    assert_eq!(audit_count, 3, "old audit rows follow their TTL policy");

    let wh_count = store.count_table("webhook_deliveries").unwrap();
    assert_eq!(wh_count, 3, "old webhook rows follow their TTL policy");

    // Expired standalone rows are removed, not orphaned by the summary
    // cascade through their ON DELETE SET NULL relationship.
    let null_audits: i64 = store.conn().query_row(
        "SELECT COUNT(*) FROM audit_entries WHERE occurred_at < '2025-03-01T00:00:00Z' AND request_id IS NULL",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(null_audits, 0);
    let null_webhooks: i64 = store.conn().query_row(
        "SELECT COUNT(*) FROM webhook_deliveries WHERE occurred_at < '2025-03-01T00:00:00Z' AND request_id IS NULL",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(null_webhooks, 0);

    // Verify Mar/Apr/May audit/webhook rows still reference their summaries.
    let non_null_audits: i64 = store.conn().query_row(
        "SELECT COUNT(*) FROM audit_entries WHERE occurred_at >= '2025-03-01T00:00:00Z' AND request_id IS NOT NULL",
        [], |row| row.get::<_, i64>(0),
    ).unwrap();
    assert_eq!(non_null_audits, 3);
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
            .insert_webhook_delivery(&format!("webhook-{month}"), None, &occurred_at, 1, None)
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
fn cascade_cleanup_reloads_pending_work_without_deleting_an_active_owner() {
    let (store, _, _tmp) = open_store();
    store
        .conn()
        .execute(
            "INSERT INTO summaries (request_id, state, created_at) VALUES ('req-active', 'active', '2025-03-01T00:00:00Z')",
            [],
        )
        .expect("insert retained summary");
    store
        .conn()
        .execute(
            "INSERT INTO artifact_pointers (artifact_id, request_id, occurred_at, kind) VALUES ('art-active', 'req-active', '2025-01-01T00:00:00Z', 'log')",
            [],
        )
        .expect("insert expired artifact pointer");
    store
        .conn()
        .execute(
            "INSERT INTO pending_artifact_deletions (artifact_id, request_id) VALUES ('art-prior', 'req-deleted')",
            [],
        )
        .expect("queue prior artifact deletion");

    let (deleted, pending) = store
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("cleanup queued artifact pointer");

    assert_eq!(deleted, 0);
    assert_eq!(
        pending,
        vec![super::repositories::CascadeArtifactPointer {
            artifact_id: "art-prior".to_owned(),
            request_id: "req-deleted".to_owned(),
        }]
    );
    assert_eq!(store.count_table("artifact_pointers").unwrap(), 1);
}

// ════════════════════════════════════
//  FOREIGN KEY ENFORCEMENT TESTS
// ════════════════════════════════════

#[test]
fn foreign_keys_enforced() {
    let (store, _, _tmp) = open_store();

    // Attempt to insert a lifecycle_event for a nonexistent request_id.
    let result = store.insert_lifecycle_event(
        "nonexistent-request",
        "evt-orph",
        r#"{"type":"admitted"}"#,
        "2025-01-01T00:00:00Z",
    );

    assert!(
        matches!(
            result,
            Err(LogStoreError::ForeignKeyViolation { entity }) if entity == "lifecycle_event"
        ),
        "orphan inserts should report a lifecycle-event foreign-key violation"
    );
}
