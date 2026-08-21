use super::*;

// ════════════════════════════════════
//  EMPTY / SINGLE ITEM PAGINATION TESTS
// ════════════════════════════════════

#[test]
fn empty_table_pagination() {
    let (store, _, _tmp) = open_store();

    let page = store.list_summaries(10, None).unwrap();
    assert!(page.items.is_empty());
    assert!(page.next_cursor.is_none());

    // Also test lifecycle events and artifacts.
    let ev_page = store.list_lifecycle_events(10, None).unwrap();
    assert!(ev_page.items.is_empty());

    let art_page = store.list_artifact_pointers(10, None).unwrap();
    assert!(art_page.items.is_empty());
}

#[test]
fn single_item_pagination() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "only-one",
            Some("llama3"),
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let page = store.list_summaries(10, None).unwrap();
    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].request_id, "only-one");
    assert!(page.next_cursor.is_none());
}

// ════════════════════════════════════
//  SUMMARY STATUS COUNTS TEST
// ════════════════════════════════════

#[test]
fn summary_status_counts() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "s-active-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    // Insert + terminal update.
    store
        .insert_summary(
            "s-completed-1",
            None,
            Some("route-a"),
            Some("provider-x"),
            Some("engine-y"),
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    let payload = r#"{"type":"completed","status_code":200}"#;
    store
        .write_terminal_event(
            "s-completed-1",
            "evt-c1",
            payload,
            "completed",
            Some(200),
            &clock.now(),
        )
        .unwrap();

    // Failed terminal.
    store
        .insert_summary(
            "s-failed-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    let failed_payload = r#"{"type":"failed","error":"timeout"}"#;
    store
        .write_terminal_event(
            "s-failed-1",
            "evt-f1",
            failed_payload,
            "failed",
            None,
            &clock.now(),
        )
        .unwrap();

    let counts = store.count_summaries_by_status().unwrap();
    assert_eq!(counts.len(), 3); // active, completed, failed states

    // Verify specific counts.
    for (state, count) in &counts {
        match state.as_str() {
            "active" => assert_eq!(*count, 1),
            "completed" => assert_eq!(*count, 1),
            "failed" => assert_eq!(*count, 1),
            _ => panic!("unexpected state: {}", state),
        }
    }
}

// ════════════════════════════════════
//  HAPPY PATH INSERT + COUNT TESTS
// ════════════════════════════════════

#[test]
fn artifact_insert_and_count() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    store
        .insert_artifact_pointer(
            "art-1",
            "req-1",
            &clock.now(),
            "log",
            Some(r#"{"size": 42}"#),
        )
        .expect("insert artifact");

    assert_eq!(store.count_table("artifact_pointers").unwrap(), 1);

    // Duplicate PK should fail with AlreadyExists.
    let err = store
        .insert_artifact_pointer("art-1", "req-1", &clock.now(), "log", None)
        .unwrap_err();
    assert!(matches!(err, LogStoreError::AlreadyExists { .. }));
}

#[test]
fn proxy_record_insert_and_count() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    store
        .insert_proxy_record(
            "att-1",
            "req-1",
            &clock.now(),
            "http://target.api",
            Some("provider-x"),
            Some("engine-y"),
            Some(&clock.now()),
            Some(&clock.now()),
            Some(200),
            None,
        )
        .expect("insert proxy record");

    assert_eq!(store.count_table("proxy_records").unwrap(), 1);

    // Duplicate PK fails.
    let err = store
        .insert_proxy_record(
            "att-1",
            "req-1",
            &clock.now(),
            "http://other.api",
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
    assert!(matches!(err, LogStoreError::AlreadyExists { .. }));
}

#[test]
fn audit_entry_insert_and_count() {
    let (store, clock, _tmp) = open_store();

    // With request_id.
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    store
        .insert_summary(
            "req-2",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    store
        .insert_audit_entry(
            "aud-1",
            Some("req-1"),
            &clock.now(),
            "user-alice",
            "model_added",
            Some(r#"{"model":"llama3"}"#),
        )
        .expect("insert audit with request_id");

    // Without request_id (standalone).
    store
        .insert_audit_entry("aud-2", None, &clock.now(), "system", "startup", None)
        .expect("insert audit without request_id");

    assert_eq!(store.count_table("audit_entries").unwrap(), 2);

    // Duplicate PK fails.
    let err = store
        .insert_audit_entry(
            "aud-1",
            Some("req-1"),
            &clock.now(),
            "user-bob",
            "other_action",
            None,
        )
        .unwrap_err();
    assert!(matches!(err, LogStoreError::AlreadyExists { .. }));

    // UNIQUE(request_id, entry_id) — same request_id + different entry_id should work.
    store
        .insert_audit_entry(
            "aud-3",
            Some("req-1"),
            &clock.now(),
            "user-carol",
            "action_3",
            None,
        )
        .expect("different entry_id with same request_id is fine");

    // UNIQUE(request_id, entry_id) — different request_id + different entry_id works.
    store
        .insert_audit_entry(
            "aud-5",
            Some("req-2"),
            &clock.now(),
            "user-carol",
            "action_3",
            None,
        )
        .expect("different request and entry are fine");

    // Different entry_id always works (entry_id is PK).
    store
        .insert_audit_entry(
            "aud-4",
            Some("req-1"),
            &clock.now(),
            "user-dave",
            "action_4",
            None,
        )
        .expect("another unique entry_id with same request_id is fine");

    assert_eq!(store.count_table("audit_entries").unwrap(), 5);
}

#[test]
fn audit_entries_page_by_occurred_at_and_entry_id_without_detail_leakage() {
    let (store, _, _tmp) = open_store();
    let timestamp = "2025-06-15T12:00:00Z";
    for entry_id in ["audit-0001", "audit-0003", "audit-0002"] {
        store
            .insert_audit_entry(
                entry_id,
                None,
                timestamp,
                "runtime",
                "runtime_ready",
                Some(r#"{"severity":"info","secret":"SENTINEL-AUDIT-SECRET"}"#),
            )
            .unwrap();
    }

    let first_page = store
        .list_audit_entries(Some(2), None, AuditEntryFilters::default())
        .unwrap();
    assert_eq!(
        first_page
            .items
            .iter()
            .map(|entry| entry.entry_id.as_str())
            .collect::<Vec<_>>(),
        vec!["audit-0003", "audit-0002"]
    );
    assert_eq!(first_page.items[0].severity, Some(AuditEntrySeverity::Info));
    assert!(
        !format!("{:?}", first_page.items).contains("SENTINEL-AUDIT-SECRET"),
        "detail_json must never cross the AuditEntryRow boundary"
    );

    let second_page = store
        .list_audit_entries(
            Some(2),
            first_page.next_cursor.as_deref(),
            AuditEntryFilters::default(),
        )
        .unwrap();
    assert_eq!(second_page.items.len(), 1);
    assert_eq!(second_page.items[0].entry_id, "audit-0001");
    assert!(second_page.next_cursor.is_none());
}

#[test]
fn audit_entries_reconcile_forward_by_durable_sequence() {
    let (store, clock, tmp) = open_store();
    let external_writer = LogStore::open(tmp.path(), clock).expect("open independent writer");
    for entry_id in ["audit-sequence-1", "audit-sequence-2", "audit-sequence-3"] {
        external_writer
            .insert_audit_entry(
                entry_id,
                None,
                "2026-08-12T12:00:00Z",
                "runtime",
                "runtime_ready",
                Some(r#"{"severity":"info"}"#),
            )
            .unwrap();
    }

    let first = store
        .list_audit_entries_after_sequence(0, 2, AuditEntryFilters::default())
        .unwrap();
    assert_eq!(
        first
            .iter()
            .map(|entry| entry.entry_id.as_str())
            .collect::<Vec<_>>(),
        ["audit-sequence-1", "audit-sequence-2"]
    );
    let second = store
        .list_audit_entries_after_sequence(
            u64::try_from(first[1].sequence).unwrap(),
            2,
            AuditEntryFilters::default(),
        )
        .unwrap();
    assert_eq!(second[0].entry_id, "audit-sequence-3");
}

#[test]
fn audit_entries_project_only_versioned_bounded_context() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-context-0001",
            None,
            "2026-08-12T12:00:00Z",
            "runtime",
            "runtime_model_ready",
            Some(
                r#"{"severity":"info","context_version":1,"subject_kind":"model","subject_id":"local-gguf/sha256-safe","operation_id":"runtime-7","outcome":"ready","duration_ms":42,"numeric_summaries":{"bytes":4096},"secret":"SENTINEL-AUDIT-SECRET"}"#,
            ),
        )
        .unwrap();

    let page = store
        .list_audit_entries(Some(1), None, AuditEntryFilters::default())
        .unwrap();
    let row = &page.items[0];
    assert_eq!(row.context_version, Some(1));
    assert_eq!(row.subject_kind.as_deref(), Some("model"));
    assert_eq!(row.subject_id.as_deref(), Some("local-gguf/sha256-safe"));
    assert_eq!(row.operation_id.as_deref(), Some("runtime-7"));
    assert_eq!(row.outcome.as_deref(), Some("ready"));
    assert_eq!(row.duration_ms, Some(42));
    assert_eq!(row.numeric_summaries.get("bytes"), Some(&4096));
    assert!(!format!("{row:?}").contains("SENTINEL-AUDIT-SECRET"));
}

#[test]
fn audit_entry_query_rejects_malformed_cursor_and_out_of_range_limit() {
    let (store, _, _tmp) = open_store();

    let cursor_error = store
        .list_audit_entries(Some(1), Some("not-a-cursor"), AuditEntryFilters::default())
        .unwrap_err();
    assert!(matches!(cursor_error, LogStoreError::CursorMalformed(_)));

    for limit in [Some(0), Some(101)] {
        let limit_error = store
            .list_audit_entries(limit, None, AuditEntryFilters::default())
            .unwrap_err();
        assert!(matches!(limit_error, LogStoreError::InvalidQuery(_)));
    }
}

#[test]
fn audit_entry_query_handles_nullable_request_empty_page_and_allowlisted_filters() {
    let (store, _, _tmp) = open_store();
    let empty_page = store
        .list_audit_entries(None, None, AuditEntryFilters::default())
        .unwrap();
    assert!(empty_page.items.is_empty());
    assert!(empty_page.next_cursor.is_none());

    store
        .insert_audit_entry(
            "audit-null-request",
            None,
            "2025-06-15T12:00:00Z",
            "runtime",
            "runtime_ready",
            Some(r#"{"severity":"info"}"#),
        )
        .unwrap();
    store
        .insert_audit_entry(
            "audit-mesh-warning",
            None,
            "2025-06-15T12:00:01Z",
            "mesh",
            "mesh_quic_handler_failed",
            Some(r#"{"severity":"warning"}"#),
        )
        .unwrap();

    let page = store
        .list_audit_entries(
            None,
            None,
            AuditEntryFilters {
                source: Some(AuditEntrySource::Mesh),
                severity: Some(AuditEntrySeverity::Warning),
                ..AuditEntryFilters::default()
            },
        )
        .unwrap();
    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].entry_id, "audit-mesh-warning");
    assert!(page.items[0].request_id.is_none());
}

#[test]
fn nullable_column_conversion_errors_are_not_silenced() {
    let (store, clock, _tmp) = open_store();
    store
        .insert_summary(
            "req-invalid-column",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("insert summary");
    store
        .conn()
        .execute(
            "UPDATE summaries SET status_code = 'not-an-integer' WHERE request_id = 'req-invalid-column'",
            [],
        )
        .expect("store invalid status code");

    assert!(matches!(
        store.get_summary("req-invalid-column"),
        Err(LogStoreError::QueryFailed(_))
    ));
    assert!(matches!(
        store.list_summaries(10, None),
        Err(LogStoreError::QueryFailed(_))
    ));

    store
        .conn()
        .execute(
            "UPDATE summaries SET status_code = NULL WHERE request_id = 'req-invalid-column'",
            [],
        )
        .expect("restore nullable summary column");
    store
        .insert_artifact_pointer(
            "art-invalid-column",
            "req-invalid-column",
            &clock.now(),
            "log",
            None,
        )
        .expect("insert artifact pointer");
    store
        .conn()
        .execute(
            "UPDATE artifact_pointers SET checksum = x'00' WHERE artifact_id = 'art-invalid-column'",
            [],
        )
        .expect("store invalid checksum type");

    assert!(matches!(
        store.get_artifact_pointer("art-invalid-column"),
        Err(LogStoreError::QueryFailed(_))
    ));
    assert!(matches!(
        store.list_artifact_pointers_for_request("req-invalid-column"),
        Err(LogStoreError::QueryFailed(_))
    ));
    assert!(matches!(
        store.list_artifact_pointers(10, None),
        Err(LogStoreError::QueryFailed(_))
    ));
}
