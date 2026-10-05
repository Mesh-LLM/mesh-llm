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
