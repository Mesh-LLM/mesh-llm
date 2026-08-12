use crate::error::LogStoreError;

use super::store_setup::open_store;

#[test]
fn summary_status_counts() {
    let (store, clock, _tmp) = open_store();
    for request_id in ["s-active-1", "s-completed-1", "s-failed-1"] {
        store
            .insert_summary(
                request_id,
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
    }
    store
        .write_terminal_event(
            "s-completed-1",
            "evt-c1",
            r#"{"type":"completed","status_code":200}"#,
            "completed",
            &clock.now(),
        )
        .unwrap();
    store
        .write_terminal_event(
            "s-failed-1",
            "evt-f1",
            r#"{"type":"failed","error":"timeout"}"#,
            "failed",
            &clock.now(),
        )
        .unwrap();

    let counts = store.count_summaries_by_status().unwrap();
    assert_eq!(counts.len(), 3);
    for (state, count) in counts {
        assert_eq!(count, 1, "{state}");
    }
}

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
        .unwrap();
    assert_eq!(store.count_table("artifact_pointers").unwrap(), 1);
    assert!(matches!(
        store.insert_artifact_pointer("art-1", "req-1", &clock.now(), "log", None),
        Err(LogStoreError::AlreadyExists { .. })
    ));
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
        .unwrap();
    assert_eq!(store.count_table("proxy_records").unwrap(), 1);
    assert!(matches!(
        store.insert_proxy_record(
            "att-1",
            "req-1",
            &clock.now(),
            "http://other.api",
            None,
            None,
            None,
            None,
            None,
            None
        ),
        Err(LogStoreError::AlreadyExists { .. })
    ));
}

#[test]
fn audit_entry_insert_and_count() {
    let (store, clock, _tmp) = open_store();
    for request_id in ["req-1", "req-2"] {
        store
            .insert_summary(
                request_id,
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
    }
    store
        .insert_audit_entry(
            "aud-1",
            Some("req-1"),
            &clock.now(),
            "user-alice",
            "model_added",
            Some(r#"{"model":"llama3"}"#),
        )
        .unwrap();
    store
        .insert_audit_entry("aud-2", None, &clock.now(), "system", "startup", None)
        .unwrap();
    assert!(matches!(
        store.insert_audit_entry(
            "aud-1",
            Some("req-1"),
            &clock.now(),
            "user-bob",
            "other_action",
            None
        ),
        Err(LogStoreError::AlreadyExists { .. })
    ));
    for (entry_id, request_id) in [("aud-3", "req-1"), ("aud-4", "req-1"), ("aud-5", "req-2")] {
        store
            .insert_audit_entry(
                entry_id,
                Some(request_id),
                &clock.now(),
                "user-carol",
                "action",
                None,
            )
            .unwrap();
    }
    assert_eq!(store.count_table("audit_entries").unwrap(), 5);
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
        .unwrap();
    store.conn().execute("UPDATE summaries SET status_code = 'not-an-integer' WHERE request_id = 'req-invalid-column'", []).unwrap();
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
        .unwrap();
    store
        .insert_artifact_pointer(
            "art-invalid-column",
            "req-invalid-column",
            &clock.now(),
            "log",
            None,
        )
        .unwrap();
    store.conn().execute("UPDATE artifact_pointers SET checksum = x'00' WHERE artifact_id = 'art-invalid-column'", []).unwrap();
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

#[test]
fn webhook_delivery_insert_and_count() {
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
        .insert_webhook_delivery(
            "wh-1",
            Some("req-1"),
            &clock.now(),
            "https://example.com/hook",
            1,
            Some(200),
            Some(r#"{"ok":true}"#),
            None,
        )
        .unwrap();
    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 1);
    assert!(matches!(
        store.insert_webhook_delivery(
            "wh-1",
            Some("req-1"),
            &clock.now(),
            "https://other.com/hook",
            2,
            None,
            None,
            None
        ),
        Err(LogStoreError::AlreadyExists { .. })
    ));
    store
        .insert_webhook_delivery(
            "wh-2",
            None,
            &clock.now(),
            "https://standalone.com/hook",
            1,
            Some(500),
            None,
            Some("connection refused"),
        )
        .unwrap();
    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 2);
}

#[test]
fn cleanup_run_insert_and_count() {
    let (store, clock, _tmp) = open_store();
    store
        .insert_cleanup_run(
            "cr-1",
            &clock.now(),
            "daily-cleanup",
            "2025-01-01T00:00:00Z",
            42,
            Some(150),
        )
        .unwrap();
    assert_eq!(store.count_table("cleanup_runs").unwrap(), 1);
    assert!(matches!(
        store.insert_cleanup_run(
            "cr-1",
            &clock.now(),
            "other-policy",
            "2025-02-01T00:00:00Z",
            10,
            None
        ),
        Err(LogStoreError::AlreadyExists { .. })
    ));
}

#[test]
fn has_terminal_event_detects_correctly() {
    let (store, clock, _tmp) = open_store();
    store
        .insert_summary(
            "term-s1",
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
    assert!(!store.has_terminal_event("term-s1").unwrap());
    store
        .insert_lifecycle_event(
            "term-s1",
            "evt-term",
            r#"{"type":"completed","status_code":200}"#,
            &clock.now(),
        )
        .unwrap();
    assert!(store.has_terminal_event("term-s1").unwrap());
    store
        .insert_lifecycle_event(
            "term-s1",
            "evt-admit",
            r#"{"type":"admitted","model":"llama3"}"#,
            &clock.now(),
        )
        .unwrap();
    store
        .insert_summary(
            "term-s2",
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
    assert!(!store.has_terminal_event("term-s2").unwrap());
}

#[test]
fn list_events_for_summary_ordered_chronologically() {
    let (store, _, _tmp) = open_store();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    for (event_id, occurred_at, payload) in [
        ("evt-c", "2025-03-01T00:00:00Z", r#"{"type":"completed"}"#),
        ("evt-a", "2025-01-01T00:00:00Z", r#"{"type":"admitted"}"#),
        (
            "evt-b",
            "2025-02-01T00:00:00Z",
            r#"{"type":"stream_started"}"#,
        ),
    ] {
        store.conn().execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, 'req-1', ?, ?)",
            rusqlite::params![event_id, occurred_at, payload],
        ).unwrap();
    }
    let events = store.list_events_for_summary("req-1").unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.event_id.as_str())
            .collect::<Vec<_>>(),
        ["evt-a", "evt-b", "evt-c"]
    );
}
