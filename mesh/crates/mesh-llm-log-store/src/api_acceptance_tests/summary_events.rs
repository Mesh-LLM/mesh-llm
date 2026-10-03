use super::*;

// ════════════════════════════════════
//  HAS_TERMINAL_EVENT TESTS
// ════════════════════════════════════

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
    assert!(
        !store.has_terminal_event("term-s1").unwrap(),
        "no terminal yet"
    );

    let payload = r#"{"type":"completed","status_code":200}"#;
    store
        .insert_lifecycle_event("term-s1", "evt-term", payload, &clock.now())
        .unwrap();
    assert!(
        store.has_terminal_event("term-s1").unwrap(),
        "terminal exists now"
    );

    // Non-terminal events should not trigger has_terminal.
    let non_term_payload = r#"{"type":"admitted","model":"llama3"}"#;
    store
        .insert_lifecycle_event("term-s1", "evt-admit", non_term_payload, &clock.now())
        .unwrap();
    assert!(
        store.has_terminal_event("term-s1").unwrap(),
        "still has terminal despite new non-terminal event"
    );

    // New summary without any events.
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

// ════════════════════════════════════
//  LIST EVENTS FOR SUMMARY TESTS
// ════════════════════════════════════

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

    // Insert events in reverse chronological order.
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, 'req-1', ?, ?)",
        rusqlite::params!["evt-c", "2025-03-01T00:00:00Z", r#"{"type":"completed"}"#],
    ).unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, 'req-1', ?, ?)",
        rusqlite::params!["evt-a", "2025-01-01T00:00:00Z", r#"{"type":"admitted"}"#],
    ).unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, 'req-1', ?, ?)",
        rusqlite::params!["evt-b", "2025-02-01T00:00:00Z", r#"{"type":"stream_started"}"#],
    ).unwrap();

    let events = store.list_events_for_summary("req-1").unwrap();
    assert_eq!(events.len(), 3);
    // Should be ordered ASC by occurred_at.
    assert_eq!(events[0].event_id, "evt-a");
    assert_eq!(events[1].event_id, "evt-b");
    assert_eq!(events[2].event_id, "evt-c");
}
