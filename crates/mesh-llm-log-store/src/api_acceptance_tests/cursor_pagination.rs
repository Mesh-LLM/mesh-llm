use super::*;

// ════════════════════════════════════
//  CURSOR PAGINATION TESTS
// ════════════════════════════════════

#[test]
fn cursor_pages_no_overlap_or_omission() {
    let (store, _clock, _tmp) = open_store();

    // Mix repeated timestamps to exercise the request-id tiebreaker across pages.
    for i in 0..7u32 {
        let ts = if i % 2 == 0 {
            "2025-01-01T00:00:10Z"
        } else {
            "2025-01-01T00:00:20Z"
        };
        store
            .insert_summary(
                &format!("page-{:04}", i),
                None,
                None,
                None,
                None,
                ts,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let page_size = 3;
    let mut all_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    loop {
        let page = store.list_summaries(page_size, cursor.as_deref()).unwrap();
        assert!(page.items.len() <= page_size);
        all_ids.extend(page.items.iter().map(|r| r.request_id.clone()));
        if let Some(c) = page.next_cursor {
            cursor = Some(c);
        } else {
            break;
        }
    }

    // All 7 IDs present, no duplicates.
    assert_eq!(all_ids.len(), 7, "expected all 7 summaries");
    let mut sorted = all_ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 7, "no duplicate IDs across pages");

    // Verify expected IDs.
    for i in 0..7u32 {
        assert!(all_ids.iter().any(|id| id == &format!("page-{:04}", i)));
    }
}

#[test]
fn opaque_keyset_cursor_respects_boundaries_and_limits_for_summaries_and_events() {
    let (store, _clock, _tmp) = open_store();
    for (request_id, created_at) in [
        ("page-a", "2025-01-01T00:00:01Z"),
        ("page-b", "2025-01-01T00:00:02Z"),
        ("page-c", "2025-01-01T00:00:03Z"),
    ] {
        store
            .insert_summary(
                request_id, None, None, None, None, created_at, None, None, None,
            )
            .expect("insert summary");
    }
    for (event_id, occurred_at) in [
        ("event-a", "2025-01-01T00:00:01Z"),
        ("event-b", "2025-01-01T00:00:02Z"),
        ("event-c", "2025-01-01T00:00:03Z"),
    ] {
        store
            .insert_lifecycle_event("page-a", event_id, r#"{"type":"admitted"}"#, occurred_at)
            .expect("insert lifecycle event");
    }

    let first_summaries = store.list_summaries(1, None).expect("first summaries page");
    assert_eq!(first_summaries.items[0].request_id, "page-c");
    let second_summaries = store
        .list_summaries(1, first_summaries.next_cursor.as_deref())
        .expect("second summaries page");
    assert_eq!(second_summaries.items[0].request_id, "page-b");

    let first_events = store
        .list_lifecycle_events(1, None)
        .expect("first lifecycle page");
    assert_eq!(first_events.items[0].event_id, "event-c");
    let second_events = store
        .list_lifecycle_events(1, first_events.next_cursor.as_deref())
        .expect("second lifecycle page");
    assert_eq!(second_events.items[0].event_id, "event-b");

    assert!(
        store
            .list_summaries(0, None)
            .expect("zero summary limit")
            .items
            .is_empty()
    );
    assert!(
        store
            .list_lifecycle_events(0, None)
            .expect("zero lifecycle limit")
            .items
            .is_empty()
    );
    assert!(matches!(
        store.list_summaries(1, Some("not-an-opaque-cursor")),
        Err(LogStoreError::CursorMalformed(_))
    ));
}

#[test]
fn cursor_pages_no_gap_after_reopen() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Insert data and get first page.
    let store1 = LogStore::open(tmp.path(), clock.clone()).unwrap();
    for i in 0..5u32 {
        let ts = if i % 2 == 0 {
            "2025-01-01T00:00:10Z"
        } else {
            "2025-01-01T00:00:20Z"
        };
        store1
            .insert_summary(
                &format!("reopen-{:04}", i),
                None,
                None,
                None,
                None,
                ts,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let first_page = store1.list_summaries(2, None).unwrap();
    assert_eq!(first_page.items.len(), 2);
    let cursor_str = first_page.next_cursor.expect("has next cursor");
    drop(store1);

    // Reopen and fetch page 2 using the same cursor.
    let store2 = LogStore::reopen_at(tmp.path(), clock.clone()).unwrap();
    let second_page = store2.list_summaries(2, Some(&cursor_str)).unwrap();

    // No overlap with first page.
    assert!(
        second_page.items.iter().all(|r| {
            !first_page
                .items
                .iter()
                .any(|f| f.request_id == r.request_id)
        }),
        "page 2 should not contain items from page 1"
    );

    let total: Vec<String> = first_page
        .items
        .into_iter()
        .chain(second_page.items)
        .map(|r| r.request_id)
        .collect();
    assert_eq!(total.len(), 4, "should see 4 items across two pages");
}

#[test]
fn cursor_same_timestamp_no_overlap_or_omission() {
    let (store, _, _tmp) = open_store();

    // Insert all rows with the same created_at — tiebreak by request_id.
    for i in 0..5u32 {
        store
            .insert_summary(
                &format!("same-ts-{i:04}"),
                None,
                None,
                None,
                None,
                "2025-06-15T12:00:00Z",
                None,
                None,
                None,
            )
            .expect("insert whole-second summary through the canonical boundary");
    }

    let page_size = 3;
    let mut all_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    loop {
        let page = store.list_summaries(page_size, cursor.as_deref()).unwrap();
        assert!(page.items.len() <= page_size);
        all_ids.extend(page.items.iter().map(|r| r.request_id.clone()));
        if let Some(c) = page.next_cursor {
            cursor = Some(c);
        } else {
            break;
        }
    }

    assert_eq!(
        all_ids.len(),
        5,
        "expected all 5 summaries with same timestamp"
    );

    // Verify ordering: DESC on (created_at, request_id), so highest ID first.
    let expected = (0..5u32)
        .rev()
        .map(|i| format!("same-ts-{i:04}"))
        .collect::<Vec<_>>();
    assert_eq!(
        all_ids, expected,
        "pages must arrive in DESC (created_at, request_id) order"
    );
}

#[test]
fn cursor_encode_decode_roundtrip() {
    let ts = "2025-06-15T12:34:56Z";
    let id = "abc-def-123";
    let encoded = encode_cursor(ts, id);

    assert!(!encoded.is_empty());

    let (dec_ts, dec_id) = decode_cursor(&encoded).expect("decode valid cursor");
    assert_eq!(dec_ts, ts);
    assert_eq!(dec_id, id);
}

#[test]
fn cursor_decode_malformed_returns_error() {
    // Empty string.
    let err = decode_cursor("").unwrap_err();
    match &err {
        LogStoreError::CursorMalformed(msg) => assert!(!msg.is_empty()),
        other => panic!("expected CursorMalformed, got: {:?}", other),
    }

    // Invalid base64 characters.
    let err = decode_cursor("v1:!!!invalid!!!").unwrap_err();
    match &err {
        LogStoreError::CursorMalformed(_) => {} // expected
        other => panic!("expected CursorMalformed, got: {:?}", other),
    }
}

#[test]
fn cursor_decode_unknown_version_returns_error() {
    let err = decode_cursor("v99:dGVzdA==").unwrap_err();
    match &err {
        LogStoreError::CursorMalformed(msg) => assert!(msg.contains("unknown cursor version")),
        other => panic!("expected CursorMalformed, got: {:?}", other),
    }
}
