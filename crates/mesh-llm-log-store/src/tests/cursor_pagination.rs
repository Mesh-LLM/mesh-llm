use std::sync::Arc;

use crate::cursor::{decode_cursor, encode_cursor};
use crate::error::LogStoreError;
use crate::{Clock as ClockTrait, LogStore};

use super::store_setup::{TestClock, open_store};

#[test]
fn cursor_pages_no_overlap_or_omission() {
    let (store, _clock, _tmp) = open_store();
    for i in 0..7_u32 {
        let timestamp = if i % 2 == 0 {
            "2025-01-01T00:00:10Z"
        } else {
            "2025-01-01T00:00:20Z"
        };
        store
            .insert_summary(
                &format!("page-{i:04}"),
                None,
                None,
                None,
                None,
                timestamp,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let page_size = 3;
    let mut all_ids = Vec::new();
    let mut cursor = None;
    loop {
        let page = store.list_summaries(page_size, cursor.as_deref()).unwrap();
        assert!(page.items.len() <= page_size);
        all_ids.extend(page.items.iter().map(|record| record.request_id.clone()));
        if let Some(next) = page.next_cursor {
            cursor = Some(next);
        } else {
            break;
        }
    }

    assert_eq!(all_ids.len(), 7, "expected all 7 summaries");
    let mut unique_ids = all_ids.clone();
    unique_ids.sort();
    unique_ids.dedup();
    assert_eq!(unique_ids.len(), 7, "no duplicate IDs across pages");
    for i in 0..7_u32 {
        assert!(all_ids.iter().any(|id| id == &format!("page-{i:04}")));
    }
}

#[test]
fn cursor_pages_no_gap_after_reopen() {
    let temporary_directory = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store1 = LogStore::open(temporary_directory.path(), clock.clone()).unwrap();
    for i in 0..5_u32 {
        let timestamp = if i % 2 == 0 {
            "2025-01-01T00:00:10Z"
        } else {
            "2025-01-01T00:00:20Z"
        };
        store1
            .insert_summary(
                &format!("reopen-{i:04}"),
                None,
                None,
                None,
                None,
                timestamp,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let first_page = store1.list_summaries(2, None).unwrap();
    assert_eq!(first_page.items.len(), 2);
    let cursor = first_page.next_cursor.clone().expect("has next cursor");
    drop(store1);

    let store2 = LogStore::reopen_at(temporary_directory.path(), clock).unwrap();
    let second_page = store2.list_summaries(2, Some(&cursor)).unwrap();
    assert!(second_page.items.iter().all(|record| {
        !first_page
            .items
            .iter()
            .any(|first| first.request_id == record.request_id)
    }));
    assert_eq!(first_page.items.len() + second_page.items.len(), 4);
}

#[test]
fn cursor_same_timestamp_no_overlap_or_omission() {
    let (store, _, _tmp) = open_store();
    for i in 0..5_u32 {
        store
            .conn()
            .execute(
                "INSERT INTO summaries (request_id, state, created_at) VALUES (?, 'active', ?)",
                rusqlite::params![format!("same-ts-{i:04}"), "2025-06-15T12:00:00Z"],
            )
            .unwrap();
    }

    let mut all_ids = Vec::new();
    let mut cursor = None;
    loop {
        let page = store.list_summaries(3, cursor.as_deref()).unwrap();
        all_ids.extend(page.items.iter().map(|record| record.request_id.clone()));
        if let Some(next) = page.next_cursor {
            cursor = Some(next);
        } else {
            break;
        }
    }

    assert_eq!(all_ids.len(), 5);
    assert_eq!(all_ids.first(), Some(&"same-ts-0004".to_string()));
}

#[test]
fn cursor_encode_decode_roundtrip() {
    let encoded = encode_cursor("2025-06-15T12:34:56Z", "abc-def-123");
    assert!(!encoded.is_empty());
    let (timestamp, identifier) = decode_cursor(&encoded).expect("decode valid cursor");
    assert_eq!(timestamp, "2025-06-15T12:34:56Z");
    assert_eq!(identifier, "abc-def-123");
}

#[test]
fn cursor_decode_malformed_returns_error() {
    for cursor in ["", "v1:!!!invalid!!!"] {
        assert!(matches!(
            decode_cursor(cursor),
            Err(LogStoreError::CursorMalformed(_))
        ));
    }
}

#[test]
fn cursor_decode_unknown_version_returns_error() {
    let error = decode_cursor("v99:dGVzdA==").unwrap_err();
    assert!(
        matches!(error, LogStoreError::CursorMalformed(message) if message.contains("unknown cursor version"))
    );
}

#[test]
fn empty_table_pagination() {
    let (store, _, _tmp) = open_store();
    let page = store.list_summaries(10, None).unwrap();
    assert!(page.items.is_empty());
    assert!(page.next_cursor.is_none());
    assert!(
        store
            .list_lifecycle_events(10, None)
            .unwrap()
            .items
            .is_empty()
    );
    assert!(
        store
            .list_artifact_pointers(10, None)
            .unwrap()
            .items
            .is_empty()
    );
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
