use std::sync::Arc;

use crate::{
    Clock, LogStore, LogStoreError, MAX_QUERY_LIMIT, PageQuery, ProxyQuery, QuerySort,
    RequestOutcome, RequestQuery, WebhookDeliveryErrorCode, encode_cursor,
};

#[derive(Debug)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> String {
        "2026-08-03T00:00:00Z".to_string()
    }
}

fn open_store() -> (tempfile::TempDir, LogStore) {
    let root = tempfile::tempdir().expect("create query store root");
    let store = LogStore::open(root.path(), Arc::new(FixedClock)).expect("open query store");
    (root, store)
}

fn request_query() -> RequestQuery {
    RequestQuery {
        limit: 10,
        cursor: None,
        from: None,
        to: None,
        route: None,
        model: None,
        provider: None,
        engine: None,
        status_code: None,
        outcome: None,
        sort: QuerySort::Descending,
    }
}

#[test]
fn request_query_applies_all_filters_and_normalizes_time_bounds() {
    let (_root, store) = open_store();
    store
        .insert_summary(
            "matching-request",
            Some("model-a"),
            Some("chat"),
            Some("provider-a"),
            Some("engine-a"),
            "2026-08-03T00:00:05Z",
            None,
            None,
            None,
        )
        .expect("insert matching summary");
    store
        .write_terminal_event(
            "matching-request",
            "matching-event",
            r#"{"type":"completed","status_code":201}"#,
            "completed",
            Some(201),
            "2026-08-03T00:00:06Z",
        )
        .expect("complete matching summary");
    store
        .insert_summary(
            "non-matching-request",
            Some("model-b"),
            Some("completion"),
            Some("provider-b"),
            Some("engine-b"),
            "2026-08-03T00:00:05Z",
            None,
            None,
            None,
        )
        .expect("insert non-matching summary");

    let page = store
        .query_requests(&RequestQuery {
            limit: 10,
            cursor: None,
            from: Some("2026-08-02T20:00:00-04:00".to_string()),
            to: Some("2026-08-03T00:01:00Z".to_string()),
            route: Some("chat".to_string()),
            model: Some("model-a".to_string()),
            provider: Some("provider-a".to_string()),
            engine: Some("engine-a".to_string()),
            status_code: Some(201),
            outcome: Some(RequestOutcome::Completed),
            sort: QuerySort::Descending,
        })
        .expect("query requests");

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].request_id, "matching-request");
}

#[test]
fn request_time_bounds_compare_instants_within_the_boundary_second() {
    let (_root, store) = open_store();
    for (request_id, created_at) in [
        ("request-at-boundary", "2026-08-03T00:00:00Z"),
        ("request-after-boundary", "2026-08-03T00:00:00.123Z"),
    ] {
        store
            .insert_summary(
                request_id, None, None, None, None, created_at, None, None, None,
            )
            .expect("insert boundary summary");
    }

    let mut from_query = request_query();
    from_query.from = Some("2026-08-03T00:00:00Z".to_string());
    let from_page = store
        .query_requests(&from_query)
        .expect("query inclusive lower bound");
    assert_eq!(
        from_page
            .items
            .iter()
            .map(|record| record.request_id.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        ["request-after-boundary", "request-at-boundary"]
            .into_iter()
            .collect()
    );

    let mut to_query = request_query();
    to_query.to = Some("2026-08-03T00:00:00Z".to_string());
    let to_page = store
        .query_requests(&to_query)
        .expect("query inclusive upper bound");
    assert_eq!(
        to_page
            .items
            .iter()
            .map(|record| record.request_id.as_str())
            .collect::<Vec<_>>(),
        ["request-at-boundary"]
    );

    let mut cursor_query = request_query();
    cursor_query.sort = QuerySort::Ascending;
    cursor_query.cursor = Some(encode_cursor("2026-08-03T00:00:00Z", "request-at-boundary"));
    let cursor_page = store
        .query_requests(&cursor_query)
        .expect("whole-second cursor resumes canonical fractional ordering");
    assert_eq!(
        cursor_page
            .items
            .iter()
            .map(|record| record.request_id.as_str())
            .collect::<Vec<_>>(),
        ["request-after-boundary"]
    );
    assert!(
        cursor_page
            .items
            .iter()
            .all(|record| record.created_at.len() == "2026-08-03T00:00:00.000000000Z".len())
    );
}

#[test]
fn standalone_retention_compares_a_fractional_boundary_chronologically() {
    let (_root, store) = open_store();
    store
        .insert_audit_entry(
            "audit-before-fractional-cutoff",
            None,
            "2026-08-03T00:00:00Z",
            "runtime",
            "before_cutoff",
            None,
        )
        .expect("insert audit before fractional cutoff");
    store
        .insert_audit_entry(
            "audit-after-fractional-cutoff",
            None,
            "2026-08-03T00:00:00.750Z",
            "runtime",
            "after_cutoff",
            None,
        )
        .expect("insert audit after fractional cutoff");

    store
        .cascade_cleanup_before("2026-08-03T00:00:00.500Z")
        .expect("apply fractional retention cutoff");

    let retained = store
        .conn()
        .prepare("SELECT entry_id FROM audit_entries ORDER BY entry_id")
        .expect("prepare retained audit query")
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query retained audit rows")
        .collect::<Result<Vec<_>, _>>()
        .expect("read retained audit rows");
    assert_eq!(retained, ["audit-after-fractional-cutoff"]);
}

#[test]
fn proxy_optional_timestamps_are_canonicalized_with_the_ordering_key() {
    let (_root, store) = open_store();
    store
        .insert_summary(
            "request-proxy-timestamps",
            None,
            None,
            None,
            None,
            "2026-08-02T23:59:59Z",
            None,
            None,
            None,
        )
        .expect("insert proxy request");
    store
        .insert_proxy_record(
            "attempt-canonical",
            "request-proxy-timestamps",
            "2026-08-03T01:00:00.100+01:00",
            "http://example.invalid",
            None,
            None,
            Some("2026-08-03T00:00:00Z"),
            Some("2026-08-03T01:00:00.200+01:00"),
            Some(200),
            None,
        )
        .expect("insert proxy timestamps");

    let page = store
        .query_proxy_records(&ProxyQuery {
            page: PageQuery {
                limit: 1,
                cursor: None,
                sort: QuerySort::Descending,
            },
            request_id: Some("request-proxy-timestamps".to_string()),
            provider: None,
            engine: None,
            status_code: None,
        })
        .expect("query canonical proxy timestamps");
    let record = &page.items[0];
    assert_eq!(record.occurred_at, "2026-08-03T00:00:00.100000000Z");
    assert_eq!(
        record.started_at.as_deref(),
        Some("2026-08-03T00:00:00.000000000Z")
    );
    assert_eq!(
        record.completed_at.as_deref(),
        Some("2026-08-03T00:00:00.200000000Z")
    );
}

#[test]
fn webhook_scheduling_uses_canonical_fractional_eligibility_keys() {
    let (_root, store) = open_store();
    store
        .insert_summary(
            "request-webhook-timestamps",
            None,
            None,
            None,
            None,
            "2026-08-02T23:59:59Z",
            None,
            None,
            None,
        )
        .expect("insert webhook request");
    store
        .write_terminal_event(
            "request-webhook-timestamps",
            "event-webhook-terminal",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2026-08-03T00:00:00Z",
        )
        .expect("terminalize webhook request");
    store
        .enqueue_webhook_delivery(
            "delivery-canonical",
            "request-webhook-timestamps",
            "2026-08-03T01:00:00.100+01:00",
            2,
        )
        .expect("enqueue canonical delivery");
    let claimed = store
        .claim_next_webhook_delivery("2026-08-03T01:00:00.200+01:00", "2026-08-03T01:00:01+01:00")
        .expect("claim canonical delivery")
        .expect("delivery is eligible");
    assert_eq!(claimed.updated_at, "2026-08-03T00:00:00.200000000Z");
    assert_eq!(
        claimed.lease_expires_at.as_deref(),
        Some("2026-08-03T00:00:01.000000000Z")
    );
    store
        .retry_or_dead_letter_webhook_delivery(
            "delivery-canonical",
            claimed.claim_generation,
            "2026-08-03T01:00:00.300+01:00",
            "2026-08-03T00:00:00.750Z",
            WebhookDeliveryErrorCode::Timeout,
            None,
        )
        .expect("schedule canonical retry")
        .expect("retry transition");
    assert!(
        store
            .claim_next_webhook_delivery("2026-08-03T00:00:00.700Z", "2026-08-03T00:00:02Z",)
            .expect("check before retry boundary")
            .is_none()
    );
    assert!(
        store
            .claim_next_webhook_delivery("2026-08-03T01:00:00.750+01:00", "2026-08-03T00:00:02Z",)
            .expect("claim at retry boundary")
            .is_some()
    );
}

#[test]
fn durable_query_rejects_unbounded_or_invalid_inputs() {
    let (_root, store) = open_store();
    let mut query = request_query();
    query.limit = 0;
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::InvalidQuery(_))
    ));

    query.limit = MAX_QUERY_LIMIT + 1;
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::InvalidQuery(_))
    ));

    query.limit = 1;
    query.from = Some("not-a-time".to_string());
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::InvalidQuery(_))
    ));

    query.from = Some("2026-08-03T00:01:00Z".to_string());
    query.to = Some("2026-08-03T00:00:00Z".to_string());
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::InvalidQuery(_))
    ));
}

#[test]
fn forged_or_scope_mismatched_request_cursor_is_rejected() {
    let (_root, store) = open_store();
    store
        .insert_summary(
            "request-a",
            None,
            Some("chat"),
            None,
            None,
            "2026-08-03T00:00:05Z",
            None,
            None,
            None,
        )
        .expect("insert request");
    let mut query = request_query();
    query.limit = 1;
    query.cursor = Some(encode_cursor("2026-08-03T00:00:04Z", "request-a"));
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::CursorInvalid)
    ));

    query.cursor = Some(encode_cursor("2026-08-03T00:00:05Z", "request-a"));
    query.route = Some("completion".to_string());
    assert!(matches!(
        store.query_requests(&query),
        Err(LogStoreError::CursorInvalid)
    ));
}

#[test]
fn related_records_are_typed_scoped_and_path_free() {
    let (_root, store) = open_store();
    for request_id in ["request-a", "request-b"] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                "2026-08-03T00:00:00Z",
                None,
                None,
                None,
            )
            .expect("insert summary");
    }
    store
        .insert_lifecycle_event(
            "request-a",
            "event-a",
            r#"{"type":"stream_chunk","tokens":3}"#,
            "2026-08-03T00:00:01Z",
        )
        .expect("insert lifecycle event");
    store
        .insert_artifact_pointer(
            "artifact-a",
            "request-a",
            "2026-08-03T00:00:02Z",
            "response",
            None,
        )
        .expect("insert artifact pointer");
    store
        .update_artifact_pointer_storage("artifact-a", Some("text/plain"), "abc", 3, 1, true, false)
        .expect("store artifact metadata");
    store
        .update_artifact_pointer_missing("artifact-a")
        .expect("mark artifact missing");
    store
        .insert_proxy_record(
            "attempt-a",
            "request-a",
            "2026-08-03T00:00:03Z",
            "local-target",
            Some("provider-a"),
            Some("engine-a"),
            None,
            None,
            Some(200),
            None,
        )
        .expect("insert request-a proxy");
    store
        .insert_proxy_record(
            "attempt-b",
            "request-b",
            "2026-08-03T00:00:03Z",
            "other-target",
            None,
            None,
            None,
            None,
            Some(503),
            None,
        )
        .expect("insert request-b proxy");

    let page = PageQuery {
        limit: 10,
        cursor: None,
        sort: QuerySort::Ascending,
    };
    let events = store
        .query_events("request-a", &page)
        .expect("query events");
    let artifacts = store
        .query_artifacts("request-a", &page)
        .expect("query artifact metadata");
    let event_windows = store
        .query_events_for_requests(&["request-a".to_string(), "request-b".to_string()], 1)
        .expect("batch query event windows");
    let artifact_windows = store
        .query_artifacts_for_requests(&["request-a".to_string(), "request-b".to_string()], 1)
        .expect("batch query artifact windows");
    let proxies = store
        .query_proxy_records(&ProxyQuery {
            page,
            request_id: Some("request-a".to_string()),
            provider: Some("provider-a".to_string()),
            engine: Some("engine-a".to_string()),
            status_code: Some(200),
        })
        .expect("query proxy records");

    assert_eq!(
        events.items[0].payload_json,
        r#"{"type":"stream_chunk","tokens":3}"#
    );
    assert!(artifacts.items[0].redacted);
    assert!(artifacts.items[0].missing);
    assert_eq!(artifacts.items[0].checksum.as_deref(), Some("abc"));
    assert_eq!(
        event_windows
            .get("request-a")
            .expect("request-a event window")[0]
            .event_id,
        "event-a"
    );
    assert!(
        !event_windows.contains_key("request-b"),
        "owners without children do not materialize empty rows"
    );
    assert_eq!(
        artifact_windows
            .get("request-a")
            .expect("request-a artifact window")[0]
            .artifact_id,
        "artifact-a"
    );
    assert_eq!(proxies.items.len(), 1);
    assert_eq!(proxies.items[0].attempt_id, "attempt-a");
}
