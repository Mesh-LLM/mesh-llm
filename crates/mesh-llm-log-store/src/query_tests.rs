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
        exclude_route: None,
        exclude_route_prefix: None,
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
            exclude_route: None,
            exclude_route_prefix: None,
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
fn request_route_exclusions_apply_before_pagination_and_preserve_null_routes() {
    // Given
    let (_root, store) = open_store();
    for (request_id, route, created_at) in [
        ("visible-null-route", None, "2026-08-03T00:00:01Z"),
        (
            "visible-chat-route",
            Some("chat_completions"),
            "2026-08-03T00:00:02Z",
        ),
        (
            "hidden-management-route",
            Some("management_get_status"),
            "2026-08-03T00:00:03Z",
        ),
        (
            "hidden-models-route",
            Some("models"),
            "2026-08-03T00:00:04Z",
        ),
    ] {
        store
            .insert_summary(
                request_id, None, route, None, None, created_at, None, None, None,
            )
            .expect("insert request summary");
    }

    // When
    let page = store
        .query_requests(&RequestQuery {
            limit: 2,
            exclude_route: Some("models".to_string()),
            exclude_route_prefix: Some("management_".to_string()),
            ..request_query()
        })
        .expect("query visible request summaries");

    // Then
    assert_eq!(
        page.items
            .iter()
            .map(|record| record.request_id.as_str())
            .collect::<Vec<_>>(),
        ["visible-chat-route", "visible-null-route"]
    );
    assert!(page.next_cursor.is_none());
}

#[test]
fn caller_identity_round_trips_without_replacing_principal_identity() {
    let (_root, store) = open_store();
    let endpoint_id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    store
        .conn()
        .execute(
            "INSERT INTO summaries \
             (request_id, created_at, tenant_id, account_id, user_id, caller_endpoint_id, caller_addr, caller_path_type) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                "request-with-caller",
                "2026-08-20T12:00:00Z",
                "tenant-a",
                "account-a",
                "user-a",
                endpoint_id,
                "192.0.2.42:11204",
                "remote_quic_http",
            ],
        )
        .expect("insert caller summary");

    let repository_row = store
        .get_summary("request-with-caller")
        .expect("query repository summary")
        .expect("repository summary");
    let query_row = store
        .query_request_with_caller("request-with-caller")
        .expect("query request record")
        .expect("request record");

    assert_eq!(repository_row.tenant_id.as_deref(), Some("tenant-a"));
    assert_eq!(repository_row.account_id.as_deref(), Some("account-a"));
    assert_eq!(repository_row.user_id.as_deref(), Some("user-a"));
    assert_eq!(query_row.caller_endpoint_id.as_deref(), Some(endpoint_id));
    assert_eq!(query_row.caller_addr.as_deref(), Some("192.0.2.42:11204"));
    assert_eq!(
        query_row.caller_path_type.as_deref(),
        Some("remote_quic_http")
    );
}

#[test]
fn authenticated_relay_upsert_replaces_and_protects_the_complete_caller_tuple() {
    let (_root, store) = open_store();
    let request_id = "request-with-authenticated-relay";
    let endpoint_id = "7272727272727272727272727272727272727272727272727272727272727272";

    store
        .upsert_summary_metadata_with_caller(
            request_id,
            Some("model-a"),
            Some("responses"),
            Some("provider-a"),
            Some("engine-a"),
            None,
            Some("127.0.0.1:40123"),
            Some("local_http"),
            "2026-08-20T12:00:00Z",
        )
        .expect("persist provisional local caller");
    store
        .upsert_summary_metadata_with_caller(
            request_id,
            None,
            None,
            None,
            None,
            Some(endpoint_id),
            None,
            Some("relay"),
            "2026-08-20T12:00:01Z",
        )
        .expect("replace with authenticated relay caller");
    store
        .upsert_summary_metadata_with_caller(
            request_id,
            None,
            None,
            None,
            None,
            None,
            Some("127.0.0.1:49999"),
            Some("local_http"),
            "2026-08-20T12:00:02Z",
        )
        .expect("ignore later provisional local caller");

    let repository_row = store
        .get_summary(request_id)
        .expect("query summary")
        .expect("summary");
    let query_row = store
        .query_request_with_caller(request_id)
        .expect("query request record")
        .expect("request record");
    assert_eq!(repository_row.model.as_deref(), Some("model-a"));
    assert_eq!(repository_row.route.as_deref(), Some("responses"));
    assert_eq!(repository_row.provider.as_deref(), Some("provider-a"));
    assert_eq!(repository_row.engine.as_deref(), Some("engine-a"));
    assert_eq!(query_row.caller_endpoint_id.as_deref(), Some(endpoint_id));
    assert_eq!(query_row.caller_addr, None);
    assert_eq!(query_row.caller_path_type.as_deref(), Some("relay"));
}

#[test]
fn authenticated_endpoint_only_caller_round_trips_through_public_query_api() {
    let (_root, store) = open_store();
    let request_id = "request-with-endpoint-only-caller";
    let endpoint_id = "7373737373737373737373737373737373737373737373737373737373737373";
    store
        .upsert_summary_metadata_with_caller(
            request_id,
            Some("model-a"),
            Some("responses"),
            Some("provider-a"),
            Some("engine-a"),
            Some(endpoint_id),
            None,
            None,
            "2026-08-20T12:00:00Z",
        )
        .expect("persist endpoint-only caller");

    let direct = store
        .query_request_with_caller(request_id)
        .expect("query request")
        .expect("request record");
    let page = store
        .query_requests_with_caller(&request_query())
        .expect("query request page");
    let listed = page
        .items
        .iter()
        .find(|record| record.request.request_id == request_id)
        .expect("listed request");

    for record in [&direct, listed] {
        assert_eq!(record.caller_endpoint_id.as_deref(), Some(endpoint_id));
        assert_eq!(record.caller_addr, None);
        assert_eq!(record.caller_path_type, None);
    }
}

#[test]
fn endpoint_only_upsert_replaces_local_or_empty_and_protects_first_authenticated_caller() {
    let (_root, store) = open_store();
    let endpoint_id = "7474747474747474747474747474747474747474747474747474747474747474";
    let later_endpoint_id = "7575757575757575757575757575757575757575757575757575757575757575";

    for (request_id, caller_addr, caller_path_type) in [
        ("endpoint-only-replaces-empty", None, None),
        (
            "endpoint-only-replaces-local",
            Some("127.0.0.1:40123"),
            Some("local_http"),
        ),
    ] {
        store
            .upsert_summary_metadata_with_caller(
                request_id,
                Some("model-a"),
                Some("responses"),
                Some("provider-a"),
                Some("engine-a"),
                None,
                caller_addr,
                caller_path_type,
                "2026-08-20T12:00:00Z",
            )
            .expect("persist initial caller");
        store
            .upsert_summary_metadata_with_caller(
                request_id,
                None,
                None,
                None,
                None,
                Some(endpoint_id),
                None,
                None,
                "2026-08-20T12:00:01Z",
            )
            .expect("replace with endpoint-only caller");

        for (later_endpoint, later_addr, later_path) in [
            (None, Some("127.0.0.1:49999"), Some("local_http")),
            (
                Some(later_endpoint_id),
                Some("192.0.2.75:11204"),
                Some("remote_quic_http"),
            ),
            (Some(later_endpoint_id), None, Some("relay")),
            (Some(later_endpoint_id), None, None),
        ] {
            store
                .upsert_summary_metadata_with_caller(
                    request_id,
                    None,
                    None,
                    None,
                    None,
                    later_endpoint,
                    later_addr,
                    later_path,
                    "2026-08-20T12:00:02Z",
                )
                .expect("ignore later caller");
        }

        let record = store
            .query_request_with_caller(request_id)
            .expect("query request")
            .expect("request record");
        assert_eq!(record.caller_endpoint_id.as_deref(), Some(endpoint_id));
        assert_eq!(record.caller_addr, None);
        assert_eq!(record.caller_path_type, None);
        assert_eq!(record.request.model.as_deref(), Some("model-a"));
        assert_eq!(record.request.route.as_deref(), Some("responses"));
        assert_eq!(record.request.provider.as_deref(), Some("provider-a"));
        assert_eq!(record.request.engine.as_deref(), Some("engine-a"));
    }
}

#[test]
fn unrecognized_stage_caller_does_not_replace_local_request_metadata() {
    let (_root, store) = open_store();
    let request_id = "request-with-stage-second";
    let endpoint_id = "7575757575757575757575757575757575757575757575757575757575757575";

    store
        .upsert_summary_metadata_with_caller(
            request_id,
            Some("model-a"),
            Some("responses"),
            Some("provider-a"),
            Some("engine-a"),
            None,
            Some("127.0.0.1:40123"),
            Some("local_http"),
            "2026-08-20T12:00:00Z",
        )
        .expect("persist provisional local caller");
    store
        .upsert_summary_metadata_with_caller(
            request_id,
            None,
            None,
            None,
            None,
            Some(endpoint_id),
            Some("192.0.2.75:11204"),
            Some("remote_quic_stage"),
            "2026-08-20T12:00:01Z",
        )
        .expect("ignore unrecognized stage caller");

    let repository_row = store
        .get_summary(request_id)
        .expect("query summary")
        .expect("summary");
    let query_row = store
        .query_request_with_caller(request_id)
        .expect("query request record")
        .expect("request record");
    assert_eq!(repository_row.model.as_deref(), Some("model-a"));
    assert_eq!(repository_row.route.as_deref(), Some("responses"));
    assert_eq!(repository_row.provider.as_deref(), Some("provider-a"));
    assert_eq!(repository_row.engine.as_deref(), Some("engine-a"));
    assert_eq!(query_row.caller_endpoint_id, None);
    assert_eq!(query_row.caller_addr.as_deref(), Some("127.0.0.1:40123"));
    assert_eq!(query_row.caller_path_type.as_deref(), Some("local_http"));
}

#[test]
fn unrecognized_stage_caller_does_not_enter_empty_summary_metadata() {
    let (_root, store) = open_store();
    let endpoint_id = "7676767676767676767676767676767676767676767676767676767676767676";

    for (request_id, path_type) in [
        ("request-with-stage-first", "remote_quic_stage"),
        ("request-with-empty-path", ""),
        ("request-with-partial-path", "remote_quic_http/"),
    ] {
        store
            .upsert_summary_metadata_with_caller(
                request_id,
                Some("model-a"),
                Some("responses"),
                Some("provider-a"),
                Some("engine-a"),
                Some(endpoint_id),
                Some("192.0.2.76:11204"),
                Some(path_type),
                "2026-08-20T12:00:00Z",
            )
            .expect("ignore unrecognized caller on empty summary");

        let repository_row = store
            .get_summary(request_id)
            .expect("query repository summary")
            .expect("summary");
        let query_row = store
            .query_request_with_caller(request_id)
            .expect("query request record")
            .expect("request record");
        assert_eq!(repository_row.model.as_deref(), Some("model-a"));
        assert_eq!(repository_row.route.as_deref(), Some("responses"));
        assert_eq!(repository_row.provider.as_deref(), Some("provider-a"));
        assert_eq!(repository_row.engine.as_deref(), Some("engine-a"));
        assert_eq!(query_row.caller_endpoint_id, None);
        assert_eq!(query_row.caller_addr, None);
        assert_eq!(query_row.caller_path_type, None);
    }
}

#[test]
fn partial_caller_tuples_do_not_enter_empty_summary_metadata() {
    let (_root, store) = open_store();
    let endpoint_id = "7777777777777777777777777777777777777777777777777777777777777777";
    let partial_tuples = [
        (
            "request-local-without-address",
            None,
            None,
            Some("local_http"),
        ),
        (
            "request-remote-without-endpoint",
            None,
            Some("192.0.2.77:11204"),
            Some("remote_quic_http"),
        ),
        ("request-relay-without-endpoint", None, None, Some("relay")),
        (
            "request-values-without-path",
            Some(endpoint_id),
            Some("192.0.2.77:11204"),
            None,
        ),
        (
            "request-invalid-endpoint-only",
            Some("not-an-authenticated-endpoint"),
            None,
            None,
        ),
        (
            "request-remote-invalid-endpoint",
            Some("not-an-authenticated-endpoint"),
            Some("192.0.2.77:11204"),
            Some("remote_quic_http"),
        ),
        (
            "request-local-invalid-address",
            None,
            Some("not-a-socket-address"),
            Some("local_http"),
        ),
    ];

    for (request_id, caller_endpoint_id, caller_addr, caller_path_type) in partial_tuples {
        store
            .upsert_summary_metadata_with_caller(
                request_id,
                None,
                None,
                None,
                None,
                caller_endpoint_id,
                caller_addr,
                caller_path_type,
                "2026-08-20T12:00:00Z",
            )
            .expect("ignore partial caller tuple");

        let row = store
            .query_request_with_caller(request_id)
            .expect("query request record")
            .expect("request record");
        assert_eq!(row.caller_endpoint_id, None);
        assert_eq!(row.caller_addr, None);
        assert_eq!(row.caller_path_type, None);
    }
}

#[test]
fn supported_caller_tuples_are_canonicalized_before_persistence() {
    let (_root, store) = open_store();
    let endpoint_id = "7878787878787878787878787878787878787878787878787878787878787878";

    for (request_id, caller_endpoint_id, caller_addr, caller_path_type) in [
        (
            "request-local-canonical",
            Some("ignored-endpoint"),
            Some("[2001:0db8:0:0::1]:40123"),
            Some("local_http"),
        ),
        (
            "request-remote-canonical",
            Some(endpoint_id),
            Some("[2001:0db8:0:0::2]:11204"),
            Some("remote_quic_http"),
        ),
        (
            "request-relay-clears-address",
            Some(endpoint_id),
            Some("192.0.2.78:11204"),
            Some("relay"),
        ),
        (
            "request-remote-clears-invalid-address",
            Some(endpoint_id),
            Some("not-a-socket-address"),
            Some("remote_quic_http"),
        ),
    ] {
        store
            .upsert_summary_metadata_with_caller(
                request_id,
                None,
                None,
                None,
                None,
                caller_endpoint_id,
                caller_addr,
                caller_path_type,
                "2026-08-20T12:00:00Z",
            )
            .expect("persist supported caller tuple");
    }

    let local = store
        .query_request_with_caller("request-local-canonical")
        .expect("query local caller")
        .expect("local caller");
    assert_eq!(local.caller_endpoint_id, None);
    assert_eq!(local.caller_addr.as_deref(), Some("[2001:db8::1]:40123"));
    assert_eq!(local.caller_path_type.as_deref(), Some("local_http"));

    let remote = store
        .query_request_with_caller("request-remote-canonical")
        .expect("query remote caller")
        .expect("remote caller");
    assert_eq!(remote.caller_endpoint_id.as_deref(), Some(endpoint_id));
    assert_eq!(remote.caller_addr.as_deref(), Some("[2001:db8::2]:11204"));
    assert_eq!(remote.caller_path_type.as_deref(), Some("remote_quic_http"));

    let relay = store
        .query_request_with_caller("request-relay-clears-address")
        .expect("query relay caller")
        .expect("relay caller");
    assert_eq!(relay.caller_endpoint_id.as_deref(), Some(endpoint_id));
    assert_eq!(relay.caller_addr, None);
    assert_eq!(relay.caller_path_type.as_deref(), Some("relay"));

    let remote_without_addr = store
        .query_request_with_caller("request-remote-clears-invalid-address")
        .expect("query remote caller with invalid address")
        .expect("remote caller with invalid address");
    assert_eq!(
        remote_without_addr.caller_endpoint_id.as_deref(),
        Some(endpoint_id)
    );
    assert_eq!(remote_without_addr.caller_addr, None);
    assert_eq!(
        remote_without_addr.caller_path_type.as_deref(),
        Some("remote_quic_http")
    );
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

    query.route = None;
    query.exclude_route = Some("chat".to_string());
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
