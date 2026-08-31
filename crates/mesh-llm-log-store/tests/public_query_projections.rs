use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use mesh_llm_log_store::{
    AuditEntryDetail, AuditEntryFilters, AuditEntryRow, AuditEntrySeverity, LogStore, QuerySort,
    RealClock, RequestQuery, RequestRecord, RequestRecordWithCaller,
};

const CALLER_ENDPOINT_ID: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const CALLER_ADDR: &str = "192.0.2.42:11204";
const CALLER_PATH_TYPE: &str = "remote_quic_http";
const AUDIT_REMOTE_ADDR: &str = "198.51.100.27:443";
const AUDIT_PATH_TYPE: &str = "direct";
const AUDIT_COMMAND: &str = "mesh-llm runtime guardrails --mode metrics --port 41731";

fn open_store() -> (tempfile::TempDir, LogStore) {
    let root = tempfile::tempdir().expect("create public projection store");
    let store = LogStore::open(root.path(), Arc::new(RealClock)).expect("open log store");
    (root, store)
}

fn request_query(limit: usize, cursor: Option<String>) -> RequestQuery {
    RequestQuery {
        limit,
        cursor,
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

fn insert_request_with_caller(
    store: &LogStore,
    request_id: &str,
    created_at: &str,
    caller_endpoint_id: &str,
    caller_addr: &str,
) {
    store
        .upsert_summary_metadata_with_caller(
            request_id,
            Some("caller-model"),
            Some("responses"),
            Some("caller-provider"),
            Some("caller-engine"),
            Some(caller_endpoint_id),
            Some(caller_addr),
            Some(CALLER_PATH_TYPE),
            created_at,
        )
        .expect("insert request caller metadata");
}

fn assert_released_request(record: &RequestRecord, request_id: &str) {
    assert_eq!(record.request_id, request_id);
    assert_eq!(record.outcome, "active");
    assert_eq!(record.route.as_deref(), Some("responses"));
    assert_eq!(record.model.as_deref(), Some("caller-model"));
    assert_eq!(record.provider.as_deref(), Some("caller-provider"));
    assert_eq!(record.engine.as_deref(), Some("caller-engine"));
}

fn assert_caller(detailed: &RequestRecordWithCaller, endpoint_id: &str, caller_addr: &str) {
    assert_eq!(detailed.caller_endpoint_id.as_deref(), Some(endpoint_id));
    assert_eq!(detailed.caller_addr.as_deref(), Some(caller_addr));
    assert_eq!(detailed.caller_path_type.as_deref(), Some(CALLER_PATH_TYPE));
}

#[test]
fn detailed_request_preserves_caller_while_legacy_query_keeps_released_projection() {
    // Given
    let (_root, store) = open_store();
    insert_request_with_caller(
        &store,
        "request-with-caller",
        "2026-08-22T12:00:00Z",
        CALLER_ENDPOINT_ID,
        CALLER_ADDR,
    );

    // When
    let detailed = store
        .query_request_with_caller("request-with-caller")
        .expect("query detailed request")
        .expect("detailed request");
    let legacy = store
        .query_request("request-with-caller")
        .expect("query legacy request")
        .expect("legacy request");

    // Then
    assert_released_request(&detailed.request, "request-with-caller");
    assert_caller(&detailed, CALLER_ENDPOINT_ID, CALLER_ADDR);
    assert_released_request(&legacy, "request-with-caller");
}

#[test]
fn detailed_request_pagination_keeps_legacy_order_and_cursor() {
    // Given
    let (_root, store) = open_store();
    for (request_id, created_at, endpoint_digit, caller_addr) in [
        ("request-a", "2026-08-22T12:00:01Z", 'a', "192.0.2.10:11204"),
        ("request-b", "2026-08-22T12:00:02Z", 'b', "192.0.2.11:11204"),
        ("request-c", "2026-08-22T12:00:03Z", 'c', "192.0.2.12:11204"),
    ] {
        let endpoint_id = endpoint_digit.to_string().repeat(64);
        insert_request_with_caller(&store, request_id, created_at, &endpoint_id, caller_addr);
    }

    // When
    let detailed_first = store
        .query_requests_with_caller(&request_query(2, None))
        .expect("query first detailed page");
    let legacy_first = store
        .query_requests(&request_query(2, None))
        .expect("query first legacy page");
    let detailed_second = store
        .query_requests_with_caller(&request_query(2, detailed_first.next_cursor.clone()))
        .expect("query second detailed page");

    // Then
    assert_eq!(detailed_first.next_cursor, legacy_first.next_cursor);
    assert_eq!(
        detailed_first
            .items
            .iter()
            .map(|item| item.request.request_id.as_str())
            .collect::<Vec<_>>(),
        ["request-c", "request-b"]
    );
    assert_eq!(
        legacy_first
            .items
            .iter()
            .map(|item| item.request_id.as_str())
            .collect::<Vec<_>>(),
        ["request-c", "request-b"]
    );
    assert_eq!(detailed_second.items[0].request.request_id, "request-a");
    assert!(detailed_second.next_cursor.is_none());
}

#[test]
fn detailed_request_batch_omits_missing_ids_across_chunk_boundary() {
    // Given
    let (_root, store) = open_store();
    let mut requested_ids = Vec::new();
    for index in 0..=100_u8 {
        let request_id = format!("request-{index:03}");
        let endpoint_id = format!("{index:064x}");
        let caller_addr = format!("192.0.2.{}:11204", index + 1);
        insert_request_with_caller(
            &store,
            &request_id,
            "2026-08-22T12:00:00Z",
            &endpoint_id,
            &caller_addr,
        );
        requested_ids.push(request_id);
    }
    requested_ids.push("request-missing".to_owned());

    // When
    let detailed = store
        .query_requests_by_ids_with_caller(&requested_ids)
        .expect("query detailed request batch");
    let legacy = store
        .query_requests_by_ids(&requested_ids)
        .expect("query legacy request batch");

    // Then
    assert_eq!(detailed.len(), 101);
    assert_eq!(legacy.len(), 101);
    let detailed_by_id = detailed
        .iter()
        .map(|item| (item.request.request_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let legacy_ids = legacy
        .iter()
        .map(|item| item.request_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(!detailed_by_id.contains_key("request-missing"));
    assert!(!legacy_ids.contains("request-missing"));
    for index in [99_u8, 100_u8] {
        let request_id = format!("request-{index:03}");
        let endpoint_id = format!("{index:064x}");
        let caller_addr = format!("192.0.2.{}:11204", index + 1);
        assert_caller(
            detailed_by_id
                .get(request_id.as_str())
                .expect("detailed request at chunk boundary"),
            &endpoint_id,
            &caller_addr,
        );
    }
}

fn insert_detailed_audit(
    store: &LogStore,
    entry_id: &str,
    occurred_at: &str,
    remote_addr: &str,
    command_summary: &str,
) {
    let detail = serde_json::json!({
        "severity": "info",
        "context_version": 1,
        "subject_kind": "mesh_peer",
        "subject_id": entry_id,
        "remote_addr": remote_addr,
        "path_type": AUDIT_PATH_TYPE,
        "operation_id": format!("operation-{entry_id}"),
        "request_id": format!("correlation-{entry_id}"),
        "reason_code": "accepted",
        "outcome": "ready",
        "command_summary": command_summary,
        "duration_ms": 17,
        "numeric_summaries": {"peers": 3}
    });
    store
        .insert_audit_entry(
            entry_id,
            None,
            occurred_at,
            "cli",
            "command_completed",
            Some(&detail.to_string()),
        )
        .expect("insert detailed audit entry");
}

fn assert_released_audit(row: &AuditEntryRow, entry_id: &str) {
    assert_eq!(row.entry_id, entry_id);
    assert_eq!(row.source, "cli");
    assert_eq!(row.code, "command_completed");
    assert_eq!(row.severity, Some(AuditEntrySeverity::Info));
    assert_eq!(row.context_version, Some(1));
    assert_eq!(row.subject_kind.as_deref(), Some("mesh_peer"));
    assert_eq!(row.subject_id.as_deref(), Some(entry_id));
    assert_eq!(row.reason_code.as_deref(), Some("accepted"));
    assert_eq!(row.outcome.as_deref(), Some("ready"));
    assert_eq!(row.duration_ms, Some(17));
    assert_eq!(row.numeric_summaries.get("peers"), Some(&3));
}

fn assert_audit_detail(
    detail: &AuditEntryDetail,
    entry_id: &str,
    remote_addr: &str,
    command_summary: &str,
) {
    assert_released_audit(&detail.entry, entry_id);
    assert_eq!(detail.remote_addr.as_deref(), Some(remote_addr));
    assert_eq!(detail.path_type.as_deref(), Some(AUDIT_PATH_TYPE));
    assert_eq!(detail.command_summary.as_deref(), Some(command_summary));
}

#[test]
fn detailed_audit_preserves_path_remote_and_command_while_legacy_list_stays_released() {
    // Given
    let (_root, store) = open_store();
    insert_detailed_audit(
        &store,
        "audit-detailed",
        "2026-08-22T12:00:00Z",
        AUDIT_REMOTE_ADDR,
        AUDIT_COMMAND,
    );

    // When
    let detailed = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .expect("list detailed audit entries");
    let legacy = store
        .list_audit_entries(Some(1), None, AuditEntryFilters::default())
        .expect("list legacy audit entries");

    // Then
    assert_audit_detail(
        &detailed.items[0],
        "audit-detailed",
        AUDIT_REMOTE_ADDR,
        AUDIT_COMMAND,
    );
    assert_released_audit(&legacy.items[0], "audit-detailed");
}

#[test]
fn detailed_audit_pagination_keeps_legacy_order_and_cursor_boundary() {
    // Given
    let (_root, store) = open_store();
    for (entry_id, remote_addr, command) in [
        ("audit-a", "198.51.100.10:443", AUDIT_COMMAND),
        ("audit-b", "198.51.100.11:443", AUDIT_COMMAND),
        ("audit-c", "198.51.100.12:443", AUDIT_COMMAND),
    ] {
        insert_detailed_audit(
            &store,
            entry_id,
            "2026-08-22T12:00:00Z",
            remote_addr,
            command,
        );
    }

    // When
    let detailed_first = store
        .list_audit_entry_details(Some(2), None, AuditEntryFilters::default())
        .expect("list first detailed audit page");
    let legacy_first = store
        .list_audit_entries(Some(2), None, AuditEntryFilters::default())
        .expect("list first legacy audit page");
    let detailed_second = store
        .list_audit_entry_details(
            Some(2),
            detailed_first.next_cursor.as_deref(),
            AuditEntryFilters::default(),
        )
        .expect("list second detailed audit page");

    // Then
    assert_eq!(detailed_first.next_cursor, legacy_first.next_cursor);
    assert_eq!(
        detailed_first
            .items
            .iter()
            .map(|item| item.entry.entry_id.as_str())
            .collect::<Vec<_>>(),
        ["audit-c", "audit-b"]
    );
    assert_eq!(detailed_second.items[0].entry.entry_id, "audit-a");
    assert!(detailed_second.next_cursor.is_none());
}

#[test]
fn detailed_audit_sequence_reconciliation_resumes_after_exact_boundary() {
    // Given
    let (root, store) = open_store();
    let writer = LogStore::open(root.path(), Arc::new(RealClock)).expect("open audit writer");
    for (entry_id, remote_addr, command) in [
        ("audit-sequence-a", "203.0.113.10:443", AUDIT_COMMAND),
        ("audit-sequence-b", "203.0.113.11:443", AUDIT_COMMAND),
        ("audit-sequence-c", "203.0.113.12:443", AUDIT_COMMAND),
    ] {
        insert_detailed_audit(
            &writer,
            entry_id,
            "2026-08-22T12:00:00Z",
            remote_addr,
            command,
        );
    }

    // When
    let first = store
        .list_audit_entry_details_after_sequence(0, 2, AuditEntryFilters::default())
        .expect("reconcile first detailed audit batch");
    let boundary = u64::try_from(first[1].entry.sequence).expect("positive audit sequence");
    let second = store
        .list_audit_entry_details_after_sequence(boundary, 2, AuditEntryFilters::default())
        .expect("reconcile second detailed audit batch");

    // Then
    assert_eq!(
        first
            .iter()
            .map(|item| item.entry.entry_id.as_str())
            .collect::<Vec<_>>(),
        ["audit-sequence-a", "audit-sequence-b"]
    );
    assert_eq!(second.len(), 1);
    assert_audit_detail(
        &second[0],
        "audit-sequence-c",
        "203.0.113.12:443",
        AUDIT_COMMAND,
    );
    assert!(second[0].entry.sequence > i64::try_from(boundary).expect("sequence fits i64"));
}
