use std::collections::BTreeMap;

use mesh_llm_log_store::{
    AuditEntryFilters, AuditEntryRow, LogStore, LogStoreError, Page, QueryPage, RequestQuery,
    RequestRecord,
};

type LegacyAuditPageMethod = fn(
    &LogStore,
    Option<usize>,
    Option<&str>,
    AuditEntryFilters,
) -> Result<Page<AuditEntryRow>, LogStoreError>;
type LegacyAuditSequenceMethod =
    fn(&LogStore, u64, usize, AuditEntryFilters) -> Result<Vec<AuditEntryRow>, LogStoreError>;
type LegacySummaryMetadataMethod = fn(
    &LogStore,
    &str,
    Option<&str>,
    Option<&str>,
    Option<&str>,
    Option<&str>,
    &str,
) -> Result<(), LogStoreError>;

#[test]
fn released_request_record_struct_literal_remains_source_compatible() {
    let record = RequestRecord {
        request_id: "released-request".to_owned(),
        outcome: "completed".to_owned(),
        created_at: "2026-08-22T12:00:00.000000000Z".to_owned(),
        terminal_at: Some("2026-08-22T12:00:01.000000000Z".to_owned()),
        route: Some("chat".to_owned()),
        model: Some("released-model".to_owned()),
        provider: Some("released-provider".to_owned()),
        engine: Some("released-engine".to_owned()),
        status_code: Some(200),
    };

    assert_eq!(record.request_id, "released-request");
}

#[test]
fn released_audit_entry_row_struct_literal_remains_source_compatible() {
    let row = AuditEntryRow {
        sequence: 7,
        entry_id: "released-audit".to_owned(),
        request_id: Some("released-request".to_owned()),
        occurred_at: "2026-08-22T12:00:02.000000000Z".to_owned(),
        source: "runtime".to_owned(),
        code: "runtime_ready".to_owned(),
        severity: None,
        context_version: Some(1),
        subject_kind: Some("runtime".to_owned()),
        subject_id: Some("released-host".to_owned()),
        operation_id: Some("released-operation".to_owned()),
        correlation_request_id: Some("released-correlation".to_owned()),
        reason_code: Some("released-reason".to_owned()),
        outcome: Some("ready".to_owned()),
        duration_ms: Some(11),
        numeric_summaries: BTreeMap::from([("models".to_owned(), 2)]),
    };

    assert_eq!(row.entry_id, "released-audit");
}

#[test]
fn released_query_and_list_methods_keep_legacy_return_shapes() {
    let _: fn(&LogStore, &str) -> Result<Option<RequestRecord>, LogStoreError> =
        LogStore::query_request;
    let _: fn(&LogStore, &[String]) -> Result<Vec<RequestRecord>, LogStoreError> =
        LogStore::query_requests_by_ids;
    let _: fn(&LogStore, &RequestQuery) -> Result<QueryPage<RequestRecord>, LogStoreError> =
        LogStore::query_requests;
    let _: LegacyAuditPageMethod = LogStore::list_audit_entries;
    let _: LegacyAuditSequenceMethod = LogStore::list_audit_entries_after_sequence;
}

#[test]
fn released_summary_metadata_method_keeps_legacy_signature() {
    let _: LegacySummaryMetadataMethod = LogStore::upsert_summary_metadata;
}
