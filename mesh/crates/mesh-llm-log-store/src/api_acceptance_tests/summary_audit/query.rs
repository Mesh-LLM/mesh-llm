use super::*;

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
fn audit_entries_drop_malformed_command_summaries_at_durable_boundary() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-command-summary-malformed",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(
                r#"{"context_version":1,"command_summary":"mesh-llm gpus --draft run-benchmark --backend cuda"}"#,
            ),
        )
        .unwrap();

    let page = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .unwrap();
    assert!(page.items[0].command_summary.is_none());
}

#[test]
fn audit_entries_drop_duplicate_command_summary_flags_at_durable_boundary() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-command-summary-duplicate",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(r#"{"context_version":1,"command_summary":"mesh-llm models list --json --json"}"#),
        )
        .unwrap();

    let page = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .unwrap();
    assert!(page.items[0].command_summary.is_none());
}

#[test]
fn audit_entries_drop_deep_malformed_command_summaries_at_durable_boundary() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-command-summary-deep",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(
                r#"{"context_version":1,"command_summary":"mesh-llm load unload status discover rotate-key setup --port 1234"}"#,
            ),
        )
        .unwrap();

    let page = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .unwrap();
    assert!(page.items[0].command_summary.is_none());
}

#[test]
fn audit_entries_drop_noncanonical_and_raw_relay_command_summaries_at_durable_boundary() {
    let (store, _, _tmp) = open_store();
    for (request_id, summary) in [
        (
            "audit-command-summary-whitespace",
            "mesh-llm  models list --json",
        ),
        (
            "audit-command-summary-raw-relay",
            "mesh-llm load name [REDACTED] --relay private-relay",
        ),
    ] {
        store
            .insert_audit_entry(
                request_id,
                None,
                "2026-08-12T12:00:00Z",
                "cli",
                "command_completed",
                Some(&format!(
                    r#"{{"context_version":1,"command_summary":"{summary}"}}"#
                )),
            )
            .unwrap();
    }

    let page = store
        .list_audit_entry_details(Some(2), None, AuditEntryFilters::default())
        .unwrap();
    assert!(page.items.iter().all(|item| item.command_summary.is_none()));
}

#[test]
fn audit_entries_preserve_valid_command_summaries_at_durable_boundary() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-command-summary-valid",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(
                r#"{"context_version":1,"command_summary":"mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"}"#,
            ),
        )
        .unwrap();

    let page = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .unwrap();
    assert_eq!(
        page.items[0].command_summary.as_deref(),
        Some("mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]")
    );
}

#[test]
fn audit_entries_preserve_direct_mesh_peer_identity_and_path() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-peer-direct",
            None,
            "2026-08-12T12:00:00Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","remote_addr":"192.168.1.44:11204","path_type":"direct"}"#,
            ),
        )
        .expect("insert direct peer audit");

    let page = store
        .list_audit_entry_details(Some(1), None, AuditEntryFilters::default())
        .expect("list direct peer audit");
    let row = &page.items[0];

    assert_eq!(row.entry.subject_kind.as_deref(), Some("mesh_peer"));
    assert_eq!(
        row.entry.subject_id.as_deref(),
        Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    );
    assert_eq!(row.remote_addr.as_deref(), Some("192.168.1.44:11204"));
    assert_eq!(row.path_type.as_deref(), Some("direct"));
}

#[test]
fn audit_entries_omit_relay_address_and_parse_legacy_path_as_absent() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-peer-relay",
            None,
            "2026-08-12T12:00:01Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"peer-relay","remote_addr":"203.0.113.10:443","path_type":"relay"}"#,
            ),
        )
        .expect("insert relay peer audit");
    store
        .insert_audit_entry(
            "audit-legacy",
            None,
            "2026-08-12T12:00:00Z",
            "runtime",
            "runtime_ready",
            Some(r#"{"context_version":1,"subject_kind":"runtime","subject_id":"host"}"#),
        )
        .expect("insert legacy audit");

    let page = store
        .list_audit_entry_details(Some(2), None, AuditEntryFilters::default())
        .expect("list relay and legacy audits");
    let relay = &page.items[0];
    let legacy = &page.items[1];

    assert_eq!(relay.path_type.as_deref(), Some("relay"));
    assert!(relay.remote_addr.is_none());
    assert!(legacy.path_type.is_none());
    assert!(legacy.remote_addr.is_none());
}

#[test]
fn audit_entries_drop_unknown_paths_and_invalid_direct_addresses() {
    let (store, _, _tmp) = open_store();
    store
        .insert_audit_entry(
            "audit-peer-unknown-path",
            None,
            "2026-08-12T12:00:01Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"peer-unknown","remote_addr":"192.168.1.44:11204","path_type":"proxy"}"#,
            ),
        )
        .expect("insert unknown peer path audit");
    store
        .insert_audit_entry(
            "audit-peer-invalid-address",
            None,
            "2026-08-12T12:00:00Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"peer-direct","remote_addr":"not-a-socket-address","path_type":"direct"}"#,
            ),
        )
        .expect("insert invalid direct address audit");

    let page = store
        .list_audit_entry_details(Some(2), None, AuditEntryFilters::default())
        .expect("list bounded peer path audits");
    let unknown_path = &page.items[0];
    let invalid_address = &page.items[1];

    assert!(unknown_path.path_type.is_none());
    assert!(unknown_path.remote_addr.is_none());
    assert_eq!(invalid_address.path_type.as_deref(), Some("direct"));
    assert!(invalid_address.remote_addr.is_none());
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
