use super::*;

#[tokio::test]
async fn audit_list_returns_sparse_dto_and_never_exposes_detail_json() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    for (entry_id, occurred_at) in [
        (
            "00000000-0000-4000-8000-000000000001",
            "2026-01-01T00:00:00Z",
        ),
        (
            "00000000-0000-4000-8000-000000000002",
            "2026-01-01T00:00:01Z",
        ),
    ] {
        store
            .insert_audit_entry(
                entry_id,
                None,
                occurred_at,
                "runtime",
                "startup_complete",
                Some(r#"{"severity":"info","secret":"SENTINEL-AUDIT-SECRET"}"#),
            )
            .expect("seed audit row");
    }

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list audits");
    let json = serde_json::to_value(page).expect("serialize page");
    let items = json["items"].as_array().expect("items");
    assert_eq!(items.len(), 2);
    let item = &items[0];
    assert_eq!(item["entryId"], "00000000-0000-4000-8000-000000000002");
    assert_eq!(item["occurredAt"], "2026-01-01T00:00:01.000000000Z");
    assert_eq!(item["source"], "runtime");
    assert_eq!(item["code"], "startup_complete");
    assert_eq!(item["severity"], "info");
    let first_sequence = items[0]["sequence"].as_u64().expect("positive sequence");
    let second_sequence = items[1]["sequence"].as_u64().expect("positive sequence");
    assert!(first_sequence > 0);
    assert!(second_sequence > 0);
    assert_ne!(first_sequence, second_sequence);
    assert!(!json.to_string().contains("SENTINEL-AUDIT-SECRET"));
    assert!(items.iter().all(|item| item.get("detailJson").is_none()));
    assert!(!json.to_string().contains("requestId"));
}

#[tokio::test]
async fn audit_pagination_resumes_correctly_with_next_cursor() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    for i in 1..=3 {
        store
            .insert_audit_entry(
                &format!("00000000-0000-4000-8000-00000000000{i}"),
                None,
                &format!("2026-01-01T00:00:0{i}Z"),
                "cli",
                &format!("action-{i}"),
                None,
            )
            .expect("seed audit row");
    }

    let first = list_audits(&state, "/api/logs/audit?limit=1")
        .await
        .expect("first page");
    let first_json = serde_json::to_value(&first).expect("serialize first page");
    let cursor = first.next_cursor.expect("has next cursor");
    assert_eq!(first_json["items"].as_array().expect("items").len(), 1);

    let second = list_audits(&state, &format!("/api/logs/audit?limit=1&cursor={cursor}"))
        .await
        .expect("second page");
    let second_json = serde_json::to_value(&second).expect("serialize second page");
    assert_eq!(second_json["items"].as_array().expect("items").len(), 1);
    assert_ne!(
        second_json["items"][0]["entryId"], first_json["items"][0]["entryId"],
        "cursor should advance to a different row"
    );
}

#[tokio::test]
async fn audit_filters_by_source() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000010",
            None,
            "2026-01-01T00:00:00Z",
            "mesh",
            "peer_joined",
            None,
        )
        .expect("seed mesh row");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000011",
            None,
            "2026-01-01T00:00:01Z",
            "runtime",
            "startup_complete",
            None,
        )
        .expect("seed runtime row");

    let page = list_audits(&state, "/api/logs/audit?source=mesh&limit=10")
        .await
        .expect("filter by source");
    let json = serde_json::to_value(page).expect("serialize page");
    assert_eq!(json["items"].as_array().expect("items").len(), 1);
    assert_eq!(json["items"][0]["source"], "mesh");
}

#[tokio::test]
async fn audit_logging_service_filter_includes_legacy_rows_with_canonical_source() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    for (entry_id, occurred_at, source) in [
        (
            "00000000-0000-4000-8000-000000000012",
            "2026-01-01T00:00:00Z",
            "logging-runtime",
        ),
        (
            "00000000-0000-4000-8000-000000000013",
            "2026-01-01T00:00:01Z",
            "logging_service",
        ),
        (
            "00000000-0000-4000-8000-000000000014",
            "2026-01-01T00:00:02Z",
            "runtime",
        ),
    ] {
        store
            .insert_audit_entry(entry_id, None, occurred_at, source, "health_check", None)
            .expect("seed audit row");
    }

    let page = list_audits(&state, "/api/logs/audit?source=logging_service&limit=10")
        .await
        .expect("filter logging service rows");
    let json = serde_json::to_value(page).expect("serialize page");
    let items = json["items"].as_array().expect("items");

    assert_eq!(items.len(), 2);
    assert_eq!(
        items
            .iter()
            .map(|item| item["entryId"].as_str().expect("entry id"))
            .collect::<Vec<_>>(),
        vec![
            "00000000-0000-4000-8000-000000000013",
            "00000000-0000-4000-8000-000000000012",
        ]
    );
    assert!(items.iter().all(|item| item["source"] == "logging_service"));
    assert!(!json.to_string().contains("logging-runtime"));
}

#[tokio::test]
async fn audit_filters_by_severity() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000020",
            None,
            "2026-01-01T00:00:00Z",
            "logging_service",
            "health_check",
            Some(r#"{"severity":"info"}"#),
        )
        .expect("seed info row");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000021",
            None,
            "2026-01-01T00:00:01Z",
            "logging_service",
            "disk_pressure",
            Some(r#"{"severity":"warning"}"#),
        )
        .expect("seed warning row");

    let page = list_audits(&state, "/api/logs/audit?severity=warning&limit=10")
        .await
        .expect("filter by severity");
    let json = serde_json::to_value(page).expect("serialize page");
    assert_eq!(json["items"].as_array().expect("items").len(), 1);
    assert_eq!(json["items"][0]["severity"], "warning");
}

#[tokio::test]
async fn audit_list_exposes_typed_context_without_arbitrary_detail() {
    let (_temp, state) = runtime();
    state
        .store()
        .expect("store")
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000022",
            None,
            "2026-08-12T12:00:00Z",
            "runtime",
            "runtime_model_ready",
            Some(
                r#"{"severity":"info","context_version":1,"subject_kind":"model","subject_id":"local-gguf/sha256-safe","operation_id":"runtime-7","outcome":"ready","duration_ms":42,"secret":"SENTINEL-AUDIT-SECRET"}"#,
            ),
        )
        .expect("seed typed audit row");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list typed audit row");
    let json = serde_json::to_value(page).expect("serialize page");
    let row = &json["items"][0];
    assert_eq!(row["contextVersion"], 1);
    assert_eq!(row["subjectKind"], "model");
    assert_eq!(row["subjectId"], "local-gguf/sha256-safe");
    assert_eq!(row["operationId"], "runtime-7");
    assert_eq!(row["outcome"], "ready");
    assert_eq!(row["durationMs"], 42);
    assert!(!json.to_string().contains("SENTINEL-AUDIT-SECRET"));
}

#[tokio::test]
async fn audit_list_omits_malformed_command_summary() {
    let (_temp, state) = runtime();
    state
        .store()
        .expect("store")
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000024",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(
                r#"{"context_version":1,"command_summary":"mesh-llm gpus --draft run-benchmark --backend cuda"}"#,
            ),
        )
        .expect("seed malformed command summary");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list malformed command summary");
    let json = serde_json::to_value(page).expect("serialize audit page");
    assert!(json["items"][0].get("commandSummary").is_none());
}

#[tokio::test]
async fn audit_list_omits_duplicate_command_summary_flags() {
    let (_temp, state) = runtime();
    state
        .store()
        .expect("store")
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000027",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(r#"{"context_version":1,"command_summary":"mesh-llm models list --json --json"}"#),
        )
        .expect("seed duplicate command summary");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list duplicate command summary");
    let json = serde_json::to_value(page).expect("serialize audit page");
    assert!(json["items"][0].get("commandSummary").is_none());
}

#[tokio::test]
async fn audit_list_omits_deep_malformed_command_summary() {
    let (_temp, state) = runtime();
    state
        .store()
        .expect("store")
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000026",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
            Some(
                r#"{"context_version":1,"command_summary":"mesh-llm load unload status discover rotate-key setup --port 1234"}"#,
            ),
        )
        .expect("seed deep malformed command summary");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list deep malformed command summary");
    let json = serde_json::to_value(page).expect("serialize audit page");
    assert!(json["items"][0].get("commandSummary").is_none());
}

#[tokio::test]
async fn audit_list_preserves_valid_command_summary() {
    let (_temp, state) = runtime();
    state
        .store()
        .expect("store")
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000025",
            None,
            "2026-08-12T12:00:00Z",
            "cli",
            "command_completed",
             Some(r#"{"context_version":1,"command_summary":"mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"}"#),
        )
        .expect("seed valid command summary");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list valid command summary");
    let json = serde_json::to_value(page).expect("serialize audit page");
    assert_eq!(
        json["items"][0]["commandSummary"],
        "mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"
    );
}

#[tokio::test]
async fn audit_list_exposes_mesh_peer_direct_path_and_omits_relay_address() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000023",
            None,
            "2026-08-12T12:00:01Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"peer-direct","remote_addr":"192.168.1.44:11204","path_type":"direct"}"#,
            ),
        )
        .expect("seed direct peer audit");
    store
        .insert_audit_entry(
            "00000000-0000-4000-8000-000000000024",
            None,
            "2026-08-12T12:00:00Z",
            "mesh",
            "mesh_quic_inbound_accepted",
            Some(
                r#"{"context_version":1,"subject_kind":"mesh_peer","subject_id":"peer-relay","remote_addr":"203.0.113.10:443","path_type":"relay"}"#,
            ),
        )
        .expect("seed relay peer audit");

    let page = list_audits(&state, "/api/logs/audit?limit=10")
        .await
        .expect("list peer audits");
    let json = serde_json::to_value(page).expect("serialize page");
    let direct = &json["items"][0];
    let relay = &json["items"][1];

    assert_eq!(direct["subjectKind"], "mesh_peer");
    assert_eq!(direct["subjectId"], "peer-direct");
    assert_eq!(direct["remoteAddr"], "192.168.1.44:11204");
    assert_eq!(direct["pathType"], "direct");
    assert_eq!(relay["pathType"], "relay");
    assert!(relay.get("remoteAddr").is_none());
}

#[tokio::test]
async fn audit_filters_by_inclusive_canonical_time_bounds_before_pagination() {
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    for (suffix, occurred_at) in [
        (30, "2026-01-01T00:00:00Z"),
        (31, "2026-01-02T00:00:00Z"),
        (32, "2026-01-03T00:00:00Z"),
    ] {
        store
            .insert_audit_entry(
                &format!("00000000-0000-4000-8000-{suffix:012}"),
                None,
                occurred_at,
                "runtime",
                "bounded_action",
                None,
            )
            .expect("seed bounded audit row");
    }

    let page = list_audits(
        &state,
        "/api/logs/audit?limit=1&from=2026-01-02T00%3A00%3A00Z&to=2026-01-02T00%3A00%3A00Z",
    )
    .await
    .expect("filter by inclusive bounds");
    let json = serde_json::to_value(page).expect("serialize page");
    assert_eq!(json["items"].as_array().expect("items").len(), 1);
    assert_eq!(
        json["items"][0]["occurredAt"],
        "2026-01-02T00:00:00.000000000Z"
    );
    assert!(json["nextCursor"].is_null());
}

#[tokio::test]
async fn audit_rejects_invalid_query_parameters() {
    let (_temp, state) = runtime();

    for path in [
        "/api/logs/audit?limit=0",
        "/api/logs/audit?limit=101",
        "/api/logs/audit?limit=abc",
        "/api/logs/audit?source=bogus",
        "/api/logs/audit?severity=bogus",
        "/api/logs/audit?cursor=garbage",
        "/api/logs/audit?from=not-a-time",
        "/api/logs/audit?to=2026-01-01T00%3A00%3A00Z&from=2026-01-02T00%3A00%3A00Z",
        "/api/logs/audit?unknown=1",
    ] {
        let result = list_audits(&state, path).await;
        assert!(result.is_err(), "expected error for invalid query: {path}");
    }
}

#[tokio::test]
async fn audit_unmatched_path_returns_not_found() {
    assert!(matches!(classify("/api/logs/audit/extra"), Route::Unknown));
    assert!(matches!(
        classify("/api/logs/audit/sub/path"),
        Route::Unknown
    ));
}
