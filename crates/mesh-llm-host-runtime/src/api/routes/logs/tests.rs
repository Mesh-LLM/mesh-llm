use mesh_llm_events::logging::identifiers::RequestId;
use openai_frontend::{
    OpenAiBackendOperation, OpenAiFrontendRoute, OpenAiLifecycleContext, OpenAiLifecycleEvent,
    OpenAiRequestMethod, OpenAiTerminalResult, OpenAiUsage, Usage,
};

use super::*;

#[test]
fn export_route_classifies_before_method_validation() {
    let get_route = classify_mutating_route(Some("/api/logs/requests/export"), "GET")
        .expect("export remains a recognized mutation endpoint");
    assert!(matches!(get_route, MutatingRoute::Export));
    assert_eq!(method_error(get_route), LogsError::ExportMethodNotAllowed);

    let post_route = classify_mutating_route(Some("/api/logs/requests/export"), "POST")
        .expect("export remains dispatchable with POST");
    assert!(matches!(post_route, MutatingRoute::Export));
}

fn runtime() -> (tempfile::TempDir, LoggingRuntimeState) {
    let temp = tempfile::tempdir().expect("temporary application state root");
    let root = temp.path().join("logging-state");
    let foundation = crate::logging::LoggingFoundation::init(true, Some(&root));
    let config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(root),
        ..Default::default()
    };
    (temp, LoggingRuntimeState::initialize(&foundation, &config))
}

#[tokio::test]
async fn disk_prefix_cached_tokens_reach_durable_usage_dto() {
    let (_temp, state) = runtime();
    let request_id = RequestId::new();
    let context = OpenAiLifecycleContext::new(
        request_id,
        OpenAiRequestMethod::Post,
        OpenAiFrontendRoute::ChatCompletions,
    );
    let operation = OpenAiBackendOperation::ChatCompletionStream;
    let observer = state
        .openai_lifecycle_observer()
        .expect("OpenAI lifecycle observer");

    observer.observe(&OpenAiLifecycleEvent::Admitted {
        context: context.clone(),
    });
    // A disk-prefix restoration is surfaced by the backend through the
    // OpenAI usage detail; exercise that production conversion before logging.
    let disk_prefix_usage = Usage::new(21, 8).with_cached_tokens(13);
    observer.observe(&OpenAiLifecycleEvent::ResponseCompleted {
        context: context.clone(),
        operation,
        usage: OpenAiUsage::from(&disk_prefix_usage),
    });
    observer.observe(&OpenAiLifecycleEvent::StreamTerminal {
        context,
        result: OpenAiTerminalResult::Completed { status_code: 200 },
    });
    assert!(state.pump_persistence_for_test().await > 0);

    let request_key = request_id.as_uuid().to_string();
    let events = request_events(
        &state,
        "/api/logs/requests/id/events?limit=20",
        &request_key,
    )
    .await
    .expect("durable request events");
    let wire = serde_json::to_value(events).expect("event DTO JSON");
    let usage = wire["items"]
        .as_array()
        .expect("event items")
        .iter()
        .find(|event| event["kind"] == "usage_recorded")
        .expect("durable usage record");

    assert_eq!(usage["cachedPromptTokens"], 13);
}

#[tokio::test]
async fn list_merges_active_and_durable_without_duplicate_ids() {
    let (_temp, state) = runtime();
    let durable_id = RequestId::new().as_uuid().to_string();
    state
        .store()
        .expect("store")
        .insert_summary(
            &durable_id,
            Some("durable-model"),
            Some("management"),
            None,
            None,
            "2026-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("seed durable summary");

    let active_id = RequestId::new();
    let service = state.service_for_test().expect("logging service");
    let (active_guard, _) = service.register_request(active_id);

    let page = list_requests(&state, "/api/logs/requests?limit=10")
        .await
        .expect("list request core");
    let json = serde_json::to_value(page).expect("serialize page");
    let items = json["items"].as_array().expect("items");
    assert_eq!(items.len(), 2);
    assert!(items.iter().any(|item| item["requestId"] == durable_id));
    assert!(
        items
            .iter()
            .any(|item| item["requestId"] == active_id.as_uuid().to_string())
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| item["requestId"] == active_id.as_uuid().to_string())
            .count(),
        1
    );

    let first_page = list_requests(&state, "/api/logs/requests?limit=1")
        .await
        .expect("first merged page");
    let cursor = first_page.next_cursor.expect("active cursor");
    let second_page = list_requests(
        &state,
        &format!("/api/logs/requests?limit=1&cursor={cursor}"),
    )
    .await
    .expect("active cursor resumes into durable history");
    let json = serde_json::to_value(second_page).expect("serialize resumed page");
    assert_eq!(json["items"][0]["requestId"], durable_id);
    drop(active_guard);
}

#[test]
fn active_request_time_bounds_compare_instants_within_the_boundary_second() {
    let (_temp, state) = runtime();
    let entries = [
        (RequestId::new(), "2026-08-03T00:00:00Z"),
        (RequestId::new(), "2026-08-03T00:00:00.123Z"),
    ]
    .map(|(request_id, created_at)| RequestSummaryEntry {
        request_id: request_id.as_uuid().to_string(),
        state: "active".to_string(),
        created_at: created_at.to_string(),
        terminal_at: None,
        metadata: crate::logging::RequestSummaryMetadata::default(),
    });
    let facade = state.query_facade().expect("query facade");

    let from_page = list_requests_blocking(
        facade.clone(),
        entries.to_vec(),
        parse::request_query("/api/logs/requests?source=active&from=2026-08-03T00:00:00Z")
            .expect("parse lower bound"),
    )
    .expect("list active requests after lower bound");
    assert_eq!(from_page.items.len(), 2);
    assert!(
        from_page
            .items
            .iter()
            .all(|item| item.created_at().len() == "2026-08-03T00:00:00.000000000Z".len())
    );

    let to_page = list_requests_blocking(
        facade,
        entries.to_vec(),
        parse::request_query("/api/logs/requests?source=active&to=2026-08-03T00:00:00Z")
            .expect("parse upper bound"),
    )
    .expect("list active requests before upper bound");
    assert_eq!(to_page.items.len(), 1);
    assert_eq!(to_page.items[0].request_id(), entries[0].request_id);
}

#[test]
fn merged_request_cursor_orders_active_and_durable_within_one_second() {
    let (_temp, state) = runtime();
    let durable_id = RequestId::new().as_uuid().to_string();
    state
        .store()
        .expect("store")
        .insert_summary(
            &durable_id,
            None,
            None,
            None,
            None,
            "2026-08-03T00:00:00.100Z",
            None,
            None,
            None,
        )
        .expect("insert durable request");
    let active_id = RequestId::new().as_uuid().to_string();
    let active = RequestSummaryEntry {
        request_id: active_id.clone(),
        state: "active".to_string(),
        created_at: "2026-08-03T00:00:00.200Z".to_string(),
        terminal_at: None,
        metadata: crate::logging::RequestSummaryMetadata::default(),
    };
    let facade = state.query_facade().expect("query facade");

    let first = list_requests_blocking(
        facade.clone(),
        vec![active.clone()],
        parse::request_query("/api/logs/requests?limit=1").expect("parse first merged page"),
    )
    .expect("list first merged page");
    assert_eq!(first.items[0].request_id(), active_id);
    let cursor = first.next_cursor.expect("active cursor");

    let second = list_requests_blocking(
        facade,
        vec![active],
        parse::request_query(&format!("/api/logs/requests?limit=1&cursor={cursor}"))
            .expect("parse second merged page"),
    )
    .expect("list second merged page");
    assert_eq!(second.items[0].request_id(), durable_id);
    assert!(second.next_cursor.is_none());
}

#[tokio::test]
async fn active_request_uses_registered_metadata_before_durable_persistence() {
    let (_temp, state) = runtime();
    let request_id = RequestId::new();
    let service = state.service_for_test().expect("logging service");
    let (guard, _) = service.register_request_with_metadata(
        request_id,
        crate::logging::RequestSummaryMetadata::from_parts(
            Some("chat_completions"),
            None,
            None,
            None,
        ),
    );
    service.merge_request_metadata(
        request_id,
        crate::logging::RequestSummaryMetadata::from_parts(
            None,
            Some("acme/model"),
            Some("mesh"),
            Some("raw_ingress"),
        ),
    );

    assert!(
        state
            .store()
            .expect("store")
            .query_request(&request_id.as_uuid().to_string())
            .expect("query active durable row")
            .is_none()
    );
    let page = list_requests(
        &state,
        "/api/logs/requests?source=active&route=chat_completions&model=acme%2Fmodel&provider=mesh&engine=raw_ingress",
    )
    .await
    .expect("active metadata query");
    let json = serde_json::to_value(page).expect("active metadata JSON");
    assert_eq!(json["items"].as_array().expect("items").len(), 1);
    assert_eq!(
        json["items"][0]["requestId"],
        request_id.as_uuid().to_string()
    );
    assert_eq!(json["items"][0]["route"], "chat_completions");
    assert_eq!(json["items"][0]["model"], "acme/model");
    assert_eq!(json["items"][0]["provider"], "mesh");
    assert_eq!(json["items"][0]["engine"], "raw_ingress");
    assert_eq!(json["items"][0]["source"], "active");
    drop(guard);
}

#[tokio::test]
async fn active_request_listing_batches_durable_metadata_fallback() {
    let (_temp, state) = runtime();
    let service = state.service_for_test().expect("logging service");
    let registrations = (0..32)
        .map(|_| {
            let request_id = RequestId::new();
            let (guard, _) = service.register_request(request_id);
            (guard, request_id.as_uuid().to_string())
        })
        .collect::<Vec<_>>();
    state
        .store()
        .expect("store")
        .insert_summary(
            &registrations[0].1,
            Some("durable-fallback-model"),
            Some("chat_completions"),
            Some("mesh"),
            Some("raw_ingress"),
            "2026-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("seed durable metadata fallback");

    let page = list_requests(
        &state,
        "/api/logs/requests?source=active&limit=1&route=chat_completions&model=durable-fallback-model&provider=mesh&engine=raw_ingress",
    )
    .await
    .expect("active request page");

    assert_eq!(page.items.len(), 1);
    assert!(page.next_cursor.is_none());
    let item = serde_json::to_value(&page.items[0]).expect("active request DTO");
    assert_eq!(item["requestId"], registrations[0].1);
    assert_eq!(item["model"], "durable-fallback-model");
    let counts = state.query_counts_for_test();
    assert_eq!(
        counts.point_requests, 0,
        "listing active requests must not issue one durable point query per request"
    );
    assert_eq!(
        counts.batch_requests, 1,
        "listing active requests must use one bounded durable metadata batch"
    );
    drop(registrations);
}

#[tokio::test]
async fn registered_metadata_is_persisted_and_durably_filterable_without_fabrication() {
    let (_temp, state) = runtime();
    let service = state.service_for_test().expect("logging service");

    let known = RequestId::new();
    let (known_guard, _) = service.register_request_with_metadata(
        known,
        crate::logging::RequestSummaryMetadata::from_parts(
            Some("chat_completions"),
            None,
            None,
            None,
        ),
    );
    service.merge_request_metadata(
        known,
        crate::logging::RequestSummaryMetadata::from_parts(
            None,
            Some("acme/model"),
            Some("mesh"),
            Some("raw_ingress"),
        ),
    );
    service
        .transition_terminal(
            known,
            &known_guard,
            crate::logging::TerminalOutcome::Completed,
        )
        .expect("known request terminalizes");

    let absent = RequestId::new();
    let (absent_guard, _) = service.register_request(absent);
    service
        .transition_terminal(
            absent,
            &absent_guard,
            crate::logging::TerminalOutcome::Completed,
        )
        .expect("metadata-absent request terminalizes");
    assert!(service.pump_sync().await > 0);

    let page = list_requests(
        &state,
        "/api/logs/requests?source=durable&route=chat_completions&model=acme%2Fmodel&provider=mesh&engine=raw_ingress&outcome=completed",
    )
    .await
    .expect("durable metadata query");
    let json = serde_json::to_value(page).expect("durable metadata JSON");
    assert_eq!(json["items"].as_array().expect("items").len(), 1);
    assert_eq!(json["items"][0]["requestId"], known.as_uuid().to_string());
    assert_eq!(json["items"][0]["route"], "chat_completions");
    assert_eq!(json["items"][0]["model"], "acme/model");
    assert_eq!(json["items"][0]["provider"], "mesh");
    assert_eq!(json["items"][0]["engine"], "raw_ingress");
    assert_eq!(json["items"][0]["source"], "durable");

    let absent_detail = request_detail(&state, &absent.as_uuid().to_string())
        .await
        .expect("metadata-absent detail");
    let absent_json = serde_json::to_value(absent_detail).expect("absent detail JSON");
    assert_eq!(absent_json["route"], serde_json::Value::Null);
    assert_eq!(absent_json["model"], serde_json::Value::Null);
    assert_eq!(absent_json["provider"], serde_json::Value::Null);
    assert_eq!(absent_json["engine"], serde_json::Value::Null);
}

#[tokio::test]
async fn detail_and_related_routes_are_typed_and_keep_envelope_private() {
    let (_temp, state) = runtime();
    let request_id = RequestId::new();
    let request_key = request_id.as_uuid().to_string();
    let store = state.store().expect("store");
    store
        .insert_summary(
            &request_key,
            None,
            None,
            None,
            None,
            "2026-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("seed durable summary");
    let event_id = mesh_llm_events::logging::identifiers::EventId::new();
    let envelope = mesh_llm_events::logging::envelope::CanonicalEnvelope::new(
        event_id,
        request_id,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        1,
        "2026-01-01T00:00:01.000000000Z".into(),
        mesh_llm_events::logging::events::LifecycleEvent::Admitted {
            model: Some("Bearer never-expose".into()),
            method: Some("GET".into()),
        },
    );
    store
        .insert_lifecycle_event(
            &request_key,
            &event_id.as_uuid().to_string(),
            &serde_json::to_string(&envelope).expect("canonical envelope"),
            "2026-01-01T00:00:01Z",
        )
        .expect("seed lifecycle record");

    let detail = request_detail(&state, &request_key)
        .await
        .expect("detail core");
    assert_eq!(
        serde_json::to_value(detail).expect("detail json")["requestId"],
        request_key
    );

    let events = request_events(&state, "/api/logs/requests/id/events", &request_key)
        .await
        .expect("related events core");
    let json = serde_json::to_string(&events).expect("event dto JSON");
    assert!(json.contains("admitted"));
    assert!(!json.contains("never-expose"));
    assert!(!json.contains("payload_json"));
}

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
