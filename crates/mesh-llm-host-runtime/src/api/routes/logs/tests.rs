use mesh_llm_events::logging::identifiers::RequestId;
use openai_frontend::{
    OpenAiBackendOperation, OpenAiFrontendRoute, OpenAiLifecycleContext, OpenAiLifecycleEvent,
    OpenAiRequestMethod, OpenAiTerminalResult, OpenAiUsage, Usage,
};

use super::*;

mod audit;
mod audit_sanitization;

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
async fn cached_prefix_tokens_reach_durable_usage_dto() {
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
    // Cached-prefix restoration is surfaced by the backend through the OpenAI
    // usage detail; exercise that production conversion before logging.
    let cached_prefix_usage = Usage::new(21, 8).with_cached_tokens(13);
    observer.observe(&OpenAiLifecycleEvent::ResponseCompleted {
        context: context.clone(),
        operation,
        usage: OpenAiUsage::from(&cached_prefix_usage),
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
fn route_exclusions_apply_to_every_merged_active_and_durable_page() {
    // Given
    let (_temp, state) = runtime();
    let store = state.store().expect("store");
    for (request_id, route, created_at) in [
        ("durable-visible", Some("responses"), "2026-08-03T00:00:01Z"),
        (
            "durable-hidden-management",
            Some("management_get_status"),
            "2026-08-03T00:00:02Z",
        ),
        (
            "durable-hidden-models",
            Some("models"),
            "2026-08-03T00:00:03Z",
        ),
    ] {
        store
            .insert_summary(
                request_id, None, route, None, None, created_at, None, None, None,
            )
            .expect("seed durable summary");
    }
    let active = [
        (
            "active-hidden-management",
            Some("management_post"),
            "2026-08-03T00:00:04Z",
        ),
        ("active-visible-null", None, "2026-08-03T00:00:05Z"),
        (
            "active-visible-chat",
            Some("chat_completions"),
            "2026-08-03T00:00:06Z",
        ),
        (
            "active-hidden-models",
            Some("models"),
            "2026-08-03T00:00:07Z",
        ),
    ]
    .map(|(request_id, route, created_at)| RequestSummaryEntry {
        request_id: request_id.to_string(),
        state: "active".to_string(),
        created_at: created_at.to_string(),
        terminal_at: None,
        metadata: crate::logging::RequestSummaryMetadata::from_parts(route, None, None, None),
    })
    .to_vec();
    let facade = state.query_facade().expect("query facade");
    let base_path =
        "/api/logs/requests?limit=1&exclude_route=models&exclude_route_prefix=management_";

    // When
    let first = list_requests_blocking(
        facade.clone(),
        active.clone(),
        parse::request_query(base_path).expect("parse first page"),
    )
    .expect("list first page");
    let first_cursor = first.next_cursor.expect("first page cursor");
    let second = list_requests_blocking(
        facade.clone(),
        active.clone(),
        parse::request_query(&format!("{base_path}&cursor={first_cursor}"))
            .expect("parse second page"),
    )
    .expect("list second page");
    let second_cursor = second.next_cursor.expect("second page cursor");
    let third = list_requests_blocking(
        facade,
        active,
        parse::request_query(&format!("{base_path}&cursor={second_cursor}"))
            .expect("parse third page"),
    )
    .expect("list third page");

    // Then
    assert_eq!(first.items[0].request_id(), "active-visible-chat");
    assert_eq!(second.items[0].request_id(), "active-visible-null");
    assert_eq!(third.items[0].request_id(), "durable-visible");
    assert!(third.next_cursor.is_none());
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
        )
        .with_caller_identity(
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
            Some("192.0.2.42:11204"),
            Some(crate::logging::CallerPathType::RemoteQuicHttp),
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
    assert_eq!(
        json["items"][0]["callerEndpointId"],
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );
    assert_eq!(json["items"][0]["callerAddr"], "192.0.2.42:11204");
    assert_eq!(json["items"][0]["callerPathType"], "remote_quic_http");
    assert_eq!(json["items"][0]["source"], "active");
    drop(guard);
}

#[tokio::test]
async fn active_and_durable_api_json_expose_endpoint_only_caller_without_path_fields() {
    let (_temp, state) = runtime();
    let service = state.service_for_test().expect("logging service");
    let active_request_id = RequestId::new();
    let durable_request_id = RequestId::new();
    let active_endpoint_id = "ad".repeat(32);
    let durable_endpoint_id = "ae".repeat(32);
    let (active_guard, _) = service.register_request_with_metadata(
        active_request_id,
        crate::logging::RequestSummaryMetadata::from_parts(
            Some("responses"),
            Some("active-model"),
            None,
            None,
        )
        .with_caller_identity(Some(&active_endpoint_id), None, None),
    );
    let (durable_guard, _) = service.register_request_with_metadata(
        durable_request_id,
        crate::logging::RequestSummaryMetadata::from_parts(
            Some("responses"),
            Some("durable-model"),
            None,
            None,
        )
        .with_caller_identity(Some(&durable_endpoint_id), None, None),
    );
    service
        .transition_terminal(
            durable_request_id,
            &durable_guard,
            crate::logging::TerminalOutcome::Completed,
        )
        .expect("durable request terminalizes");
    assert!(service.pump_sync().await > 0);

    let active_page = list_requests(
        &state,
        "/api/logs/requests?source=active&model=active-model",
    )
    .await
    .expect("active endpoint-only request");
    let durable_page = list_requests(
        &state,
        "/api/logs/requests?source=durable&model=durable-model&outcome=completed",
    )
    .await
    .expect("durable endpoint-only request");

    for (page, endpoint_id) in [
        (active_page, active_endpoint_id.as_str()),
        (durable_page, durable_endpoint_id.as_str()),
    ] {
        let json = serde_json::to_value(page).expect("request page JSON");
        assert_eq!(json["items"].as_array().expect("items").len(), 1);
        assert_eq!(json["items"][0]["callerEndpointId"], endpoint_id);
        assert!(json["items"][0].get("callerAddr").is_none());
        assert!(json["items"][0].get("callerPathType").is_none());
    }
    drop(active_guard);
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
        )
        .with_caller_identity(
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
            None,
            Some(crate::logging::CallerPathType::Relay),
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
    assert_eq!(
        json["items"][0]["callerEndpointId"],
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    );
    assert!(json["items"][0].get("callerAddr").is_none());
    assert_eq!(json["items"][0]["callerPathType"], "relay");
    assert_eq!(json["items"][0]["source"], "durable");

    let absent_detail = request_detail(&state, &absent.as_uuid().to_string())
        .await
        .expect("metadata-absent detail");
    let absent_json = serde_json::to_value(absent_detail).expect("absent detail JSON");
    assert_eq!(absent_json["route"], serde_json::Value::Null);
    assert_eq!(absent_json["model"], serde_json::Value::Null);
    assert_eq!(absent_json["provider"], serde_json::Value::Null);
    assert_eq!(absent_json["engine"], serde_json::Value::Null);
    assert!(absent_json.get("callerEndpointId").is_none());
    assert!(absent_json.get("callerAddr").is_none());
    assert!(absent_json.get("callerPathType").is_none());
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
