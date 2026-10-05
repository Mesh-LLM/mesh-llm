use super::*;

fn initialized_state() -> (Arc<LoggingRuntimeState>, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = Arc::new(LoggingRuntimeState::initialize(
        &foundation,
        &mesh_llm_config::LoggingConfig::default(),
    ));
    (state, root)
}

fn provisional_local_metadata() -> RequestSummaryMetadata {
    RequestSummaryMetadata::from_parts(
        Some("responses"),
        Some("model-a"),
        Some("provider-a"),
        Some("engine-a"),
    )
    .with_source(Some("direct_http"))
    .with_method(Some("POST"))
    .with_caller_identity(
        None,
        Some("127.0.0.1:40123"),
        Some(CallerPathType::LocalHttp),
    )
}

fn authenticated_direct_metadata(endpoint_id: &str) -> RequestSummaryMetadata {
    RequestSummaryMetadata::default().with_caller_identity(
        Some(endpoint_id),
        Some("192.0.2.71:11204"),
        Some(CallerPathType::RemoteQuicHttp),
    )
}

fn authenticated_relay_metadata(endpoint_id: &str) -> RequestSummaryMetadata {
    RequestSummaryMetadata::default().with_caller_identity(
        Some(endpoint_id),
        None,
        Some(CallerPathType::Relay),
    )
}

fn assert_active_caller(
    state: &LoggingRuntimeState,
    request_id: mesh_llm_events::logging::identifiers::RequestId,
    endpoint_id: &str,
    address: Option<&str>,
    path_type: &str,
) {
    let active = state
        .service
        .as_ref()
        .expect("logging service")
        .registry_ref()
        .get_active(&request_id.as_uuid().to_string())
        .expect("active request");
    assert_eq!(active.metadata.caller_endpoint_id(), Some(endpoint_id));
    assert_eq!(active.metadata.caller_addr(), address);
    assert_eq!(active.metadata.caller_path_type(), Some(path_type));
    assert_eq!(active.metadata.route(), Some("responses"));
    assert_eq!(active.metadata.model(), Some("model-a"));
    assert_eq!(active.metadata.provider(), Some("provider-a"));
    assert_eq!(active.metadata.engine(), Some("engine-a"));
    assert_eq!(active.metadata.source(), Some("direct_http"));
    assert_eq!(active.metadata.method(), Some("POST"));
}

#[tokio::test]
async fn production_order_authenticated_direct_survives_local_registration_and_persistence() {
    let (state, _root) = initialized_state();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let endpoint_id = "71".repeat(32);

    let attribution = state
        .attribute_remote_tunneled_request(request_id, authenticated_direct_metadata(&endpoint_id))
        .expect("remote attribution lease");
    let suppression = state
        .suppress_remote_tunneled_request(request_id)
        .expect("remote suppression lease");
    let mut attachment = state.openai_ingress_attachment(request_id, provisional_local_metadata());

    assert_active_caller(
        &state,
        request_id,
        &endpoint_id,
        Some("192.0.2.71:11204"),
        "remote_quic_http",
    );
    attachment.terminal(TerminalOutcome::Completed);
    drop(suppression);
    drop(attribution);
    state.pump_persistence_for_test().await;

    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("query request")
        .expect("durable request");
    assert_eq!(
        durable.caller_endpoint_id.as_deref(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(durable.caller_addr.as_deref(), Some("192.0.2.71:11204"));
    assert_eq!(
        durable.caller_path_type.as_deref(),
        Some("remote_quic_http")
    );
    assert_eq!(durable.request.model.as_deref(), Some("model-a"));
    assert_eq!(durable.request.route.as_deref(), Some("responses"));
    assert_eq!(durable.request.provider.as_deref(), Some("provider-a"));
    assert_eq!(durable.request.engine.as_deref(), Some("engine-a"));
}

#[tokio::test]
async fn reverse_order_authenticated_relay_atomically_clears_local_address() {
    let (state, _root) = initialized_state();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let endpoint_id = "72".repeat(32);
    let mut attachment = state.openai_ingress_attachment(request_id, provisional_local_metadata());

    let first = state
        .attribute_remote_tunneled_request(request_id, authenticated_relay_metadata(&endpoint_id))
        .expect("first remote attribution lease");
    let second = state
        .attribute_remote_tunneled_request(request_id, authenticated_relay_metadata(&endpoint_id))
        .expect("idempotent remote attribution lease");

    assert_active_caller(&state, request_id, &endpoint_id, None, "relay");
    attachment.terminal(TerminalOutcome::Completed);
    drop(first);
    drop(second);
    state.pump_persistence_for_test().await;

    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("query request")
        .expect("durable request");
    assert_eq!(
        durable.caller_endpoint_id.as_deref(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(durable.caller_addr, None);
    assert_eq!(durable.caller_path_type.as_deref(), Some("relay"));
    assert_eq!(durable.request.model.as_deref(), Some("model-a"));
    assert_eq!(durable.request.route.as_deref(), Some("responses"));
    assert_eq!(durable.request.provider.as_deref(), Some("provider-a"));
    assert_eq!(durable.request.engine.as_deref(), Some("engine-a"));
}

#[tokio::test]
async fn markerless_authenticated_relay_survives_production_order_without_suppression() {
    let (state, _root) = initialized_state();
    let (forwarded, request_id) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec(),
        );
    let request_id = request_id.expect("tunnel-generated request ID");
    let endpoint_id = "73".repeat(32);

    assert_eq!(
        crate::network::openai::request_parse::canonical_request_id_from_header_prefix(&forwarded),
        Some(request_id)
    );

    let attribution = state
        .attribute_remote_tunneled_request(request_id, authenticated_relay_metadata(&endpoint_id))
        .expect("markerless remote attribution lease");
    let mut attachment = state.openai_ingress_attachment(request_id, provisional_local_metadata());

    assert_active_caller(&state, request_id, &endpoint_id, None, "relay");
    attachment.terminal(TerminalOutcome::Completed);
    drop(attribution);
    state.pump_persistence_for_test().await;

    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("query request")
        .expect("durable request");
    assert_eq!(
        durable.caller_endpoint_id.as_deref(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(durable.caller_addr, None);
    assert_eq!(durable.caller_path_type.as_deref(), Some("relay"));
    assert_eq!(durable.request.model.as_deref(), Some("model-a"));
    assert_eq!(durable.request.route.as_deref(), Some("responses"));
    assert_eq!(durable.request.provider.as_deref(), Some("provider-a"));
    assert_eq!(durable.request.engine.as_deref(), Some("engine-a"));
}

#[tokio::test]
async fn no_header_authenticated_direct_survives_active_and_durable_lifecycle() {
    let (state, _root) = initialized_state();
    let (forwarded, request_id) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec(),
        );
    let request_id = request_id.expect("tunnel-generated request ID");
    let endpoint_id = "76".repeat(32);

    assert_eq!(
        crate::network::openai::request_parse::canonical_request_id_from_header_prefix(&forwarded),
        Some(request_id)
    );
    let attribution = state
        .attribute_remote_tunneled_request(request_id, authenticated_direct_metadata(&endpoint_id))
        .expect("markerless remote attribution lease");
    let mut attachment = state.openai_ingress_attachment(request_id, provisional_local_metadata());

    assert_active_caller(
        &state,
        request_id,
        &endpoint_id,
        Some("192.0.2.71:11204"),
        "remote_quic_http",
    );
    attachment.terminal(TerminalOutcome::Completed);
    drop(attribution);
    state.pump_persistence_for_test().await;

    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("query request")
        .expect("durable request");
    assert_eq!(
        durable.caller_endpoint_id.as_deref(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(durable.caller_addr.as_deref(), Some("192.0.2.71:11204"));
    assert_eq!(
        durable.caller_path_type.as_deref(),
        Some("remote_quic_http")
    );
}

#[tokio::test]
async fn endpoint_only_caller_survives_pending_active_terminal_and_durable_lifecycle() {
    let (state, _root) = initialized_state();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let endpoint_id = "79".repeat(32);
    let later_endpoint_id = "7a".repeat(32);

    let endpoint_only = state
        .attribute_remote_tunneled_request(
            request_id,
            RequestSummaryMetadata::default().with_caller_identity(Some(&endpoint_id), None, None),
        )
        .expect("pending endpoint-only attribution");
    let mut attachment = state.openai_ingress_attachment(request_id, provisional_local_metadata());
    let direct = state
        .attribute_remote_tunneled_request(
            request_id,
            authenticated_direct_metadata(&later_endpoint_id),
        )
        .expect("later direct attribution lease");
    let relay = state
        .attribute_remote_tunneled_request(
            request_id,
            authenticated_relay_metadata(&later_endpoint_id),
        )
        .expect("later relay attribution lease");

    let active = state
        .service
        .as_ref()
        .expect("logging service")
        .registry_ref()
        .get_active(&request_id.as_uuid().to_string())
        .expect("active request");
    assert_eq!(
        active.metadata.caller_endpoint_id(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(active.metadata.caller_addr(), None);
    assert_eq!(active.metadata.caller_path_type(), None);
    assert_eq!(active.metadata.route(), Some("responses"));
    assert_eq!(active.metadata.model(), Some("model-a"));
    assert_eq!(active.metadata.provider(), Some("provider-a"));
    assert_eq!(active.metadata.engine(), Some("engine-a"));
    assert_eq!(active.metadata.source(), Some("direct_http"));
    assert_eq!(active.metadata.method(), Some("POST"));

    attachment.terminal(TerminalOutcome::Completed);
    drop(relay);
    drop(direct);
    drop(endpoint_only);
    state.pump_persistence_for_test().await;

    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_id.as_uuid().to_string())
        .expect("query request")
        .expect("durable request");
    assert_eq!(
        durable.caller_endpoint_id.as_deref(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(durable.caller_addr, None);
    assert_eq!(durable.caller_path_type, None);
    assert_eq!(durable.request.model.as_deref(), Some("model-a"));
    assert_eq!(durable.request.route.as_deref(), Some("responses"));
    assert_eq!(durable.request.provider.as_deref(), Some("provider-a"));
    assert_eq!(durable.request.engine.as_deref(), Some("engine-a"));
}
