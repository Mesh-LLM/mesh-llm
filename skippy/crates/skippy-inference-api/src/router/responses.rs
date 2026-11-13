//! The direct `/v1/responses` handler: one [`ResponsesTurn`] per request,
//! translated onto the shared chat generation boundary, answered as JSON or
//! through the shared [`ResponsesStream`] state machine.
use std::net::SocketAddr;

use axum::extract::ConnectInfo;

use super::*;
use crate::{
    chat::ChatCompletionResponse,
    generation::GenerationRequest,
    responses::{
        CallerScope, PreparedResponsesTurn, ResponseContinuation, ResponsesSessionExecutor,
        ResponsesStreamEvent, ResponsesTurn, caller_scope_for_http_peer, prepare_request,
    },
};

pub(super) async fn responses(
    State(state): State<FrontendState>,
    Extension(context): Extension<InferenceLifecycleContext>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, InferenceError> {
    let Json(value) = json_payload(payload)?;
    let peer = connect_info.map(|Extension(ConnectInfo(addr))| addr.ip());
    let continuation = ResponseContinuation::new(
        state.config.response_store.clone(),
        http_caller_scope(peer, &headers),
    );
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let previous_response_id = value
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let prepared = prepare_request(value, continuation, header_session, previous_response_id)?;
    match prepared.response_adapter() {
        ResponseAdapterMode::OpenAiResponsesStream => {
            stream_responses(&state, &context, prepared).await
        }
        _ => {
            let (turn, request, trusted_agent_session, _) = prepared.into_execution_parts();
            non_streaming_responses(&state, &context, request, trusted_agent_session, &turn).await
        }
    }
}

pub(super) async fn responses_compact(
    State(state): State<FrontendState>,
    Extension(context): Extension<InferenceLifecycleContext>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, InferenceError> {
    let Json(mut value) = json_payload(payload)?;
    let peer = connect_info.map(|Extension(ConnectInfo(addr))| addr.ip());
    let continuation = ResponseContinuation::new(
        state.config.response_store.clone(),
        http_caller_scope(peer, &headers),
    );
    let turn = Arc::new(ResponsesTurn::prepare_compact(&mut value, continuation)?);
    normalize_openai_compat_request("/v1/responses/compact", &mut value)?;
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(|error| {
        InferenceError::invalid_request(format!("invalid compact request: {error}"))
    })?;
    let mut request = GenerationRequest::from(request);
    request.validate()?;
    request.set_agent_session(None);
    non_streaming_responses(&state, &context, request, false, &turn).await
}

/// The caller scope of a direct HTTP request: loopback peers without
/// forwarding headers share the host-local scope; anything else has none.
fn http_caller_scope(peer: Option<std::net::IpAddr>, headers: &HeaderMap) -> Option<CallerScope> {
    let forwarded = crate::responses::FORWARDING_HEADERS
        .iter()
        .any(|name| headers.contains_key(*name));
    caller_scope_for_http_peer(peer, forwarded)
}

async fn stream_responses(
    state: &FrontendState,
    context: &InferenceLifecycleContext,
    prepared: PreparedResponsesTurn,
) -> Result<Response, InferenceError> {
    let executor = ResponsesSessionExecutor::new(
        state.backend.clone(),
        state.config.backend_timeout,
        state.config.lifecycle_observer.clone(),
    );
    let mut execution = executor.start(prepared, context.clone()).await?;
    let exchange_id = execution.exchange_id().map(ToString::to_string);
    let cancellation = execution.cancellation_token();
    let lifecycle = execution.lifecycle();
    let prelude = sse_events(execution.start_events());
    let body_events = stream::unfold(execution, |mut execution| async move {
        execution
            .next_events()
            .await
            .map(|events| (sse_events_with_done(events), execution))
    })
    .flat_map(stream::iter);
    let events = stream::iter(prelude).chain(body_events);
    let mut response = sse_response(events, cancellation, lifecycle);
    if let Some(exchange_id) = exchange_id {
        response
            .extensions_mut()
            .insert(ExchangeIdExtension(exchange_id));
    }
    Ok(response)
}

fn sse_events(events: Vec<ResponsesStreamEvent>) -> Vec<Result<Event, Infallible>> {
    events
        .into_iter()
        .map(|event| {
            Ok(Event::default()
                .event(event.event)
                .json_data(&event.data)
                .unwrap_or_else(|_| Event::default().event(event.event).data("{}")))
        })
        .collect()
}

fn sse_events_with_done(events: Vec<ResponsesStreamEvent>) -> Vec<Result<Event, Infallible>> {
    let terminal = events
        .iter()
        .any(|event| matches!(event.event, "response.completed" | "response.incomplete"));
    let mut framed = sse_events(events);
    if terminal {
        framed.push(done_event());
    }
    framed
}

async fn non_streaming_responses(
    state: &FrontendState,
    context: &InferenceLifecycleContext,
    request: GenerationRequest,
    trusted_agent_session: bool,
    turn: &ResponsesTurn,
) -> Result<Response, InferenceError> {
    let backend_context = request_context(context.request_id, trusted_agent_session, false);
    let response = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        context,
        InferenceBackendOperation::Responses,
        "responses",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_with_context(request, backend_context.clone()),
    )
    .await?;
    let usage = response.usage.clone();
    let capsule_marker = response.capsule_marker.clone();
    let exchange_id = response.exchange_id.clone();
    let chat = serde_json::to_value(ChatCompletionResponse::from(response)).map_err(|error| {
        InferenceError::internal(format!("serialize chat completion response: {error}"))
    })?;
    let translated = turn.try_finish_chat_completion(&chat)?;
    state.response_completed(context, InferenceBackendOperation::Responses, &usage);
    let mut http_response = json_response_with_usage(translated, &usage);
    if let Some(marker) = capsule_marker {
        http_response
            .extensions_mut()
            .insert(CapsuleMarkerExtension(marker));
    }
    if let Some(exchange_id) = exchange_id {
        http_response
            .extensions_mut()
            .insert(ExchangeIdExtension(exchange_id));
    }
    Ok(http_response)
}
