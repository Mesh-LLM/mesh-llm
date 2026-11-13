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
        CallerScope, ResponseContinuation, ResponsesStream, ResponsesStreamEvent, ResponsesTurn,
        caller_scope_for_http_peer,
    },
};

pub(super) async fn responses(
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
    let turn = Arc::new(ResponsesTurn::prepare(&mut value, continuation)?);
    let normalization = normalize_openai_compat_request("/v1/responses", &mut value)?;
    let request: ChatCompletionRequest = serde_json::from_value(value).map_err(|error| {
        InferenceError::invalid_request(format!("invalid Responses request: {error}"))
    })?;
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let trusted_agent_session = header_session.is_some();
    let mut request = GenerationRequest::from(request);
    let responses_session = normalization
        .agent_session_id
        .map(|id| AgentSessionIdentity::new(id, AgentSessionSource::ResponsesConversation))
        .transpose()?;
    request.set_agent_session(resolve_agent_session(header_session, responses_session)?);
    request.validate()?;
    match normalization.response_adapter {
        ResponseAdapterMode::OpenAiResponsesStream => {
            stream_responses(&state, &context, request, trusted_agent_session, turn).await
        }
        _ => non_streaming_responses(&state, &context, request, trusted_agent_session, &turn).await,
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
    request: GenerationRequest,
    trusted_agent_session: bool,
    turn: Arc<ResponsesTurn>,
) -> Result<Response, InferenceError> {
    let backend_context = request_context(context.request_id, trusted_agent_session, true);
    let cancellation = backend_context.cancellation_token();
    let stream = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        context,
        InferenceBackendOperation::ResponsesStream,
        "responses_stream",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_stream(request, backend_context.clone()),
    )
    .await?;
    let lifecycle =
        state.stream_lifecycle(context.clone(), InferenceBackendOperation::ResponsesStream);
    let exchange_id = backend_context.exchange_id();
    if let Some(exchange_id) = exchange_id.clone() {
        lifecycle.record_exchange_id(exchange_id);
    }
    let stream = observe_backend_stream(stream, lifecycle.clone());
    let machine = Arc::new(Mutex::new(ResponsesStream::new(turn)));
    let prelude = sse_events(lock(&machine).start());
    let body_machine = machine.clone();
    let usage_lifecycle = lifecycle.clone();
    let backend_cancellation = cancellation.clone();
    let body_events = stream.flat_map(move |item| {
        let events = stream_item_events(&body_machine, &usage_lifecycle, item.map(Into::into));
        if lock(&body_machine).is_failed() {
            backend_cancellation.cancel();
        }
        stream::iter(sse_events(events))
    });
    let completion_lifecycle = lifecycle.clone();
    let tail_events = stream::once(async move {
        let mut machine = lock(&machine);
        let already_terminated = machine.is_terminated();
        let mut events = sse_events(machine.finish());
        if machine.is_failed() {
            completion_lifecycle.mark_protocol_failed("Responses stream ended unsuccessfully");
        } else if !already_terminated {
            completion_lifecycle.mark_protocol_complete();
        }
        if !already_terminated {
            events.push(done_event());
        }
        events
    })
    .flat_map(stream::iter);
    let events = stream::iter(prelude).chain(body_events).chain(tail_events);
    let mut response = sse_response(events, cancellation, lifecycle);
    if let Some(exchange_id) = exchange_id {
        response
            .extensions_mut()
            .insert(ExchangeIdExtension(exchange_id));
    }
    Ok(response)
}

fn lock(machine: &Mutex<ResponsesStream>) -> std::sync::MutexGuard<'_, ResponsesStream> {
    machine
        .lock()
        .expect("responses stream state lock poisoned")
}

fn stream_item_events(
    machine: &Mutex<ResponsesStream>,
    usage_lifecycle: &StreamLifecycle,
    item: InferenceResult<ChatCompletionChunk>,
) -> Vec<ResponsesStreamEvent> {
    let mut machine = lock(machine);
    match item {
        Ok(chunk) => {
            if let Some(usage) = chunk.usage.as_ref() {
                usage_lifecycle.capture_usage(usage);
            }
            match serde_json::to_value(&chunk) {
                Ok(chunk) => machine.push_chunk(&chunk),
                Err(error) => machine.fail("server_error", &error.to_string()),
            }
        }
        Err(error) => {
            let body = error.body().error;
            let code = body.code.unwrap_or_else(|| body.error_type.clone());
            machine.fail(&code, &body.message)
        }
    }
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
