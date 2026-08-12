use std::{
    convert::Infallible,
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Extension, State, rejection::JsonRejection},
    http::{HeaderMap, Method, Request, StatusCode, Uri, header::HeaderName},
    middleware::{self, Next},
    response::{IntoResponse, Response, sse::Event},
    routing::{get, post},
};
use futures_util::{StreamExt, stream};
use mesh_llm_events::logging::events::TokenUsage;
use serde::Serialize;
use serde_json::Value;

use crate::{
    backend::{OpenAiBackend, OpenAiRequestContext, OpenAiResult, SharedBackend},
    chat::{ChatCompletionChunk, ChatCompletionRequest},
    common::{AgentSessionIdentity, AgentSessionSource, Usage},
    completions::CompletionRequest,
    errors::OpenAiError,
    lifecycle::{
        OpenAiBackendOperation, OpenAiFailure, OpenAiFrontendRoute, OpenAiLifecycleContext,
        OpenAiLifecycleEvent, OpenAiLifecycleObserver, OpenAiRejection, OpenAiRequestMethod,
        OpenAiTerminalResult, request_id_from_headers_or_generate, request_id_response_header,
    },
    models::ModelsResponse,
    responses::{
        ResponseAdapterMode, ResponseSseState, chunk_delta_text, normalize_openai_compat_request,
        responses_stream_completed_event_with_sequence, responses_stream_content_part_added_event,
        responses_stream_content_part_done_event, responses_stream_created_event_with_sequence,
        responses_stream_delta_event_with_logprobs_and_sequence,
        responses_stream_output_item_added_event, responses_stream_output_item_done_event,
        responses_stream_text_done_event_with_sequence,
        translate_chat_completion_response_to_responses, usage_to_responses_usage,
    },
    sse::{done_event, json_event},
};

mod stream_lifecycle;
use stream_lifecycle::{StreamLifecycle, StreamingResponse, sse_response};

const AGENT_SESSION_HEADER_ENV: &str = "MESH_AGENT_SESSION_HEADER";

fn parse_agent_session_header(value: &str) -> Option<HeaderName> {
    HeaderName::from_bytes(value.as_bytes()).ok()
}

fn configured_agent_session_header() -> Option<HeaderName> {
    let value = match std::env::var(AGENT_SESSION_HEADER_ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return None,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                env = AGENT_SESSION_HEADER_ENV,
                "ignoring non-UTF-8 trusted agent-session header configuration"
            );
            return None;
        }
    };
    match parse_agent_session_header(&value) {
        Some(header) => Some(header),
        None => {
            tracing::warn!(
                env = AGENT_SESSION_HEADER_ENV,
                value = %value,
                "ignoring invalid trusted agent-session header configuration"
            );
            None
        }
    }
}

pub use crate::lifecycle::RequestId;

#[derive(Clone)]
struct FrontendState {
    backend: SharedBackend,
    config: OpenAiFrontendConfig,
}

impl FrontendState {
    fn observe(&self, event: OpenAiLifecycleEvent) {
        if let Some(observer) = &self.config.lifecycle_observer {
            observer.observe(&event);
        }
    }

    fn stream_lifecycle(&self, context: OpenAiLifecycleContext) -> StreamLifecycle {
        StreamLifecycle::new(self.config.lifecycle_observer.clone(), context)
    }
}

#[derive(Clone)]
pub struct OpenAiFrontendConfig {
    pub max_request_body_bytes: usize,
    pub backend_timeout: Option<Duration>,
    /// Header accepted as stable agent-session identity from the endpoint's
    /// trusted immediate upstream. `None` disables header-derived identity.
    pub agent_session_header: Option<HeaderName>,
    lifecycle_observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
}

impl std::fmt::Debug for OpenAiFrontendConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiFrontendConfig")
            .field("max_request_body_bytes", &self.max_request_body_bytes)
            .field("backend_timeout", &self.backend_timeout)
            .field("agent_session_header", &self.agent_session_header)
            .field("has_lifecycle_observer", &self.lifecycle_observer.is_some())
            .finish()
    }
}

impl OpenAiFrontendConfig {
    pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 4 * 1024 * 1024;
    pub const DEFAULT_BACKEND_TIMEOUT: Duration = Duration::from_secs(300);

    pub fn with_max_request_body_bytes(mut self, max_request_body_bytes: usize) -> Self {
        self.max_request_body_bytes = max_request_body_bytes;
        self
    }

    pub fn with_backend_timeout(mut self, backend_timeout: Duration) -> Self {
        self.backend_timeout = Some(backend_timeout);
        self
    }

    pub fn without_backend_timeout(mut self) -> Self {
        self.backend_timeout = None;
        self
    }

    pub fn with_agent_session_header(mut self, header: HeaderName) -> Self {
        self.agent_session_header = Some(header);
        self
    }

    /// Observe metadata-only lifecycle boundaries for frontend ingress.
    pub fn with_lifecycle_observer(mut self, observer: Arc<dyn OpenAiLifecycleObserver>) -> Self {
        self.lifecycle_observer = Some(observer);
        self
    }
}

impl Default for OpenAiFrontendConfig {
    fn default() -> Self {
        Self {
            max_request_body_bytes: Self::DEFAULT_MAX_REQUEST_BODY_BYTES,
            backend_timeout: Some(Self::DEFAULT_BACKEND_TIMEOUT),
            agent_session_header: configured_agent_session_header(),
            lifecycle_observer: None,
        }
    }
}

pub fn router<B>(backend: Arc<B>) -> Router
where
    B: OpenAiBackend,
{
    router_for(backend)
}

pub fn router_for(backend: Arc<dyn OpenAiBackend>) -> Router {
    router_for_with_config(backend, OpenAiFrontendConfig::default())
}

pub fn router_with_config<B>(backend: Arc<B>, config: OpenAiFrontendConfig) -> Router
where
    B: OpenAiBackend,
{
    router_for_with_config(backend, config)
}

pub fn router_for_with_config(
    backend: Arc<dyn OpenAiBackend>,
    config: OpenAiFrontendConfig,
) -> Router {
    let state = FrontendState { backend, config };
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/responses", post(responses))
        .method_not_allowed_fallback(method_not_allowed)
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            frontend_lifecycle_middleware,
        ))
        .layer(DefaultBodyLimit::max(state.config.max_request_body_bytes))
        .with_state(state)
}

#[derive(Debug, Clone, Copy, Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn ready(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
) -> Result<Json<HealthResponse>, OpenAiError> {
    backend_call(
        &state,
        &context,
        OpenAiBackendOperation::Models,
        "models",
        state.backend.models(),
    )
    .await?;
    Ok(Json(HealthResponse { status: "ready" }))
}

async fn models(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
) -> Result<Json<ModelsResponse>, OpenAiError> {
    let data = backend_call(
        &state,
        &context,
        OpenAiBackendOperation::Models,
        "models",
        state.backend.models(),
    )
    .await?;
    Ok(Json(ModelsResponse {
        object: "list",
        data,
    }))
}

async fn chat_completions(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(mut request) = json_payload(payload)?;
    request.set_agent_session(agent_session_from_header(&state.config, &headers)?);
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let model = request.model.clone();
        let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
        let cancellation = backend_context.cancellation_token();
        let stream = backend_call_with_cancellation(
            &state,
            &context,
            OpenAiBackendOperation::ChatCompletionStream,
            "chat_completion_stream",
            &backend_context,
            state
                .backend
                .chat_completion_stream(request, backend_context.clone()),
        )
        .await?;
        let prelude = stream::once(async move { json_event(&ChatCompletionChunk::role(model)) });
        let lifecycle = state.stream_lifecycle(context);
        let error_lifecycle = lifecycle.clone();
        let usage_lifecycle = lifecycle.clone();
        let events = prelude
            .chain(stream.filter_map(move |item| {
                let error_lifecycle = error_lifecycle.clone();
                let usage_lifecycle = usage_lifecycle.clone();
                async move {
                    match item {
                        Ok(chunk) => {
                            if let Some(usage) = chunk.usage.as_ref() {
                                usage_lifecycle.observe_usage(usage);
                            }
                            (include_usage || chunk.usage.is_none()).then(|| json_event(&chunk))
                        }
                        Err(error) => {
                            error_lifecycle.failed(&error);
                            Some(json_event(&error.body()))
                        }
                    }
                }
            }))
            .chain(stream::once(async { done_event() }));
        Ok(sse_response(events, cancellation, lifecycle))
    } else {
        let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
        let response = backend_call_with_cancellation(
            &state,
            &context,
            OpenAiBackendOperation::ChatCompletion,
            "chat_completion",
            &backend_context,
            state
                .backend
                .chat_completion_with_context(request, backend_context.clone()),
        )
        .await?;
        Ok(json_response_with_usage(response.clone(), &response.usage))
    }
}

async fn responses(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(mut value) = json_payload(payload)?;
    let normalization = normalize_openai_compat_request("/v1/responses", &mut value)?;
    let mut request: ChatCompletionRequest = serde_json::from_value(value).map_err(|error| {
        OpenAiError::invalid_request(format!("invalid Responses request: {error}"))
    })?;
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let responses_session = normalization
        .agent_session_id
        .map(|id| AgentSessionIdentity::new(id, AgentSessionSource::ResponsesConversation))
        .transpose()?;
    request.set_agent_session(resolve_agent_session(header_session, responses_session)?);
    request.validate()?;
    match normalization.response_adapter {
        ResponseAdapterMode::OpenAiResponsesStream => {
            let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
            let cancellation = backend_context.cancellation_token();
            let state_machine = Arc::new(Mutex::new(ResponseSseState::new(request.model.clone())));
            let stream = backend_call_with_cancellation(
                &state,
                &context,
                OpenAiBackendOperation::ResponsesStream,
                "responses_stream",
                &backend_context,
                state
                    .backend
                    .chat_completion_stream(request, backend_context.clone()),
            )
            .await?;
            let body_state = state_machine.clone();
            let lifecycle = state.stream_lifecycle(context);
            let error_lifecycle = lifecycle.clone();
            let usage_lifecycle = lifecycle.clone();
            let body_events = stream.flat_map(move |item| {
                let mut out = Vec::new();
                let mut state_machine = body_state
                    .lock()
                    .expect("responses stream state lock poisoned");
                if state_machine.failed {
                    return stream::iter(out.into_iter().map(Ok::<_, Infallible>));
                }
                match item {
                    Ok(chunk) => {
                        if !state_machine.created_emitted {
                            state_machine.model = chunk.model.clone();
                            let sequence_number = state_machine.next_sequence_number();
                            out.push(
                                Event::default()
                                    .event("response.created")
                                    .json_data(responses_stream_created_event_with_sequence(
                                        &state_machine.model,
                                        state_machine.created_at,
                                        sequence_number,
                                    ))
                                    .unwrap_or_else(|_| Event::default().data("{}")),
                            );
                            state_machine.created_emitted = true;
                        }
                        if let Some(delta) = chunk_delta_text(&chunk) {
                            if !state_machine.output_item_emitted {
                                let sequence_number = state_machine.next_sequence_number();
                                out.push(
                                    Event::default()
                                        .event("response.output_item.added")
                                        .json_data(responses_stream_output_item_added_event(
                                            &state_machine.item_id,
                                            sequence_number,
                                        ))
                                        .unwrap_or_else(|_| Event::default().data("{}")),
                                );
                                let sequence_number = state_machine.next_sequence_number();
                                out.push(
                                    Event::default()
                                        .event("response.content_part.added")
                                        .json_data(responses_stream_content_part_added_event(
                                            &state_machine.item_id,
                                            sequence_number,
                                        ))
                                        .unwrap_or_else(|_| Event::default().data("{}")),
                                );
                                state_machine.output_item_emitted = true;
                            }
                            let logprobs = chunk
                                .choices
                                .first()
                                .and_then(|choice| choice.logprobs.clone());
                            state_machine.output_text.push_str(&delta);
                            let sequence_number = state_machine.next_sequence_number();
                            out.push(
                                Event::default()
                                    .event("response.output_text.delta")
                                    .json_data(
                                        responses_stream_delta_event_with_logprobs_and_sequence(
                                            &state_machine.item_id,
                                            &delta,
                                            logprobs,
                                            sequence_number,
                                        ),
                                    )
                                    .unwrap_or_else(|_| Event::default().data("{}")),
                            );
                        }
                        if let Some(usage) = chunk.usage.as_ref() {
                            state_machine.usage = Some(usage_to_responses_usage(usage));
                            usage_lifecycle.observe_usage(usage);
                        }
                    }
                    Err(error) => {
                        error_lifecycle.failed(&error);
                        state_machine.failed = true;
                        out.push(
                            Event::default()
                                .event("error")
                                .json_data(error.body())
                                .unwrap_or_else(|_| Event::default().data("{}")),
                        );
                    }
                }
                stream::iter(out.into_iter().map(Ok::<_, Infallible>))
            });
            let tail_events = stream::once(async move {
                let mut state_machine = state_machine
                    .lock()
                    .expect("responses stream state lock poisoned");
                let mut out = Vec::new();
                if state_machine.failed {
                    return out;
                }
                if !state_machine.created_emitted {
                    let sequence_number = state_machine.next_sequence_number();
                    out.push(
                        Event::default()
                            .event("response.created")
                            .json_data(responses_stream_created_event_with_sequence(
                                &state_machine.model,
                                state_machine.created_at,
                                sequence_number,
                            ))
                            .unwrap_or_else(|_| Event::default().data("{}")),
                    );
                    state_machine.created_emitted = true;
                }
                if !state_machine.output_item_emitted {
                    let sequence_number = state_machine.next_sequence_number();
                    out.push(
                        Event::default()
                            .event("response.output_item.added")
                            .json_data(responses_stream_output_item_added_event(
                                &state_machine.item_id,
                                sequence_number,
                            ))
                            .unwrap_or_else(|_| Event::default().data("{}")),
                    );
                    let sequence_number = state_machine.next_sequence_number();
                    out.push(
                        Event::default()
                            .event("response.content_part.added")
                            .json_data(responses_stream_content_part_added_event(
                                &state_machine.item_id,
                                sequence_number,
                            ))
                            .unwrap_or_else(|_| Event::default().data("{}")),
                    );
                    state_machine.output_item_emitted = true;
                }
                let sequence_number = state_machine.next_sequence_number();
                out.push(
                    Event::default()
                        .event("response.output_text.done")
                        .json_data(responses_stream_text_done_event_with_sequence(
                            &state_machine.item_id,
                            &state_machine.output_text,
                            sequence_number,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                let sequence_number = state_machine.next_sequence_number();
                out.push(
                    Event::default()
                        .event("response.content_part.done")
                        .json_data(responses_stream_content_part_done_event(
                            &state_machine.item_id,
                            &state_machine.output_text,
                            sequence_number,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                let sequence_number = state_machine.next_sequence_number();
                out.push(
                    Event::default()
                        .event("response.output_item.done")
                        .json_data(responses_stream_output_item_done_event(
                            &state_machine.item_id,
                            &state_machine.output_text,
                            sequence_number,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                let sequence_number = state_machine.next_sequence_number();
                out.push(
                    Event::default()
                        .event("response.completed")
                        .json_data(responses_stream_completed_event_with_sequence(
                            &state_machine.response_id,
                            state_machine.created_at,
                            &state_machine.model,
                            &state_machine.item_id,
                            &state_machine.output_text,
                            state_machine.usage.clone(),
                            sequence_number,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
                out
            })
            .flat_map(|out| stream::iter(out.into_iter().map(Ok::<_, Infallible>)));
            let events = body_events
                .chain(tail_events)
                .chain(stream::once(async { done_event() }));
            Ok(sse_response(events, cancellation, lifecycle))
        }
        _ => {
            let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
            let response = backend_call_with_cancellation(
                &state,
                &context,
                OpenAiBackendOperation::Responses,
                "responses",
                &backend_context,
                state
                    .backend
                    .chat_completion_with_context(request, backend_context.clone()),
            )
            .await?;
            let translated = translate_chat_completion_response_to_responses(&response)?;
            Ok(json_response_with_usage(translated, &response.usage))
        }
    }
}

async fn completions(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(mut request) = json_payload(payload)?;
    request.set_agent_session(agent_session_from_header(&state.config, &headers)?);
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
        let cancellation = backend_context.cancellation_token();
        let stream = backend_call_with_cancellation(
            &state,
            &context,
            OpenAiBackendOperation::CompletionStream,
            "completion_stream",
            &backend_context,
            state
                .backend
                .completion_stream(request, backend_context.clone()),
        )
        .await?;
        let lifecycle = state.stream_lifecycle(context);
        let error_lifecycle = lifecycle.clone();
        let usage_lifecycle = lifecycle.clone();
        let events = stream
            .filter_map(move |item| {
                let error_lifecycle = error_lifecycle.clone();
                let usage_lifecycle = usage_lifecycle.clone();
                async move {
                    match item {
                        Ok(chunk) => {
                            if let Some(usage) = chunk.usage.as_ref() {
                                usage_lifecycle.observe_usage(usage);
                            }
                            (include_usage || chunk.usage.is_none()).then(|| json_event(&chunk))
                        }
                        Err(error) => {
                            error_lifecycle.failed(&error);
                            Some(json_event(&error.body()))
                        }
                    }
                }
            })
            .chain(stream::once(async { done_event() }));
        Ok(sse_response(events, cancellation, lifecycle))
    } else {
        let backend_context = OpenAiRequestContext::with_request_id(context.request_id);
        let response = backend_call_with_cancellation(
            &state,
            &context,
            OpenAiBackendOperation::Completion,
            "completion",
            &backend_context,
            state
                .backend
                .completion_with_context(request, backend_context.clone()),
        )
        .await?;
        Ok(json_response_with_usage(response.clone(), &response.usage))
    }
}

#[derive(Clone, Copy)]
struct TerminalUsage(TokenUsage);

fn authoritative_usage(usage: &Usage) -> Option<TokenUsage> {
    TokenUsage::from_counts(
        Some(u64::from(usage.prompt_tokens)),
        Some(u64::from(usage.completion_tokens)),
        Some(u64::from(usage.total_tokens)),
    )
}

fn json_response_with_usage<T: Serialize>(value: T, usage: &Usage) -> Response {
    let mut response = Json(value).into_response();
    if let Some(usage) = authoritative_usage(usage) {
        response.extensions_mut().insert(TerminalUsage(usage));
    }
    response
}

fn agent_session_from_header(
    config: &OpenAiFrontendConfig,
    headers: &HeaderMap,
) -> OpenAiResult<Option<AgentSessionIdentity>> {
    let Some(name) = config.agent_session_header.as_ref() else {
        return Ok(None);
    };
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| {
        OpenAiError::invalid_request("configured agent-session header is not valid UTF-8")
    })?;
    AgentSessionIdentity::new(
        value,
        AgentSessionSource::TrustedHeader(name.as_str().to_owned()),
    )
    .map(Some)
}

fn resolve_agent_session(
    header: Option<AgentSessionIdentity>,
    protocol: Option<AgentSessionIdentity>,
) -> OpenAiResult<Option<AgentSessionIdentity>> {
    match (header, protocol) {
        (Some(header), Some(protocol)) if header.id() != protocol.id() => {
            Err(OpenAiError::invalid_request(
                "trusted agent-session header conflicts with Responses conversation identity",
            ))
        }
        (Some(header), _) => Ok(Some(header)),
        (None, protocol) => Ok(protocol),
    }
}

async fn backend_call<T, F>(
    state: &FrontendState,
    context: &OpenAiLifecycleContext,
    backend_operation: OpenAiBackendOperation,
    operation_name: &'static str,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
{
    state.observe(OpenAiLifecycleEvent::BackendDispatched {
        context: context.clone(),
        operation: backend_operation,
    });
    match state.config.backend_timeout {
        Some(timeout) => tokio::time::timeout(timeout, future).await.map_err(|_| {
            OpenAiError::timeout(format!(
                "{operation_name} timed out after {} ms",
                timeout.as_millis()
            ))
        })?,
        None => future.await,
    }
}

struct CancelOnDrop {
    context: OpenAiRequestContext,
    armed: bool,
}

impl CancelOnDrop {
    fn new(context: &OpenAiRequestContext) -> Self {
        Self {
            context: context.clone(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.context.cancel();
        }
    }
}

async fn backend_call_with_cancellation<T, F>(
    state: &FrontendState,
    lifecycle_context: &OpenAiLifecycleContext,
    backend_operation: OpenAiBackendOperation,
    operation_name: &'static str,
    request_context: &OpenAiRequestContext,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
{
    state.observe(OpenAiLifecycleEvent::BackendDispatched {
        context: lifecycle_context.clone(),
        operation: backend_operation,
    });
    let mut cancel_on_drop = CancelOnDrop::new(request_context);
    let result = match state.config.backend_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, future).await {
            Ok(result) => result,
            Err(_) => {
                request_context.cancel();
                return Err(OpenAiError::timeout(format!(
                    "{operation_name} timed out after {} ms",
                    timeout.as_millis()
                )));
            }
        },
        None => future.await,
    };
    cancel_on_drop.disarm();
    result
}

fn json_payload<T>(payload: Result<Json<T>, JsonRejection>) -> Result<Json<T>, OpenAiError> {
    payload.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            return OpenAiError::payload_too_large(format!("request body too large: {rejection}"));
        }
        OpenAiError::invalid_request(format!("invalid JSON request body: {rejection}"))
    })
}

async fn not_found(uri: Uri) -> OpenAiError {
    OpenAiError::route_not_found(uri)
}

async fn method_not_allowed(method: Method) -> OpenAiError {
    OpenAiError::method_not_allowed(method)
}

async fn frontend_lifecycle_middleware(
    State(state): State<FrontendState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request_id_from_headers_or_generate(request.headers());
    let method = request.method().clone();
    let uri = request.uri().clone();
    let context =
        OpenAiLifecycleContext::new(request_id, lifecycle_method(&method), lifecycle_route(&uri));
    request.extensions_mut().insert(request_id);
    request.extensions_mut().insert(context.clone());
    state.observe(OpenAiLifecycleEvent::Admitted {
        context: context.clone(),
    });

    let mut response = next.run(request).await;
    let (header_name, header_value) = request_id_response_header(&request_id);
    response.headers_mut().insert(header_name, header_value);
    if response.extensions().get::<StreamingResponse>().is_none() {
        let usage = response
            .extensions()
            .get::<TerminalUsage>()
            .map(|usage| usage.0);
        observe_non_stream_terminal(&state, context.clone(), response.status(), usage);
    }
    tracing::info!(
        request_id = %request_id.as_ref(),
        method = %method,
        uri = %uri,
        status = %response.status(),
        "openai frontend request"
    );
    response
}

fn lifecycle_method(method: &Method) -> OpenAiRequestMethod {
    match *method {
        Method::GET => OpenAiRequestMethod::Get,
        Method::POST => OpenAiRequestMethod::Post,
        _ => OpenAiRequestMethod::Other,
    }
}

fn lifecycle_route(uri: &Uri) -> OpenAiFrontendRoute {
    match uri.path() {
        "/health" => OpenAiFrontendRoute::Health,
        "/healthz" => OpenAiFrontendRoute::Healthz,
        "/readyz" => OpenAiFrontendRoute::Readyz,
        "/v1/models" => OpenAiFrontendRoute::Models,
        "/v1/chat/completions" => OpenAiFrontendRoute::ChatCompletions,
        "/v1/completions" => OpenAiFrontendRoute::Completions,
        "/v1/responses" => OpenAiFrontendRoute::Responses,
        _ => OpenAiFrontendRoute::Unknown,
    }
}

fn observe_non_stream_terminal(
    state: &FrontendState,
    context: OpenAiLifecycleContext,
    status: StatusCode,
    usage: Option<TokenUsage>,
) {
    if status.is_client_error() {
        state.observe(OpenAiLifecycleEvent::Rejected {
            context,
            status_code: status.as_u16(),
            rejection: rejection_for_status(status),
        });
        return;
    }

    let result = if status.is_server_error() {
        OpenAiTerminalResult::Failed {
            status_code: status.as_u16(),
            failure: failure_for_status(status),
        }
    } else if let Some(usage) = usage {
        OpenAiTerminalResult::CompletedWithUsage {
            status_code: status.as_u16(),
            usage,
        }
    } else {
        OpenAiTerminalResult::Completed {
            status_code: status.as_u16(),
        }
    };
    state.observe(OpenAiLifecycleEvent::NonStreamTerminal { context, result });
}

fn rejection_for_status(status: StatusCode) -> OpenAiRejection {
    match status {
        StatusCode::PAYLOAD_TOO_LARGE => OpenAiRejection::PayloadTooLarge,
        StatusCode::METHOD_NOT_ALLOWED => OpenAiRejection::MethodNotAllowed,
        StatusCode::NOT_FOUND => OpenAiRejection::NotFound,
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => OpenAiRejection::AdmissionDenied,
        _ => OpenAiRejection::InvalidRequest,
    }
}

fn failure_for_status(status: StatusCode) -> OpenAiFailure {
    match status {
        StatusCode::GATEWAY_TIMEOUT => OpenAiFailure::Timeout,
        StatusCode::INTERNAL_SERVER_ERROR => OpenAiFailure::Internal,
        _ => OpenAiFailure::Backend,
    }
}

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
