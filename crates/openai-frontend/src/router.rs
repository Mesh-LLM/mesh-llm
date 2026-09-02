use std::{
    convert::Infallible,
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
    backend_lifecycle::{call_backend, call_backend_with_context},
    chat::{ChatCompletionChunk, ChatCompletionRequest},
    common::{AgentSessionIdentity, AgentSessionSource, Usage},
    completions::CompletionRequest,
    errors::OpenAiError,
    lifecycle::{
        OpenAiBackendOperation, OpenAiFrontendRoute, OpenAiLifecycleContext, OpenAiLifecycleEvent,
        OpenAiLifecycleObserver, OpenAiRequestMethod, OpenAiUsage,
        request_id_from_headers_or_generate, request_id_response_header,
    },
    models::ModelsResponse,
    request_lifecycle::RequestLifecycle,
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
    stream_lifecycle::{
        StreamLifecycle, is_streaming_response, observe_backend_stream, sse_response,
    },
};

const AGENT_SESSION_HEADER_ENV: &str = "MESH_AGENT_SESSION_HEADER";
const BACKEND_TIMEOUT_SECS_ENV: &str = "MESH_OPENAI_BACKEND_TIMEOUT_SECS";

/// Backend timeout override, in whole seconds. `0` disables the timeout.
///
/// Returning `None` means "no override configured", which leaves
/// [`OpenAiFrontendConfig::DEFAULT_BACKEND_TIMEOUT`] in place. Returning
/// `Some(None)` means the operator explicitly disabled the timeout.
fn configured_backend_timeout() -> Option<Option<Duration>> {
    let value = match std::env::var(BACKEND_TIMEOUT_SECS_ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return None,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                env = BACKEND_TIMEOUT_SECS_ENV,
                "ignoring non-UTF-8 backend timeout configuration"
            );
            return None;
        }
    };
    match parse_backend_timeout_secs(&value) {
        Some(timeout) => Some(timeout),
        None => {
            tracing::warn!(
                env = BACKEND_TIMEOUT_SECS_ENV,
                value = %value,
                "ignoring invalid backend timeout configuration; expected whole seconds"
            );
            None
        }
    }
}

fn parse_backend_timeout_secs(value: &str) -> Option<Option<Duration>> {
    match value.trim().parse::<u64>() {
        Ok(0) => Some(None),
        Ok(secs) => Some(Some(Duration::from_secs(secs))),
        Err(_) => None,
    }
}

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

    fn stream_lifecycle(
        &self,
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
    ) -> StreamLifecycle {
        StreamLifecycle::new(self.config.lifecycle_observer.clone(), context, operation)
    }

    fn response_completed(
        &self,
        context: &OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
        usage: &crate::Usage,
    ) {
        self.observe(OpenAiLifecycleEvent::ResponseCompleted {
            context: context.clone(),
            operation,
            usage: OpenAiUsage::from(usage),
        });
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
    /// Safety net for a wedged backend, not a latency budget.
    ///
    /// A cold prefill of a large prompt on a single local machine is measured
    /// in minutes: a 60k-token prompt takes ~250s on an M5 Max. A tighter
    /// budget here makes the frontend give up while the runtime is still
    /// legitimately working. This matches the proxy's own local first-byte
    /// safety net so neither layer aborts a healthy long prefill.
    pub const DEFAULT_BACKEND_TIMEOUT: Duration = Duration::from_secs(600);

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
            backend_timeout: configured_backend_timeout()
                .unwrap_or(Some(Self::DEFAULT_BACKEND_TIMEOUT)),
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
    call_backend(
        state.config.lifecycle_observer.clone(),
        &context,
        OpenAiBackendOperation::Models,
        "models",
        state.config.backend_timeout,
        state.backend.models(),
    )
    .await?;
    Ok(Json(HealthResponse { status: "ready" }))
}

async fn models(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
) -> Result<Json<ModelsResponse>, OpenAiError> {
    let data = call_backend(
        state.config.lifecycle_observer.clone(),
        &context,
        OpenAiBackendOperation::Models,
        "models",
        state.config.backend_timeout,
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
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let trusted_agent_session = header_session.is_some();
    request.set_agent_session(header_session);
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let model = request.model.clone();
        let backend_context = request_context(context.request_id, trusted_agent_session, true);
        let cancellation = backend_context.cancellation_token();
        let stream = call_backend_with_context(
            state.config.lifecycle_observer.clone(),
            &context,
            OpenAiBackendOperation::ChatCompletionStream,
            "chat_completion_stream",
            state.config.backend_timeout,
            &backend_context,
            state
                .backend
                .chat_completion_stream(request, backend_context.clone()),
        )
        .await?;
        let lifecycle =
            state.stream_lifecycle(context, OpenAiBackendOperation::ChatCompletionStream);
        let stream = observe_backend_stream(stream, lifecycle.clone());
        let prelude = stream::once(async move { json_event(&ChatCompletionChunk::role(model)) });
        let usage_lifecycle = lifecycle.clone();
        let completion_lifecycle = lifecycle.clone();
        let events = prelude
            .chain(stream.filter_map(move |item| {
                let usage_lifecycle = usage_lifecycle.clone();
                async move {
                    match item {
                        Ok(chunk) => {
                            if let Some(usage) = chunk.usage.as_ref() {
                                usage_lifecycle.capture_usage(usage);
                            }
                            if !include_usage && chunk.usage.is_some() {
                                None
                            } else {
                                Some(json_event(&chunk))
                            }
                        }
                        Err(error) => Some(json_event(&error.body())),
                    }
                }
            }))
            .chain(stream::once(async move {
                completion_lifecycle.mark_protocol_complete();
                done_event()
            }));
        Ok(sse_response(events, cancellation, lifecycle))
    } else {
        let backend_context = request_context(context.request_id, trusted_agent_session, false);
        let response = call_backend_with_context(
            state.config.lifecycle_observer.clone(),
            &context,
            OpenAiBackendOperation::ChatCompletion,
            "chat_completion",
            state.config.backend_timeout,
            &backend_context,
            state
                .backend
                .chat_completion_with_context(request, backend_context.clone()),
        )
        .await?;
        state.response_completed(
            &context,
            OpenAiBackendOperation::ChatCompletion,
            &response.usage,
        );
        let usage = response.usage.clone();
        Ok(json_response_with_usage(response, &usage))
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
    let trusted_agent_session = header_session.is_some();
    let responses_session = normalization
        .agent_session_id
        .map(|id| AgentSessionIdentity::new(id, AgentSessionSource::ResponsesConversation))
        .transpose()?;
    request.set_agent_session(resolve_agent_session(header_session, responses_session)?);
    request.validate()?;
    match normalization.response_adapter {
        ResponseAdapterMode::OpenAiResponsesStream => {
            stream_responses(&state, &context, request, trusted_agent_session).await
        }
        _ => non_streaming_responses(&state, &context, request, trusted_agent_session).await,
    }
}

async fn stream_responses(
    state: &FrontendState,
    context: &OpenAiLifecycleContext,
    request: ChatCompletionRequest,
    trusted_agent_session: bool,
) -> Result<Response, OpenAiError> {
    let include_usage = request.include_usage();
    let backend_context = request_context(context.request_id, trusted_agent_session, true);
    let cancellation = backend_context.cancellation_token();
    let state_machine = Arc::new(Mutex::new(ResponseSseState::new(request.model.clone())));
    let stream = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        context,
        OpenAiBackendOperation::ResponsesStream,
        "responses_stream",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_stream(request, backend_context.clone()),
    )
    .await?;
    let lifecycle =
        state.stream_lifecycle(context.clone(), OpenAiBackendOperation::ResponsesStream);
    let stream = observe_backend_stream(stream, lifecycle.clone());
    let body_state = state_machine.clone();
    let usage_lifecycle = lifecycle.clone();
    let body_events = stream.flat_map(move |item| {
        let out = responses_stream_body_events(&body_state, &usage_lifecycle, item);
        stream::iter(out.into_iter().map(Ok::<_, Infallible>))
    });
    let tail_events =
        stream::once(async move { responses_stream_tail_events(&state_machine, include_usage) })
            .flat_map(|out| stream::iter(out.into_iter().map(Ok::<_, Infallible>)));
    let completion_lifecycle = lifecycle.clone();
    let events = body_events
        .chain(tail_events)
        .chain(stream::once(async move {
            completion_lifecycle.mark_protocol_complete();
            done_event()
        }));
    Ok(sse_response(events, cancellation, lifecycle))
}

async fn non_streaming_responses(
    state: &FrontendState,
    context: &OpenAiLifecycleContext,
    request: ChatCompletionRequest,
    trusted_agent_session: bool,
) -> Result<Response, OpenAiError> {
    let backend_context = request_context(context.request_id, trusted_agent_session, false);
    let response = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        context,
        OpenAiBackendOperation::Responses,
        "responses",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_with_context(request, backend_context.clone()),
    )
    .await?;
    state.response_completed(context, OpenAiBackendOperation::Responses, &response.usage);
    let usage = response.usage.clone();
    let translated = translate_chat_completion_response_to_responses(&response)?;
    Ok(json_response_with_usage(translated, &usage))
}

fn responses_stream_body_events(
    state_machine: &Mutex<ResponseSseState>,
    usage_lifecycle: &StreamLifecycle,
    item: OpenAiResult<ChatCompletionChunk>,
) -> Vec<Event> {
    let mut state_machine = state_machine
        .lock()
        .expect("responses stream state lock poisoned");
    if state_machine.failed {
        return Vec::new();
    }
    let mut out = Vec::new();
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
                        .json_data(responses_stream_delta_event_with_logprobs_and_sequence(
                            &state_machine.item_id,
                            &delta,
                            logprobs,
                            sequence_number,
                        ))
                        .unwrap_or_else(|_| Event::default().data("{}")),
                );
            }
            if let Some(usage) = chunk.usage.as_ref() {
                usage_lifecycle.capture_usage(usage);
                state_machine.usage = Some(usage_to_responses_usage(usage));
            }
        }
        Err(error) => {
            state_machine.failed = true;
            out.push(
                Event::default()
                    .event("error")
                    .json_data(error.body())
                    .unwrap_or_else(|_| Event::default().data("{}")),
            );
        }
    }
    out
}

fn responses_stream_tail_events(
    state_machine: &Mutex<ResponseSseState>,
    include_usage: bool,
) -> Vec<Event> {
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
                if include_usage {
                    state_machine.usage.clone()
                } else {
                    None
                },
                sequence_number,
            ))
            .unwrap_or_else(|_| Event::default().data("{}")),
    );
    out
}

async fn completions(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(mut request) = json_payload(payload)?;
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let trusted_agent_session = header_session.is_some();
    request.set_agent_session(header_session);
    request.validate()?;
    if request.stream {
        let include_usage = request.include_usage();
        let backend_context = request_context(context.request_id, trusted_agent_session, true);
        let cancellation = backend_context.cancellation_token();
        let stream = call_backend_with_context(
            state.config.lifecycle_observer.clone(),
            &context,
            OpenAiBackendOperation::CompletionStream,
            "completion_stream",
            state.config.backend_timeout,
            &backend_context,
            state
                .backend
                .completion_stream(request, backend_context.clone()),
        )
        .await?;
        let lifecycle = state.stream_lifecycle(context, OpenAiBackendOperation::CompletionStream);
        let stream = observe_backend_stream(stream, lifecycle.clone());
        let usage_lifecycle = lifecycle.clone();
        let completion_lifecycle = lifecycle.clone();
        let events = stream
            .filter_map(move |item| {
                let usage_lifecycle = usage_lifecycle.clone();
                async move {
                    match item {
                        Ok(chunk) => {
                            if let Some(usage) = chunk.usage.as_ref() {
                                usage_lifecycle.capture_usage(usage);
                            }
                            if !include_usage && chunk.usage.is_some() {
                                None
                            } else {
                                Some(json_event(&chunk))
                            }
                        }
                        Err(error) => Some(json_event(&error.body())),
                    }
                }
            })
            .chain(stream::once(async move {
                completion_lifecycle.mark_protocol_complete();
                done_event()
            }));
        Ok(sse_response(events, cancellation, lifecycle))
    } else {
        let backend_context = request_context(context.request_id, trusted_agent_session, false);
        let response = call_backend_with_context(
            state.config.lifecycle_observer.clone(),
            &context,
            OpenAiBackendOperation::Completion,
            "completion",
            state.config.backend_timeout,
            &backend_context,
            state
                .backend
                .completion_with_context(request, backend_context.clone()),
        )
        .await?;
        state.response_completed(
            &context,
            OpenAiBackendOperation::Completion,
            &response.usage,
        );
        let usage = response.usage.clone();
        Ok(json_response_with_usage(response, &usage))
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
    .map(|authoritative| {
        authoritative.with_cached_prompt_tokens(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| u64::from(details.cached_tokens)),
        )
    })
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

fn request_context(
    request_id: RequestId,
    trusted_agent_session: bool,
    observe_stream_usage: bool,
) -> OpenAiRequestContext {
    let mut context = OpenAiRequestContext::with_request_id(request_id);
    if trusted_agent_session {
        context = context.with_trusted_agent_session();
    }
    if observe_stream_usage {
        context = context.with_stream_usage_observation();
    }
    context
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
    let mut lifecycle =
        RequestLifecycle::admit(state.config.lifecycle_observer.clone(), context.clone());

    let mut response = next.run(request).await;
    let (header_name, header_value) = request_id_response_header(&request_id);
    response.headers_mut().insert(header_name, header_value);
    if is_streaming_response(&response) {
        lifecycle.transfer_to_stream();
    } else {
        let usage = response
            .extensions()
            .get::<TerminalUsage>()
            .map(|usage| usage.0);
        lifecycle.finish_with_usage(response.status(), usage);
    }
    tracing::info!(
        request_id = %request_id.as_ref(),
        method = ?context.method,
        route = ?context.route,
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

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
