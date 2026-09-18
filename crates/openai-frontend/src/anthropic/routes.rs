//! Axum handlers for the Anthropic Messages surface.
//!
//! Mirrors the `responses` route shape in `crate::router`: same backend
//! dispatch, lifecycle observation, timeout, and SSE plumbing — only the
//! wire protocol differs.

use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::sse::Event,
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use crate::anthropic::protocol::{AnthropicError, AnthropicMessagesStreamEvent};
use crate::anthropic::translate::{
    MessagesStreamAssembler, message_start_event, messages_request_to_chat_request,
    messages_response_from_chat_response, translate_stream_error_body,
};
use crate::backend_lifecycle::call_backend_with_context;
use crate::errors::OpenAiError;
use crate::lifecycle::{OpenAiBackendOperation, OpenAiLifecycleContext, RequestId};
use crate::router::{FrontendState, agent_session_from_header, json_payload, request_context};
use crate::stream_lifecycle::{observe_backend_stream, sse_response};

/// Renders backend/lifecycle failures in the Anthropic error envelope.
pub(crate) struct AnthropicRejection(OpenAiError);

impl From<OpenAiError> for AnthropicRejection {
    fn from(error: OpenAiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AnthropicRejection {
    fn into_response(self) -> Response {
        let status = self.0.status();
        let kind = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "authentication_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
            StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            _ => "api_error",
        };
        let body = AnthropicError::new(kind, self.0.message().to_string());
        (status, Json(body)).into_response()
    }
}

/// `POST /v1/messages` — non-streaming JSON or Anthropic SSE stream.
pub(crate) async fn messages(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<
        Json<crate::anthropic::protocol::AnthropicMessagesRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Result<Response, AnthropicRejection> {
    let Json(request) = json_payload(payload)?;
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let trusted_agent_session = header_session.is_some();
    let mut chat_request = messages_request_to_chat_request(request)?;
    chat_request.set_agent_session(header_session);
    chat_request.validate()?;

    if chat_request.stream {
        streaming_messages(&state, context, chat_request, trusted_agent_session).await
    } else {
        non_streaming_messages(&state, context, chat_request, trusted_agent_session).await
    }
}

async fn non_streaming_messages(
    state: &FrontendState,
    context: OpenAiLifecycleContext,
    request: crate::chat::ChatCompletionRequest,
    trusted_agent_session: bool,
) -> Result<Response, AnthropicRejection> {
    let backend_context = request_context(context.request_id, trusted_agent_session, false);
    let response = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        &context,
        OpenAiBackendOperation::Messages,
        "messages",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_with_context(request, backend_context.clone()),
    )
    .await?;
    state.response_completed(&context, OpenAiBackendOperation::Messages, &response.usage);
    let translated = messages_response_from_chat_response(&response)?;
    Ok(Json(translated).into_response())
}

async fn streaming_messages(
    state: &FrontendState,
    context: OpenAiLifecycleContext,
    request: crate::chat::ChatCompletionRequest,
    trusted_agent_session: bool,
) -> Result<Response, AnthropicRejection> {
    let model = request.model.clone();
    let request_id: RequestId = context.request_id;
    let backend_context = request_context(request_id, trusted_agent_session, true);
    let cancellation = backend_context.cancellation_token();
    let stream = call_backend_with_context(
        state.config.lifecycle_observer.clone(),
        &context,
        OpenAiBackendOperation::MessagesStream,
        "messages_stream",
        state.config.backend_timeout,
        &backend_context,
        state
            .backend
            .chat_completion_stream(request, backend_context.clone()),
    )
    .await?;
    let lifecycle = state.stream_lifecycle(context, OpenAiBackendOperation::MessagesStream);
    let stream = observe_backend_stream(stream, lifecycle.clone());

    let prelude = stream::once(async move {
        Ok::<_, std::convert::Infallible>(anthropic_event(&message_start_event(
            request_id.as_uuid().to_string().as_str(),
            &model,
        )))
    });

    let assembler = Arc::new(Mutex::new(MessagesStreamAssembler::new()));
    let assembler_lifecycle = lifecycle.clone();
    let scan_assembler = Arc::clone(&assembler);
    let body_events = stream.scan((), move |(): &mut (), item| {
        let lifecycle = assembler_lifecycle.clone();
        let assembler = Arc::clone(&scan_assembler);
        let wire_events = match item {
            Ok(chunk) => {
                if let Some(usage) = chunk.usage.as_ref() {
                    lifecycle.capture_usage(usage);
                }
                let events = match assembler.lock() {
                    Ok(mut assembler) => {
                        let mut events = assembler.absorb(&chunk);
                        if let Some(finish) = chunk
                            .choices
                            .first()
                            .and_then(|choice| choice.finish_reason)
                        {
                            events.extend(assembler.finish(Some(finish)));
                        }
                        events
                    }
                    Err(_) => return std::future::ready(None),
                };
                events
                    .iter()
                    .map(|event| Ok::<_, std::convert::Infallible>(anthropic_event(event)))
                    .collect::<Vec<_>>()
            }
            Err(error) => {
                let body = serde_json::to_value(error.body()).unwrap_or(json!({}));
                let error_event = translate_stream_error_body(&body);
                vec![Ok::<_, std::convert::Infallible>(anthropic_event(
                    &error_event,
                ))]
            }
        };
        std::future::ready(Some(stream::iter(wire_events)))
    });

    let completion_lifecycle = lifecycle.clone();
    // The Anthropic protocol has no `[DONE]` sentinel: `message_stop` closes
    // the stream. The scan above only closes it when the backend sends a
    // `finish_reason`, so an epilogue guarantees the terminator for backends
    // that simply end their chunk stream (and no-ops otherwise — `finish` is
    // idempotent). After a mid-stream `error` event the protocol is already
    // broken, so no closing sequence is attempted.
    let epilogue_assembler = Arc::clone(&assembler);
    let events = prelude
        .chain(body_events.flatten())
        .chain(
            stream::once(async move {
                let tail = epilogue_assembler
                    .lock()
                    .map(|mut assembler| assembler.finish(None))
                    .unwrap_or_default();
                completion_lifecycle.mark_protocol_complete();
                stream::iter(
                    tail.iter()
                        .map(|event| Ok::<_, std::convert::Infallible>(anthropic_event(event)))
                        .collect::<Vec<_>>(),
                )
            })
            .flatten(),
        )
        .chain(stream::once(async {
            // Empty comment frame as the Sse terminator.
            Ok::<_, std::convert::Infallible>(Event::default().comment("stream complete"))
        }));
    Ok(sse_response(events, cancellation, lifecycle))
}

fn anthropic_event(event: &AnthropicMessagesStreamEvent) -> Event {
    Event::default()
        .event(event.event_name())
        .json_data(event)
        .unwrap_or_else(|_| {
            Event::default().data(
                r#"{"type":"error","error":{"type":"api_error","message":"failed to serialize SSE event"}}"#,
            )
        })
}

impl AnthropicMessagesStreamEvent {
    /// The Anthropic SSE `event:` name. message_start/content_block_*/
    /// message_delta/message_stop/error per the streaming document.
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop { .. } => "message_stop",
            Self::Ping { .. } => "ping",
            Self::Error { .. } => "error",
        }
    }
}

/// `POST /v1/messages/count_tokens`.
///
/// The frontend has no tokenizer handle, so the count is a documented
/// chars/4 estimate over the serialized prompt; the response shape matches
/// the Anthropic contract (`{"input_tokens": N}`).
pub(crate) async fn messages_count_tokens(
    State(state): State<FrontendState>,
    Extension(context): Extension<OpenAiLifecycleContext>,
    headers: HeaderMap,
    payload: Result<
        Json<crate::anthropic::protocol::AnthropicMessagesRequest>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Result<Json<Value>, AnthropicRejection> {
    let _ = (&state, &context);
    let Json(request) = json_payload(payload)?;
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let _ = header_session;
    let chat_request = messages_request_to_chat_request(request)?;
    let mut serialized = String::new();
    for message in &chat_request.messages {
        if let Some(text) = message
            .content
            .as_ref()
            .and_then(crate::message_content_to_text)
        {
            serialized.push_str(&text);
            serialized.push('\n');
        }
        if let Some(tool_calls) = message.extra.get("tool_calls").and_then(Value::as_array) {
            for call in tool_calls {
                if let Some(name) = call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                {
                    serialized.push_str(name);
                    serialized.push(' ');
                }
            }
        }
    }
    if let Some(tools) = chat_request.tools.as_ref() {
        serialized.push('\n');
        serialized.push_str(&tools.to_string());
    }
    Ok(Json(
        json!({ "input_tokens": estimate_tokens(&serialized) }),
    ))
}

/// Conservative chars/4 estimate; matches the OpenAI-surface budget heuristic.
fn estimate_tokens(text: &str) -> u32 {
    u32::try_from(text.len().div_ceil(4)).unwrap_or(u32::MAX)
}
