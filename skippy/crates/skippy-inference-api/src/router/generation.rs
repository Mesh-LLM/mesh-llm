//! The OpenAI Chat adapter onto the protocol-neutral generation boundary.
use super::*;
use crate::{chat::ChatCompletionResponse, generation::GenerationRequest};

pub(super) async fn chat_completions(
    State(state): State<FrontendState>,
    Extension(context): Extension<InferenceLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, InferenceError> {
    let Json(request) = json_payload(payload)?;
    let mut request = GenerationRequest::from(request);
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
            InferenceBackendOperation::ChatCompletionStream,
            "chat_completion_stream",
            state.config.backend_timeout,
            &backend_context,
            state
                .backend
                .chat_completion_stream(request, backend_context.clone()),
        )
        .await?;
        let lifecycle =
            state.stream_lifecycle(context, InferenceBackendOperation::ChatCompletionStream);
        let exchange_id = backend_context.exchange_id();
        if let Some(exchange_id) = exchange_id.clone() {
            lifecycle.record_exchange_id(exchange_id);
        }
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
                                Some(json_event(&ChatCompletionChunk::from(chunk)))
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
        let mut response = sse_response(events, cancellation, lifecycle);
        if let Some(exchange_id) = exchange_id {
            response
                .extensions_mut()
                .insert(ExchangeIdExtension(exchange_id));
        }
        Ok(response)
    } else {
        let backend_context = request_context(context.request_id, trusted_agent_session, false);
        let response = call_backend_with_context(
            state.config.lifecycle_observer.clone(),
            &context,
            InferenceBackendOperation::ChatCompletion,
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
            InferenceBackendOperation::ChatCompletion,
            &response.usage,
        );
        let usage = response.usage.clone();
        let capsule_marker = response.capsule_marker.clone();
        let exchange_id = response.exchange_id.clone();
        let mut http_response =
            json_response_with_usage(ChatCompletionResponse::from(response), &usage);
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
}
