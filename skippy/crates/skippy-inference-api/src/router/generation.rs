//! OpenAI Chat/Responses adapters onto the protocol-neutral generation boundary.
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

pub(super) async fn responses(
    State(state): State<FrontendState>,
    Extension(context): Extension<InferenceLifecycleContext>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Response, InferenceError> {
    let Json(mut value) = json_payload(payload)?;
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
            stream_responses(&state, &context, request, trusted_agent_session).await
        }
        _ => non_streaming_responses(&state, &context, request, trusted_agent_session).await,
    }
}

async fn stream_responses(
    state: &FrontendState,
    context: &InferenceLifecycleContext,
    request: GenerationRequest,
    trusted_agent_session: bool,
) -> Result<Response, InferenceError> {
    let include_usage = request.include_usage();
    let backend_context = request_context(context.request_id, trusted_agent_session, true);
    let cancellation = backend_context.cancellation_token();
    let state_machine = Arc::new(Mutex::new(ResponseSseState::new(request.model.clone())));
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
    let body_state = state_machine.clone();
    let usage_lifecycle = lifecycle.clone();
    let body_events = stream.flat_map(move |item| {
        let out = responses_stream_body_events(&body_state, &usage_lifecycle, item.map(Into::into));
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
    let mut response = sse_response(events, cancellation, lifecycle);
    if let Some(exchange_id) = exchange_id {
        response
            .extensions_mut()
            .insert(ExchangeIdExtension(exchange_id));
    }
    Ok(response)
}

async fn non_streaming_responses(
    state: &FrontendState,
    context: &InferenceLifecycleContext,
    request: GenerationRequest,
    trusted_agent_session: bool,
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
    state.response_completed(
        context,
        InferenceBackendOperation::Responses,
        &response.usage,
    );
    let usage = response.usage.clone();
    let capsule_marker = response.capsule_marker.clone();
    let exchange_id = response.exchange_id.clone();
    let translated =
        translate_chat_completion_response_to_responses(&ChatCompletionResponse::from(response))?;
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

fn responses_stream_body_events(
    state_machine: &Mutex<ResponseSseState>,
    usage_lifecycle: &StreamLifecycle,
    item: InferenceResult<ChatCompletionChunk>,
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
