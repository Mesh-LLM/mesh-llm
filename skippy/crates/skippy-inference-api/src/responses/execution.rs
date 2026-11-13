//! Shared backend execution for streaming Responses transports.

use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;

use super::{PreparedResponsesTurn, ResponsesStream, ResponsesStreamEvent};
use crate::{
    InferenceBackend, InferenceBackendOperation, InferenceError, InferenceLifecycleContext,
    InferenceLifecycleObserver, InferenceRequestContext,
    backend::CancellationToken,
    backend_lifecycle::call_backend_with_context,
    chat::ChatCompletionChunk,
    generation::GenerationEventStream,
    stream_lifecycle::{StreamLifecycle, observe_backend_stream},
};

/// Starts prepared Responses turns against an OpenAI-compatible backend.
///
/// The session/preparation layer is independent of this direct executor; a
/// future Mesh ingress can route the prepared request through its own pipeline
/// and still use [`ResponsesStream`] for the returned chunks.
#[derive(Clone)]
pub struct ResponsesSessionExecutor {
    backend: Arc<dyn InferenceBackend>,
    backend_timeout: Option<Duration>,
    lifecycle_observer: Option<Arc<dyn InferenceLifecycleObserver>>,
}

impl std::fmt::Debug for ResponsesSessionExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResponsesSessionExecutor")
            .field("backend_timeout", &self.backend_timeout)
            .field("has_lifecycle_observer", &self.lifecycle_observer.is_some())
            .finish_non_exhaustive()
    }
}

impl ResponsesSessionExecutor {
    pub fn new(
        backend: Arc<dyn InferenceBackend>,
        backend_timeout: Option<Duration>,
        lifecycle_observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    ) -> Self {
        Self {
            backend,
            backend_timeout,
            lifecycle_observer,
        }
    }

    pub async fn start(
        &self,
        prepared: PreparedResponsesTurn,
        lifecycle_context: InferenceLifecycleContext,
    ) -> Result<ResponsesTurnEventStream, InferenceError> {
        let (turn, request, trusted_agent_session, previous_response_id) =
            prepared.into_execution_parts();
        let mut backend_context =
            InferenceRequestContext::with_request_id(lifecycle_context.request_id)
                .with_stream_usage_observation();
        if trusted_agent_session {
            backend_context = backend_context.with_trusted_agent_session();
        }
        let cancellation = backend_context.cancellation_token();
        let mut start_cancellation = CancelBackendStartOnDrop::new(cancellation.clone());
        let stream = call_backend_with_context(
            self.lifecycle_observer.clone(),
            &lifecycle_context,
            InferenceBackendOperation::ResponsesStream,
            "responses_stream",
            self.backend_timeout,
            &backend_context,
            self.backend
                .chat_completion_stream(request, backend_context.clone()),
        )
        .await;
        start_cancellation.disarm();
        let stream = stream?;
        let lifecycle = StreamLifecycle::new(
            self.lifecycle_observer.clone(),
            lifecycle_context,
            InferenceBackendOperation::ResponsesStream,
        );
        let exchange_id = backend_context.exchange_id();
        if let Some(exchange_id) = exchange_id.clone() {
            lifecycle.record_exchange_id(exchange_id);
        }
        let stream: GenerationEventStream =
            Box::pin(observe_backend_stream(stream, lifecycle.clone()));
        Ok(ResponsesTurnEventStream {
            stream,
            machine: ResponsesStream::new(turn),
            cancellation,
            lifecycle,
            exchange_id,
            previous_response_id,
            settled: false,
            delivery_finished: false,
        })
    }
}

/// Keeps backend-start cancellation owned while no stream guard exists yet.
///
/// `call_backend_with_context` also cancels its request context when its future
/// is dropped. This outer guard makes that ownership explicit across the
/// executor's whole start phase, including future refactors around the backend
/// lifecycle wrapper.
struct CancelBackendStartOnDrop {
    cancellation: CancellationToken,
    armed: bool,
}

impl CancelBackendStartOnDrop {
    fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelBackendStartOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// One running turn yielding the exact JSON event objects used by SSE.
pub struct ResponsesTurnEventStream {
    stream: GenerationEventStream,
    machine: ResponsesStream,
    cancellation: CancellationToken,
    lifecycle: StreamLifecycle,
    exchange_id: Option<String>,
    previous_response_id: Option<String>,
    settled: bool,
    delivery_finished: bool,
}

impl ResponsesTurnEventStream {
    pub fn start_events(&mut self) -> Vec<ResponsesStreamEvent> {
        self.machine.start()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub(crate) fn lifecycle(&self) -> StreamLifecycle {
        self.lifecycle.clone()
    }

    pub fn exchange_id(&self) -> Option<&str> {
        self.exchange_id.as_deref()
    }

    pub fn previous_response_id(&self) -> Option<&str> {
        self.previous_response_id.as_deref()
    }

    pub fn is_failed(&self) -> bool {
        self.machine.is_failed()
    }

    /// Return the next non-empty group of semantic Responses events.
    pub async fn next_events(&mut self) -> Option<Vec<ResponsesStreamEvent>> {
        if self.settled {
            self.finish_delivery();
            return None;
        }
        loop {
            match self.stream.next().await {
                Some(item) => {
                    let events = self.item_events(item.map(ChatCompletionChunk::from));
                    if self.machine.is_failed() {
                        self.cancellation.cancel();
                        self.lifecycle
                            .mark_protocol_failed("Responses stream ended unsuccessfully");
                        self.settled = true;
                    }
                    if !events.is_empty() {
                        return Some(events);
                    }
                }
                None => {
                    let already_terminated = self.machine.is_terminated();
                    let events = self.machine.finish();
                    if self.machine.is_failed() {
                        self.lifecycle
                            .mark_protocol_failed("Responses stream ended unsuccessfully");
                    } else if !already_terminated {
                        self.lifecycle.mark_protocol_complete();
                    }
                    self.settled = true;
                    return (!events.is_empty()).then_some(events);
                }
            }
        }
    }

    fn finish_delivery(&mut self) {
        if !self.delivery_finished {
            self.lifecycle.finish_natural();
            self.delivery_finished = true;
        }
    }

    fn item_events(
        &mut self,
        item: Result<ChatCompletionChunk, InferenceError>,
    ) -> Vec<ResponsesStreamEvent> {
        match item {
            Ok(chunk) => {
                if let Some(usage) = chunk.usage.as_ref() {
                    self.lifecycle.capture_usage(usage);
                }
                match serde_json::to_value(&chunk) {
                    Ok(chunk) => self.machine.push_chunk(&chunk),
                    Err(error) => self.machine.fail("server_error", &error.to_string()),
                }
            }
            Err(error) => {
                let body = error.body().error;
                let code = body.code.unwrap_or_else(|| body.error_type.clone());
                self.machine.fail(&code, &body.message)
            }
        }
    }
}

impl Drop for ResponsesTurnEventStream {
    fn drop(&mut self) {
        if !self.delivery_finished {
            self.lifecycle.finish_drop(self.cancellation.is_cancelled());
        }
        self.cancellation.cancel();
    }
}
