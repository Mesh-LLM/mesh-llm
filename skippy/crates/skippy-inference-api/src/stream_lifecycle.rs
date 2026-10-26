//! Stream observation and terminal ownership for OpenAI SSE responses.

use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

use axum::response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
};
use futures_util::{Stream, StreamExt};
use skippy_events::usage::TokenUsage;

use crate::{
    backend::{CancellationToken, InferenceResult},
    common::Usage,
    errors::InferenceError,
    lifecycle::{
        InferenceBackendOperation, InferenceLifecycleContext, InferenceLifecycleEvent,
        InferenceLifecycleObserver, InferenceTerminalResult, InferenceUsage,
        terminal_result_for_error,
    },
};

#[derive(Clone, Copy)]
struct StreamingResponse;

#[derive(Clone)]
pub(crate) struct StreamLifecycle {
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    context: InferenceLifecycleContext,
    operation: InferenceBackendOperation,
    terminal: Arc<AtomicBool>,
    backend_error: Arc<AtomicBool>,
    protocol_complete: Arc<AtomicBool>,
    usage: Arc<Mutex<Option<InferenceUsage>>>,
}

impl StreamLifecycle {
    pub(crate) fn new(
        observer: Option<Arc<dyn InferenceLifecycleObserver>>,
        context: InferenceLifecycleContext,
        operation: InferenceBackendOperation,
    ) -> Self {
        Self {
            observer,
            context,
            operation,
            terminal: Arc::new(AtomicBool::new(false)),
            backend_error: Arc::new(AtomicBool::new(false)),
            protocol_complete: Arc::new(AtomicBool::new(false)),
            usage: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn capture_usage(&self, usage: &Usage) {
        *self.usage.lock().expect("stream usage lock poisoned") = Some(usage.into());
    }

    pub(crate) fn record_exchange_id(&self, exchange_id: String) {
        self.observe(&InferenceLifecycleEvent::ExchangeIdentified {
            context: self.context.clone(),
            exchange_id,
        });
    }

    /// Mark the client-visible protocol terminal before yielding `[DONE]`.
    ///
    /// Some clients close immediately after that frame and never poll the body
    /// to EOF. The marker lets drop classify that path as success.
    pub(crate) fn mark_protocol_complete(&self) {
        self.protocol_complete.store(true, Ordering::Release);
    }

    fn first_item(&self) {
        self.observe(&InferenceLifecycleEvent::StreamFirstItem {
            context: self.context.clone(),
            operation: self.operation,
        });
    }

    fn failed(&self, error: &InferenceError) {
        self.backend_error.store(true, Ordering::Release);
        self.finish_terminal(InferenceLifecycleEvent::StreamTerminal {
            context: self.context.clone(),
            result: terminal_result_for_error(error),
        });
    }

    fn finish_natural(&self) {
        if self.backend_error.load(Ordering::Acquire) {
            return;
        }
        self.finish_success();
    }

    fn finish_drop(&self, cancellation_already_requested: bool) {
        match self.drop_outcome(cancellation_already_requested) {
            StreamDropOutcome::BackendError => {}
            StreamDropOutcome::Completed => self.finish_success(),
            StreamDropOutcome::Cancelled => {
                self.finish_terminal(InferenceLifecycleEvent::StreamCancelled {
                    context: self.context.clone(),
                });
            }
            StreamDropOutcome::ClientDisconnect => {
                self.finish_terminal(InferenceLifecycleEvent::StreamDropped {
                    context: self.context.clone(),
                });
            }
        }
    }

    fn drop_outcome(&self, cancellation_already_requested: bool) -> StreamDropOutcome {
        if self.backend_error.load(Ordering::Acquire) {
            StreamDropOutcome::BackendError
        } else if self.protocol_complete.load(Ordering::Acquire) {
            StreamDropOutcome::Completed
        } else if cancellation_already_requested {
            StreamDropOutcome::Cancelled
        } else {
            StreamDropOutcome::ClientDisconnect
        }
    }

    fn finish_success(&self) {
        if !self.claim_terminal() {
            return;
        }
        let usage = *self.usage.lock().expect("stream usage lock poisoned");
        if let Some(usage) = usage {
            self.observe(&InferenceLifecycleEvent::ResponseCompleted {
                context: self.context.clone(),
                operation: self.operation,
                usage,
            });
        }
        let result = usage
            .and_then(|usage| {
                TokenUsage::from_counts(
                    Some(u64::from(usage.prompt_tokens)),
                    Some(u64::from(usage.completion_tokens)),
                    Some(u64::from(usage.total_tokens)),
                )
                .map(|authoritative| {
                    authoritative.with_cached_prompt_tokens(usage.cached_tokens.map(u64::from))
                })
            })
            .map_or(
                InferenceTerminalResult::Completed { status_code: 200 },
                |usage| InferenceTerminalResult::CompletedWithUsage {
                    status_code: 200,
                    usage,
                },
            );
        self.observe(&InferenceLifecycleEvent::StreamTerminal {
            context: self.context.clone(),
            result,
        });
    }

    fn finish_terminal(&self, event: InferenceLifecycleEvent) {
        if self.claim_terminal() {
            self.observe(&event);
        }
    }

    fn claim_terminal(&self) -> bool {
        !self.terminal.swap(true, Ordering::AcqRel)
    }

    fn observe(&self, event: &InferenceLifecycleEvent) {
        if let Some(observer) = &self.observer {
            observer.observe(event);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamDropOutcome {
    BackendError,
    Completed,
    Cancelled,
    ClientDisconnect,
}

pub(crate) fn observe_backend_stream<S, T>(
    stream: S,
    lifecycle: StreamLifecycle,
) -> impl Stream<Item = InferenceResult<T>> + Send + 'static
where
    S: Stream<Item = InferenceResult<T>> + Send + 'static,
    T: Send + 'static,
{
    let mut first_item = true;
    stream.map(move |item| {
        if first_item {
            first_item = false;
            lifecycle.first_item();
        }
        if let Err(error) = &item {
            lifecycle.failed(error);
        }
        item
    })
}

pub(crate) fn sse_response<S>(
    events: S,
    cancellation: CancellationToken,
    lifecycle: StreamLifecycle,
) -> Response
where
    S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    let mut response = Sse::new(CancelOnDropSseStream::new(events, cancellation, lifecycle))
        .keep_alive(KeepAlive::default())
        .into_response();
    response.extensions_mut().insert(StreamingResponse);
    response
}

pub(crate) fn is_streaming_response(response: &Response) -> bool {
    response.extensions().get::<StreamingResponse>().is_some()
}

struct CancelOnDropSseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send + 'static>>,
    cancellation: CancellationToken,
    lifecycle: StreamLifecycle,
}

impl CancelOnDropSseStream {
    fn new<S>(inner: S, cancellation: CancellationToken, lifecycle: StreamLifecycle) -> Self
    where
        S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            cancellation,
            lifecycle,
        }
    }
}

impl Stream for CancelOnDropSseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            self.lifecycle.finish_natural();
        }
        poll
    }
}

impl Drop for CancelOnDropSseStream {
    fn drop(&mut self) {
        self.lifecycle.finish_drop(self.cancellation.is_cancelled());
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::stream;

    use super::*;
    use crate::lifecycle::{
        InferenceFailure, InferenceFrontendRoute, InferenceRequestMethod, parse_request_id,
    };

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<InferenceLifecycleEvent>>);

    impl RecordingObserver {
        fn events(&self) -> Vec<InferenceLifecycleEvent> {
            self.0.lock().expect("observer lock").clone()
        }
    }

    impl InferenceLifecycleObserver for RecordingObserver {
        fn observe(&self, event: &InferenceLifecycleEvent) {
            self.0.lock().expect("observer lock").push(event.clone());
        }
    }

    fn lifecycle(observer: Arc<RecordingObserver>) -> StreamLifecycle {
        StreamLifecycle::new(
            Some(observer),
            InferenceLifecycleContext::new(
                parse_request_id("ac04bc97-ab30-4111-826d-60b7c4b6e720").expect("request ID"),
                InferenceRequestMethod::Post,
                InferenceFrontendRoute::ChatCompletions,
            ),
            InferenceBackendOperation::ChatCompletionStream,
        )
    }

    #[test]
    fn protocol_completion_then_drop_emits_one_success_and_cached_usage() {
        let observer = Arc::new(RecordingObserver::default());
        let lifecycle = lifecycle(observer.clone());
        lifecycle.capture_usage(&Usage::new(12, 3).with_cached_tokens(9));
        lifecycle.mark_protocol_complete();

        lifecycle.finish_drop(false);
        lifecycle.finish_natural();
        lifecycle.finish_drop(false);

        let events = observer.events();
        assert!(matches!(
            events.as_slice(),
            [
                InferenceLifecycleEvent::ResponseCompleted {
                    usage: InferenceUsage {
                        prompt_tokens: 12,
                        cached_tokens: Some(9),
                        completion_tokens: 3,
                        total_tokens: 15,
                    },
                    ..
                },
                InferenceLifecycleEvent::StreamTerminal {
                    result: InferenceTerminalResult::CompletedWithUsage {
                        status_code: 200,
                        usage: TokenUsage {
                            prompt_tokens: Some(12),
                            cached_prompt_tokens: Some(9),
                            completion_tokens: Some(3),
                            total_tokens: Some(15),
                        },
                    },
                    ..
                },
            ]
        ));
    }

    #[test]
    fn stream_error_wins_over_protocol_completion_and_drop() {
        let observer = Arc::new(RecordingObserver::default());
        let lifecycle = lifecycle(observer.clone());
        lifecycle.capture_usage(&Usage::new(3, 1));
        lifecycle.mark_protocol_complete();
        lifecycle.failed(&InferenceError::backend("private backend detail"));

        lifecycle.finish_natural();
        lifecycle.finish_drop(false);

        let events = observer.events();
        assert!(matches!(
            events.as_slice(),
            [InferenceLifecycleEvent::StreamTerminal {
                result: InferenceTerminalResult::Failed {
                    status_code: 502,
                    failure: InferenceFailure::Backend,
                },
                ..
            }]
        ));
    }

    #[test]
    fn incomplete_stream_drop_distinguishes_cancel_from_client_disconnect() {
        let cancelled_observer = Arc::new(RecordingObserver::default());
        lifecycle(cancelled_observer.clone()).finish_drop(true);
        assert!(matches!(
            cancelled_observer.events().as_slice(),
            [InferenceLifecycleEvent::StreamCancelled { .. }]
        ));

        let dropped_observer = Arc::new(RecordingObserver::default());
        lifecycle(dropped_observer.clone()).finish_drop(false);
        assert!(matches!(
            dropped_observer.events().as_slice(),
            [InferenceLifecycleEvent::StreamDropped { .. }]
        ));
    }

    #[tokio::test]
    async fn delivered_done_frame_then_body_drop_is_success_without_eof_poll() {
        let observer = Arc::new(RecordingObserver::default());
        let lifecycle = lifecycle(observer.clone());
        let completion = lifecycle.clone();
        let events = stream::once(async move {
            completion.mark_protocol_complete();
            Ok(Event::default().data("[DONE]"))
        });
        let mut stream = CancelOnDropSseStream::new(events, CancellationToken::new(), lifecycle);

        assert!(stream.next().await.is_some());
        drop(stream);

        assert!(matches!(
            observer.events().as_slice(),
            [InferenceLifecycleEvent::StreamTerminal {
                result: InferenceTerminalResult::Completed { .. },
                ..
            }]
        ));
    }

    #[tokio::test]
    async fn backend_stream_emits_first_item_once_and_correlated_failure_once() {
        let observer = Arc::new(RecordingObserver::default());
        let lifecycle = lifecycle(observer.clone());
        let source = stream::iter(vec![
            Ok::<_, InferenceError>(1_u8),
            Err(InferenceError::backend("private backend detail")),
        ]);
        let items = observe_backend_stream(source, lifecycle)
            .collect::<Vec<_>>()
            .await;

        assert_eq!(items.len(), 2);
        assert!(matches!(
            observer.events().as_slice(),
            [
                InferenceLifecycleEvent::StreamFirstItem { .. },
                InferenceLifecycleEvent::StreamTerminal {
                    result: InferenceTerminalResult::Failed { .. },
                    ..
                },
            ]
        ));
    }
}
