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
use futures_util::Stream;
use mesh_llm_events::logging::events::TokenUsage;

use crate::{
    backend::CancellationToken,
    common::Usage,
    errors::OpenAiError,
    lifecycle::{
        OpenAiLifecycleContext, OpenAiLifecycleEvent, OpenAiLifecycleObserver, OpenAiTerminalResult,
    },
};

use super::{authoritative_usage, failure_for_status};

#[derive(Clone, Copy)]
pub(super) struct StreamingResponse;

pub(super) fn sse_response<S>(
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

#[derive(Clone)]
pub(super) struct StreamLifecycle {
    observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
    context: OpenAiLifecycleContext,
    terminal: Arc<AtomicBool>,
    usage: Arc<Mutex<Option<TokenUsage>>>,
}

impl StreamLifecycle {
    pub(super) fn new(
        observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
        context: OpenAiLifecycleContext,
    ) -> Self {
        Self {
            observer,
            context,
            terminal: Arc::new(AtomicBool::new(false)),
            usage: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn observe_usage(&self, usage: &Usage) {
        if let Some(usage) = authoritative_usage(usage) {
            *self.usage.lock().expect("stream usage lock poisoned") = Some(usage);
        }
    }

    fn completed(&self) {
        self.observe_terminal(OpenAiLifecycleEvent::StreamTerminal {
            context: self.context.clone(),
            result: self
                .usage
                .lock()
                .expect("stream usage lock poisoned")
                .map_or(
                    OpenAiTerminalResult::Completed { status_code: 200 },
                    |usage| OpenAiTerminalResult::CompletedWithUsage {
                        status_code: 200,
                        usage,
                    },
                ),
        });
    }

    pub(super) fn failed(&self, error: &OpenAiError) {
        self.observe_terminal(OpenAiLifecycleEvent::StreamTerminal {
            context: self.context.clone(),
            result: OpenAiTerminalResult::Failed {
                status_code: error.status().as_u16(),
                failure: failure_for_status(error.status()),
            },
        });
    }

    fn dropped(&self, cancelled: bool) {
        let event = if cancelled {
            OpenAiLifecycleEvent::StreamCancelled {
                context: self.context.clone(),
            }
        } else {
            OpenAiLifecycleEvent::StreamDropped {
                context: self.context.clone(),
            }
        };
        self.observe_terminal(event);
    }

    fn observe_terminal(&self, event: OpenAiLifecycleEvent) {
        if self.terminal.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(observer) = &self.observer {
            observer.observe(&event);
        }
    }
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
            self.lifecycle.completed();
        }
        poll
    }
}

impl Drop for CancelOnDropSseStream {
    fn drop(&mut self) {
        self.lifecycle.dropped(self.cancellation.is_cancelled());
        self.cancellation.cancel();
    }
}
