//! Completion stream termination signals backend outcomes independently of HTTP status.
use super::*;
use skippy_inference_api::InferenceHookPolicy;
use std::pin::Pin;
use std::task::{Context, Poll};

type TerminalFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
pub(super) struct CompletionTerminalGuard {
    hooks: Arc<dyn InferenceHookPolicy>,
    id: String,
    finished: bool,
}
impl CompletionTerminalGuard {
    pub(super) fn new(hooks: Arc<dyn InferenceHookPolicy>, id: String) -> Self {
        Self {
            hooks,
            id,
            finished: false,
        }
    }
    pub(super) fn finish(&mut self, outcome: &'static str) {
        if let Some(future) = self.terminal_future(outcome)
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(future);
        }
    }
    fn terminal_future(&mut self, outcome: &'static str) -> Option<TerminalFuture> {
        if self.finished {
            return None;
        }
        self.finished = true;
        let hooks = self.hooks.clone();
        let id = self.id.clone();
        Some(Box::pin(async move {
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                hooks.on_completion_terminal(&id, outcome),
            )
            .await;
        }))
    }
}
impl Drop for CompletionTerminalGuard {
    fn drop(&mut self) {
        self.finish("client_cancelled");
    }
}

struct GuardedCompletionStream {
    inner: CompletionStream,
    guard: Option<CompletionTerminalGuard>,
    terminal: Option<TerminalFuture>,
    pending: Option<PendingEmission>,
}
impl Drop for GuardedCompletionStream {
    fn drop(&mut self) {
        if let Some(terminal) = self.terminal.take()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(terminal);
        }
    }
}
enum PendingEmission {
    Error(InferenceError),
    End,
}
pub(super) fn error_outcome(error: &InferenceError) -> &'static str {
    if error.status().as_u16() == 504 {
        "timed_out"
    } else {
        "backend_error"
    }
}
impl futures_util::Stream for GuardedCompletionStream {
    type Item = InferenceResult<skippy_inference_api::CompletionChunk>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(terminal) = self.terminal.as_mut() {
            if terminal.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.terminal = None;
            return Poll::Ready(match self.pending.take() {
                Some(PendingEmission::Error(error)) => Some(Err(error)),
                _ => None,
            });
        }
        let next = self.inner.as_mut().poll_next(cx);
        let outcome = match &next {
            Poll::Ready(Some(Err(error))) => Some(error_outcome(error)),
            Poll::Ready(None) => Some("completed"),
            _ => None,
        };
        let terminal = outcome.and_then(|outcome| self.guard.as_mut()?.terminal_future(outcome));
        if let Some(terminal) = terminal {
            self.terminal = Some(terminal);
            self.pending = Some(match next {
                Poll::Ready(Some(Err(error))) => PendingEmission::Error(error),
                _ => PendingEmission::End,
            });
            return self.poll_next(cx);
        }
        next
    }
}
pub(super) fn guarded(
    stream: CompletionStream,
    guard: Option<CompletionTerminalGuard>,
) -> CompletionStream {
    Box::pin(GuardedCompletionStream {
        inner: stream,
        guard,
        terminal: None,
        pending: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct BlockingTerminal {
        started: Notify,
        release: Notify,
        completed: Notify,
        calls: AtomicUsize,
        expected: &'static str,
    }
    #[async_trait]
    impl InferenceHookPolicy for BlockingTerminal {
        async fn on_completion_terminal(&self, _id: &str, outcome: &str) {
            assert_eq!(outcome, self.expected);
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            self.completed.notify_one();
        }
    }
    #[tokio::test]
    async fn completion_error_waits_for_backend_outcome_signal() {
        let hooks = Arc::new(BlockingTerminal {
            started: Notify::new(),
            release: Notify::new(),
            completed: Notify::new(),
            calls: AtomicUsize::new(0),
            expected: "backend_error",
        });
        let inner: CompletionStream = Box::pin(futures_util::stream::once(async {
            Err(InferenceError::backend("failure"))
        }));
        let mut stream = guarded(
            inner,
            Some(CompletionTerminalGuard::new(hooks.clone(), "id".into())),
        );
        let pending = tokio::spawn(async move { stream.next().await });
        hooks.started.notified().await;
        assert!(!pending.is_finished());
        hooks.release.notify_one();
        assert!(pending.await.unwrap().unwrap().is_err());
    }

    #[tokio::test]
    async fn dropping_pending_terminal_resumes_completion_callback_exactly_once() {
        for error in [false, true] {
            let hooks = Arc::new(BlockingTerminal {
                started: Notify::new(),
                release: Notify::new(),
                completed: Notify::new(),
                calls: AtomicUsize::new(0),
                expected: if error { "backend_error" } else { "completed" },
            });
            let inner: CompletionStream = if error {
                Box::pin(futures_util::stream::once(async {
                    Err(InferenceError::backend("failure"))
                }))
            } else {
                Box::pin(futures_util::stream::empty())
            };
            let mut stream = guarded(
                inner,
                Some(CompletionTerminalGuard::new(hooks.clone(), "id".into())),
            );
            assert!(futures_util::poll!(stream.next()).is_pending());
            hooks.started.notified().await;
            drop(stream);
            hooks.release.notify_one();
            tokio::time::timeout(Duration::from_secs(1), hooks.completed.notified())
                .await
                .expect("dropping the body must preserve the pending callback");
            assert_eq!(hooks.calls.load(Ordering::SeqCst), 1);
        }
    }
}
