//! Await bounded terminal observation before exposing stream EOF or error.
use super::*;
use crate::InferenceError;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pub(super) type TerminalFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
enum PendingEmission {
    Error(InferenceError),
    End,
}
pub struct TerminalGuardedChatStream {
    inner: ChatCompletionStream,
    guard: Option<TerminalGuard>,
    terminal: Option<TerminalFuture>,
    pending: Option<PendingEmission>,
}
impl TerminalGuardedChatStream {
    pub fn pinned(inner: ChatCompletionStream, guard: TerminalGuard) -> ChatCompletionStream {
        Box::pin(Self {
            inner,
            guard: Some(guard),
            terminal: None,
            pending: None,
        })
    }
}
impl Drop for TerminalGuardedChatStream {
    fn drop(&mut self) {
        if let Some(terminal) = self.terminal.take()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(terminal);
        }
    }
}
impl futures_core::Stream for TerminalGuardedChatStream {
    type Item = InferenceResult<ChatCompletionChunk>;
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
            Poll::Ready(Some(Err(error))) => Some(OwnedChatCompletionOutcome::Error {
                status: error.status().as_u16(),
                message: error.to_string(),
            }),
            Poll::Ready(None) => Some(OwnedChatCompletionOutcome::StreamCompleted),
            _ => None,
        };
        if let Some(outcome) = outcome
            && let Some(guard) = self.guard.take()
        {
            self.terminal = Some(guard.into_terminal_future(outcome));
            self.pending = Some(match next {
                Poll::Ready(Some(Err(error))) => PendingEmission::Error(error),
                _ => PendingEmission::End,
            });
            return self.poll_next(cx);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceError;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct BlockingTerminal {
        started: Notify,
        release: Notify,
        completed: Notify,
        calls: AtomicUsize,
        expected_error: bool,
    }
    #[async_trait]
    impl InferenceHookPolicy for BlockingTerminal {
        async fn on_chat_completion_terminal(
            &self,
            _request: &ChatCompletionRequest,
            _id: &str,
            outcome: &ChatCompletionOutcome<'_>,
        ) {
            assert!(matches!(
                outcome,
                ChatCompletionOutcome::Error { .. } | ChatCompletionOutcome::StreamCompleted
            ));
            assert_eq!(
                matches!(outcome, ChatCompletionOutcome::Error { .. }),
                self.expected_error
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            self.completed.notify_one();
        }
    }
    #[tokio::test]
    async fn terminal_callback_finishes_before_stream_error_or_eof_is_exposed() {
        for error in [false, true] {
            let hooks = Arc::new(BlockingTerminal {
                started: Notify::new(),
                release: Notify::new(),
                completed: Notify::new(),
                calls: AtomicUsize::new(0),
                expected_error: error,
            });
            let inner: ChatCompletionStream = if error {
                Box::pin(futures_util::stream::once(async {
                    Err(InferenceError::backend("failure"))
                }))
            } else {
                Box::pin(futures_util::stream::empty())
            };
            let guard =
                TerminalGuard::new(hooks.clone(), ChatCompletionRequest::default(), "id".into());
            let mut stream = TerminalGuardedChatStream::pinned(inner, guard);
            let pending = tokio::spawn(async move { stream.next().await });
            hooks.started.notified().await;
            assert!(
                !pending.is_finished(),
                "terminal signal must precede emission termination"
            );
            hooks.release.notify_one();
            let result = pending.await.unwrap();
            assert_eq!(result.is_some(), error);
        }
    }

    #[tokio::test]
    async fn dropping_pending_terminal_resumes_chat_callback_exactly_once() {
        for error in [false, true] {
            let hooks = Arc::new(BlockingTerminal {
                started: Notify::new(),
                release: Notify::new(),
                completed: Notify::new(),
                calls: AtomicUsize::new(0),
                expected_error: error,
            });
            let inner: ChatCompletionStream = if error {
                Box::pin(futures_util::stream::once(async {
                    Err(InferenceError::backend("failure"))
                }))
            } else {
                Box::pin(futures_util::stream::empty())
            };
            let guard =
                TerminalGuard::new(hooks.clone(), ChatCompletionRequest::default(), "id".into());
            let mut stream = TerminalGuardedChatStream::pinned(inner, guard);
            assert!(futures_util::poll!(stream.next()).is_pending());
            hooks.started.notified().await;
            drop(stream);
            hooks.release.notify_one();
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                hooks.completed.notified(),
            )
            .await
            .expect("dropping the body must preserve the pending callback");
            assert_eq!(hooks.calls.load(Ordering::SeqCst), 1);
        }
    }
}
