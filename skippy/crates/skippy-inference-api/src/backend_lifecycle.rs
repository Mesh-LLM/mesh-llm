//! Correlated lifecycle ownership for one frontend-to-backend dispatch.

use std::{future::Future, sync::Arc, time::Duration};

use crate::{
    backend::{InferenceRequestContext, InferenceResult},
    errors::InferenceError,
    lifecycle::{
        CLIENT_CLOSED_REQUEST_STATUS, InferenceBackendOperation, InferenceFailure,
        InferenceLifecycleContext, InferenceLifecycleEvent, InferenceLifecycleObserver,
        InferenceTerminalResult, terminal_result_for_error,
    },
};

pub(crate) async fn call_backend<T, F>(
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    context: &InferenceLifecycleContext,
    operation: InferenceBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    future: F,
) -> InferenceResult<T>
where
    F: Future<Output = InferenceResult<T>>,
{
    call_backend_inner(
        observer,
        context,
        operation,
        operation_name,
        timeout,
        None,
        future,
    )
    .await
}

pub(crate) async fn call_backend_with_context<T, F>(
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    lifecycle_context: &InferenceLifecycleContext,
    operation: InferenceBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    request_context: &InferenceRequestContext,
    future: F,
) -> InferenceResult<T>
where
    F: Future<Output = InferenceResult<T>>,
{
    call_backend_inner(
        observer,
        lifecycle_context,
        operation,
        operation_name,
        timeout,
        Some(request_context),
        future,
    )
    .await
}

async fn call_backend_inner<T, F>(
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    lifecycle_context: &InferenceLifecycleContext,
    operation: InferenceBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    request_context: Option<&InferenceRequestContext>,
    future: F,
) -> InferenceResult<T>
where
    F: Future<Output = InferenceResult<T>>,
{
    let mut lifecycle = BackendLifecycle::start(
        observer,
        lifecycle_context.clone(),
        operation,
        request_context.cloned(),
    );
    let result = match timeout {
        Some(timeout) => match tokio::time::timeout(timeout, future).await {
            Ok(result) => result,
            Err(_) => {
                if let Some(context) = request_context {
                    context.cancel();
                }
                let error = InferenceError::timeout(format!(
                    "{operation_name} timed out after {} ms",
                    timeout.as_millis()
                ));
                lifecycle.finish(terminal_result_for_error(&error));
                return Err(error);
            }
        },
        None => future.await,
    };
    let terminal = match &result {
        Ok(_) => InferenceTerminalResult::Completed { status_code: 200 },
        Err(error) => terminal_result_for_error(error),
    };
    lifecycle.finish(terminal);
    result
}

struct BackendLifecycle {
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    context: InferenceLifecycleContext,
    operation: InferenceBackendOperation,
    request_context: Option<InferenceRequestContext>,
    terminal: bool,
}

impl BackendLifecycle {
    fn start(
        observer: Option<Arc<dyn InferenceLifecycleObserver>>,
        context: InferenceLifecycleContext,
        operation: InferenceBackendOperation,
        request_context: Option<InferenceRequestContext>,
    ) -> Self {
        if let Some(observer) = &observer {
            observer.observe(&InferenceLifecycleEvent::BackendDispatched {
                context: context.clone(),
                operation,
            });
        }
        Self {
            observer,
            context,
            operation,
            request_context,
            terminal: false,
        }
    }

    fn finish(&mut self, result: InferenceTerminalResult) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if let Some(observer) = &self.observer {
            observer.observe(&InferenceLifecycleEvent::BackendTerminal {
                context: self.context.clone(),
                operation: self.operation,
                result,
            });
        }
    }
}

impl Drop for BackendLifecycle {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        if let Some(context) = &self.request_context {
            context.cancel();
        }
        self.finish(InferenceTerminalResult::Failed {
            status_code: CLIENT_CLOSED_REQUEST_STATUS,
            failure: InferenceFailure::Cancelled,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, time::Duration};

    use super::*;
    use crate::lifecycle::{InferenceFrontendRoute, InferenceRequestMethod, parse_request_id};

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<InferenceLifecycleEvent>>);

    impl InferenceLifecycleObserver for RecordingObserver {
        fn observe(&self, event: &InferenceLifecycleEvent) {
            self.0.lock().expect("observer lock").push(event.clone());
        }
    }

    fn context() -> InferenceLifecycleContext {
        InferenceLifecycleContext::new(
            parse_request_id("f84fa37c-a268-4d3a-962e-aa4b229672fa").expect("request ID"),
            InferenceRequestMethod::Post,
            InferenceFrontendRoute::ChatCompletions,
        )
    }

    #[tokio::test]
    async fn successful_dispatch_has_exactly_one_correlated_terminal() {
        let observer = Arc::new(RecordingObserver::default());
        let result = call_backend(
            Some(observer.clone()),
            &context(),
            InferenceBackendOperation::Models,
            "models",
            None,
            async { Ok::<_, InferenceError>(()) },
        )
        .await;

        assert!(result.is_ok());
        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [
                InferenceLifecycleEvent::BackendDispatched { .. },
                InferenceLifecycleEvent::BackendTerminal {
                    result: InferenceTerminalResult::Completed { status_code: 200 },
                    ..
                },
            ]
        ));
    }

    #[tokio::test]
    async fn timeout_cancels_context_and_classifies_backend_terminal() {
        let observer = Arc::new(RecordingObserver::default());
        let request_context = InferenceRequestContext::with_request_id(context().request_id);
        let result = call_backend_with_context(
            Some(observer.clone()),
            &context(),
            InferenceBackendOperation::ChatCompletion,
            "chat_completion",
            Some(Duration::from_millis(1)),
            &request_context,
            std::future::pending::<InferenceResult<()>>(),
        )
        .await;

        assert!(result.is_err());
        assert!(request_context.is_cancelled());
        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [
                InferenceLifecycleEvent::BackendDispatched { .. },
                InferenceLifecycleEvent::BackendTerminal {
                    result: InferenceTerminalResult::Failed {
                        status_code: 504,
                        failure: InferenceFailure::Timeout,
                    },
                    ..
                },
            ]
        ));
    }
}
