//! Correlated lifecycle ownership for one frontend-to-backend dispatch.

use std::{future::Future, sync::Arc, time::Duration};

use crate::{
    backend::{OpenAiRequestContext, OpenAiResult},
    errors::OpenAiError,
    lifecycle::{
        CLIENT_CLOSED_REQUEST_STATUS, OpenAiBackendOperation, OpenAiFailure,
        OpenAiLifecycleContext, OpenAiLifecycleEvent, OpenAiLifecycleObserver,
        OpenAiTerminalResult, terminal_result_for_error,
    },
};

pub(crate) async fn call_backend<T, F>(
    observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
    context: &OpenAiLifecycleContext,
    operation: OpenAiBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
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
    observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
    lifecycle_context: &OpenAiLifecycleContext,
    operation: OpenAiBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    request_context: &OpenAiRequestContext,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
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
    observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
    lifecycle_context: &OpenAiLifecycleContext,
    operation: OpenAiBackendOperation,
    operation_name: &'static str,
    timeout: Option<Duration>,
    request_context: Option<&OpenAiRequestContext>,
    future: F,
) -> OpenAiResult<T>
where
    F: Future<Output = OpenAiResult<T>>,
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
                let error = OpenAiError::timeout(format!(
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
        Ok(_) => OpenAiTerminalResult::Completed { status_code: 200 },
        Err(error) => terminal_result_for_error(error),
    };
    lifecycle.finish(terminal);
    result
}

struct BackendLifecycle {
    observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
    context: OpenAiLifecycleContext,
    operation: OpenAiBackendOperation,
    request_context: Option<OpenAiRequestContext>,
    terminal: bool,
}

impl BackendLifecycle {
    fn start(
        observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
        request_context: Option<OpenAiRequestContext>,
    ) -> Self {
        if let Some(observer) = &observer {
            observer.observe(&OpenAiLifecycleEvent::BackendDispatched {
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

    fn finish(&mut self, result: OpenAiTerminalResult) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if let Some(observer) = &self.observer {
            observer.observe(&OpenAiLifecycleEvent::BackendTerminal {
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
        self.finish(OpenAiTerminalResult::Failed {
            status_code: CLIENT_CLOSED_REQUEST_STATUS,
            failure: OpenAiFailure::Cancelled,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, time::Duration};

    use super::*;
    use crate::lifecycle::{OpenAiFrontendRoute, OpenAiRequestMethod, parse_request_id};

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<OpenAiLifecycleEvent>>);

    impl OpenAiLifecycleObserver for RecordingObserver {
        fn observe(&self, event: &OpenAiLifecycleEvent) {
            self.0.lock().expect("observer lock").push(event.clone());
        }
    }

    fn context() -> OpenAiLifecycleContext {
        OpenAiLifecycleContext::new(
            parse_request_id("f84fa37c-a268-4d3a-962e-aa4b229672fa").expect("request ID"),
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::ChatCompletions,
        )
    }

    #[tokio::test]
    async fn successful_dispatch_has_exactly_one_correlated_terminal() {
        let observer = Arc::new(RecordingObserver::default());
        let result = call_backend(
            Some(observer.clone()),
            &context(),
            OpenAiBackendOperation::Models,
            "models",
            None,
            async { Ok::<_, OpenAiError>(()) },
        )
        .await;

        assert!(result.is_ok());
        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [
                OpenAiLifecycleEvent::BackendDispatched { .. },
                OpenAiLifecycleEvent::BackendTerminal {
                    result: OpenAiTerminalResult::Completed { status_code: 200 },
                    ..
                },
            ]
        ));
    }

    #[tokio::test]
    async fn timeout_cancels_context_and_classifies_backend_terminal() {
        let observer = Arc::new(RecordingObserver::default());
        let request_context = OpenAiRequestContext::with_request_id(context().request_id);
        let result = call_backend_with_context(
            Some(observer.clone()),
            &context(),
            OpenAiBackendOperation::ChatCompletion,
            "chat_completion",
            Some(Duration::from_millis(1)),
            &request_context,
            std::future::pending::<OpenAiResult<()>>(),
        )
        .await;

        assert!(result.is_err());
        assert!(request_context.is_cancelled());
        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [
                OpenAiLifecycleEvent::BackendDispatched { .. },
                OpenAiLifecycleEvent::BackendTerminal {
                    result: OpenAiTerminalResult::Failed {
                        status_code: 504,
                        failure: OpenAiFailure::Timeout,
                    },
                    ..
                },
            ]
        ));
    }
}
