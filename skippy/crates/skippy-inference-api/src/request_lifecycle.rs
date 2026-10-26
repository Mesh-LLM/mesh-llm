//! Terminal ownership for an OpenAI HTTP request before any stream handoff.

use std::sync::Arc;

use axum::http::StatusCode;
use skippy_events::usage::TokenUsage;

use crate::lifecycle::{
    CLIENT_CLOSED_REQUEST_STATUS, InferenceFailure, InferenceLifecycleContext,
    InferenceLifecycleEvent, InferenceLifecycleObserver, InferenceRejection,
    InferenceTerminalResult, failure_for_status,
};

pub(crate) struct RequestLifecycle {
    observer: Option<Arc<dyn InferenceLifecycleObserver>>,
    context: InferenceLifecycleContext,
    terminal_or_transferred: bool,
}

impl RequestLifecycle {
    pub(crate) fn admit(
        observer: Option<Arc<dyn InferenceLifecycleObserver>>,
        context: InferenceLifecycleContext,
    ) -> Self {
        if let Some(observer) = &observer {
            observer.observe(&InferenceLifecycleEvent::Admitted {
                context: context.clone(),
            });
        }
        Self {
            observer,
            context,
            terminal_or_transferred: false,
        }
    }

    pub(crate) fn finish_with_usage(
        &mut self,
        status: StatusCode,
        usage: Option<TokenUsage>,
        exchange_id: Option<String>,
    ) {
        if self.terminal_or_transferred {
            return;
        }
        self.terminal_or_transferred = true;
        let event = if status.as_u16() == CLIENT_CLOSED_REQUEST_STATUS {
            InferenceLifecycleEvent::NonStreamTerminal {
                context: self.context.clone(),
                result: InferenceTerminalResult::Failed {
                    status_code: CLIENT_CLOSED_REQUEST_STATUS,
                    failure: InferenceFailure::Cancelled,
                },
                exchange_id,
            }
        } else if status.is_client_error() {
            InferenceLifecycleEvent::Rejected {
                context: self.context.clone(),
                status_code: status.as_u16(),
                rejection: rejection_for_status(status),
            }
        } else {
            let result = if status.is_server_error() {
                InferenceTerminalResult::Failed {
                    status_code: status.as_u16(),
                    failure: failure_for_status(status),
                }
            } else if let Some(usage) = usage {
                InferenceTerminalResult::CompletedWithUsage {
                    status_code: status.as_u16(),
                    usage,
                }
            } else {
                InferenceTerminalResult::Completed {
                    status_code: status.as_u16(),
                }
            };
            InferenceLifecycleEvent::NonStreamTerminal {
                context: self.context.clone(),
                result,
                exchange_id,
            }
        };
        self.observe(&event);
    }

    pub(crate) fn transfer_to_stream(&mut self) {
        self.terminal_or_transferred = true;
    }

    fn observe(&self, event: &InferenceLifecycleEvent) {
        if let Some(observer) = &self.observer {
            observer.observe(event);
        }
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        if self.terminal_or_transferred {
            return;
        }
        self.terminal_or_transferred = true;
        self.observe(&InferenceLifecycleEvent::RequestCancelled {
            context: self.context.clone(),
        });
    }
}

fn rejection_for_status(status: StatusCode) -> InferenceRejection {
    match status {
        StatusCode::PAYLOAD_TOO_LARGE => InferenceRejection::PayloadTooLarge,
        StatusCode::METHOD_NOT_ALLOWED => InferenceRejection::MethodNotAllowed,
        StatusCode::NOT_FOUND => InferenceRejection::NotFound,
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => InferenceRejection::AdmissionDenied,
        _ => InferenceRejection::InvalidRequest,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

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
            parse_request_id("54c2252a-0ce7-41e8-9884-897e902e2df5").expect("request ID"),
            InferenceRequestMethod::Post,
            InferenceFrontendRoute::ChatCompletions,
        )
    }

    #[test]
    fn dropped_request_future_is_terminalized_once_as_cancelled() {
        let observer = Arc::new(RecordingObserver::default());
        {
            let _lifecycle = RequestLifecycle::admit(Some(observer.clone()), context());
        }

        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [
                InferenceLifecycleEvent::Admitted { .. },
                InferenceLifecycleEvent::RequestCancelled { .. },
            ]
        ));
    }

    #[test]
    fn stream_transfer_prevents_request_scope_terminal_duplication() {
        let observer = Arc::new(RecordingObserver::default());
        {
            let mut lifecycle = RequestLifecycle::admit(Some(observer.clone()), context());
            lifecycle.transfer_to_stream();
        }

        let events = observer.0.lock().expect("observer lock");
        assert!(matches!(
            events.as_slice(),
            [InferenceLifecycleEvent::Admitted { .. }]
        ));
    }

    #[test]
    fn client_closed_status_is_cancelled_instead_of_rejected() {
        let observer = Arc::new(RecordingObserver::default());
        let mut lifecycle = RequestLifecycle::admit(Some(observer.clone()), context());

        lifecycle.finish_with_usage(crate::lifecycle::client_closed_request_status(), None, None);

        assert!(matches!(
            observer.0.lock().expect("observer lock").as_slice(),
            [
                InferenceLifecycleEvent::Admitted { .. },
                InferenceLifecycleEvent::NonStreamTerminal {
                    result: InferenceTerminalResult::Failed {
                        status_code: CLIENT_CLOSED_REQUEST_STATUS,
                        failure: InferenceFailure::Cancelled,
                    },
                    ..
                },
            ]
        ));
    }
}
