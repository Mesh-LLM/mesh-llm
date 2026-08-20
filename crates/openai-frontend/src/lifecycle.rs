//! Dependency-safe lifecycle observation contracts for OpenAI ingress.
//!
//! The frontend owns request correlation and classifies request boundaries;
//! runtimes provide an observer that persists or forwards the metadata. These
//! types intentionally have no request or response payload fields.

use axum::http::{HeaderMap, HeaderValue, StatusCode, header::HeaderName};
use mesh_llm_events::logging::events::TokenUsage;
pub use mesh_llm_events::logging::identifiers::RequestId;
use mesh_llm_events::logging::lifecycle::LifecycleState;
use uuid::Uuid;

use crate::{common::Usage, errors::OpenAiError};

/// The canonical request correlation header used by the OpenAI frontend.
pub static REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

/// A caller-supplied fresh, unpredictable per-request value. Presence alone
/// buys equivocation resistance (a downstream recorder can no longer fabricate
/// or replay the freshness value) — it does not establish who sent it.
pub static CLIENT_NONCE_HEADER: HeaderName = HeaderName::from_static("x-capsule-client-nonce");

/// Set only when this ingress minted the nonce itself, never when forwarding
/// a value the inbound request already carried. Without this marker, "the
/// harness sent a nonce" and "the ingress minted one on the harness's behalf"
/// are indistinguishable once the header is present downstream.
pub static CLIENT_NONCE_ORIGIN_HEADER: HeaderName =
    HeaderName::from_static("x-capsule-nonce-origin");

/// The [`CLIENT_NONCE_ORIGIN_HEADER`] value stamped when this ingress minted
/// the nonce rather than forwarding one the client already supplied.
pub const CLIENT_NONCE_ORIGIN_LOCAL_INGRESS: &str = "local_ingress";

/// Metadata that identifies a frontend request without retaining its payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiLifecycleContext {
    pub request_id: RequestId,
    pub method: OpenAiRequestMethod,
    pub route: OpenAiFrontendRoute,
}

impl OpenAiLifecycleContext {
    pub const fn new(
        request_id: RequestId,
        method: OpenAiRequestMethod,
        route: OpenAiFrontendRoute,
    ) -> Self {
        Self {
            request_id,
            method,
            route,
        }
    }
}

/// A bounded HTTP method classification for lifecycle metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiRequestMethod {
    Get,
    Post,
    Other,
}

/// A bounded frontend route classification for lifecycle metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiFrontendRoute {
    Health,
    Healthz,
    Readyz,
    Models,
    ChatCompletions,
    Completions,
    Responses,
    Unknown,
}

/// The backend operation dispatched by a frontend route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiBackendOperation {
    Models,
    ChatCompletion,
    ChatCompletionStream,
    Completion,
    CompletionStream,
    Responses,
    ResponsesStream,
}

/// A bounded classification for a request rejected before backend execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiRejection {
    InvalidRequest,
    PayloadTooLarge,
    MethodNotAllowed,
    NotFound,
    AdmissionDenied,
}

/// A bounded classification for a terminal backend failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiFailure {
    Backend,
    Timeout,
    Internal,
    Cancelled,
}

/// Numeric-only token accounting attached to a completed response.
///
/// This intentionally cannot retain prompts, completions, model labels, or
/// arbitrary backend metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<&Usage> for OpenAiUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

/// A typed terminal outcome for non-streaming execution and stream completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiTerminalResult {
    Completed {
        status_code: u16,
    },
    CompletedWithUsage {
        status_code: u16,
        usage: TokenUsage,
    },
    Failed {
        status_code: u16,
        failure: OpenAiFailure,
    },
}

impl OpenAiTerminalResult {
    /// Return the shared lifecycle state corresponding to this terminal result.
    pub const fn lifecycle_state(self) -> LifecycleState {
        match self {
            Self::Completed { .. } | Self::CompletedWithUsage { .. } => LifecycleState::Completed,
            Self::Failed { .. } => LifecycleState::Failed,
        }
    }
}

/// Metadata-only lifecycle events emitted by a frontend ingress owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenAiLifecycleEvent {
    Admitted {
        context: OpenAiLifecycleContext,
    },
    Rejected {
        context: OpenAiLifecycleContext,
        status_code: u16,
        rejection: OpenAiRejection,
    },
    BackendDispatched {
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
    },
    BackendTerminal {
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
        result: OpenAiTerminalResult,
    },
    StreamFirstItem {
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
    },
    ResponseCompleted {
        context: OpenAiLifecycleContext,
        operation: OpenAiBackendOperation,
        usage: OpenAiUsage,
    },
    NonStreamTerminal {
        context: OpenAiLifecycleContext,
        result: OpenAiTerminalResult,
    },
    StreamTerminal {
        context: OpenAiLifecycleContext,
        result: OpenAiTerminalResult,
    },
    StreamDropped {
        context: OpenAiLifecycleContext,
    },
    StreamCancelled {
        context: OpenAiLifecycleContext,
    },
    RequestCancelled {
        context: OpenAiLifecycleContext,
    },
}

/// Receives metadata-only lifecycle events from the owning frontend ingress.
///
/// Implementations must remain non-blocking for request serving. The frontend
/// deliberately does not prescribe persistence, capture, or runtime adapters.
pub trait OpenAiLifecycleObserver: Send + Sync + 'static {
    fn observe(&self, event: &OpenAiLifecycleEvent);
}

/// Parse an inbound request identifier only when it is a valid UUID.
pub fn parse_request_id(value: &str) -> Option<RequestId> {
    Uuid::parse_str(value).ok().map(RequestId::from)
}

/// Parse one canonical request ID from a header-like value sequence.
///
/// A missing value, malformed value, or duplicate header is rejected. The
/// `Option<&str>` item shape lets byte-oriented ingress preserve invalid UTF-8
/// as an invalid value instead of silently treating it as absent.
pub fn parse_single_request_id<'a, I>(values: I) -> Option<RequestId>
where
    I: IntoIterator<Item = Option<&'a str>>,
{
    let mut values = values.into_iter();
    let value = values.next()??;
    if values.next().is_some() {
        return None;
    }
    parse_request_id(value)
}

/// Parse the canonical inbound request identifier header only when it is a valid UUID.
pub fn parse_request_id_header(headers: &HeaderMap) -> Option<RequestId> {
    headers
        .get(&REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_request_id)
}

/// Generate a fresh canonical UUID request identifier.
pub fn generate_request_id() -> RequestId {
    RequestId::new()
}

/// Reuse a valid inbound UUID request ID or generate a replacement identifier.
pub fn request_id_from_headers_or_generate(headers: &HeaderMap) -> RequestId {
    parse_request_id_header(headers).unwrap_or_else(generate_request_id)
}

/// Construct the response header that propagates the canonical request identifier.
pub fn request_id_response_header(request_id: &RequestId) -> (HeaderName, HeaderValue) {
    let value = HeaderValue::from_str(&request_id.as_ref().hyphenated().to_string())
        .expect("a UUID is always a valid x-request-id header value");
    (REQUEST_ID_HEADER.clone(), value)
}

/// Forward an inbound client nonce unchanged, or mint a fresh CSPRNG UUIDv4
/// when the header is absent — never reused across requests, never derived
/// from a counter, timestamp, or session. Returns the value to forward and,
/// only when this call minted it, the origin-marker value to attach alongside
/// it so a downstream reader can tell the two cases apart.
pub fn client_nonce_from_headers_or_generate(
    headers: &HeaderMap,
) -> (HeaderValue, Option<HeaderValue>) {
    match headers.get(&CLIENT_NONCE_HEADER) {
        Some(value) => (value.clone(), None),
        None => {
            let minted = HeaderValue::from_str(&Uuid::new_v4().to_string())
                .expect("a UUID is always a valid header value");
            let origin = HeaderValue::from_static(CLIENT_NONCE_ORIGIN_LOCAL_INGRESS);
            (minted, Some(origin))
        }
    }
}

pub(crate) const CLIENT_CLOSED_REQUEST_STATUS: u16 = 499;

pub(crate) fn client_closed_request_status() -> StatusCode {
    StatusCode::from_u16(CLIENT_CLOSED_REQUEST_STATUS)
        .expect("the client-closed status is a valid HTTP status")
}

pub(crate) fn failure_for_status(status: StatusCode) -> OpenAiFailure {
    match status {
        StatusCode::GATEWAY_TIMEOUT => OpenAiFailure::Timeout,
        StatusCode::INTERNAL_SERVER_ERROR => OpenAiFailure::Internal,
        _ => OpenAiFailure::Backend,
    }
}

pub(crate) fn terminal_result_for_error(error: &OpenAiError) -> OpenAiTerminalResult {
    OpenAiTerminalResult::Failed {
        status_code: error.status().as_u16(),
        failure: if error.status().as_u16() == CLIENT_CLOSED_REQUEST_STATUS {
            OpenAiFailure::Cancelled
        } else {
            failure_for_status(error.status())
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    const REQUEST_ID: &str = "c0a801ef-2a39-4f52-99f5-bdc849127cde";

    #[test]
    fn valid_uuid_header_is_reused_and_propagated() {
        let mut headers = HeaderMap::new();
        headers.insert(
            REQUEST_ID_HEADER.clone(),
            HeaderValue::from_static(REQUEST_ID),
        );

        let request_id = request_id_from_headers_or_generate(&headers);
        assert_eq!(request_id.as_ref().to_string(), REQUEST_ID);

        let (name, value) = request_id_response_header(&request_id);
        assert_eq!(name, REQUEST_ID_HEADER);
        assert_eq!(value, HeaderValue::from_static(REQUEST_ID));
    }

    #[test]
    fn invalid_or_missing_header_generates_a_uuid() {
        let mut invalid = HeaderMap::new();
        invalid.insert(
            REQUEST_ID_HEADER.clone(),
            HeaderValue::from_static("client-request-42"),
        );

        let invalid_id = request_id_from_headers_or_generate(&invalid);
        assert_ne!(invalid_id.as_ref().to_string(), "client-request-42");
        assert!(Uuid::parse_str(&invalid_id.as_ref().to_string()).is_ok());

        let missing_id = request_id_from_headers_or_generate(&HeaderMap::new());
        assert!(Uuid::parse_str(&missing_id.as_ref().to_string()).is_ok());
    }

    #[test]
    fn parse_single_request_id_golden_table_rejects_missing_invalid_and_duplicate_values() {
        let valid = REQUEST_ID;
        let cases = [
            (vec![Some(valid)], true),
            (vec![Some("not-a-uuid")], false),
            (Vec::new(), false),
            (vec![Some(valid), Some(valid)], false),
            (vec![None], false),
            (vec![None, Some(valid)], false),
        ];

        for (values, expected) in cases {
            assert_eq!(
                parse_single_request_id(values),
                expected.then(|| parse_request_id(valid).unwrap())
            );
        }
    }

    #[test]
    fn client_nonce_already_present_is_forwarded_unchanged() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CLIENT_NONCE_HEADER.clone(),
            HeaderValue::from_static("harness-supplied-nonce"),
        );

        let (value, origin) = client_nonce_from_headers_or_generate(&headers);
        assert_eq!(value, HeaderValue::from_static("harness-supplied-nonce"));
        assert!(
            origin.is_none(),
            "forwarding a caller-supplied nonce must not add an origin marker"
        );
    }

    #[test]
    fn client_nonce_absent_is_minted_and_marked_local_ingress() {
        let (value, origin) = client_nonce_from_headers_or_generate(&HeaderMap::new());

        assert!(Uuid::parse_str(value.to_str().expect("minted nonce is ASCII")).is_ok());
        assert_eq!(
            origin,
            Some(HeaderValue::from_static(CLIENT_NONCE_ORIGIN_LOCAL_INGRESS))
        );
    }

    #[test]
    fn client_nonce_minting_is_fresh_every_call() {
        let (first, _) = client_nonce_from_headers_or_generate(&HeaderMap::new());
        let (second, _) = client_nonce_from_headers_or_generate(&HeaderMap::new());
        assert_ne!(first, second, "a nonce reused across requests buys nothing");
    }

    #[test]
    fn lifecycle_events_keep_context_and_terminal_results_typed() {
        let context = OpenAiLifecycleContext::new(
            parse_request_id(REQUEST_ID).expect("test UUID should parse"),
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::ChatCompletions,
        );
        let event = OpenAiLifecycleEvent::NonStreamTerminal {
            context: context.clone(),
            result: OpenAiTerminalResult::Failed {
                status_code: 504,
                failure: OpenAiFailure::Timeout,
            },
        };

        assert!(matches!(
            event,
            OpenAiLifecycleEvent::NonStreamTerminal {
                context: OpenAiLifecycleContext {
                    route: OpenAiFrontendRoute::ChatCompletions,
                    ..
                },
                result: OpenAiTerminalResult::Failed {
                    status_code: 504,
                    failure: OpenAiFailure::Timeout,
                },
            }
        ));
        assert_eq!(
            OpenAiTerminalResult::Completed { status_code: 200 }.lifecycle_state(),
            LifecycleState::Completed
        );
        assert_eq!(
            OpenAiTerminalResult::Failed {
                status_code: 502,
                failure: OpenAiFailure::Backend,
            }
            .lifecycle_state(),
            LifecycleState::Failed
        );
        assert_eq!(context.request_id.as_ref().to_string(), REQUEST_ID);
    }

    #[test]
    fn usage_projection_contains_only_numeric_counts_and_cached_tokens() {
        let usage = OpenAiUsage::from(&Usage::new(17, 4).with_cached_tokens(11));

        assert_eq!(
            usage,
            OpenAiUsage {
                prompt_tokens: 17,
                cached_tokens: 11,
                completion_tokens: 4,
                total_tokens: 21,
            }
        );
    }

    #[test]
    fn observer_receives_metadata_only_stream_drop_and_cancel_events() {
        struct RecordingObserver(Mutex<Vec<OpenAiLifecycleEvent>>);

        impl OpenAiLifecycleObserver for RecordingObserver {
            fn observe(&self, event: &OpenAiLifecycleEvent) {
                self.0
                    .lock()
                    .expect("test observer lock poisoned")
                    .push(event.clone());
            }
        }

        let context = OpenAiLifecycleContext::new(
            parse_request_id(REQUEST_ID).expect("test UUID should parse"),
            OpenAiRequestMethod::Post,
            OpenAiFrontendRoute::Responses,
        );
        let observer = RecordingObserver(Mutex::new(Vec::new()));
        observer.observe(&OpenAiLifecycleEvent::StreamDropped {
            context: context.clone(),
        });
        observer.observe(&OpenAiLifecycleEvent::StreamCancelled { context });

        assert!(matches!(
            observer
                .0
                .lock()
                .expect("test observer lock poisoned")
                .as_slice(),
            [
                OpenAiLifecycleEvent::StreamDropped { .. },
                OpenAiLifecycleEvent::StreamCancelled { .. }
            ]
        ));
    }

    #[test]
    fn manifest_has_no_host_runtime_dependency() {
        let manifest = include_str!("../Cargo.toml");
        assert!(
            !manifest.contains("mesh-llm-host-runtime"),
            "openai-frontend must not depend on mesh-llm-host-runtime"
        );
    }
}
