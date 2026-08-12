//! Lifecycle event payloads for canonical logging envelopes.

use serde::{Deserialize, Serialize};

use super::identifiers::AttemptId;

/// Lifecycle event payloads carried inside [`CanonicalEnvelope`].
///
/// Bounded metadata only — never raw request/response payloads or secrets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LifecycleEvent {
    /// Request admitted into the system.
    Admitted {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        method: Option<String>,
    },
    /// A route was selected for the request.
    RouteSelected {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        engine: Option<String>,
    },
    /// A transport attempt started.
    AttemptStarted {
        #[serde(skip_serializing_if = "Option::is_none")]
        attempt_id: Option<AttemptId>,
    },
    /// A transport attempt completed.
    AttemptCompleted {
        #[serde(skip_serializing_if = "Option::is_none")]
        attempt_id: Option<AttemptId>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status_code: Option<u16>,
    },
    /// A transport attempt failed.
    AttemptFailed {
        #[serde(skip_serializing_if = "Option::is_none")]
        attempt_id: Option<AttemptId>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A streaming response started.
    StreamStarted {
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// A streaming chunk was produced.
    StreamChunk {
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<u64>,
    },
    /// A stream completed successfully.
    StreamCompleted {
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<u64>,
    },
    /// A stream errored.
    StreamError {
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// A bounded, sanitized operational error emitted by the logging service
    /// itself. These are carried on the System replay channel and are never
    /// treated as request terminal outcomes.
    AuditError { message: String },
    /// Request completed successfully.
    Completed {
        #[serde(skip_serializing_if = "Option::is_none")]
        status_code: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    /// Request failed.
    Failed { error: String },
    /// Request rejected before processing.
    Rejected {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Request cancelled.
    Cancelled {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Request dropped without terminal processing.
    Dropped {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}
