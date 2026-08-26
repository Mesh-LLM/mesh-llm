//! Lifecycle event payloads for canonical logging envelopes.

use serde::{Deserialize, Serialize};

use super::identifiers::AttemptId;

/// Upstream token counts exactly as reported by a compatible API. Each field
/// remains absent when upstream omitted it; consumers must never estimate it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

impl TokenUsage {
    /// Construct an authoritative usage record only when the provider supplied
    /// every count and the total is arithmetically consistent. Missing,
    /// overflowing, or internally inconsistent usage must not be estimated.
    pub fn from_counts(
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        total_tokens: Option<u64>,
    ) -> Option<Self> {
        let (Some(prompt_tokens), Some(completion_tokens), Some(total_tokens)) =
            (prompt_tokens, completion_tokens, total_tokens)
        else {
            return None;
        };
        if prompt_tokens.checked_add(completion_tokens) != Some(total_tokens) {
            return None;
        }
        Some(Self {
            prompt_tokens: Some(prompt_tokens),
            cached_prompt_tokens: None,
            completion_tokens: Some(completion_tokens),
            total_tokens: Some(total_tokens),
        })
    }

    pub fn with_cached_prompt_tokens(mut self, cached_prompt_tokens: Option<u64>) -> Self {
        self.cached_prompt_tokens = cached_prompt_tokens;
        self
    }
}

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
    /// The first item became available from a streaming backend.
    ///
    /// This is not a first-token event: the item may be usage-only or an
    /// error, and it remains distinct from client-visible stream chunks.
    BackendStreamFirstItem,
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
        /// Legacy compatibility field containing completion tokens only.
        #[serde(skip_serializing_if = "Option::is_none")]
        tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    /// Bounded token accounting reported by an OpenAI-compatible backend.
    UsageRecorded {
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cached_prompt_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        completion_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total_tokens: Option<u64>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    /// Request failed.
    Failed {
        error: String,
        /// The terminal upstream HTTP status when one was received.
        #[serde(skip_serializing_if = "Option::is_none")]
        status_code: Option<u16>,
    },
    /// Request rejected before processing.
    Rejected {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// The terminal upstream HTTP status when one was received.
        #[serde(skip_serializing_if = "Option::is_none")]
        status_code: Option<u16>,
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
