//! Generation results and ordered stream events. They contain inference data,
//! not HTTP envelopes: protocol adapters own object tags and serialization.

use crate::common::{FinishReason, Usage, completion_id, now_unix_secs};
use serde_json::Value;
use std::collections::BTreeMap;

/// Completed generation, including metadata consumed by lifecycle hooks.
/// Constructors keep callers independent of future metadata additions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GenerationResponse {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<GenerationChoice>,
    pub usage: Usage,

    pub timings: Option<BTreeMap<String, Value>>,
    /// Response attestation metadata consumed by the frontend header adapter.
    pub capsule_marker: Option<CapsuleMarker>,
    /// Host-minted exchange identity shared with the terminal hook and frontend.
    pub exchange_id: Option<String>,
}

/// A rung-ladder response-leg marker: the `capsule_id` written into the
/// response as `X-Capsule-Id`, and the `nonce` the marker is correlated
/// against, so an out-of-process plugin observing the terminal event (which
/// carries this marker) knows what a later client ack must sign over to lift
/// `acknowledged_receipt` -> `full_bilateral` (see
/// `capsule-emit-mesh/README.md` "Bilateral attestation demo", move 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleMarker {
    pub capsule_id: String,
    pub nonce: String,
}

/// Whether `id` is safe to publish as both an `X-Capsule-Id` response header
/// (set by `frontend_lifecycle_middleware` in `router.rs`) and a mesh
/// plugin's terminal-event `capsule_id` — the single check that keeps those
/// two views of the same exchange from disagreeing. The router already
/// silently drops the header on a value `HeaderValue::from_str` rejects; a
/// hook-minted marker that fails this same check must not be attached to the
/// response at all, so a plugin observing the terminal event never sees a
/// capsule id the client's response never carried.
pub fn capsule_id_is_valid(id: &str) -> bool {
    !id.is_empty() && axum::http::HeaderValue::from_str(id).is_ok()
}

impl GenerationResponse {
    pub fn new(model: impl Into<String>, content: impl Into<String>, usage: Usage) -> Self {
        Self::new_with_reason(model, content, usage, FinishReason::Stop)
    }

    pub fn new_with_reason(
        model: impl Into<String>,
        content: impl Into<String>,
        usage: Usage,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![GenerationChoice {
                index: 0,
                message: GeneratedMessage {
                    role: "assistant",
                    content: Some(content.into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage,
            timings: None,
            capsule_marker: None,
            exchange_id: None,
        }
    }

    /// Construct a response with an explicit id/creation time/choice set —
    /// for callers (e.g. a backend reconstructing a parsed generation) that
    /// need full control over those fields, unlike [`Self::new`]/
    /// [`Self::new_with_reason`], which always mint a fresh id and a single
    /// choice. `capsule_marker` always starts `None`; it is attached later,
    /// by [`crate::hooks::HookedInferenceBackend`]. Exists so a downstream
    /// crate that previously built this via struct literal has a
    /// `#[non_exhaustive]`-safe replacement.
    pub fn from_parts(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        choices: Vec<GenerationChoice>,
        usage: Usage,
        timings: Option<BTreeMap<String, Value>>,
    ) -> Self {
        Self {
            id: id.into(),
            created,
            model: model.into(),
            choices,
            usage,
            timings,
            capsule_marker: None,
            exchange_id: None,
        }
    }

    pub fn with_timings(mut self, timings: Option<BTreeMap<String, Value>>) -> Self {
        self.timings = timings;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationChoice {
    pub index: u32,
    pub message: GeneratedMessage,

    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedMessage {
    pub role: &'static str,
    pub content: Option<String>,

    pub reasoning_content: Option<String>,

    pub tool_calls: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationEvent {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<GenerationEventChoice>,

    pub usage: Option<Usage>,

    pub timings: Option<BTreeMap<String, Value>>,
}

impl GenerationEvent {
    pub fn role(model: impl Into<String>) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![GenerationEventChoice {
                index: 0,
                delta: GenerationDelta {
                    role: Some("assistant"),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
            timings: None,
        }
    }

    pub fn delta(model: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![GenerationEventChoice {
                index: 0,
                delta: GenerationDelta {
                    role: None,
                    content: Some(content.into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
            timings: None,
        }
    }

    pub fn done(model: impl Into<String>) -> Self {
        Self::done_with_reason(model, FinishReason::Stop)
    }

    pub fn done_with_reason(model: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            created: now_unix_secs(),
            model: model.into(),
            choices: vec![GenerationEventChoice {
                index: 0,
                delta: GenerationDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: None,
            timings: None,
        }
    }

    pub fn usage(model: impl Into<String>, usage: Usage) -> Self {
        Self {
            id: completion_id("chatcmpl"),
            created: now_unix_secs(),
            model: model.into(),
            choices: Vec::new(),
            usage: Some(usage),
            timings: None,
        }
    }

    pub fn usage_with_timings(
        model: impl Into<String>,
        usage: Usage,
        timings: Option<BTreeMap<String, Value>>,
    ) -> Self {
        Self {
            timings,
            ..Self::usage(model, usage)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationEventChoice {
    pub index: u32,
    pub delta: GenerationDelta,

    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationDelta {
    pub role: Option<&'static str>,

    pub content: Option<String>,

    pub reasoning_content: Option<String>,

    pub tool_calls: Option<Value>,
}
