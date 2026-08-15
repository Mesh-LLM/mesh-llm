use super::envelope::{CanonicalEnvelope, CanonicalPresentationContext};
use super::events::LifecycleEvent;
use crate::OutputLevel;

/// The compact, payload-free vocabulary used by terminal and JSONL output.
///
/// Trusted local output retains bounded correlation metadata (event and request
/// IDs, replay channel/sequence, numeric lifecycle counters, and closed route /
/// source / destination classifications). Identity fields, artifacts, model
/// input/output, credentials, and free-form error detail deliberately never
/// cross this presentation boundary. Network and telemetry projections remain
/// stricter and do not use these local IDs.
impl CanonicalEnvelope {
    pub fn presentation_event_name(&self) -> &'static str {
        match self.event {
            LifecycleEvent::Admitted { .. } => "request_admitted",
            LifecycleEvent::RouteSelected { .. } => "request_route_selected",
            LifecycleEvent::AttemptStarted { .. } => "request_attempt_started",
            LifecycleEvent::AttemptCompleted { .. } => "request_attempt_completed",
            LifecycleEvent::AttemptFailed { .. } => "request_attempt_failed",
            LifecycleEvent::BackendStreamFirstItem => "request_backend_stream_first_item",
            LifecycleEvent::StreamStarted { .. } => "request_stream_started",
            LifecycleEvent::StreamChunk { .. } => "request_stream_chunk",
            LifecycleEvent::StreamCompleted { .. } => "request_stream_completed",
            LifecycleEvent::UsageRecorded { .. } => "request_usage_recorded",
            LifecycleEvent::StreamError { .. } => "request_stream_error",
            LifecycleEvent::AuditError { .. } => "logging_audit_error",
            LifecycleEvent::Completed { .. } => "request_completed",
            LifecycleEvent::Failed { .. } => "request_failed",
            LifecycleEvent::Rejected { .. } => "request_rejected",
            LifecycleEvent::Cancelled { .. } => "request_cancelled",
            LifecycleEvent::Dropped { .. } => "request_dropped",
        }
    }

    pub fn presentation_level(&self) -> OutputLevel {
        match self.event {
            LifecycleEvent::AttemptFailed { .. }
            | LifecycleEvent::StreamError { .. }
            | LifecycleEvent::AuditError { .. }
            | LifecycleEvent::Failed { .. }
            | LifecycleEvent::Rejected { .. }
            | LifecycleEvent::Cancelled { .. }
            | LifecycleEvent::Dropped { .. } => OutputLevel::Warn,
            LifecycleEvent::Admitted { .. }
            | LifecycleEvent::RouteSelected { .. }
            | LifecycleEvent::AttemptStarted { .. }
            | LifecycleEvent::AttemptCompleted { .. }
            | LifecycleEvent::BackendStreamFirstItem
            | LifecycleEvent::StreamStarted { .. }
            | LifecycleEvent::StreamChunk { .. }
            | LifecycleEvent::StreamCompleted { .. }
            | LifecycleEvent::UsageRecorded { .. }
            | LifecycleEvent::Completed { .. } => OutputLevel::Info,
        }
    }

    pub fn presentation_message(&self) -> String {
        match self.event {
            LifecycleEvent::AuditError { .. } => "logging audit warning".to_string(),
            LifecycleEvent::Admitted { .. } => append_context("request admitted", self),
            LifecycleEvent::RouteSelected { .. } => append_context("request route selected", self),
            LifecycleEvent::AttemptStarted { .. } => {
                append_context("request attempt started", self)
            }
            LifecycleEvent::AttemptCompleted { status_code, .. } => append_status(
                append_context("request attempt completed", self),
                status_code,
            ),
            LifecycleEvent::AttemptFailed { .. } => append_context("request attempt failed", self),
            LifecycleEvent::BackendStreamFirstItem => {
                append_context("request backend stream first item", self)
            }
            LifecycleEvent::StreamStarted { .. } => append_context("request stream started", self),
            LifecycleEvent::StreamChunk { .. } => append_context("request stream chunk", self),
            LifecycleEvent::StreamCompleted { .. } => {
                append_context("request stream completed", self)
            }
            LifecycleEvent::UsageRecorded { .. } => append_context("request usage recorded", self),
            LifecycleEvent::StreamError { .. } => append_context("request stream failed", self),
            LifecycleEvent::Completed {
                status_code,
                duration_ms,
                ..
            } => append_duration(
                append_status(append_context("request completed", self), status_code),
                duration_ms,
            ),
            LifecycleEvent::Failed { .. } => append_context("request failed", self),
            LifecycleEvent::Rejected { .. } => append_context("request rejected", self),
            LifecycleEvent::Cancelled { .. } => append_context("request cancelled", self),
            LifecycleEvent::Dropped { .. } => append_context("request dropped", self),
        }
    }

    /// A bounded local-console summary with stable correlation metadata.
    ///
    /// This is intentionally for JSONL/pretty/TUI presentation only. It does
    /// not include identity fields, artifacts, free-form payloads, or secrets.
    pub fn presentation_local_summary(&self) -> String {
        self.presentation_local_summary_with_limit(DEFAULT_PRESENTATION_SUMMARY_LIMIT)
    }

    /// A deterministic, character-bounded local presentation summary.
    ///
    /// The source summary is already payload-free. Limiting happens only after
    /// that safe projection has been constructed, so callers cannot accidentally
    /// truncate and expose a raw lifecycle payload instead.
    pub fn presentation_local_summary_with_limit(&self, limit: usize) -> String {
        let mut message = format!(
            "{} request_id={} event_id={} channel={} sequence={}",
            self.presentation_message(),
            self.request_id.as_uuid(),
            self.event_id.as_uuid(),
            presentation_channel_name(self.channel),
            self.sequence,
        );
        if let Some(tokens) = self.presentation_token_count() {
            message.push_str(&format!(" tokens={tokens}"));
        }
        truncate_presentation_summary(message, limit)
    }

    /// Numeric token counters are safe local operational metadata; token
    /// content is never represented by canonical lifecycle events.
    pub fn presentation_token_count(&self) -> Option<u64> {
        match self.event {
            LifecycleEvent::StreamChunk { tokens }
            | LifecycleEvent::StreamCompleted { tokens, .. } => tokens,
            LifecycleEvent::UsageRecorded { total_tokens, .. } => total_tokens,
            LifecycleEvent::Admitted { .. }
            | LifecycleEvent::RouteSelected { .. }
            | LifecycleEvent::AttemptStarted { .. }
            | LifecycleEvent::AttemptCompleted { .. }
            | LifecycleEvent::AttemptFailed { .. }
            | LifecycleEvent::BackendStreamFirstItem
            | LifecycleEvent::StreamStarted { .. }
            | LifecycleEvent::StreamError { .. }
            | LifecycleEvent::AuditError { .. }
            | LifecycleEvent::Completed { .. }
            | LifecycleEvent::Failed { .. }
            | LifecycleEvent::Rejected { .. }
            | LifecycleEvent::Cancelled { .. }
            | LifecycleEvent::Dropped { .. } => None,
        }
    }

    pub fn presentation_outcome(&self) -> Option<&'static str> {
        match self.event {
            LifecycleEvent::Completed { .. } => Some("completed"),
            LifecycleEvent::Failed { .. } => Some("failed"),
            LifecycleEvent::Rejected { .. } => Some("rejected"),
            LifecycleEvent::Cancelled { .. } => Some("cancelled"),
            LifecycleEvent::Dropped { .. } => Some("dropped"),
            LifecycleEvent::Admitted { .. }
            | LifecycleEvent::RouteSelected { .. }
            | LifecycleEvent::AttemptStarted { .. }
            | LifecycleEvent::AttemptCompleted { .. }
            | LifecycleEvent::AttemptFailed { .. }
            | LifecycleEvent::BackendStreamFirstItem
            | LifecycleEvent::StreamStarted { .. }
            | LifecycleEvent::StreamChunk { .. }
            | LifecycleEvent::StreamCompleted { .. }
            | LifecycleEvent::StreamError { .. }
            | LifecycleEvent::UsageRecorded { .. }
            | LifecycleEvent::AuditError { .. } => None,
        }
    }

    /// Closed request classification used by local pretty, JSONL, and TUI
    /// projections. Sparse/legacy envelopes truthfully fall back to unknown.
    pub fn presentation_request_kind(&self) -> &'static str {
        self.presentation_context
            .as_ref()
            .map_or("unknown", CanonicalPresentationContext::request_kind)
    }

    pub fn presentation_route(&self) -> Option<&str> {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::route)
    }

    pub fn presentation_source(&self) -> &str {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::source)
            .unwrap_or("unknown")
    }

    pub fn presentation_model(&self) -> Option<&str> {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::model)
            .or(match &self.event {
                LifecycleEvent::Admitted { model, .. }
                | LifecycleEvent::RouteSelected { model, .. }
                | LifecycleEvent::StreamStarted { model } => model.as_deref(),
                _ => None,
            })
    }

    pub fn presentation_provider(&self) -> Option<&str> {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::provider)
            .or(match &self.event {
                LifecycleEvent::RouteSelected { provider, .. } => provider.as_deref(),
                _ => None,
            })
    }

    pub fn presentation_engine(&self) -> Option<&str> {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::engine)
            .or(match &self.event {
                LifecycleEvent::RouteSelected { engine, .. } => engine.as_deref(),
                _ => None,
            })
    }

    pub fn presentation_method(&self) -> Option<&str> {
        self.presentation_context
            .as_ref()
            .and_then(CanonicalPresentationContext::method)
            .or(match &self.event {
                LifecycleEvent::Admitted { method, .. } => method.as_deref(),
                _ => None,
            })
    }
}

/// Default local presentation bound. Host configuration may supply a stricter
/// or larger validated limit at runtime.
pub const DEFAULT_PRESENTATION_SUMMARY_LIMIT: usize = 2_048;

fn truncate_presentation_summary(summary: String, limit: usize) -> String {
    summary.chars().take(limit).collect()
}

fn append_status(mut message: String, status_code: Option<u16>) -> String {
    if let Some(status_code) = status_code {
        message.push_str(&format!(" status={status_code}"));
    }
    message
}

fn append_duration(mut message: String, duration_ms: Option<u64>) -> String {
    if let Some(duration_ms) = duration_ms {
        message.push_str(&format!(" duration={duration_ms}ms"));
    }
    message
}

fn append_context(message: impl Into<String>, envelope: &CanonicalEnvelope) -> String {
    let mut message = contextual_phase_prefix(message.into(), envelope);
    if envelope.presentation_context().is_some() {
        if envelope.presentation_request_kind() == "unknown" {
            message.push_str(" kind=unknown");
        }
        if let Some(route) = envelope.presentation_route() {
            message.push_str(" route=");
            message.push_str(route);
        }
        if envelope.presentation_source() != "unknown" {
            message.push_str(" source=");
            message.push_str(envelope.presentation_source());
        }
    }
    for (key, value) in [
        ("model", envelope.presentation_model()),
        ("provider", envelope.presentation_provider()),
        ("engine", envelope.presentation_engine()),
    ] {
        if let Some(value) = value {
            message.push(' ');
            message.push_str(key);
            message.push('=');
            message.push_str(value);
        }
    }
    if let Some(method) = envelope.presentation_method() {
        message.push_str(" method=");
        message.push_str(method);
    }
    message
}

fn contextual_phase_prefix(mut message: String, envelope: &CanonicalEnvelope) -> String {
    let Some(_) = envelope.presentation_context() else {
        return message;
    };
    let kind = match envelope.presentation_request_kind() {
        "probe" => "probe",
        "model_listing" => "model listing",
        "inference" => "inference",
        "management" => "management",
        _ => return message,
    };
    if message == "request admitted" {
        return format!("{kind} admitted");
    }
    message.insert_str(0, kind);
    message.insert(kind.len(), ' ');
    message
}

fn presentation_channel_name(channel: super::replay::ReplayChannel) -> &'static str {
    match channel {
        super::replay::ReplayChannel::Requests => "requests",
        super::replay::ReplayChannel::Operations => "operations",
        super::replay::ReplayChannel::System => "system",
    }
}
