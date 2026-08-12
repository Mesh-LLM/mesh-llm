use super::envelope::CanonicalEnvelope;
use super::events::LifecycleEvent;
use crate::OutputLevel;

/// The compact, payload-free vocabulary used by terminal and JSONL output.
///
/// Trusted local output retains only bounded correlation metadata (event and
/// request IDs, replay channel/sequence, and numeric lifecycle counters).
/// Identity fields, artifacts, model input/output, credentials, and free-form
/// error detail deliberately never cross this presentation boundary. Network
/// and telemetry projections remain stricter and do not use these local IDs.
impl CanonicalEnvelope {
    pub fn presentation_event_name(&self) -> &'static str {
        match self.event {
            LifecycleEvent::Admitted { .. } => "request_admitted",
            LifecycleEvent::RouteSelected { .. } => "request_route_selected",
            LifecycleEvent::AttemptStarted { .. } => "request_attempt_started",
            LifecycleEvent::AttemptCompleted { .. } => "request_attempt_completed",
            LifecycleEvent::AttemptFailed { .. } => "request_attempt_failed",
            LifecycleEvent::StreamStarted { .. } => "request_stream_started",
            LifecycleEvent::StreamChunk { .. } => "request_stream_chunk",
            LifecycleEvent::StreamCompleted { .. } => "request_stream_completed",
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
            | LifecycleEvent::StreamStarted { .. }
            | LifecycleEvent::StreamChunk { .. }
            | LifecycleEvent::StreamCompleted { .. }
            | LifecycleEvent::Completed { .. } => OutputLevel::Info,
        }
    }

    pub fn presentation_message(&self) -> String {
        match self.event {
            LifecycleEvent::Admitted { .. } => "request admitted".to_string(),
            LifecycleEvent::RouteSelected { .. } => "request route selected".to_string(),
            LifecycleEvent::AttemptStarted { .. } => "request attempt started".to_string(),
            LifecycleEvent::AttemptCompleted { status_code, .. } => {
                append_status("request attempt completed".to_string(), status_code)
            }
            LifecycleEvent::AttemptFailed { .. } => "request attempt failed".to_string(),
            LifecycleEvent::StreamStarted { .. } => "request stream started".to_string(),
            LifecycleEvent::StreamChunk { .. } => "request stream chunk".to_string(),
            LifecycleEvent::StreamCompleted { .. } => "request stream completed".to_string(),
            LifecycleEvent::StreamError { .. } => "request stream failed".to_string(),
            LifecycleEvent::AuditError { .. } => "logging audit warning".to_string(),
            LifecycleEvent::Completed {
                status_code,
                duration_ms,
            } => append_duration(
                append_status("request completed".to_string(), status_code),
                duration_ms,
            ),
            LifecycleEvent::Failed { .. } => "request failed".to_string(),
            LifecycleEvent::Rejected { .. } => "request rejected".to_string(),
            LifecycleEvent::Cancelled { .. } => "request cancelled".to_string(),
            LifecycleEvent::Dropped { .. } => "request dropped".to_string(),
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
            LifecycleEvent::StreamChunk { tokens } | LifecycleEvent::StreamCompleted { tokens } => {
                tokens
            }
            LifecycleEvent::Admitted { .. }
            | LifecycleEvent::RouteSelected { .. }
            | LifecycleEvent::AttemptStarted { .. }
            | LifecycleEvent::AttemptCompleted { .. }
            | LifecycleEvent::AttemptFailed { .. }
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
            | LifecycleEvent::StreamStarted { .. }
            | LifecycleEvent::StreamChunk { .. }
            | LifecycleEvent::StreamCompleted { .. }
            | LifecycleEvent::StreamError { .. }
            | LifecycleEvent::AuditError { .. } => None,
        }
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

fn presentation_channel_name(channel: super::replay::ReplayChannel) -> &'static str {
    match channel {
        super::replay::ReplayChannel::Requests => "requests",
        super::replay::ReplayChannel::Operations => "operations",
        super::replay::ReplayChannel::System => "system",
    }
}
