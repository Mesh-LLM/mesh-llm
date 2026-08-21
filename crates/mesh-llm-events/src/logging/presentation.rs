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

    /// A deterministic local presentation summary with a bounded message body.
    ///
    /// Only the message/context portion is bounded. The trailing correlation
    /// metadata (request/event IDs, channel, sequence, token counts) is
    /// appended afterward so operator correlation survives even at the
    /// smallest limits. The source summary is already payload-free. Limiting
    /// happens only after that safe projection has been constructed, so
    /// callers cannot accidentally truncate and expose a raw lifecycle payload
    /// instead.
    pub fn presentation_local_summary_with_limit(&self, limit: usize) -> String {
        let mut correlation = format!(
            " request_id={} event_id={} channel={} sequence={}",
            self.request_id.as_uuid(),
            self.event_id.as_uuid(),
            presentation_channel_name(self.channel),
            self.sequence,
        );
        if let Some(tokens) = self.presentation_token_count() {
            correlation.push_str(&format!(" tokens={tokens}"));
        }
        let mut message = truncate_presentation_message(self.presentation_message(), limit);
        message.push_str(&correlation);
        message
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

/// Marker appended to a message body that was truncated to its character
/// budget. Correlation metadata is never part of this budget.
const ELLIPSIS: &str = "...";

fn truncate_presentation_message(message: String, limit: usize) -> String {
    if message.chars().count() <= limit {
        return message;
    }
    // The ellipsis marker itself must fit inside `limit`; below that, it is
    // truncated too so the returned string never exceeds the documented
    // character budget.
    let ellipsis_len = ELLIPSIS.chars().count().min(limit);
    let keep = limit - ellipsis_len;
    let truncated: String = message.chars().take(keep).collect();
    let ellipsis: String = ELLIPSIS.chars().take(ellipsis_len).collect();
    format!("{truncated}{ellipsis}")
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

/// The closed request classification used by local presentation rendering.
///
/// This is the typed counterpart of the string vocabulary produced by
/// [`CanonicalEnvelope::presentation_request_kind`]; rendering matches on the
/// enum so an unknown or misspelled kind falls through explicitly instead of
/// silently dropping the phase prefix or context classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestKind {
    Probe,
    ModelListing,
    Inference,
    Management,
    Unknown,
}

impl std::str::FromStr for RequestKind {
    type Err = std::convert::Infallible;

    /// Parse the compact request-kind vocabulary; anything unrecognized maps
    /// to [`RequestKind::Unknown`].
    fn from_str(kind: &str) -> Result<Self, Self::Err> {
        Ok(match kind {
            "probe" => Self::Probe,
            "model_listing" => Self::ModelListing,
            "inference" => Self::Inference,
            "management" => Self::Management,
            _ => Self::Unknown,
        })
    }
}

impl RequestKind {
    /// The human-readable label used as a request phase prefix.
    fn as_str(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::ModelListing => "model listing",
            Self::Inference => "inference",
            Self::Management => "management",
            Self::Unknown => "unknown",
        }
    }
}

fn append_context(message: impl Into<String>, envelope: &CanonicalEnvelope) -> String {
    let mut message = contextual_phase_prefix(message.into(), envelope);
    if envelope.presentation_context().is_some() {
        let kind = envelope
            .presentation_request_kind()
            .parse::<RequestKind>()
            .unwrap_or(RequestKind::Unknown);
        if kind == RequestKind::Unknown {
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
    let kind = envelope
        .presentation_request_kind()
        .parse::<RequestKind>()
        .unwrap_or(RequestKind::Unknown);
    let phase = match kind {
        RequestKind::Unknown => return message,
        RequestKind::Probe
        | RequestKind::ModelListing
        | RequestKind::Inference
        | RequestKind::Management => kind.as_str(),
    };
    if message == "request admitted" {
        return format!("{phase} admitted");
    }
    message.insert_str(0, phase);
    message.insert(phase.len(), ' ');
    message
}

fn presentation_channel_name(channel: super::replay::ReplayChannel) -> &'static str {
    match channel {
        super::replay::ReplayChannel::Requests => "requests",
        super::replay::ReplayChannel::Operations => "operations",
        super::replay::ReplayChannel::System => "system",
    }
}

#[cfg(test)]
mod tests {
    use super::super::identifiers::{EventId, RequestId};
    use super::super::replay::ReplayChannel;
    use super::*;

    fn completed_envelope() -> CanonicalEnvelope {
        CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            7,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: Some(3),
                usage: None,
            },
        )
        .with_presentation_context(CanonicalPresentationContext::from_parts(
            Some("health"),
            Some("direct_http"),
            Some("llama-3"),
            Some("openai_frontend"),
            Some("health"),
            Some("GET"),
        ))
    }

    fn admitted_envelope_with_kind(route: Option<&str>) -> CanonicalEnvelope {
        let mut envelope = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            1,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: Some("llama-3".into()),
                method: Some("POST".into()),
            },
        );
        if let Some(route) = route {
            envelope =
                envelope.with_presentation_context(CanonicalPresentationContext::from_parts(
                    Some(route),
                    Some("direct_http"),
                    Some("llama-3"),
                    Some("openai_frontend"),
                    None,
                    Some("POST"),
                ));
        }
        envelope
    }

    #[test]
    fn local_summary_within_limit_keeps_full_message_and_correlation() {
        let envelope = completed_envelope();
        let summary = envelope.presentation_local_summary_with_limit(2_048);

        assert!(summary.starts_with("probe request completed"));
        assert!(summary.contains(" status=200"));
        assert!(summary.contains(" duration=3ms"));
        assert!(summary.contains("request_id="));
        assert!(summary.contains("event_id="));
        assert!(summary.contains("channel=requests"));
        assert!(summary.contains("sequence=7"));
    }

    #[test]
    fn local_summary_truncates_message_but_keeps_correlation() {
        let envelope = completed_envelope();
        let summary = envelope.presentation_local_summary_with_limit(40);

        let message_portion = summary.split(" request_id=").next().unwrap();
        assert!(message_portion.chars().count() <= 40);
        assert!(summary.contains("..."));
        assert!(summary.contains("request_id="));
        assert!(summary.contains("event_id="));
        assert!(summary.contains("channel=requests"));
        assert!(summary.contains("sequence=7"));
        assert!(!summary.contains("duration=3ms"));
        assert!(!summary.contains("engine=health"));
    }

    #[test]
    fn local_summary_at_limit_one_keeps_only_correlation_metadata() {
        let envelope = completed_envelope();
        let summary = envelope.presentation_local_summary_with_limit(1);

        let message_portion = summary.split(" request_id=").next().unwrap();
        assert_eq!(message_portion, ".");
        assert!(summary.contains("request_id="));
        assert!(summary.contains("event_id="));
        assert!(summary.contains("channel=requests"));
        assert!(summary.contains("sequence=7"));
        assert!(!summary.contains("request completed"));
        assert!(!summary.contains(" route="));
    }

    #[test]
    fn local_summary_at_limit_zero_has_no_message_portion() {
        let envelope = completed_envelope();
        let summary = envelope.presentation_local_summary_with_limit(0);

        let message_portion = summary.split(" request_id=").next().unwrap();
        assert_eq!(message_portion, "");
        assert!(summary.contains("request_id="));
    }

    #[test]
    fn request_kind_parses_known_kinds_and_unknown_fallback() {
        assert_eq!("probe".parse::<RequestKind>().unwrap(), RequestKind::Probe);
        assert_eq!(
            "model_listing".parse::<RequestKind>().unwrap(),
            RequestKind::ModelListing
        );
        assert_eq!(
            "inference".parse::<RequestKind>().unwrap(),
            RequestKind::Inference
        );
        assert_eq!(
            "management".parse::<RequestKind>().unwrap(),
            RequestKind::Management
        );
        assert_eq!("typo".parse::<RequestKind>().unwrap(), RequestKind::Unknown);
        assert_eq!("".parse::<RequestKind>().unwrap(), RequestKind::Unknown);
    }

    #[test]
    fn request_kind_labels_render_expected_vocabulary() {
        for (kind, label) in [
            (RequestKind::Probe, "probe"),
            (RequestKind::ModelListing, "model listing"),
            (RequestKind::Inference, "inference"),
            (RequestKind::Management, "management"),
            (RequestKind::Unknown, "unknown"),
        ] {
            assert_eq!(kind.as_str(), label);
        }
    }

    #[test]
    fn contextual_phase_prefix_uses_typed_kind_labels() {
        assert_eq!(
            admitted_envelope_with_kind(Some("health")).presentation_message(),
            "probe admitted route=health source=direct_http model=llama-3 \
             provider=openai_frontend method=POST"
        );
        assert_eq!(
            admitted_envelope_with_kind(Some("models")).presentation_message(),
            "model listing admitted route=models source=direct_http model=llama-3 \
             provider=openai_frontend method=POST"
        );
        assert_eq!(
            admitted_envelope_with_kind(Some("chat_completions")).presentation_message(),
            "inference admitted route=chat_completions source=direct_http model=llama-3 \
             provider=openai_frontend method=POST"
        );
        assert_eq!(
            admitted_envelope_with_kind(Some("management_get_status")).presentation_message(),
            "management admitted route=management_get_status source=direct_http model=llama-3 \
             provider=openai_frontend method=POST"
        );
    }

    #[test]
    fn contextual_phase_prefix_falls_back_without_kind_label() {
        let envelope = admitted_envelope_with_kind(None);
        assert_eq!(
            envelope.presentation_message(),
            "request admitted model=llama-3 method=POST"
        );
    }

    #[test]
    fn contextual_phase_prefix_with_unknown_kind_marks_kind_unknown() {
        let envelope = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            1,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        )
        .with_presentation_context(CanonicalPresentationContext::from_parts(
            Some("not_a_closed_route"),
            Some("direct_http"),
            None,
            None,
            None,
            None,
        ));

        assert_eq!(
            envelope.presentation_message(),
            "request admitted kind=unknown source=direct_http"
        );
    }
}
