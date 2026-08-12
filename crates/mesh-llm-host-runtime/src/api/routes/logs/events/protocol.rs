use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::replay::ReplayChannel;
use serde::Serialize;

use super::query::{AuditCursor, Cursor};
use crate::logging::{AuditReplayRecord, ReplayRecord};

pub(in crate::api::routes::logs) const MAX_FRAME_BYTES: usize = 16 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ChannelName {
    Requests,
    Operations,
    System,
    Audit,
}

impl From<ReplayChannel> for ChannelName {
    fn from(channel: ReplayChannel) -> Self {
        match channel {
            ReplayChannel::Requests => Self::Requests,
            ReplayChannel::Operations => Self::Operations,
            ReplayChannel::System => Self::System,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicEvent {
    event_id: String,
    request_id: String,
    occurred_at: String,
    channel: ChannelName,
    sequence: u64,
    kind: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GapData {
    channel: ChannelName,
    from_sequence: u64,
    to_sequence: u64,
    recovery: RestRecovery,
}

impl GapData {
    pub(super) fn new(
        channel: ReplayChannel,
        from_sequence: u64,
        to_sequence: u64,
        recovery_cursor: Option<String>,
    ) -> Self {
        Self {
            channel: channel.into(),
            from_sequence,
            to_sequence,
            recovery: RestRecovery {
                endpoint: "/api/logs/requests",
                cursor: recovery_cursor,
            },
        }
    }

    pub(super) fn audit(
        from_sequence: u64,
        to_sequence: u64,
        recovery_cursor: Option<String>,
    ) -> Self {
        Self {
            channel: ChannelName::Audit,
            from_sequence,
            to_sequence,
            recovery: RestRecovery {
                endpoint: "/api/logs/audit",
                cursor: recovery_cursor,
            },
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RestRecovery {
    endpoint: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cursor: Option<String>,
}

/// Render one bounded privacy-safe lifecycle event. The bus's raw serialized
/// payload is never sent; only canonical identifiers, sequencing, timestamp,
/// and an exhaustive event-kind label cross the SSE boundary.
pub(super) fn event_frame(record: &ReplayRecord) -> Result<String, ()> {
    let envelope = envelope(record)?;
    let data = PublicEvent {
        event_id: envelope.event_id.as_uuid().to_string(),
        request_id: envelope.request_id.as_uuid().to_string(),
        occurred_at: envelope.occurred_at.clone(),
        channel: record.replay.channel.into(),
        sequence: record.replay.sequence,
        kind: event_kind(&envelope.event),
    };
    frame("log_event", &cursor_id(record.cursor), &data)
}

pub(super) fn gap_frame(cursor: Cursor, gap: &GapData) -> Result<String, ()> {
    frame("replay_gap", &cursor.event_id(), gap)
}

pub(super) fn error_frame(cursor: Cursor) -> String {
    frame(
        "stream_error",
        &cursor.event_id(),
        &serde_json::json!({"code":"invalid_event"}),
    )
    .expect("fixed stream error frame fits the SSE bound")
}

pub(in crate::api::routes::logs) fn heartbeat_frame() -> &'static str {
    ": keepalive\n\n"
}

fn envelope(record: &ReplayRecord) -> Result<CanonicalEnvelope, ()> {
    let parsed: serde_json::Value = serde_json::from_str(&record.entry.payload).map_err(|_| ())?;
    let envelope = parsed
        .get("canonical_envelope")
        .ok_or(())
        .and_then(|value| CanonicalEnvelope::from_json_str(&value.to_string()).map_err(|_| ()))?;
    if envelope.channel != record.replay.channel || envelope.sequence != record.replay.sequence {
        return Err(());
    }
    Ok(envelope)
}

fn cursor_id(cursor: crate::logging::ReplayCursor) -> String {
    Cursor::from_sequences(
        cursor.sequence(ReplayChannel::Requests),
        cursor.sequence(ReplayChannel::Operations),
        cursor.sequence(ReplayChannel::System),
    )
    .event_id()
}

fn event_kind(event: &LifecycleEvent) -> &'static str {
    match event {
        LifecycleEvent::Admitted { .. } => "admitted",
        LifecycleEvent::RouteSelected { .. } => "route_selected",
        LifecycleEvent::AttemptStarted { .. } => "attempt_started",
        LifecycleEvent::AttemptCompleted { .. } => "attempt_completed",
        LifecycleEvent::AttemptFailed { .. } => "attempt_failed",
        LifecycleEvent::StreamStarted { .. } => "stream_started",
        LifecycleEvent::StreamChunk { .. } => "stream_chunk",
        LifecycleEvent::StreamCompleted { .. } => "stream_completed",
        LifecycleEvent::StreamError { .. } => "stream_error",
        LifecycleEvent::AuditError { .. } => "audit_error",
        LifecycleEvent::Completed { .. } => "completed",
        LifecycleEvent::Failed { .. } => "failed",
        LifecycleEvent::Rejected { .. } => "rejected",
        LifecycleEvent::Cancelled { .. } => "cancelled",
        LifecycleEvent::Dropped { .. } => "dropped",
    }
}

fn frame<T: Serialize>(event: &str, id: &str, data: &T) -> Result<String, ()> {
    let data = serde_json::to_string(data).map_err(|_| ())?;
    let frame = format!("event: {event}\nid: {id}\ndata: {data}\n\n");
    (frame.len() <= MAX_FRAME_BYTES).then_some(frame).ok_or(())
}

/// Audit entry frame: privacy-safe projection of an audit replay record.
/// Never contains `canonical_envelope`, request IDs, or `detail_json`.
pub(super) fn audit_entry_frame(record: &AuditReplayRecord) -> Result<String, ()> {
    let payload: serde_json::Value = serde_json::from_str(&record.entry.payload).map_err(|_| ())?;
    let entry_id = payload
        .get("entry_id")
        .and_then(|v| v.as_str())
        .ok_or(())?
        .to_owned();
    let occurred_at = payload
        .get("occurred_at")
        .and_then(|v| v.as_str())
        .ok_or(())?
        .to_owned();
    let source = payload
        .get("source")
        .and_then(|v| v.as_str())
        .ok_or(())?
        .to_owned();
    let code = payload
        .get("code")
        .and_then(|v| v.as_str())
        .ok_or(())?
        .to_owned();
    let severity = payload
        .get("severity")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct AuditEntryData {
        entry_id: String,
        occurred_at: String,
        source: String,
        code: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        severity: Option<String>,
        sequence: u64,
    }

    let data = AuditEntryData {
        entry_id,
        occurred_at,
        source,
        code,
        severity,
        sequence: record.sequence,
    };
    frame(
        "audit_entry",
        &AuditCursor(record.sequence).event_id(),
        &data,
    )
}

/// Audit gap frame: points to `/api/logs/audit` for recovery.
pub(super) fn audit_gap_frame(
    from_sequence: u64,
    to_sequence: u64,
    recovery_cursor: Option<String>,
) -> Result<String, ()> {
    let gap = GapData::audit(from_sequence, to_sequence, recovery_cursor);
    frame("replay_gap", &AuditCursor(to_sequence).event_id(), &gap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::BusEntry;

    fn audit_record(sequence: u64) -> AuditReplayRecord {
        AuditReplayRecord {
            entry: BusEntry {
                payload: serde_json::json!({
                    "kind": "audit",
                    "entry_id": "test-entry-id",
                    "occurred_at": "2026-01-01T00:00:00Z",
                    "source": "runtime",
                    "code": "startup_complete",
                    "severity": "info",
                })
                .to_string(),
                channel_hint: 2,
            },
            sequence,
            cursor: sequence,
        }
    }

    #[test]
    fn audit_entry_frame_shape_and_fields() {
        let record = audit_record(7);
        let frame = audit_entry_frame(&record).expect("audit entry frame");

        assert!(frame.contains("event: audit_entry"));
        assert!(frame.contains("id: a1:7"));
        assert!(frame.contains("\"entryId\":\"test-entry-id\""));
        assert!(frame.contains("\"occurredAt\":\"2026-01-01T00:00:00Z\""));
        assert!(frame.contains("\"source\":\"runtime\""));
        assert!(frame.contains("\"code\":\"startup_complete\""));
        assert!(frame.contains("\"severity\":\"info\""));
        assert!(frame.contains("\"sequence\":7"));
        assert!(!frame.contains("canonical_envelope"));
        assert!(!frame.contains("detail_json"));
        assert!(frame.len() <= MAX_FRAME_BYTES);
    }

    #[test]
    fn audit_entry_frame_omits_severity_when_none() {
        let mut record = audit_record(3);
        record.entry.payload = serde_json::json!({
            "kind": "audit",
            "entry_id": "id-3",
            "occurred_at": "2026-01-01T00:00:00Z",
            "source": "cli",
            "code": "command_executed",
        })
        .to_string();
        let frame = audit_entry_frame(&record).expect("audit entry without severity");
        assert!(!frame.contains("severity"));
    }

    #[test]
    fn audit_entry_frame_rejects_malformed_payload() {
        let mut record = audit_record(1);
        record.entry.payload = "not-json".to_string();
        assert!(audit_entry_frame(&record).is_err());
    }

    #[test]
    fn audit_gap_frame_carries_audit_endpoint() {
        let frame = audit_gap_frame(5, 10, Some("a1:10".to_owned())).expect("audit gap frame");
        assert!(frame.contains("event: replay_gap"));
        assert!(frame.contains("id: a1:10"));
        assert!(frame.contains("/api/logs/audit"));
        assert!(!frame.contains("/api/logs/requests"));
    }

    #[test]
    fn lifecycle_gap_frame_still_carries_requests_endpoint() {
        let gap = GapData::new(ReplayChannel::Requests, 1, 5, Some("v1:1.0.0".to_owned()));
        let frame = gap_frame(Cursor::from_sequences(5, 0, 0), &gap).expect("lifecycle gap frame");
        assert!(frame.contains("/api/logs/requests"));
        assert!(!frame.contains("/api/logs/audit"));
        assert!(frame.contains("id: v1:5.0.0"));
    }
}
