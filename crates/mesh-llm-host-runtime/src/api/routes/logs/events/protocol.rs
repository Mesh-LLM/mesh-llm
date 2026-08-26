use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::replay::ReplayChannel;
use serde::Serialize;

use super::super::dto::safe_metadata;
use super::super::event_kind;
use super::query::{AuditCursor, Cursor};
use crate::logging::ReplayRecord;
use crate::logging::RequestSummaryEventSnapshots;

mod audit_entry;
pub(super) use audit_entry::{audit_entry_frame, durable_audit_entry_frame};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    request: Option<PublicRequest>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicRequest {
    request_id: String,
    outcome: String,
    created_at: String,
    terminal_at: Option<String>,
    route: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    caller_endpoint_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    caller_addr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    caller_path_type: Option<String>,
    status_code: Option<u16>,
    source: &'static str,
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
    let request = request_projection(record, &envelope);
    let data = PublicEvent {
        event_id: envelope.event_id.as_uuid().to_string(),
        request_id: envelope.request_id.as_uuid().to_string(),
        occurred_at: envelope.occurred_at.clone(),
        channel: record.replay.channel.into(),
        sequence: record.replay.sequence,
        kind: event_kind(&envelope.event),
        request,
    };
    frame("log_event", &cursor_id(record.cursor), &data)
}

fn request_projection(
    record: &ReplayRecord,
    envelope: &CanonicalEnvelope,
) -> Option<PublicRequest> {
    let payload: serde_json::Value = serde_json::from_str(&record.entry.payload).ok()?;
    let snapshots = payload
        .get("request_summary_snapshots")
        .cloned()
        .and_then(|value| serde_json::from_value::<RequestSummaryEventSnapshots>(value).ok())?;
    let snapshot = snapshots.after()?;
    let metadata = snapshot.metadata();
    Some(PublicRequest {
        request_id: envelope.request_id.as_uuid().to_string(),
        outcome: snapshot.state().to_owned(),
        created_at: snapshot.created_at().to_owned(),
        terminal_at: snapshot.terminal_at().map(str::to_owned),
        route: metadata.route().map(safe_metadata),
        model: metadata.model().map(safe_metadata),
        provider: metadata.provider().map(safe_metadata),
        engine: metadata.engine().map(safe_metadata),
        caller_endpoint_id: metadata.caller_endpoint_id().map(safe_metadata),
        caller_addr: metadata.caller_addr().map(safe_metadata),
        caller_path_type: metadata.caller_path_type().map(safe_metadata),
        status_code: lifecycle_status_code(&envelope.event),
        source: "active",
    })
}

const fn lifecycle_status_code(event: &LifecycleEvent) -> Option<u16> {
    match event {
        LifecycleEvent::AttemptCompleted { status_code, .. }
        | LifecycleEvent::Completed { status_code, .. } => *status_code,
        LifecycleEvent::Failed { status_code, .. }
        | LifecycleEvent::Rejected { status_code, .. } => *status_code,
        _ => None,
    }
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

/// Fixed `stream_error` frame emitted exactly once when the durable audit
/// reconcile query fails. The audit cursor keeps the client's replay position
/// stable while the code distinguishes a failed reconcile from the
/// `invalid_event` frame used for malformed entries.
pub(super) fn audit_reconcile_error_frame(sequence: u64) -> String {
    frame(
        "stream_error",
        &AuditCursor(sequence).event_id(),
        &serde_json::json!({"code":"audit_reconcile_failed"}),
    )
    .expect("fixed audit reconcile error frame fits the SSE bound")
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

fn frame<T: Serialize>(event: &str, id: &str, data: &T) -> Result<String, ()> {
    let data = serde_json::to_string(data).map_err(|_| ())?;
    let frame = format!("event: {event}\nid: {id}\ndata: {data}\n\n");
    (frame.len() <= MAX_FRAME_BYTES).then_some(frame).ok_or(())
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
