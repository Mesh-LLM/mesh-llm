use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::replay::ReplayChannel;
use mesh_llm_log_store::{AuditEntryRow, AuditEntrySeverity};
use serde::Serialize;
use std::collections::BTreeMap;

use super::super::event_kind;
use super::query::{AuditCursor, Cursor};
use crate::logging::OperationalAuditContext;
use crate::logging::RequestSummaryEventSnapshots;
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
        route: metadata.route().map(str::to_owned),
        model: metadata.model().map(str::to_owned),
        provider: metadata.provider().map(str::to_owned),
        engine: metadata.engine().map(str::to_owned),
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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditEntryData {
    entry_id: String,
    occurred_at: String,
    source: String,
    code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    severity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_version: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    subject_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    subject_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    numeric_summaries: BTreeMap<String, u64>,
    sequence: u64,
}

/// Audit entry frame: privacy-safe projection of an audit replay record.
/// Never contains `canonical_envelope` or arbitrary `detail_json`.
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
    let context_version = payload
        .get("context_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u8::try_from(value).ok())
        .filter(|value| *value == 1);
    let context_string = |key: &str| {
        context_version.and_then(|_| {
            payload
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty() && value.chars().count() <= 256)
                .map(str::to_owned)
        })
    };
    let context_code = |key: &str| {
        context_version.and_then(|_| {
            payload
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|value| OperationalAuditContext::valid_static_code(value))
                .map(str::to_owned)
        })
    };
    let subject_kind = context_version.and_then(|_| {
        payload
            .get("subject_kind")
            .and_then(serde_json::Value::as_str)
            .and_then(crate::logging::OperationalAuditSubjectKind::parse)
            .map(|kind| kind.as_str().to_owned())
    });
    let numeric_summaries = context_version.map_or_else(BTreeMap::new, |_| {
        payload
            .get("numeric_summaries")
            .and_then(serde_json::Value::as_object)
            .into_iter()
            .flatten()
            .filter(|(key, _)| OperationalAuditContext::valid_static_code(key))
            .filter_map(|(key, value)| value.as_u64().map(|value| (key.clone(), value)))
            .take(8)
            .collect()
    });

    let data = AuditEntryData {
        entry_id,
        occurred_at,
        source,
        code,
        severity,
        context_version,
        subject_kind,
        subject_id: context_string("subject_id"),
        operation_id: context_string("operation_id"),
        request_id: context_string("request_id"),
        reason_code: context_code("reason_code"),
        outcome: context_code("outcome"),
        duration_ms: context_version.and_then(|_| {
            payload
                .get("duration_ms")
                .and_then(serde_json::Value::as_u64)
        }),
        numeric_summaries,
        sequence: record.sequence,
    };
    frame(
        "audit_entry",
        &AuditCursor(record.sequence).event_id(),
        &data,
    )
}

/// Render a row read back from the durable store. This is the production audit
/// reconciliation path and deliberately uses the database sequence as the SSE
/// cursor so entries written by another process share the same ordering.
pub(super) fn durable_audit_entry_frame(record: AuditEntryRow) -> Result<String, ()> {
    let sequence = u64::try_from(record.sequence).map_err(|_| ())?;
    let severity = record.severity.map(|severity| match severity {
        AuditEntrySeverity::Info => "info".to_string(),
        AuditEntrySeverity::Warning => "warning".to_string(),
        AuditEntrySeverity::Error => "error".to_string(),
    });
    let data = AuditEntryData {
        entry_id: record.entry_id,
        occurred_at: record.occurred_at,
        source: record.source,
        code: record.code,
        severity,
        context_version: record.context_version,
        subject_kind: record.subject_kind,
        subject_id: record.subject_id,
        operation_id: record.operation_id,
        request_id: record.correlation_request_id,
        reason_code: record.reason_code,
        outcome: record.outcome,
        duration_ms: record.duration_ms,
        numeric_summaries: record.numeric_summaries,
        sequence,
    };
    frame("audit_entry", &AuditCursor(sequence).event_id(), &data)
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
    fn audit_entry_frame_filters_invalid_typed_context_fields() {
        let mut record = audit_record(4);
        record.entry.payload = serde_json::json!({
            "kind": "audit",
            "entry_id": "id-4",
            "occurred_at": "2026-01-01T00:00:00Z",
            "source": "runtime",
            "code": "startup_complete",
            "context_version": 1,
            "subject_kind": "not_a_subject",
            "reason_code": "NOT VALID",
            "outcome": "completed",
            "numeric_summaries": {
                "bad-key": 0,
                "metric_0": 0,
                "metric_1": 1,
                "metric_2": 2,
                "metric_3": 3,
                "metric_4": 4,
                "metric_5": 5,
                "metric_6": 6,
                "metric_7": 7,
                "metric_8": 8
            }
        })
        .to_string();
        let frame = audit_entry_frame(&record).expect("audit context frame");
        let data = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .expect("audit frame data");

        assert!(data.get("subjectKind").is_none());
        assert!(data.get("reasonCode").is_none());
        assert_eq!(data["outcome"], "completed");
        assert_eq!(data["numericSummaries"].as_object().unwrap().len(), 8);
        assert!(data["numericSummaries"].get("metric_7").is_some());
        assert!(data["numericSummaries"].get("metric_8").is_none());
        assert!(data["numericSummaries"].get("bad-key").is_none());
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
