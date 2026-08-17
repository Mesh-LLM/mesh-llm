use base64::Engine;
use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::{LifecycleEvent, TokenUsage};
use mesh_llm_log_store::{
    ArtifactContent, ArtifactRecord, AuditEntryRow, AuditEntrySeverity, EventRecord, ProxyRecord,
    RequestRecord,
};
use serde::Serialize;
use std::collections::BTreeMap;

use super::{LogsError, event_kind};
use crate::logging::RequestSummaryEntry;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PageDto<T> {
    pub(crate) items: Vec<T>,
    pub(crate) next_cursor: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RequestDto {
    request_id: String,
    outcome: String,
    created_at: String,
    terminal_at: Option<String>,
    route: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    status_code: Option<i64>,
    source: &'static str,
}

impl RequestDto {
    pub(super) fn durable(record: RequestRecord) -> Self {
        Self {
            request_id: record.request_id,
            outcome: record.outcome,
            created_at: record.created_at,
            terminal_at: record.terminal_at,
            route: record.route.as_deref().map(safe_metadata),
            model: record.model.as_deref().map(safe_metadata),
            provider: record.provider.as_deref().map(safe_metadata),
            engine: record.engine.as_deref().map(safe_metadata),
            status_code: record.status_code,
            source: "durable",
        }
    }

    pub(super) fn active(entry: RequestSummaryEntry, metadata: Option<RequestRecord>) -> Self {
        let summary_metadata = entry.metadata.clone();
        let metadata = metadata.map(|record| {
            (
                record.route.as_deref().map(safe_metadata),
                record.model.as_deref().map(safe_metadata),
                record.provider.as_deref().map(safe_metadata),
                record.engine.as_deref().map(safe_metadata),
                record.status_code,
            )
        });
        let (route, model, provider, engine, status_code) = metadata.unwrap_or_default();
        Self {
            request_id: entry.request_id,
            outcome: entry.state,
            created_at: entry.created_at,
            terminal_at: entry.terminal_at,
            route: summary_metadata.route().map(safe_metadata).or(route),
            model: summary_metadata.model().map(safe_metadata).or(model),
            provider: summary_metadata.provider().map(safe_metadata).or(provider),
            engine: summary_metadata.engine().map(safe_metadata).or(engine),
            status_code,
            source: "active",
        }
    }

    pub(super) fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(super) fn created_at(&self) -> &str {
        &self.created_at
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EventDto {
    event_id: String,
    request_id: String,
    occurred_at: String,
    kind: &'static str,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    attempt_id: Option<String>,
    status_code: Option<u16>,
    duration_ms: Option<u64>,
    tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    cached_prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl TryFrom<EventRecord> for EventDto {
    type Error = LogsError;

    fn try_from(record: EventRecord) -> Result<Self, Self::Error> {
        let envelope = CanonicalEnvelope::from_json_str(&record.payload_json)
            .map_err(|_| LogsError::StoreUnavailable)?;
        if envelope.event_id.as_uuid().to_string() != record.event_id
            || envelope.request_id.as_uuid().to_string() != record.request_id
            || envelope.occurred_at != record.occurred_at
        {
            return Err(LogsError::StoreUnavailable);
        }
        let mut dto = Self {
            event_id: record.event_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            kind: event_kind(&envelope.event),
            model: None,
            provider: None,
            engine: None,
            attempt_id: None,
            status_code: None,
            duration_ms: None,
            tokens: None,
            prompt_tokens: None,
            cached_prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
        };
        match envelope.event {
            LifecycleEvent::Admitted { model, .. } | LifecycleEvent::StreamStarted { model } => {
                dto.model = model.as_deref().map(safe_metadata);
            }
            LifecycleEvent::RouteSelected {
                model,
                provider,
                engine,
            } => {
                dto.model = model.as_deref().map(safe_metadata);
                dto.provider = provider.as_deref().map(safe_metadata);
                dto.engine = engine.as_deref().map(safe_metadata);
            }
            LifecycleEvent::AttemptStarted { attempt_id }
            | LifecycleEvent::AttemptFailed { attempt_id, .. } => {
                dto.attempt_id = attempt_id.map(|id| id.as_uuid().to_string());
            }
            LifecycleEvent::AttemptCompleted {
                attempt_id,
                status_code,
            } => {
                dto.attempt_id = attempt_id.map(|id| id.as_uuid().to_string());
                dto.status_code = status_code;
            }
            LifecycleEvent::StreamChunk { tokens } => {
                dto.tokens = tokens;
            }
            LifecycleEvent::StreamCompleted { tokens, usage } => {
                dto.tokens = tokens;
                dto.set_usage(usage);
            }
            LifecycleEvent::UsageRecorded {
                prompt_tokens,
                cached_prompt_tokens,
                completion_tokens,
                total_tokens,
            } => {
                dto.prompt_tokens = prompt_tokens;
                dto.cached_prompt_tokens = cached_prompt_tokens;
                dto.completion_tokens = completion_tokens;
                dto.total_tokens = total_tokens;
            }
            LifecycleEvent::Completed {
                status_code,
                duration_ms,
                usage,
            } => {
                dto.status_code = status_code;
                dto.duration_ms = duration_ms;
                dto.set_usage(usage);
            }
            LifecycleEvent::StreamError { .. }
            | LifecycleEvent::BackendStreamFirstItem
            | LifecycleEvent::AuditError { .. }
            | LifecycleEvent::Cancelled { .. }
            | LifecycleEvent::Dropped { .. } => {}
            LifecycleEvent::Failed { status_code, .. }
            | LifecycleEvent::Rejected { status_code, .. } => {
                dto.status_code = status_code;
            }
        }
        Ok(dto)
    }
}

impl EventDto {
    fn set_usage(&mut self, usage: Option<TokenUsage>) {
        if let Some(usage) = usage {
            self.prompt_tokens = usage.prompt_tokens;
            self.completion_tokens = usage.completion_tokens;
            self.total_tokens = usage.total_tokens;
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ArtifactDto {
    artifact_id: String,
    request_id: String,
    occurred_at: String,
    kind: String,
    media_kind: Option<String>,
    checksum: Option<String>,
    bytes: i64,
    version: i32,
    redacted: bool,
    truncated: bool,
    content_state: &'static str,
    unavailable_reason: Option<String>,
    content_base64: Option<String>,
}

impl ArtifactDto {
    pub(super) fn metadata(record: ArtifactRecord) -> Self {
        Self::from_parts(record, None)
    }

    pub(super) fn content(record: ArtifactRecord, content: ArtifactContent) -> Self {
        Self::from_parts(
            record,
            Some(base64::engine::general_purpose::STANDARD.encode(content.bytes)),
        )
    }

    /// Whether this read returned redacted artifact bytes. The route audit
    /// records only this outcome classification, never this DTO or its body.
    pub(super) fn has_available_content(&self) -> bool {
        self.content_state == "available"
    }

    fn from_parts(record: ArtifactRecord, content_base64: Option<String>) -> Self {
        let content_state = artifact_state(&record);
        Self {
            artifact_id: record.artifact_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            kind: safe_metadata(&record.kind),
            media_kind: record.media_kind.as_deref().map(safe_metadata),
            checksum: record.checksum,
            bytes: record.bytes,
            version: record.version,
            redacted: record.redacted,
            truncated: record.truncated,
            content_state,
            unavailable_reason: record.unavailable_reason,
            content_base64,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProxyDto {
    attempt_id: String,
    request_id: String,
    occurred_at: String,
    target: String,
    provider: Option<String>,
    engine: Option<String>,
    started_at: Option<String>,
    completed_at: Option<String>,
    status_code: Option<i64>,
}

impl From<ProxyRecord> for ProxyDto {
    fn from(record: ProxyRecord) -> Self {
        Self {
            attempt_id: record.attempt_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            target: safe_target(&record.target),
            provider: record.provider.as_deref().map(safe_metadata),
            engine: record.engine.as_deref().map(safe_metadata),
            started_at: record.started_at,
            completed_at: record.completed_at,
            status_code: record.status_code,
        }
    }
}

/// Privacy-safe projection of a durable audit entry for the trusted-local
/// management API. Arbitrary `detail_json` and message text never cross this
/// boundary; only the versioned typed context allowlist is exposed.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AuditDto {
    sequence: i64,
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
}

impl From<AuditEntryRow> for AuditDto {
    fn from(row: AuditEntryRow) -> Self {
        Self {
            sequence: row.sequence,
            entry_id: row.entry_id,
            occurred_at: row.occurred_at,
            source: row.source,
            code: row.code,
            severity: row.severity.map(|s| match s {
                AuditEntrySeverity::Info => "info".to_string(),
                AuditEntrySeverity::Warning => "warning".to_string(),
                AuditEntrySeverity::Error => "error".to_string(),
            }),
            context_version: row.context_version,
            subject_kind: row.subject_kind,
            subject_id: row.subject_id.as_deref().map(safe_metadata),
            operation_id: row.operation_id.as_deref().map(safe_metadata),
            request_id: row.correlation_request_id.as_deref().map(safe_metadata),
            reason_code: row.reason_code,
            outcome: row.outcome,
            duration_ms: row.duration_ms,
            numeric_summaries: row.numeric_summaries,
        }
    }
}

pub(super) fn artifact_state(record: &ArtifactRecord) -> &'static str {
    if record.corrupt {
        "corrupt"
    } else if record.unavailable_reason.is_some() {
        "unavailable"
    } else if record.missing || (record.checksum.is_none() && record.bytes == 0) {
        "missing"
    } else if record.redacted {
        "available"
    } else {
        "unavailable"
    }
}

fn safe_target(target: &str) -> String {
    let Ok(url) = url::Url::parse(target) else {
        return "opaque".to_string();
    };
    let Some(host) = url.host_str() else {
        return "opaque".to_string();
    };
    match url.port() {
        Some(port) => format!("{}://{host}:{port}", url.scheme()),
        None => format!("{}://{host}", url.scheme()),
    }
}

#[cfg(test)]
mod usage_tests {
    use mesh_llm_events::logging::{
        envelope::CanonicalEnvelope,
        events::LifecycleEvent,
        identifiers::{EventId, RequestId},
        replay::ReplayChannel,
    };

    use super::*;

    #[test]
    fn usage_event_projects_numeric_counters() {
        let event_id = EventId::new();
        let request_id = RequestId::new();
        let occurred_at = "2026-08-12T12:00:00Z".to_string();
        let envelope = CanonicalEnvelope::new(
            event_id,
            request_id,
            ReplayChannel::Requests,
            1,
            occurred_at.clone(),
            LifecycleEvent::UsageRecorded {
                prompt_tokens: Some(21),
                cached_prompt_tokens: Some(13),
                completion_tokens: Some(8),
                total_tokens: Some(29),
            },
        );
        let dto = EventDto::try_from(EventRecord {
            event_id: event_id.as_uuid().to_string(),
            request_id: request_id.as_uuid().to_string(),
            occurred_at,
            payload_json: serde_json::to_string(&envelope).unwrap(),
        })
        .unwrap();
        let wire = serde_json::to_value(dto).unwrap();

        assert_eq!(wire["kind"], "usage_recorded");
        assert_eq!(wire["promptTokens"], 21);
        assert_eq!(wire["cachedPromptTokens"], 13);
        assert_eq!(wire["completionTokens"], 8);
        assert_eq!(wire["totalTokens"], 29);
    }
}

fn safe_metadata(value: &str) -> String {
    let trimmed = value.trim();
    let path_shaped = trimmed.starts_with('/')
        || trimmed.starts_with("~/")
        || trimmed.as_bytes().get(1) == Some(&b':')
        || trimmed.contains('\\');
    if path_shaped || trimmed.contains('?') || trimmed.contains("://") {
        return "[REDACTED]".to_string();
    }
    crate::logging::policy::apply_redaction(trimmed).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_events::logging::{
        envelope::CanonicalEnvelope,
        identifiers::{EventId, RequestId},
    };

    #[test]
    fn event_dto_projects_authoritative_usage_and_terminal_status() {
        let request_id = RequestId::new();
        let event_id = EventId::new();
        let occurred_at = "2026-08-11T12:00:00Z";
        let envelope = CanonicalEnvelope::new(
            event_id,
            request_id,
            mesh_llm_events::logging::replay::ReplayChannel::Operations,
            1,
            occurred_at.into(),
            LifecycleEvent::Completed {
                status_code: Some(201),
                duration_ms: Some(9),
                usage: Some(TokenUsage {
                    prompt_tokens: Some(8),
                    completion_tokens: Some(3),
                    total_tokens: Some(11),
                }),
            },
        );
        let record = EventRecord {
            event_id: event_id.as_uuid().to_string(),
            request_id: request_id.as_uuid().to_string(),
            occurred_at: occurred_at.into(),
            payload_json: serde_json::to_string(&envelope).unwrap(),
        };

        let json = serde_json::to_value(EventDto::try_from(record).unwrap()).unwrap();
        assert_eq!(json["statusCode"], 201);
        assert_eq!(json["promptTokens"], 8);
        assert_eq!(json["completionTokens"], 3);
        assert_eq!(json["totalTokens"], 11);
        assert!(json["tokens"].is_null());
    }

    #[test]
    fn event_dto_preserves_moa_failure_and_committed_stream_statuses() {
        for (sequence, event, expected_status) in [
            (
                1,
                LifecycleEvent::Failed {
                    status_code: Some(502),
                    error: "moa_turn_failed".into(),
                },
                502,
            ),
            (
                2,
                LifecycleEvent::Failed {
                    status_code: Some(200),
                    error: "moa_turn_failed_after_commit".into(),
                },
                200,
            ),
        ] {
            let request_id = RequestId::new();
            let event_id = EventId::new();
            let occurred_at = "2026-08-11T12:00:00Z";
            let envelope = CanonicalEnvelope::new(
                event_id,
                request_id,
                mesh_llm_events::logging::replay::ReplayChannel::Operations,
                sequence,
                occurred_at.into(),
                event,
            );
            let record = EventRecord {
                event_id: event_id.as_uuid().to_string(),
                request_id: request_id.as_uuid().to_string(),
                occurred_at: occurred_at.into(),
                payload_json: serde_json::to_string(&envelope).unwrap(),
            };

            let json = serde_json::to_value(EventDto::try_from(record).unwrap()).unwrap();
            assert_eq!(json["kind"], "failed");
            assert_eq!(json["statusCode"], expected_status);
            assert!(json["promptTokens"].is_null());
            assert!(json["completionTokens"].is_null());
            assert!(json["totalTokens"].is_null());
        }
    }

    #[test]
    fn metadata_and_proxy_dtos_do_not_leak_paths_or_credentials() {
        let dto = RequestDto::durable(RequestRecord {
            request_id: uuid::Uuid::new_v4().to_string(),
            outcome: "completed".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            terminal_at: None,
            route: Some("/private/secret".into()),
            model: Some("Bearer secret-token".into()),
            provider: None,
            engine: None,
            status_code: None,
        });
        let json = serde_json::to_string(&dto).expect("serialize dto");
        assert!(!json.contains("secret-token"));
        assert!(!json.contains("/private/secret"));

        let proxy = ProxyDto::from(ProxyRecord {
            attempt_id: uuid::Uuid::new_v4().to_string(),
            request_id: uuid::Uuid::new_v4().to_string(),
            occurred_at: "2026-01-01T00:00:00Z".into(),
            target: "https://user:password@example.test/path?token=secret".into(),
            provider: None,
            engine: None,
            started_at: None,
            completed_at: None,
            status_code: None,
        });
        let json = serde_json::to_string(&proxy).expect("serialize proxy");
        assert!(json.contains("https://example.test"));
        assert!(!json.contains("password"));
        assert!(!json.contains("token=secret"));
    }

    #[test]
    fn intentionally_omitted_artifact_is_unavailable_not_missing() {
        let record = ArtifactRecord {
            artifact_id: uuid::Uuid::new_v4().to_string(),
            request_id: uuid::Uuid::new_v4().to_string(),
            occurred_at: "2026-08-04T12:00:00Z".into(),
            kind: "response".into(),
            media_kind: Some("application/json".into()),
            checksum: None,
            bytes: 0,
            version: 1,
            redacted: true,
            truncated: false,
            missing: false,
            corrupt: false,
            unavailable_reason: Some("streaming_response_not_assembled".into()),
        };

        assert_eq!(artifact_state(&record), "unavailable");
        let json = serde_json::to_value(ArtifactDto::metadata(record)).expect("serialize DTO");
        assert_eq!(json["contentState"], "unavailable");
        assert_eq!(
            json["unavailableReason"],
            "streaming_response_not_assembled"
        );
        assert!(json["contentBase64"].is_null());
        assert!(json["checksum"].is_null());
    }
}
