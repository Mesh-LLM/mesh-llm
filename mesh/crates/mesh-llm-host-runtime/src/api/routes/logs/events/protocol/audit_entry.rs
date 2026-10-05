use mesh_llm_events::CliCommandSummary;
use mesh_llm_log_store::{AuditEntryDetail, AuditEntrySeverity};
use serde::Serialize;
use std::collections::BTreeMap;
use std::net::SocketAddr;

use super::super::query::AuditCursor;
use crate::api::routes::logs::dto::safe_metadata;
use crate::logging::{AuditReplayRecord, OperationalAuditContext};

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
    remote_addr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    numeric_summaries: BTreeMap<String, u64>,
    sequence: u64,
}

#[derive(Default)]
struct AuditPathProjection {
    remote_addr: Option<String>,
    path_type: Option<&'static str>,
}

fn audit_path_projection(
    context_version: Option<u8>,
    path_type: Option<&str>,
    remote_addr: Option<&str>,
) -> AuditPathProjection {
    if context_version != Some(1) {
        return AuditPathProjection::default();
    }
    match path_type {
        Some("direct") => AuditPathProjection {
            remote_addr: remote_addr
                .and_then(|value| value.parse::<SocketAddr>().ok())
                .map(|value| value.to_string()),
            path_type: Some("direct"),
        },
        Some("relay") => AuditPathProjection {
            remote_addr: None,
            path_type: Some("relay"),
        },
        Some(_) | None => AuditPathProjection::default(),
    }
}

/// Audit entry frame: privacy-safe projection of an audit replay record.
/// Never contains `canonical_envelope` or arbitrary `detail_json`.
pub(in super::super) fn audit_entry_frame(record: &AuditReplayRecord) -> Result<String, ()> {
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
    let path = audit_path_projection(
        context_version,
        payload.get("path_type").and_then(serde_json::Value::as_str),
        payload
            .get("remote_addr")
            .and_then(serde_json::Value::as_str),
    );

    let data = AuditEntryData {
        entry_id,
        occurred_at,
        source,
        code,
        severity,
        context_version,
        subject_kind,
        subject_id: context_string("subject_id"),
        remote_addr: path.remote_addr,
        path_type: path.path_type,
        operation_id: context_string("operation_id"),
        request_id: context_string("request_id"),
        reason_code: context_code("reason_code"),
        outcome: context_code("outcome"),
        command_summary: context_version.and_then(|_| {
            payload
                .get("command_summary")
                .and_then(serde_json::Value::as_str)
                .and_then(CliCommandSummary::sanitize)
                .map(|summary| summary.as_str().to_owned())
        }),
        duration_ms: context_version.and_then(|_| {
            payload
                .get("duration_ms")
                .and_then(serde_json::Value::as_u64)
        }),
        numeric_summaries,
        sequence: record.sequence,
    };
    super::frame(
        "audit_entry",
        &AuditCursor(record.sequence).event_id(),
        &data,
    )
}

/// Render a row read back from the durable store. This is the production audit
/// reconciliation path and deliberately uses the database sequence as the SSE
/// cursor so entries written by another process share the same ordering.
pub(in super::super) fn durable_audit_entry_frame(detail: AuditEntryDetail) -> Result<String, ()> {
    let record = detail.entry;
    let sequence = u64::try_from(record.sequence).map_err(|_| ())?;
    let path = audit_path_projection(
        record.context_version,
        detail.path_type.as_deref(),
        detail.remote_addr.as_deref(),
    );
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
        subject_id: record.subject_id.as_deref().map(safe_metadata),
        remote_addr: path.remote_addr,
        path_type: path.path_type,
        operation_id: record.operation_id.as_deref().map(safe_metadata),
        request_id: record.correlation_request_id.as_deref().map(safe_metadata),
        reason_code: record.reason_code,
        outcome: record.outcome,
        command_summary: detail
            .command_summary
            .and_then(|summary| CliCommandSummary::sanitize(&summary))
            .map(|summary| summary.as_str().to_owned()),
        duration_ms: record.duration_ms,
        numeric_summaries: record.numeric_summaries,
        sequence,
    };
    super::frame("audit_entry", &AuditCursor(sequence).event_id(), &data)
}

#[cfg(test)]
mod tests;
