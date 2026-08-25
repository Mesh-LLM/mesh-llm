use mesh_llm_events::audit::SanitizedAuditScalar;

use super::detail::StoredAuditDetail;
use super::model::canonicalize_persisted_source;
use super::{AuditEntryRow, AuditEntrySeverity};
use crate::AuditEntryDetail;

pub(super) fn audit_entry_detail(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuditEntryDetail> {
    let detail = StoredAuditDetail::parse(row.get(6)?).bounded();
    let source = SanitizedAuditScalar::sanitize(&row.get::<_, String>(4)?);
    Ok(AuditEntryDetail {
        entry: AuditEntryRow {
            sequence: row.get(0)?,
            entry_id: row.get(1)?,
            request_id: row.get(2)?,
            occurred_at: row.get(3)?,
            source: canonicalize_persisted_source(source.as_str()).to_owned(),
            code: SanitizedAuditScalar::sanitize(&row.get::<_, String>(5)?)
                .as_str()
                .to_owned(),
            severity: AuditEntrySeverity::parse(detail.severity),
            context_version: detail.context_version,
            subject_kind: detail.subject_kind,
            subject_id: detail.subject_id,
            operation_id: detail.operation_id,
            correlation_request_id: detail.request_id,
            reason_code: detail.reason_code,
            outcome: detail.outcome,
            duration_ms: detail.duration_ms,
            numeric_summaries: detail.numeric_summaries,
        },
        remote_addr: detail.remote_addr,
        path_type: detail.path_type,
        command_summary: detail.command_summary,
    })
}
