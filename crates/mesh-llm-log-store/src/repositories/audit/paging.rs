use rusqlite::types::Value;

use super::{AuditEntryFilters, DEFAULT_AUDIT_ENTRY_LIMIT, MAX_AUDIT_ENTRY_LIMIT};
use crate::error::LogStoreError;
use crate::timestamps::canonical_comparison_timestamp;

pub(super) fn validate_limit(limit: Option<usize>) -> Result<usize, LogStoreError> {
    let limit = limit.unwrap_or(DEFAULT_AUDIT_ENTRY_LIMIT);
    if (1..=MAX_AUDIT_ENTRY_LIMIT).contains(&limit) {
        Ok(limit)
    } else {
        Err(LogStoreError::InvalidQuery(format!(
            "audit entry limit must be within 1..={MAX_AUDIT_ENTRY_LIMIT}"
        )))
    }
}

pub(super) fn query_parts(
    cursor: Option<&(String, String)>,
    filters: AuditEntryFilters,
) -> Result<(String, Vec<Value>), LogStoreError> {
    let mut clauses = Vec::new();
    let mut parameters = Vec::new();
    if let Some((occurred_at, entry_id)) = cursor {
        clauses.push("(occurred_at, entry_id) < (?, ?)".to_string());
        parameters.push(Value::Text(occurred_at.clone()));
        parameters.push(Value::Text(entry_id.clone()));
    }
    if let Some(source) = filters.source {
        clauses.push("actor = ?".to_string());
        parameters.push(Value::Text(source.as_str().to_string()));
    }
    if let Some(severity) = filters.severity {
        clauses.push("CASE WHEN json_valid(detail_json) THEN json_extract(detail_json, '$.severity') END = ?".to_string());
        parameters.push(Value::Text(severity.as_str().to_string()));
    }
    let from = filters
        .from
        .as_deref()
        .map(canonical_comparison_timestamp)
        .transpose()?;
    let to = filters
        .to
        .as_deref()
        .map(canonical_comparison_timestamp)
        .transpose()?;
    if from > to && to.is_some() {
        return Err(LogStoreError::InvalidQuery(
            "audit from must not be after to".to_string(),
        ));
    }
    if let Some(from) = from {
        clauses.push("occurred_at >= ?".to_string());
        parameters.push(Value::Text(from));
    }
    if let Some(to) = to {
        clauses.push("occurred_at <= ?".to_string());
        parameters.push(Value::Text(to));
    }
    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    Ok((where_clause, parameters))
}
