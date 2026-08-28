use mesh_llm_events::audit::{SanitizedAuditDetailJson, SanitizedAuditScalar};
use rusqlite::types::Value;

use super::{Page, map_insert_constraint_error};
use crate::cursor::{decode_ordering_cursor, encode_cursor};
use crate::error::LogStoreError;
use crate::store::LogStore;
use crate::timestamps::canonical_persisted_timestamp;

mod detail;
mod model;
mod paging;
mod projection;

pub use model::{
    AuditEntryFilters, AuditEntryRow, AuditEntrySeverity, AuditEntrySource,
    DEFAULT_AUDIT_ENTRY_LIMIT, MAX_AUDIT_ENTRY_LIMIT,
};

use crate::AuditEntryDetail;
use paging::{
    query_parts as audit_entry_query_parts, validate_limit as validate_audit_entry_limit,
};
use projection::audit_entry_detail;

impl LogStore {
    pub fn insert_audit_entry(
        &self,
        entry_id: &str,
        request_id: Option<&str>,
        occurred_at: &str,
        source: &str,
        code: &str,
        detail_json: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let source = SanitizedAuditScalar::sanitize(source);
        let code = SanitizedAuditScalar::sanitize(code);
        let detail_json = detail_json.map(SanitizedAuditDetailJson::sanitize);
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action, detail_json) VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                entry_id,
                request_id,
                occurred_at,
                source.as_str(),
                code.as_str(),
                detail_json.as_ref().map(SanitizedAuditDetailJson::as_str)
            ],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("audit_entry {entry_id}")) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    /// List privacy-safe operational audit rows with cursor pagination.
    ///
    /// The query mirrors lifecycle ordering on `(occurred_at, entry_id)` while
    /// fetching one extra row to determine the next cursor. `detail_json` is
    /// only used internally for the bounded severity projection.
    pub fn list_audit_entries(
        &self,
        limit: Option<usize>,
        after_cursor: Option<&str>,
        filters: AuditEntryFilters,
    ) -> Result<Page<AuditEntryRow>, LogStoreError> {
        self.list_audit_entry_details(limit, after_cursor, filters)
            .map(|page| Page {
                items: page.items.into_iter().map(Into::into).collect(),
                next_cursor: page.next_cursor,
            })
    }

    pub fn list_audit_entry_details(
        &self,
        limit: Option<usize>,
        after_cursor: Option<&str>,
        filters: AuditEntryFilters,
    ) -> Result<Page<AuditEntryDetail>, LogStoreError> {
        let limit = validate_audit_entry_limit(limit)?;
        let cursor = after_cursor.map(decode_ordering_cursor).transpose()?;
        let (where_clause, parameters) = audit_entry_query_parts(cursor.as_ref(), filters)?;
        let sql = format!(
            "SELECT sequence, entry_id, request_id, occurred_at, actor, action, detail_json \
             FROM audit_entries {where_clause} \
             ORDER BY occurred_at DESC, entry_id DESC LIMIT {}",
            limit + 1
        );
        let conn = self.conn();
        let mut statement = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;
        let mut items: Vec<AuditEntryDetail> = statement
            .query_map(rusqlite::params_from_iter(parameters), audit_entry_detail)
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;

        let has_more = items.len() > limit;
        if has_more {
            items.pop();
        }
        let next_cursor = has_more.then(|| {
            let last = &items[items.len() - 1];
            encode_cursor(&last.entry.occurred_at, &last.entry.entry_id)
        });
        Ok(Page { items, next_cursor })
    }

    /// Read newly committed audit rows in database sequence order for the
    /// trusted-local event stream. Unlike the public keyset listing, this is a
    /// forward-only reconciliation cursor shared by every process writing the
    /// durable store.
    pub fn list_audit_entries_after_sequence(
        &self,
        after_sequence: u64,
        limit: usize,
        filters: AuditEntryFilters,
    ) -> Result<Vec<AuditEntryRow>, LogStoreError> {
        self.list_audit_entry_details_after_sequence(after_sequence, limit, filters)
            .map(|entries| entries.into_iter().map(Into::into).collect())
    }

    pub fn list_audit_entry_details_after_sequence(
        &self,
        after_sequence: u64,
        limit: usize,
        filters: AuditEntryFilters,
    ) -> Result<Vec<AuditEntryDetail>, LogStoreError> {
        let limit = validate_audit_entry_limit(Some(limit))?;
        let after_sequence = i64::try_from(after_sequence).map_err(|_| {
            LogStoreError::InvalidQuery("audit sequence is outside the supported range".to_string())
        })?;
        let (where_clause, mut parameters) = audit_entry_query_parts(None, filters)?;
        let predicate = if where_clause.is_empty() {
            "WHERE sequence > ?".to_string()
        } else {
            format!("{where_clause} AND sequence > ?")
        };
        parameters.push(Value::Integer(after_sequence));
        let sql = format!(
            "SELECT sequence, entry_id, request_id, occurred_at, actor, action, detail_json \
             FROM audit_entries {predicate} ORDER BY sequence ASC LIMIT {limit}"
        );
        let conn = self.conn();
        let mut statement = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;
        statement
            .query_map(rusqlite::params_from_iter(parameters), audit_entry_detail)
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
    }
}
