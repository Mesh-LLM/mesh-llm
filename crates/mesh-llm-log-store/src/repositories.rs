//! Typed repositories for log-store persistence operations.

use crate::cursor::{decode_cursor, encode_cursor};
use crate::error::LogStoreError;
use crate::store::LogStore;

mod cleanup;

// ─── Row types returned by queries ──────────────────────

#[derive(Debug, Clone)]
pub struct SummaryRow {
    pub request_id: String,
    pub state: String,
    pub created_at: String,
    #[allow(dead_code)]
    pub terminal_at: Option<String>,
    #[allow(dead_code)]
    pub route: Option<String>,
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub provider: Option<String>,
    #[allow(dead_code)]
    pub engine: Option<String>,
    #[allow(dead_code)]
    pub status_code: Option<i64>,
    #[allow(dead_code)]
    pub error_msg: Option<String>,
    #[allow(dead_code)]
    pub tenant_id: Option<String>,
    #[allow(dead_code)]
    pub account_id: Option<String>,
    #[allow(dead_code)]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LifecycleEventRow {
    pub event_id: String,
    pub request_id: String,
    pub occurred_at: String,
}

#[derive(Debug, Clone)]
pub struct ArtifactPointerRow {
    pub artifact_id: String,
    #[allow(dead_code)]
    pub request_id: String,
    #[allow(dead_code)]
    pub occurred_at: String,
    #[allow(dead_code)]
    pub kind: String,
    #[allow(dead_code)]
    pub media_kind: Option<String>,
    #[allow(dead_code)]
    pub checksum: Option<String>,
    #[allow(dead_code)]
    pub bytes: i64,
    #[allow(dead_code)]
    pub version: i32,
    #[allow(dead_code)]
    pub redacted: bool,
    #[allow(dead_code)]
    pub truncated: bool,
}

/// A file artifact queued durably for post-commit deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadeArtifactPointer {
    pub artifact_id: String,
    pub request_id: String,
}

/// Paginated query result with an optional cursor for the next page.
#[derive(Debug)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
}

// ─── Internal helpers ──────────────

fn is_unique_constraint_error(e: &rusqlite::Error) -> bool {
    if let rusqlite::Error::SqliteFailure(err, _) = e {
        matches!(
            err.extended_code,
            rusqlite::ffi::SQLITE_CONSTRAINT_PRIMARYKEY | rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
        )
    } else {
        false
    }
}

fn is_foreign_key_constraint_error(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(err, _)
            if err.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FOREIGNKEY
    )
}

fn map_insert_constraint_error(e: &rusqlite::Error, entity: String) -> Option<LogStoreError> {
    if is_foreign_key_constraint_error(e) {
        Some(LogStoreError::ForeignKeyViolation { entity })
    } else if is_unique_constraint_error(e) {
        Some(LogStoreError::AlreadyExists { entity })
    } else {
        None
    }
}

/// Check whether a payload JSON string represents a terminal event type.
fn is_terminal_payload(payload_json: &str) -> bool {
    payload_json.contains(r#""type":"completed""#)
        || payload_json.contains(r#""type":"failed""#)
        || payload_json.contains(r#""type":"rejected""#)
        || payload_json.contains(r#""type":"cancelled""#)
        || payload_json.contains(r#""type":"dropped""#)
}

/// Check if a request already has any terminal event. Works on Connection or Transaction.
fn check_existing_terminal_raw(
    cxn: &rusqlite::Connection,
    request_id: &str,
) -> Result<bool, LogStoreError> {
    let count: i64 = cxn
        .query_row(
            "SELECT COUNT(*) FROM lifecycle_events \
         WHERE request_id = ? \
         AND (payload_json LIKE '%\"type\":\"completed\"%' \
            OR payload_json LIKE '%\"type\":\"failed\"%' \
            OR payload_json LIKE '%\"type\":\"rejected\"%' \
            OR payload_json LIKE '%\"type\":\"cancelled\"%' \
            OR payload_json LIKE '%\"type\":\"dropped\"%')",
            rusqlite::params![request_id],
            |row| row.get(0),
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(count > 0)
}

// ─── LogStore repository methods ──────────────

impl LogStore {
    // ════════════════════════════
    //  Summaries
    // ════════════════════════════

    /// Insert a new summary. Returns AlreadyExists on duplicate PK.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_summary(
        &self,
        request_id: &str,
        model: Option<&str>,
        route: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
        occurred_at: &str,
        tenant_id: Option<&str>,
        account_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO summaries (request_id, created_at, model, route, provider, engine, tenant_id, account_id, user_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![request_id, occurred_at, model, route, provider, engine, tenant_id, account_id, user_id],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("summary {}", request_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    /// Get a summary by request_id. Returns None if not found (no-op style).
    pub fn get_summary(&self, request_id: &str) -> Result<Option<SummaryRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT request_id, state, created_at, terminal_at, route, model, provider, engine, \
             status_code, error_msg, tenant_id, account_id, user_id \
             FROM summaries WHERE request_id = ?",
        ).map_err(LogStoreError::Sqlite)?;

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SummaryRow> {
            Ok(SummaryRow {
                request_id: row.get(0)?,
                state: row.get(1)?,
                created_at: row.get(2)?,
                terminal_at: row.get(3)?,
                route: row.get(4)?,
                model: row.get(5)?,
                provider: row.get(6)?,
                engine: row.get(7)?,
                status_code: row.get(8)?,
                error_msg: row.get(9)?,
                tenant_id: row.get(10)?,
                account_id: row.get(11)?,
                user_id: row.get(12)?,
            })
        };

        match stmt.query_row(rusqlite::params![request_id], row_fn) {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(LogStoreError::QueryFailed(e.to_string())),
        }
    }

    /// Paginated summary listing keyed on (created_at, request_id).
    pub fn list_summaries(
        &self,
        limit: usize,
        after_cursor: Option<&str>,
    ) -> Result<Page<SummaryRow>, LogStoreError> {
        let conn = self.conn();

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SummaryRow> {
            Ok(SummaryRow {
                request_id: row.get(0)?,
                state: row.get(1)?,
                created_at: row.get(2)?,
                terminal_at: row.get(3)?,
                route: row.get(4)?,
                model: row.get(5)?,
                provider: row.get(6)?,
                engine: row.get(7)?,
                status_code: row.get(8)?,
                error_msg: row.get(9)?,
                tenant_id: row.get(10)?,
                account_id: row.get(11)?,
                user_id: row.get(12)?,
            })
        };

        if let Some(cursor_str) = after_cursor {
            let (ts, id) = decode_cursor(cursor_str)?;

            // Fetch exactly `limit` rows with cursor boundary.
            let sql = format!(
                "SELECT request_id, state, created_at, terminal_at, route, model, provider, engine, \
                 status_code, error_msg, tenant_id, account_id, user_id \
                 FROM summaries WHERE (created_at, request_id) < (?, ?) ORDER BY created_at DESC, request_id DESC LIMIT {}",
                limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            // Collect exactly `limit` rows.
            let items: Vec<SummaryRow> = stmt
                .query_map(rusqlite::params![ts, id], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe: is there at least one more row beyond the last returned item?
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM summaries \
                             WHERE (created_at, request_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.created_at, &last.request_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.created_at, &last.request_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        } else {
            // First page: fetch exactly `limit` rows, then probe separately for more.
            let sql = format!(
                "SELECT request_id, state, created_at, terminal_at, route, model, provider, engine, \
                 status_code, error_msg, tenant_id, account_id, user_id \
                 FROM summaries ORDER BY created_at DESC, request_id DESC LIMIT {}",
                limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            let items: Vec<SummaryRow> = stmt
                .query_map(rusqlite::params![], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe for more rows beyond the last returned item.
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM summaries \
                              WHERE (created_at, request_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.created_at, &last.request_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.created_at, &last.request_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        }
    }

    /// Update summary terminal state. No-op if request_id not found (returns 0 rows affected).
    pub fn update_summary_terminal(
        &self,
        request_id: &str,
        terminal_status: &str,
        terminal_at: &str,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "UPDATE summaries SET state = ?, terminal_at = ? WHERE request_id = ?",
            rusqlite::params![terminal_status, terminal_at, request_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    // ════════════════════════════
    //  Lifecycle Events
    // ════════════════════════════

    /// Insert a lifecycle event. Caller serializes the payload to JSON before calling.
    pub fn insert_lifecycle_event(
        &self,
        request_id: &str,
        event_id: &str,
        payload_json: &str,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();

        // Pre-check for terminal duplicates.
        if is_terminal_payload(payload_json) && check_existing_terminal_raw(&conn, request_id)? {
            return Err(LogStoreError::DuplicateTerminalEvent {
                summary_id: request_id.to_string(),
                event_type: payload_json.to_string(),
            });
        }

        match conn.execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
            rusqlite::params![event_id, request_id, occurred_at, payload_json],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) if is_foreign_key_constraint_error(e) => {
                Err(LogStoreError::ForeignKeyViolation {
                    entity: "lifecycle_event".to_string(),
                })
            }
            Err(ref e) if is_unique_constraint_error(e) => {
                // Could be UNIQUE(request_id, event_id) or the partial terminal index.
                if is_terminal_payload(payload_json) {
                    return Err(LogStoreError::DuplicateTerminalEvent {
                        summary_id: request_id.to_string(),
                        event_type: payload_json.to_string(),
                    });
                }
                Err(LogStoreError::AlreadyExists {
                    entity: "lifecycle_event".to_string(),
                })
            }
            Err(e) => Err(LogStoreError::InsertFailed(e.to_string())),
        }
    }

    /// Atomic write of a terminal event + summary state update. Both succeed or neither does.
    pub fn write_terminal_event(
        &self,
        request_id: &str,
        event_id: &str,
        payload_json: &str,
        terminal_status: &str,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        self.txn(|tx| {
            let has_terminal = tx.query_row(
                "SELECT COUNT(*) FROM lifecycle_events \
                 WHERE request_id = ? \
                 AND (payload_json LIKE '%\"type\":\"completed\"%' \
                    OR payload_json LIKE '%\"type\":\"failed\"%' \
                    OR payload_json LIKE '%\"type\":\"rejected\"%' \
                    OR payload_json LIKE '%\"type\":\"cancelled\"%' \
                    OR payload_json LIKE '%\"type\":\"dropped\"%')",
                rusqlite::params![request_id],
                |row| row.get::<_, i64>(0),
            ).map(|c: i64| c > 0).map_err(LogStoreError::Sqlite)?;

            if has_terminal {
                return Err(LogStoreError::DuplicateTerminalEvent {
                    summary_id: request_id.to_string(),
                    event_type: payload_json.to_string(),
                });
            }

            tx.execute(
                "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
                rusqlite::params![event_id, request_id, occurred_at, payload_json],
            ).map_err(LogStoreError::Sqlite)?;

            tx.execute(
                "UPDATE summaries SET state = ?, terminal_at = ? WHERE request_id = ?",
                rusqlite::params![terminal_status, occurred_at, request_id],
            ).map_err(LogStoreError::Sqlite)?;

            Ok(()) as Result<(), LogStoreError>
        })
    }

    /// Check if a summary already has any terminal event.
    pub fn has_terminal_event(&self, request_id: &str) -> Result<bool, LogStoreError> {
        let conn = self.conn();
        check_existing_terminal_raw(&conn, request_id)
    }

    /// Paginated lifecycle event listing keyed on (occurred_at, event_id).
    pub fn list_lifecycle_events(
        &self,
        limit: usize,
        after_cursor: Option<&str>,
    ) -> Result<Page<LifecycleEventRow>, LogStoreError> {
        let conn = self.conn();

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<LifecycleEventRow> {
            Ok(LifecycleEventRow {
                event_id: row.get(0)?,
                request_id: row.get(1)?,
                occurred_at: row.get(2)?,
            })
        };

        if let Some(cursor_str) = after_cursor {
            let (ts, id) = decode_cursor(cursor_str)?;

            // Fetch exactly `limit` rows with cursor boundary.
            let sql = format!(
                "SELECT event_id, request_id, occurred_at FROM lifecycle_events \
                 WHERE (occurred_at, event_id) < (?, ?) ORDER BY occurred_at DESC, event_id DESC LIMIT {}",
                limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            // Collect exactly `limit` rows.
            let items: Vec<LifecycleEventRow> = stmt
                .query_map(rusqlite::params![ts, id], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe: is there at least one more row beyond the last returned item?
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM lifecycle_events \
                             WHERE (occurred_at, event_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.occurred_at, &last.event_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.occurred_at, &last.event_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        } else {
            // No cursor: fetch first page of exactly `limit` rows, then probe.
            let sql = format!(
                "SELECT event_id, request_id, occurred_at FROM lifecycle_events \
                 ORDER BY occurred_at DESC, event_id DESC LIMIT {}",
                limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            let items: Vec<LifecycleEventRow> = stmt
                .query_map(rusqlite::params![], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe for more rows beyond the last returned item.
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM lifecycle_events \
                              WHERE (occurred_at, event_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.occurred_at, &last.event_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.occurred_at, &last.event_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        }
    }

    /// List all lifecycle events for a specific summary, ordered chronologically.
    pub fn list_events_for_summary(
        &self,
        request_id: &str,
    ) -> Result<Vec<LifecycleEventRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT event_id, request_id, occurred_at FROM lifecycle_events \
             WHERE request_id = ? ORDER BY occurred_at ASC",
            )
            .map_err(LogStoreError::Sqlite)?;

        let rows_iter = stmt
            .query_map(rusqlite::params![request_id], |row: &rusqlite::Row<'_>| {
                Ok(LifecycleEventRow {
                    event_id: row.get(0)?,
                    request_id: row.get(1)?,
                    occurred_at: row.get(2)?,
                })
            })
            .map_err(LogStoreError::Sqlite)?;

        let mut items = Vec::new();
        for result in rows_iter {
            match result {
                Ok(item) => items.push(item),
                Err(e) => return Err(LogStoreError::QueryFailed(e.to_string())),
            }
        }
        Ok(items)
    }

    // ════════════════════════════
    //  Artifact Pointers
    // ════════════════════════════

    pub fn insert_artifact_pointer(
        &self,
        artifact_id: &str,
        request_id: &str,
        occurred_at: &str,
        kind: &str,
        metadata_json: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO artifact_pointers (artifact_id, request_id, occurred_at, kind, metadata_json) VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![artifact_id, request_id, occurred_at, kind, metadata_json],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("artifact_pointer {}", artifact_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    /// Update storage fields on an existing pointer row after file write.
    #[allow(clippy::too_many_arguments)] // mirrors DB columns; grouping into a struct adds no clarity for single caller
    pub fn update_artifact_pointer_storage(
        &self,
        artifact_id: &str,
        media_kind: Option<&str>,
        checksum: &str,
        bytes: i64,
        version: i32,
        redacted: bool,
        truncated: bool,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "UPDATE artifact_pointers \
             SET media_kind = ?, checksum = ?, bytes = ?, version = ?, \
                 redacted = ?, truncated = ? \
             WHERE artifact_id = ?",
            rusqlite::params![
                media_kind,
                checksum,
                bytes,
                version,
                redacted as i32,
                truncated as i32,
                artifact_id
            ],
        )
        .map_err(LogStoreError::Sqlite)
    }

    /// Get a single artifact pointer row by artifact_id. Returns None if not found.
    pub fn get_artifact_pointer(
        &self,
        artifact_id: &str,
    ) -> Result<Option<ArtifactPointerRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
             version, redacted, truncated \
             FROM artifact_pointers WHERE artifact_id = ?",
            )
            .map_err(LogStoreError::Sqlite)?;

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ArtifactPointerRow> {
            Ok(ArtifactPointerRow {
                artifact_id: row.get(0)?,
                request_id: row.get(1)?,
                occurred_at: row.get(2)?,
                kind: row.get(3)?,
                media_kind: row.get(4)?,
                checksum: row.get(5)?,
                bytes: row.get::<_, i64>(6)?,
                version: row.get::<_, i32>(7)?,
                redacted: row.get::<_, i32>(8)? != 0,
                truncated: row.get::<_, i32>(9)? != 0,
            })
        };

        match stmt.query_row(rusqlite::params![artifact_id], row_fn) {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(LogStoreError::QueryFailed(e.to_string())),
        }
    }

    /// List artifact pointers for a request_id. Returns rows in occurred_at ASC order.
    pub fn list_artifact_pointers_for_request(
        &self,
        request_id: &str,
    ) -> Result<Vec<ArtifactPointerRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
             version, redacted, truncated \
             FROM artifact_pointers WHERE request_id = ? ORDER BY occurred_at ASC",
            )
            .map_err(LogStoreError::Sqlite)?;

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ArtifactPointerRow> {
            Ok(ArtifactPointerRow {
                artifact_id: row.get(0)?,
                request_id: row.get(1)?,
                occurred_at: row.get(2)?,
                kind: row.get(3)?,
                media_kind: row.get(4)?,
                checksum: row.get(5)?,
                bytes: row.get::<_, i64>(6)?,
                version: row.get::<_, i32>(7)?,
                redacted: row.get::<_, i32>(8)? != 0,
                truncated: row.get::<_, i32>(9)? != 0,
            })
        };

        let rows_iter = stmt
            .query_map(rusqlite::params![request_id], &row_fn)
            .map_err(LogStoreError::Sqlite)?;

        let mut items = Vec::new();
        for result in rows_iter {
            match result {
                Ok(item) => items.push(item),
                Err(e) => return Err(LogStoreError::QueryFailed(e.to_string())),
            }
        }
        Ok(items)
    }

    /// Sum of bytes column for all artifact pointers belonging to a request. Returns 0 if none.
    pub fn sum_artifact_bytes_for_request(&self, request_id: &str) -> Result<i64, LogStoreError> {
        let conn = self.conn();
        let total: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(bytes), 0) FROM artifact_pointers WHERE request_id = ?",
                rusqlite::params![request_id],
                |row| row.get(0),
            )
            .map_err(LogStoreError::Sqlite)?;
        Ok(total)
    }

    /// Delete a single artifact pointer row by ID. Returns rows affected (0 or 1).
    pub fn delete_artifact_pointer_row(&self, artifact_id: &str) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "DELETE FROM artifact_pointers WHERE artifact_id = ?",
            rusqlite::params![artifact_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    /// Delete all artifact pointer rows for a request. Returns count of deleted rows.
    pub fn delete_artifact_pointer_rows_for_request(
        &self,
        request_id: &str,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "DELETE FROM artifact_pointers WHERE request_id = ?",
            rusqlite::params![request_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    /// Paginated listing keyed on (occurred_at, artifact_id).
    pub fn list_artifact_pointers(
        &self,
        limit: usize,
        after_cursor: Option<&str>,
    ) -> Result<Page<ArtifactPointerRow>, LogStoreError> {
        let conn = self.conn();

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ArtifactPointerRow> {
            Ok(ArtifactPointerRow {
                artifact_id: row.get(0)?,
                request_id: row.get(1)?,
                occurred_at: row.get(2)?,
                kind: row.get(3)?,
                media_kind: row.get(4)?,
                checksum: row.get(5)?,
                bytes: row.get::<_, i64>(6)?,
                version: row.get::<_, i32>(7)?,
                redacted: row.get::<_, i32>(8)? != 0,
                truncated: row.get::<_, i32>(9)? != 0,
            })
        };

        let cols = "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
                    version, redacted, truncated";

        if let Some(cursor_str) = after_cursor {
            let (ts, id) = decode_cursor(cursor_str)?;

            // Fetch exactly `limit` rows with cursor boundary.
            let sql = format!(
                "SELECT {} FROM artifact_pointers \
                 WHERE (occurred_at, artifact_id) < (?, ?) ORDER BY occurred_at DESC, artifact_id DESC LIMIT {}",
                cols, limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            // Collect exactly `limit` rows.
            let items: Vec<ArtifactPointerRow> = stmt
                .query_map(rusqlite::params![ts, id], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe: is there at least one more row beyond the last returned item?
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM artifact_pointers \
                             WHERE (occurred_at, artifact_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.occurred_at, &last.artifact_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.occurred_at, &last.artifact_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        } else {
            // No cursor: fetch first page of exactly `limit` rows, then probe.
            let sql = format!(
                "SELECT {} FROM artifact_pointers \
                 ORDER BY occurred_at DESC, artifact_id DESC LIMIT {}",
                cols, limit
            );
            let mut stmt = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;

            let items: Vec<ArtifactPointerRow> = stmt
                .query_map(rusqlite::params![], &row_fn)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| LogStoreError::QueryFailed(e.to_string()))?;

            if items.is_empty() {
                return Ok(Page {
                    items,
                    next_cursor: None,
                });
            }

            // Probe for more rows beyond the last returned item.
            let last = &items[items.len() - 1];
            let probe_sql = "SELECT EXISTS(SELECT 1 FROM artifact_pointers \
                              WHERE (occurred_at, artifact_id) < (?, ?) LIMIT 1)";
            let has_more: bool = conn
                .query_row(
                    probe_sql,
                    rusqlite::params![&last.occurred_at, &last.artifact_id],
                    |r| r.get::<_, i32>(0),
                )
                .map(|v| v != 0)
                .map_err(LogStoreError::Sqlite)?;

            let next_cursor = if has_more {
                Some(encode_cursor(&last.occurred_at, &last.artifact_id))
            } else {
                None
            };

            Ok(Page { items, next_cursor })
        }
    }

    // ════════════════════════════
    //  Proxy Records
    // ════════════════════════════

    #[allow(clippy::too_many_arguments)]
    pub fn insert_proxy_record(
        &self,
        attempt_id: &str,
        request_id: &str,
        occurred_at: &str,
        target: &str,
        provider: Option<&str>,
        engine: Option<&str>,
        started_at: Option<&str>,
        completed_at: Option<&str>,
        status_code: Option<i64>,
        error_msg: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO proxy_records (attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code, error_msg) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code, error_msg],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("proxy_record {}", attempt_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    // ════════════════════════════
    //  Audit Entries
    // ════════════════════════════

    pub fn insert_audit_entry(
        &self,
        entry_id: &str,
        request_id: Option<&str>,
        occurred_at: &str,
        actor: &str,
        action: &str,
        detail_json: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action, detail_json) VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![entry_id, request_id, occurred_at, actor, action, detail_json],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("audit_entry {}", entry_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    // ════════════════════════════
    //  Webhook Deliveries
    // ════════════════════════════

    #[allow(clippy::too_many_arguments)]
    pub fn insert_webhook_delivery(
        &self,
        delivery_id: &str,
        request_id: Option<&str>,
        occurred_at: &str,
        target_url: &str,
        attempt_number: i64,
        status_code: Option<i64>,
        response_body: Option<&str>,
        error_msg: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO webhook_deliveries (delivery_id, request_id, occurred_at, target_url, attempt_number, status_code, response_body, error_msg) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![delivery_id, request_id, occurred_at, target_url, attempt_number, status_code, response_body, error_msg],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("webhook_delivery {}", delivery_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    // ════════════════════════════
    //  Cleanup Runs
    // ════════════════════════════

    pub fn insert_cleanup_run(
        &self,
        run_id: &str,
        occurred_at: &str,
        policy_name: &str,
        cutoff_before: &str,
        deleted_count: i64,
        duration_ms: Option<i64>,
    ) -> Result<(), LogStoreError> {
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO cleanup_runs (run_id, occurred_at, policy_name, cutoff_before, deleted_count, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![run_id, occurred_at, policy_name, cutoff_before, deleted_count, duration_ms],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("cleanup_run {}", run_id)) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    // ════════════════════════════
    //  Aggregation Queries
    // ════════════════════════════

    /// Count summaries grouped by state.
    pub fn count_summaries_by_status(&self) -> Result<Vec<(String, i64)>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT state, COUNT(*) FROM summaries GROUP BY state")
            .map_err(LogStoreError::Sqlite)?;

        let rows_iter = stmt
            .query_map([], |row: &rusqlite::Row<'_>| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(LogStoreError::Sqlite)?;

        let mut items = Vec::new();
        for result in rows_iter {
            match result {
                Ok(item) => items.push(item),
                Err(e) => return Err(LogStoreError::QueryFailed(e.to_string())),
            }
        }
        Ok(items)
    }

    // ════════════════════════════
    //  Artifact Pointer Status Updates
    // ════════════════════════════

    /// Mark an artifact pointer as missing (file gone from disk).
    pub fn update_artifact_pointer_missing(
        &self,
        artifact_id: &str,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "UPDATE artifact_pointers SET missing = 1 WHERE artifact_id = ?",
            rusqlite::params![artifact_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    /// Mark an artifact pointer as corrupt (checksum mismatch).
    pub fn update_artifact_pointer_corrupt(
        &self,
        artifact_id: &str,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "UPDATE artifact_pointers SET corrupt = 1 WHERE artifact_id = ?",
            rusqlite::params![artifact_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    // ════════════════════════════
    //  Test Helpers
    // ════════════════════════════

    #[cfg(test)]
    pub fn count_table(&self, table_name: &str) -> Result<i64, LogStoreError> {
        let conn = self.conn();
        let sql = format!("SELECT COUNT(*) FROM {}", table_name);
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .map_err(LogStoreError::Sqlite)
    }

    #[cfg(test)]
    pub fn clear_table(&self, table_name: &str) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        let sql = format!("DELETE FROM {}", table_name);
        conn.execute(&sql, []).map_err(LogStoreError::Sqlite)
    }
}
