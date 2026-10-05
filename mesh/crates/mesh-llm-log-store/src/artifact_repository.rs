//! Durable artifact-pointer repository operations.

use crate::cursor::{decode_ordering_cursor, encode_cursor};
use crate::error::LogStoreError;
use crate::repositories::{Page, map_insert_constraint_error};
use crate::store::LogStore;
use crate::timestamps::canonical_persisted_timestamp;

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
    #[allow(dead_code)]
    pub unavailable_reason: Option<String>,
}

/// Metadata for an artifact body intentionally omitted by the capture owner.
pub struct UnavailableArtifactPointer<'a> {
    pub artifact_id: &'a str,
    pub request_id: &'a str,
    pub occurred_at: &'a str,
    pub kind: &'a str,
    pub media_kind: Option<&'a str>,
    pub version: u32,
    pub reason: &'a str,
}

fn artifact_pointer_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactPointerRow> {
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
        unavailable_reason: row.get(10)?,
    })
}

impl LogStore {
    /// Persist metadata for content that was intentionally not captured.
    ///
    /// `reason` is constrained by the schema to a closed, path-free
    /// vocabulary. No placeholder file or checksum is created.
    pub fn insert_unavailable_artifact_pointer(
        &self,
        pointer: UnavailableArtifactPointer<'_>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(pointer.occurred_at)?;
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO artifact_pointers \
             (artifact_id, request_id, occurred_at, kind, media_kind, version, redacted, unavailable_reason) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            rusqlite::params![
                pointer.artifact_id,
                pointer.request_id,
                occurred_at,
                pointer.kind,
                pointer.media_kind,
                i64::from(pointer.version),
                pointer.reason
            ],
        ) {
            Ok(_) => Ok(()),
            Err(ref error) => {
                match map_insert_constraint_error(
                    error,
                    format!("artifact_pointer {}", pointer.artifact_id),
                ) {
                    Some(mapped) => Err(mapped),
                    None => Err(LogStoreError::InsertFailed(error.to_string())),
                }
            }
        }
    }

    pub fn insert_artifact_pointer(
        &self,
        artifact_id: &str,
        request_id: &str,
        occurred_at: &str,
        kind: &str,
        metadata_json: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO artifact_pointers (artifact_id, request_id, occurred_at, kind, metadata_json) VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![artifact_id, request_id, occurred_at, kind, metadata_json],
        ) {
            Ok(_) => Ok(()),
            Err(ref error) => {
                match map_insert_constraint_error(error, format!("artifact_pointer {artifact_id}")) {
                    Some(mapped) => Err(mapped),
                    None => Err(LogStoreError::InsertFailed(error.to_string())),
                }
            }
        }
    }

    /// Update storage fields on an existing pointer row after file write.
    #[allow(clippy::too_many_arguments)]
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
        let mut statement = conn
            .prepare(
                "SELECT artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
                 version, redacted, truncated, unavailable_reason \
                 FROM artifact_pointers WHERE artifact_id = ?",
            )
            .map_err(LogStoreError::Sqlite)?;

        match statement.query_row(rusqlite::params![artifact_id], artifact_pointer_row) {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    /// List artifact pointers for a request_id. Returns rows in occurred_at ASC order.
    pub fn list_artifact_pointers_for_request(
        &self,
        request_id: &str,
    ) -> Result<Vec<ArtifactPointerRow>, LogStoreError> {
        let conn = self.conn();
        let mut statement = conn
            .prepare(
                "SELECT artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
                 version, redacted, truncated, unavailable_reason \
                 FROM artifact_pointers WHERE request_id = ? ORDER BY occurred_at ASC",
            )
            .map_err(LogStoreError::Sqlite)?;

        statement
            .query_map(rusqlite::params![request_id], artifact_pointer_row)
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
    }

    /// Sum of bytes column for all artifact pointers belonging to a request. Returns 0 if none.
    pub fn sum_artifact_bytes_for_request(&self, request_id: &str) -> Result<i64, LogStoreError> {
        let conn = self.conn();
        conn.query_row(
            "SELECT COALESCE(SUM(bytes), 0) FROM artifact_pointers WHERE request_id = ?",
            rusqlite::params![request_id],
            |row| row.get(0),
        )
        .map_err(LogStoreError::Sqlite)
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
        let columns = "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, \
                       version, redacted, truncated, unavailable_reason";

        let items = if let Some(cursor) = after_cursor {
            let (timestamp, artifact_id) = decode_ordering_cursor(cursor)?;
            let sql = format!(
                "SELECT {columns} FROM artifact_pointers \
                 WHERE (occurred_at, artifact_id) < (?, ?) \
                 ORDER BY occurred_at DESC, artifact_id DESC LIMIT {limit}"
            );
            let mut statement = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;
            statement
                .query_map(
                    rusqlite::params![timestamp, artifact_id],
                    artifact_pointer_row,
                )
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?
        } else {
            let sql = format!(
                "SELECT {columns} FROM artifact_pointers \
                 ORDER BY occurred_at DESC, artifact_id DESC LIMIT {limit}"
            );
            let mut statement = conn.prepare(&sql).map_err(LogStoreError::Sqlite)?;
            statement
                .query_map([], artifact_pointer_row)
                .map_err(LogStoreError::Sqlite)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?
        };

        let Some(last) = items.last() else {
            return Ok(Page {
                items,
                next_cursor: None,
            });
        };
        let has_more: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_pointers \
                 WHERE (occurred_at, artifact_id) < (?, ?) LIMIT 1)",
                rusqlite::params![&last.occurred_at, &last.artifact_id],
                |row| row.get::<_, i32>(0),
            )
            .map(|value| value != 0)
            .map_err(LogStoreError::Sqlite)?;

        let next_cursor = has_more.then(|| encode_cursor(&last.occurred_at, &last.artifact_id));
        Ok(Page { items, next_cursor })
    }

    /// Mark an artifact pointer as missing (file gone from disk).
    pub fn update_artifact_pointer_missing(
        &self,
        artifact_id: &str,
    ) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        conn.execute(
            "UPDATE artifact_pointers SET missing = 1 \
             WHERE artifact_id = ? AND unavailable_reason IS NULL",
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
}
