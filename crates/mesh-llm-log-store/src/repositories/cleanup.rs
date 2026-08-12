//! Transactional retention cleanup for durable logging repositories.

use rusqlite::Transaction;

use super::{CascadeArtifactPointer, LogStore, LogStoreError};
use crate::SQLiteSpaceMaintenance;

impl LogStore {
    /// Single-transaction cascade cleanup before a shared cutoff timestamp.
    pub fn cascade_cleanup_before(
        &self,
        cutoff_occurred_at: &str,
    ) -> Result<(i64, Vec<CascadeArtifactPointer>), LogStoreError> {
        self.cascade_cleanup_with_retention_cutoffs(
            cutoff_occurred_at,
            cutoff_occurred_at,
            cutoff_occurred_at,
        )
    }

    /// Cleanup application rows, audit entries, and webhook deliveries using
    /// their independently configured retention cutoffs.
    pub fn cascade_cleanup_with_retention_cutoffs(
        &self,
        application_cutoff: &str,
        audit_cutoff: &str,
        webhook_cutoff: &str,
    ) -> Result<(i64, Vec<CascadeArtifactPointer>), LogStoreError> {
        let result = self.txn(|transaction| {
            let queued = Self::queue_artifact_deletions(transaction, application_cutoff)?;
            let mut total = queued;
            total += Self::cleanup_application_rows(transaction, application_cutoff)?;
            total += Self::cleanup_table_before(transaction, "audit_entries", audit_cutoff)?;
            total += Self::cleanup_table_before(transaction, "webhook_deliveries", webhook_cutoff)?;
            let pending = Self::pending_artifact_deletions(transaction)?;
            Ok((total, pending))
        })?;
        if result.0 > 0 {
            return preserve_cleanup_result(Ok(result), self.maintain_space_after_cleanup());
        }
        Ok(result)
    }

    fn cleanup_application_rows(
        transaction: &Transaction<'_>,
        cutoff: &str,
    ) -> Result<i64, LogStoreError> {
        let mut total = 0;
        for table in ["lifecycle_events", "proxy_records"] {
            total += Self::cleanup_table_before(transaction, table, cutoff)?;
        }
        let orphans = transaction
            .execute(
                "DELETE FROM summaries \
                 WHERE request_id NOT IN (SELECT DISTINCT request_id FROM lifecycle_events) \
                 AND request_id NOT IN (SELECT DISTINCT request_id FROM artifact_pointers) \
                 AND created_at < ?",
                rusqlite::params![cutoff],
            )
            .map_err(LogStoreError::Sqlite)?;
        Ok(total + orphans as i64)
    }

    fn cleanup_table_before(
        transaction: &Transaction<'_>,
        table: &str,
        cutoff: &str,
    ) -> Result<i64, LogStoreError> {
        transaction
            .execute(
                &format!("DELETE FROM {table} WHERE occurred_at < ?"),
                rusqlite::params![cutoff],
            )
            .map(|deleted| deleted as i64)
            .map_err(LogStoreError::Sqlite)
    }

    fn queue_artifact_deletions(
        transaction: &Transaction<'_>,
        cutoff: &str,
    ) -> Result<i64, LogStoreError> {
        transaction
            .execute(
                "INSERT OR IGNORE INTO pending_artifact_deletions (artifact_id, request_id) \
                 SELECT artifact_id, request_id FROM artifact_pointers WHERE occurred_at < ?",
                rusqlite::params![cutoff],
            )
            .map_err(LogStoreError::Sqlite)?;
        let deleted = transaction
            .execute(
                "DELETE FROM artifact_pointers WHERE occurred_at < ?",
                rusqlite::params![cutoff],
            )
            .map_err(LogStoreError::Sqlite)? as i64;
        Ok(deleted)
    }

    fn pending_artifact_deletions(
        transaction: &Transaction<'_>,
    ) -> Result<Vec<CascadeArtifactPointer>, LogStoreError> {
        let mut statement = transaction
            .prepare(
                "SELECT artifact_id, request_id FROM pending_artifact_deletions \
                 ORDER BY artifact_id ASC",
            )
            .map_err(LogStoreError::Sqlite)?;
        statement
            .query_map([], |row| {
                Ok(CascadeArtifactPointer {
                    artifact_id: row.get(0)?,
                    request_id: row.get(1)?,
                })
            })
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
    }

    pub(crate) fn acknowledge_artifact_deletion(
        &self,
        pointer: &CascadeArtifactPointer,
    ) -> Result<(), LogStoreError> {
        self.conn()
            .execute(
                "DELETE FROM pending_artifact_deletions \
                 WHERE artifact_id = ? AND request_id = ?",
                rusqlite::params![pointer.artifact_id, pointer.request_id],
            )
            .map(|_| ())
            .map_err(LogStoreError::Sqlite)
    }
}

/// Logical cleanup commits before physical maintenance starts. Maintenance is
/// deliberately best-effort so its failure cannot make a caller retry a
/// deletion that already succeeded.
fn preserve_cleanup_result<T>(
    cleanup_result: Result<T, LogStoreError>,
    _maintenance_result: Result<SQLiteSpaceMaintenance, LogStoreError>,
) -> Result<T, LogStoreError> {
    cleanup_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_cleanup_result_survives_maintenance_failure() {
        let result =
            preserve_cleanup_result::<usize>(Ok(7), Err(LogStoreError::PrivacyNotGuaranteed));

        assert_eq!(result.expect("logical cleanup must remain successful"), 7);
    }
}
