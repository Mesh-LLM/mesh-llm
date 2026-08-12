//! Transactional retention cleanup for durable logging repositories.

use std::collections::BTreeMap;

use rusqlite::Transaction;

use super::{CascadeArtifactPointer, LogStore, LogStoreError};
use crate::SQLiteSpaceMaintenance;
use crate::timestamps::canonical_comparison_timestamp;

/// A single retention pass never deletes more than this many terminal
/// summaries. Further passes resume from the same deterministic oldest-first
/// ordering, keeping SQLite work and post-commit artifact file cleanup bounded.
const MAX_SUMMARIES_PER_CAP_PRUNE: i64 = 1_000;
/// Keep dynamically generated `IN` predicates below SQLite's conservative
/// bind-variable floor while replacing per-owner retention queries with a
/// bounded set-based operation.
const RETENTION_OWNER_BATCH_SIZE: usize = 250;
/// Pending filesystem work is replayed in deterministic bounded batches.
const MAX_PENDING_ARTIFACT_DELETIONS_PER_BATCH: i64 = 1_000;
/// Durable tables governed by the logging retention policy.  These stable,
/// path-free names are suitable for cleanup receipts and local health views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RetentionTable {
    Summaries,
    LifecycleEvents,
    ArtifactPointers,
    ProxyRecords,
    AuditEntries,
    WebhookDeliveries,
    CleanupRuns,
}

impl RetentionTable {
    pub const ALL: [Self; 7] = [
        Self::Summaries,
        Self::LifecycleEvents,
        Self::ArtifactPointers,
        Self::ProxyRecords,
        Self::AuditEntries,
        Self::WebhookDeliveries,
        Self::CleanupRuns,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Summaries => "summaries",
            Self::LifecycleEvents => "lifecycle_events",
            Self::ArtifactPointers => "artifact_pointers",
            Self::ProxyRecords => "proxy_records",
            Self::AuditEntries => "audit_entries",
            Self::WebhookDeliveries => "webhook_deliveries",
            Self::CleanupRuns => "cleanup_runs",
        }
    }
}

/// Bounded retention settings for one durable table.  A cutoff is used rather
/// than a duration so the store remains deterministic under an injected clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionTablePolicy {
    pub cutoff_occurred_at: String,
    pub max_rows: u64,
}

/// Explicit policy map for every durable logging table.  The constructor
/// rejects missing tables, so adding a table cannot silently make it
/// unbounded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionPolicy {
    table_policies: BTreeMap<RetentionTable, RetentionTablePolicy>,
    webhook_dead_letter_cutoff_at: Option<String>,
}

impl RetentionPolicy {
    pub fn new(
        mut table_policies: BTreeMap<RetentionTable, RetentionTablePolicy>,
    ) -> Result<Self, LogStoreError> {
        for table in RetentionTable::ALL {
            let Some(policy) = table_policies.get_mut(&table) else {
                return Err(LogStoreError::QueryFailed(format!(
                    "logging retention policy is missing {}",
                    table.label()
                )));
            };
            if policy.max_rows == 0 {
                return Err(LogStoreError::QueryFailed(format!(
                    "logging retention max rows for {} must be at least one",
                    table.label()
                )));
            }
            policy.cutoff_occurred_at = canonical_comparison_timestamp(&policy.cutoff_occurred_at)?;
        }
        Ok(Self {
            table_policies,
            webhook_dead_letter_cutoff_at: None,
        })
    }

    /// Compatibility policy for the existing global config.  It deliberately
    /// expands that config into an explicit, complete map rather than allowing
    /// standalone audit, webhook, or cleanup tables to become unbounded.
    pub fn uniform(
        cutoff_occurred_at: impl Into<String>,
        max_rows: u64,
    ) -> Result<Self, LogStoreError> {
        let cutoff_occurred_at = cutoff_occurred_at.into();
        let table_policies = RetentionTable::ALL
            .into_iter()
            .map(|table| {
                (
                    table,
                    RetentionTablePolicy {
                        cutoff_occurred_at: cutoff_occurred_at.clone(),
                        max_rows,
                    },
                )
            })
            .collect();
        Self::new(table_policies)
    }

    pub fn table(&self, table: RetentionTable) -> &RetentionTablePolicy {
        // `new` proves complete coverage and this is private-state immutable.
        &self.table_policies[&table]
    }

    /// Add a dead-letter-only cutoff to generic webhook retention.
    ///
    /// `updated_at` is written by every transition into `dead_letter`, so it
    /// is the durable dead-letter transition timestamp. All other delivery
    /// states are unaffected by this additional cutoff.
    pub fn with_webhook_dead_letter_cutoff(mut self, cutoff_updated_at: impl Into<String>) -> Self {
        self.webhook_dead_letter_cutoff_at = Some(cutoff_updated_at.into());
        self
    }

    fn webhook_dead_letter_cutoff_at(&self) -> Option<&str> {
        self.webhook_dead_letter_cutoff_at.as_deref()
    }
}

/// Per-table deletion counts from one committed policy pass.  Names are
/// schema labels only; no filesystem location or artifact path is exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionTableResult {
    pub table: RetentionTable,
    pub ttl_deleted_count: i64,
    pub max_rows_deleted_count: i64,
}

/// The committed outcome of one bounded retention pass.
///
/// Artifact pointers are loaded from the durable post-commit deletion queue.
/// This includes work committed by an earlier process, so callers can retry
/// file reconciliation after a crash without rediscovering deleted owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionCleanupResult {
    pub ttl_deleted_count: i64,
    pub max_rows_deleted_count: i64,
    pub artifact_pointers: Vec<CascadeArtifactPointer>,
    pub table_results: Vec<RetentionTableResult>,
}

impl LogStore {
    // ════════════════════════════
    //  Cascade Cleanup
    // ════════════════════════════

    /// Compatibility entry point for the safe TTL half of retention.
    ///
    /// Unlike the former timestamp-only implementation, this never deletes an
    /// artifact pointer independently of its request summary.  Callers get
    /// only pointers whose owning terminal summary was removed transactionally.
    /// The row cap is deliberately disabled here; runtime retention supplies
    /// its configured bounded cap through [`Self::apply_retention_policy`].
    pub fn cascade_cleanup_before(
        &self,
        cutoff_occurred_at: &str,
    ) -> Result<(i64, Vec<CascadeArtifactPointer>), LogStoreError> {
        let result = self.apply_retention_policy(cutoff_occurred_at, i64::MAX as u64)?;
        Ok((result.ttl_deleted_count, result.artifact_pointers))
    }

    /// Compatibility entry point for independently configured application,
    /// audit, and webhook retention cutoffs. Request-owned rows still use the
    /// terminal-summary cascade rules enforced by the complete policy map.
    pub fn cascade_cleanup_with_retention_cutoffs(
        &self,
        application_cutoff: &str,
        audit_cutoff: &str,
        webhook_cutoff: &str,
    ) -> Result<(i64, Vec<CascadeArtifactPointer>), LogStoreError> {
        let table_policies = RetentionTable::ALL
            .into_iter()
            .map(|table| {
                let cutoff_occurred_at = match table {
                    RetentionTable::AuditEntries => audit_cutoff,
                    RetentionTable::WebhookDeliveries => webhook_cutoff,
                    _ => application_cutoff,
                };
                (
                    table,
                    RetentionTablePolicy {
                        cutoff_occurred_at: cutoff_occurred_at.to_string(),
                        max_rows: i64::MAX as u64,
                    },
                )
            })
            .collect();
        let result = self.apply_retention_policy_map(&RetentionPolicy::new(table_policies)?)?;
        Ok((result.ttl_deleted_count, result.artifact_pointers))
    }

    /// Apply the durable logging retention policy in one transaction.
    ///
    /// A terminal summary owns its lifecycle, proxy, and artifact-pointer
    /// children.  Time-to-live and row-cap selection therefore delete those
    /// summaries first and rely on foreign-key cascade for their children.
    /// This prevents an old artifact pointer from removing its file while the
    /// summary that still references it remains available.  Active summaries
    /// and all of their owned rows are deliberately retained.
    ///
    /// The backwards-compatible config pair is expanded into a complete map:
    /// independent audit, webhook, and cleanup receipt rows are each capped;
    /// request-owned lifecycle, proxy, and artifact rows use terminal-owner
    /// cascade selection so their parent/detail invariants remain intact.
    pub fn apply_retention_policy(
        &self,
        cutoff_occurred_at: &str,
        max_terminal_summaries: u64,
    ) -> Result<RetentionCleanupResult, LogStoreError> {
        self.apply_retention_policy_map(&RetentionPolicy::uniform(
            cutoff_occurred_at,
            max_terminal_summaries,
        )?)
    }

    /// Apply a complete per-table retention policy transactionally.  Request
    /// owned rows are never removed from an active summary.  Artifact-pointer
    /// TTL/caps select a terminal owner summary for cascade deletion rather
    /// than deleting a pointer/file independently.
    pub fn apply_retention_policy_map(
        &self,
        policy: &RetentionPolicy,
    ) -> Result<RetentionCleanupResult, LogStoreError> {
        let webhook_dead_letter_cutoff = policy
            .webhook_dead_letter_cutoff_at()
            .map(canonical_comparison_timestamp)
            .transpose()?;
        let result = self.txn(|tx| {
            let mut results = RetentionTable::ALL
                .into_iter()
                .map(|table| {
                    (
                        table,
                        RetentionTableResult {
                            table,
                            ttl_deleted_count: 0,
                            max_rows_deleted_count: 0,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let mut artifact_pointers = Vec::new();

            Self::apply_summary_owner_ttl(
                tx,
                policy.table(RetentionTable::Summaries),
                &mut results,
                &mut artifact_pointers,
            )?;
            Self::apply_artifact_owner_ttl(
                tx,
                policy.table(RetentionTable::ArtifactPointers),
                &mut results,
                &mut artifact_pointers,
            )?;
            Self::apply_terminal_owned_ttl(
                tx,
                RetentionTable::LifecycleEvents,
                policy.table(RetentionTable::LifecycleEvents),
                &mut results,
                &mut artifact_pointers,
            )?;
            Self::apply_terminal_owned_ttl(
                tx,
                RetentionTable::ProxyRecords,
                policy.table(RetentionTable::ProxyRecords),
                &mut results,
                &mut artifact_pointers,
            )?;
            for table in [
                RetentionTable::AuditEntries,
                RetentionTable::WebhookDeliveries,
                RetentionTable::CleanupRuns,
            ] {
                Self::apply_standalone_ttl(tx, table, policy.table(table), &mut results)?;
            }
            if let Some(cutoff_updated_at) = webhook_dead_letter_cutoff.as_deref() {
                Self::apply_webhook_dead_letter_ttl(tx, cutoff_updated_at, &mut results)?;
            }

            Self::apply_summary_owner_cap(
                tx,
                policy.table(RetentionTable::Summaries),
                &mut results,
                &mut artifact_pointers,
            )?;
            Self::apply_artifact_owner_cap(
                tx,
                policy.table(RetentionTable::ArtifactPointers),
                &mut results,
                &mut artifact_pointers,
            )?;
            for table in [
                RetentionTable::LifecycleEvents,
                RetentionTable::ProxyRecords,
            ] {
                Self::apply_terminal_owned_cap(
                    tx,
                    table,
                    policy.table(table),
                    &mut results,
                    &mut artifact_pointers,
                )?;
            }
            for table in [
                RetentionTable::AuditEntries,
                RetentionTable::WebhookDeliveries,
                RetentionTable::CleanupRuns,
            ] {
                Self::apply_standalone_cap(tx, table, policy.table(table), &mut results)?;
            }

            // The per-phase accumulator documents which owners were selected
            // by this pass. Filesystem work itself always comes from the
            // durable queue so an earlier committed pass is replayable.
            drop(artifact_pointers);
            let artifact_pointers = Self::load_pending_artifact_deletions(tx)?;
            let table_results = results.into_values().collect::<Vec<_>>();
            let ttl_deleted_count = table_results
                .iter()
                .map(|result| result.ttl_deleted_count)
                .sum();
            let max_rows_deleted_count = table_results
                .iter()
                .map(|result| result.max_rows_deleted_count)
                .sum();
            Ok(RetentionCleanupResult {
                ttl_deleted_count,
                max_rows_deleted_count,
                artifact_pointers,
                table_results,
            })
        })?;
        if result.ttl_deleted_count > 0 || result.max_rows_deleted_count > 0 {
            return preserve_cleanup_result(Ok(result), self.maintain_space_after_cleanup());
        }
        Ok(result)
    }

    fn apply_summary_owner_ttl(
        tx: &Transaction,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates = Self::select_terminal_summary_ids_before(tx, &policy.cutoff_occurred_at)?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, true)
    }

    fn apply_artifact_owner_ttl(
        tx: &Transaction,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates = Self::select_terminal_owner_ids_before(
            tx,
            RetentionTable::ArtifactPointers,
            &policy.cutoff_occurred_at,
        )?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, true)
    }

    fn apply_terminal_owned_ttl(
        tx: &Transaction,
        table: RetentionTable,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates =
            Self::select_terminal_owner_ids_before(tx, table, &policy.cutoff_occurred_at)?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, true)
    }

    fn apply_standalone_ttl(
        tx: &Transaction,
        table: RetentionTable,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
    ) -> Result<(), LogStoreError> {
        let (table_name, id_column) = Self::table_sql(table);
        let deleted =
            Self::delete_rows_before(tx, table_name, id_column, &policy.cutoff_occurred_at)?;
        Self::record_retention_count(results, table, deleted, true)
    }

    fn apply_webhook_dead_letter_ttl(
        tx: &Transaction,
        cutoff_updated_at: &str,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
    ) -> Result<(), LogStoreError> {
        let deleted = tx
            .execute(
                &format!(
                    r#"
                DELETE FROM webhook_deliveries
                WHERE delivery_id IN (
                    SELECT delivery_id FROM webhook_deliveries
                    WHERE state = 'dead_letter' AND updated_at < ?1
                    ORDER BY updated_at ASC, delivery_id ASC
                    LIMIT {MAX_SUMMARIES_PER_CAP_PRUNE}
                )
                "#
                ),
                rusqlite::params![cutoff_updated_at],
            )
            .map_err(LogStoreError::Sqlite)?;
        Self::record_retention_count(
            results,
            RetentionTable::WebhookDeliveries,
            deleted as i64,
            true,
        )
    }

    fn apply_summary_owner_cap(
        tx: &Transaction,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates =
            Self::select_terminal_summary_ids_for_cap(tx, Self::policy_max_rows(policy)?)?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, false)
    }

    fn apply_artifact_owner_cap(
        tx: &Transaction,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates = Self::select_terminal_owner_ids_for_cap(
            tx,
            RetentionTable::ArtifactPointers,
            Self::policy_max_rows(policy)?,
        )?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, false)
    }

    fn apply_terminal_owned_cap(
        tx: &Transaction,
        table: RetentionTable,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        artifact_pointers: &mut Vec<CascadeArtifactPointer>,
    ) -> Result<(), LogStoreError> {
        let candidates =
            Self::select_terminal_owner_ids_for_cap(tx, table, Self::policy_max_rows(policy)?)?;
        let (deltas, pointers) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
        artifact_pointers.extend(pointers);
        Self::record_retention_deltas(results, deltas, false)
    }

    fn apply_standalone_cap(
        tx: &Transaction,
        table: RetentionTable,
        policy: &RetentionTablePolicy,
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
    ) -> Result<(), LogStoreError> {
        let (table_name, id_column) = Self::table_sql(table);
        let deleted =
            Self::delete_rows_to_max(tx, table_name, id_column, Self::policy_max_rows(policy)?)?;
        Self::record_retention_count(results, table, deleted, false)
    }

    fn policy_max_rows(policy: &RetentionTablePolicy) -> Result<i64, LogStoreError> {
        i64::try_from(policy.max_rows).map_err(|_| {
            LogStoreError::QueryFailed("logging retention max rows is out of range".to_string())
        })
    }

    fn table_sql(table: RetentionTable) -> (&'static str, &'static str) {
        match table {
            RetentionTable::Summaries => ("summaries", "request_id"),
            RetentionTable::LifecycleEvents => ("lifecycle_events", "event_id"),
            RetentionTable::ArtifactPointers => ("artifact_pointers", "artifact_id"),
            RetentionTable::ProxyRecords => ("proxy_records", "attempt_id"),
            RetentionTable::AuditEntries => ("audit_entries", "entry_id"),
            RetentionTable::WebhookDeliveries => ("webhook_deliveries", "delivery_id"),
            RetentionTable::CleanupRuns => ("cleanup_runs", "run_id"),
        }
    }

    fn empty_retention_deltas() -> BTreeMap<RetentionTable, i64> {
        RetentionTable::ALL
            .into_iter()
            .map(|table| (table, 0))
            .collect()
    }

    fn record_retention_deltas(
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        deltas: BTreeMap<RetentionTable, i64>,
        is_ttl: bool,
    ) -> Result<(), LogStoreError> {
        for (table, deleted) in deltas {
            Self::record_retention_count(results, table, deleted, is_ttl)?;
        }
        Ok(())
    }

    fn record_retention_count(
        results: &mut BTreeMap<RetentionTable, RetentionTableResult>,
        table: RetentionTable,
        deleted: i64,
        is_ttl: bool,
    ) -> Result<(), LogStoreError> {
        let entry = results
            .get_mut(&table)
            .expect("retention result map covers every durable table");
        if is_ttl {
            entry.ttl_deleted_count += deleted;
        } else {
            entry.max_rows_deleted_count += deleted;
        }
        Ok(())
    }

    /// Trim oldest terminal summaries until at most `max_rows` terminal
    /// summaries remain, deleting no more than one bounded batch per call.
    ///
    /// Active summaries are deliberately excluded: request serving must never
    /// lose its live lifecycle owner merely because durable history exceeds a
    /// retention cap. Candidate ordering is stable by terminal timestamp (or
    /// creation timestamp for legacy terminal rows) and then request ID. The
    /// returned artifact pointers are captured in the same transaction before
    /// the summary cascade removes their rows, so callers can safely perform
    /// post-commit file deletion only for pointers they owned.
    pub fn cascade_prune_terminal_summaries_to_max_rows(
        &self,
        max_rows: u64,
    ) -> Result<(i64, Vec<CascadeArtifactPointer>), LogStoreError> {
        let max_rows = i64::try_from(max_rows).map_err(|_| {
            LogStoreError::QueryFailed("logging retention max rows is out of range".to_string())
        })?;
        if max_rows < 1 {
            return Err(LogStoreError::QueryFailed(
                "logging retention max rows must be at least one".to_string(),
            ));
        }

        self.txn(|tx| {
            let candidates = Self::select_terminal_summary_ids_for_cap(tx, max_rows)?;
            let (deltas, _) = Self::delete_terminal_summary_candidates(tx, &candidates)?;
            let pending = Self::load_pending_artifact_deletions(tx)?;
            Ok((deltas.values().sum(), pending))
        })
    }

    fn select_terminal_summary_ids_before(
        tx: &Transaction,
        cutoff_occurred_at: &str,
    ) -> Result<Vec<String>, LogStoreError> {
        let mut statement = tx
            .prepare(&format!(
                "SELECT request_id FROM summaries\n\
                 WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')\n\
                   AND COALESCE(terminal_at, created_at) < ?1\n\
                 ORDER BY COALESCE(terminal_at, created_at) ASC, request_id ASC\n\
                 LIMIT {MAX_SUMMARIES_PER_CAP_PRUNE}",
            ))
            .map_err(LogStoreError::Sqlite)?;
        Self::collect_request_ids(&mut statement, rusqlite::params![cutoff_occurred_at])
    }

    fn select_terminal_summary_ids_for_cap(
        tx: &Transaction,
        max_terminal_summaries: i64,
    ) -> Result<Vec<String>, LogStoreError> {
        let terminal_count: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM summaries\n\
                 WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')",
                [],
                |row| row.get(0),
            )
            .map_err(LogStoreError::Sqlite)?;
        let prune_count = terminal_count
            .saturating_sub(max_terminal_summaries)
            .clamp(0, MAX_SUMMARIES_PER_CAP_PRUNE);
        if prune_count == 0 {
            return Ok(Vec::new());
        }

        let mut statement = tx
            .prepare(
                "SELECT request_id FROM summaries\n\
                 WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')\n\
                 ORDER BY COALESCE(terminal_at, created_at) ASC, request_id ASC\n\
                 LIMIT ?1",
            )
            .map_err(LogStoreError::Sqlite)?;
        Self::collect_request_ids(&mut statement, rusqlite::params![prune_count])
    }

    fn select_terminal_owner_ids_before(
        tx: &Transaction,
        table: RetentionTable,
        cutoff_occurred_at: &str,
    ) -> Result<Vec<String>, LogStoreError> {
        let (table_name, id_column) = Self::table_sql(table);
        let mut statement = tx
            .prepare(&format!(
                "SELECT {table_name}.request_id\n\
                 FROM {table_name}\n\
                 INNER JOIN summaries ON summaries.request_id = {table_name}.request_id\n\
                 WHERE summaries.state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')\n\
                   AND {table_name}.occurred_at < ?1\n\
                 GROUP BY {table_name}.request_id\n\
                 ORDER BY MIN({table_name}.occurred_at) ASC, MIN({table_name}.{id_column}) ASC, {table_name}.request_id ASC\n\
                 LIMIT {MAX_SUMMARIES_PER_CAP_PRUNE}"
            ))
            .map_err(LogStoreError::Sqlite)?;
        Self::collect_request_ids(&mut statement, rusqlite::params![cutoff_occurred_at])
    }

    fn select_terminal_owner_ids_for_cap(
        tx: &Transaction,
        table: RetentionTable,
        max_rows: i64,
    ) -> Result<Vec<String>, LogStoreError> {
        let (table_name, id_column) = Self::table_sql(table);
        let terminal_row_count: i64 = tx
            .query_row(
                &format!(
                    "SELECT COUNT(*) FROM {table_name}\n\
                     INNER JOIN summaries ON summaries.request_id = {table_name}.request_id\n\
                     WHERE summaries.state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')"
                ),
                [],
                |row| row.get(0),
            )
            .map_err(LogStoreError::Sqlite)?;
        let prune_count = terminal_row_count
            .saturating_sub(max_rows)
            .clamp(0, MAX_SUMMARIES_PER_CAP_PRUNE);
        if prune_count == 0 {
            return Ok(Vec::new());
        }
        let mut statement = tx
            .prepare(&format!(
                "SELECT {table_name}.request_id\n\
                 FROM {table_name}\n\
                 INNER JOIN summaries ON summaries.request_id = {table_name}.request_id\n\
                 WHERE summaries.state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')\n\
                 GROUP BY {table_name}.request_id\n\
                 ORDER BY MIN({table_name}.occurred_at) ASC, MIN({table_name}.{id_column}) ASC, {table_name}.request_id ASC\n\
                 LIMIT ?1"
            ))
            .map_err(LogStoreError::Sqlite)?;
        Self::collect_request_ids(&mut statement, rusqlite::params![prune_count])
    }

    fn collect_request_ids(
        statement: &mut rusqlite::Statement<'_>,
        parameters: impl rusqlite::Params,
    ) -> Result<Vec<String>, LogStoreError> {
        statement
            .query_map(parameters, |row| row.get(0))
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
    }

    fn delete_terminal_summary_candidates(
        tx: &Transaction,
        request_ids: &[String],
    ) -> Result<(BTreeMap<RetentionTable, i64>, Vec<CascadeArtifactPointer>), LogStoreError> {
        let mut deltas = Self::empty_retention_deltas();
        let mut pointers = Vec::new();
        for request_ids in request_ids.chunks(RETENTION_OWNER_BATCH_SIZE) {
            let placeholders = std::iter::repeat_n("?", request_ids.len())
                .collect::<Vec<_>>()
                .join(", ");
            let counts_sql = format!(
                "SELECT request_id, \
                    (SELECT COUNT(*) FROM lifecycle_events WHERE request_id = summaries.request_id), \
                    (SELECT COUNT(*) FROM artifact_pointers WHERE request_id = summaries.request_id), \
                    (SELECT COUNT(*) FROM proxy_records WHERE request_id = summaries.request_id) \
                 FROM summaries \
                 WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped') \
                   AND request_id IN ({placeholders})"
            );
            let selected_counts = {
                let mut statement = tx.prepare(&counts_sql).map_err(LogStoreError::Sqlite)?;
                statement
                    .query_map(rusqlite::params_from_iter(request_ids), |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, i64>(2)?,
                            row.get::<_, i64>(3)?,
                        ))
                    })
                    .map_err(LogStoreError::Sqlite)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(LogStoreError::Sqlite)?
            };
            if selected_counts.len() != request_ids.len() {
                return Err(LogStoreError::QueryFailed(
                    "retention owner changed during transactional deletion".to_string(),
                ));
            }

            let pointer_sql = format!(
                "SELECT artifact_id, request_id FROM artifact_pointers \
                 WHERE request_id IN ({placeholders}) ORDER BY request_id ASC, artifact_id ASC"
            );
            let mut selected_pointers = {
                let mut statement = tx.prepare(&pointer_sql).map_err(LogStoreError::Sqlite)?;
                statement
                    .query_map(rusqlite::params_from_iter(request_ids), |row| {
                        Ok(CascadeArtifactPointer {
                            artifact_id: row.get(0)?,
                            request_id: row.get(1)?,
                        })
                    })
                    .map_err(LogStoreError::Sqlite)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(LogStoreError::Sqlite)?
            };

            // Ownership must become durable before the summary cascade makes
            // these filesystem paths impossible to rediscover.
            Self::queue_artifact_deletions(tx, &selected_pointers)?;
            let delete_sql = format!(
                "DELETE FROM summaries \
                 WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped') \
                   AND request_id IN ({placeholders})"
            );
            let deleted = tx
                .execute(&delete_sql, rusqlite::params_from_iter(request_ids))
                .map_err(LogStoreError::Sqlite)?;
            if deleted != selected_counts.len() {
                return Err(LogStoreError::QueryFailed(
                    "retention owner changed during transactional deletion".to_string(),
                ));
            }
            for (_, events, artifacts, proxy_records) in selected_counts {
                *deltas
                    .get_mut(&RetentionTable::Summaries)
                    .expect("all tables") += 1;
                *deltas
                    .get_mut(&RetentionTable::LifecycleEvents)
                    .expect("all tables") += events;
                *deltas
                    .get_mut(&RetentionTable::ArtifactPointers)
                    .expect("all tables") += artifacts;
                *deltas
                    .get_mut(&RetentionTable::ProxyRecords)
                    .expect("all tables") += proxy_records;
            }
            pointers.append(&mut selected_pointers);
        }
        Ok((deltas, pointers))
    }

    pub(crate) fn queue_artifact_deletions(
        tx: &Transaction<'_>,
        pointers: &[CascadeArtifactPointer],
    ) -> Result<(), LogStoreError> {
        for pointers in pointers.chunks(RETENTION_OWNER_BATCH_SIZE) {
            let placeholders = std::iter::repeat_n("?", pointers.len())
                .collect::<Vec<_>>()
                .join(", ");
            let pending_sql = format!(
                "SELECT artifact_id, request_id FROM pending_artifact_deletions \
                 WHERE artifact_id IN ({placeholders})"
            );
            let pending = {
                let mut statement = tx.prepare(&pending_sql).map_err(LogStoreError::Sqlite)?;
                statement
                    .query_map(
                        rusqlite::params_from_iter(
                            pointers.iter().map(|pointer| &pointer.artifact_id),
                        ),
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                    )
                    .map_err(LogStoreError::Sqlite)?
                    .collect::<Result<BTreeMap<_, _>, _>>()
                    .map_err(LogStoreError::Sqlite)?
            };
            for pointer in pointers {
                if pending
                    .get(&pointer.artifact_id)
                    .is_some_and(|request_id| request_id != &pointer.request_id)
                {
                    return Err(LogStoreError::QueryFailed(
                        "pending artifact deletion ownership conflict".to_string(),
                    ));
                }
            }
            let mut insert = tx
                .prepare(
                    "INSERT OR IGNORE INTO pending_artifact_deletions (artifact_id, request_id) \
                     VALUES (?1, ?2)",
                )
                .map_err(LogStoreError::Sqlite)?;
            for pointer in pointers {
                insert
                    .execute(rusqlite::params![pointer.artifact_id, pointer.request_id])
                    .map_err(LogStoreError::Sqlite)?;
            }
        }
        Ok(())
    }

    pub(crate) fn load_pending_artifact_deletions(
        tx: &Transaction<'_>,
    ) -> Result<Vec<CascadeArtifactPointer>, LogStoreError> {
        let mut statement = tx
            .prepare(
                "SELECT pending.artifact_id, pending.request_id \
                 FROM pending_artifact_deletions AS pending \
                 WHERE NOT EXISTS (\
                     SELECT 1 FROM summaries \
                     WHERE summaries.request_id = pending.request_id\
                 ) \
                 ORDER BY artifact_id ASC, request_id ASC LIMIT ?1",
            )
            .map_err(LogStoreError::Sqlite)?;
        statement
            .query_map([MAX_PENDING_ARTIFACT_DELETIONS_PER_BATCH], |row| {
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
                 WHERE artifact_id = ?1 AND request_id = ?2",
                rusqlite::params![pointer.artifact_id, pointer.request_id],
            )
            .map(|_| ())
            .map_err(LogStoreError::Sqlite)
    }

    fn delete_rows_before(
        tx: &Transaction,
        table: &str,
        id_column: &str,
        cutoff_occurred_at: &str,
    ) -> Result<i64, LogStoreError> {
        let deleted = tx
            .execute(
                &format!(
                    "DELETE FROM {table} WHERE {id_column} IN (\n\
                     SELECT {id_column} FROM {table}\n\
                     WHERE occurred_at < ?1\n\
                     ORDER BY occurred_at ASC, {id_column} ASC\n\
                     LIMIT {MAX_SUMMARIES_PER_CAP_PRUNE}\n\
                     )"
                ),
                rusqlite::params![cutoff_occurred_at],
            )
            .map_err(LogStoreError::Sqlite)?;
        Ok(deleted as i64)
    }

    fn delete_rows_to_max(
        tx: &Transaction,
        table: &str,
        id_column: &str,
        max_rows: i64,
    ) -> Result<i64, LogStoreError> {
        let count: i64 = tx
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .map_err(LogStoreError::Sqlite)?;
        let prune_count = count
            .saturating_sub(max_rows)
            .clamp(0, MAX_SUMMARIES_PER_CAP_PRUNE);
        if prune_count == 0 {
            return Ok(0);
        }
        let deleted = tx
            .execute(
                &format!(
                    "DELETE FROM {table} WHERE {id_column} IN (\n\
                     SELECT {id_column} FROM {table}\n\
                     ORDER BY occurred_at ASC, {id_column} ASC\n\
                     LIMIT ?1)"
                ),
                rusqlite::params![prune_count],
            )
            .map_err(LogStoreError::Sqlite)?;
        Ok(deleted as i64)
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
