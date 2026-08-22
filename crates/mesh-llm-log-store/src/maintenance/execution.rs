use super::*;

impl LogStore {
    /// Retain only a bounded history of maintenance receipts.
    ///
    /// Completed receipts and non-retryable partial cleanup receipts are
    /// subject to both TTL and row-cap retention. A
    /// preview-only cleanup receipt contains no destructive work in progress,
    /// so it is also TTL eligible. A previewed `delete_one` receipt, however,
    /// is durable retry state for interrupted artifact deletion and is never
    /// collected here. Deleting an eligible operation cascades its immutable
    /// targets in the same SQLite transaction.
    pub fn cleanup_maintenance_receipts(
        &self,
        cutoff_before: &str,
        max_rows: u64,
    ) -> Result<u64, LogStoreError> {
        let cutoff_before = canonical_persisted_timestamp(cutoff_before)?;
        let max_rows = i64::try_from(max_rows).map_err(|_| {
            LogStoreError::QueryFailed("maintenance receipt max rows is out of range".to_string())
        })?;
        self.txn(|transaction| {
            let ttl_deleted = transaction
                .execute(
                    "DELETE FROM maintenance_operations WHERE operation_id IN (\
                     SELECT operation_id FROM maintenance_operations \
                     WHERE (state = 'completed' \
                            OR (state = 'partial' AND action = 'cleanup' \
                                AND artifact_files_failed = 0) \
                            OR (state = 'previewed' AND action = 'cleanup')) \
                       AND COALESCE(completed_at, created_at) < ?1 \
                     ORDER BY COALESCE(completed_at, created_at) ASC, operation_id ASC \
                     LIMIT 1000)",
                    [&cutoff_before],
                )
                .map_err(LogStoreError::Sqlite)? as u64;
            let terminal_count: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM maintenance_operations \
                     WHERE state = 'completed' \
                        OR (state = 'partial' AND action = 'cleanup' \
                            AND artifact_files_failed = 0)",
                    [],
                    |row| row.get(0),
                )
                .map_err(LogStoreError::Sqlite)?;
            let excess = terminal_count.saturating_sub(max_rows).min(1_000);
            let capped_deleted = if excess == 0 {
                0
            } else {
                transaction
                    .execute(
                        "DELETE FROM maintenance_operations WHERE operation_id IN (\
                         SELECT operation_id FROM maintenance_operations \
                         WHERE state = 'completed' \
                            OR (state = 'partial' AND action = 'cleanup' \
                                AND artifact_files_failed = 0) \
                         ORDER BY COALESCE(completed_at, created_at) ASC, operation_id ASC \
                         LIMIT ?1)",
                        [excess],
                    )
                    .map_err(LogStoreError::Sqlite)? as u64
            };
            Ok(ttl_deleted + capped_deleted)
        })
    }

    /// Load one cleanup receipt after validating its immutable action and reason.
    ///
    /// Metadata-only runtimes use this read-only path to preserve typed
    /// not-found/conflict outcomes and replay an already completed operation
    /// even though they deliberately have no artifact-file owner.
    pub fn cleanup_receipt(
        &self,
        operation_id: MaintenanceOperationId,
        expected_reason: &MaintenanceReason,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        self.txn(|transaction| {
            let receipt = load_receipt(transaction, operation_id)?
                .ok_or(LogStoreError::MaintenanceOperationNotFound)?;
            if receipt.action != MaintenanceAction::Cleanup
                || load_reason(transaction, operation_id)? != expected_reason.as_str()
            {
                return Err(LogStoreError::MaintenanceOperationConflict);
            }
            Ok(receipt)
        })
    }

    /// Return the completed receipt for one matching delete-one operation
    /// without opening a new delete operation. Callers use this to replay a
    /// successful deletion after its request owner has been removed.
    pub fn delete_one_receipt(
        &self,
        request: &DeleteOneRequest,
    ) -> Result<Option<MaintenanceReceipt>, LogStoreError> {
        self.txn(|transaction| {
            let Some(receipt) = load_receipt(transaction, request.operation_id)? else {
                return Ok(None);
            };
            ensure_same_delete_one(transaction, &receipt, request)?;
            Ok(Some(receipt))
        })
    }

    /// Snapshot terminal request owners before a caller-supplied cutoff and
    /// persist the exact target list plus a stable trusted-local audit record.
    pub fn preview_cleanup(
        &self,
        request: &CleanupPreviewRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        self.txn(|transaction| {
            if let Some(existing) = load_receipt(transaction, request.operation_id)? {
                ensure_same_intent(
                    &existing,
                    &load_reason(transaction, request.operation_id)?,
                    request,
                )?;
                return Ok(existing);
            }

            if control.is_cancelled() {
                return Err(LogStoreError::MaintenanceExecutionCancelled);
            }
            let (targets, has_more) = select_targets(transaction, &request.scope)?;
            let planned = count_targets(transaction, &targets)?;
            // Reads may take longer than a route's wall-clock budget. Check
            // again immediately before the first write so a caller that has
            // already timed out cannot later observe an unrequested receipt.
            if control.is_cancelled() {
                return Err(LogStoreError::MaintenanceExecutionCancelled);
            }
            let preview_audit_id = Uuid::new_v4().to_string();
            let receipt = MaintenanceReceipt {
                operation_id: request.operation_id,
                action: MaintenanceAction::Cleanup,
                scope: request.scope.clone(),
                state: MaintenanceReceiptState::Previewed,
                planned,
                executed: MaintenanceCounts::default(),
                artifact_deletion: ArtifactDeletionProgress::default(),
                has_more,
                fingerprint: selection_fingerprint(MaintenanceAction::Cleanup, &request.scope, &targets),
                preview_audit_id: Some(preview_audit_id.clone()),
                execution_audit_id: None,
            };
            persist_receipt(transaction, &receipt, request.reason.as_str(), &self.now())?;
            for (ordinal, request_id) in targets.iter().enumerate() {
                transaction
                    .execute(
                        "INSERT INTO maintenance_operation_targets (operation_id, ordinal, request_id) VALUES (?1, ?2, ?3)",
                        rusqlite::params![request.operation_id.to_string(), ordinal, request_id],
                    )
                    .map_err(LogStoreError::Sqlite)?;
            }
            let preview_result = if receipt.has_more { "partial" } else { "previewed" };
            write_audit(
                transaction,
                &self.now(),
                &preview_audit_id,
                request.operation_id,
                "log_cleanup_preview",
                preview_result,
                request.reason.as_str(),
            )?;
            Ok(receipt)
        })
    }
}

impl ArtifactFileStore {
    /// Persist the immutable receipt and target for one delete-one operation
    /// before any artifact file is touched.
    ///
    /// The returned receipt is safe to return to a bounded HTTP caller as an
    /// accepted, retryable pending operation. Filesystem work remains owned by
    /// [`Self::execute_prepared_delete_request`].
    pub fn prepare_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let _maintenance = self.maintenance_lock();
        self.store_ref().txn(|transaction| {
            if let Some(existing) = load_receipt(transaction, request.operation_id)? {
                ensure_same_delete_one(transaction, &existing, request)?;
                return Ok(existing);
            }
            ensure_maintenance_active(control)?;
            let scope = delete_one_scope()?;
            let targets = vec![request.request_id.clone()];
            let planned = count_terminal_request_owner(transaction, &request.request_id)?;
            // Keep the queue durable with the receipt. This never performs
            // file I/O and a retry may safely rebuild it from current pointers.
            let pointers = terminal_request_artifacts(transaction, &request.request_id)?;
            ensure_maintenance_active(control)?;
            LogStore::queue_artifact_deletions(transaction, &pointers)?;
            let fingerprint = selection_fingerprint(MaintenanceAction::DeleteOne, &scope, &targets);
            let receipt = MaintenanceReceipt {
                operation_id: request.operation_id,
                action: MaintenanceAction::DeleteOne,
                scope,
                state: MaintenanceReceiptState::Previewed,
                planned,
                executed: MaintenanceCounts::default(),
                artifact_deletion: ArtifactDeletionProgress::default(),
                has_more: false,
                fingerprint,
                preview_audit_id: None,
                execution_audit_id: None,
            };
            let occurred_at = self.store_ref().now();
            persist_receipt(transaction, &receipt, request.reason.as_str(), &occurred_at)?;
            transaction
                .execute(
                    "INSERT INTO maintenance_operation_targets (operation_id, ordinal, request_id) VALUES (?1, 0, ?2)",
                    rusqlite::params![request.operation_id.to_string(), request.request_id],
                )
                .map_err(LogStoreError::Sqlite)?;
            Ok(receipt)
        })
    }

    /// Execute a previously prepared delete-one operation through the confined
    /// artifact owner. A completed receipt is replayed verbatim.
    pub fn execute_prepared_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let _maintenance = self.maintenance_lock();
        let plan = self.store_ref().txn(|transaction| {
            let receipt = load_receipt(transaction, request.operation_id)?
                .ok_or(LogStoreError::MaintenanceOperationNotFound)?;
            ensure_same_delete_one(transaction, &receipt, request)?;
            if receipt.state == MaintenanceReceiptState::Completed {
                return Ok(DeleteRequestPlan::Replay(receipt));
            }
            ensure_maintenance_active(control)?;
            let pointers = terminal_request_artifacts(transaction, &request.request_id)?;
            LogStore::queue_artifact_deletions(transaction, &pointers)?;
            Ok(DeleteRequestPlan::Execute { receipt, pointers })
        })?;

        let DeleteRequestPlan::Replay(receipt) = plan else {
            ensure_maintenance_active(control)?;
            let pointers = plan.pointers();
            let results = self.delete_artifact_files_for_maintenance(pointers, control)?;
            ensure_maintenance_active(control)?;
            let occurred_at = self.store_ref().now();
            return self.store_ref().txn(|transaction| {
                complete_delete_request(transaction, request, plan, &results, &occurred_at, control)
            });
        };
        Ok(receipt)
    }

    /// Execute one previously previewed cleanup operation. A completed receipt
    /// is returned verbatim on every retry, so the operation ID is idempotent.
    pub fn execute_cleanup(
        &self,
        operation_id: MaintenanceOperationId,
        expected_reason: &MaintenanceReason,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let _maintenance = self.maintenance_lock();
        let plan = self.store_ref().txn(|transaction| {
            let receipt = load_receipt(transaction, operation_id)?
                .ok_or(LogStoreError::MaintenanceOperationNotFound)?;
            if receipt.action != MaintenanceAction::Cleanup {
                return Err(LogStoreError::MaintenanceOperationConflict);
            }
            let reason = load_reason(transaction, operation_id)?;
            if reason != expected_reason.as_str() {
                return Err(LogStoreError::MaintenanceOperationConflict);
            }
            if receipt.state == MaintenanceReceiptState::Completed
                || (receipt.state == MaintenanceReceiptState::Partial
                    && receipt.artifact_deletion.failed == 0)
            {
                return Ok(CleanupExecutionPlan::Replay(receipt));
            }
            ensure_maintenance_active(control)?;
            let targets = load_targets(transaction, operation_id)?;
            ensure_maintenance_active(control)?;
            let pointers = terminal_target_artifacts(transaction, &targets)?;
            LogStore::queue_artifact_deletions(transaction, &pointers)?;
            Ok(CleanupExecutionPlan::Execute {
                receipt,
                targets: targets.clone(),
                pointers,
                reason,
            })
        })?;
        match plan {
            CleanupExecutionPlan::Replay(receipt) => Ok(receipt),
            CleanupExecutionPlan::Execute {
                receipt,
                targets,
                pointers,
                reason,
            } => {
                ensure_maintenance_active(control)?;
                let results = self.delete_artifact_files_for_maintenance(&pointers, control)?;
                ensure_maintenance_active(control)?;
                let occurred_at = self.store_ref().now();
                let context = CleanupExecutionContext {
                    targets: &targets,
                    operation_id,
                    reason: &reason,
                    occurred_at: &occurred_at,
                    control,
                };
                self.store_ref().txn(|transaction| {
                    complete_cleanup_execution(transaction, receipt, &results, &context)
                })
            }
        }
    }

    /// Delete one request owner and all of its durable children. The request
    /// ID is frozen into the operation target table even when absent, so a
    /// retry receives the original completed receipt instead of becoming a
    /// new deletion with a broader meaning.
    pub fn delete_request_cascade(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        self.prepare_delete_request(request, control)?;
        self.execute_prepared_delete_request(request, control)
    }
}

pub(super) enum DeleteRequestPlan {
    Replay(MaintenanceReceipt),
    Execute {
        receipt: MaintenanceReceipt,
        pointers: Vec<CascadeArtifactPointer>,
    },
}

enum CleanupExecutionPlan {
    Replay(MaintenanceReceipt),
    Execute {
        receipt: MaintenanceReceipt,
        targets: Vec<String>,
        pointers: Vec<CascadeArtifactPointer>,
        reason: String,
    },
}

struct CleanupExecutionContext<'a> {
    targets: &'a [String],
    operation_id: MaintenanceOperationId,
    reason: &'a str,
    occurred_at: &'a str,
    control: &'a dyn MaintenanceExecutionControl,
}

impl DeleteRequestPlan {
    fn pointers(&self) -> &[CascadeArtifactPointer] {
        match self {
            Self::Replay(_) => &[],
            Self::Execute { pointers, .. } => pointers,
        }
    }
}

pub(super) fn ensure_maintenance_active(
    control: &dyn MaintenanceExecutionControl,
) -> Result<(), LogStoreError> {
    if control.is_cancelled() {
        Err(LogStoreError::MaintenanceExecutionCancelled)
    } else {
        Ok(())
    }
}

pub(super) fn complete_delete_request(
    transaction: &Transaction<'_>,
    request: &DeleteOneRequest,
    plan: DeleteRequestPlan,
    results: &[CascadeArtifactDeleteResult],
    occurred_at: &str,
    control: &dyn MaintenanceExecutionControl,
) -> Result<MaintenanceReceipt, LogStoreError> {
    ensure_maintenance_active(control)?;
    let mut receipt = match plan {
        DeleteRequestPlan::Execute { receipt, .. } => receipt,
        DeleteRequestPlan::Replay(_) => {
            return Err(LogStoreError::MaintenanceOperationConflict);
        }
    };
    let removed = remove_successful_artifact_pointers(transaction, results, control)?;
    receipt.executed.artifacts += removed;
    receipt.executed.database_rows += removed;
    receipt.artifact_deletion.removed += removed;
    receipt.artifact_deletion.failed =
        results.iter().filter(|result| !result.succeeded()).count() as u64;
    receipt.artifact_deletion.failure_class = results
        .iter()
        .find_map(CascadeArtifactDeleteResult::failure_class)
        .map(ArtifactDeletionFailureClass::from);

    if receipt.artifact_deletion.failed == 0 {
        let (final_counts, unexpected_pointers) =
            delete_request_owner(transaction, &request.request_id, control)?;
        if !unexpected_pointers.is_empty() {
            return Err(LogStoreError::MaintenanceOperationConflict);
        }
        add_counts(&mut receipt.executed, final_counts);
        receipt.state = if receipt.executed == receipt.planned {
            MaintenanceReceiptState::Completed
        } else {
            MaintenanceReceiptState::Partial
        };
    } else {
        receipt.state = MaintenanceReceiptState::Partial;
    }

    ensure_maintenance_active(control)?;
    let execution_audit_id = Uuid::new_v4().to_string();
    receipt.execution_audit_id = Some(execution_audit_id.clone());
    update_maintenance_receipt(transaction, &receipt, occurred_at)?;
    ensure_maintenance_active(control)?;
    write_audit(
        transaction,
        occurred_at,
        &execution_audit_id,
        request.operation_id,
        "log_delete_request",
        receipt.state.as_str(),
        request.reason.as_str(),
    )?;
    Ok(receipt)
}

fn complete_cleanup_execution(
    transaction: &Transaction<'_>,
    mut receipt: MaintenanceReceipt,
    results: &[CascadeArtifactDeleteResult],
    context: &CleanupExecutionContext<'_>,
) -> Result<MaintenanceReceipt, LogStoreError> {
    ensure_maintenance_active(context.control)?;
    let removed = remove_successful_artifact_pointers(transaction, results, context.control)?;
    receipt.executed.artifacts += removed;
    receipt.executed.database_rows += removed;
    receipt.artifact_deletion.removed += removed;
    let failed_targets = failed_cleanup_targets(results);
    receipt.artifact_deletion.failed =
        results.iter().filter(|result| !result.succeeded()).count() as u64;
    receipt.artifact_deletion.failure_class = results
        .iter()
        .find_map(CascadeArtifactDeleteResult::failure_class)
        .map(ArtifactDeletionFailureClass::from);

    let completed_targets = delete_reconciled_cleanup_targets(
        transaction,
        context.targets,
        &failed_targets,
        context.control,
    )?;
    add_counts(&mut receipt.executed, completed_targets);
    receipt.state = if receipt.artifact_deletion.failed > 0
        || receipt.has_more
        || receipt.executed != receipt.planned
    {
        MaintenanceReceiptState::Partial
    } else {
        MaintenanceReceiptState::Completed
    };
    ensure_maintenance_active(context.control)?;
    let execution_audit_id = Uuid::new_v4().to_string();
    receipt.execution_audit_id = Some(execution_audit_id.clone());
    update_maintenance_receipt(transaction, &receipt, context.occurred_at)?;
    ensure_maintenance_active(context.control)?;
    write_audit(
        transaction,
        context.occurred_at,
        &execution_audit_id,
        context.operation_id,
        "log_cleanup_execute",
        receipt.state.as_str(),
        context.reason,
    )?;
    Ok(receipt)
}

fn failed_cleanup_targets(results: &[CascadeArtifactDeleteResult]) -> HashSet<String> {
    results
        .iter()
        .filter(|result| !result.succeeded())
        .map(|result| result.pointer().request_id.clone())
        .collect()
}

fn delete_reconciled_cleanup_targets(
    transaction: &Transaction<'_>,
    targets: &[String],
    failed_targets: &HashSet<String>,
    control: &dyn MaintenanceExecutionControl,
) -> Result<MaintenanceCounts, LogStoreError> {
    let mut executed = MaintenanceCounts::default();
    for request_id in targets {
        ensure_maintenance_active(control)?;
        if failed_targets.contains(request_id) {
            continue;
        }
        if !terminal_request_artifacts(transaction, request_id)?.is_empty() {
            return Err(LogStoreError::MaintenanceOperationConflict);
        }
        let counts = count_terminal_request_owner(transaction, request_id)?;
        ensure_maintenance_active(control)?;
        let deleted = transaction
            .execute(
                "DELETE FROM summaries WHERE request_id = ?1 AND state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')",
                [request_id],
            )
            .map_err(LogStoreError::Sqlite)?;
        if deleted == 1 {
            add_counts(&mut executed, counts);
        }
    }
    Ok(executed)
}

fn remove_successful_artifact_pointers(
    transaction: &Transaction<'_>,
    results: &[CascadeArtifactDeleteResult],
    control: &dyn MaintenanceExecutionControl,
) -> Result<u64, LogStoreError> {
    let mut removed = 0;
    for result in results.iter().filter(|result| result.succeeded()) {
        ensure_maintenance_active(control)?;
        let pointer = result.pointer();
        removed += transaction
            .execute(
                "DELETE FROM artifact_pointers WHERE artifact_id = ?1 AND request_id = ?2",
                rusqlite::params![pointer.artifact_id, pointer.request_id],
            )
            .map_err(LogStoreError::Sqlite)? as u64;
        transaction
            .execute(
                "DELETE FROM pending_artifact_deletions \
                 WHERE artifact_id = ?1 AND request_id = ?2",
                rusqlite::params![pointer.artifact_id, pointer.request_id],
            )
            .map_err(LogStoreError::Sqlite)?;
    }
    Ok(removed)
}

fn update_maintenance_receipt(
    transaction: &Transaction<'_>,
    receipt: &MaintenanceReceipt,
    occurred_at: &str,
) -> Result<(), LogStoreError> {
    let occurred_at = canonical_persisted_timestamp(occurred_at)?;
    transaction
        .execute(
            "UPDATE maintenance_operations SET state = ?2, executed_requests = ?3, executed_events = ?4, executed_artifacts = ?5, executed_proxy_records = ?6, executed_database_rows = ?7, artifact_files_removed = ?8, artifact_files_failed = ?9, artifact_file_failure_class = ?10, completed_at = ?11, execution_audit_id = ?12 WHERE operation_id = ?1",
            rusqlite::params![
                receipt.operation_id.to_string(),
                receipt.state.as_str(),
                receipt.executed.requests,
                receipt.executed.events,
                receipt.executed.artifacts,
                receipt.executed.proxy_records,
                receipt.executed.database_rows,
                receipt.artifact_deletion.removed,
                receipt.artifact_deletion.failed,
                receipt.artifact_deletion.failure_class.map(ArtifactDeletionFailureClass::as_str),
                occurred_at,
                receipt.execution_audit_id,
            ],
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(())
}

fn add_counts(target: &mut MaintenanceCounts, added: MaintenanceCounts) {
    target.requests += added.requests;
    target.events += added.events;
    target.artifacts += added.artifacts;
    target.proxy_records += added.proxy_records;
    target.database_rows += added.database_rows;
}

fn select_targets(
    transaction: &Transaction<'_>,
    scope: &CleanupScope,
) -> Result<(Vec<String>, bool), LogStoreError> {
    let filters = scope.filters();
    let mut sql = String::from(
        "SELECT request_id FROM summaries WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped') AND created_at < ? AND (route IS NULL OR route NOT GLOB 'management_*')",
    );
    let mut parameters = vec![rusqlite::types::Value::Text(
        scope.cutoff_before.as_str().to_owned(),
    )];
    for (column, value) in [
        ("created_at >=", filters.from()),
        ("created_at <=", filters.to()),
        ("route =", filters.route()),
        ("model =", filters.model()),
        ("provider =", filters.provider()),
        ("engine =", filters.engine()),
    ] {
        if let Some(value) = value {
            sql.push_str(" AND ");
            sql.push_str(column);
            sql.push_str(" ?");
            parameters.push(rusqlite::types::Value::Text(value.to_owned()));
        }
    }
    if let Some(outcome) = filters.outcome() {
        sql.push_str(" AND state = ?");
        parameters.push(rusqlite::types::Value::Text(outcome.as_str().to_owned()));
    }
    sql.push_str(" ORDER BY created_at ASC, request_id ASC LIMIT ?");
    parameters.push(rusqlite::types::Value::Integer(
        i64::from(scope.request_limit) + 1,
    ));
    let mut statement = transaction.prepare(&sql).map_err(LogStoreError::Sqlite)?;
    let mut targets = statement
        .query_map(rusqlite::params_from_iter(parameters), |row| {
            row.get::<_, String>(0)
        })
        .map_err(LogStoreError::Sqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(LogStoreError::Sqlite)?;
    let has_more = targets.len() > usize::from(scope.request_limit);
    targets.truncate(usize::from(scope.request_limit));
    Ok((targets, has_more))
}

fn count_targets(
    transaction: &Transaction<'_>,
    targets: &[String],
) -> Result<MaintenanceCounts, LogStoreError> {
    if targets.is_empty() {
        return Ok(MaintenanceCounts::default());
    }
    let placeholders = std::iter::repeat_n("?", targets.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT \
             COUNT(*), \
             COALESCE(SUM((SELECT COUNT(*) FROM lifecycle_events WHERE request_id = summaries.request_id)), 0), \
             COALESCE(SUM((SELECT COUNT(*) FROM artifact_pointers WHERE request_id = summaries.request_id)), 0), \
             COALESCE(SUM((SELECT COUNT(*) FROM proxy_records WHERE request_id = summaries.request_id)), 0) \
         FROM summaries \
         WHERE state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped') \
           AND request_id IN ({placeholders})"
    );
    transaction
        .query_row(&sql, rusqlite::params_from_iter(targets), |row| {
            let requests = row.get::<_, u64>(0)?;
            let events = row.get::<_, u64>(1)?;
            let artifacts = row.get::<_, u64>(2)?;
            let proxy_records = row.get::<_, u64>(3)?;
            Ok(MaintenanceCounts {
                requests,
                events,
                artifacts,
                proxy_records,
                database_rows: requests + events + artifacts + proxy_records,
            })
        })
        .map_err(LogStoreError::Sqlite)
}

pub(super) fn persist_receipt(
    transaction: &Transaction<'_>,
    receipt: &MaintenanceReceipt,
    reason: &str,
    occurred_at: &str,
) -> Result<(), LogStoreError> {
    let occurred_at = canonical_persisted_timestamp(occurred_at)?;
    let completed_at = if matches!(receipt.state, MaintenanceReceiptState::Previewed) {
        None
    } else {
        Some(occurred_at.as_str())
    };
    transaction
        .execute(
            "INSERT INTO maintenance_operations (operation_id, action, cutoff_before, request_limit, reason, state, planned_requests, planned_events, planned_artifacts, planned_proxy_records, planned_database_rows, executed_requests, executed_events, executed_artifacts, executed_proxy_records, executed_database_rows, artifact_files_removed, artifact_files_failed, artifact_file_failure_class, has_more, created_at, completed_at, selection_fingerprint, preview_audit_id, execution_audit_id, cleanup_filters_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            rusqlite::params![
                receipt.operation_id.to_string(),
                receipt.action.as_str(),
                receipt.scope.cutoff_before.as_str(),
                i64::from(receipt.scope.request_limit),
                reason,
                receipt.state.as_str(),
                receipt.planned.requests,
                receipt.planned.events,
                receipt.planned.artifacts,
                receipt.planned.proxy_records,
                receipt.planned.database_rows,
                receipt.executed.requests,
                receipt.executed.events,
                receipt.executed.artifacts,
                receipt.executed.proxy_records,
                receipt.executed.database_rows,
                receipt.artifact_deletion.removed,
                receipt.artifact_deletion.failed,
                receipt.artifact_deletion.failure_class.map(ArtifactDeletionFailureClass::as_str),
                receipt.has_more,
                occurred_at,
                completed_at,
                receipt.fingerprint.as_str(),
                receipt.preview_audit_id,
                receipt.execution_audit_id,
                serde_json::to_string(receipt.scope.filters()).map_err(|error| LogStoreError::QueryFailed(error.to_string()))?,
            ],
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(())
}

pub(super) fn load_receipt(
    connection: &rusqlite::Connection,
    operation_id: MaintenanceOperationId,
) -> Result<Option<MaintenanceReceipt>, LogStoreError> {
    connection
        .query_row(
            "SELECT action, cutoff_before, request_limit, state, planned_requests, planned_events, planned_artifacts, planned_proxy_records, planned_database_rows, executed_requests, executed_events, executed_artifacts, executed_proxy_records, executed_database_rows, artifact_files_removed, artifact_files_failed, artifact_file_failure_class, has_more, selection_fingerprint, preview_audit_id, execution_audit_id, cleanup_filters_json FROM maintenance_operations WHERE operation_id = ?1",
            [operation_id.to_string()],
            |row| {
                let action: String = row.get(0)?;
                let cutoff: String = row.get(1)?;
                let request_limit: usize = row.get(2)?;
                let filters: CleanupFilters = serde_json::from_str(&row.get::<_, String>(21)?)
                    .map_err(|error| to_sql_error(LogStoreError::QueryFailed(error.to_string())))?;
                let scope = CleanupScope::new(
                    MaintenanceTimestamp::try_from(cutoff.as_str()).map_err(to_sql_error)?,
                    request_limit,
                )
                .map_err(to_sql_error)?
                .with_filters(filters);
                Ok(MaintenanceReceipt {
                    operation_id,
                    action: MaintenanceAction::from_str(&action).map_err(to_sql_error)?,
                    scope,
                    state: MaintenanceReceiptState::from_str(&row.get::<_, String>(3)?).map_err(to_sql_error)?,
                    planned: MaintenanceCounts {
                        requests: row.get(4)?, events: row.get(5)?, artifacts: row.get(6)?, proxy_records: row.get(7)?, database_rows: row.get(8)?,
                    },
                    executed: MaintenanceCounts { requests: row.get(9)?, events: row.get(10)?, artifacts: row.get(11)?, proxy_records: row.get(12)?, database_rows: row.get(13)? },
                    artifact_deletion: ArtifactDeletionProgress {
                        removed: row.get(14)?,
                        failed: row.get(15)?,
                        failure_class: row.get::<_, Option<String>>(16)?.map(|value| ArtifactDeletionFailureClass::from_str(&value)).transpose().map_err(to_sql_error)?,
                    },
                    has_more: row.get(17)?,
                    fingerprint: MaintenanceFingerprint(row.get(18)?),
                    preview_audit_id: row.get(19)?,
                    execution_audit_id: row.get(20)?,
                })
            },
        )
        .optional()
        .map_err(LogStoreError::Sqlite)
}

fn load_targets(
    transaction: &Transaction<'_>,
    operation_id: MaintenanceOperationId,
) -> Result<Vec<String>, LogStoreError> {
    let mut statement = transaction
        .prepare("SELECT request_id FROM maintenance_operation_targets WHERE operation_id = ?1 ORDER BY ordinal ASC")
        .map_err(LogStoreError::Sqlite)?;
    statement
        .query_map([operation_id.to_string()], |row| row.get(0))
        .map_err(LogStoreError::Sqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(LogStoreError::Sqlite)
}

fn delete_request_owner(
    transaction: &Transaction<'_>,
    request_id: &str,
    control: &dyn MaintenanceExecutionControl,
) -> Result<(MaintenanceCounts, Vec<CascadeArtifactPointer>), LogStoreError> {
    ensure_maintenance_active(control)?;
    let counts = count_terminal_request_owner(transaction, request_id)?;
    let pointers = load_request_artifact_pointers(transaction, request_id)?;
    ensure_maintenance_active(control)?;
    LogStore::queue_artifact_deletions(transaction, &pointers)?;
    let deleted = transaction
        .execute(
            "DELETE FROM summaries WHERE request_id = ?1 AND state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')",
            [request_id],
        )
        .map_err(LogStoreError::Sqlite)?;
    if deleted == 1 {
        Ok((counts, pointers))
    } else if pointers.is_empty() {
        Ok((MaintenanceCounts::default(), Vec::new()))
    } else {
        Err(LogStoreError::MaintenanceOperationConflict)
    }
}

pub(super) fn count_terminal_request_owner(
    transaction: &Transaction<'_>,
    request_id: &str,
) -> Result<MaintenanceCounts, LogStoreError> {
    let terminal = transaction
        .query_row(
            "SELECT 1 FROM summaries WHERE request_id = ?1 AND state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')",
            [request_id],
            |_| Ok(()),
        )
        .optional()
        .map_err(LogStoreError::Sqlite)?
        .is_some();
    if !terminal {
        return Ok(MaintenanceCounts::default());
    }
    count_targets(transaction, std::slice::from_ref(&request_id.to_owned()))
}

pub(super) fn terminal_request_artifacts(
    transaction: &Transaction<'_>,
    request_id: &str,
) -> Result<Vec<CascadeArtifactPointer>, LogStoreError> {
    if count_terminal_request_owner(transaction, request_id)?.requests == 0 {
        return Ok(Vec::new());
    }
    load_request_artifact_pointers(transaction, request_id)
}

fn terminal_target_artifacts(
    transaction: &Transaction<'_>,
    targets: &[String],
) -> Result<Vec<CascadeArtifactPointer>, LogStoreError> {
    let mut pointers = Vec::new();
    for request_id in targets {
        pointers.extend(terminal_request_artifacts(transaction, request_id)?);
    }
    Ok(pointers)
}

fn load_request_artifact_pointers(
    transaction: &Transaction<'_>,
    request_id: &str,
) -> Result<Vec<CascadeArtifactPointer>, LogStoreError> {
    let mut statement = transaction
        .prepare(
            "SELECT artifact_id, request_id FROM artifact_pointers WHERE request_id = ?1 ORDER BY artifact_id ASC",
        )
        .map_err(LogStoreError::Sqlite)?;
    statement
        .query_map([request_id], |row| {
            Ok(CascadeArtifactPointer {
                artifact_id: row.get(0)?,
                request_id: row.get(1)?,
            })
        })
        .map_err(LogStoreError::Sqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(LogStoreError::Sqlite)
}

fn ensure_same_intent(
    receipt: &MaintenanceReceipt,
    existing_reason: &str,
    request: &CleanupPreviewRequest,
) -> Result<(), LogStoreError> {
    if receipt.action != MaintenanceAction::Cleanup
        || receipt.scope != request.scope
        || existing_reason != request.reason.as_str()
    {
        return Err(LogStoreError::MaintenanceOperationConflict);
    }
    Ok(())
}

pub(super) fn ensure_same_delete_one(
    transaction: &Transaction<'_>,
    receipt: &MaintenanceReceipt,
    request: &DeleteOneRequest,
) -> Result<(), LogStoreError> {
    let reason = load_reason(transaction, request.operation_id)?;
    let targets = load_targets(transaction, request.operation_id)?;
    if receipt.action != MaintenanceAction::DeleteOne
        || reason != request.reason.as_str()
        || targets != [request.request_id.clone()]
    {
        return Err(LogStoreError::MaintenanceOperationConflict);
    }
    Ok(())
}

pub(super) fn delete_one_scope() -> Result<CleanupScope, LogStoreError> {
    CleanupScope::new(MaintenanceTimestamp::try_from(DELETE_ONE_SCOPE_CUTOFF)?, 1)
}

fn load_reason(
    transaction: &Transaction<'_>,
    operation_id: MaintenanceOperationId,
) -> Result<String, LogStoreError> {
    transaction
        .query_row(
            "SELECT reason FROM maintenance_operations WHERE operation_id = ?1",
            [operation_id.to_string()],
            |row| row.get(0),
        )
        .map_err(LogStoreError::Sqlite)
}

fn write_audit(
    transaction: &Transaction<'_>,
    occurred_at: &str,
    entry_id: &str,
    operation_id: MaintenanceOperationId,
    action: &'static str,
    result: &'static str,
    reason: &str,
) -> Result<(), LogStoreError> {
    let occurred_at = canonical_persisted_timestamp(occurred_at)?;
    let detail = mesh_llm_events::audit::SanitizedAuditDetailJson::sanitize(
        &serde_json::json!({
            "actor": "trusted_local_operator",
            "source": "logs_api",
            "result": result,
            "reason": reason,
            "operationId": operation_id.to_string(),
        })
        .to_string(),
    );
    transaction
        .execute(
            "INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action, detail_json) VALUES (?1, NULL, ?2, ?3, ?4, ?5)",
            rusqlite::params![entry_id, occurred_at, "logs_api", action, detail.as_str()],
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(())
}

fn to_sql_error(error: LogStoreError) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error))
}

pub(super) fn selection_fingerprint(
    action: MaintenanceAction,
    scope: &CleanupScope,
    targets: &[String],
) -> MaintenanceFingerprint {
    let mut hasher = Sha256::new();
    hasher.update(action.as_str().as_bytes());
    hasher.update([0]);
    hasher.update(scope.cutoff_before.as_str().as_bytes());
    hasher.update([0]);
    hasher.update([scope.request_limit]);
    for value in [
        scope.filters.from(),
        scope.filters.to(),
        scope.filters.route(),
        scope.filters.model(),
        scope.filters.provider(),
        scope.filters.engine(),
        scope.filters.outcome().map(CleanupOutcome::as_str),
    ] {
        hasher.update([0]);
        if let Some(value) = value {
            hasher.update(value.as_bytes());
        }
    }
    for target in targets {
        hasher.update([0]);
        hasher.update(target.as_bytes());
    }
    MaintenanceFingerprint(hex::encode(hasher.finalize()))
}
