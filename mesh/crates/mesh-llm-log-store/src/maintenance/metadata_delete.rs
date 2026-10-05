//! Database-only delete-one execution for metadata-only logging runtimes.

use super::execution::{
    DeleteRequestPlan, complete_delete_request, count_terminal_request_owner, delete_one_scope,
    ensure_maintenance_active, ensure_same_delete_one, load_receipt, persist_receipt,
    selection_fingerprint, terminal_request_artifacts,
};
use super::*;

impl LogStore {
    /// Persist a delete-one receipt for a metadata-only runtime.
    ///
    /// This database-only lane is safe only when the terminal request has no
    /// artifact pointers. A request with pointers must be handled by the
    /// trusted artifact owner so its backing files cannot be orphaned.
    pub fn prepare_metadata_only_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        self.txn(|transaction| {
            if let Some(existing) = load_receipt(transaction, request.operation_id)? {
                ensure_same_delete_one(transaction, &existing, request)?;
                return Ok(existing);
            }
            ensure_maintenance_active(control)?;
            if !terminal_request_artifacts(transaction, &request.request_id)?.is_empty() {
                return Err(LogStoreError::ArtifactDeletionUnavailable);
            }
            let scope = delete_one_scope()?;
            let targets = vec![request.request_id.clone()];
            let planned = count_terminal_request_owner(transaction, &request.request_id)?;
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
            let occurred_at = self.now();
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

    /// Complete a database-only delete-one receipt after verifying it still
    /// has no artifact pointers. This recheck prevents a stale metadata-only
    /// decision from accepting a request whose durable artifact ownership
    /// changed before execution.
    pub fn execute_prepared_metadata_only_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let occurred_at = self.now();
        self.txn(|transaction| {
            let receipt = load_receipt(transaction, request.operation_id)?
                .ok_or(LogStoreError::MaintenanceOperationNotFound)?;
            ensure_same_delete_one(transaction, &receipt, request)?;
            if receipt.state == MaintenanceReceiptState::Completed {
                return Ok(receipt);
            }
            ensure_maintenance_active(control)?;
            if !terminal_request_artifacts(transaction, &request.request_id)?.is_empty() {
                return Err(LogStoreError::ArtifactDeletionUnavailable);
            }
            complete_delete_request(
                transaction,
                request,
                DeleteRequestPlan::Execute {
                    receipt,
                    pointers: Vec::new(),
                },
                &[],
                &occurred_at,
                control,
            )
        })
    }
}
