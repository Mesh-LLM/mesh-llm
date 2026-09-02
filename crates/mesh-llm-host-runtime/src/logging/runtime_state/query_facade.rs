use super::*;
use std::collections::BTreeMap;

/// Narrow, path-free query and read facade for trusted-local log routes.
///
/// It intentionally owns the active snapshot source, SQLite query substrate,
/// and confined artifact reader together so API code cannot reach into old
/// `ArtifactFileStore` ownership or discover local filesystem paths.
#[derive(Clone)]
pub(crate) struct LoggingQueryFacade {
    store: Arc<LogStore>,
    service: Arc<LoggingService>,
    artifact_capture: Option<Arc<FailOpenArtifactCapture>>,
    artifact_export_enabled: bool,
    export_limit_bytes: usize,
    operator_audit_writer: Arc<OperatorAuditWriter>,
    #[cfg(test)]
    query_counter: Arc<LoggingQueryCounter>,
}

/// Ordered child windows collected for one bounded log export page.
///
/// Keeping the two child collections together makes the query facade's export
/// contract explicit while preserving its fixed two-query fan-in.
pub(crate) struct ExportChildren {
    pub(crate) events_by_request: BTreeMap<String, Vec<EventRecord>>,
    pub(crate) artifacts_by_request: BTreeMap<String, Vec<ArtifactRecord>>,
}

impl LoggingQueryFacade {
    pub(super) fn from_runtime(state: &LoggingRuntimeState) -> Option<Self> {
        Some(Self {
            store: state.store.clone()?,
            service: state.service.clone()?,
            artifact_capture: state.artifact_capture.clone(),
            artifact_export_enabled: state.artifact_export_enabled,
            export_limit_bytes: state.export_limit_bytes,
            operator_audit_writer: Arc::clone(&state.operator_audit_writer),
            #[cfg(test)]
            query_counter: Arc::clone(&state.query_counter),
        })
    }

    pub(crate) fn snapshot_active(&self) -> ActiveRequestSnapshot {
        self.service.registry_ref().snapshot_active()
    }

    pub(crate) fn request(
        &self,
        request_id: &str,
    ) -> Result<Option<RequestRecordWithCaller>, LogStoreError> {
        #[cfg(test)]
        self.query_counter
            .point_requests
            .fetch_add(1, Ordering::Relaxed);
        self.store.query_request_with_caller(request_id)
    }

    pub(crate) fn requests_by_ids(
        &self,
        request_ids: &[String],
    ) -> Result<Vec<RequestRecordWithCaller>, LogStoreError> {
        #[cfg(test)]
        self.query_counter
            .batch_requests
            .fetch_add(1, Ordering::Relaxed);
        self.store.query_requests_by_ids_with_caller(request_ids)
    }

    pub(crate) fn requests(
        &self,
        query: &RequestQuery,
    ) -> Result<QueryPage<RequestRecordWithCaller>, LogStoreError> {
        self.store.query_requests_with_caller(query)
    }

    pub(crate) fn events(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<EventRecord>, LogStoreError> {
        self.store.query_events(request_id, query)
    }

    pub(crate) fn artifacts(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<ArtifactRecord>, LogStoreError> {
        self.store.query_artifacts(request_id, query)
    }

    /// Batch child windows for a bounded export page so the API has stable
    /// per-owner ordering without issuing one events query and one artifacts
    /// query for every summary in that page.
    pub(crate) fn export_children(
        &self,
        request_ids: &[String],
        per_request_limit: usize,
    ) -> Result<ExportChildren, LogStoreError> {
        Ok(ExportChildren {
            events_by_request: self
                .store
                .query_events_for_requests(request_ids, per_request_limit)?,
            artifacts_by_request: self
                .store
                .query_artifacts_for_requests(request_ids, per_request_limit)?,
        })
    }

    pub(crate) fn artifact(
        &self,
        artifact_id: &str,
    ) -> Result<Option<ArtifactRecord>, LogStoreError> {
        self.store.query_artifact(artifact_id)
    }

    pub(crate) fn proxy_records(
        &self,
        query: &ProxyQuery,
    ) -> Result<QueryPage<ProxyRecord>, LogStoreError> {
        self.store.query_proxy_records(query)
    }

    pub(crate) fn audit_entries(
        &self,
        limit: Option<usize>,
        after_cursor: Option<&str>,
        filters: AuditEntryFilters,
    ) -> Result<Page<AuditEntryDetail>, LogStoreError> {
        self.store
            .list_audit_entry_details(limit, after_cursor, filters)
    }

    pub(crate) fn audit_entries_after_sequence(
        &self,
        after_sequence: u64,
        limit: usize,
        filters: AuditEntryFilters,
    ) -> Result<Vec<AuditEntryDetail>, LogStoreError> {
        self.store
            .list_audit_entry_details_after_sequence(after_sequence, limit, filters)
    }

    /// Read content only through the fail-open capture owner. The route core
    /// verifies `ArtifactRecord::redacted` before calling this method.
    pub(crate) fn read_artifact(
        &self,
        artifact_id: &str,
    ) -> Result<ArtifactContent, LogStoreError> {
        let Some(capture) = &self.artifact_capture else {
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        };
        capture.read_artifact(artifact_id)
    }

    /// Whether a trusted-local export may include redacted artifact bytes.
    /// Metadata queries do not need this opt-in because they never read files.
    pub(crate) const fn artifact_export_enabled(&self) -> bool {
        self.artifact_export_enabled
    }

    /// The configured upper bound for one operator export response.
    pub(crate) const fn export_limit_bytes(&self) -> usize {
        self.export_limit_bytes
    }

    /// Persist one operator action without routing its failure through the
    /// logging service. The shared recursion guard keeps an audit-store error
    /// from recursively creating audit records of its own.
    pub(crate) fn write_operator_audit(
        &self,
        action: &'static str,
        reason: String,
        result: &'static str,
    ) -> Result<(), LogStoreError> {
        self.operator_audit_writer
            .write(Arc::clone(&self.store), action, reason, result)
    }

    pub(crate) fn preview_cleanup(
        &self,
        request: &mesh_llm_log_store::CleanupPreviewRequest,
        control: &dyn mesh_llm_log_store::MaintenanceExecutionControl,
    ) -> Result<mesh_llm_log_store::MaintenanceReceipt, LogStoreError> {
        self.store.preview_cleanup(request, control)
    }

    pub(crate) fn execute_cleanup(
        &self,
        operation_id: mesh_llm_log_store::MaintenanceOperationId,
        reason: &mesh_llm_log_store::MaintenanceReason,
        control: &dyn mesh_llm_log_store::MaintenanceExecutionControl,
    ) -> Result<mesh_llm_log_store::MaintenanceReceipt, LogStoreError> {
        if let Some(capture) = &self.artifact_capture {
            return capture.execute_cleanup(operation_id, reason, control);
        }

        let receipt = self.store.cleanup_receipt(operation_id, reason)?;
        if receipt.state == mesh_llm_log_store::MaintenanceReceiptState::Completed
            || (receipt.state == mesh_llm_log_store::MaintenanceReceiptState::Partial
                && receipt.artifact_deletion.failed == 0)
        {
            return Ok(receipt);
        }
        Err(LogStoreError::MaintenanceExecutionCancelled)
    }

    /// Delete exactly one terminal durable request through the confined
    /// artifact owner. The API layer never receives filesystem paths.
    pub(crate) fn delete_request_cascade(
        &self,
        request: &mesh_llm_log_store::DeleteOneRequest,
        control: &dyn mesh_llm_log_store::MaintenanceExecutionControl,
    ) -> Result<mesh_llm_log_store::MaintenanceReceipt, LogStoreError> {
        let Some(capture) = &self.artifact_capture else {
            return Err(LogStoreError::MaintenanceExecutionCancelled);
        };
        capture.delete_request_cascade(request, control)
    }

    /// Persist a delete-one operation's immutable receipt and target before
    /// the route starts bounded artifact execution.
    pub(crate) fn prepare_delete_request(
        &self,
        request: &mesh_llm_log_store::DeleteOneRequest,
        control: &dyn mesh_llm_log_store::MaintenanceExecutionControl,
    ) -> Result<mesh_llm_log_store::MaintenanceReceipt, LogStoreError> {
        match self.artifact_capture.as_ref() {
            Some(capture) if !capture.is_disabled() => {
                capture.prepare_delete_request(request, control)
            }
            _ => self
                .store
                .prepare_metadata_only_delete_request(request, control),
        }
    }

    /// Continue a delete-one operation after its durable receipt has been
    /// accepted. The artifact owner keeps all filesystem paths private.
    pub(crate) fn execute_prepared_delete_request(
        &self,
        request: &mesh_llm_log_store::DeleteOneRequest,
        control: &dyn mesh_llm_log_store::MaintenanceExecutionControl,
    ) -> Result<mesh_llm_log_store::MaintenanceReceipt, LogStoreError> {
        match self.artifact_capture.as_ref() {
            Some(capture) if !capture.is_disabled() => {
                capture.execute_prepared_delete_request(request, control)
            }
            _ => self
                .store
                .execute_prepared_metadata_only_delete_request(request, control),
        }
    }

    /// Load a matching delete-one receipt before checking its former request
    /// owner, which may have been removed by the original operation.
    pub(crate) fn delete_one_receipt(
        &self,
        request: &mesh_llm_log_store::DeleteOneRequest,
    ) -> Result<Option<mesh_llm_log_store::MaintenanceReceipt>, LogStoreError> {
        self.store.delete_one_receipt(request)
    }

    /// Transition one dead-letter delivery into the separately auditable
    /// manual-retry state. The route receives only a fixed atomic outcome;
    /// endpoint and payload material remain private to the worker.
    pub(crate) fn manually_retry_webhook_delivery(
        &self,
        delivery_id: &str,
    ) -> Result<mesh_llm_log_store::WebhookManualRetryOutcome, LogStoreError> {
        self.store
            .manually_retry_webhook_delivery(delivery_id, &self.store.now())
    }

    /// Load the state needed to make a manual-retry response idempotent. The
    /// API reduces this record to a fixed outcome label and never serializes
    /// its delivery metadata.
    pub(crate) fn webhook_delivery(
        &self,
        delivery_id: &str,
    ) -> Result<Option<mesh_llm_log_store::WebhookDeliveryRecord>, LogStoreError> {
        self.store.webhook_delivery(delivery_id)
    }
}
