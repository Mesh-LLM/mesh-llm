//! Trusted-local, receipt-backed cleanup controls.
//!
//! The HTTP route owns only parsing and path-free response DTOs. Snapshot
//! selection, idempotency, owner cascade, and artifact-file confinement remain
//! inside the log-store contract and runtime facade.

use std::time::Duration;

use mesh_llm_log_store::{
    ArtifactDeletionFailureClass, ArtifactDeletionProgress, MaintenanceCounts, MaintenanceReceipt,
};
use serde::Serialize;
use tokio::net::TcpStream;

use super::{
    LoggingRuntimeState, LogsError,
    maintenance_control::{MaintenanceDeadline, timeout_maintenance},
    run_blocking,
};

const CLEANUP_TIME_CAP: Duration = Duration::from_secs(2);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CleanupReceiptDto {
    operation_id: String,
    audit_id: String,
    cutoff_before: String,
    request_limit: usize,
    scope: CleanupScopeDto,
    state: &'static str,
    has_more: bool,
    selection_fingerprint: String,
    planned: CleanupCountsDto,
    executed: CleanupCountsDto,
    artifact_deletion: ArtifactDeletionDto,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CleanupScopeDto {
    source: &'static str,
    cutoff_before: String,
    request_limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CleanupCountsDto {
    requests: u64,
    events: u64,
    artifacts: u64,
    proxy_records: u64,
    database_rows: u64,
}

impl From<MaintenanceCounts> for CleanupCountsDto {
    fn from(value: MaintenanceCounts) -> Self {
        Self {
            requests: value.requests,
            events: value.events,
            artifacts: value.artifacts,
            proxy_records: value.proxy_records,
            database_rows: value.database_rows,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactDeletionDto {
    removed: u64,
    failed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure_class: Option<&'static str>,
}

impl From<ArtifactDeletionProgress> for ArtifactDeletionDto {
    fn from(value: ArtifactDeletionProgress) -> Self {
        Self {
            removed: value.removed,
            failed: value.failed,
            failure_class: value.failure_class.map(|class| match class {
                ArtifactDeletionFailureClass::Io => "io",
                ArtifactDeletionFailureClass::UnsafePath => "unsafe_path",
            }),
        }
    }
}

impl CleanupReceiptDto {
    fn preview_from_receipt(value: MaintenanceReceipt) -> Result<Self, LogsError> {
        let audit_id = value
            .preview_audit_id
            .clone()
            .ok_or(LogsError::StoreUnavailable)?;
        Ok(Self::from_receipt(value, audit_id))
    }

    fn execution_from_receipt(value: MaintenanceReceipt) -> Result<Self, LogsError> {
        let audit_id = value
            .execution_audit_id
            .clone()
            .ok_or(LogsError::StoreUnavailable)?;
        Ok(Self::from_receipt(value, audit_id))
    }

    fn from_receipt(value: MaintenanceReceipt, audit_id: String) -> Self {
        Self {
            operation_id: value.operation_id.to_string(),
            audit_id,
            cutoff_before: value.scope.cutoff_before().as_str().to_owned(),
            request_limit: value.scope.request_limit(),
            scope: CleanupScopeDto::from_scope(&value.scope),
            state: value.state.as_str(),
            has_more: value.has_more,
            selection_fingerprint: value.fingerprint.as_str().to_owned(),
            planned: value.planned.into(),
            executed: value.executed.into(),
            artifact_deletion: value.artifact_deletion.into(),
        }
    }
}

impl CleanupScopeDto {
    fn from_scope(scope: &mesh_llm_log_store::CleanupScope) -> Self {
        let filters = scope.filters();
        Self {
            source: "durable",
            cutoff_before: scope.cutoff_before().as_str().to_owned(),
            request_limit: scope.request_limit(),
            from: filters.from().map(str::to_owned),
            to: filters.to().map(str::to_owned),
            route: filters.route().map(str::to_owned),
            model: filters.model().map(str::to_owned),
            provider: filters.provider().map(str::to_owned),
            engine: filters.engine().map(str::to_owned),
            outcome: filters
                .outcome()
                .map(mesh_llm_log_store::CleanupOutcome::as_str),
        }
    }
}

pub(super) async fn preview(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    path: &str,
    body: &str,
) -> Result<(), LogsError> {
    let request = super::parse::cleanup_preview_request(path, body)?;
    let facade = state.query_facade().ok_or(LogsError::ServiceUnavailable)?;
    let audit_facade = facade.clone();
    let reason = request.reason.as_str().to_owned();
    let control = MaintenanceDeadline::new(CLEANUP_TIME_CAP);
    let worker_control = control.clone();
    let result = timeout_maintenance(
        CLEANUP_TIME_CAP,
        &control,
        run_blocking(move || {
            let receipt = facade.preview_cleanup(&request, &worker_control)?;
            CleanupReceiptDto::preview_from_receipt(receipt)
        }),
    )
    .await;
    write_failure_audit(&audit_facade, "log_cleanup_preview", reason, &result);
    let response = result?;
    crate::api::http::respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

pub(super) async fn run(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    path: &str,
    body: &str,
) -> Result<(), LogsError> {
    let request = super::parse::cleanup_run_request(path, body)?;
    let facade = state.query_facade().ok_or(LogsError::ServiceUnavailable)?;
    let audit_facade = facade.clone();
    let reason = request.reason.as_str().to_owned();
    let control = MaintenanceDeadline::new(CLEANUP_TIME_CAP);
    let worker_control = control.clone();
    let result = timeout_maintenance(
        CLEANUP_TIME_CAP,
        &control,
        run_blocking(move || {
            let receipt =
                facade.execute_cleanup(request.operation_id, &request.reason, &worker_control)?;
            CleanupReceiptDto::execution_from_receipt(receipt)
        }),
    )
    .await;
    write_failure_audit(&audit_facade, "log_cleanup_run", reason, &result);
    let response = result?;
    crate::api::http::respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

fn write_failure_audit(
    facade: &super::LoggingQueryFacade,
    action: &'static str,
    reason: String,
    result: &Result<CleanupReceiptDto, LogsError>,
) {
    if result.is_err() {
        let _ = facade.write_operator_audit(action, reason, "failed");
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_log_store::{ArtifactDeletionFailureClass, ArtifactDeletionProgress};

    use super::ArtifactDeletionDto;

    #[test]
    fn artifact_deletion_dto_is_path_free_and_omits_default_failure_class() {
        let default = serde_json::to_value(ArtifactDeletionDto::from(
            ArtifactDeletionProgress::default(),
        ))
        .expect("serialize default progress");
        assert_eq!(default, serde_json::json!({ "removed": 0, "failed": 0 }));

        for (failure_class, expected) in [
            (ArtifactDeletionFailureClass::Io, "io"),
            (ArtifactDeletionFailureClass::UnsafePath, "unsafe_path"),
        ] {
            let partial =
                serde_json::to_value(ArtifactDeletionDto::from(ArtifactDeletionProgress {
                    removed: 1,
                    failed: 1,
                    failure_class: Some(failure_class),
                }))
                .expect("serialize partial progress");
            assert_eq!(
                partial,
                serde_json::json!({
                    "removed": 1,
                    "failed": 1,
                    "failureClass": expected,
                })
            );
        }
    }
}
