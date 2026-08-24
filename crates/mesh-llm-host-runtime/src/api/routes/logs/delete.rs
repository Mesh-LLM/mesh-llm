//! Trusted-local delete-one request endpoint.
//!
//! This handler owns only strict HTTP parsing, active-request protection, a
//! bounded cooperative deadline, and a path-free receipt DTO. Durable receipt
//! semantics and confined artifact deletion remain in the log-store facade.

use mesh_llm_log_store::{
    ArtifactDeletionFailureClass, ArtifactDeletionProgress, MaintenanceCounts, MaintenanceReceipt,
};
use serde::Serialize;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;

use super::{
    LoggingQueryFacade, LoggingRuntimeState, LogsError,
    maintenance_control::{MaintenanceDeadline, timeout_maintenance},
    run_blocking,
};

const DELETE_TIME_CAP: Duration = Duration::from_secs(2);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeleteReceiptDto {
    operation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    audit_id: Option<String>,
    request_id: String,
    state: &'static str,
    selection_fingerprint: String,
    planned: DeleteCountsDto,
    executed: DeleteCountsDto,
    artifact_deletion: ArtifactDeletionDto,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeleteCountsDto {
    requests: u64,
    events: u64,
    artifacts: u64,
    proxy_records: u64,
    database_rows: u64,
}

impl From<MaintenanceCounts> for DeleteCountsDto {
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

impl DeleteReceiptDto {
    fn from_receipt(request_id: String, receipt: MaintenanceReceipt) -> Result<Self, LogsError> {
        Ok(Self {
            operation_id: receipt.operation_id.to_string(),
            audit_id: receipt.execution_audit_id.clone(),
            request_id,
            state: delete_receipt_state(&receipt),
            selection_fingerprint: receipt.fingerprint.as_str().to_owned(),
            planned: receipt.planned.into(),
            executed: receipt.executed.into(),
            artifact_deletion: receipt.artifact_deletion.into(),
        })
    }

    fn pending_from_receipt(
        request_id: String,
        receipt: MaintenanceReceipt,
    ) -> Result<Self, LogsError> {
        let mut response = Self::from_receipt(request_id, receipt)?;
        response.state = "pending";
        Ok(response)
    }
}

fn delete_receipt_state(receipt: &MaintenanceReceipt) -> &'static str {
    match receipt.state {
        mesh_llm_log_store::MaintenanceReceiptState::Previewed => "pending",
        mesh_llm_log_store::MaintenanceReceiptState::Completed => "completed",
        mesh_llm_log_store::MaintenanceReceiptState::Partial
            if receipt.artifact_deletion.failed > 0 =>
        {
            "partial"
        }
        mesh_llm_log_store::MaintenanceReceiptState::Partial => "pending",
    }
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    request_id: &str,
    path: &str,
    body: &str,
) -> Result<(), LogsError> {
    handle_with_time_caps(
        stream,
        state,
        request_id,
        path,
        body,
        DELETE_TIME_CAP,
        DELETE_TIME_CAP,
    )
    .await
}

async fn handle_with_time_caps(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    request_id: &str,
    path: &str,
    body: &str,
    prepare_time_cap: Duration,
    execution_time_cap: Duration,
) -> Result<(), LogsError> {
    let request = super::parse::delete_request(request_id, path, body)?;
    let facade = super::query_facade(state)?;
    let audit_facade = facade.clone();
    let reason = request.reason.as_str().to_owned();
    let started = Instant::now();
    let control = MaintenanceDeadline::new(DELETE_TIME_CAP);
    let prepare_control = control.clone();
    let prepare_facade = facade.clone();
    let prepared = timeout_maintenance(
        prepare_time_cap,
        &control,
        run_blocking(move || prepare_terminal_delete(&prepare_facade, request, &prepare_control)),
    )
    .await;
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            write_failure_audit(&audit_facade, reason);
            return Err(error);
        }
    };

    if prepared.receipt.state == mesh_llm_log_store::MaintenanceReceiptState::Completed {
        let response = DeleteReceiptDto::from_receipt(prepared.request_id, prepared.receipt)?;
        return crate::api::http::respond_json(stream, 200, &response)
            .await
            .map_err(|_| LogsError::StoreUnavailable);
    }

    let pending = DeleteReceiptDto::pending_from_receipt(
        prepared.request_id.clone(),
        prepared.receipt.clone(),
    )?;
    let remaining = execution_time_cap.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return crate::api::http::respond_json(stream, 202, &pending)
            .await
            .map_err(|_| LogsError::StoreUnavailable);
    }
    let execute_control = control.clone();
    let execute_facade = facade.clone();
    let request = prepared.request;
    let result = timeout_maintenance(
        remaining,
        &control,
        run_blocking(move || {
            Ok(execute_facade.execute_prepared_delete_request(&request, &execute_control)?)
        }),
    )
    .await;
    let (status, response) = match result {
        Ok(receipt) => (
            200,
            DeleteReceiptDto::from_receipt(prepared.request_id, receipt)?,
        ),
        // The immutable receipt and target are already committed. The worker
        // observes cancellation between file and database mutations, so an
        // accepted timeout is pending/retryable rather than a failed delete.
        Err(LogsError::MaintenanceCancelled) => (202, pending),
        Err(error) => {
            write_failure_audit(&audit_facade, reason);
            return Err(error);
        }
    };
    crate::api::http::respond_json(stream, status, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

struct PreparedDelete {
    request: mesh_llm_log_store::DeleteOneRequest,
    request_id: String,
    receipt: MaintenanceReceipt,
}

fn prepare_terminal_delete(
    facade: &LoggingQueryFacade,
    request: super::parse::DeleteRequest,
    control: &MaintenanceDeadline,
) -> Result<PreparedDelete, LogsError> {
    let delete_request = mesh_llm_log_store::DeleteOneRequest::new(
        request.operation_id,
        &request.request_id,
        request.reason,
    )?;
    let request_id = delete_request.request_id.clone();
    if let Some(receipt) = facade.delete_one_receipt(&delete_request)? {
        return Ok(PreparedDelete {
            request: delete_request,
            request_id,
            receipt,
        });
    }
    match facade.request(&request_id)? {
        Some(record) if record.outcome == "active" => return Err(LogsError::ActiveRequest),
        Some(_) => {}
        None => return Err(LogsError::NotFound),
    }
    let receipt = facade.prepare_delete_request(&delete_request, control)?;
    Ok(PreparedDelete {
        request: delete_request,
        request_id,
        receipt,
    })
}

fn write_failure_audit(facade: &LoggingQueryFacade, reason: String) {
    let _ = facade.write_operator_audit("log_delete_request", reason, "failed");
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mesh_llm_log_store::{ArtifactDeletionFailureClass, ArtifactDeletionProgress};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use super::{ArtifactDeletionDto, DeleteCountsDto, DeleteReceiptDto, handle_with_time_caps};

    #[test]
    fn artifact_deletion_dto_uses_only_stable_failure_classes() {
        let value = serde_json::to_value(ArtifactDeletionDto::from(ArtifactDeletionProgress {
            removed: 0,
            failed: 1,
            failure_class: Some(ArtifactDeletionFailureClass::UnsafePath),
        }))
        .expect("serialize partial progress");
        assert_eq!(
            value,
            serde_json::json!({
                "removed": 0,
                "failed": 1,
                "failureClass": "unsafe_path",
            })
        );
    }

    #[test]
    fn partial_delete_receipt_omits_audit_id_for_clients() {
        let value = serde_json::to_value(DeleteReceiptDto {
            operation_id: "00000000-0000-4000-8000-000000000251".to_owned(),
            audit_id: None,
            request_id: "00000000-0000-4000-8000-000000000252".to_owned(),
            state: "partial",
            selection_fingerprint: "safe-fingerprint".to_owned(),
            planned: DeleteCountsDto {
                requests: 1,
                events: 2,
                artifacts: 1,
                proxy_records: 0,
                database_rows: 4,
            },
            executed: DeleteCountsDto {
                requests: 0,
                events: 0,
                artifacts: 0,
                proxy_records: 0,
                database_rows: 0,
            },
            artifact_deletion: ArtifactDeletionDto {
                removed: 0,
                failed: 1,
                failure_class: Some("io"),
            },
        })
        .expect("serialize partial receipt");
        assert_eq!(value["state"], "partial");
        assert!(value.get("auditId").is_none());
        assert_eq!(value["artifactDeletion"]["failureClass"], "io");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn accepted_delete_timeout_returns_pending_without_a_failure_audit() {
        let temporary_directory = tempfile::tempdir().expect("temporary logging root");
        let mut config = mesh_llm_config::LoggingConfig {
            application_state_root: Some(temporary_directory.path().join("logging")),
            ..Default::default()
        };
        config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
        crate::initialize_logging_foundation(&config).await;
        let logging = crate::logging_runtime_state().expect("logging runtime");
        let store = logging.store().expect("logging store");
        let request_id = "00000000-0000-4000-8000-000000000253";
        store
            .insert_summary(
                request_id,
                Some("safe-model"),
                Some("management"),
                None,
                None,
                "2026-08-01T00:00:00Z",
                None,
                None,
                None,
            )
            .expect("terminal summary");
        store
            .conn()
            .execute(
                "UPDATE summaries SET state = 'completed' WHERE request_id = ?1",
                [request_id],
            )
            .expect("mark terminal");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let address = listener.local_addr().expect("listener address");
        let mut client = TcpStream::connect(address).await.expect("client");
        let (mut server, _) = listener.accept().await.expect("server stream");
        handle_with_time_caps(
            &mut server,
            &logging,
            request_id,
            &format!("/api/logs/requests/{request_id}/delete"),
            r#"{"operationId":"00000000-0000-4000-8000-000000000254","reason":"operator delete"}"#,
            Duration::from_secs(1),
            Duration::ZERO,
        )
        .await
        .expect("accepted timeout response");
        server.shutdown().await.expect("shutdown server stream");
        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .await
            .expect("read response");
        assert!(response.starts_with("HTTP/1.1 202 Accepted"), "{response}");
        let body = response.split("\r\n\r\n").nth(1).expect("response body");
        let body: serde_json::Value = serde_json::from_str(body).expect("JSON response");
        assert_eq!(body["state"], "pending");
        assert!(body.get("auditId").is_none());
        assert_eq!(
            store
                .conn()
                .query_row(
                    "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_delete_request'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .expect("audit count"),
            0,
            "an accepted timeout is not a failed deletion"
        );
        crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
            enabled: false,
            ..Default::default()
        })
        .await;
    }
}
