use serde::Serialize;
use tokio::net::TcpStream;

use crate::api::http::respond_json;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LogsError {
    Forbidden,
    InvalidHeaders,
    InvalidJsonBody,
    NotAcceptable,
    InvalidQuery(&'static str),
    InvalidCursor,
    CursorExpired,
    InvalidId,
    InvalidWebhookDeliveryId,
    NotFound,
    ArtifactExportForbidden,
    ExportTimedOut,
    MaintenanceConflict,
    MaintenanceCancelled,
    ArtifactDeletionUnavailable,
    ActiveRequest,
    CleanupMethodNotAllowed,
    DeleteMethodNotAllowed,
    WebhookRetryMethodNotAllowed,
    ExportMethodNotAllowed,
    WebhookNotRetryable,
    MethodNotAllowed,
    ServiceUnavailable,
    StoreUnavailable,
}

impl LogsError {
    pub(crate) async fn write(self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let (status, code, message) = match self {
            Self::Forbidden => (
                403,
                "forbidden",
                "this log route requires a trusted local caller",
            ),
            Self::InvalidHeaders => (400, "invalid_request", "request headers are invalid"),
            Self::InvalidJsonBody => (400, "invalid_request", "request body is invalid"),
            Self::NotAcceptable => (
                406,
                "not_acceptable",
                "route requires Accept: text/event-stream",
            ),
            Self::InvalidQuery(message) => (400, "invalid_query", message),
            Self::InvalidCursor => (400, "invalid_cursor", "cursor is malformed"),
            Self::CursorExpired => (400, "cursor_expired", "cursor is no longer available"),
            Self::InvalidId => (400, "invalid_id", "identifier must be a UUID"),
            Self::InvalidWebhookDeliveryId => (
                400,
                "invalid_webhook_delivery_id",
                "webhook delivery identifier is invalid",
            ),
            Self::NotFound => (404, "not_found", "log record was not found"),
            Self::ArtifactExportForbidden => (
                403,
                "artifact_export_forbidden",
                "artifact bytes require redacted capture and explicit authorization",
            ),
            Self::ExportTimedOut => (
                503,
                "export_timed_out",
                "log export exceeded its bounded execution window",
            ),
            Self::MaintenanceConflict => (
                409,
                "maintenance_conflict",
                "maintenance operation conflicts with its recorded preview",
            ),
            Self::MaintenanceCancelled => (
                503,
                "maintenance_cancelled",
                "maintenance operation did not complete within its bounded window",
            ),
            Self::ArtifactDeletionUnavailable => (
                503,
                "artifact_deletion_unavailable",
                "request artifact deletion requires an unavailable trusted artifact owner",
            ),
            Self::ActiveRequest => (409, "request_active", "active requests cannot be deleted"),
            Self::CleanupMethodNotAllowed => {
                (405, "method_not_allowed", "cleanup routes require POST")
            }
            Self::DeleteMethodNotAllowed => {
                (405, "method_not_allowed", "request deletion requires POST")
            }
            Self::WebhookRetryMethodNotAllowed => {
                (405, "method_not_allowed", "webhook retry requires POST")
            }
            Self::ExportMethodNotAllowed => {
                (405, "method_not_allowed", "request export requires POST")
            }
            Self::WebhookNotRetryable => (
                409,
                "webhook_not_retryable",
                "webhook delivery is not eligible for manual retry",
            ),
            Self::MethodNotAllowed => (
                405,
                "method_not_allowed",
                "route requires GET without a request body",
            ),
            Self::ServiceUnavailable => (
                503,
                "logging_unavailable",
                "logging service is not available",
            ),
            Self::StoreUnavailable => (503, "store_unavailable", "logging store is not available"),
        };
        respond_json(
            stream,
            status,
            &ErrorResponse {
                error: ErrorBody { code, message },
            },
        )
        .await
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: &'static str,
}

impl From<mesh_llm_log_store::LogStoreError> for LogsError {
    fn from(error: mesh_llm_log_store::LogStoreError) -> Self {
        match error {
            mesh_llm_log_store::LogStoreError::CursorMalformed(_) => Self::InvalidCursor,
            mesh_llm_log_store::LogStoreError::CursorInvalid => Self::CursorExpired,
            mesh_llm_log_store::LogStoreError::PathUnsafe { .. } => Self::InvalidId,
            mesh_llm_log_store::LogStoreError::ArtifactMissing { .. }
            | mesh_llm_log_store::LogStoreError::ArtifactCorrupt { .. } => Self::NotFound,
            mesh_llm_log_store::LogStoreError::Sqlite(_)
            | mesh_llm_log_store::LogStoreError::ConnectionPoisoned
            | mesh_llm_log_store::LogStoreError::MigrationFailed(_)
            | mesh_llm_log_store::LogStoreError::InsertFailed(_)
            | mesh_llm_log_store::LogStoreError::DuplicateTerminalEvent { .. }
            | mesh_llm_log_store::LogStoreError::AlreadyExists { .. }
            | mesh_llm_log_store::LogStoreError::ForeignKeyViolation { .. }
            | mesh_llm_log_store::LogStoreError::QueryFailed(_)
            | mesh_llm_log_store::LogStoreError::IoError(_)
            | mesh_llm_log_store::LogStoreError::ArtifactLimitExceeded { .. }
            | mesh_llm_log_store::LogStoreError::PrivacyNotGuaranteed
            | mesh_llm_log_store::LogStoreError::InvalidQuery(_) => Self::StoreUnavailable,
            mesh_llm_log_store::LogStoreError::MaintenanceScopeInvalid { .. } => {
                Self::InvalidQuery("cleanup request is invalid")
            }
            mesh_llm_log_store::LogStoreError::MaintenanceOperationConflict => {
                Self::MaintenanceConflict
            }
            mesh_llm_log_store::LogStoreError::MaintenanceOperationNotFound => Self::NotFound,
            mesh_llm_log_store::LogStoreError::MaintenanceExecutionCancelled => {
                Self::MaintenanceCancelled
            }
            mesh_llm_log_store::LogStoreError::ArtifactDeletionUnavailable => {
                Self::ArtifactDeletionUnavailable
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LogsError;

    #[test]
    fn poisoned_connection_maps_to_store_unavailable() {
        assert_eq!(
            LogsError::from(mesh_llm_log_store::LogStoreError::ConnectionPoisoned),
            LogsError::StoreUnavailable
        );
    }
}
