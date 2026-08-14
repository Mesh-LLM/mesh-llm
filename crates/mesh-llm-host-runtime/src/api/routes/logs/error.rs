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
    SchemaIncompatible { found: u32, supported: u32 },
}

impl LogsError {
    pub(crate) fn unavailable(state: &crate::logging::LoggingRuntimeState) -> Self {
        match state.metadata_state() {
            crate::logging::LoggingMetadataState::SchemaIncompatible { found, supported } => {
                Self::SchemaIncompatible { found, supported }
            }
            _ => Self::ServiceUnavailable,
        }
    }

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
            Self::SchemaIncompatible { .. } => (
                503,
                "logging_schema_incompatible",
                "the local log database schema is incompatible with this MeshLLM version",
            ),
        };
        let details = match self {
            Self::SchemaIncompatible { found, supported } => Some(ErrorDetails {
                schema_version: found,
                supported_schema_version: supported,
            }),
            _ => None,
        };
        respond_json(
            stream,
            status,
            &ErrorResponse {
                error: ErrorBody {
                    code,
                    message,
                    details,
                },
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
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<ErrorDetails>,
}

#[derive(Serialize)]
struct ErrorDetails {
    schema_version: u32,
    supported_schema_version: u32,
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
            mesh_llm_log_store::LogStoreError::SchemaIncompatible { found, supported } => {
                Self::SchemaIncompatible { found, supported }
            }
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

    #[test]
    fn schema_incompatibility_preserves_only_the_version_pair() {
        assert_eq!(
            LogsError::from(mesh_llm_log_store::LogStoreError::SchemaIncompatible {
                found: 14,
                supported: 11,
            }),
            LogsError::SchemaIncompatible {
                found: 14,
                supported: 11,
            }
        );
    }
}
