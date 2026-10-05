//! SQLite persistence for mesh-llm canonical logging pipeline.

#[cfg(test)]
mod api_acceptance_tests;
#[cfg(test)]
mod artifact_unavailable_tests;
#[cfg(test)]
mod artifacts_tests;
#[cfg(test)]
mod capture_tests;
#[cfg(test)]
mod query_pagination_tests;
#[cfg(test)]
mod query_tests;
#[cfg(test)]
mod space_maintenance_tests;

mod artifact_privacy;
mod artifact_repository;
mod artifacts;
mod audit_detail;
mod capture;
mod cursor;
mod error;
mod maintenance;
mod migrations;
mod query;
mod repositories;
mod schema;
mod store;
mod timestamps;

// Re-export primary types at crate root.
#[doc(hidden)]
pub use artifact_privacy::ArtifactPrivacy;
pub use artifact_repository::UnavailableArtifactPointer;
pub use artifacts::{
    ArtifactContent, ArtifactFileStore, ArtifactRedactor, ArtifactStatus, ArtifactWriteReceipt,
};
pub use audit_detail::AuditEntryDetail;
pub use capture::{
    ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE, ArtifactCaptureDisabledReason,
    ArtifactCaptureHealthMarker, ArtifactCaptureOutcome, FailOpenArtifactCapture,
};
pub use cursor::{decode_cursor, encode_cursor};
pub use error::LogStoreError;
pub use maintenance::{
    ArtifactDeletionFailureClass, ArtifactDeletionProgress, CleanupFilters, CleanupOutcome,
    CleanupPreviewRequest, CleanupScope, DeleteOneRequest, MaintenanceAction, MaintenanceCounts,
    MaintenanceExecutionControl, MaintenanceFingerprint, MaintenanceOperationId, MaintenanceReason,
    MaintenanceReceipt, MaintenanceReceiptState, MaintenanceTimestamp,
};
pub use migrations::CURRENT_VERSION as LOG_STORE_SCHEMA_VERSION;
pub use query::{
    ArtifactRecord, EventRecord, MAX_QUERY_LIMIT, PageQuery, ProxyQuery, ProxyRecord, QueryPage,
    QuerySort, RequestOutcome, RequestQuery, RequestRecord, RequestRecordWithCaller,
};
pub use repositories::{
    AuditEntryFilters, AuditEntryRow, AuditEntrySeverity, AuditEntrySource, CascadeArtifactPointer,
    DEFAULT_AUDIT_ENTRY_LIMIT, MAX_AUDIT_ENTRY_LIMIT, Page, RetentionCleanupResult,
    RetentionPolicy, RetentionTable, RetentionTablePolicy, RetentionTableResult,
    WebhookDeliveryErrorCode, WebhookDeliveryInsertOutcome, WebhookDeliveryRecord,
    WebhookDeliveryState, WebhookManualRetryOutcome, WebhookRetryOutcome, WebhookTerminalOutcome,
};
pub use store::{Clock, LogStore, SQLiteSpaceMaintenance, SystemClock as RealClock};
