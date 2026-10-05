//! Canonical timestamp boundaries for durable logging storage.

use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;

use crate::LogStoreError;

/// Canonicalize a timestamp before it becomes durable sortable data.
pub(crate) fn canonical_persisted_timestamp(value: &str) -> Result<String, LogStoreError> {
    canonical_logging_timestamp(value)
        .map_err(|error| LogStoreError::InsertFailed(error.to_string()))
}

/// Canonicalize an optional durable timestamp without inventing a value.
pub(crate) fn canonical_optional_persisted_timestamp(
    value: Option<&str>,
) -> Result<Option<String>, LogStoreError> {
    value.map(canonical_persisted_timestamp).transpose()
}

/// Canonicalize timestamp-shaped durable metadata while preserving its
/// documented non-timestamp variants.
pub(crate) fn canonical_timestamp_metadata(value: &str) -> String {
    canonical_logging_timestamp(value).unwrap_or_else(|_| value.to_owned())
}

/// Canonicalize a timestamp decoded from an opaque ordering cursor.
pub(crate) fn canonical_cursor_timestamp(value: &str) -> Result<String, LogStoreError> {
    canonical_logging_timestamp(value).map_err(|_| LogStoreError::CursorInvalid)
}

/// Canonicalize a caller-supplied timestamp used only for durable comparison.
pub(crate) fn canonical_comparison_timestamp(value: &str) -> Result<String, LogStoreError> {
    canonical_logging_timestamp(value)
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
}
