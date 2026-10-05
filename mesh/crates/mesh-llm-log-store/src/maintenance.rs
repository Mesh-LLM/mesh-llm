//! Durable, bounded operator cleanup receipts.
//!
//! Preview snapshots a limited set of terminal-summary owners. Execute consumes
//! that snapshot exactly once in one SQLite transaction; a completed operation
//! ID is replayed without repeating either deletion or its audit entry.

use std::{collections::HashSet, convert::TryFrom, fmt};

use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;
use rusqlite::{OptionalExtension, Transaction};
use uuid::Uuid;

use crate::artifacts::{CascadeArtifactDeleteFailure, CascadeArtifactDeleteResult};
use crate::timestamps::canonical_persisted_timestamp;
use crate::{ArtifactFileStore, CascadeArtifactPointer, LogStore, LogStoreError};
use sha2::{Digest, Sha256};

const MAX_CLEANUP_REQUESTS: usize = 100;
const MAX_REASON_BYTES: usize = 256;
const DELETE_ONE_SCOPE_CUTOFF: &str = "1970-01-01T00:00:00Z";

mod metadata_delete;
mod scope_filters;
#[cfg(test)]
mod tests;

pub use scope_filters::{CleanupFilters, CleanupOutcome};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceAction {
    Cleanup,
    DeleteOne,
}

impl MaintenanceAction {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cleanup => "cleanup",
            Self::DeleteOne => "delete_one",
        }
    }

    fn from_str(value: &str) -> Result<Self, LogStoreError> {
        match value {
            "cleanup" => Ok(Self::Cleanup),
            "delete_one" => Ok(Self::DeleteOne),
            _ => Err(LogStoreError::QueryFailed(
                "invalid maintenance action".to_owned(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaintenanceOperationId(Uuid);

impl MaintenanceOperationId {
    pub const fn new(value: Uuid) -> Self {
        Self(value)
    }
}

impl fmt::Display for MaintenanceOperationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl TryFrom<&str> for MaintenanceOperationId {
    type Error = LogStoreError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Uuid::parse_str(value)
            .map(Self)
            .map_err(|_| LogStoreError::MaintenanceScopeInvalid {
                field: "operation_id",
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintenanceReason(String);

impl MaintenanceReason {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for MaintenanceReason {
    type Error = LogStoreError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let value = value.trim();
        if value.is_empty() || value.len() > MAX_REASON_BYTES || value.chars().any(char::is_control)
        {
            return Err(LogStoreError::MaintenanceScopeInvalid { field: "reason" });
        }
        Ok(Self(value.to_owned()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintenanceTimestamp(String);

impl MaintenanceTimestamp {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for MaintenanceTimestamp {
    type Error = LogStoreError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let value = value.trim();
        let value = canonical_logging_timestamp(value).map_err(|_| {
            LogStoreError::MaintenanceScopeInvalid {
                field: "cutoff_before",
            }
        })?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupScope {
    cutoff_before: MaintenanceTimestamp,
    request_limit: u8,
    filters: CleanupFilters,
}

impl CleanupScope {
    pub fn new(
        cutoff_before: MaintenanceTimestamp,
        request_limit: usize,
    ) -> Result<Self, LogStoreError> {
        if !(1..=MAX_CLEANUP_REQUESTS).contains(&request_limit) {
            return Err(LogStoreError::MaintenanceScopeInvalid {
                field: "request_limit",
            });
        }
        Ok(Self {
            cutoff_before,
            request_limit: request_limit as u8,
            filters: CleanupFilters::default(),
        })
    }

    pub fn with_filters(mut self, filters: CleanupFilters) -> Self {
        self.filters = filters;
        self
    }

    pub fn cutoff_before(&self) -> &MaintenanceTimestamp {
        &self.cutoff_before
    }

    pub const fn request_limit(&self) -> usize {
        self.request_limit as usize
    }

    pub const fn filters(&self) -> &CleanupFilters {
        &self.filters
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupPreviewRequest {
    pub operation_id: MaintenanceOperationId,
    pub scope: CleanupScope,
    pub reason: MaintenanceReason,
}

/// Delete exactly one durable request owner. The operation ID is immutable:
/// retries with the same ID replay its receipt, including a missing target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeleteOneRequest {
    pub operation_id: MaintenanceOperationId,
    pub request_id: String,
    pub reason: MaintenanceReason,
}

impl DeleteOneRequest {
    pub fn new(
        operation_id: MaintenanceOperationId,
        request_id: &str,
        reason: MaintenanceReason,
    ) -> Result<Self, LogStoreError> {
        let request_id = Uuid::parse_str(request_id)
            .map(|value| value.to_string())
            .map_err(|_| LogStoreError::MaintenanceScopeInvalid {
                field: "request_id",
            })?;
        Ok(Self {
            operation_id,
            request_id,
            reason,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaintenanceCounts {
    pub requests: u64,
    pub events: u64,
    pub artifacts: u64,
    pub proxy_records: u64,
    pub database_rows: u64,
}

/// Path-free, durable progress for one maintenance artifact cascade.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArtifactDeletionProgress {
    /// Durable artifact pointer rows reconciled after a file was removed or
    /// was already missing.
    pub removed: u64,
    /// Pointer-owned files that could not be removed and therefore remain
    /// durable and eligible for an exact same-operation retry.
    pub failed: u64,
    /// Coarse stable class for the current failed set. It never includes an
    /// OS message or filesystem path.
    pub failure_class: Option<ArtifactDeletionFailureClass>,
}

/// Stable classification for a failed maintenance artifact removal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactDeletionFailureClass {
    Io,
    UnsafePath,
}

impl From<CascadeArtifactDeleteFailure> for ArtifactDeletionFailureClass {
    fn from(value: CascadeArtifactDeleteFailure) -> Self {
        match value {
            CascadeArtifactDeleteFailure::Io => Self::Io,
            CascadeArtifactDeleteFailure::UnsafePath => Self::UnsafePath,
        }
    }
}

impl ArtifactDeletionFailureClass {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Io => "io",
            Self::UnsafePath => "unsafe_path",
        }
    }

    fn from_str(value: &str) -> Result<Self, LogStoreError> {
        match value {
            "io" => Ok(Self::Io),
            "unsafe_path" => Ok(Self::UnsafePath),
            _ => Err(LogStoreError::QueryFailed(
                "invalid artifact deletion failure class".to_owned(),
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintenanceFingerprint(String);

impl MaintenanceFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceReceiptState {
    Previewed,
    Completed,
    Partial,
}

impl MaintenanceReceiptState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Previewed => "previewed",
            Self::Completed => "completed",
            Self::Partial => "partial",
        }
    }

    fn from_str(value: &str) -> Result<Self, LogStoreError> {
        match value {
            "previewed" => Ok(Self::Previewed),
            "completed" => Ok(Self::Completed),
            "partial" => Ok(Self::Partial),
            _ => Err(LogStoreError::QueryFailed(
                "invalid maintenance receipt state".to_owned(),
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintenanceReceipt {
    pub operation_id: MaintenanceOperationId,
    pub action: MaintenanceAction,
    pub scope: CleanupScope,
    pub state: MaintenanceReceiptState,
    pub planned: MaintenanceCounts,
    pub executed: MaintenanceCounts,
    pub artifact_deletion: ArtifactDeletionProgress,
    pub has_more: bool,
    pub fingerprint: MaintenanceFingerprint,
    /// Audit entry created with the durable cleanup preview, if applicable.
    pub preview_audit_id: Option<String>,
    /// Most recent audit entry created with successful cleanup/delete execution.
    pub execution_audit_id: Option<String>,
}

/// A narrow cooperative cancellation seam. The caller may back it with a
/// deadline or shutdown flag; execution checks it before each transaction
/// mutation and never commits a partially selected target list.
pub trait MaintenanceExecutionControl: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

mod execution;
