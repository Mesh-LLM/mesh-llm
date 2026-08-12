//! Typed error types for log-store operations.

use std::fmt;

#[derive(Debug)]
pub enum LogStoreError {
    /// SQLite-level failure (connection, schema, etc.)
    Sqlite(rusqlite::Error),

    /// Schema migration failed to apply cleanly.
    MigrationFailed(String),

    /// Cursor decode/encode error.
    CursorMalformed(String),

    /// Insert failed due to a conflict.
    InsertFailed(String),

    /// Duplicate terminal event for summary + event_type pair.
    DuplicateTerminalEvent {
        summary_id: String,
        event_type: String,
    },

    /// Entity already exists (unique constraint).
    AlreadyExists { entity: String },

    /// Insert references a parent row that does not exist.
    ForeignKeyViolation { entity: String },

    /// General query failure.
    QueryFailed(String),

    /// I/O error on the store path.
    IoError(std::io::Error),

    /// Artifact write rejected: content exceeds byte_limit or aggregate limit for request.
    ArtifactLimitExceeded {
        artifact_id: String,
        limit_bytes: usize,
        kind: String, // "byte" or "aggregate"
    },

    /// Artifact file on disk has a checksum mismatch with the DB row — corrupt.
    ArtifactCorrupt { artifact_id: String },

    /// Artifact pointer exists in DB but the backing file is missing from disk.
    ArtifactMissing { artifact_id: String },

    /// Caller-supplied ID contains path-traversal characters (/ \ .. NUL).
    PathUnsafe { segment: String },

    /// Platform privacy cannot be guaranteed for an artifact path.
    PrivacyNotGuaranteed,
}

impl fmt::Display for LogStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite error: {}", e),
            Self::MigrationFailed(msg) => write!(f, "migration failed: {}", msg),
            Self::CursorMalformed(msg) => write!(f, "cursor malformed: {}", msg),
            Self::InsertFailed(msg) => write!(f, "insert failed: {}", msg),
            Self::DuplicateTerminalEvent {
                summary_id,
                event_type,
            } => {
                write!(
                    f,
                    "duplicate terminal event for summary={} type={}",
                    summary_id, event_type
                )
            }
            Self::AlreadyExists { entity } => write!(f, "{} already exists", entity),
            Self::ForeignKeyViolation { entity } => {
                write!(f, "{} references a missing parent row", entity)
            }
            Self::QueryFailed(msg) => write!(f, "query failed: {}", msg),
            Self::IoError(e) => write!(f, "io error: {}", e),
            Self::ArtifactLimitExceeded {
                artifact_id,
                limit_bytes,
                kind,
            } => {
                write!(
                    f,
                    "artifact {} exceeds {} limit of {} bytes",
                    artifact_id, kind, limit_bytes
                )
            }
            Self::ArtifactCorrupt { artifact_id } => {
                write!(
                    f,
                    "artifact {} checksum mismatch — file corrupt",
                    artifact_id
                )
            }
            Self::ArtifactMissing { artifact_id } => {
                write!(
                    f,
                    "artifact {} pointer exists but backing file is missing",
                    artifact_id
                )
            }
            Self::PathUnsafe { segment } => {
                write!(f, "unsafe path segment: {:?}", segment)
            }
            Self::PrivacyNotGuaranteed => {
                write!(
                    f,
                    "platform privacy cannot be guaranteed for artifact storage"
                )
            }
        }
    }
}

impl std::error::Error for LogStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for LogStoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for LogStoreError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}
