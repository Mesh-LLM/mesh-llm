//! SQLite persistence for mesh-llm canonical logging pipeline.

#[cfg(test)]
mod artifacts_tests;
#[cfg(test)]
mod capture_tests;
#[cfg(test)]
mod space_maintenance_tests;
#[cfg(test)]
mod tests;

mod artifact_privacy;
mod artifacts;
mod capture;
mod cursor;
mod error;
mod migrations;
mod repositories;
mod store;

// Re-export primary types at crate root.
#[doc(hidden)]
pub use artifact_privacy::ArtifactPrivacy;
pub use artifacts::{
    ArtifactContent, ArtifactFileStore, ArtifactRedactor, ArtifactStatus, ArtifactWriteReceipt,
};
pub use capture::{
    ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE, ArtifactCaptureDisabledReason,
    ArtifactCaptureHealthMarker, ArtifactCaptureOutcome, FailOpenArtifactCapture,
};
pub use cursor::{decode_cursor, encode_cursor};
pub use error::LogStoreError;
pub use repositories::CascadeArtifactPointer;
pub use store::{Clock, LogStore, SQLiteSpaceMaintenance, SystemClock as RealClock};
