//! Artifact metadata for logged request/response traces.

use serde::{Deserialize, Serialize};

use super::identifiers::ArtifactId;

/// Kind of artifact being tracked.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// The original request payload (redacted).
    Request,
    /// The response payload (redacted).
    Response,
    /// A trace or diagnostic artifact.
    Trace,
    /// An individual chunk of a streaming response.
    Chunk,
    /// An error snapshot.
    Error,
}

/// Metadata for an artifact without storing the raw content inline.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub artifact_id: ArtifactId,
    pub kind: ArtifactKind,

    /// Size of the original artifact in bytes (before truncation/redaction).
    pub bytes: u64,

    /// Content checksum for integrity verification.
    pub checksum: String,

    /// Schema version of this metadata record.
    pub version: u16,

    /// Whether sensitive content was redacted before storage.
    #[serde(default)]
    pub redacted: bool,

    /// Whether the artifact was truncated to fit size limits.
    #[serde(default)]
    pub truncated: bool,

    /// Whether the original artifact is missing from storage.
    #[serde(default)]
    pub missing: bool,

    /// Whether stored data failed integrity checks.
    #[serde(default)]
    pub corrupt: bool,
}

impl ArtifactMetadata {
    /// Create a new metadata record for an artifact of the given kind and size.
    pub fn new(kind: ArtifactKind, bytes: u64) -> Self {
        Self {
            artifact_id: ArtifactId::new(),
            kind,
            bytes,
            checksum: String::new(),
            version: 1,
            redacted: false,
            truncated: false,
            missing: false,
            corrupt: false,
        }
    }

    /// Mark this artifact as redacted. Returns `&mut self` for chaining.
    pub fn set_redacted(&mut self) -> &mut Self {
        self.redacted = true;
        self
    }

    /// Mark this artifact as truncated. Returns `&mut self` for chaining.
    pub fn set_truncated(&mut self) -> &mut Self {
        self.truncated = true;
        self
    }

    /// Set the checksum value. Returns `&mut self` for chaining.
    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = checksum;
        self
    }

    /// Check if this artifact is considered healthy (not missing and not corrupt).
    pub fn is_healthy(&self) -> bool {
        !self.missing && !self.corrupt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_artifact_metadata() {
        let meta = ArtifactMetadata::new(ArtifactKind::Request, 4096);
        assert_eq!(meta.kind, ArtifactKind::Request);
        assert!(!meta.redacted);
        assert!(!meta.truncated);
        assert!(meta.is_healthy());
    }

    #[test]
    fn test_set_redacted() {
        let mut meta = ArtifactMetadata::new(ArtifactKind::Response, 2048);
        meta.set_redacted();
        assert!(meta.redacted);
    }

    #[test]
    fn test_artifact_serde_roundtrip() {
        let meta =
            ArtifactMetadata::new(ArtifactKind::Trace, 1024).with_checksum("sha256-abc".into());

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"kind\":\"trace\""));

        let parsed: ArtifactMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, ArtifactKind::Trace);
    }

    #[test]
    fn test_defaults_for_boolean_flags() {
        // Deserialize from minimal JSON — boolean defaults should be false.
        let json = r#"{"artifact_id":"00000000-0000-4000-a000-000000000001","kind":"chunk","bytes":128,"checksum":"","version":1}"#;
        let meta: ArtifactMetadata = serde_json::from_str(json).unwrap();

        assert!(!meta.redacted);
        assert!(!meta.truncated);
        assert!(!meta.missing);
        assert!(!meta.corrupt);
    }

    #[test]
    fn test_missing_or_corrupt_unhealthy() {
        let mut meta = ArtifactMetadata::new(ArtifactKind::Error, 256);
        meta.missing = true;
        assert!(!meta.is_healthy());

        meta.missing = false;
        meta.corrupt = true;
        assert!(!meta.is_healthy());
    }

    #[test]
    fn test_all_kinds_serialize() {
        for kind in [
            ArtifactKind::Request,
            ArtifactKind::Response,
            ArtifactKind::Trace,
            ArtifactKind::Chunk,
            ArtifactKind::Error,
        ] {
            let meta = ArtifactMetadata::new(kind, 1);
            let json = serde_json::to_string(&meta).unwrap();
            assert!(json.contains("\"kind\":"));

            // Round-trip parses back.
            let parsed: ArtifactMetadata = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.kind, kind);
        }
    }

    #[test]
    fn test_with_checksum() {
        let meta = ArtifactMetadata::new(ArtifactKind::Request, 512)
            .with_checksum("sha256-deadbeef".into());

        assert_eq!(meta.checksum, "sha256-deadbeef");
    }

    #[test]
    fn test_set_truncated() {
        let mut meta = ArtifactMetadata::new(ArtifactKind::Response, 10_000);
        meta.set_truncated();
        assert!(meta.truncated);
    }

    #[test]
    fn test_artifact_clone() {
        let a = ArtifactMetadata::new(ArtifactKind::Request, 42).with_checksum("abc".into());

        let b = a.clone();
        assert_eq!(b.bytes, 42);
        assert_eq!(b.checksum, "abc");
    }
}
