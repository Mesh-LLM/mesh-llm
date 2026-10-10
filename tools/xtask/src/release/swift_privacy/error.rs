use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PrivacyError {
    #[error("cannot read privacy manifest {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid privacy plist: {0}")]
    Plist(#[from] plist::Error),
    #[error("invalid privacy plist shape: {0}")]
    Shape(&'static str),
    #[error("NSPrivacyTracking must be false")]
    Tracking,
    #[error("NSPrivacyCollectedDataTypes must stay empty unless SDK data collection changes")]
    CollectedData,
    #[error("NSPrivacyTrackingDomains must stay empty when tracking is false")]
    TrackingDomains,
    #[error("invalid required-reason API entry at index {0}")]
    InvalidEntry(usize),
    #[error("duplicate required-reason API entry: {0}")]
    DuplicateCategory(String),
    #[error("{category} reasons mismatch: {actual:?} != {expected:?}")]
    Reasons {
        category: &'static str,
        actual: Vec<String>,
        expected: Vec<&'static str>,
    },
    #[error("unexpected required-reason API categories: {}", .0.join(", "))]
    UnexpectedCategories(Vec<String>),
    #[error("XCFramework does not exist: {0}")]
    MissingFramework(PathBuf),
    #[error("PrivacyInfo.xcprivacy not embedded in {0}")]
    NotEmbedded(PathBuf),
    #[error("embedded privacy manifest differs from template: {0}")]
    EmbeddedDifference(PathBuf),
    #[error("cannot discover embedded privacy manifests in {path}: {source}")]
    Discovery {
        path: PathBuf,
        source: std::io::Error,
    },
}
