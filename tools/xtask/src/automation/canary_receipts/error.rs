use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ErrorKind {
    Json,
    Io,
    InputLimit,
    Plan,
    UnplannedFamily,
    ReceiptIdentity,
    AttemptBounds,
    DuplicateReceipt,
    WorkerOutcome,
    ResultsDigest,
    Results,
    BatteryPreflight,
    Certification,
    WorkloadClass,
    RequiredLane,
    Multimodal,
    MissingReceipts,
    PackageIdentity,
    PackageArtifact,
}

#[derive(Debug, Serialize)]
pub(crate) struct Error {
    pub(crate) kind: ErrorKind,
    pub(crate) message: String,
}

impl Error {
    pub(super) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::new(ErrorKind::Json, error.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, error.to_string())
    }
}
