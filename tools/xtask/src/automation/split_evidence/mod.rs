mod args;
mod boundary;
mod command;
mod encoding;
mod evidence;
mod reconcile;
use super::workload_oracle_evidence::serialization;
mod storage;
mod topology;
mod types;
mod verify;

pub(crate) use command::{USAGE, run};

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("{0}")]
    Contract(String),
    #[error("cannot read {label} snapshot {path}: {source}")]
    Read {
        label: &'static str,
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("invalid JSON in {label} snapshot {path}: {reason}")]
    Json {
        label: &'static str,
        path: std::path::PathBuf,
        reason: String,
    },
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests;
