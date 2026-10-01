mod command;
pub(crate) mod schema;
mod source;
mod validation;

pub(crate) use command::{USAGE, run};

use schema::{Artifact, Execution};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("qualification receipt rejected: {0}")]
    Invalid(&'static str),
    #[error("qualification input I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("qualification input JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("qualification digest mismatch: {0}")]
    DigestMismatch(std::path::PathBuf),
    #[error("qualification execution pending: {0}")]
    ExecutionPending(&'static str),
}

fn hex_identity(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn verify_artifact(artifact: &Artifact) -> Result<(), Error> {
    if !artifact.path.is_absolute() || !hex_identity(&artifact.sha256, 64) {
        return Err(Error::Invalid(
            "artifact requires absolute path and SHA-256",
        ));
    }
    let metadata = std::fs::symlink_metadata(&artifact.path)?;
    if !metadata.is_file() {
        return Err(Error::Invalid(
            "artifact must be a regular non-symlink file",
        ));
    }
    let actual = crate::product::digest::file_sha256(&artifact.path)
        .map_err(|failure| Error::Io(failure.error))?;
    if actual != artifact.sha256 {
        return Err(Error::DigestMismatch(artifact.path.clone()));
    }
    Ok(())
}

fn verify_execution(execution: &Execution) -> Result<(), Error> {
    if execution.argv.is_empty()
        || execution.argv.iter().any(String::is_empty)
        || execution.exit_code != 0
        || execution.case_count == 0
        || !execution.cleanup_complete
    {
        return Err(Error::Invalid(
            "required execution failed, skipped, empty or leaked",
        ));
    }
    verify_artifact(&execution.evidence)
}

pub(crate) fn load(path: &Path) -> Result<schema::Receipt, Error> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > 1_048_576 {
        return Err(Error::Invalid("receipt exceeds one MiB"));
    }
    Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
}

#[cfg(test)]
mod tests;
