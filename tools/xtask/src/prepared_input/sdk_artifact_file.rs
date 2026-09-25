//! Manifest-relative file references inside a native SDK artifact: forward
//! slashes only, no absolute, drive-qualified or `..` paths, and the resolved
//! target (symlinks followed) must remain a regular file inside the artifact.

use super::{Checked, Rejected};
use crate::ci_plan::document::Json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Resolves `raw` inside `root`, the canonical artifact directory.
pub(super) fn artifact_file(root: &Path, label: &str, raw: Option<&Json>) -> Checked<PathBuf> {
    let raw = match raw.and_then(Json::as_str) {
        Some(text) if !text.is_empty() && !text.contains('\0') => text,
        _ => return Err(Rejected(format!("{label} path must be a non-empty string"))),
    };
    if raw.contains('\\') {
        return Err(Rejected(format!(
            "{label} path must use forward slashes inside the artifact: {raw}"
        )));
    }
    let parts: Vec<&str> = raw
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if raw.starts_with('/') || drive_prefixed(raw) || parts.contains(&"..") {
        return Err(Rejected(format!(
            "{label} must be a relative path inside the artifact: {raw}"
        )));
    }
    let candidate = parts
        .iter()
        .fold(root.to_path_buf(), |path, part| path.join(part));
    let missing = || Rejected(format!("missing {label}: {}", candidate.display()));
    let resolved = candidate.canonicalize().map_err(|_| missing())?;
    if !resolved.starts_with(root) {
        return Err(Rejected(format!(
            "{label} path resolves outside the artifact: {raw}"
        )));
    }
    if !resolved.is_file() {
        return Err(missing());
    }
    Ok(resolved)
}

fn drive_prefixed(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

pub(super) fn sha256_file(path: &Path) -> Checked<String> {
    let bytes =
        std::fs::read(path).map_err(|error| Rejected(super::python_io::os_error(path, &error)))?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}
