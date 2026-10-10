use crate::command::DynResult;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub(super) fn select(root: &Path, backend: &str) -> DynResult<PathBuf> {
    #[derive(Deserialize)]
    struct Manifest {
        runtime: Runtime,
    }
    #[derive(Deserialize)]
    struct Runtime {
        backend: Backend,
    }
    #[derive(Deserialize)]
    struct Backend {
        kind: String,
    }
    let expected = match backend {
        "cuda-blackwell" => "cuda",
        "hip" => "rocm",
        other => other,
    };
    let mut matches = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path().join("manifest.json");
        let bytes = std::fs::read(path)?;
        let Ok(manifest) = serde_json::from_slice::<Manifest>(&bytes) else {
            continue;
        };
        if manifest.runtime.backend.kind == expected {
            matches.push(entry.path());
        }
    }
    if matches.len() != 1 {
        return Err("replay arm must identify exactly one matching runtime".into());
    }
    matches.pop().ok_or_else(|| "missing replay runtime".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_candidate_manifest_is_an_io_failure_not_a_skipped_runtime() {
        let state = tempfile::tempdir().unwrap();
        let valid = state.path().join("valid");
        std::fs::create_dir(&valid).unwrap();
        std::fs::write(
            valid.join("manifest.json"),
            br#"{"runtime":{"backend":{"kind":"cpu"}}}"#,
        )
        .unwrap();
        std::fs::create_dir(state.path().join("unreadable")).unwrap();
        assert!(select(state.path(), "cpu").is_err());
    }
}
