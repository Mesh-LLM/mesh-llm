use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

mod cli;

const MANIFEST: &str = "skippy-convert-manifest.json";

pub(crate) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let Some(parsed) = cli::parse(args)? else {
        println!("{}", cli::USAGE);
        return Ok(());
    };
    validate_converted_artifact(&parsed.artifact_dir)?;
    println!(
        "converted artifact preflight passed: {}",
        parsed.artifact_dir.display()
    );
    Ok(())
}

#[derive(Debug, Error)]
pub(crate) enum ArtifactError {
    #[error("complete converted artifact not found: {path}{details}")]
    MissingArtifact { path: PathBuf, details: String },
    #[error("invalid expected_splits in {0}")]
    InvalidSplitCount(PathBuf),
    #[error("invalid output_basename in {0}")]
    InvalidBasename(PathBuf),
    #[error("converted artifact is incomplete: missing {0}")]
    MissingShards(String),
    #[error("cannot read converted artifact manifest {path}: {source}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse converted artifact manifest {path}: {source}")]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[cfg(test)]
pub(crate) fn converted_artifact_dir(
    work_dir: &Path,
    target_prefix: &Path,
    upload_only: bool,
) -> Result<PathBuf, ArtifactError> {
    let artifact_dir = work_dir.join("target").join(target_prefix);
    if upload_only {
        validate_converted_artifact(&artifact_dir)?;
    }
    Ok(artifact_dir)
}

pub(crate) fn validate_converted_artifact(artifact_dir: &Path) -> Result<(), ArtifactError> {
    required_files(artifact_dir)?;
    let manifest_path = artifact_dir.join(MANIFEST);
    let bytes = fs::read(&manifest_path).map_err(|source| ArtifactError::ReadManifest {
        path: manifest_path.clone(),
        source,
    })?;
    let manifest: Value =
        serde_json::from_slice(&bytes).map_err(|source| ArtifactError::ParseManifest {
            path: manifest_path.clone(),
            source,
        })?;
    validate_converted_artifact_manifest(artifact_dir, &manifest)
}

fn required_files(artifact_dir: &Path) -> Result<(), ArtifactError> {
    let missing = ["README.md", MANIFEST]
        .into_iter()
        .filter(|name| !artifact_dir.join(name).is_file())
        .collect::<Vec<_>>();
    if !artifact_dir.is_dir() || !missing.is_empty() {
        let details = if missing.is_empty() {
            String::new()
        } else {
            format!("; missing {}", missing.join(", "))
        };
        return Err(ArtifactError::MissingArtifact {
            path: artifact_dir.to_path_buf(),
            details,
        });
    }

    Ok(())
}

/// Reuse preflight policy with the caller's already bounded and admitted manifest value.
/// This function does not reread or admit manifest bytes; that remains the caller's custody.
pub(crate) fn validate_converted_artifact_manifest(
    artifact_dir: &Path,
    manifest: &Value,
) -> Result<(), ArtifactError> {
    required_files(artifact_dir)?;
    let manifest_path = artifact_dir.join(MANIFEST);
    let expected_splits = match manifest.get("expected_splits") {
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count > 0)
            .ok_or_else(|| ArtifactError::InvalidSplitCount(manifest_path.clone()))?,
        _ => return Err(ArtifactError::InvalidSplitCount(manifest_path)),
    };
    let basename = manifest
        .get("output_basename")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ArtifactError::InvalidBasename(manifest_path.clone()))?;

    let missing_shards = (1..=expected_splits)
        .map(|index| {
            if expected_splits == 1 {
                format!("{basename}.gguf")
            } else {
                format!("{basename}-{index:05}-of-{expected_splits:05}.gguf")
            }
        })
        .filter(|name| !artifact_dir.join(name).is_file())
        .collect::<Vec<_>>();
    if !missing_shards.is_empty() {
        return Err(ArtifactError::MissingShards(missing_shards.join(", ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{converted_artifact_dir, run};
    use std::path::Path;

    #[test]
    fn run_rejects_missing_artifact_directory_argument() {
        let result = run(&[]);

        assert!(result.is_err());
    }

    #[test]
    fn non_upload_artifact_selection_keeps_the_target_path() {
        let temp = tempfile::TempDir::new().unwrap();

        let selected = converted_artifact_dir(temp.path(), Path::new("custom"), false).unwrap();

        assert_eq!(selected, temp.path().join("target/custom"));
    }

    #[test]
    fn parsed_preflight_reuses_admitted_manifest_without_reopening_replaced_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::write(root.join("README.md"), b"card").unwrap();
        std::fs::write(root.join("model.gguf"), b"GGUFfixture").unwrap();
        let admitted = serde_json::json!({"expected_splits":1,"output_basename":"model"});
        std::fs::write(root.join(super::MANIFEST), b"malformed replacement bytes").unwrap();
        super::validate_converted_artifact_manifest(root, &admitted).unwrap();
        assert!(super::validate_converted_artifact(root).is_err());
        let missing = serde_json::json!({"expected_splits":2,"output_basename":"model"});
        assert!(super::validate_converted_artifact_manifest(root, &missing).is_err());
        std::fs::write(
            root.join(super::MANIFEST),
            serde_json::to_vec(&admitted).unwrap(),
        )
        .unwrap();
        super::validate_converted_artifact(root).unwrap();
        temp.close().unwrap();
    }
}
