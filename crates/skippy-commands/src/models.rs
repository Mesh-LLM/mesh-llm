use anyhow::{Context, Result, ensure};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

/// Parsed `skippy models` action, decoupled from clap.
#[derive(Debug, Clone)]
pub enum ModelAction {
    /// Resolve a Hub revision and download its selected model files.
    Pull {
        /// Hub reference: org/repo@revision:filename-or-quantization.
        model_ref: String,
        /// Expected SHA-256 of the primary model file; checked on cache hits too.
        sha256: Option<String>,
        /// Expected byte count of the primary model file.
        size_bytes: Option<u64>,
    },
    /// Remove all cached revisions of one local model repository; never deletes from the Hub.
    Remove { repo: String, dry_run: bool },
    /// List local model repositories and snapshots without contacting the Hub.
    List,
}

fn validate_digest(digest: Option<&str>) -> Result<()> {
    if let Some(digest) = digest {
        ensure!(
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "--sha256 must be exactly 64 hexadecimal characters"
        );
    }
    Ok(())
}

fn verify_file(
    path: &Path,
    expected_size: Option<u64>,
    expected_sha: Option<&str>,
) -> Result<serde_json::Value> {
    validate_digest(expected_sha)?;
    let bytes = path
        .metadata()
        .with_context(|| format!("inspect downloaded file {}", path.display()))?
        .len();
    if let Some(expected) = expected_size {
        ensure!(
            bytes == expected,
            "downloaded file size mismatch for {}: expected {expected}, got {bytes}",
            path.display()
        );
    }
    let digest = skippy_api::package::file_sha256(path)?;
    if let Some(expected) = expected_sha {
        ensure!(
            digest.eq_ignore_ascii_case(expected),
            "downloaded file SHA-256 mismatch for {}",
            path.display()
        );
    }
    Ok(
        serde_json::json!({"path": path, "bytes": bytes, "sha256": digest, "expected_sha256_verified": expected_sha.is_some()}),
    )
}

pub async fn run(explicit_cache: Option<PathBuf>, command: ModelAction) -> Result<()> {
    let cache = skippy_config::paths::model_cache_dir(explicit_cache)?;
    match command {
        ModelAction::Pull {
            model_ref,
            sha256,
            size_bytes,
        } => {
            validate_digest(sha256.as_deref())?;
            let _cache_lock = model_hf::local_cache::lock_cache(&cache)?;
            let repository = model_hf::HfModelRepository::builder()
                .cache_dir(&cache)
                .retry_max_attempts(6)
                .retry_base_delay(Duration::from_millis(500))
                .build()?;
            let artifact =
                model_artifact::resolve_model_artifact_ref(&model_ref, &repository).await?;
            let paths = repository.download_artifact_files(&artifact).await?;
            ensure!(
                paths.len() == artifact.files.len(),
                "downloaded artifact file count mismatch"
            );
            let mut files = Vec::with_capacity(paths.len());
            let mut primary_path = None;
            for (file, path) in artifact.files.iter().zip(paths) {
                let primary = file.path == artifact.primary_file;
                let expected_size = if primary {
                    size_bytes.or(file.size_bytes)
                } else {
                    file.size_bytes
                };
                let expected_sha = if primary {
                    sha256.as_deref().or(file.sha256.as_deref())
                } else {
                    file.sha256.as_deref()
                };
                files.push(verify_file(&path, expected_size, expected_sha)?);
                if primary {
                    primary_path = Some(path);
                }
            }
            let primary_path =
                primary_path.context("download did not include the primary model file")?;
            crate::console::write_json(&serde_json::json!({
                "cache_dir": cache, "artifact": artifact, "primary_path": primary_path, "files": files
            }))
        }
        ModelAction::Remove { repo, dry_run } => crate::console::write_json(
            &model_hf::local_cache::remove_repository(&cache, &repo, dry_run)?,
        ),
        ModelAction::List => {
            // This operation scans only the explicit local root; it issues no Hub request.
            let _ = model_hf::configure_hf_tls_provider();
            let client = hf_hub::HFClient::builder().cache_dir(&cache).build()?;
            let scan = client.scan_cache().send().await?;
            let repos = scan.repos.iter().filter(|r| r.repo_type == "model").map(|r| {
                serde_json::json!({"repo": r.repo_id, "path": r.repo_path, "bytes": r.size_on_disk,
                    "revisions": r.revisions.iter().map(|v| serde_json::json!({
                        "revision": v.commit_hash, "path": v.snapshot_path, "refs": v.refs,
                        "files": v.files.iter().map(|f| serde_json::json!({"file":f.file_name,"path":f.file_path,"bytes":f.size_on_disk})).collect::<Vec<_>>()
                    })).collect::<Vec<_>>()})
            }).collect::<Vec<_>>();
            crate::console::write_json(
                &serde_json::json!({"cache_dir":cache,"repositories":repos,"warnings":scan.warnings}),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_file_is_rehashed_and_rejects_corruption() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        std::fs::write(&path, b"first").unwrap();
        let digest = skippy_api::package::file_sha256(&path).unwrap();
        assert!(verify_file(&path, Some(5), Some(&digest)).is_ok());
        std::fs::write(&path, b"other").unwrap();
        assert!(
            verify_file(&path, Some(5), Some(&digest))
                .unwrap_err()
                .to_string()
                .contains("SHA-256 mismatch")
        );
        assert!(
            verify_file(&path, Some(6), None)
                .unwrap_err()
                .to_string()
                .contains("size mismatch")
        );
    }

    #[test]
    fn invalid_digest_is_rejected_before_file_access() {
        assert!(
            verify_file(Path::new("/nonexistent/model.gguf"), None, Some("bad"))
                .unwrap_err()
                .to_string()
                .contains("64 hexadecimal")
        );
    }
}
