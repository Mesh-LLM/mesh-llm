use anyhow::{Context, Result, ensure};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

pub struct DownloadedModel {
    pub primary_path: PathBuf,
    pub load_path: PathBuf,
    pub projector_path: Option<PathBuf>,
    pub report: serde_json::Value,
    pub(super) transfer_stats: Option<super::transfer::DownloadTransferStats>,
}

/// Mesh and Skippy use the same Hugging Face cache resolution policy.
pub fn model_cache_dir() -> PathBuf {
    skippy_model_hf::huggingface_hub_cache_dir()
}

struct DownloadProgress {
    last_percent: Mutex<Option<u64>>,
    tracker: Arc<Mutex<super::transfer::DownloadTransferTracker>>,
    context: super::output::ModelCommandContext,
    enabled: bool,
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(event) = event
            && let Ok(mut tracker) = self.tracker.lock()
        {
            tracker.apply_download_event(event);
        }
        let (current, total) = match event {
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => (*bytes_completed, *total_bytes),
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                let Some(file) = files.last() else { return };
                (file.bytes_completed, file.total_bytes)
            }
            _ => return,
        };
        if !self.enabled || total == 0 {
            return;
        }
        let percent = current.min(total).saturating_mul(100) / total;
        let Ok(mut last) = self.last_percent.lock() else {
            return;
        };
        if *last == Some(percent) {
            return;
        }
        *last = Some(percent);
        let _ = super::output::sync_scope(self.context.clone(), || {
            super::output::progress("Model download", current, total)
        });
    }
}

pub async fn download_model(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
) -> Result<DownloadedModel> {
    download_model_with_policy(cache, model_ref, sha256, size_bytes, true, true).await
}

pub(super) async fn download_model_direct(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
    progress: bool,
) -> Result<DownloadedModel> {
    download_model_with_policy(cache, model_ref, sha256, size_bytes, false, progress).await
}

async fn download_model_with_policy(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
    use_catalog: bool,
    progress: bool,
) -> Result<DownloadedModel> {
    let mut downloaded =
        download_primary_model(cache, model_ref, sha256, size_bytes, use_catalog, progress).await?;
    let artifact = &downloaded.report["artifact"];
    let catalog = skippy_model_hf::remote_catalog::find_model_exact(model_ref).or_else(|| {
        let repo = artifact["source_repo"].as_str()?;
        let revision = artifact["source_revision"].as_str();
        let file = artifact["primary_file"].as_str()?;
        skippy_model_hf::remote_catalog::matching_model_for_huggingface(repo, revision, file)
    });
    if let Some(asset) = catalog.and_then(|model| model.mmproj) {
        let revision = asset
            .revision
            .as_deref()
            .map(|revision| format!("@{revision}"))
            .unwrap_or_default();
        let projector_ref = format!("{}{}/{}", asset.repo, revision, asset.source_file);
        let projector =
            download_primary_model(cache, &projector_ref, None, None, false, progress).await?;
        downloaded.report["projector_path"] = serde_json::json!(projector.primary_path);
        downloaded.projector_path = Some(projector.primary_path);
        downloaded.transfer_stats = super::transfer::DownloadTransferStats::combine(
            [downloaded.transfer_stats, projector.transfer_stats]
                .into_iter()
                .flatten()
                .collect(),
        );
    }
    Ok(downloaded)
}

async fn download_primary_model(
    cache: &Path,
    model_ref: &str,
    sha256: Option<&str>,
    size_bytes: Option<u64>,
    use_catalog: bool,
    progress: bool,
) -> Result<DownloadedModel> {
    validate_digest(sha256)?;
    use std::io::Write;
    if progress {
        writeln!(super::output::console_err(), "📦 Resolving {model_ref}")?;
    }
    let _cache_lock = skippy_model_hf::local_cache::lock_cache(cache)?;
    let repository = skippy_model_hf::HfModelRepository::builder()
        .cache_dir(cache)
        .retry_max_attempts(6)
        .retry_base_delay(Duration::from_millis(500))
        .build()?;
    let resolved_ref = if use_catalog {
        skippy_model_hf::remote_catalog::find_model_exact(model_ref)
            .map(|model| model.exact_ref())
            .or_else(|| {
                (!model_ref.contains('/'))
                    .then(|| skippy_model_hf::remote_catalog::resolve_model_download(model_ref))
                    .flatten()
                    .map(|model| {
                        let revision = model
                            .revision
                            .as_deref()
                            .map(|revision| format!("@{revision}"))
                            .unwrap_or_default();
                        format!("{}{}/{}", model.repo, revision, model.file)
                    })
            })
            .unwrap_or_else(|| model_ref.to_string())
    } else {
        model_ref.to_string()
    };
    let artifact =
        skippy_model_artifact::resolve_model_artifact_ref(&resolved_ref, &repository).await?;
    let cached_before = cache
        .join(format!(
            "models--{}",
            artifact.source_repo.replace('/', "--")
        ))
        .join("snapshots")
        .join(&artifact.source_revision)
        .join(&artifact.primary_file)
        .exists();
    let tracker = Arc::new(Mutex::new(
        super::transfer::DownloadTransferTracker::default(),
    ));
    let progress = hf_hub::progress::Progress::new(DownloadProgress {
        tracker: tracker.clone(),
        context: super::output::context(),
        enabled: progress,
        last_percent: Mutex::new(None),
    });
    let paths = if artifact.format == skippy_model_artifact::ModelFormat::Safetensors {
        repository
            .download_checkpoint_with_progress(&artifact, Some(progress))
            .await?
            .into_iter()
            .map(|downloaded| (downloaded.file, downloaded.path))
            .collect::<Vec<_>>()
    } else {
        let downloaded_paths = repository
            .download_artifact_files_with_progress(&artifact, Some(progress))
            .await?;
        ensure!(
            downloaded_paths.len() == artifact.files.len(),
            "downloaded artifact file count mismatch"
        );
        downloaded_paths
            .into_iter()
            .zip(artifact.files.iter().cloned())
            .map(|(path, file)| (file, path))
            .collect::<Vec<_>>()
    };
    ensure!(!paths.is_empty(), "downloaded artifact file list is empty");
    let mut files = Vec::with_capacity(paths.len());
    let mut managed_paths = Vec::with_capacity(paths.len());
    let mut primary_path = None;
    for (file, path) in paths {
        let primary = file.path == artifact.primary_file;
        let expected_size = if primary {
            size_bytes.or(file.size_bytes)
        } else {
            file.size_bytes
        };
        let expected_sha = if primary {
            sha256.or(file.sha256.as_deref())
        } else {
            file.sha256.as_deref()
        };
        files.push(verify_file(&path, expected_size, expected_sha)?);
        managed_paths.push(path.clone());
        if primary {
            primary_path = Some(path);
        }
    }
    let primary_path = primary_path.context("download did not include the primary model file")?;
    let load_path = if artifact.format == skippy_model_artifact::ModelFormat::Safetensors {
        primary_path
            .parent()
            .context("SafeTensors checkpoint has no parent directory")?
            .to_path_buf()
    } else {
        primary_path.clone()
    };
    if let Err(error) = skippy_model_hf::store::usage::track_managed_model_usage(
        &primary_path,
        &managed_paths,
        &artifact.model_id,
        Some(&artifact.canonical_ref),
        "huggingface",
    ) {
        crate::console::status(&format!("⚠ Could not record model usage: {error:#}"))?;
    }
    let report = serde_json::json!({
        "cache_dir": cache, "artifact": artifact, "primary_path": primary_path,
        "load_path": load_path, "files": files
    });
    let transfer_stats = std::mem::take(
        &mut *tracker
            .lock()
            .map_err(|_| anyhow::anyhow!("download transfer tracker poisoned"))?,
    )
    .finish_with_file_fallback(cached_before, &primary_path);
    Ok(DownloadedModel {
        transfer_stats,
        primary_path,
        load_path,
        projector_path: None,
        report,
    })
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
