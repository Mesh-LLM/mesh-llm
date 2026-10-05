//! Shared package acquisition. Callers supply storage, client policy and progress.
use super::progress::{ForwardProgress, NoPackageProgress, PackageProgress};
use super::{
    StagePackageRef, cache_resolution, is_metadata_only_package_inspection,
    manifest_artifact_bytes, resolve_local_package_files, verify_resolved_hf_package_files,
};
use crate::materialization::{safe_manifest_file_path, verify_package_v2_artifact};
use anyhow::{Context, Result};
use hf_hub::progress::Progress;
use skippy_package_format::{
    PackageManifest as PackageManifestV2,
    stage_admission::StageAdmissionDescriptor as PackageV2StageAdmissionDescriptor,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Explicit storage and transport policy shared by standalone and embedded callers.
/// The client factory is lazy: a complete cached snapshot never constructs a client.
#[derive(Clone)]
pub struct PackageAcquisition {
    pub hub_cache: PathBuf,
    pub integrity_cache: Option<PathBuf>,
    pub build_client: Arc<dyn Fn() -> Result<hf_hub::HFClientSync> + Send + Sync>,
    pub progress: Arc<dyn PackageProgress>,
}

impl PackageAcquisition {
    pub fn new(
        hub_cache: PathBuf,
        build_client: impl Fn() -> Result<hf_hub::HFClientSync> + Send + Sync + 'static,
    ) -> Self {
        Self {
            hub_cache,
            integrity_cache: None,
            build_client: Arc::new(build_client),
            progress: Arc::new(NoPackageProgress),
        }
    }
}

// A stage load can be superseded while a blocking HF transfer is still
// finishing. Serialise package downloads inside one process so the
// replacement load reuses the completed cache entry instead of racing the HF
// cache's per-blob file lock.
static LAYER_PACKAGE_DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

fn lock_layer_package_downloads() -> std::sync::MutexGuard<'static, ()> {
    LAYER_PACKAGE_DOWNLOAD_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn layer_package_progress_label(repo: &str, revision: &str) -> String {
    if revision == "main" {
        format!("layer package {repo}")
    } else {
        format!("layer package {repo}@{revision}")
    }
}

fn download_layer_package_file(
    model_api: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: &str,
    label: &str,
    file_name: &str,
    total_bytes: Option<u64>,
    observer: &dyn PackageProgress,
    completed_before: usize,
) -> Result<PathBuf> {
    let progress = observer.file(label, file_name, total_bytes, completed_before);
    progress.ensuring();
    let progress_handler: Option<Progress> =
        Some(Arc::new(ForwardProgress(progress.clone())).into());
    let path = model_api
        .download_file()
        .filename(file_name.to_string())
        .revision(revision.to_string())
        .maybe_progress(progress_handler)
        .send()
        .with_context(|| format!("download layer package file: {file_name}"))?;
    progress.ready(&path);
    Ok(path)
}

impl PackageAcquisition {
    pub fn resolve_hf_package_to_local(
        &self,
        package_ref: &str,
        layer_start: u32,
        layer_end: u32,
        include_embeddings: bool,
        include_output: bool,
    ) -> Result<String> {
        let parsed = StagePackageRef::parse(package_ref)?;
        let (repo, revision) = match &parsed {
            StagePackageRef::HuggingFacePackage { repo, revision } => (
                repo.clone(),
                revision.clone().unwrap_or_else(|| "main".to_string()),
            ),
            StagePackageRef::LocalPackage(path) => {
                return resolve_local_package_files(
                    path,
                    layer_start,
                    layer_end,
                    include_embeddings,
                    include_output,
                );
            }
            _ => return Ok(package_ref.to_string()),
        };

        // Try to resolve from the local HF cache first — avoids the HF SDK entirely,
        // which is critical on NFS (where flock fails) and inside async runtimes
        // (where the sync SDK wrapper panics with "Cannot start a runtime").
        let cache_dir = self.hub_cache.clone();
        let repo_folder = format!("models--{}", repo.replace('/', "--"));
        let revision_cache_path = safe_manifest_file_path(&revision)
            .with_context(|| format!("invalid HF revision for local cache lookup: {revision}"))?;
        let ref_path = cache_dir
            .join(&repo_folder)
            .join("refs")
            .join(&revision_cache_path);
        let direct_snapshot_dir = cache_dir
            .join(&repo_folder)
            .join("snapshots")
            .join(&revision_cache_path);
        if direct_snapshot_dir.join("model-package.json").is_file()
            && let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
                &direct_snapshot_dir,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
                self.integrity_cache.as_deref(),
            )?
        {
            return Ok(local_ref);
        }
        if let Ok(commit_hash) = fs::read_to_string(&ref_path) {
            let commit_hash = commit_hash.trim();
            let commit_hash_path = safe_manifest_file_path(commit_hash).with_context(|| {
                format!("invalid HF cache commit hash for local cache lookup: {commit_hash}")
            })?;
            let snapshot_dir = cache_dir
                .join(&repo_folder)
                .join("snapshots")
                .join(commit_hash_path);
            if snapshot_dir.join("model-package.json").is_file()
                && let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
                    &snapshot_dir,
                    layer_start,
                    layer_end,
                    include_embeddings,
                    include_output,
                    self.integrity_cache.as_deref(),
                )?
            {
                return Ok(local_ref);
            }
        }
        let acquisition = self.clone();
        let downloaded = skippy_model_hf::blocking::run_hf_sync(move || {
            acquisition.download_hf_package_to_local_sync(
                &repo,
                &revision,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            )
        })?;

        // Metadata-only probes (layer_start == layer_end == 0) download the
        // manifest and shared metadata but no layer files.  The downloaded
        // snapshot may be a skeleton whose hash must not propagate through
        // topology configs and stage loads.  Re-scan the local cache for a
        // snapshot that has at least one real layer artifact.
        //
        // Real stage loads (layer_start < layer_end) always download the
        // requested layer range, so the downloaded snapshot is guaranteed to
        // have the needed files — no fallback scan needed.
        let is_metadata_only = layer_start == 0 && layer_end == 0;
        if is_metadata_only {
            let downloaded_dir = std::path::Path::new(&downloaded);
            if downloaded_dir.join("model-package.json").is_file()
                && cache_resolution::resolve_cached_hf_package_snapshot(
                    downloaded_dir,
                    layer_start,
                    layer_end,
                    include_embeddings,
                    include_output,
                    self.integrity_cache.as_deref(),
                )?
                .is_none()
            {
                // Downloaded snapshot is a skeleton — find one with real layers.
                let cache_dir = self.hub_cache.clone();
                for snapshot_dir in
                    cache_resolution::cached_package_snapshots(&cache_dir, &repo_folder)?
                {
                    if snapshot_dir.as_path() == downloaded_dir {
                        continue;
                    }
                    if let Ok(Some(better)) = cache_resolution::resolve_cached_hf_package_snapshot(
                        &snapshot_dir,
                        layer_start,
                        layer_end,
                        include_embeddings,
                        include_output,
                        self.integrity_cache.as_deref(),
                    ) {
                        tracing::debug!(
                            downloaded = %downloaded,
                            better = %better,
                            "post-download: preferring cached snapshot with layer artifacts over skeleton"
                        );
                        return Ok(better);
                    }
                }
            }
        }

        Ok(downloaded)
    }

    fn download_hf_package_to_local_sync(
        &self,
        repo: &str,
        revision: &str,
        layer_start: u32,
        layer_end: u32,
        include_embeddings: bool,
        include_output: bool,
    ) -> Result<String> {
        let _download_guard = lock_layer_package_downloads();
        let api = (self.build_client)()?;
        let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
        let model_api = api.model(owner, name);
        let progress_label = layer_package_progress_label(repo, revision);

        // Download manifest first
        let manifest_path = download_layer_package_file(
            &model_api,
            revision,
            &progress_label,
            "model-package.json",
            None,
            self.progress.as_ref(),
            0,
        )
        .context("download layer package manifest")?;

        let package_dir = manifest_path
            .parent()
            .context("manifest has no parent directory")?
            .to_path_buf();

        let needed_files = package_download_files(
            &manifest_path,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        )?;

        let missing_files: Vec<_> = needed_files
            .iter()
            .filter(|(file, _)| !package_dir.join(file).is_file())
            .collect();
        let package_scope = self
            .progress
            .batch(&progress_label, missing_files.len() + 1);

        // Download each needed file
        for (index, (file, total_bytes)) in missing_files.into_iter().enumerate() {
            let file_name = file.to_string_lossy().to_string();
            download_layer_package_file(
                &model_api,
                revision,
                &progress_label,
                &file_name,
                *total_bytes,
                package_scope.as_ref(),
                index + 1,
            )?;
        }

        verify_resolved_hf_package_files(
            &package_dir,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
            self.integrity_cache.as_deref(),
        )
    }

    pub fn resolve_package_v2_stage_to_local(
        &self,
        package_ref: &str,
        admission: &PackageV2StageAdmissionDescriptor,
    ) -> Result<(String, Vec<PathBuf>, Option<PathBuf>)> {
        let local_ref = self.resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
        let package_dir = PathBuf::from(&local_ref);
        let manifest_path = package_dir.join("model-package.json");
        let manifest_bytes = fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
        let manifest: PackageManifestV2 =
            serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
        let manifest =
            skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
                .context("resolve package-v2 metadata carrier")?;
        let resolved = manifest
            .resolve_stage_admission(admission)
            .context("resolve exact package-v2 stage admission")?;
        let required = resolved
            .required_artifacts
            .iter()
            .map(|artifact| (*artifact).clone())
            .collect::<Vec<_>>();

        if let StagePackageRef::HuggingFacePackage { repo, revision } =
            StagePackageRef::parse(package_ref)?
        {
            let _download_guard = lock_layer_package_downloads();
            let revision = revision.unwrap_or_else(|| "main".to_string());
            let missing = required
                .iter()
                .filter(|artifact| verify_package_v2_artifact(&package_dir, artifact).is_err())
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                let package_dir_for_download = package_dir.clone();
                let required_for_download = required.clone();
                let acquisition = self.clone();
                skippy_model_hf::blocking::run_hf_sync(move || {
                    let api = (acquisition.build_client)()?;
                    let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
                    let model_api = api.model(owner, name);
                    let label = layer_package_progress_label(&repo, &revision);
                    let scope = acquisition.progress.batch(&label, missing.len());
                    for (index, artifact) in missing.iter().enumerate() {
                        download_layer_package_file(
                            &model_api,
                            &revision,
                            &label,
                            &artifact.path,
                            Some(artifact.byte_size),
                            scope.as_ref(),
                            index,
                        )?;
                    }
                    for artifact in &required_for_download {
                        verify_package_v2_artifact(&package_dir_for_download, artifact)?;
                    }
                    Ok(())
                })?;
            }
        }
        crate::materialization::resolve_local_package_stage(&package_dir, admission)
    }

    /// Resolve the complete tensor closure for single-node package-v2 serving.
    ///
    /// Split stages arrive with an exact coordinator-produced admission descriptor.
    /// A local single-node load owns every tensor, so its admission descriptor is
    /// the manifest's complete, ordered tensor catalog.
    pub fn resolve_package_v2_full_model_to_local(
        &self,
        package_ref: &str,
    ) -> Result<(Vec<PathBuf>, Option<PathBuf>)> {
        let (_, model_parts, projector) =
            self.resolve_package_v2_full_model_with_root(package_ref)?;
        Ok((model_parts, projector))
    }

    /// Download every artifact needed to use a package-v2 model and return its
    /// local package root.
    pub fn download_package_v2_to_local(&self, package_ref: &str) -> Result<PathBuf> {
        let (package_dir, _, _) = self.resolve_package_v2_full_model_with_root(package_ref)?;
        Ok(package_dir)
    }

    fn resolve_package_v2_full_model_with_root(
        &self,
        package_ref: &str,
    ) -> Result<(PathBuf, Vec<PathBuf>, Option<PathBuf>)> {
        let local_ref = self.resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
        let manifest_path = Path::new(&local_ref).join("model-package.json");
        let manifest: PackageManifestV2 =
            serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                format!("read package-v2 manifest {}", manifest_path.display())
            })?)
            .context("parse package-v2 manifest")?;
        let manifest = skippy_model::package_carrier::resolve_package_carrier_from_dir(
            manifest,
            Path::new(&local_ref),
        )
        .context("resolve package-v2 metadata carrier")?;
        let mut sidecars = manifest.sidecars.clone();
        sidecars.sort();
        let admission = PackageV2StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids: manifest
                .tensor_catalog
                .entries
                .iter()
                .filter(|tensor| {
                    matches!(
                        tensor.storage,
                        skippy_package_format::TensorStorage::Owned { .. }
                    )
                })
                .map(|tensor| tensor.id.clone())
                .collect(),
            sidecars,
        };
        let (_, model_parts, projector) =
            self.resolve_package_v2_stage_to_local(package_ref, &admission)?;
        Ok((PathBuf::from(local_ref), model_parts, projector))
    }
}

fn package_download_files(
    manifest_path: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Vec<(PathBuf, Option<u64>)>> {
    // Read manifest to determine which files we need
    let manifest_contents = fs::read(manifest_path).context("read package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;

    let package_v2 = manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION));
    if package_v2 {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
    }

    if package_v2 {
        let mut needed_files = Vec::new();
        let manifest_v2: PackageManifestV2 =
            serde_json::from_slice(&manifest_contents).context("parse package-v2 manifest")?;
        manifest_v2
            .validate_root()
            .context("validate package-v2 manifest")?;
        let metadata_artifact = manifest_v2
            .artifact_catalog
            .entries
            .iter()
            .find(|artifact| artifact.id == manifest_v2.source_model.metadata_artifact_id)
            .context("package-v2 metadata artifact is absent")?;
        needed_files.push((
            safe_manifest_file_path(&metadata_artifact.path)?,
            Some(metadata_artifact.byte_size),
        ));
        return Ok(needed_files);
    }
    legacy_download_files(
        &manifest,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}

fn legacy_download_files(
    manifest: &serde_json::Value,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Vec<(PathBuf, Option<u64>)>> {
    let mut needed_files = Vec::new();
    // Legacy packages always need shared/metadata.gguf.
    let metadata_artifact = manifest
        .pointer("/shared/metadata")
        .context("manifest missing required /shared/metadata")?;
    let metadata_path = metadata_artifact
        .get("path")
        .and_then(|v| v.as_str())
        .context("manifest missing required /shared/metadata/path")?;
    needed_files.push((
        safe_manifest_file_path(metadata_path)?,
        manifest_artifact_bytes(metadata_artifact),
    ));
    if include_embeddings
        && let Some(artifact) = manifest.pointer("/shared/embeddings")
        && let Some(path) = artifact.get("path").and_then(|v| v.as_str())
    {
        needed_files.push((
            safe_manifest_file_path(path)?,
            manifest_artifact_bytes(artifact),
        ));
    }
    if include_output
        && let Some(artifact) = manifest.pointer("/shared/output")
        && let Some(path) = artifact.get("path").and_then(|v| v.as_str())
    {
        needed_files.push((
            safe_manifest_file_path(path)?,
            manifest_artifact_bytes(artifact),
        ));
    }

    // Layer files for assigned range — use explicit layer_index if present,
    // fall back to array position.
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = layer
                .get("layer_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64) as u32;
            if idx >= layer_start
                && idx < layer_end
                && let Some(path) = layer.get("path").and_then(|a| a.as_str())
            {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(layer),
                ));
            }
        }
    }
    if layer_start == 0
        && let Some(projectors) = manifest.get("projectors").and_then(|p| p.as_array())
    {
        for projector in projectors {
            if let Some(path) = projector.get("path").and_then(|value| value.as_str()) {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(projector),
                ));
            }
        }
    }

    Ok(needed_files)
}

#[cfg(test)]
mod tests;
