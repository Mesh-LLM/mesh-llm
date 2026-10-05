//! Package selection, acquisition, artifact integrity and cached snapshot resolution.
//! Cache locations and acquisition policies are supplied by the caller.
use crate::materialization::{safe_manifest_file_path, verify_package_v2_artifact};
use anyhow::{Context, Result, bail};
use skippy_package_format::PackageManifest as PackageManifestV2;
use skippy_runtime::package::{self, PackageIntegrityOptions, PackageStageRequest};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub mod cache_resolution;
pub mod progress;
pub mod remote;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StagePackageRef {
    LocalPackage(PathBuf),
    HuggingFacePackage {
        repo: String,
        revision: Option<String>,
    },
    SyntheticDirectGguf(PathBuf),
}

impl StagePackageRef {
    pub fn parse(value: &str) -> Result<Self> {
        if let Some(rest) = value.strip_prefix("hf://") {
            let (repo, revision) = if let Some((repo, revision)) = rest.split_once('@') {
                (repo, Some(revision.to_string()))
            } else if let Some(index) = rest.rfind(':') {
                (&rest[..index], Some(rest[index + 1..].to_string()))
            } else {
                (rest, None)
            };
            if repo.split('/').count() != 2 || repo.contains(':') || repo.contains('@') {
                bail!("HF package repo id must look like namespace/repo");
            }
            return Ok(Self::HuggingFacePackage {
                repo: repo.to_string(),
                revision,
            });
        }

        let path = PathBuf::from(value);
        if path.join("model-package.json").is_file() {
            return Ok(Self::LocalPackage(path));
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            return Ok(Self::SyntheticDirectGguf(path));
        }

        bail!("not a skippy package ref: {value}");
    }

    pub fn is_distributable_package(&self) -> bool {
        matches!(
            self,
            Self::LocalPackage(_) | Self::HuggingFacePackage { .. }
        )
    }

    pub fn as_package_ref(&self) -> Option<String> {
        match self {
            Self::LocalPackage(path) => Some(path.to_string_lossy().to_string()),
            Self::HuggingFacePackage { repo, revision } => Some(match revision {
                Some(revision) => format!("hf://{repo}@{revision}"),
                None => format!("hf://{repo}"),
            }),
            Self::SyntheticDirectGguf(_) => None,
        }
    }
}

pub fn is_layer_package_ref(value: &str) -> bool {
    StagePackageRef::parse(value).is_ok_and(|package_ref| package_ref.is_distributable_package())
}

/// Resolve a layer package from the local HF cache without touching the HF SDK.
/// Verifies that needed files exist locally; returns the snapshot dir path.
pub fn resolve_local_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read local package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse local package manifest")?;

    if manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION))
    {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
        verify_package_v2_metadata(package_dir, &manifest_contents)?;
        return Ok(package_dir.to_string_lossy().to_string());
    }

    // Verify shared/metadata.gguf exists
    let metadata_path = manifest
        .pointer("/shared/metadata/path")
        .and_then(|v| v.as_str())
        .context("manifest missing /shared/metadata/path")?;
    let metadata_path = safe_manifest_file_path(metadata_path)?;
    anyhow::ensure!(
        package_dir.join(&metadata_path).is_file(),
        "missing shared metadata: {}",
        metadata_path.display()
    );
    if include_embeddings
        && let Some(path) = manifest
            .pointer("/shared/embeddings/path")
            .and_then(|v| v.as_str())
    {
        let path = safe_manifest_file_path(path)?;
        anyhow::ensure!(
            package_dir.join(&path).is_file(),
            "missing shared embeddings: {}",
            path.display()
        );
    }
    if include_output
        && let Some(path) = manifest
            .pointer("/shared/output/path")
            .and_then(|v| v.as_str())
    {
        let path = safe_manifest_file_path(path)?;
        anyhow::ensure!(
            package_dir.join(&path).is_file(),
            "missing shared output: {}",
            path.display()
        );
    }
    // Verify needed layer files exist
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
                let path = safe_manifest_file_path(path)?;
                anyhow::ensure!(
                    package_dir.join(&path).is_file(),
                    "missing layer file: {}",
                    path.display()
                );
            }
        }
    }
    Ok(package_dir.to_string_lossy().to_string())
}

pub fn is_metadata_only_package_inspection(
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> bool {
    layer_start == layer_end && !include_embeddings && !include_output
}

pub fn verify_resolved_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
    integrity_cache: Option<&Path>,
) -> Result<String> {
    let manifest_contents = fs::read(package_dir.join("model-package.json"))
        .context("read resolved package manifest")?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&manifest_contents)
        .context("parse resolved package manifest")?
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if schema_version == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION)) {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
        verify_package_v2_metadata(package_dir, &manifest_contents)?;
        return Ok(package_dir.to_string_lossy().to_string());
    }
    let local_ref = resolve_local_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )?;
    let metadata_only = is_metadata_only_package_inspection(
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    );
    let options = if metadata_only {
        // Metadata-only probes hash only the small shared metadata artifact.
        // Avoid the cross-run integrity cache here so a same-size metadata
        // rewrite cannot be hidden by coarse filesystem timestamp resolution.
        PackageIntegrityOptions::verify_without_cache()
    } else {
        match integrity_cache {
            Some(path) => PackageIntegrityOptions::verify_with_cache(path),
            None => PackageIntegrityOptions::verify_without_cache(),
        }
    };
    let report = if metadata_only {
        package::verify_layer_package_metadata_integrity(&local_ref, &options)
    } else {
        let request = PackageStageRequest {
            model_id: "hf-layer-package".to_string(),
            topology_id: "hf-layer-package-resolver".to_string(),
            package_ref: local_ref.clone(),
            stage_id: format!("layers-{layer_start}-{layer_end}"),
            layer_start,
            layer_end,
            source_stage: include_embeddings,
            terminal_stage: include_output,
        };
        package::verify_layer_package_integrity(&request, &options)
    }
    .map_err(|error| anyhow::anyhow!("verify resolved HF layer package artifacts: {error:#}"))?;
    tracing::debug!(
        artifacts = report.artifacts,
        verified_artifacts = report.verified_artifacts,
        cached_artifacts = report.cached_artifacts,
        manifest_sha256 = %report.manifest_sha256,
        metadata_only,
        "verified resolved HF layer package artifacts"
    );
    Ok(local_ref)
}

fn missing_cached_package_artifact(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.starts_with("missing shared metadata:")
        || message.starts_with("missing shared embeddings:")
        || message.starts_with("missing shared output:")
        || message.starts_with("missing layer file:")
}

pub fn verify_cached_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
    integrity_cache: Option<&Path>,
) -> Result<Option<String>> {
    match verify_resolved_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
        integrity_cache,
    ) {
        Ok(local_ref) => Ok(Some(local_ref)),
        Err(error) if missing_cached_package_artifact(&error) => {
            tracing::debug!(
                package_dir = %package_dir.display(),
                error = %error,
                "cached HF layer package snapshot is incomplete; downloading missing artifacts"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

pub fn manifest_artifact_bytes(artifact: &serde_json::Value) -> Option<u64> {
    artifact
        .get("artifact_bytes")
        .and_then(|value| value.as_u64())
}

fn verify_package_v2_metadata(package_dir: &Path, manifest_bytes: &[u8]) -> Result<()> {
    let manifest: PackageManifestV2 =
        serde_json::from_slice(manifest_bytes).context("parse package-v2 manifest")?;
    manifest
        .validate_root()
        .context("validate package-v2 manifest")?;
    let computed = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed,
        "package-v2 manifest package_id does not match its content"
    );
    let artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package-v2 metadata artifact is absent")?;
    verify_package_v2_artifact(package_dir, artifact)?;
    skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, package_dir)
        .context("resolve package-v2 metadata carrier")?;
    Ok(())
}
