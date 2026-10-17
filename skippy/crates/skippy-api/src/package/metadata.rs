//! Lightweight identity verification using the package metadata carrier.
use super::{
    PACKAGE_V2_MANIFEST, SkippyPackageIdentity, hex_lower, package_v2_generation_info,
    package_v2_layer_weight_bytes, source_file_sha256,
};
use crate::hash_cache::SidecarDigestCache;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use skippy_package_format::PackageManifest as PackageManifestV2;
use std::path::{Path, PathBuf};

/// Probe the schema marker before choosing a package resolver.
pub fn is_package_v2_ref(package_ref: &str) -> bool {
    let manifest_path = Path::new(package_ref).join(PACKAGE_V2_MANIFEST);
    std::fs::read(&manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| {
            manifest
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
        })
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION))
}

/// Verify package metadata without downloading layer artifacts.
/// The optional cache is advisory; callers choose its policy.
pub fn identity_from_package_v2_metadata(
    package_ref: &str,
    local_ref: &str,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageIdentity> {
    let package_dir = PathBuf::from(local_ref);
    let manifest_path = package_dir.join(PACKAGE_V2_MANIFEST);
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
            .context("resolve package-v2 metadata carrier")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "package-v2 manifest package_id does not match its content"
    );
    // Producer ABI is provenance; schema, carrier, and artifact validation
    // govern package compatibility. Runtime ABI is checked when loading.
    let metadata_artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package-v2 metadata artifact is absent")?;
    let metadata_relative = Path::new(&metadata_artifact.path);
    anyhow::ensure!(
        !metadata_relative.as_os_str().is_empty()
            && metadata_relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
        "package-v2 artifact path is not a safe relative path: {metadata_relative:?}"
    );
    let source_model_path = package_dir.join(metadata_relative);
    let source_metadata = source_model_path
        .metadata()
        .with_context(|| format!("stat package-v2 source {}", source_model_path.display()))?;
    anyhow::ensure!(
        source_metadata.is_file() && source_metadata.len() == metadata_artifact.byte_size,
        "package-v2 metadata artifact size differs from manifest"
    );
    let source_sha256 = source_file_sha256(&source_model_path, &source_metadata, digest_cache)?;
    anyhow::ensure!(
        source_sha256 == metadata_artifact.sha256,
        "package-v2 metadata artifact SHA-256 differs from manifest"
    );
    let architecture = manifest
        .model_metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        .context("package-v2 model metadata is missing general.architecture")?;
    let activation_width_key = format!("{architecture}.embedding_length");
    let activation_width = manifest
        .model_metadata
        .get(&activation_width_key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .with_context(|| {
            format!("package-v2 model metadata is missing positive {activation_width_key}")
        })?;
    let source_model_bytes = manifest
        .source_model
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
        .context("package-v2 source byte count overflow")?;
    anyhow::ensure!(
        source_model_bytes > 0,
        "package-v2 source model byte count must be positive"
    );
    let layer_weight_bytes = package_v2_layer_weight_bytes(&manifest)?;
    let tensor_count = u64::try_from(manifest.tensor_catalog.entries.len())
        .context("package-v2 tensor count exceeds u64")?;
    let manifest_sha256 = hex_lower(&Sha256::digest(&manifest_bytes));
    let canonical_package_ref = canonical_layer_package_ref(package_ref, local_ref);
    super::warn_if_mtp_without_generation(&manifest);
    Ok(SkippyPackageIdentity {
        package_ref: canonical_package_ref,
        manifest_sha256,
        source_model_path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes,
        source_files: Vec::new(),
        layer_weight_bytes,
        layer_count: manifest.layer_count,
        activation_width,
        tensor_count,
        generation: manifest.generation.as_ref().map(package_v2_generation_info),
        publisher_defaults: manifest.publisher_defaults,
    })
}

/// Detect if a local path is inside an HF cache directory and convert to `hf://` ref.
///
/// HF cache paths look like:
///   `.../hub/models--owner--name/snapshots/<hash>/`
///
/// Returns `Some("hf://owner/name@hash")` if detected, `None` otherwise.
fn hf_ref_from_cache_path(path: &str) -> Option<String> {
    // Walk path components looking for "models--*" followed by "snapshots"
    let path = std::path::Path::new(path);
    let components: Vec<&std::ffi::OsStr> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect();
    for (i, comp) in components.iter().enumerate() {
        let s = comp.to_str()?;
        if let Some(repo_part) = s.strip_prefix("models--") {
            // Verify next component is "snapshots" and preserve the exact
            // snapshot revision/hash so peers fetch identical package content.
            if components.get(i + 1).and_then(|c| c.to_str()) == Some("snapshots") {
                let revision = components.get(i + 2)?.to_str()?;
                // repo_part is "owner--name", convert to "owner/name"
                let repo = repo_part.replacen("--", "/", 1);
                if repo.contains('/') {
                    return Some(format!("hf://{repo}@{revision}"));
                }
            }
        }
    }
    None
}

fn canonical_layer_package_ref(package_ref: &str, local_ref: &str) -> String {
    hf_ref_from_cache_path(local_ref)
        .or_else(|| hf_ref_from_cache_path(package_ref))
        .unwrap_or_else(|| package_ref.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_v2_ref_requires_the_v2_schema_marker() {
        let root = tempfile::tempdir().unwrap();
        let manifest = root.path().join(PACKAGE_V2_MANIFEST);

        std::fs::write(&manifest, br#"{"schema_version":1}"#).unwrap();
        assert!(!is_package_v2_ref(&root.path().to_string_lossy()));

        std::fs::write(&manifest, br#"{"schema_version":2}"#).unwrap();
        assert!(is_package_v2_ref(&root.path().to_string_lossy()));
    }

    #[test]
    fn hf_ref_from_cache_path_preserves_snapshot_revision() {
        let package_ref =
            "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123/model-package.json";

        assert_eq!(
            hf_ref_from_cache_path(package_ref),
            Some("hf://meshllm/Qwen3-layers@abc123".to_string())
        );
    }

    #[test]
    fn canonical_layer_package_ref_prefers_resolved_snapshot() {
        let local_ref = "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123";

        assert_eq!(
            canonical_layer_package_ref("hf://meshllm/Qwen3-layers@main", local_ref),
            "hf://meshllm/Qwen3-layers@abc123"
        );
    }
}
