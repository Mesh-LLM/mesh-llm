//! Inspect an acquired local package without downloading layer artifacts.
use crate::hash_cache::SidecarDigestCache;
use anyhow::{Context, Result};
use skippy_package_format::{PackageManifest as PackageManifestV2, TensorStorage};
use skippy_runtime::package::{self, LayerPackageInfo, PackageGenerationInfo};
use std::{fs, path::PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct StagePackageInfo {
    pub package_ref: String,
    pub package_dir: PathBuf,
    pub manifest_sha256: String,
    pub model_id: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub layer_count: u32,
    pub activation_width: u32,
    pub generation: Option<PackageGenerationInfo>,
    pub projector_path: Option<String>,
    pub layers: Vec<StagePackageLayerInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePackageLayerInfo {
    pub layer_index: u32,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    pub artifact_bytes: u64,
}

/// Inspect the caller-resolved local package, retaining the original reference.
/// Network acquisition and advisory digest-cache policy belong to the caller.
pub fn inspect_local_stage_package(
    package_ref: &str,
    local_ref: &str,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<StagePackageInfo> {
    if super::metadata::is_package_v2_ref(local_ref) {
        return stage_package_info_v2(package_ref, local_ref, digest_cache);
    }
    let info = package::inspect_layer_package(local_ref)
        .with_context(|| format!("inspect skippy layer package {package_ref}"))?;
    stage_package_info(package_ref, info)
}

fn stage_package_info_v2(
    package_ref: &str,
    local_ref: &str,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<StagePackageInfo> {
    let identity =
        super::metadata::identity_from_package_v2_metadata(package_ref, local_ref, digest_cache)?;
    let package_dir = PathBuf::from(local_ref);
    let manifest_path = package_dir.join("model-package.json");
    let manifest: PackageManifestV2 = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
            .context("resolve package-v2 metadata carrier")?;

    let mut layers = (0..manifest.layer_count)
        .map(|layer_index| StagePackageLayerInfo {
            layer_index,
            tensor_count: 0,
            tensor_bytes: 0,
            artifact_bytes: 0,
        })
        .collect::<Vec<_>>();
    let mut layer_artifacts = vec![std::collections::BTreeSet::new(); layers.len()];
    for tensor in &manifest.tensor_catalog.entries {
        let Some(layer_index) = tensor.layer_ordinal else {
            continue;
        };
        let layer = layers.get_mut(layer_index as usize).with_context(|| {
            format!("package-v2 tensor layer {layer_index} exceeds layer count")
        })?;
        layer.tensor_count += 1;
        if let TensorStorage::Owned {
            artifact_id,
            stored_length,
            ..
        } = &tensor.storage
        {
            layer.tensor_bytes = layer
                .tensor_bytes
                .checked_add(*stored_length)
                .context("package-v2 layer byte count overflow")?;
            layer_artifacts[layer_index as usize].insert(artifact_id.as_str());
        }
    }
    for (layer, artifact_ids) in layers.iter_mut().zip(layer_artifacts) {
        layer.artifact_bytes = artifact_ids.into_iter().try_fold(0_u64, |total, id| {
            let bytes = manifest
                .artifact_catalog
                .entries
                .iter()
                .find(|artifact| artifact.id == id)
                .with_context(|| format!("package-v2 layer references absent artifact {id:?}"))?
                .byte_size;
            total
                .checked_add(bytes)
                .context("package-v2 layer artifact byte count overflow")
        })?;
    }
    let projector_path = manifest
        .sidecars
        .iter()
        .find_map(|sidecar| {
            (sidecar.kind == skippy_package_format::SidecarKind::Mmproj)
                .then_some(sidecar.artifact_id.as_str())
        })
        .and_then(|artifact_id| {
            manifest
                .artifact_catalog
                .entries
                .iter()
                .find(|artifact| artifact.id == artifact_id)
        })
        .map(|artifact| {
            package_dir
                .join(&artifact.path)
                .to_string_lossy()
                .into_owned()
        });

    Ok(StagePackageInfo {
        package_ref: package_ref.to_string(),
        package_dir,
        manifest_sha256: identity.manifest_sha256,
        model_id: manifest.model_id,
        source_model_path: identity.source_model_path.to_string_lossy().into_owned(),
        source_model_sha256: identity.source_model_sha256,
        source_model_bytes: Some(identity.source_model_bytes),
        layer_count: identity.layer_count,
        activation_width: identity.activation_width,
        generation: identity.generation,
        projector_path,
        layers,
    })
}

fn stage_package_info(package_ref: &str, info: LayerPackageInfo) -> Result<StagePackageInfo> {
    Ok(StagePackageInfo {
        package_ref: package_ref.to_string(),
        package_dir: info.package_dir,
        manifest_sha256: info.manifest_sha256,
        model_id: info.model_id,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        activation_width: 0,
        generation: info.generation,
        projector_path: info
            .projectors
            .first()
            .map(|projector| projector.path.to_string_lossy().to_string()),
        layers: info
            .layers
            .into_iter()
            .map(|layer| StagePackageLayerInfo {
                layer_index: layer.layer_index,
                tensor_count: layer.tensor_count,
                tensor_bytes: layer.tensor_bytes,
                artifact_bytes: layer.artifact_bytes,
            })
            .collect(),
    })
}
