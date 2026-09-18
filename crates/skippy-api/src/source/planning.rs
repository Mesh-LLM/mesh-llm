//! Source-complete graph-planning manifests from verified local GGUF files.
use super::content_addressed;
use crate::package::SkippyPackageIdentity;
use anyhow::{Context, Result};
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest as PackageManifestV2,
    SourceFile, SourceModel, Tensor, TensorCatalog, TensorIntegrity, TensorStorage,
};
use std::path::PathBuf;
/// Build the source-complete metadata envelope required by generation-10
/// planning directly from local GGUF shards. The envelope is in memory: the
/// source files remain at their original paths and no package or layer shard
/// is written.
pub fn direct_gguf_planning_manifest_from_identity(
    model_id: &str,
    identity: &SkippyPackageIdentity,
) -> Result<(PackageManifestV2, Vec<PathBuf>)> {
    let shard_paths = identity
        .source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    let mut artifacts = Vec::with_capacity(shard_paths.len());
    let mut source_files = Vec::with_capacity(shard_paths.len());
    let mut tensors = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    let mut primary_metadata = None;
    let mut shard_metadata = Vec::with_capacity(shard_paths.len());
    let mut total_tensors = 0_usize;

    for (index, (path, source)) in shard_paths.iter().zip(&identity.source_files).enumerate() {
        let artifact_id = format!("source-{index:05}");
        let logical_path = format!("source-{index:05}.gguf");
        let directory = skippy_model::gguf_catalog::read_gguf_catalog(path)
            .with_context(|| format!("read direct GGUF catalog {}", path.display()))?;
        anyhow::ensure!(
            directory.artifact_bytes == source.bytes,
            "direct GGUF size changed while planning: {}",
            path.display()
        );
        if index == 0 {
            primary_metadata = Some(directory.metadata.clone());
        }
        let native = skippy_runtime::ModelInfo::open(path)
            .with_context(|| format!("open direct GGUF metadata {}", path.display()))?
            .tensors()
            .with_context(|| format!("read direct GGUF tensors {}", path.display()))?
            .into_iter()
            // Zero-element placeholders are skipped by the GGUF catalog
            // reader; drop them from the native side so counts agree.
            .filter(|tensor| tensor.element_count > 0)
            .collect::<Vec<_>>();
        let shard_tensors = direct_planning_tensor_catalog(&directory, &native, &artifact_id)?;
        total_tensors = total_tensors
            .checked_add(shard_tensors.entries.len())
            .context("direct GGUF tensor count overflow")?;
        shard_metadata.push(directory.metadata.clone());
        for tensor in shard_tensors.entries {
            anyhow::ensure!(
                names.insert(tensor.name.clone()),
                "duplicate direct GGUF tensor {:?}",
                tensor.name
            );
            tensors.push(tensor);
        }
        source_files.push(SourceFile {
            path: logical_path.clone(),
            byte_size: source.bytes,
            sha256: source.sha256.clone(),
        });
        artifacts.push(Artifact {
            id: artifact_id,
            path: logical_path,
            byte_size: source.bytes,
            sha256: source.sha256.clone(),
        });
    }
    anyhow::ensure!(!tensors.is_empty(), "direct GGUF tensor inventory is empty");
    validate_direct_gguf_shards(&shard_metadata, total_tensors)?;

    let primary = source_files
        .first()
        .context("direct GGUF source file list is empty")?;
    let mut manifest = PackageManifestV2 {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: model_id.to_string(),
        source_model: SourceModel {
            sha256: primary.sha256.clone(),
            metadata_artifact_id: "source-00000".to_string(),
            repo: None,
            revision: None,
            primary_file: Some(primary.path.clone()),
            canonical_ref: None,
            distribution_id: Some(content_addressed::aggregate_source_sha256(
                &identity.source_files,
            )),
            files: source_files,
        },
        format: "gguf".to_string(),
        layer_count: identity.layer_count,
        model_metadata: primary_metadata.context("direct GGUF metadata is empty")?,
        artifact_catalog: ArtifactCatalog { entries: artifacts },
        tensor_catalog: TensorCatalog { entries: tensors },
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        generator_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_unix_secs: 0,
    };
    manifest.package_id = manifest
        .computed_package_id()
        .context("compute direct GGUF planning identity")?;
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("validate direct GGUF planning manifest")?;
    Ok((manifest, shard_paths))
}

fn validate_direct_gguf_shards(
    shard_metadata: &[std::collections::BTreeMap<String, serde_json::Value>],
    total_tensors: usize,
) -> Result<()> {
    let first = shard_metadata
        .first()
        .context("direct GGUF shard inventory is empty")?;
    for (index, metadata) in shard_metadata.iter().enumerate() {
        if shard_metadata.len() > 1 || metadata.contains_key("split.count") {
            anyhow::ensure!(
                metadata
                    .get("split.count")
                    .and_then(serde_json::Value::as_u64)
                    == Some(shard_metadata.len() as u64),
                "incomplete direct GGUF shard set: split.count mismatch"
            );
            anyhow::ensure!(
                metadata.get("split.no").and_then(serde_json::Value::as_u64) == Some(index as u64),
                "direct GGUF split.no mismatch"
            );
        }
        for (key, value) in metadata {
            if key != "split.no" && key != "split.count" && key != "split.tensors.count" {
                anyhow::ensure!(
                    first.get(key) == Some(value),
                    "inconsistent direct GGUF metadata {key:?} across shards"
                );
            }
        }
    }
    if shard_metadata.len() > 1 || first.contains_key("split.count") {
        anyhow::ensure!(
            first
                .get("split.tensors.count")
                .and_then(serde_json::Value::as_u64)
                == Some(total_tensors as u64),
            "incomplete direct GGUF tensor inventory: split.tensors.count mismatch"
        );
    }
    Ok(())
}

fn direct_planning_tensor_catalog(
    directory: &skippy_model::gguf_catalog::GgufCatalog,
    native: &[skippy_runtime::TensorInfo],
    artifact_id: &str,
) -> Result<TensorCatalog> {
    let by_name = native
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<std::collections::BTreeMap<_, _>>();
    anyhow::ensure!(
        by_name.len() == native.len() && native.len() == directory.tensors.len(),
        "native and direct GGUF tensor inventories disagree"
    );
    let mut source = directory.tensors.iter().collect::<Vec<_>>();
    source.sort_by(|left, right| {
        (left.data_offset, &left.name).cmp(&(right.data_offset, &right.name))
    });
    let mut entries: Vec<Tensor> = Vec::with_capacity(source.len());
    let mut last_owned: Option<(u64, u64, usize)> = None;
    for tensor in source {
        let info = by_name
            .get(tensor.name.as_str())
            .context("direct GGUF tensor is missing from native inventory")?;
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .context("direct GGUF tensor element count overflow")?;
        anyhow::ensure!(
            info.ggml_type == tensor.ggml_type && info.element_count == elements,
            "native and direct GGUF metadata disagree for {:?}",
            tensor.name
        );
        anyhow::ensure!(
            info.byte_size > 0,
            "empty direct GGUF tensor storage for {:?}",
            tensor.name
        );
        let end = tensor
            .data_offset
            .checked_add(info.byte_size)
            .context("direct GGUF tensor extent overflow")?;
        anyhow::ensure!(
            end <= directory.artifact_bytes,
            "direct GGUF tensor {:?} storage exceeds artifact bounds",
            tensor.name
        );
        let storage = if let Some((start, previous_end, owner)) =
            last_owned.filter(|(_, previous_end, _)| tensor.data_offset < *previous_end)
        {
            let target = &entries[owner];
            anyhow::ensure!(
                tensor.data_offset == start
                    && end == previous_end
                    && tensor.ggml_type == target.ggml_type
                    && tensor.dimensions == target.dimensions,
                "overlapping or mismatched direct GGUF alias storage for {:?}",
                tensor.name
            );
            TensorStorage::Alias {
                target_tensor_id: target.id.clone(),
            }
        } else {
            last_owned = Some((tensor.data_offset, end, entries.len()));
            TensorStorage::Owned {
                artifact_id: artifact_id.to_string(),
                data_offset: tensor.data_offset,
                stored_length: info.byte_size,
                alignment: directory.alignment,
                integrity: TensorIntegrity::ArtifactSha256,
            }
        };
        entries.push(Tensor {
            id: tensor.name.clone(),
            name: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
            dimensions: tensor.dimensions.clone(),
            layer_ordinal: None,
            storage,
        });
    }
    Ok(TensorCatalog { entries })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn split_metadata(
        split_no: u64,
        split_count: u64,
        tensor_count: u64,
    ) -> std::collections::BTreeMap<String, serde_json::Value> {
        [
            ("split.no".to_string(), split_no.into()),
            ("split.count".to_string(), split_count.into()),
            ("split.tensors.count".to_string(), tensor_count.into()),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn direct_gguf_inventory_uses_primary_global_tensor_count() {
        let metadata = vec![split_metadata(0, 2, 3), split_metadata(1, 2, 2)];

        validate_direct_gguf_shards(&metadata, 3).unwrap();

        let stale_primary = vec![split_metadata(0, 2, 2), split_metadata(1, 2, 3)];
        let error = validate_direct_gguf_shards(&stale_primary, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("split.tensors.count mismatch"), "{error}");
    }
}
