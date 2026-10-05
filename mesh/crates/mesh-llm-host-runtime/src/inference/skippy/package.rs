#[cfg(test)]
use std::path::{Path, PathBuf};

use anyhow::Result;
#[cfg(test)]
use sha2::{Digest, Sha256};
#[cfg(test)]
use skippy_package_format::PackageManifest as PackageManifestV2;
#[cfg(test)]
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, SourceFile, SourceModel, Tensor,
    TensorCatalog, TensorIntegrity, TensorStorage,
};

pub use mesh_llm_skippy_adapter::package::{
    identity_from_package_v2, identity_from_package_v2_metadata, is_package_v2_ref,
};

#[cfg(test)]
const PACKAGE_V2_MANIFEST: &str = "model-package.json";

#[cfg(test)]
pub(crate) fn write_test_package_v2_fixture(
    package_dir: &Path,
    model_id: &str,
    payloads: &[(&str, &str, &str)],
) -> Result<PackageManifestV2> {
    anyhow::ensure!(!payloads.is_empty(), "test package needs payloads");
    let mut payloads = payloads.to_vec();
    payloads.sort_by_key(|(artifact_id, _, _)| *artifact_id);
    let mut payload_locators = Vec::with_capacity(payloads.len());
    let mut artifacts = Vec::with_capacity(payloads.len() + 1);
    for (index, (artifact_id, relative_path, tensor_name)) in payloads.iter().enumerate() {
        let path = package_dir.join(relative_path);
        write_test_payload_gguf(&path, tensor_name, u8::try_from(index).unwrap_or(u8::MAX))?;
        let catalog = skippy_model::gguf_catalog::read_gguf_catalog(&path)?;
        payload_locators.push((
            *tensor_name,
            u32::try_from(index)?,
            catalog.tensors[0].data_offset,
            4_u64,
            u32::try_from(catalog.alignment)?,
        ));
        let bytes = std::fs::read(&path)?;
        artifacts.push(Artifact {
            id: (*artifact_id).to_string(),
            path: (*relative_path).to_string(),
            byte_size: bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(&bytes)),
        });
    }
    let metadata_relative = "shared/metadata.gguf";
    let metadata_path = package_dir.join(metadata_relative);
    write_test_metadata_carrier(&metadata_path, &payload_locators)?;
    let metadata_bytes = std::fs::read(&metadata_path)?;
    artifacts.insert(
        0,
        Artifact {
            id: "metadata".to_string(),
            path: metadata_relative.to_string(),
            byte_size: metadata_bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(&metadata_bytes)),
        },
    );
    let source = &artifacts[1];
    let mut root = PackageManifestV2 {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: model_id.to_string(),
        source_model: SourceModel {
            sha256: source.sha256.clone(),
            metadata_artifact_id: "metadata".to_string(),
            repo: None,
            revision: None,
            primary_file: Some(source.path.clone()),
            canonical_ref: None,
            distribution_id: None,
            files: vec![SourceFile {
                path: source.path.clone(),
                byte_size: source.byte_size,
                sha256: source.sha256.clone(),
            }],
        },
        format: "gguf".to_string(),
        layer_count: 2,
        model_metadata: Default::default(),
        artifact_catalog: ArtifactCatalog { entries: artifacts },
        tensor_catalog: TensorCatalog {
            entries: Vec::new(),
        },
        sidecars: Vec::new(),
        publisher_metadata: Vec::new(),
        publisher_defaults: None,
        generation: None,
        native_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        generator_version: "test".to_string(),
        created_at_unix_secs: 1,
    };
    root.package_id = root.computed_package_id()?;
    let manifest_bytes = serde_json::to_vec_pretty(&root)?;
    std::fs::create_dir_all(package_dir)?;
    std::fs::write(package_dir.join(PACKAGE_V2_MANIFEST), manifest_bytes)?;
    skippy_model::package_carrier::resolve_package_carrier_from_dir(root, package_dir)
}

#[cfg(test)]
fn write_test_metadata_carrier(path: &Path, tensors: &[(&str, u32, u64, u64, u32)]) -> Result<()> {
    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32_array(bytes: &mut Vec<u8>, key: &str, values: impl Iterator<Item = u32>) {
        let values = values.collect::<Vec<_>>();
        push_string(bytes, key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_u64_array(bytes: &mut Vec<u8>, key: &str, values: impl Iterator<Item = u64>) {
        let values = values.collect::<Vec<_>>();
        push_string(bytes, key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&10_u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&10_u64.to_le_bytes());
    push_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    push_string(&mut bytes, "llama");
    push_u32(&mut bytes, "llama.block_count", 2);
    push_u32(&mut bytes, "llama.embedding_length", 4);
    push_string(&mut bytes, "skippy.package.metadata_only");
    bytes.extend_from_slice(&7_u32.to_le_bytes());
    bytes.push(1);
    push_u32(
        &mut bytes,
        "skippy.package.part_count",
        u32::try_from(tensors.len())?,
    );
    push_u32(&mut bytes, "skippy.package.locator_schema", 1);
    push_u32_array(
        &mut bytes,
        "skippy.package.tensor_part",
        tensors.iter().map(|(_, part, _, _, _)| *part),
    );
    push_u64_array(
        &mut bytes,
        "skippy.package.tensor_offset",
        tensors.iter().map(|(_, _, offset, _, _)| *offset),
    );
    push_u64_array(
        &mut bytes,
        "skippy.package.tensor_size",
        tensors.iter().map(|(_, _, _, size, _)| *size),
    );
    push_u32_array(
        &mut bytes,
        "skippy.package.tensor_alignment",
        tensors.iter().map(|(_, _, _, _, alignment)| *alignment),
    );
    for (name, _, _, _, _) in tensors {
        push_string(&mut bytes, name);
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
fn write_test_payload_gguf(path: &Path, tensor_name: &str, value: u8) -> Result<()> {
    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    push_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    push_string(&mut bytes, "llama");
    push_string(&mut bytes, "llama.block_count");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    push_string(&mut bytes, tensor_name);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend_from_slice(&[value, value, value, value]);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

pub use skippy_api::package::SkippyPackageIdentity;
#[cfg(test)]
use skippy_api::package::package_v2_layer_weight_bytes;

pub use skippy_api::source::planning::direct_gguf_planning_manifest_from_identity;
pub use skippy_api::source::{direct_gguf_source_paths, synthetic_content_addressed_gguf_package};

pub use mesh_llm_skippy_adapter::synthetic_direct_gguf_package;

pub fn synthetic_huggingface_gguf_package(
    model_id: &str,
    identity: &crate::models::HuggingFaceModelIdentity,
) -> Result<SkippyPackageIdentity> {
    let snapshot_root = skippy_model_hf::store::local::huggingface_snapshot_path(
        &identity.repo_id,
        hf_hub::RepoTypeModel,
        &identity.revision,
    );
    skippy_api::source::synthetic_huggingface_gguf_package(
        model_id,
        identity,
        &snapshot_root,
        || crate::models::build_hf_api(false),
    )
}

#[cfg(test)]
fn hex_lower(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Build a `SkippyPackageIdentity` from a remote HF layer package.
///
/// Resolves the package into the local HF cache for inspection, downloading
/// the manifest and shared metadata that the resolver requires, but not layer
/// files. Layer artifacts are fetched later by the node that materializes or
/// loads its assigned stage.
pub fn identity_from_layer_package(package_ref: &str) -> Result<SkippyPackageIdentity> {
    // Resolve hf:// to a local package dir for lightweight package inspection.
    let local_ref =
        super::materialization::resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    if is_package_v2_ref(&local_ref) {
        return identity_from_package_v2_metadata(package_ref, &local_ref);
    }
    anyhow::bail!(
        "layer-package schema v1 is offline-only; split serving requires a package-v2 manifest"
    )
}

fn layer_weight_bytes_from_info(info: &skippy_runtime::package::LayerPackageInfo) -> Vec<u64> {
    let mut layers = info.layers.clone();
    layers.sort_by_key(|layer| layer.layer_index);
    if layers.len() != info.layer_count as usize
        || layers
            .iter()
            .enumerate()
            .any(|(index, layer)| layer.layer_index as usize != index)
    {
        return Vec::new();
    }
    let mut weights = layers
        .into_iter()
        .map(|layer| layer.tensor_bytes.max(layer.artifact_bytes))
        .collect::<Vec<_>>();
    let accounted = weights.iter().copied().sum::<u64>();
    let unaccounted = info
        .source_model_bytes
        .unwrap_or_default()
        .saturating_sub(accounted);
    if let Some((first, rest)) = weights.split_first_mut() {
        *first = first.saturating_add(unaccounted.div_ceil(2));
        if let Some(last) = rest.last_mut() {
            *last = last.saturating_add(unaccounted / 2);
        } else {
            *first = first.saturating_add(unaccounted / 2);
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_v2_identity_rejects_v1_without_fallback() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join(PACKAGE_V2_MANIFEST),
            br#"{"schema_version":1}"#,
        )
        .unwrap();

        let error = identity_from_package_v2(root.path())
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires package schema 2"), "{error}");
    }

    #[test]
    fn package_v2_identity_keeps_source_provenance_separate_from_generated_metadata() {
        let root = tempfile::tempdir().unwrap();
        let metadata_path = root.path().join("shared/metadata.gguf");
        let mut manifest = write_test_package_v2_fixture(
            root.path(),
            "fixture/model",
            &[("payload", "shared/payload.gguf", "blk.0.weight")],
        )
        .unwrap();
        let source_sha256 = "a".repeat(64);
        manifest.source_model.sha256 = source_sha256.clone();
        manifest.source_model.repo = Some("fixture/model".to_string());
        manifest.source_model.revision = Some("revision".to_string());
        manifest.source_model.primary_file = Some("source.gguf".to_string());
        manifest.source_model.canonical_ref =
            Some("hf://fixture/model@revision/source.gguf".to_string());
        manifest.source_model.files = vec![SourceFile {
            path: "source.gguf".to_string(),
            byte_size: 1024,
            sha256: source_sha256.clone(),
        }];
        manifest.package_id = manifest.computed_package_id().unwrap();
        std::fs::write(
            root.path().join(PACKAGE_V2_MANIFEST),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let direct = identity_from_package_v2(root.path()).unwrap();
        let metadata = identity_from_package_v2_metadata(
            &root.path().to_string_lossy(),
            &root.path().to_string_lossy(),
        )
        .unwrap();

        assert_eq!(direct.source_model_sha256, source_sha256);
        assert_eq!(
            direct.source_model_path,
            metadata_path.canonicalize().unwrap()
        );
        assert_eq!(metadata.source_model_sha256, direct.source_model_sha256);
        assert_eq!(
            metadata.source_model_path,
            root.path().join("shared/metadata.gguf")
        );
    }

    #[test]
    fn package_v2_layer_weights_fall_back_when_ordinals_are_unavailable() {
        let tensor = |id: &str, layer_ordinal, stored_length| Tensor {
            id: id.to_string(),
            name: id.to_string(),
            ggml_type: 0,
            dimensions: vec![1],
            layer_ordinal,
            storage: TensorStorage::Owned {
                artifact_id: "source".to_string(),
                data_offset: 0,
                stored_length,
                alignment: 1,
                integrity: TensorIntegrity::ArtifactSha256,
            },
        };
        let mut manifest = PackageManifestV2 {
            schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "fixture/model".to_string(),
            source_model: SourceModel {
                sha256: String::new(),
                metadata_artifact_id: "source".to_string(),
                repo: None,
                revision: None,
                primary_file: None,
                canonical_ref: None,
                distribution_id: None,
                files: Vec::new(),
            },
            format: "gguf".to_string(),
            layer_count: 2,
            model_metadata: Default::default(),
            artifact_catalog: ArtifactCatalog {
                entries: Vec::new(),
            },
            tensor_catalog: TensorCatalog {
                entries: vec![tensor("first", None, 10), tensor("second", None, 20)],
            },
            sidecars: Vec::new(),
            publisher_metadata: Vec::new(),
            publisher_defaults: None,
            generation: None,
            native_abi_version: String::new(),
            generator_version: String::new(),
            created_at_unix_secs: 0,
        };

        assert!(package_v2_layer_weight_bytes(&manifest).unwrap().is_empty());

        manifest.tensor_catalog.entries[0].layer_ordinal = Some(0);
        assert!(package_v2_layer_weight_bytes(&manifest).unwrap().is_empty());

        manifest.tensor_catalog.entries[1].layer_ordinal = Some(1);
        assert_eq!(
            package_v2_layer_weight_bytes(&manifest).unwrap(),
            vec![10, 20]
        );
    }

    fn manifest_with_mtp_tensor(
        tensor_name: &str,
        generation: Option<skippy_package_format::Generation>,
    ) -> PackageManifestV2 {
        PackageManifestV2 {
            schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
            package_id: "fixture-package".to_string(),
            model_id: "fixture/model".to_string(),
            source_model: SourceModel {
                sha256: String::new(),
                metadata_artifact_id: "source".to_string(),
                repo: None,
                revision: None,
                primary_file: None,
                canonical_ref: None,
                distribution_id: None,
                files: Vec::new(),
            },
            format: "gguf".to_string(),
            layer_count: 1,
            model_metadata: Default::default(),
            artifact_catalog: ArtifactCatalog {
                entries: Vec::new(),
            },
            tensor_catalog: TensorCatalog {
                entries: vec![Tensor {
                    id: "tensor-0".to_string(),
                    name: tensor_name.to_string(),
                    ggml_type: 0,
                    dimensions: vec![1],
                    layer_ordinal: Some(0),
                    storage: TensorStorage::Owned {
                        artifact_id: "source".to_string(),
                        data_offset: 0,
                        stored_length: 10,
                        alignment: 1,
                        integrity: TensorIntegrity::ArtifactSha256,
                    },
                }],
            },
            sidecars: Vec::new(),
            publisher_metadata: Vec::new(),
            publisher_defaults: None,
            generation,
            native_abi_version: String::new(),
            generator_version: String::new(),
            created_at_unix_secs: 0,
        }
    }

    #[test]
    fn package_v2_warns_when_mtp_tensors_ship_without_generation() {
        let manifest = manifest_with_mtp_tensor("blk.40.nextn.eh_proj.weight", None);

        assert!(manifest_ships_mtp_without_generation(&manifest));
    }

    #[test]
    fn package_v2_warns_when_generation_lacks_speculative_decoding() {
        let manifest = manifest_with_mtp_tensor(
            "blk.40.nextn.eh_proj.weight",
            Some(skippy_package_format::Generation {
                request_defaults: None,
                speculative_decoding: None,
            }),
        );

        assert!(manifest_ships_mtp_without_generation(&manifest));
    }

    #[test]
    fn package_v2_stays_silent_when_mtp_generation_is_declared() {
        let manifest = manifest_with_mtp_tensor(
            "blk.40.nextn.eh_proj.weight",
            Some(skippy_package_format::Generation {
                request_defaults: None,
                speculative_decoding: Some(skippy_package_format::SpeculativeDecoding {
                    default: "mtp".to_string(),
                    proposers: Default::default(),
                    strategies: std::collections::BTreeMap::from([(
                        "mtp".to_string(),
                        skippy_package_format::StrategySpec {
                            kind: StrategyKind::NativeMtp {
                                proposer: None,
                                prediction_depth: Some(1),
                                layer_indices: vec![1],
                                window_policy: None,
                            },
                        },
                    )]),
                }),
            }),
        );

        assert!(!manifest_ships_mtp_without_generation(&manifest));
    }

    #[test]
    fn package_v2_stays_silent_for_non_mtp_tensors_without_generation() {
        let manifest = manifest_with_mtp_tensor("blk.40.attn_q.weight", None);

        assert!(!manifest_ships_mtp_without_generation(&manifest));
    }

    #[test]
    #[ignore = "requires SKIPPY_PACKAGE_V2_TEST_DIR"]
    fn package_v2_identity_reads_a_real_package() {
        let package_dir = std::env::var_os("SKIPPY_PACKAGE_V2_TEST_DIR")
            .map(PathBuf::from)
            .expect("SKIPPY_PACKAGE_V2_TEST_DIR is required");

        let identity = identity_from_package_v2(&package_dir).unwrap();

        assert_eq!(identity.layer_count, 32);
        assert!(identity.activation_width > 0);
        assert_eq!(identity.layer_weight_bytes.len(), 32);
        assert!(identity.source_model_path.is_file());
        assert_eq!(identity.manifest_sha256.len(), 64);

        let expected_manifest_sha256 = identity.manifest_sha256.clone();
        let expected_source_sha256 = identity.source_model_sha256.clone();
        let strict = super::super::local_source::into_content_addressed_identity(identity)
            .expect("index package-v2 source for strict-local loading");
        let verified = super::super::local_source::verify_registered_content_source(
            "granite-v2-test",
            &strict.package_ref,
            &expected_manifest_sha256,
            &expected_source_sha256,
        )
        .expect("resolve indexed package-v2 source");
        assert_eq!(verified, strict);
    }

    #[test]
    fn package_layer_weights_include_shared_model_bytes_at_endpoints() {
        let info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(120),
            layer_count: 2,
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 1,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert_eq!(layer_weight_bytes_from_info(&info), vec![55, 65]);
    }

    #[test]
    fn package_layer_weights_require_contiguous_indices() {
        let mut info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(70),
            layer_count: 2,
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 2,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert!(layer_weight_bytes_from_info(&info).is_empty());
        info.layers[1].layer_index = 1;
        assert_eq!(layer_weight_bytes_from_info(&info), vec![30, 40]);
    }
}

#[cfg(test)]
mod abi_provenance_tests;
#[cfg(test)]
use skippy_api::package::manifest_ships_mtp_without_generation;
#[cfg(test)]
use skippy_package_format::StrategyKind;
