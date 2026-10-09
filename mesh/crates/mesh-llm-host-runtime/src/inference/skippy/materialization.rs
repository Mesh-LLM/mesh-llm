#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
#[cfg(test)]
use sha2::{Digest, Sha256};
use skippy_protocol::LoadMode;

use super::StageLoadRequest;

mod cache_management;
mod package_download;

pub use cache_management::{
    materialized_stages_for_sources, prune_unpinned_materialized_stages,
    remove_materialized_stages_for_sources,
};
pub use package_download::{
    StagePackageRef, download_package_v2_to_local, is_layer_package_ref,
    resolve_hf_package_to_local, resolve_package_v2_full_model_to_local,
    resolve_package_v2_stage_to_local,
};

pub fn configure_materialized_stage_cache() {
    if std::env::var_os("SKIPPY_MATERIALIZED_DIR").is_none() {
        // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
        // runtime work can access the process environment. The current call
        // graph does not enforce that startup boundary; retain the audit TODO.
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("SKIPPY_MATERIALIZED_DIR", materialized_stage_cache_dir()) };
    }
}

pub fn materialized_stage_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-stages")
}

pub use mesh_llm_skippy_adapter::package::StagePackageInfo;
#[cfg(test)]
pub use mesh_llm_skippy_adapter::package::StagePackageLayerInfo;

pub use skippy_api::stage_load::ResolvedStagePackage;

pub fn inspect_stage_package(package_ref: &str) -> Result<StagePackageInfo> {
    // Resolve hf:// to local for inspection, downloading the manifest and any
    // shared package metadata that resolver path needs.
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    mesh_llm_skippy_adapter::package::inspect_local_stage_package(package_ref, &local_ref)
}

/// Resolve an `hf://` package ref in a stage load request to a local directory.
/// Returns the resolved local path if the package ref needed resolution, or `None`
/// if it was already local / not a layer package.
pub fn resolve_stage_load_package(load: &StageLoadRequest) -> Result<Option<ResolvedStagePackage>> {
    if load.load_mode == LoadMode::RuntimeSlice && is_layer_package_ref(&load.package_ref) {
        let descriptor = skippy_api::materialization::package_admission_descriptor(&load.admission);
        let (local_ref, model_part_paths, projector_path) =
            resolve_package_v2_stage_to_local(&load.package_ref, &descriptor)?;
        return skippy_api::materialization::stage_package_from_verified_parts(
            local_ref,
            &load.manifest_sha256,
            model_part_paths,
            projector_path,
        )
        .map(Some);
    }

    anyhow::ensure!(
        load.load_mode != LoadMode::LayerPackage,
        "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
    );
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{FlashAttentionType, LoadMode};

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn write_local_package_v2_fixture(root: &Path) -> (String, String) {
        let manifest = crate::inference::skippy::write_test_package_v2_fixture(
            root,
            "model-a",
            &[
                ("primary", "artifacts/primary.gguf", "input.weight"),
                ("resident", "artifacts/resident.gguf", "resident-tensor"),
                ("unused", "artifacts/unused.gguf", "unused-tensor"),
            ],
        )
        .unwrap();
        let package_id = manifest.package_id.clone();
        let bytes = fs::read(root.join("model-package.json")).unwrap();
        let manifest_sha = sha256_hex(&bytes);
        (manifest_sha, package_id)
    }

    #[test]
    fn inspect_package_v2_uses_the_compact_root_and_metadata_carrier() {
        let root = tempfile::tempdir().unwrap();
        crate::inference::skippy::write_test_package_v2_fixture(
            root.path(),
            "fixture/llama-1b",
            &[
                (
                    "layer-00000",
                    "layers/layer-00000.gguf",
                    "blk.0.attn.weight",
                ),
                (
                    "layer-00001",
                    "layers/layer-00001.gguf",
                    "blk.1.attn.weight",
                ),
            ],
        )
        .unwrap();
        let layer_bytes = fs::metadata(root.path().join("layers/layer-00000.gguf"))
            .unwrap()
            .len();
        fs::remove_file(root.path().join("layers/layer-00000.gguf")).unwrap();
        fs::remove_file(root.path().join("layers/layer-00001.gguf")).unwrap();

        let info = inspect_stage_package(&root.path().to_string_lossy()).unwrap();

        assert_eq!(info.model_id, "fixture/llama-1b");
        assert_eq!(info.layer_count, 2);
        assert_eq!(info.activation_width, 4);
        assert_eq!(info.layers.len(), 2);
        assert_eq!(info.layers[0].tensor_count, 1);
        assert_eq!(info.layers[1].tensor_count, 1);
        assert_eq!(info.layers[0].tensor_bytes, 4);
        assert_eq!(info.layers[0].artifact_bytes, layer_bytes);

        let original_ref = "hf://fixture/llama-1b@revision/package";
        let direct = skippy_api::package::inspection::inspect_local_stage_package(
            original_ref,
            &root.path().to_string_lossy(),
            None,
        )
        .unwrap();
        assert_eq!(direct.package_ref, original_ref);
        assert_eq!(direct.layers, info.layers);
        assert_eq!(direct.manifest_sha256, info.manifest_sha256);
        assert!(info.source_model_path.ends_with("shared/metadata.gguf"));
    }

    #[test]
    fn package_inspection_rejects_changed_metadata_before_reporting_layer_inventory() {
        let root = tempfile::tempdir().unwrap();
        write_local_package_v2_fixture(root.path());
        let metadata_path = root.path().join("shared/metadata.gguf");
        let mut bytes = fs::read(&metadata_path).unwrap();
        // Same-size corruption must still fail carrier/digest validation.
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(metadata_path, bytes).unwrap();
        let error = inspect_stage_package(&root.path().to_string_lossy()).unwrap_err();
        assert!(format!("{error:#}").contains("SHA-256"), "{error:#}");
    }

    fn stage_load_request_for_package(
        package_dir: &Path,
        manifest_sha256: String,
    ) -> StageLoadRequest {
        StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            runtime_profile: Some(String::new()),
            backend: "skippy".to_string(),
            package_ref: package_dir.to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            admission: crate::inference::skippy::test_stage_admission(0, 1),
            participant_set_hash: "participants".to_string(),
            topology_hash: "topology".to_string(),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
            topology_stages: Vec::new(),
            model_path: Some(package_dir.to_string_lossy().to_string()),
            source_model_bytes: None,
            source_model_sha256: None,
            split_certification: None,
            local_source_required: false,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            ctx_size: 8192,
            lane_count: 1,
            continuous_batching: true,
            last_stage_decode_batch: None,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            runtime_settings: Default::default(),
            native_mtp_enabled: true,
            shutdown_generation: 1,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        }
    }

    fn write_cached_package_snapshot(snapshot: &Path, layer_sha: String) {
        fs::create_dir_all(snapshot.join("shared")).unwrap();
        fs::create_dir_all(snapshot.join("layers")).unwrap();
        fs::write(snapshot.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(snapshot.join("layers/layer-000.gguf"), b"layer").unwrap();
        fs::write(
            snapshot.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "model_id": "model-a",
                "source_model": {
                    "path": "model-a.gguf",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "files": [
                        {
                            "path": "model-a.gguf",
                            "size_bytes": 123,
                            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        }
                    ]
                },
                "format": "layer-package",
                "layer_count": 1,
                "shared": {
                    "metadata": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "embeddings": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "output": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    }
                },
                "layers": [
                    {
                        "layer_index": 0,
                        "path": "layers/layer-000.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 5,
                        "sha256": layer_sha
                    }
                ],
                "skippy_abi_version": "0.1.0",
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn resolve_stage_load_package_rejects_legacy_package_mode() {
        let dir = tempfile::tempdir().unwrap();
        let load = stage_load_request_for_package(dir.path(), "0".repeat(64));
        let error = resolve_stage_load_package(&load).unwrap_err().to_string();
        assert!(error.contains("package-v2 graph admission"), "{error}");
    }

    #[test]
    fn runtime_slice_package_v2_resolves_only_admitted_model_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let (manifest_sha, package_id) = write_local_package_v2_fixture(dir.path());
        let mut load = stage_load_request_for_package(dir.path(), manifest_sha);
        load.load_mode = LoadMode::RuntimeSlice;
        load.admission.package_id = package_id;
        load.admission.resident_tensor_ids = vec!["resident-tensor".to_string()];

        let resolved = resolve_stage_load_package(&load).unwrap().unwrap();
        assert_eq!(
            resolved.model_part_paths,
            vec![
                dir.path().join("shared/metadata.gguf"),
                dir.path().join("artifacts/resident.gguf")
            ]
        );
        assert!(
            !resolved
                .model_part_paths
                .iter()
                .any(|path| path.ends_with("unused.gguf"))
        );
    }

    #[test]
    fn resolved_stage_load_package_does_not_accept_v1_manifest() {
        let dir = tempfile::tempdir().unwrap();
        write_cached_package_snapshot(dir.path(), sha256_hex(b"layer"));
        let manifest_bytes = fs::read(dir.path().join("model-package.json")).unwrap();
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let load = StageLoadRequest {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: "model-a".to_string(),
            runtime_profile: Some(String::new()),
            backend: "skippy".to_string(),
            package_ref: dir.path().to_string_lossy().to_string(),
            manifest_sha256,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            admission: crate::inference::skippy::test_stage_admission(0, 1),
            participant_set_hash: "participants".to_string(),
            topology_hash: "topology".to_string(),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
            topology_stages: Vec::new(),
            model_path: None,
            source_model_bytes: None,
            source_model_sha256: None,
            split_certification: None,
            local_source_required: false,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            selected_device: None,
            bind_addr: "127.0.0.1:0".to_string(),
            ctx_size: 512,
            lane_count: 1,
            continuous_batching: true,
            last_stage_decode_batch: None,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
            runtime_settings: Default::default(),
            native_mtp_enabled: true,
            shutdown_generation: 0,
            coordinator_term: 0,
            coordinator_id: None,
            lease_until_unix_ms: 0,
            load_mode: LoadMode::LayerPackage,
            upstream: None,
            downstream: None,
        };

        let error = resolve_stage_load_package(&load)
            .expect_err("v1 package selection must not remain reachable from serving");
        assert!(error.to_string().contains("package-v2 graph admission"));
    }
}
