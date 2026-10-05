//! Host wiring for Skippy-owned package acquisition.
use anyhow::Result;
pub use mesh_llm_skippy_adapter::package::acquisition::{StagePackageRef, is_layer_package_ref};
use skippy_package_format::stage_admission::StageAdmissionDescriptor as PackageV2StageAdmissionDescriptor;
use std::path::PathBuf;

fn acquisition() -> skippy_api::package::acquisition::remote::PackageAcquisition {
    mesh_llm_skippy_adapter::package::acquisition::with_client(|| {
        crate::models::build_hf_api(false)
    })
}

pub fn resolve_hf_package_to_local(
    package_ref: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    acquisition().resolve_hf_package_to_local(
        package_ref,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}
pub fn resolve_package_v2_stage_to_local(
    package_ref: &str,
    admission: &PackageV2StageAdmissionDescriptor,
) -> Result<(String, Vec<PathBuf>, Option<PathBuf>)> {
    acquisition().resolve_package_v2_stage_to_local(package_ref, admission)
}
pub fn resolve_package_v2_full_model_to_local(
    package_ref: &str,
) -> Result<(Vec<PathBuf>, Option<PathBuf>)> {
    acquisition().resolve_package_v2_full_model_to_local(package_ref)
}
pub fn download_package_v2_to_local(package_ref: &str) -> Result<PathBuf> {
    acquisition().download_package_v2_to_local(package_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_skippy_adapter::package::acquisition::{
        resolve_local_package_files, verify_cached_hf_package_files,
    };
    use skippy_api::materialization::safe_manifest_file_path;
    use std::ffi::OsString;
    use std::{fs, path::Path};

    use serial_test::serial;
    use sha2::{Digest, Sha256};

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::set_var(key, value) };
        } else {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::remove_var(key) };
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    #[test]
    fn complete_package_v2_download_returns_the_package_root() {
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

        let package_dir = download_package_v2_to_local(&root.path().to_string_lossy()).unwrap();

        assert_eq!(package_dir, root.path());
        assert!(package_dir.join("model-package.json").is_file());
        assert!(package_dir.join("layers/layer-00000.gguf").is_file());
        assert!(package_dir.join("layers/layer-00001.gguf").is_file());
    }

    struct EnvRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvRestore {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            restore_env(self.key, self.previous.take());
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
                "activation_width": 4096,
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
    fn layer_package_ref_detects_local_manifest_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-package.json"), "{}").unwrap();

        assert!(is_layer_package_ref(&dir.path().to_string_lossy()));
        assert!(!is_layer_package_ref("/tmp/not-a-package"));
        assert!(is_layer_package_ref("hf://Mesh-LLM/demo-package"));
    }

    #[test]
    fn package_ref_distinguishes_direct_gguf_from_distributable_packages() {
        let direct = StagePackageRef::parse("/models/model.gguf").unwrap();
        assert_eq!(
            direct,
            StagePackageRef::SyntheticDirectGguf(PathBuf::from("/models/model.gguf"))
        );
        assert!(!direct.is_distributable_package());
        assert!(direct.as_package_ref().is_none());

        let hf = StagePackageRef::parse("hf://Mesh-LLM/demo-package@abc123").unwrap();
        assert!(hf.is_distributable_package());
        assert_eq!(
            hf.as_package_ref().as_deref(),
            Some("hf://Mesh-LLM/demo-package@abc123")
        );
    }

    #[test]
    fn safe_manifest_file_path_rejects_escaping_paths() {
        assert_eq!(
            safe_manifest_file_path("shared/metadata.gguf").unwrap(),
            PathBuf::from("shared/metadata.gguf")
        );

        for path in [
            "",
            "/tmp/metadata.gguf",
            "../metadata.gguf",
            "shared/../metadata.gguf",
        ] {
            let error = safe_manifest_file_path(path).unwrap_err().to_string();
            assert!(
                error.contains("manifest file path"),
                "unexpected error for {path:?}: {error}"
            );
        }
    }

    #[test]
    #[serial]
    fn env_restore_preserves_previous_value_after_unwind() {
        const KEY: &str = "MESH_LLM_TEST_ENV_RESTORE_PANIC";
        let _restore_outer = EnvRestore::capture(KEY);
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var(KEY, "before") };

        let panic_result = std::panic::catch_unwind(|| {
            let _restore_inner = EnvRestore::capture(KEY);
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::set_var(KEY, "during") };
            panic!("force unwind after env mutation");
        });

        assert!(panic_result.is_err());
        assert_eq!(std::env::var(KEY).unwrap(), "before");
    }

    #[test]
    fn local_package_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_local_package_files(dir.path(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn local_package_ref_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_hf_package_to_local(&dir.path().to_string_lossy(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn cached_hf_package_verification_treats_missing_artifacts_as_incomplete_cache() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "shared/metadata.gguf" },
                    "embeddings": { "path": "shared/embeddings.gguf" },
                    "output": { "path": "shared/output.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let resolved = verify_cached_hf_package_files(dir.path(), 0, 0, true, false).unwrap();

        assert_eq!(resolved, None);
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_revision_cache_traversal() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let error = resolve_hf_package_to_local("hf://owner/repo@../../evil", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF revision") || error.contains("safe relative path"),
            "{error}"
        );
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_ref_target_cache_traversal() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let refs_dir = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("refs");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), "../../evil").unwrap();

        let error = resolve_hf_package_to_local("hf://owner/repo", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF cache commit hash") || error.contains("safe relative path"),
            "{error}"
        );
    }

    #[test]
    #[serial]
    fn hf_package_resolution_uses_direct_snapshot_revision_cache() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(&snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), snapshot);
    }

    #[test]
    #[serial]
    /// With an explicit pinned revision that has all requested layers, the
    /// cache lookup returns it directly without downloading or scanning other
    /// snapshots.  A stale snapshot with different content must NOT be picked.
    fn pinned_revision_resolves_directly_from_cache() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let repo_cache = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo");

        // Create a complete snapshot at the pinned revision.
        let pinned_snapshot = repo_cache.join("snapshots").join("abc123");
        write_cached_package_snapshot(&pinned_snapshot, sha256_hex(b"layer"));

        // Create a stale snapshot that also has layers — must NOT be used.
        let stale_snapshot = repo_cache.join("snapshots").join("old-stale");
        write_cached_package_snapshot(&stale_snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), pinned_snapshot);
    }

    #[test]
    #[serial]
    fn hf_package_metadata_only_cache_resolution_uses_metadata_integrity_scope() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();
        assert_eq!(PathBuf::from(resolved), snapshot);

        let info = super::super::inspect_stage_package("hf://owner/repo@abc123").unwrap();
        assert_eq!(info.model_id, "model-a");
        assert_eq!(info.layer_count, 1);

        fs::write(snapshot.join("shared/metadata.gguf"), b"metadota").unwrap();
        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(error.contains("shared/metadata.gguf"), "{error}");
    }

    #[test]
    #[serial]
    fn hf_package_resolution_verifies_cached_snapshot_artifact_checksums() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("checksum mismatch"), "{error}");
    }

    /// Integration test: resolves package metadata without downloading layer files from HF.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_metadata_only -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_metadata_only() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request 0 layers — should download manifest/shared metadata, but no layer files.
        let local_path = resolve_hf_package_to_local(package_ref, 0, 0, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(
            manifest.is_file(),
            "manifest should exist at {}",
            manifest.display()
        );

        // Verify manifest is valid JSON with expected fields
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert!(parsed["layers"].as_array().unwrap().len() > 50);

        // Verify the function didn't request any layer downloads
        // (we can't check the cache dir because previous test runs may have cached files)
    }

    /// Integration test: downloads manifest + a single layer file.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_single_layer -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_single_layer() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request just layer 0
        let local_path = resolve_hf_package_to_local(package_ref, 0, 1, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(manifest.is_file());

        // Read manifest to find layer 0's artifact path
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        let layer0_artifact = parsed["layers"][0]["path"].as_str().unwrap();

        // Verify that specific layer file was downloaded
        let layer0_path = std::path::Path::new(&local_path).join(layer0_artifact);
        assert!(
            layer0_path.is_file(),
            "layer 0 should be downloaded at {}",
            layer0_path.display()
        );
        // Should be non-trivial size (layer files are typically > 1 MB)
        let size = std::fs::metadata(&layer0_path).unwrap().len();
        assert!(size > 1_000_000, "layer file should be > 1MB, got {size}");
    }
}
