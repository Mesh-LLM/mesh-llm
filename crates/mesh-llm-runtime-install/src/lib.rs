mod cache;
mod discovery;
mod install;
mod manifest;
mod types;

pub use discovery::{
    NATIVE_RUNTIME_BUNDLE_DIR_ENV, discover_local_native_runtimes,
    discover_native_runtime_bundle_dirs,
};
pub use mesh_llm_native_runtime::{
    CachePrunePlan, CandidateEvaluation, CandidateRejection, HostGpuProfile, HostRuntimeProfile,
    InstalledNativeRuntime, NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeArtifact,
    NativeRuntimeCache, NativeRuntimeCacheRoot, NativeRuntimeFlavor, NativeRuntimeFlavorParseError,
    NativeRuntimeLoadPlan, NativeRuntimeManifest, NativeRuntimePruneMode,
    NativeRuntimeReleaseManifest, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, native_runtime_cache_root, select_native_runtime,
};

pub use cache::{
    current_skippy_abi_version, default_native_runtime_cache, host_runtime_profile,
    native_runtime_cache, native_runtime_versions_match_current_sdk,
};
pub use install::install_native_runtime;
pub use manifest::{default_manifest_url, default_release_manifest_url, load_release_manifest};
pub use types::{
    CURRENT_MESH_VERSION, NATIVE_RUNTIME_CACHE_DIR_ENV, NATIVE_RUNTIME_MANIFEST_URL_ENV,
    NativeRuntimeBundleInstallPolicy, NativeRuntimeDownloadProgress,
    NativeRuntimeDownloadProgressCallback, NativeRuntimeInstallOptions,
    NativeRuntimeInstallOutcome, NativeRuntimeInstallStatus, NativeRuntimeManifestOptions,
    NativeRuntimeVerificationPolicy,
};

#[cfg(test)]
pub(crate) use cache::resolve_cache_root;
#[cfg(test)]
pub(crate) use install::{
    bundle_path_matches_explicit_root, emit_download_progress, install_resolved_runtime,
    verify_download_policy_before_fetch,
};
#[cfg(test)]
pub(crate) use manifest::{
    manifest_url, release_manifest_checksum_url, url_without_query,
    verify_release_manifest_checksum,
};

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
    use sha2::Digest;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    static MANIFEST_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn artifact_with_sha(signature: Option<&str>) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: "meshllm-runtime-linux-x86_64-cpu".to_string(),
            mesh_version: Some(CURRENT_MESH_VERSION.to_string()),
            skippy_abi: current_skippy_abi_version(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: Some("x86_64-unknown-linux-gnu".to_string()),
            },
            backend: NativeRuntimeBackend::cpu(),
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: Some("https://example.invalid/runtime.tar.gz".to_string()),
            sha256: Some("a".repeat(64)),
            signature: signature.map(str::to_string),
        }
    }

    #[test]
    fn checksum_policy_requires_sha256() {
        let mut artifact = artifact_with_sha(None);
        artifact.sha256 = None;

        let err = verify_download_policy_before_fetch(
            &artifact,
            NativeRuntimeVerificationPolicy::RequireChecksum,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("missing required sha256"),
            "{err:?}"
        );
    }

    #[test]
    fn download_progress_redacts_url_query_tokens() {
        let captured = Arc::new(Mutex::new(None));
        let captured_for_callback = Arc::clone(&captured);
        let options = NativeRuntimeInstallOptions {
            progress: Some(Arc::new(move |progress| {
                *captured_for_callback.lock().unwrap() = Some(progress);
            })),
            ..Default::default()
        };

        emit_download_progress(
            &artifact_with_sha(None),
            "https://example.invalid/runtime.tar.gz?token=secret",
            10,
            Some(20),
            false,
            &options,
        );

        let progress = captured.lock().unwrap().clone().expect("progress event");
        assert_eq!(progress.url, "https://example.invalid/runtime.tar.gz");
    }

    #[test]
    fn legacy_cached_manifest_does_not_block_valid_bundle_install() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle");
        let profile = host_runtime_profile();
        let mut artifact = artifact_with_sha(None);
        artifact.id = "valid-bundle-runtime".to_string();
        artifact.platform.os = profile.os;
        artifact.platform.arch = profile.arch;
        artifact.platform.target = profile.target_triple;
        artifact.url = None;
        artifact.sha256 = None;
        std::fs::create_dir_all(bundle.join("lib")).unwrap();
        std::fs::write(bundle.join("lib/libllama.so"), b"valid runtime").unwrap();
        NativeRuntimeManifest {
            runtime: artifact.clone(),
        }
        .write_to_dir(&bundle)
        .unwrap();

        let install = |cache_dir: PathBuf| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(install_native_runtime(NativeRuntimeInstallOptions {
                    selection: RuntimeSelection::Id(artifact.id.clone()),
                    bundle_dirs: vec![bundle.clone()],
                    cache_dir: Some(cache_dir),
                    bundle_install_policy:
                        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache,
                    allow_download: false,
                    ..Default::default()
                }))
        };

        install(temp.path().join("fresh-cache"))
            .expect("the valid bundle should install into a fresh cache");

        let polluted_cache = temp.path().join("polluted-cache");
        let legacy_runtime = polluted_cache.join("0.74.0/legacy-cache-runtime");
        std::fs::create_dir_all(legacy_runtime.join("lib")).unwrap();
        std::fs::write(legacy_runtime.join("lib/libllama.so"), b"legacy runtime").unwrap();
        std::fs::write(
            legacy_runtime.join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "legacy-cache-runtime",
    "mesh_version": "0.74.0",
    "skippy_abi": "0.1.25",
    "platform": {"os": "windows", "arch": "x86_64"},
    "backend": {"kind": "vulkan"},
    "libraries": ["lib/libllama.so"]
  }
}"#,
        )
        .unwrap();

        install(polluted_cache)
            .expect("a legacy cached manifest must not block the valid bundle install");
    }

    #[test]
    fn bundled_runtime_is_used_in_place_without_cache_copy() {
        let bundle = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let mut artifact = artifact_with_sha(None);
        artifact.url = None;
        artifact.sha256 = None;
        std::fs::create_dir_all(bundle.path().join("lib")).unwrap();
        std::fs::write(bundle.path().join("lib/libllama.so"), b"runtime").unwrap();
        NativeRuntimeManifest {
            runtime: artifact.clone(),
        }
        .write_to_dir(bundle.path())
        .unwrap();
        let cache = NativeRuntimeCache::new(cache_root.path());
        let resolution = NativeRuntimeResolution {
            selected: artifact.clone(),
            source: NativeRuntimeSource::Bundle {
                path: bundle.path().to_path_buf(),
            },
            evaluated: Vec::new(),
        };

        let outcome = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(install_resolved_runtime(
                &cache,
                resolution,
                &NativeRuntimeInstallOptions {
                    allow_download: false,
                    ..Default::default()
                },
            ))
            .unwrap();

        assert_eq!(outcome.status, NativeRuntimeInstallStatus::AlreadyInstalled);
        assert_eq!(outcome.runtime.path, bundle.path());
        assert!(
            !cache
                .runtime_dir(
                    artifact.mesh_version.as_deref().unwrap(),
                    artifact.native_runtime_id()
                )
                .exists()
        );
    }

    #[test]
    fn bundled_runtime_is_used_in_place_when_policy_has_no_explicit_root_match() {
        let bundle = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let mut artifact = artifact_with_sha(None);
        artifact.url = None;
        artifact.sha256 = None;
        std::fs::create_dir_all(bundle.path().join("lib")).unwrap();
        std::fs::write(bundle.path().join("lib/libllama.so"), b"runtime").unwrap();
        NativeRuntimeManifest {
            runtime: artifact.clone(),
        }
        .write_to_dir(bundle.path())
        .unwrap();
        let cache = NativeRuntimeCache::new(cache_root.path());
        let resolution = NativeRuntimeResolution {
            selected: artifact.clone(),
            source: NativeRuntimeSource::Bundle {
                path: bundle.path().to_path_buf(),
            },
            evaluated: Vec::new(),
        };

        let outcome = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(install_resolved_runtime(
                &cache,
                resolution,
                &NativeRuntimeInstallOptions {
                    bundle_install_policy:
                        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache,
                    allow_download: false,
                    ..Default::default()
                },
            ))
            .unwrap();

        assert_eq!(outcome.status, NativeRuntimeInstallStatus::AlreadyInstalled);
        assert_eq!(outcome.runtime.path, bundle.path());
        assert!(
            !cache
                .runtime_dir(
                    artifact.mesh_version.as_deref().unwrap(),
                    artifact.native_runtime_id()
                )
                .exists()
        );
    }

    #[test]
    fn explicit_product_bundle_root_is_installed_into_cache_when_policy_requires_it() {
        let temp = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let product_bundle = temp.path().join("mesh-bundle");
        let runtime_bundle = product_bundle.join("native-runtimes/runtime-a");
        let mut artifact = artifact_with_sha(None);
        artifact.url = None;
        artifact.sha256 = None;
        std::fs::create_dir_all(runtime_bundle.join("lib")).unwrap();
        std::fs::write(runtime_bundle.join("lib/libllama.so"), b"runtime").unwrap();
        NativeRuntimeManifest {
            runtime: artifact.clone(),
        }
        .write_to_dir(&runtime_bundle)
        .unwrap();
        let cache = NativeRuntimeCache::new(cache_root.path());
        let resolution = NativeRuntimeResolution {
            selected: artifact.clone(),
            source: NativeRuntimeSource::Bundle {
                path: runtime_bundle.clone(),
            },
            evaluated: Vec::new(),
        };

        let outcome = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(install_resolved_runtime(
                &cache,
                resolution,
                &NativeRuntimeInstallOptions {
                    bundle_dirs: vec![product_bundle],
                    bundle_install_policy:
                        NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache,
                    allow_download: false,
                    ..Default::default()
                },
            ))
            .unwrap();

        let cached_path = cache.runtime_dir(
            artifact.mesh_version.as_deref().unwrap(),
            artifact.native_runtime_id(),
        );
        assert_eq!(outcome.status, NativeRuntimeInstallStatus::Installed);
        assert_eq!(outcome.runtime.path, cached_path);
        assert!(outcome.runtime.path.join("lib/libllama.so").exists());
    }

    #[test]
    fn explicit_bundle_root_matching_accepts_runtime_native_runtimes_and_product_roots() {
        let temp = tempfile::tempdir().unwrap();
        let product_bundle = temp.path().join("mesh-bundle");
        let native_runtimes_root = product_bundle.join("native-runtimes");
        let runtime_bundle = native_runtimes_root.join("runtime-a");
        let sibling = temp.path().join("other-bundle");
        std::fs::create_dir_all(&runtime_bundle).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();

        assert!(
            bundle_path_matches_explicit_root(
                &runtime_bundle,
                std::slice::from_ref(&runtime_bundle)
            )
            .unwrap()
        );
        assert!(
            bundle_path_matches_explicit_root(
                &runtime_bundle,
                std::slice::from_ref(&native_runtimes_root)
            )
            .unwrap()
        );
        assert!(
            bundle_path_matches_explicit_root(
                &runtime_bundle,
                std::slice::from_ref(&product_bundle)
            )
            .unwrap()
        );
        assert!(
            !bundle_path_matches_explicit_root(&runtime_bundle, std::slice::from_ref(&sibling))
                .unwrap()
        );
    }

    #[test]
    fn explicit_bundle_root_matching_skips_uncanonicalizable_roots() {
        let temp = tempfile::tempdir().unwrap();
        let product_bundle = temp.path().join("mesh-bundle");
        let runtime_bundle = product_bundle.join("native-runtimes/runtime-a");
        std::fs::create_dir_all(&runtime_bundle).unwrap();

        let matches = bundle_path_matches_explicit_root(
            &runtime_bundle,
            &[temp.path().join("missing"), product_bundle.clone()],
        )
        .unwrap();

        assert!(matches);
    }

    #[test]
    fn modified_release_manifest_is_rejected_before_parsing() {
        let expected = hex::encode(sha2::Sha256::digest(b"expected manifest"));
        let error = verify_release_manifest_checksum(
            b"modified manifest",
            &format!("{expected}  native-runtimes.json"),
        )
        .expect_err("modified release manifest must fail verification");
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }

    #[test]
    fn release_manifest_checksum_url_preserves_query_parameters() {
        assert_eq!(
            release_manifest_checksum_url(
                "https://example.invalid/native-runtimes.json?token=secret"
            ),
            "https://example.invalid/native-runtimes.json.sha256?token=secret"
        );
    }

    #[test]
    fn manifest_diagnostic_urls_redact_query_parameters() {
        assert_eq!(
            url_without_query("https://example.invalid/native-runtimes.json?token=secret"),
            "https://example.invalid/native-runtimes.json"
        );
        assert_eq!(
            url_without_query("https://example.invalid/native-runtimes.json"),
            "https://example.invalid/native-runtimes.json"
        );
    }

    #[test]
    fn manifest_diagnostic_urls_redact_userinfo() {
        let redacted =
            url_without_query("https://user:secret@example.invalid/native-runtimes.json?token=abc");
        assert!(!redacted.contains("secret"), "{redacted}");
        assert!(!redacted.contains("abc"), "{redacted}");
        assert_eq!(
            redacted,
            "https://[REDACTED]@example.invalid/native-runtimes.json"
        );
    }

    #[test]
    fn resolve_cache_root_treats_empty_env_value_as_unset() {
        let empty_env = resolve_cache_root(None, Some(std::ffi::OsString::new())).unwrap();
        let unset_env = resolve_cache_root(None, None).unwrap();
        assert_eq!(empty_env, unset_env);
    }

    #[test]
    fn resolve_cache_root_honours_non_empty_env_value() {
        let root =
            resolve_cache_root(None, Some(std::ffi::OsString::from("/tmp/custom-cache"))).unwrap();
        assert_eq!(root, PathBuf::from("/tmp/custom-cache"));
    }

    #[test]
    fn resolve_cache_root_prefers_explicit_override_over_env() {
        let root = resolve_cache_root(
            Some(Path::new("/tmp/explicit-cache")),
            Some(std::ffi::OsString::from("/tmp/env-cache")),
        )
        .unwrap();
        assert_eq!(root, PathBuf::from("/tmp/explicit-cache"));
    }

    #[test]
    fn matching_release_manifest_checksum_is_accepted() {
        let manifest = b"{\"mesh_version\":\"0.73.1\"}";
        let expected = hex::encode(sha2::Sha256::digest(manifest));
        verify_release_manifest_checksum(manifest, &format!("{expected}  native-runtimes.json"))
            .unwrap();
    }

    #[test]
    fn signature_policy_fails_closed_until_implemented() {
        let artifact = artifact_with_sha(Some("signature"));

        let err = verify_download_policy_before_fetch(
            &artifact,
            NativeRuntimeVerificationPolicy::RequireChecksumAndSignature,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("signature verification is not implemented"),
            "{err:?}"
        );
    }

    #[test]
    fn default_manifest_url_is_skipped_for_bundle_only_resolution() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }

        let options = NativeRuntimeManifestOptions {
            bundle_dirs: vec![PathBuf::from("runtime-bundle")],
            ..Default::default()
        };

        assert!(manifest_url(&options).is_none());
    }

    #[test]
    fn explicit_manifest_url_wins_over_env_and_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/from-env.json",
            );
        }

        let options = NativeRuntimeManifestOptions {
            manifest_url: Some("https://example.invalid/from-arg.json".to_string()),
            ..Default::default()
        };

        assert_eq!(
            manifest_url(&options).as_deref(),
            Some("https://example.invalid/from-arg.json")
        );

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }

    #[test]
    fn env_manifest_url_wins_over_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/from-env.json",
            );
        }

        let url = manifest_url(&NativeRuntimeManifestOptions::default());

        assert_eq!(
            url.as_deref(),
            Some("https://example.invalid/from-env.json")
        );

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }

    #[test]
    fn default_manifest_url_uses_release_download_for_release_builds() {
        assert_eq!(
            default_manifest_url("0.68.0", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.68.0/native-runtimes.json"
        );
    }

    #[test]
    fn default_manifest_url_uses_latest_download_for_sha_builds() {
        assert_eq!(
            default_manifest_url("0.68.0+gAB131C", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
        );
        assert_eq!(
            default_manifest_url("0.68.0+gAB131C.dirty", "0.68.0"),
            "https://github.com/Mesh-LLM/mesh-llm/releases/latest/download/native-runtimes.json"
        );
    }

    #[test]
    fn non_default_mesh_version_request_uses_versioned_release_url() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }

        let options = NativeRuntimeManifestOptions {
            mesh_version: "0.67.0".to_string(),
            allow_default_manifest_url: true,
            ..Default::default()
        };

        assert_eq!(
            manifest_url(&options).as_deref(),
            Some(
                "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.67.0/native-runtimes.json"
            )
        );
    }

    #[test]
    fn current_mesh_version_uses_release_version() {
        assert_eq!(CURRENT_MESH_VERSION, mesh_llm_build_info::RELEASE_VERSION);
    }

    #[test]
    fn sdk_runtime_version_check_requires_exact_mesh_and_skippy_versions() {
        let current_abi = current_skippy_abi_version();
        assert!(native_runtime_versions_match_current_sdk(
            CURRENT_MESH_VERSION,
            &current_abi
        ));
        assert!(!native_runtime_versions_match_current_sdk(
            "0.0.0",
            &current_abi
        ));
        assert!(!native_runtime_versions_match_current_sdk(
            CURRENT_MESH_VERSION,
            "0.0.0"
        ));
    }

    #[test]
    fn load_release_manifest_prefers_explicit_path_over_env_and_default() {
        let _guard = MANIFEST_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(
                NATIVE_RUNTIME_MANIFEST_URL_ENV,
                "https://example.invalid/should-not-be-fetched.json",
            );
        }

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("native-runtimes.json");
        std::fs::write(
            &path,
            format!(
                r#"{{
  "mesh_version": "0.68.0",
  "skippy_abi": "{}",
  "artifacts": []
}}"#,
                current_skippy_abi_version()
            ),
        )
        .unwrap();

        let manifest = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(load_release_manifest(NativeRuntimeManifestOptions {
                mesh_version: "0.0.0+gLOCAL".to_string(),
                manifest_path: Some(path),
                manifest_url: Some("https://example.invalid/from-arg.json".to_string()),
                bundle_dirs: Vec::new(),
                allow_default_manifest_url: true,
            }))
            .unwrap();

        assert_eq!(manifest.mesh_version, "0.68.0");
        assert!(manifest.artifacts.is_empty());

        unsafe {
            std::env::remove_var(NATIVE_RUNTIME_MANIFEST_URL_ENV);
        }
    }
}
