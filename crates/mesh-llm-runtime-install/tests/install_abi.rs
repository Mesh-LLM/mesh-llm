use mesh_llm_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
use mesh_llm_runtime_install::{
    CURRENT_MESH_VERSION, CandidateRejection, NativeRuntimeArtifact,
    NativeRuntimeBundleInstallPolicy, NativeRuntimeCache, NativeRuntimeInstallOptions,
    NativeRuntimeManifest, NativeRuntimeReleaseManifest, NativeRuntimeResolutionError,
    current_skippy_abi_version, host_runtime_profile, install_native_runtime,
};
use std::path::Path;

const CATALOG_ABI: &str = "0.0.0";

fn write_bundle(root: &Path, id: &str, version: &str, abi: &str) -> NativeRuntimeArtifact {
    let profile = host_runtime_profile();
    let min_glibc = (profile.os == "linux").then(|| "2.17".to_string());
    let artifact = NativeRuntimeArtifact {
        id: id.to_string(),
        mesh_version: Some(version.to_string()),
        skippy_abi: abi.to_string(),
        platform: NativeRuntimePlatform {
            os: profile.os,
            arch: profile.arch,
            target: profile.target_triple,
            min_glibc,
        },
        backend: NativeRuntimeBackend::cpu(),
        rank: 0,
        libraries: vec!["lib/runtime.bin".to_string()],
        files: Default::default(),
        tools: Default::default(),
        url: None,
        sha256: None,
        signature: None,
    };
    std::fs::create_dir_all(root.join("lib")).unwrap();
    std::fs::write(root.join("lib/runtime.bin"), b"fixture runtime").unwrap();
    NativeRuntimeManifest {
        runtime: artifact.clone(),
    }
    .write_to_dir(root)
    .unwrap();
    artifact
}

fn install_options(root: &Path, version: &str) -> NativeRuntimeInstallOptions {
    assert_ne!(CATALOG_ABI, current_skippy_abi_version());
    let old_bundle = root.join("old-bundle");
    let mut old = write_bundle(&old_bundle, "catalog-runtime", version, CATALOG_ABI);
    // Make the stale entry win on rank if compatibility trusts the catalog.
    old.rank = 1000;
    NativeRuntimeManifest {
        runtime: old.clone(),
    }
    .write_to_dir(&old_bundle)
    .unwrap();
    let cache_dir = root.join("cache");
    NativeRuntimeCache::new(&cache_dir)
        .install_from_dir(&old_bundle)
        .unwrap();
    let manifest_path = root.join("catalog.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&NativeRuntimeReleaseManifest {
            mesh_version: version.to_string(),
            skippy_abi: CATALOG_ABI.to_string(),
            artifacts: vec![old],
        })
        .unwrap(),
    )
    .unwrap();
    NativeRuntimeInstallOptions {
        mesh_version: version.to_string(),
        manifest_path: Some(manifest_path),
        cache_dir: Some(cache_dir),
        bundle_install_policy: NativeRuntimeBundleInstallPolicy::InstallExplicitBundlesIntoCache,
        allow_download: false,
        ..Default::default()
    }
}

fn run_install(
    options: NativeRuntimeInstallOptions,
) -> anyhow::Result<mesh_llm_runtime_install::NativeRuntimeInstallOutcome> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(install_native_runtime(options))
}

#[test]
fn current_install_uses_matching_bundle_despite_stale_catalog_and_cache() {
    let temp = tempfile::tempdir().unwrap();
    let mut options = install_options(temp.path(), CURRENT_MESH_VERSION);
    let bundle = temp.path().join("current-bundle");
    let abi = current_skippy_abi_version();
    write_bundle(&bundle, "current-runtime", CURRENT_MESH_VERSION, &abi);
    options.bundle_dirs.push(bundle);
    let outcome = run_install(options).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "current-runtime");
    assert_eq!(outcome.runtime.manifest.runtime.skippy_abi, abi);
    assert!(outcome.runtime.path.starts_with(temp.path().join("cache")));
    assert!(outcome.resolution.evaluated.iter().any(|candidate| {
        candidate.artifact.id == "catalog-runtime"
            && candidate
                .rejection_reasons
                .contains(&CandidateRejection::SkippyAbiMismatch {
                    expected: abi.clone(),
                    actual: CATALOG_ABI.to_string(),
                })
    }));
}

#[test]
fn current_install_rejects_stale_catalog_instead_of_returning_cached_runtime() {
    let temp = tempfile::tempdir().unwrap();
    let error = run_install(install_options(temp.path(), CURRENT_MESH_VERSION)).unwrap_err();
    let rejection = error
        .downcast_ref::<NativeRuntimeResolutionError>()
        .unwrap();
    assert!(rejection.candidates.iter().any(|candidate| {
        candidate
            .reasons
            .contains(&CandidateRejection::SkippyAbiMismatch {
                expected: current_skippy_abi_version(),
                actual: CATALOG_ABI.to_string(),
            })
    }));
}

#[test]
fn explicit_abi_override_is_preserved_for_current_version() {
    let temp = tempfile::tempdir().unwrap();
    let mut options = install_options(temp.path(), CURRENT_MESH_VERSION);
    options.skippy_abi_version = Some(CATALOG_ABI.to_string());
    let outcome = run_install(options).unwrap();
    assert_eq!(outcome.runtime.manifest.runtime.skippy_abi, CATALOG_ABI);
}

#[test]
fn staging_another_version_preserves_its_catalog_abi() {
    let temp = tempfile::tempdir().unwrap();
    let version = "0.0.0";
    assert_ne!(version, CURRENT_MESH_VERSION);
    let outcome = run_install(install_options(temp.path(), version)).unwrap();
    assert_eq!(outcome.runtime.manifest.runtime.skippy_abi, CATALOG_ABI);
    assert_eq!(outcome.runtime.mesh_version, version);
}
