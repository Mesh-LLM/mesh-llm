use super::*;
use serial_test::serial;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(name: &str) -> PathBuf {
    let unique = format!(
        "mesh-llm-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let path = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn test_release_asset(name: impl Into<String>) -> ReleaseAsset {
    ReleaseAsset {
        name: name.into(),
        sha256: "a".repeat(64),
    }
}

#[test]
fn test_version_newer() {
    assert!(version_newer("0.33.1", "0.33.0"));
    assert!(!version_newer("0.33.0", "0.33.0"));
    assert!(!version_newer("0.32.0", "0.33.0"));
    assert!(version_newer("0.33.0", "0.33.0-rc.1"));
    assert!(!version_newer("0.33.0-rc.1", "0.33.0"));
    assert!(version_newer("0.33.0-rc.2", "0.33.0-rc.1"));
    assert!(!version_newer("0.99.0", "0.68.0+gAB131C"));
    assert!(version_newer("0.68.0+gAB131C", "0.99.0"));
    assert!(!version_newer("0.99.0", "0.68.0+gAB131C.dirty"));
    assert!(version_newer("0.68.0+gAB131C.dirty", "0.99.0"));
    assert!(version_newer("0.69.0", "0.68.0"));
    assert!(!version_newer("not-a-version", "0.68.0"));
    assert!(!version_newer("0.68.0", "not-a-version"));
    assert!(!version_newer("not-a-version+gAB131C", "0.99.0"));
    assert!(!version_newer("not-a-version+gAB131C.dirty", "0.99.0"));
    assert!(!version_newer("0.99.0", "not-a-version+gAB131C"));
}

#[test]
#[serial]
fn test_release_asset_url() {
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
    assert_eq!(
        release_asset_url("v0.60.0", "mesh-llm-aarch64-apple-darwin.tar.gz"),
        "https://github.com/Mesh-LLM/mesh-llm/releases/download/v0.60.0/mesh-llm-aarch64-apple-darwin.tar.gz"
    );
}

#[test]
#[serial]
fn test_release_repo_defaults_to_main_repo() {
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
    assert_eq!(release_repo(), "Mesh-LLM/mesh-llm");
    assert_eq!(
        latest_release_api_url(),
        "https://api.github.com/repos/Mesh-LLM/mesh-llm/releases/latest"
    );
}

#[test]
#[serial]
fn test_release_repo_can_be_overridden_for_testing() {
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::set_var(SELF_UPDATE_REPO_ENV, "jdumay/mesh-llm") };
    assert_eq!(release_repo(), "jdumay/mesh-llm");
    assert_eq!(
        latest_release_api_url(),
        "https://api.github.com/repos/jdumay/mesh-llm/releases/latest"
    );
    assert_eq!(
        release_api_url_for_tag("v0.60.0"),
        "https://api.github.com/repos/jdumay/mesh-llm/releases/tags/v0.60.0"
    );
    assert_eq!(
        release_asset_url("v0.60.0", "mesh-llm-x86_64-unknown-linux-gnu.tar.gz"),
        "https://github.com/jdumay/mesh-llm/releases/download/v0.60.0/mesh-llm-x86_64-unknown-linux-gnu.tar.gz"
    );
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::remove_var(SELF_UPDATE_REPO_ENV) };
}

#[test]
fn test_normalize_release_tag() {
    assert_eq!(normalize_release_tag("v0.60.0").unwrap(), "v0.60.0");
    assert_eq!(
        normalize_release_tag("0.60.0-rc.1").unwrap(),
        "v0.60.0-rc.1"
    );
    assert!(normalize_release_tag("latest").is_err());
}

#[test]
fn test_describe_requested_update() {
    assert_eq!(
        describe_requested_update("0.60.0", "0.68.0", false),
        "Updating"
    );
    assert_eq!(
        describe_requested_update("0.68.0", "0.68.0", true),
        "Reinstalling"
    );
    assert_eq!(
        describe_requested_update("0.0.1", "0.68.0", true),
        "Downgrading"
    );
    assert_eq!(
        describe_requested_update("999.0.0", "0.68.0", true),
        "Installing"
    );
}

#[test]
fn test_stable_release_asset_name_matches_platform() {
    let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some((
            backend::BinaryFlavor::Metal,
            "mesh-llm-aarch64-apple-darwin.tar.gz",
        )),
        ("linux", "x86_64") => Some((
            backend::BinaryFlavor::Cpu,
            "mesh-llm-x86_64-unknown-linux-gnu.tar.gz",
        )),
        _ => None,
    };

    let Some((flavor, asset)) = expected else {
        return;
    };
    assert_eq!(
        stable_release_asset_name_for(std::env::consts::OS, std::env::consts::ARCH, flavor),
        Some(asset.to_string())
    );
}

#[test]
fn test_windows_release_asset_names() {
    assert!(platform_has_release_assets_for("windows", "x86_64"));
    assert_eq!(
        stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cpu),
        Some("mesh-llm-x86_64-pc-windows-msvc.zip".to_string())
    );
    assert_eq!(
        stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Cuda),
        Some("mesh-llm-x86_64-pc-windows-msvc-cuda.zip".to_string())
    );
    assert_eq!(
        stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Rocm),
        Some("mesh-llm-x86_64-pc-windows-msvc-rocm.zip".to_string())
    );
    assert_eq!(
        stable_release_asset_name_for("windows", "x86_64", backend::BinaryFlavor::Vulkan),
        Some("mesh-llm-x86_64-pc-windows-msvc-vulkan.zip".to_string())
    );
    let release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![test_release_asset(
            "mesh-llm-v0.60.0-x86_64-pc-windows-msvc.zip",
        )],
    };
    assert!(release_has_any_platform_asset(
        &release, "windows", "x86_64"
    ));
    let empty_release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: Vec::new(),
    };
    assert!(!release_has_any_platform_asset(
        &empty_release,
        "windows",
        "x86_64"
    ));
}

#[test]
fn test_linux_arm64_release_asset_names() {
    let stable_asset = "mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string();
    assert!(platform_has_release_assets_for("linux", "aarch64"));
    assert_eq!(
        stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu),
        Some(stable_asset.clone())
    );

    let published_release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![test_release_asset(stable_asset)],
    };
    assert!(release_has_any_platform_asset(
        &published_release,
        "linux",
        "aarch64"
    ));

    let missing_release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: Vec::new(),
    };
    assert!(!release_has_any_platform_asset(
        &missing_release,
        "linux",
        "aarch64"
    ));
}

#[test]
fn test_linux_arm64_aliases_resolve_identical_release_assets() {
    let arm64_asset = stable_release_asset_name_for("linux", "arm64", backend::BinaryFlavor::Cpu);
    let aarch64_asset =
        stable_release_asset_name_for("linux", "aarch64", backend::BinaryFlavor::Cpu);
    assert_eq!(arm64_asset, aarch64_asset);
    assert_eq!(
        arm64_asset,
        Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
    );
}

#[test]
fn test_resolve_release_asset_name_prefers_stable_linux_arm64_asset() {
    let release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![
            test_release_asset("mesh-llm-aarch64-unknown-linux-gnu.tar.gz"),
            test_release_asset("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz"),
        ],
    };

    assert_eq!(
        resolve_release_asset_name(
            &release,
            ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
            ReleaseAssetPreference::StableFirst,
            CudaBundleOrder::Cuda12First,
        ),
        Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
    );
}

#[test]
fn test_resolve_release_asset_name_falls_back_to_versioned_linux_arm64_asset() {
    let release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![test_release_asset(
            "mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz",
        )],
    };

    assert_eq!(
        resolve_release_asset_name(
            &release,
            ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
            ReleaseAssetPreference::StableFirst,
            CudaBundleOrder::Cuda12First,
        ),
        Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
    );
}

/// A Windows update keeps the runtime packaged in the bundle, and the CUDA
/// 12 bundle stops at SM 90, so a Blackwell host updates to CUDA 13.
#[test]
fn test_resolve_release_asset_name_follows_the_gpu_for_split_windows_cuda_assets() {
    let release = ReleaseInfo {
        tag: "v0.79.0".to_string(),
        version: "0.79.0".to_string(),
        assets: vec![
            test_release_asset("mesh-llm-x86_64-pc-windows-msvc-cuda-12.zip"),
            test_release_asset("mesh-llm-x86_64-pc-windows-msvc-cuda-13.zip"),
            test_release_asset("mesh-llm-v0.79.0-x86_64-pc-windows-msvc-cuda-12.zip"),
            test_release_asset("mesh-llm-v0.79.0-x86_64-pc-windows-msvc-cuda-13.zip"),
        ],
    };
    let target = ReleaseTarget::from_raw("windows", "x86_64", backend::BinaryFlavor::Cuda).unwrap();

    for (gpu_arches, expected) in [
        (&["120"][..], "mesh-llm-x86_64-pc-windows-msvc-cuda-13.zip"),
        (
            &["86", "120"][..],
            "mesh-llm-x86_64-pc-windows-msvc-cuda-13.zip",
        ),
        (&["89"][..], "mesh-llm-x86_64-pc-windows-msvc-cuda-12.zip"),
        (&[][..], "mesh-llm-x86_64-pc-windows-msvc-cuda-12.zip"),
    ] {
        let gpu_arches = gpu_arches.iter().map(|arch| arch.to_string()).collect();
        let cuda_order = cuda_bundle_order(target, &gpu_arches);
        for preference in [
            ReleaseAssetPreference::StableFirst,
            ReleaseAssetPreference::VersionedFirst,
        ] {
            assert_eq!(
                resolve_release_asset_name(&release, target, preference, cuda_order),
                Some(expected.to_string()),
                "{gpu_arches:?}"
            );
        }
    }
}

/// Other platforms install the runtime for the host after updating, so the
/// GPU does not change which bundle they download.
#[test]
fn test_cuda_bundle_order_ignores_the_gpu_outside_windows() {
    let gpu_arches = BTreeSet::from(["120".to_string()]);
    for (os, arch) in [("linux", "x86_64"), ("linux", "aarch64")] {
        let target = ReleaseTarget::from_raw(os, arch, backend::BinaryFlavor::Cuda).unwrap();
        assert_eq!(
            cuda_bundle_order(target, &gpu_arches),
            CudaBundleOrder::Cuda12First
        );
    }
}

#[test]
fn test_resolve_release_asset_name_prefers_versioned_for_explicit_install() {
    let release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![
            test_release_asset("mesh-llm-aarch64-unknown-linux-gnu.tar.gz"),
            test_release_asset("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz"),
        ],
    };

    assert_eq!(
        resolve_release_asset_name(
            &release,
            ReleaseTarget::from_raw("linux", "aarch64", backend::BinaryFlavor::Cpu).unwrap(),
            ReleaseAssetPreference::VersionedFirst,
            CudaBundleOrder::Cuda12First,
        ),
        Some("mesh-llm-v0.60.0-aarch64-unknown-linux-gnu.tar.gz".to_string())
    );
}

#[test]
fn test_resolve_release_asset_name_versioned_first_falls_back_to_stable() {
    let release = ReleaseInfo {
        tag: "v0.60.0".to_string(),
        version: "0.60.0".to_string(),
        assets: vec![test_release_asset(
            "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        )],
    };

    assert_eq!(
        resolve_release_asset_name(
            &release,
            ReleaseTarget::from_raw("linux", "arm64", backend::BinaryFlavor::Cpu).unwrap(),
            ReleaseAssetPreference::VersionedFirst,
            CudaBundleOrder::Cuda12First,
        ),
        Some("mesh-llm-aarch64-unknown-linux-gnu.tar.gz".to_string())
    );
}

#[test]
fn test_path_is_writable_for_temp_dir() {
    let dir = temp_dir("self-update-writable");
    assert!(path_is_writable(&dir));
    assert!(std::fs::read_dir(&dir).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(PATH_WRITE_PROBE_PREFIX)
    }));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_path_is_writable_rejects_non_directory_path() {
    let dir = temp_dir("self-update-writable-file");
    let path = dir.join("mesh-llm");
    std::fs::write(&path, b"binary").unwrap();

    assert!(!path_is_writable(&path));
    assert_eq!(std::fs::read(&path).unwrap(), b"binary");
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
#[test]
fn test_replace_bundle_files_rolls_back_backup_failure() {
    let dir = temp_dir("self-update-backup-rollback");
    let install_dir = dir.join("install");
    let extracted = dir.join("extracted");
    let backup = dir.join("backup");
    std::fs::create_dir_all(&install_dir).unwrap();
    std::fs::create_dir_all(&extracted).unwrap();
    std::fs::create_dir_all(backup.join("sidecar")).unwrap();
    std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
    std::fs::write(install_dir.join("sidecar"), b"old-sidecar").unwrap();

    let err = replace_bundle_files(&install_dir, &extracted, &backup, &["sidecar".to_string()])
        .unwrap_err();

    assert!(err.to_string().contains("Failed to move"));
    assert_eq!(
        std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
        b"old-binary"
    );
    assert_eq!(
        std::fs::read(install_dir.join("sidecar")).unwrap(),
        b"old-sidecar"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
#[test]
fn test_replace_bundle_files_rolls_back_install_failure() {
    let dir = temp_dir("self-update-install-rollback");
    let install_dir = dir.join("install");
    let extracted = dir.join("extracted");
    let backup = dir.join("backup");
    std::fs::create_dir_all(&install_dir).unwrap();
    std::fs::create_dir_all(&extracted).unwrap();
    std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
    std::fs::write(install_dir.join("sidecar"), b"old-sidecar").unwrap();
    std::fs::write(extracted.join("sidecar"), b"new-sidecar").unwrap();

    let staged_files = ["sidecar".to_string(), mesh_binary_name()];
    let err = replace_bundle_files(&install_dir, &extracted, &backup, &staged_files).unwrap_err();

    assert!(err.to_string().contains("Failed to install"));
    assert_eq!(
        std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
        b"old-binary"
    );
    assert_eq!(
        std::fs::read(install_dir.join("sidecar")).unwrap(),
        b"old-sidecar"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
#[test]
fn test_replace_bundle_files_installs_and_backs_up_runtime_tree() {
    let dir = temp_dir("self-update-runtime-tree");
    let install_dir = dir.join("install");
    let extracted = dir.join("extracted");
    let backup = dir.join("backup");
    std::fs::create_dir_all(&install_dir).unwrap();
    std::fs::create_dir_all(installed_runtime_tree(&install_dir).join("lib")).unwrap();
    std::fs::create_dir_all(
        extracted
            .join("native-runtimes")
            .join("runtime")
            .join("lib"),
    )
    .unwrap();
    std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
    std::fs::write(
        installed_runtime_tree(&install_dir)
            .join("lib")
            .join("core.dylib"),
        b"old-runtime",
    )
    .unwrap();
    std::fs::write(install_dir.join(PRODUCT_MANIFEST_NAME), b"old-manifest").unwrap();
    std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
    std::fs::write(extracted.join(PRODUCT_MANIFEST_NAME), b"new-manifest").unwrap();
    std::fs::write(
        extracted
            .join("native-runtimes")
            .join("runtime")
            .join("lib")
            .join("core.dylib"),
        b"new-runtime",
    )
    .unwrap();

    let staged = vec![
        PRODUCT_MANIFEST_NAME.to_string(),
        NATIVE_RUNTIMES_DIR_NAME.to_string(),
        mesh_binary_name(),
    ];
    replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap();

    assert_eq!(
        std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
        b"new-binary"
    );
    assert_eq!(
        std::fs::read(install_dir.join(PRODUCT_MANIFEST_NAME)).unwrap(),
        b"new-manifest"
    );
    assert_eq!(
        std::fs::read(
            installed_runtime_tree(&install_dir)
                .join("lib")
                .join("core.dylib")
        )
        .unwrap(),
        b"new-runtime"
    );
    // Superseded files must be restorable from the backup.
    assert_eq!(
        std::fs::read(
            backup
                .join("native-runtimes")
                .join("runtime")
                .join("lib")
                .join("core.dylib")
        )
        .unwrap(),
        b"old-runtime"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
#[test]
fn test_replace_bundle_files_rolls_back_runtime_tree() {
    let dir = temp_dir("self-update-runtime-rollback");
    let install_dir = dir.join("install");
    let extracted = dir.join("extracted");
    let backup = dir.join("backup");
    std::fs::create_dir_all(&install_dir).unwrap();
    std::fs::create_dir_all(extracted.join("native-runtimes").join("runtime")).unwrap();
    std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
    std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
    std::fs::write(
        extracted
            .join("native-runtimes")
            .join("runtime")
            .join("core.dylib"),
        b"new-runtime",
    )
    .unwrap();

    let staged = vec![
        NATIVE_RUNTIMES_DIR_NAME.to_string(),
        mesh_binary_name(),
        "missing.bin".to_string(),
    ];
    let err = replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap_err();

    assert!(err.to_string().contains("Failed to install"));
    assert_eq!(
        std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
        b"old-binary"
    );
    assert!(
        !install_dir.join(NATIVE_RUNTIMES_DIR_NAME).exists(),
        "runtime tree installed by a failed update must be rolled back"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(not(windows))]
#[test]
fn test_replace_bundle_files_restores_previous_runtime_tree() {
    let dir = temp_dir("self-update-runtime-restore");
    let install_dir = dir.join("install");
    let extracted = dir.join("extracted");
    let backup = dir.join("backup");
    std::fs::create_dir_all(installed_runtime_tree(&install_dir).join("lib")).unwrap();
    std::fs::create_dir_all(extracted.join("native-runtimes").join("runtime")).unwrap();
    std::fs::write(install_dir.join(mesh_binary_name()), b"old-binary").unwrap();
    std::fs::write(
        installed_runtime_tree(&install_dir)
            .join("lib")
            .join("core.dylib"),
        b"old-runtime",
    )
    .unwrap();
    std::fs::write(extracted.join(mesh_binary_name()), b"new-binary").unwrap();
    std::fs::write(
        extracted
            .join("native-runtimes")
            .join("runtime")
            .join("core.dylib"),
        b"new-runtime",
    )
    .unwrap();

    let staged = vec![
        NATIVE_RUNTIMES_DIR_NAME.to_string(),
        mesh_binary_name(),
        "missing.bin".to_string(),
    ];
    let err = replace_bundle_files(&install_dir, &extracted, &backup, &staged).unwrap_err();

    assert!(err.to_string().contains("Failed to install"));
    assert_eq!(
        std::fs::read(install_dir.join(mesh_binary_name())).unwrap(),
        b"old-binary"
    );
    assert_eq!(
        std::fs::read(
            installed_runtime_tree(&install_dir)
                .join("lib")
                .join("core.dylib")
        )
        .unwrap(),
        b"old-runtime"
    );
    assert!(
        !installed_runtime_tree(&install_dir)
            .join("core.dylib")
            .exists(),
        "the failed update's runtime files must not survive rollback"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_collect_bundle_files_accepts_product_v2_layout() {
    let dir = temp_dir("self-update-pv2-collect");
    std::fs::create_dir_all(dir.join("native-runtimes").join("runtime").join("lib")).unwrap();
    std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();
    std::fs::write(dir.join("host-imports.json"), b"{}").unwrap();
    std::fs::write(dir.join(PRODUCT_MANIFEST_NAME), b"{}").unwrap();
    std::fs::write(
        dir.join("native-runtimes")
            .join("runtime")
            .join("core.dylib"),
        b"runtime",
    )
    .unwrap();

    let staged = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap();
    assert_eq!(
        staged,
        vec![
            mesh_binary_name(),
            "host-imports.json".to_string(),
            NATIVE_RUNTIMES_DIR_NAME.to_string(),
            PRODUCT_MANIFEST_NAME.to_string(),
        ]
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_collect_bundle_files_rejects_product_v2_without_manifest() {
    let dir = temp_dir("self-update-pv2-manifest");
    std::fs::create_dir_all(dir.join("native-runtimes").join("runtime")).unwrap();
    std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();

    let err = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap_err();
    assert!(err.to_string().contains(PRODUCT_MANIFEST_NAME));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_collect_bundle_files_rejects_unexpected_directory() {
    let dir = temp_dir("self-update-unexpected-dir");
    std::fs::create_dir_all(dir.join("surprise")).unwrap();
    std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();

    let err = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap_err();
    assert!(err.to_string().contains("Unexpected directory in bundle"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_collect_bundle_files_accepts_legacy_flat_bundle() {
    let dir = temp_dir("self-update-legacy-collect");
    std::fs::write(dir.join(mesh_binary_name()), b"binary").unwrap();
    std::fs::write(dir.join("sidecar.txt"), b"sidecar").unwrap();

    let staged = collect_bundle_files(&dir, backend::BinaryFlavor::Cpu).unwrap();
    assert_eq!(staged, vec![mesh_binary_name(), "sidecar.txt".to_string()]);
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_staged_binary_version_must_match_release() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_dir("self-update-version-check");
    let binary = dir.join(mesh_binary_name());
    std::fs::write(&binary, "#!/bin/sh\necho 'mesh-llm 0.68.0'\n").unwrap();
    let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).unwrap();

    verify_staged_mesh_binary_version(&dir, "0.68.0").unwrap();
    let err = verify_staged_mesh_binary_version(&dir, "0.69.0").unwrap_err();
    assert!(
        err.to_string()
            .contains("contains mesh-llm v0.68.0; refusing to install")
    );
    let _ = std::fs::remove_dir_all(dir);
}
