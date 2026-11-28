//! Transition fixtures: a 0.80.0 GPU product install updated in place by the
//! released updater's own staging and replacement functions.
//!
//! `extract_bundle_archive`, `collect_bundle_files`, `replace_bundle_files`
//! and the helpers they call are unchanged from `v0.80.0`, so these tests
//! show what an already-installed 0.80.0 updater does with a new archive.
//! The new host's product check (`product_runtime_release`) then decides
//! whether the result can start.

use super::*;
use crate::native_runtime_install::{
    current_skippy_abi_version, product_runtime_release, product_runtime_tree_sha256,
};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{
    NativeRuntimeArtifact, NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
};
use std::io::Write as _;

/// Skippy ABI of the runtimes bundled in Mesh 0.80.0
/// (`v0.80.0:skippy/crates/skippy-ffi/src/lib.rs`).
const OLD_ABI: &str = "0.1.80";
const RUNTIME_RELEASE: &str = "0.76.1";

/// Write a product-v2 bundle (host, runtime, product-manifest.json) into
/// `root`, the way `package-release.sh` composes it.
fn write_product(root: &Path, id: &str, abi: &str, backend: NativeRuntimeBackend, host: &[u8]) {
    let runtime = root.join(NATIVE_RUNTIMES_DIR_NAME).join(id);
    std::fs::create_dir_all(runtime.join("lib")).unwrap();
    std::fs::write(runtime.join("lib/libruntime.so"), id.as_bytes()).unwrap();
    NativeRuntimeManifest {
        runtime: NativeRuntimeArtifact {
            id: id.to_string(),
            release_version: Some(RUNTIME_RELEASE.to_string()),
            skippy_abi: abi.to_string(),
            platform: NativeRuntimePlatform {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                target: None,
                min_glibc: cfg!(target_os = "linux").then(|| "2.17".to_string()),
            },
            backend,
            rank: 0,
            libraries: vec!["lib/libruntime.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        },
    }
    .write_to_dir(&runtime)
    .unwrap();
    let manifest_sha = hex::encode(Sha256::digest(
        std::fs::read(runtime.join("manifest.json")).unwrap(),
    ));
    let product = serde_json::json!({
        "schema_version": 2, "contract": "mesh-llm-product-v2",
        "host": {"required_skippy_abi": abi},
        "runtime": {"id": id, "path": format!("{NATIVE_RUNTIMES_DIR_NAME}/{id}"),
            "release_version": RUNTIME_RELEASE, "skippy_abi": abi,
            "manifest_sha256": manifest_sha,
            "sha256": product_runtime_tree_sha256(&runtime).unwrap()},
    });
    std::fs::write(
        root.join(PRODUCT_MANIFEST_NAME),
        serde_json::to_vec(&product).unwrap(),
    )
    .unwrap();
    std::fs::write(root.join(mesh_binary_name()), host).unwrap();
}

/// Zip `dir` under a top-level `mesh-bundle/` directory, as release archives are.
fn zip_dir(dir: &Path, archive: &Path) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(archive).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let rel = current.strip_prefix(dir).unwrap();
        let name = format!("mesh-bundle/{}", rel.to_string_lossy());
        writer
            .add_directory(format!("{}/", name.trim_end_matches('/')), options)
            .unwrap();
        for entry in std::fs::read_dir(&current).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(dir).unwrap();
                writer
                    .start_file(format!("mesh-bundle/{}", rel.to_string_lossy()), options)
                    .unwrap();
                writer.write_all(&std::fs::read(&path).unwrap()).unwrap();
            }
        }
    }
    writer.finish().unwrap();
}

/// Run the released updater's extract → stage → replace sequence against
/// `install`, as `install_latest_bundle` does (minus download and `--version`).
fn apply_with_released_updater(install: &Path, new_product: &Path, work: &Path) {
    std::fs::create_dir_all(work).unwrap();
    let archive = work.join("mesh-llm-x86_64-unknown-linux-gnu-cuda.zip");
    zip_dir(new_product, &archive);
    let extracted = work.join("bundle");
    std::fs::create_dir_all(&extracted).unwrap();
    extract_bundle_archive(&archive, &extracted).unwrap();
    let staged = collect_bundle_files(&extracted, backend::BinaryFlavor::Cuda).unwrap();
    replace_bundle_files(install, &extracted, &work.join("backup"), &staged).unwrap();
}

/// A 0.80.0 CUDA install: host plus its bundled CUDA runtime at the old ABI.
fn old_cuda_install(root: &Path) -> PathBuf {
    let install = root.join("install");
    write_product(
        &install,
        "meshllm-runtime-cuda-old",
        OLD_ABI,
        NativeRuntimeBackend::cuda(12, vec!["sm_89".to_string()]),
        b"old host",
    );
    install
}

fn runtime_bundles(install: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(install.join(NATIVE_RUNTIMES_DIR_NAME))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn full_new_product_replaces_old_gpu_product_coherently() {
    let temp = tempfile::tempdir().unwrap();
    let abi = current_skippy_abi_version();
    assert_ne!(abi, OLD_ABI, "fixture assumes the ABI moved since 0.80.0");
    let install = old_cuda_install(temp.path());
    assert!(
        product_runtime_release(&runtime_bundles(&install), &abi).is_err(),
        "the old product must be incompatible with the new host"
    );

    let new_product = temp.path().join("new");
    write_product(
        &new_product,
        "meshllm-runtime-cpu",
        &abi,
        NativeRuntimeBackend::cpu(),
        b"new host",
    );
    let work = temp.path().join("work");
    apply_with_released_updater(&install, &new_product, &work);

    assert_eq!(
        std::fs::read(install.join(mesh_binary_name())).unwrap(),
        b"new host"
    );
    let bundles = runtime_bundles(&install);
    assert_eq!(
        bundles
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>(),
        vec!["meshllm-runtime-cpu"],
        "the old GPU runtime tree must be replaced, not merged"
    );
    assert_eq!(
        product_runtime_release(&bundles, &abi).unwrap().as_deref(),
        Some(RUNTIME_RELEASE)
    );
}

#[test]
fn host_only_replacement_leaves_stale_product_that_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let abi = current_skippy_abi_version();
    let install = old_cuda_install(temp.path());

    let new_host_only = temp.path().join("new");
    std::fs::create_dir_all(&new_host_only).unwrap();
    std::fs::write(new_host_only.join(mesh_binary_name()), b"new host").unwrap();
    let work = temp.path().join("work");
    apply_with_released_updater(&install, &new_host_only, &work);

    assert_eq!(
        std::fs::read(install.join(mesh_binary_name())).unwrap(),
        b"new host"
    );
    let bundles = runtime_bundles(&install);
    assert_eq!(bundles.len(), 1, "old runtime tree is left in place");
    let error = product_runtime_release(&bundles, &abi).unwrap_err();
    assert!(
        format!("{error:#}").contains("ABI"),
        "stale 0.80.0 product must be rejected for its ABI: {error:#}"
    );
}
