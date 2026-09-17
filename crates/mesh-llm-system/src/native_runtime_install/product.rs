//! Release selection from a composed product's verified adjacent runtime.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use skippy_native_runtime::NativeRuntimeManifest;
use std::fs;
use std::path::{Component, PathBuf};

#[derive(Deserialize)]
struct ProductManifest {
    schema_version: u32,
    contract: String,
    host: Host,
    runtime: Runtime,
}

#[derive(Deserialize)]
struct Host {
    required_skippy_abi: String,
}

#[derive(Deserialize)]
struct Runtime {
    id: String,
    path: String,
    release_version: String,
    skippy_abi: String,
    manifest_sha256: String,
}

/// Read only product manifests adjacent to discovered `native-runtimes` roots.
/// Explicit release pins must bypass this default-selection policy.
/// Payload checksums, manifest digest, path and identity must all agree before
/// the product can override the compiled fallback release.
pub fn product_runtime_release(
    bundle_dirs: &[PathBuf],
    required_abi: &str,
) -> Result<Option<String>> {
    let mut selected: Option<(String, PathBuf)> = None;
    for bundle in bundle_dirs {
        let Some(native_root) = bundle.parent() else {
            continue;
        };
        if native_root
            .file_name()
            .is_none_or(|name| name != "native-runtimes")
        {
            continue;
        }
        let Some(root) = native_root.parent() else {
            continue;
        };
        let path = root.join("product-manifest.json");
        if !path.exists() {
            continue;
        }
        let product: ProductManifest = serde_json::from_slice(
            &fs::read(&path)
                .with_context(|| format!("read product manifest {}", path.display()))?,
        )
        .with_context(|| format!("parse product manifest {}", path.display()))?;
        if product.schema_version != 2 || product.contract != "mesh-llm-product-v2" {
            bail!(
                "unsupported product manifest generation: {}",
                path.display()
            );
        }
        let runtime = product.runtime;
        semver::Version::parse(&runtime.release_version)
            .with_context(|| format!("invalid product runtime release in {}", path.display()))?;
        if product.host.required_skippy_abi != required_abi || runtime.skippy_abi != required_abi {
            bail!(
                "product runtime ABI does not match host-required ABI {required_abi}: {}",
                path.display()
            );
        }
        let relative = std::path::Path::new(&runtime.path);
        if relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        {
            bail!("unsafe product runtime path in {}", path.display());
        }
        let declared = root
            .join(relative)
            .canonicalize()
            .with_context(|| format!("resolve product runtime path in {}", path.display()))?;
        if declared != bundle.canonicalize()? || !declared.starts_with(root.canonicalize()?) {
            bail!(
                "product runtime path does not identify its adjacent bundle: {}",
                path.display()
            );
        }
        let manifest_path = declared.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path)?;
        if hex::encode(Sha256::digest(&manifest_bytes)) != runtime.manifest_sha256 {
            bail!(
                "product runtime manifest checksum mismatch: {}",
                path.display()
            );
        }
        let manifest = NativeRuntimeManifest::read_from_dir(&declared)?;
        if manifest.runtime.id != runtime.id
            || manifest.runtime.mesh_version.as_deref() != Some(runtime.release_version.as_str())
            || manifest.runtime.skippy_abi != runtime.skippy_abi
        {
            bail!(
                "product runtime identity disagrees with bundled artifact (requested release {}): {}",
                runtime.release_version,
                path.display()
            );
        }
        if let Some((release, previous)) = &selected {
            if release != &runtime.release_version {
                bail!(
                    "conflicting product runtime releases in {} and {}",
                    previous.display(),
                    path.display()
                );
            }
        } else {
            selected = Some((runtime.release_version, path));
        }
    }
    Ok(selected.map(|(release, _)| release))
}
