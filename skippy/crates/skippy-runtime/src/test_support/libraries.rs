//! Selects a verified native runtime load plan for tests.

use anyhow::{Context, Result};
use skippy_native_runtime::{InstalledNativeRuntime, NativeRuntimeLoadPlan, NativeRuntimeManifest};
use std::path::{Path, PathBuf};

/// Resolves a verified load plan from a runtime bundle or bundle-set directory.
pub fn discover_load_plan(bundle_dir: &Path) -> Result<NativeRuntimeLoadPlan> {
    let candidates = runtime_directories(bundle_dir)?;
    for path in candidates {
        let Ok(manifest) = NativeRuntimeManifest::read_from_dir(&path) else {
            continue;
        };
        let runtime = InstalledNativeRuntime {
            release_version: manifest
                .runtime
                .release_version
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            native_runtime_id: manifest.runtime.id.clone(),
            flavor: manifest.runtime.backend.kind.to_string(),
            path,
            manifest,
        };
        if let Ok(plan) = runtime.load_plan() {
            return Ok(plan);
        }
    }
    Err(anyhow::anyhow!(
        "no compatible verified runtime load plan found under {}",
        bundle_dir.display()
    ))
}

fn runtime_directories(bundle_dir: &Path) -> Result<Vec<PathBuf>> {
    if bundle_dir.join("manifest.json").is_file() {
        return Ok(vec![bundle_dir.to_path_buf()]);
    }
    let entries = std::fs::read_dir(bundle_dir).with_context(|| {
        format!(
            "read native runtime bundle directory {}",
            bundle_dir.display()
        )
    })?;
    let mut directories = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    directories.sort();
    Ok(directories)
}
