//! Native runtime cache and version helpers.

use crate::types::{CURRENT_MESH_VERSION, NATIVE_RUNTIME_CACHE_DIR_ENV};
use anyhow::{Context, Result};
use mesh_llm_native_runtime::{HostRuntimeProfile, NativeRuntimeCache};
use std::path::{Path, PathBuf};
pub fn current_skippy_abi_version() -> String {
    format!(
        "{}.{}.{}",
        skippy_ffi::ABI_VERSION_MAJOR,
        skippy_ffi::ABI_VERSION_MINOR,
        skippy_ffi::ABI_VERSION_PATCH
    )
}

/// Returns whether native-runtime metadata matches the exact MeshLLM and
/// Skippy ABI versions linked into this SDK build.
pub fn native_runtime_versions_match_current_sdk(mesh_version: &str, skippy_abi: &str) -> bool {
    mesh_version == CURRENT_MESH_VERSION && skippy_abi == current_skippy_abi_version()
}

pub fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
    native_runtime_cache(None)
}

pub fn native_runtime_cache(cache_dir: Option<&Path>) -> Result<NativeRuntimeCache> {
    let root = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => match std::env::var_os(NATIVE_RUNTIME_CACHE_DIR_ENV) {
            Some(path) => PathBuf::from(path),
            None => dirs::cache_dir()
                .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
                .context("cannot determine native runtime cache directory")?
                .join("mesh-llm")
                .join("native-runtimes"),
        },
    };
    Ok(NativeRuntimeCache::new(root))
}

pub fn host_runtime_profile() -> HostRuntimeProfile {
    mesh_llm_hardware_profile::host_runtime_profile()
}
