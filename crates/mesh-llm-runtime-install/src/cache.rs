//! Native runtime cache and version helpers.

use crate::types::{CURRENT_MESH_VERSION, NATIVE_RUNTIME_CACHE_DIR_ENV};
use anyhow::{Context, Result};
use mesh_llm_native_runtime::{HostRuntimeProfile, NativeRuntimeCache};
use std::ffi::OsString;
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

/// Resolves the cache root from an explicit override, then the environment
/// variable, then the platform default. An env value that is set but empty
/// is treated the same as unset rather than resolving to a CWD-relative
/// empty path.
pub(crate) fn resolve_cache_root(
    cache_dir: Option<&Path>,
    env_value: Option<OsString>,
) -> Result<PathBuf> {
    if let Some(path) = cache_dir {
        return Ok(path.to_path_buf());
    }
    match env_value {
        Some(path) if !path.is_empty() => Ok(PathBuf::from(path)),
        _ => dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
            .context("cannot determine native runtime cache directory")
            .map(|dir| dir.join("mesh-llm").join("native-runtimes")),
    }
}

pub fn native_runtime_cache(cache_dir: Option<&Path>) -> Result<NativeRuntimeCache> {
    let root = resolve_cache_root(cache_dir, std::env::var_os(NATIVE_RUNTIME_CACHE_DIR_ENV))?;
    Ok(NativeRuntimeCache::new(root))
}

pub fn host_runtime_profile() -> HostRuntimeProfile {
    mesh_llm_hardware_profile::host_runtime_profile()
}
