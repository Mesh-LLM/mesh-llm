//! Cache path policy for standalone Skippy.
//!
//! Resolution order mirrors the original CLI behavior: an explicit flag
//! wins, then the `SKIPPY_*` environment override, then the platform cache
//! directory.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Model artifact cache root: `--cache-dir` override, then
/// `SKIPPY_MODEL_CACHE_DIR`, then the platform Skippy model cache.
pub fn model_cache_dir(explicit: Option<PathBuf>) -> Result<PathBuf> {
    explicit
        .or_else(|| {
            std::env::var_os("SKIPPY_MODEL_CACHE_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| dirs::cache_dir().map(|p| p.join("skippy/models")))
        .context("cannot determine model cache directory; supply models --cache-dir")
}

/// Native runtime cache root default: `SKIPPY_NATIVE_RUNTIME_CACHE_DIR`, then
/// the platform cache directory (falling back to `~/.cache`).
pub fn native_runtime_cache_default() -> Result<PathBuf> {
    match std::env::var_os("SKIPPY_NATIVE_RUNTIME_CACHE_DIR").filter(|v| !v.is_empty()) {
        Some(path) => Ok(path.into()),
        None => Ok(dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|p| p.join(".cache")))
            .context("cannot determine Skippy runtime cache directory; supply --runtime-cache")?
            .join("skippy")
            .join("native-runtimes")),
    }
}

/// Additional native runtime bundle search roots parsed from
/// `SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR` (a `PATH`-style list).
pub fn native_runtime_bundle_dirs_from_env() -> Vec<PathBuf> {
    std::env::var_os("SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR")
        .map(|paths| {
            std::env::split_paths(&paths)
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cache_dir_explicit_wins() {
        assert_eq!(
            model_cache_dir(Some(PathBuf::from("chosen"))).unwrap(),
            PathBuf::from("chosen")
        );
    }
}
