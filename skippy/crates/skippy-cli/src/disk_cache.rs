//! Resolve standalone disk cache settings before entering the shared Skippy lifecycle.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use skippy_api::serving::LocalDiskCacheOptions;
use skippy_cache::disk_policy::{DiskCacheBudget, parse_disk_budget};

pub(crate) fn from_public_settings(
    mode: Option<&str>,
    directory: Option<PathBuf>,
    minimum_free: Option<&str>,
) -> Result<Option<LocalDiskCacheOptions>> {
    let public_mode = mode
        .map(str::to_owned)
        .or(env_utf8("MESH_LLM_KV_CACHE_DISK")?);
    let legacy_directory = std::env::var_os("SKIPPY_L3_DIR").map(PathBuf::from);
    let budget = match public_mode {
        Some(mode) => parse_disk_budget(&mode)?,
        None if legacy_directory.is_some() => {
            let bytes = env_utf8("SKIPPY_L3_BUDGET_BYTES")?
                .map(|value| value.parse::<u64>())
                .transpose()
                .context("SKIPPY_L3_BUDGET_BYTES must be a byte count")?
                .filter(|bytes| *bytes != 0)
                .unwrap_or(32 * 1024 * 1024 * 1024);
            DiskCacheBudget::Fixed(bytes)
        }
        None => DiskCacheBudget::Off,
    };
    if budget == DiskCacheBudget::Off {
        return Ok(None);
    }
    let directory = directory
        .or_else(|| std::env::var_os("MESH_LLM_KV_CACHE_DISK_DIR").map(PathBuf::from))
        .or(legacy_directory)
        .unwrap_or_else(|| {
            std::env::var_os("MESH_LLM_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".mesh-llm"))
                .join("kv-cache")
        });
    if !directory.is_absolute() {
        bail!(
            "disk cache directory must be absolute: {}",
            directory.display()
        );
    }
    let minimum_free_bytes = minimum_free
        .map(str::to_owned)
        .or(env_utf8("MESH_LLM_KV_CACHE_MIN_FREE")?)
        .map(|value| parse_disk_budget(&value))
        .transpose()?
        .map_or(16 * 1024 * 1024 * 1024, |budget| match budget {
            DiskCacheBudget::Fixed(bytes) => bytes,
            _ => 0,
        });
    if minimum_free_bytes < 1024 * 1024 * 1024 {
        bail!("disk cache minimum free space must be at least 1GiB");
    }
    Ok(Some(LocalDiskCacheOptions {
        directory,
        budget,
        minimum_free_bytes,
    }))
}

fn env_utf8(name: &str) -> Result<Option<String>> {
    std::env::var_os(name)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow::anyhow!("{name} must contain valid UTF-8"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_settings_resolve_mesh_compatible_budget() {
        let directory = tempfile::tempdir().unwrap();
        let settings = from_public_settings(
            Some("32GiB"),
            Some(directory.path().to_path_buf()),
            Some("1GiB"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            settings.budget,
            DiskCacheBudget::Fixed(32 * 1024 * 1024 * 1024)
        );
        assert_eq!(settings.minimum_free_bytes, 1024 * 1024 * 1024);
    }
}
