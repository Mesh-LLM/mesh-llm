//! Disk cache budget policy shared by standalone Skippy and Mesh.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::{L3CacheManager, StoreLimits};

const AUTO_MAX_BUDGET_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiskCacheBudget {
    Off,
    Auto,
    Fixed(u64),
}

/// Parse the public disk cache setting shared by both products.
pub fn parse_disk_budget(value: &str) -> Result<DiskCacheBudget> {
    match value.trim() {
        "off" => Ok(DiskCacheBudget::Off),
        "auto" => Ok(DiskCacheBudget::Auto),
        size => {
            let (digits, multiplier) = [
                ("KiB", 1024_u64),
                ("MiB", 1024_u64.pow(2)),
                ("GiB", 1024_u64.pow(3)),
                ("TiB", 1024_u64.pow(4)),
            ]
            .into_iter()
            .find_map(|(suffix, multiplier)| {
                size.strip_suffix(suffix).map(|digits| (digits, multiplier))
            })
            .context("disk cache size needs an IEC suffix (KiB, MiB, GiB, TiB)")?;
            ensure_positive_digits(digits)?;
            let units = digits
                .parse::<u64>()
                .context("disk cache size is too large")?;
            let bytes = units
                .checked_mul(multiplier)
                .context("disk cache size is too large")?;
            Ok(DiskCacheBudget::Fixed(bytes))
        }
    }
}

fn ensure_positive_digits(digits: &str) -> Result<()> {
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || digits.bytes().all(|byte| byte == b'0')
    {
        bail!("disk cache size must be a positive whole number");
    }
    Ok(())
}

pub fn acquire_disk_cache(
    root: &Path,
    budget: DiskCacheBudget,
    minimum_free_bytes: u64,
) -> Result<Option<L3CacheManager>> {
    if budget == DiskCacheBudget::Off {
        return Ok(None);
    }
    std::fs::create_dir_all(root)
        .with_context(|| format!("create disk prompt-cache root {}", root.display()))?;
    let budget_bytes = match budget {
        DiskCacheBudget::Off => unreachable!(),
        DiskCacheBudget::Auto => auto_budget_bytes(root, minimum_free_bytes)?,
        DiskCacheBudget::Fixed(bytes) => bytes,
    };
    if budget_bytes == 0 {
        return Ok(None);
    }
    Ok(Some(L3CacheManager::acquire(
        root,
        StoreLimits::new(budget_bytes, minimum_free_bytes),
    )?))
}

pub fn auto_budget_bytes(root: &Path, minimum_free_bytes: u64) -> Result<u64> {
    let available = crate::fsinfo::available_bytes(root)?;
    let managed = managed_root_bytes(root)?;
    Ok(auto_budget_from_space(
        available,
        managed,
        minimum_free_bytes,
    ))
}

pub fn auto_budget_from_space(available: u64, managed: u64, minimum_free: u64) -> u64 {
    let capacity_basis = available.saturating_add(managed);
    let twenty_percent = capacity_basis / 5;
    let allocatable = available
        .saturating_sub(minimum_free)
        .saturating_add(managed);
    twenty_percent.min(allocatable).min(AUTO_MAX_BUDGET_BYTES)
}

fn managed_root_bytes(root: &Path) -> Result<u64> {
    let mut total = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("read cache directory {}", directory.display()))?
        {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() {
                bail!(
                    "disk prompt-cache root contains a symlink: {}",
                    entry.path().display()
                );
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_budget_modes_and_iec_sizes() {
        assert_eq!(parse_disk_budget("off").unwrap(), DiskCacheBudget::Off);
        assert_eq!(parse_disk_budget("auto").unwrap(), DiskCacheBudget::Auto);
        assert_eq!(
            parse_disk_budget("32GiB").unwrap(),
            DiskCacheBudget::Fixed(32 * 1024 * 1024 * 1024)
        );
        for invalid in ["0GiB", "32", "32GB", "1.5GiB", "-1GiB"] {
            assert!(parse_disk_budget(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn auto_budget_preserves_minimum_free_and_managed_bytes() {
        let gib = 1024_u64.pow(3);
        assert_eq!(auto_budget_from_space(100 * gib, 0, 16 * gib), 20 * gib);
        assert_eq!(
            auto_budget_from_space(84 * gib, 16 * gib, 16 * gib),
            20 * gib
        );
        assert_eq!(auto_budget_from_space(8 * gib, 0, 16 * gib), 0);
    }
}
