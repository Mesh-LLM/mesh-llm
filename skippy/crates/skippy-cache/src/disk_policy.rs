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
    match budget {
        DiskCacheBudget::Off | DiskCacheBudget::Fixed(0) => Ok(None),
        DiskCacheBudget::Auto => L3CacheManager::acquire_auto(root, minimum_free_bytes),
        DiskCacheBudget::Fixed(bytes) => Ok(Some(L3CacheManager::acquire(
            root,
            StoreLimits::new(bytes, minimum_free_bytes),
        )?)),
    }
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

    struct TestDirectory(std::path::PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            Self(std::env::temp_dir().join(format!(
                "skippy-auto-budget-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

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

    #[test]
    fn auto_reacquire_preserves_live_owner_budget() {
        let directory = TestDirectory::new();
        let owner = acquire_disk_cache(directory.path(), DiskCacheBudget::Auto, 0)
            .unwrap()
            .expect("temporary filesystem has cache capacity");
        // An owner may adjust its budget during its lifetime. Later automatic
        // stage attachments must join that owner rather than overwrite it
        // with a fresh free-space estimate.
        owner.update_limits(StoreLimits::new(1024, 0)).unwrap();
        let attached = acquire_disk_cache(directory.path(), DiskCacheBudget::Auto, 0)
            .unwrap()
            .unwrap();
        assert_eq!(attached.limits(), owner.limits());
        assert_eq!(attached.limits().budget_bytes, 1024);
    }

    #[test]
    fn concurrent_first_auto_acquisitions_share_canonical_root_owner() {
        let directory = TestDirectory::new();
        let barrier = std::sync::Barrier::new(8);
        let owners = std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|index| {
                    let barrier = &barrier;
                    let root = if index % 2 == 0 {
                        directory.path().to_path_buf()
                    } else {
                        directory.path().join(".")
                    };
                    scope.spawn(move || {
                        barrier.wait();
                        acquire_disk_cache(&root, DiskCacheBudget::Auto, 0)
                            .unwrap()
                            .expect("temporary filesystem has cache capacity")
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let canonical_root = std::fs::canonicalize(directory.path()).unwrap();
        for owner in &owners {
            assert!(owner.shares_root_with(&owners[0]));
            assert_eq!(owner.limits(), owners[0].limits());
            assert_eq!(owner.root(), canonical_root.as_path());
        }
        drop(owners);
        let reopened = acquire_disk_cache(directory.path(), DiskCacheBudget::Auto, 0)
            .unwrap()
            .expect("dropping every attachment releases the physical root lock");
        assert_eq!(reopened.root(), canonical_root.as_path());
    }

    #[test]
    fn concurrent_auto_attachments_share_live_limits_and_enforce_free_space_policy() {
        let directory = TestDirectory::new();
        let owner = acquire_disk_cache(directory.path(), DiskCacheBudget::Auto, 0)
            .unwrap()
            .unwrap();
        owner.update_limits(StoreLimits::new(1024, 0)).unwrap();
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let barrier = &barrier;
                    let root = directory.path();
                    scope.spawn(move || {
                        barrier.wait();
                        acquire_disk_cache(root, DiskCacheBudget::Auto, 0)
                            .unwrap()
                            .unwrap()
                    })
                })
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap().limits(), owner.limits());
            }
        });
        assert!(acquire_disk_cache(directory.path(), DiskCacheBudget::Auto, 1).is_err());
        assert!(acquire_disk_cache(directory.path(), DiskCacheBudget::Fixed(2048), 0).is_err());
        assert!(
            acquire_disk_cache(directory.path(), DiskCacheBudget::Off, 0)
                .unwrap()
                .is_none()
        );
    }
}
