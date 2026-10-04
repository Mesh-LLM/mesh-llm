//! Canonical cache-root registration, owner attachment, and store acquisition.

use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, RwLock, Weak},
};

use anyhow::{Context, Result, bail};

use super::{L3Activity, L3BenefitAdmission, L3CacheManager, L3EffectiveStatus, L3ManagerInner};
use crate::{HandoffSegmentStore, StoreLimits};

static ROOT_MANAGERS: LazyLock<Mutex<BTreeMap<PathBuf, Weak<L3ManagerInner>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

#[derive(Clone, Copy)]
enum AcquisitionBudget {
    Fixed(StoreLimits),
    Auto { minimum_free_bytes: u64 },
}

impl AcquisitionBudget {
    fn accepts(self, limits: StoreLimits) -> bool {
        match self {
            Self::Fixed(expected) => limits == expected,
            Self::Auto { minimum_free_bytes } => limits.minimum_free_bytes == minimum_free_bytes,
        }
    }

    fn new_owner_limits(self, root: &Path) -> Result<Option<StoreLimits>> {
        match self {
            Self::Fixed(limits) => Ok(Some(limits)),
            Self::Auto { minimum_free_bytes } => {
                let budget_bytes = crate::disk_policy::auto_budget_bytes(root, minimum_free_bytes)?;
                Ok((budget_bytes > 0).then(|| StoreLimits::new(budget_bytes, minimum_free_bytes)))
            }
        }
    }
}

pub(super) fn acquire(root: &Path, limits: StoreLimits) -> Result<L3CacheManager> {
    Ok(acquire_with_budget(root, AcquisitionBudget::Fixed(limits))?
        .expect("a fixed budget always acquires an owner or returns an error"))
}

pub(super) fn acquire_auto(root: &Path, minimum_free_bytes: u64) -> Result<Option<L3CacheManager>> {
    acquire_with_budget(root, AcquisitionBudget::Auto { minimum_free_bytes })
}

fn acquire_with_budget(root: &Path, budget: AcquisitionBudget) -> Result<Option<L3CacheManager>> {
    let root = canonical_cache_root(root)?;
    let mut managers = ROOT_MANAGERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // An entry whose strong count has reached zero may still be dropping its
    // physical store lock on another thread.
    let expiring_owner = managers
        .get(&root)
        .is_some_and(|manager| manager.strong_count() == 0);
    managers.retain(|_, manager| manager.strong_count() > 0);
    if let Some(inner) = managers.get(&root).and_then(Weak::upgrade) {
        if !budget.accepts(inner.store.limits()) {
            bail!(
                "cache root {} is already open with different limits",
                root.display()
            );
        }
        return Ok(Some(L3CacheManager { inner }));
    }

    // Estimate automatic capacity only after excluding a live owner, and keep
    // the registry locked through creation so concurrent callers join it.
    let Some(limits) = budget.new_owner_limits(&root)? else {
        return Ok(None);
    };
    let manager = create_manager(&root, limits, expiring_owner)?;
    managers.insert(root, Arc::downgrade(&manager.inner));
    Ok(Some(manager))
}

fn create_manager(
    root: &Path,
    limits: StoreLimits,
    expiring_owner: bool,
) -> Result<L3CacheManager> {
    let store = Arc::new(open_store_for_acquire(root, limits, expiring_owner)?);
    let reconciliation = store.reconcile_startup()?;
    let inner = Arc::new(L3ManagerInner {
        store,
        activity: Arc::new(L3Activity::default()),
        fill_claims: Arc::new(Mutex::new(std::collections::BTreeSet::new())),
        record_claims: Mutex::new(BTreeMap::new()),
        effective: Mutex::new(L3EffectiveStatus::default()),
        transitions: Mutex::new(VecDeque::new()),
        operations: RwLock::new(()),
        reconciliation,
        benefit_admission: Mutex::new(L3BenefitAdmission::default()),
    });
    Ok(L3CacheManager { inner })
}

/// Retry briefly while the previous in-process owner releases its root lock.
fn open_store_for_acquire(
    root: &Path,
    limits: StoreLimits,
    expiring_owner: bool,
) -> Result<HandoffSegmentStore> {
    const HANDOFF_ATTEMPTS: u32 = 20;
    const HANDOFF_BACKOFF: std::time::Duration = std::time::Duration::from_millis(5);

    let attempts = if expiring_owner { HANDOFF_ATTEMPTS } else { 1 };
    let mut last = None;
    for attempt in 0..attempts {
        match HandoffSegmentStore::open_unreconciled_with_limits(root, limits) {
            Ok(store) => return Ok(store),
            Err(error) => {
                last = Some(error);
                if attempt + 1 < attempts {
                    std::thread::sleep(HANDOFF_BACKOFF);
                }
            }
        }
    }
    Err(last.expect("at least one attempt was made"))
}

fn canonical_cache_root(root: &Path) -> Result<PathBuf> {
    if !root.is_absolute() {
        bail!("cache root must be absolute: {}", root.display());
    }
    crate::fsinfo::refuse_symlink(root)?;
    crate::fsinfo::create_dir_all_without_links(root)
        .with_context(|| format!("failed to create cache root {}", root.display()))?;
    fs::canonicalize(root)
        .with_context(|| format!("failed to resolve cache root {}", root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(name: &str) -> PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-l3-acquisition-{name}-{}-{timestamp}",
            std::process::id()
        ))
    }

    #[test]
    fn one_live_manager_owns_each_root() {
        let root = test_root("owner");
        let limits = StoreLimits::new(1_000_000, 0);
        let first = L3CacheManager::acquire(&root, limits).expect("first manager");
        let second = L3CacheManager::acquire(&root, limits).expect("shared manager");

        assert!(first.shares_root_with(&second));
        assert_eq!(first.root(), second.root());
        assert!(
            L3CacheManager::acquire(&root, StoreLimits::new(2_000_000, 0)).is_err(),
            "one root accepted contradictory budgets"
        );
        drop(first);
        drop(second);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn acquisition_rejects_symlink_roots_and_parents_including_dot_aliases() {
        let directory = test_root("redirected-parent");
        let target = directory.join("target");
        let redirected = directory.join("redirected");
        fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, &redirected).unwrap();
        for root in [
            redirected.clone(),
            redirected.join("."),
            redirected.join("cache"),
            redirected.join("cache").join("."),
        ] {
            let error = L3CacheManager::acquire(&root, StoreLimits::new(1024, 0)).unwrap_err();
            assert!(format!("{error:#}").contains("symlink"), "{error:#}");
            assert!(!target.join("cache").exists());
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
