//! Node-scoped ownership for the durable L3 cache root.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::{
    l3::{HandoffSegmentStore, StoreLimits, StoreReconciliation, StoreUsage},
    tier::L3Tier,
};

static ROOT_MANAGERS: LazyLock<Mutex<BTreeMap<PathBuf, Weak<L3ManagerInner>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// What every stage attached to the node's L3 root has done since open.
#[derive(Debug, Default)]
pub(crate) struct L3Activity {
    pub(crate) fills: AtomicU64,
    pub(crate) hits: AtomicU64,
    pub(crate) misses: AtomicU64,
    pub(crate) writes: AtomicU64,
    pub(crate) geometry_rejected: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) bytes_written: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl L3Activity {
    pub(crate) fn record_error(&self, error: &anyhow::Error) {
        *self
            .last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(format!("{error:#}"));
    }

    fn snapshot(&self, usage: Option<&StoreUsage>) -> L3ActivitySnapshot {
        L3ActivitySnapshot {
            fills: self.fills.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            evictions: usage.map_or(0, |usage| usage.evicted_manifests),
            corrupt_entries: usage.map_or(0, |usage| usage.quarantined_objects),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            geometry_rejected: self.geometry_rejected.load(Ordering::Relaxed),
            last_error: self
                .last_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        }
    }
}

/// Point-in-time activity across every stage attached to one root manager.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct L3ActivitySnapshot {
    pub fills: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub evictions: u64,
    pub corrupt_entries: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub geometry_rejected: u64,
    pub last_error: Option<String>,
}

#[derive(Debug)]
struct L3ManagerInner {
    store: Arc<HandoffSegmentStore>,
    activity: Arc<L3Activity>,
    fill_claims: Arc<Mutex<std::collections::BTreeSet<String>>>,
    operations: RwLock<()>,
    reconciliation: StoreReconciliation,
}

/// The single physical owner of a node-local L3 root.
///
/// Clones are cheap stage handles into the same reservation, pin, lifecycle,
/// activity, and filesystem-lock domain.
#[derive(Clone, Debug)]
pub struct L3CacheManager {
    inner: Arc<L3ManagerInner>,
}

impl L3CacheManager {
    /// Acquire the manager for `root`, reusing the live node owner when one
    /// exists. A second process is rejected by the store's root lock.
    pub fn acquire(root: impl AsRef<Path>, limits: StoreLimits) -> Result<Self> {
        let root = canonical_cache_root(root.as_ref())?;
        let mut managers = ROOT_MANAGERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        managers.retain(|_, manager| manager.strong_count() > 0);
        if let Some(inner) = managers.get(&root).and_then(Weak::upgrade) {
            if inner.store.limits() != limits {
                bail!(
                    "cache root {} is already open with different limits",
                    root.display()
                );
            }
            return Ok(Self { inner });
        }

        let store = Arc::new(HandoffSegmentStore::open_with_limits(&root, limits)?);
        let reconciliation = store.reconcile_startup()?;
        let inner = Arc::new(L3ManagerInner {
            store,
            activity: Arc::new(L3Activity::default()),
            fill_claims: Arc::new(Mutex::new(std::collections::BTreeSet::new())),
            operations: RwLock::new(()),
            reconciliation,
        });
        managers.insert(root, Arc::downgrade(&inner));
        Ok(Self { inner })
    }

    pub fn tier(&self, state_identity: String, segment_bytes: usize) -> L3Tier {
        L3Tier::from_manager(self.clone(), state_identity, segment_bytes)
    }

    pub fn root(&self) -> &Path {
        self.inner.store.root()
    }

    pub fn limits(&self) -> StoreLimits {
        self.inner.store.limits()
    }

    pub fn reconciliation(&self) -> StoreReconciliation {
        self.inner.reconciliation
    }

    pub fn usage(&self) -> Result<StoreUsage> {
        self.inner.store.usage()
    }

    pub fn activity(&self) -> Result<L3ActivitySnapshot> {
        let usage = self.usage()?;
        Ok(self.inner.activity.snapshot(Some(&usage)))
    }

    pub fn prune_to(&self, target_bytes: u64) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.prune_to(target_bytes)
    }

    pub fn clear(&self) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.clear()
    }

    pub fn shares_root_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Manifest keys currently being filled from this root. Keeping claims
    /// here makes single-flight node-wide instead of duplicating physical
    /// reads when placement-equivalent stages miss at the same time.
    pub fn fill_claims(&self) -> Arc<Mutex<std::collections::BTreeSet<String>>> {
        self.inner.fill_claims.clone()
    }

    pub(crate) fn store(&self) -> &HandoffSegmentStore {
        &self.inner.store
    }

    pub(crate) fn activity_counters(&self) -> &L3Activity {
        &self.inner.activity
    }

    pub(crate) fn activity_snapshot(&self) -> L3ActivitySnapshot {
        let usage = self.usage().ok();
        self.inner.activity.snapshot(usage.as_ref())
    }

    pub(crate) fn operation_guard(&self) -> RwLockReadGuard<'_, ()> {
        self.inner
            .operations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn lifecycle_guard(&self) -> RwLockWriteGuard<'_, ()> {
        self.inner
            .operations
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn canonical_cache_root(root: &Path) -> Result<PathBuf> {
    if !root.is_absolute() {
        bail!("cache root must be absolute: {}", root.display());
    }
    crate::fsinfo::refuse_symlink(root)?;
    fs::create_dir_all(root)
        .with_context(|| format!("failed to create cache root {}", root.display()))?;
    fs::canonicalize(root)
        .with_context(|| format!("failed to resolve cache root {}", root.display()))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::ExactStatePayload;

    fn temp_root(name: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir()
            .join("skippy-l3-manager-tests")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn one_live_manager_owns_each_root() {
        let root = temp_root("shared-owner");
        let limits = StoreLimits::new(1_000_000, 0);
        let first = L3CacheManager::acquire(&root, limits).expect("first manager");
        let second = L3CacheManager::acquire(&root, limits).expect("shared manager");

        assert!(first.shares_root_with(&second));
        assert_eq!(first.root(), second.root());
        assert!(
            L3CacheManager::acquire(&root, StoreLimits::new(2_000_000, 0)).is_err(),
            "one root accepted contradictory budgets"
        );
    }

    #[test]
    fn concurrent_stages_cannot_double_reserve_the_budget() {
        let root = temp_root("atomic-reservation");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(10_000, 0)).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut tasks = Vec::new();
        for _ in 0..2 {
            let manager = manager.clone();
            let barrier = barrier.clone();
            tasks.push(std::thread::spawn(move || {
                let reservation = manager.store().reserve(8_000).unwrap();
                let admitted = reservation.is_ok();
                barrier.wait();
                admitted
            }));
        }
        barrier.wait();
        let admitted = tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .filter(|admitted| *admitted)
            .count();

        assert_eq!(admitted, 1, "two stages reserved the same bytes");
        assert_eq!(manager.usage().unwrap().reserved_inflight_bytes, 0);
    }

    #[test]
    fn activity_and_store_accounting_are_node_wide() {
        let root = temp_root("node-status");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let stage_a = manager.tier("state-a".to_string(), 4);
        let stage_b = manager.tier("state-b".to_string(), 4);

        stage_a
            .spill(
                "namespace-a",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-a-state".to_vec()),
                None,
                None,
            )
            .unwrap();

        assert_eq!(stage_b.activity().writes, 1);
        assert_eq!(
            stage_a.status().unwrap().usage,
            stage_b.status().unwrap().usage
        );
        assert_eq!(stage_a.status().unwrap().restorable_manifests, 1);
        assert_eq!(stage_b.status().unwrap().restorable_manifests, 0);
    }

    #[test]
    fn one_stage_cannot_evict_another_stages_pinned_manifest() {
        let root = temp_root("cross-stage-pin");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let stage_a = manager.tier("state-a".to_string(), 4);
        let stage_b = manager.tier("state-b".to_string(), 4);
        let key = stage_a
            .spill(
                "namespace-a",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-a-state".to_vec()),
                None,
                None,
            )
            .unwrap();

        let pin = stage_a.store().pin(&key);
        assert_eq!(stage_b.manager().prune_to(0).unwrap(), 0);
        assert!(stage_a.store().load_manifest(&key).is_ok());

        drop(pin);
        assert!(stage_b.manager().prune_to(0).unwrap() > 0);
        assert!(stage_a.store().load_manifest(&key).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn root_lock_rejects_an_independent_store_owner() {
        let root = temp_root("root-lock");
        let limits = StoreLimits::new(1_000_000, 0);
        let _manager = L3CacheManager::acquire(&root, limits).unwrap();

        let error = HandoffSegmentStore::open_with_limits(&root, limits)
            .expect_err("a second physical root owner acquired the lock");
        assert!(error.to_string().contains("already owned"));
    }

    #[test]
    fn startup_reconciles_temps_corrupt_manifests_links_and_orphans() {
        let root = temp_root("reconcile");
        let limits = StoreLimits::new(1_000_000, 0);
        let manager = L3CacheManager::acquire(&root, limits).unwrap();
        fs::write(root.join("segments/.tmp-dead"), b"partial").unwrap();
        fs::write(root.join("segments/orphan.seg"), b"orphan").unwrap();
        fs::write(root.join("manifests/corrupt.json"), b"{").unwrap();
        let namespace = root.join("prefixes/namespace");
        fs::create_dir_all(&namespace).unwrap();
        fs::write(namespace.join("000000000001-prefix.key"), b"missing").unwrap();
        drop(manager);

        let reopened = L3CacheManager::acquire(&root, limits).unwrap();
        let report = reopened.reconciliation();
        assert_eq!(report.removed_temporary_files, 1);
        assert_eq!(report.quarantined_manifests, 1);
        assert_eq!(report.removed_prefix_links, 1);
        assert_eq!(report.removed_orphan_bytes, 6);
        assert!(!root.join("segments/orphan.seg").exists());
        assert!(root.join("quarantine/corrupt.json").exists());
    }
}
