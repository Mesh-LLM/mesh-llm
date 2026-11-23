//! How full the store is, from its running counters. `usage` lists and loads
//! every manifest, which is right for the status contract and too slow for a
//! sampler that reads the store several times a second.

use std::sync::atomic::Ordering;

use super::HandoffSegmentStore;
use crate::fsinfo;

/// The store's budget and occupancy, read without touching its files except
/// for one filesystem free-space query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StorePressure {
    pub budget_bytes: u64,
    /// Managed bytes on disk plus in-flight reservations; `None` until a scan
    /// has established the running total the store keeps between scans.
    pub used_bytes: Option<u64>,
    pub reserved_inflight_bytes: u64,
    pub filesystem_available_bytes: Option<u64>,
    pub minimum_free_bytes: u64,
    /// Manifests removed by budget enforcement or prune since open.
    pub evicted_manifests: u64,
    /// Objects quarantined after failing verification since open.
    pub quarantined_objects: u64,
}

impl HandoffSegmentStore {
    pub fn pressure(&self) -> StorePressure {
        let limits = self.limits();
        let reserved = self.reserved_inflight.load(Ordering::Acquire);
        let used_bytes = self.usage_valid.load(Ordering::Acquire).then(|| {
            self.usage_bytes
                .load(Ordering::Acquire)
                .saturating_add(reserved)
        });
        StorePressure {
            budget_bytes: limits.budget_bytes,
            used_bytes,
            reserved_inflight_bytes: reserved,
            filesystem_available_bytes: fsinfo::available_bytes(&self.root).ok(),
            minimum_free_bytes: limits.minimum_free_bytes,
            evicted_manifests: self.evicted_manifests.load(Ordering::Relaxed),
            quarantined_objects: self.quarantined_objects.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_reports_the_budget_and_the_running_total_once_known() {
        let root = std::env::temp_dir()
            .join("skippy-l3-tests")
            .join(format!("pressure-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = HandoffSegmentStore::open(&root, 1 << 20).expect("open store");
        let before = store.pressure();
        assert_eq!(before.budget_bytes, 1 << 20);
        assert!(before.filesystem_available_bytes.is_some());

        store.put_segment(&[7u8; 4096]).expect("put segment");
        let usage = store.usage().expect("usage");
        let after = store.pressure();
        assert_eq!(after.used_bytes, Some(usage.used_bytes));
        assert_eq!(after.evicted_manifests, usage.evicted_manifests);
        assert_eq!(after.quarantined_objects, usage.quarantined_objects);
        drop(store);
        let _ = std::fs::remove_dir_all(&root);
    }
}
