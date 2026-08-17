use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

pub const PREFIX_AFFINITY_TTL: Duration = Duration::from_secs(20 * 60);
pub const PREFIX_AFFINITY_MAX_ENTRIES: usize = 4096;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixAffinityStats {
    pub entries: usize,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub stale: u64,
    pub routes: u64,
    pub sticky_routes: u64,
    pub session_routes: u64,
    pub learned: u64,
    pub evicted: u64,
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_prefix_affinity_stats_snapshot {
    ($snapshot:ty $({ $($field:ident: $value:expr),+ $(,)? })?) => {
        impl $snapshot {
            fn from_prefix_affinity_stats(
                stats: $crate::prefix_affinity::PrefixAffinityStats,
                prefix_enabled: bool,
                sticky_enabled: bool,
            ) -> Self {
                Self {
                    prefix_enabled,
                    sticky_enabled,
                    prefix_entries: stats.entries,
                    prefix_lookups: stats.lookups,
                    prefix_hits: stats.hits,
                    prefix_misses: stats.misses,
                    prefix_stale: stats.stale,
                    prefix_routes: stats.routes,
                    sticky_routes: stats.sticky_routes,
                    session_routes: stats.session_routes,
                    learned: stats.learned,
                    evicted: stats.evicted,
                    $($($field: $value),+)?
                }
            }
        }
    };
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AffinityKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct AffinityEntry<T> {
    target: T,
    last_used: Instant,
    order: u64,
}

pub struct PrefixAffinity<T> {
    entries: HashMap<AffinityKey, AffinityEntry<T>>,
    lru: BTreeMap<u64, AffinityKey>,
    next_order: u64,
    stats: PrefixAffinityStats,
    ttl: Duration,
    max_entries: usize,
}

impl<T> PrefixAffinity<T> {
    pub fn record_sticky_route(&mut self) {
        self.stats.sticky_routes += 1;
    }

    pub fn record_session_route(&mut self) {
        self.stats.session_routes += 1;
    }

    pub fn snapshot(&mut self) -> PrefixAffinityStats {
        self.prune_expired();
        let mut stats = self.stats.clone();
        stats.entries = self.entries.len();
        stats
    }

    fn prune_expired(&mut self) {
        let now = Instant::now();
        while let Some((_, front_key)) = self.lru.first_key_value() {
            if self
                .entries
                .get(front_key)
                .is_some_and(|entry| now.duration_since(entry.last_used) <= self.ttl)
            {
                break;
            }
            let Some((_, front_key)) = self.lru.pop_first() else {
                break;
            };
            if self.entries.remove(&front_key).is_some() {
                self.stats.stale += 1;
            }
        }
    }

    fn touch_key(&mut self, key: &AffinityKey) {
        let Some(previous_order) = self.entries.get(key).map(|entry| entry.order) else {
            return;
        };
        self.lru.remove(&previous_order);
        let order = self.take_next_order();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.order = order;
        }
        self.lru.insert(order, key.clone());
    }

    fn remove_key(&mut self, key: &AffinityKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.lru.remove(&entry.order);
        }
    }

    fn take_next_order(&mut self) -> u64 {
        if self.next_order == u64::MAX {
            self.rebase_orders();
        }
        let order = self.next_order;
        self.next_order += 1;
        order
    }

    fn rebase_orders(&mut self) {
        let mut next_order = 0;
        let mut rebased = BTreeMap::new();
        for (_, key) in std::mem::take(&mut self.lru) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.order = next_order;
                rebased.insert(next_order, key);
                next_order += 1;
            }
        }
        self.lru = rebased;
        self.next_order = next_order;
    }
}

impl<T: Clone + PartialEq> PrefixAffinity<T> {
    pub fn lookup(&mut self, model: &str, prefix_hash: u64, candidates: &[T]) -> Option<T> {
        self.prune_expired();
        self.stats.lookups += 1;
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let Some(entry) = self.entries.get(&key) else {
            self.stats.misses += 1;
            return None;
        };
        if !candidates.contains(&entry.target) {
            self.remove_key(&key);
            self.stats.stale += 1;
            self.stats.misses += 1;
            return None;
        }
        let target = entry.target.clone();
        self.touch_key(&key);
        if let Some(existing) = self.entries.get_mut(&key) {
            existing.last_used = Instant::now();
        }
        self.stats.hits += 1;
        self.stats.routes += 1;
        Some(target)
    }

    pub fn learn(&mut self, model: &str, prefix_hash: u64, target: &T) {
        self.prune_expired();
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        self.remove_key(&key);
        let order = self.take_next_order();
        self.entries.insert(
            key.clone(),
            AffinityEntry {
                target: target.clone(),
                last_used: Instant::now(),
                order,
            },
        );
        self.lru.insert(order, key);
        self.stats.learned += 1;
        while self.entries.len() > self.max_entries {
            let Some((_, oldest)) = self.lru.pop_first() else {
                break;
            };
            if self.entries.remove(&oldest).is_some() {
                self.stats.evicted += 1;
            }
        }
    }

    pub fn forget(&mut self, model: &str, prefix_hash: u64, target: &T) {
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        if self
            .entries
            .get(&key)
            .is_some_and(|entry| &entry.target == target)
        {
            self.remove_key(&key);
            self.stats.stale += 1;
        }
    }
}

impl<T> Default for PrefixAffinity<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            lru: BTreeMap::new(),
            next_order: 0,
            stats: PrefixAffinityStats::default(),
            ttl: PREFIX_AFFINITY_TTL,
            max_entries: PREFIX_AFFINITY_MAX_ENTRIES,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[derive(Debug)]
    struct CloneTracked {
        id: u8,
        clone_count: Arc<AtomicUsize>,
    }

    impl Clone for CloneTracked {
        fn clone(&self) -> Self {
            self.clone_count.fetch_add(1, Ordering::Relaxed);
            Self {
                id: self.id,
                clone_count: Arc::clone(&self.clone_count),
            }
        }
    }

    impl PartialEq for CloneTracked {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for CloneTracked {}

    #[test]
    fn expired_entry_is_pruned_before_snapshot() {
        let mut affinity = PrefixAffinity::<u8>::default();
        affinity.learn("model", 1, &7);
        affinity.entries.values_mut().next().unwrap().last_used =
            Instant::now() - PREFIX_AFFINITY_TTL - Duration::from_secs(1);

        let stats = affinity.snapshot();

        assert_eq!(stats.entries, 0);
        assert_eq!(stats.stale, 1);
    }

    #[test]
    fn lookup_learn_forget_and_lru_stats_are_stable() {
        let mut affinity = PrefixAffinity {
            max_entries: 2,
            ..PrefixAffinity::default()
        };
        affinity.learn("model", 1, &1u8);
        affinity.learn("model", 2, &2u8);
        assert_eq!(affinity.lookup("model", 1, &[1, 2]), Some(1));
        affinity.learn("model", 3, &3u8);
        assert_eq!(affinity.lookup("model", 2, &[1, 2, 3]), None);
        affinity.forget("model", 1, &9);
        assert_eq!(affinity.lookup("model", 1, &[1, 3]), Some(1));
        affinity.forget("model", 1, &1);

        let stats = affinity.snapshot();

        assert_eq!(stats.entries, 1);
        assert_eq!(stats.lookups, 3);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.stale, 1);
        assert_eq!(stats.routes, 2);
        assert_eq!(stats.learned, 3);
        assert_eq!(stats.evicted, 1);
    }

    #[test]
    fn lookup_clones_only_returned_target() {
        let clone_count = Arc::new(AtomicUsize::new(0));
        let target = CloneTracked {
            id: 1,
            clone_count: Arc::clone(&clone_count),
        };
        let mut affinity = PrefixAffinity::default();
        affinity.learn("model", 1, &target);
        clone_count.store(0, Ordering::Relaxed);

        let unavailable = CloneTracked {
            id: 2,
            clone_count: Arc::new(AtomicUsize::new(0)),
        };
        assert_eq!(affinity.lookup("model", 1, &[unavailable]), None);
        assert_eq!(clone_count.load(Ordering::Relaxed), 0);

        affinity.learn("model", 1, &target);
        clone_count.store(0, Ordering::Relaxed);
        let available = CloneTracked {
            id: 1,
            clone_count: Arc::clone(&clone_count),
        };
        assert_eq!(affinity.lookup("model", 1, &[available]).unwrap().id, 1);
        assert_eq!(clone_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn relearning_key_refreshes_lru_without_duplicate_order_entries() {
        let mut affinity = PrefixAffinity {
            max_entries: 2,
            ..PrefixAffinity::default()
        };
        affinity.learn("model", 1, &1u8);
        affinity.learn("model", 2, &2u8);
        affinity.learn("model", 1, &1u8);
        affinity.learn("model", 3, &3u8);

        assert_eq!(affinity.lookup("model", 2, &[1, 2, 3]), None);
        assert_eq!(affinity.lookup("model", 1, &[1, 2, 3]), Some(1));
        let stats = affinity.snapshot();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.evicted, 1);
    }
}
