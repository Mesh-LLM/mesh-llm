use std::collections::{HashMap, VecDeque};
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AffinityKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct AffinityEntry<T> {
    target: T,
    last_used: Instant,
}

pub struct PrefixAffinity<T> {
    entries: HashMap<AffinityKey, AffinityEntry<T>>,
    lru: VecDeque<AffinityKey>,
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
        while let Some(front_key) = self.lru.front().cloned() {
            match self.entries.get(&front_key) {
                Some(entry) if now.duration_since(entry.last_used) > self.ttl => {
                    self.lru.pop_front();
                    if self.entries.remove(&front_key).is_some() {
                        self.stats.stale += 1;
                    }
                }
                Some(_) => break,
                None => {
                    self.lru.pop_front();
                }
            }
        }
    }

    fn touch_key(&mut self, key: &AffinityKey) {
        if let Some(position) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(position);
        }
        self.lru.push_back(key.clone());
    }

    fn remove_key(&mut self, key: &AffinityKey) {
        self.entries.remove(key);
        if let Some(position) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(position);
        }
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
        let entry = match self.entries.get(&key).cloned() {
            Some(entry) => entry,
            None => {
                self.stats.misses += 1;
                return None;
            }
        };
        if !candidates.contains(&entry.target) {
            self.remove_key(&key);
            self.stats.stale += 1;
            self.stats.misses += 1;
            return None;
        }
        self.touch_key(&key);
        if let Some(existing) = self.entries.get_mut(&key) {
            existing.last_used = Instant::now();
        }
        self.stats.hits += 1;
        self.stats.routes += 1;
        Some(entry.target)
    }

    pub fn learn(&mut self, model: &str, prefix_hash: u64, target: &T) {
        self.prune_expired();
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        self.entries.insert(
            key.clone(),
            AffinityEntry {
                target: target.clone(),
                last_used: Instant::now(),
            },
        );
        self.touch_key(&key);
        self.stats.learned += 1;
        while self.entries.len() > self.max_entries {
            let Some(oldest) = self.lru.pop_front() else {
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
            lru: VecDeque::new(),
            stats: PrefixAffinityStats::default(),
            ttl: PREFIX_AFFINITY_TTL,
            max_entries: PREFIX_AFFINITY_MAX_ENTRIES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
