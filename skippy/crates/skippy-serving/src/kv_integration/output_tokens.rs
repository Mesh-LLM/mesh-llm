use std::collections::BTreeMap;

#[derive(Debug, Default)]
struct OutputTokenEntry {
    first: Option<i32>,
    replay: Vec<i32>,
    last_used: u64,
}

/// Sampling-aware output accelerators with one shared entry budget.
///
/// First-token and exact-replay data for the same prompt/sampling key share an
/// entry. The LRU cap is intentionally independent of the radix population:
/// failed or skipped KV records must not let auxiliary output metadata grow
/// without bound.
#[derive(Debug)]
pub(crate) struct OutputTokenCache {
    max_entries: usize,
    clock: u64,
    entries: BTreeMap<String, OutputTokenEntry>,
}

impl OutputTokenCache {
    pub(super) fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            clock: 0,
            entries: BTreeMap::new(),
        }
    }

    pub(super) fn record_first(&mut self, cache_key: &str, predicted: i32) -> bool {
        if self.max_entries == 0 {
            return false;
        }
        let last_used = self.next_clock();
        let entry = self.entries.entry(cache_key.to_string()).or_default();
        entry.last_used = last_used;
        let inserted = entry.first.replace(predicted).is_none();
        self.enforce_capacity();
        inserted
    }

    pub(super) fn lookup_first(&mut self, cache_key: &str) -> Option<i32> {
        let last_used = self.next_clock();
        let entry = self.entries.get_mut(cache_key)?;
        entry.last_used = last_used;
        entry.first
    }

    pub(super) fn record_replay(
        &mut self,
        cache_key: &str,
        previous: &[i32],
        predicted: i32,
        max_replay_tokens: usize,
    ) -> Option<usize> {
        if self.max_entries == 0 || max_replay_tokens == 0 || previous.len() >= max_replay_tokens {
            return None;
        }
        let last_used = self.next_clock();
        let recorded = {
            let entry = self.entries.entry(cache_key.to_string()).or_default();
            entry.last_used = last_used;
            if entry.replay.len() > previous.len() {
                Some(entry.replay.len().min(max_replay_tokens))
            } else if entry.replay.as_slice() != previous {
                None
            } else {
                entry.replay.push(predicted);
                Some(entry.replay.len())
            }
        };
        self.enforce_capacity();
        recorded
    }

    pub(super) fn lookup_replay(&mut self, cache_key: &str, max_tokens: usize) -> Vec<i32> {
        if max_tokens == 0 {
            return Vec::new();
        }
        let last_used = self.next_clock();
        let Some(entry) = self.entries.get_mut(cache_key) else {
            return Vec::new();
        };
        entry.last_used = last_used;
        entry.replay.iter().copied().take(max_tokens).collect()
    }

    fn next_clock(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn enforce_capacity(&mut self) {
        while self.entries.len() > self.max_entries {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.entries.remove(&victim);
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::OutputTokenCache;

    #[test]
    fn unique_prompt_churn_never_exceeds_the_shared_entry_cap() {
        let mut cache = OutputTokenCache::new(8);

        for index in 0..2_048 {
            let key = format!("prompt-{index}");
            assert!(cache.record_first(&key, index));
            assert_eq!(cache.record_replay(&key, &[], index, 4), Some(1));
            assert!(cache.len() <= 8);
        }

        assert_eq!(cache.len(), 8);
        assert_eq!(cache.lookup_first("prompt-0"), None);
        assert_eq!(cache.lookup_replay("prompt-2047", 4), vec![2_047]);
    }

    #[test]
    fn first_and_replay_tokens_share_one_lru_entry() {
        let mut cache = OutputTokenCache::new(1);

        assert!(cache.record_first("shared", 7));
        assert_eq!(cache.record_replay("shared", &[], 8, 4), Some(1));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.lookup_first("shared"), Some(7));
        assert_eq!(cache.lookup_replay("shared", 4), vec![8]);

        assert!(cache.record_first("replacement", 9));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.lookup_first("shared"), None);
    }

    #[test]
    fn lookups_refresh_lru_recency() {
        let mut cache = OutputTokenCache::new(2);
        assert!(cache.record_first("old", 1));
        assert!(cache.record_first("new", 2));
        assert_eq!(cache.lookup_first("old"), Some(1));

        assert!(cache.record_first("newest", 3));

        assert_eq!(cache.lookup_first("new"), None);
        assert_eq!(cache.lookup_first("old"), Some(1));
        assert_eq!(cache.lookup_first("newest"), Some(3));
    }
}
