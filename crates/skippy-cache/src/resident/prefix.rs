use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::ResidentCacheConfig;

#[derive(Debug)]
pub struct ResidentPrefixCache {
    max_entries: usize,
    max_bytes: u64,
    /// 0 means "unlimited (legacy)".
    max_resident_tokens: u64,
    min_tokens: u64,
    reserved_seq_count: i32,
    next_seq_id: i32,
    clock: u64,
    resident_tokens: u64,
    estimated_bytes: u64,
    entries: HashMap<String, ResidentPrefixEntry>,
    free_seq_ids: Vec<i32>,
}

#[derive(Debug)]
struct ResidentPrefixEntry {
    seq_id: i32,
    token_count: u64,
    estimated_bytes: u64,
    last_used: u64,
    borrowed: bool,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixLookup {
    pub seq_id: i32,
    pub entries: usize,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixEviction {
    pub page_id: String,
    pub seq_id: i32,
    pub token_count: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixAllocation {
    pub seq_id: i32,
    pub evictions: Vec<ResidentPrefixEviction>,
    pub should_save: bool,
    pub should_retain: bool,
}

impl ResidentPrefixAllocation {
    fn existing(seq_id: i32) -> Self {
        Self {
            seq_id,
            evictions: Vec::new(),
            should_save: false,
            should_retain: true,
        }
    }

    fn new_record(seq_id: i32, evictions: Vec<ResidentPrefixEviction>) -> Self {
        Self {
            seq_id,
            evictions,
            should_save: true,
            should_retain: true,
        }
    }

    fn uncacheable() -> Self {
        Self {
            seq_id: -1,
            evictions: Vec::new(),
            should_save: false,
            should_retain: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentPrefixCacheStats {
    pub entries: usize,
    pub resident_tokens: u64,
    pub estimated_bytes: u64,
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl ResidentPrefixCache {
    pub fn new(config: ResidentCacheConfig) -> Self {
        Self {
            max_entries: config.max_entries,
            max_bytes: config.max_bytes,
            max_resident_tokens: config.max_resident_tokens,
            min_tokens: config.min_tokens,
            reserved_seq_count: config.reserved_seq_count,
            next_seq_id: config.reserved_seq_count,
            clock: 0,
            resident_tokens: 0,
            estimated_bytes: 0,
            entries: HashMap::new(),
            free_seq_ids: Vec::new(),
        }
    }

    pub fn lookup(&mut self, page_id: &str) -> Option<ResidentPrefixLookup> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        if entry.borrowed {
            return None;
        }
        entry.last_used = self.clock;
        Some(ResidentPrefixLookup {
            seq_id: entry.seq_id,
            entries,
        })
    }

    pub fn acquire(&mut self, page_id: &str) -> Option<ResidentPrefixLookup> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        let entry = self.entries.get_mut(page_id)?;
        if entry.borrowed {
            return None;
        }
        entry.borrowed = true;
        entry.last_used = self.clock;
        Some(ResidentPrefixLookup {
            seq_id: entry.seq_id,
            entries,
        })
    }

    pub fn release(&mut self, page_id: &str) {
        if let Some(entry) = self.entries.get_mut(page_id) {
            entry.borrowed = false;
            self.clock = self.clock.saturating_add(1);
            entry.last_used = self.clock;
        }
    }

    pub fn allocate_for_record(
        &mut self,
        page_id: &str,
        token_count: u64,
        estimated_bytes: u64,
        mut drop_evicted: impl FnMut(i32) -> Result<()>,
    ) -> Result<ResidentPrefixAllocation> {
        if token_count < self.min_tokens {
            bail!("resident prefix has fewer tokens than cache minimum");
        }
        self.clock = self.clock.saturating_add(1);
        if let Some(entry) = self.entries.get(page_id) {
            return Ok(ResidentPrefixAllocation::existing(entry.seq_id));
        }
        if self.candidate_exceeds_single_record_budget(estimated_bytes, token_count) {
            return Ok(ResidentPrefixAllocation::uncacheable());
        }

        let evictions =
            self.evict_until_room_for(estimated_bytes, token_count, &mut drop_evicted)?;
        let seq_id = self.next_sequence_id()?;
        Ok(ResidentPrefixAllocation::new_record(seq_id, evictions))
    }

    pub fn commit_record(
        &mut self,
        page_id: String,
        seq_id: i32,
        token_count: u64,
        estimated_bytes: u64,
    ) {
        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&page_id) {
            self.resident_tokens = self.resident_tokens.saturating_sub(previous.token_count);
            self.estimated_bytes = self
                .estimated_bytes
                .saturating_sub(previous.estimated_bytes);
        }
        self.resident_tokens = self.resident_tokens.saturating_add(token_count);
        self.estimated_bytes = self.estimated_bytes.saturating_add(estimated_bytes);
        self.entries.insert(
            page_id,
            ResidentPrefixEntry {
                seq_id,
                token_count,
                estimated_bytes,
                last_used: self.clock,
                borrowed: false,
            },
        );
    }

    /// Evict one LRU entry (the entry with the smallest `last_used`
    /// that is not currently borrowed). Returns the evicted entry
    /// metadata, or `None` if there is nothing to evict.
    ///
    /// This is the **proactive eviction** path: it runs outside the
    /// normal `allocate_for_record` flow so that the decode loop can
    /// free KV cells before grammar-triggered retries need them.
    pub fn evict_one_lru_entry(
        &mut self,
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Option<ResidentPrefixEviction>> {
        self.evict_lru_entry(drop_evicted)
    }

    /// Evict LRU entries until at least `min_tokens` have been released,
    /// or until no non-borrowed entry remains. Returns the entries that
    /// were actually evicted.
    pub fn evict_lru_until_tokens(
        &mut self,
        min_tokens: u64,
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Vec<ResidentPrefixEviction>> {
        let mut evictions = Vec::new();
        let mut evicted_tokens = 0_u64;
        while evicted_tokens < min_tokens {
            let Some(eviction) = self.evict_lru_entry(drop_evicted)? else {
                break;
            };
            evicted_tokens = evicted_tokens.saturating_add(eviction.token_count);
            evictions.push(eviction);
        }
        Ok(evictions)
    }

    fn evict_lru_entry(
        &mut self,
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Option<ResidentPrefixEviction>> {
        let victim = self
            .entries
            .iter()
            .filter(|(_, entry)| !entry.borrowed)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        let Some(victim) = victim else {
            return Ok(None);
        };
        let entry = self
            .entries
            .get(&victim)
            .expect("selected resident prefix victim should exist");
        drop_evicted(entry.seq_id)?;
        let entry = self
            .entries
            .remove(&victim)
            .expect("selected resident prefix victim should still exist after native drop");
        self.free_seq_ids.push(entry.seq_id);
        self.resident_tokens = self.resident_tokens.saturating_sub(entry.token_count);
        self.estimated_bytes = self.estimated_bytes.saturating_sub(entry.estimated_bytes);
        Ok(Some(ResidentPrefixEviction {
            page_id: victim,
            seq_id: entry.seq_id,
            token_count: entry.token_count,
        }))
    }

    pub fn stats(&self) -> ResidentPrefixCacheStats {
        ResidentPrefixCacheStats {
            entries: self.entries.len(),
            resident_tokens: self.resident_tokens,
            estimated_bytes: self.estimated_bytes,
            max_entries: self.max_entries,
            max_bytes: self.max_bytes,
        }
    }

    fn evict_until_room_for(
        &mut self,
        estimated_bytes: u64,
        token_count: u64,
        drop_evicted: &mut impl FnMut(i32) -> Result<()>,
    ) -> Result<Vec<ResidentPrefixEviction>> {
        let mut evictions = Vec::new();
        loop {
            let over_entries = self.entries.len().saturating_add(1) > self.max_entries;
            let over_bytes = self.max_bytes > 0
                && self.estimated_bytes.saturating_add(estimated_bytes) > self.max_bytes;
            // Under unified-KV serving the prefix cache shares the
            // `n_ctx` cell pool with the active lanes. `max_entries`
            // and `max_bytes` alone do not bound the cell footprint:
            // 12 entries averaging 10k tokens each pin 120k cells in
            // a 131k-`n_ctx` pool with no LRU pressure. Add an
            // explicit token budget so eviction kicks in before the
            // cells run out.
            let over_tokens = self.max_resident_tokens > 0
                && self.resident_tokens.saturating_add(token_count) > self.max_resident_tokens;
            if !over_entries && !over_bytes && !over_tokens {
                break;
            }
            let Some(eviction) = self.evict_lru_entry(drop_evicted)? else {
                bail!("resident prefix cache has no releasable entries");
            };
            evictions.push(eviction);
        }
        Ok(evictions)
    }

    fn candidate_exceeds_single_record_budget(
        &self,
        estimated_bytes: u64,
        token_count: u64,
    ) -> bool {
        let over_bytes = self.max_bytes > 0 && estimated_bytes > self.max_bytes;
        let over_tokens = self.max_resident_tokens > 0 && token_count > self.max_resident_tokens;
        over_bytes || over_tokens
    }

    fn next_sequence_id(&mut self) -> Result<i32> {
        if let Some(seq_id) = self.free_seq_ids.pop() {
            return Ok(seq_id);
        }
        let seq_id = self.next_seq_id;
        self.next_seq_id = self
            .next_seq_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resident prefix sequence id overflow"))?;
        if seq_id < self.reserved_seq_count || seq_id >= 1024 {
            bail!("resident prefix sequence id capacity exhausted");
        }
        Ok(seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_entries: usize, max_bytes: u64, max_resident_tokens: u64) -> ResidentCacheConfig {
        ResidentCacheConfig {
            max_entries,
            max_bytes,
            max_resident_tokens,
            min_tokens: 256,
            reserved_seq_count: 2,
        }
    }

    #[test]
    fn token_budget_triggers_lru_before_entry_cap_under_unified_kv() {
        // Regression: under skippy `kv_unified = true` the prefix cache
        // shares the model's `n_ctx` cell pool with the active lanes.
        // Before this fix, the cache only evicted on `max_entries` and
        // `max_bytes`. Live data on a 131k-`n_ctx` MiniMax showed the
        // cache happily filling to ~124k pinned tokens (~95% of the
        // pool) across just 12 entries with no LRU pressure, starving
        // the lanes and surfacing as HTTP 502
        // `RuntimeError: llama_decode failed`.
        //
        // With `max_resident_tokens = 4096` and entries of 1500 tokens
        // each, the third record must trigger eviction even though
        // we're well under `max_entries = 16`.
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 4096));
        let mut dropped: Vec<i32> = Vec::new();

        let alloc1 = cache
            .allocate_for_record("page-1", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        assert!(alloc1.should_save);
        cache.commit_record("page-1".to_string(), alloc1.seq_id, 1500, 100);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 1500);

        let alloc2 = cache
            .allocate_for_record("page-2", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-2".to_string(), alloc2.seq_id, 1500, 100);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().resident_tokens, 3000);
        // Two entries fit; we are at 3000 / 4096 tokens.
        assert!(dropped.is_empty(), "should not have evicted yet");

        let alloc3 = cache
            .allocate_for_record("page-3", 1500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-3".to_string(), alloc3.seq_id, 1500, 100);
        // 3000 + 1500 = 4500 > 4096 — must have evicted at least one
        // entry to make room. LRU picks page-1.
        assert_eq!(dropped, vec![alloc1.seq_id], "LRU should evict oldest");
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().resident_tokens, 3000);
    }

    #[test]
    fn oversized_resident_prefix_candidate_is_nonfatal() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 4096));
        let small = cache
            .allocate_for_record("small", 1000, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("small".to_string(), small.seq_id, 1000, 100);

        let allocation = cache
            .allocate_for_record("huge", 5000, 100, |_| {
                panic!("oversized candidates should not evict existing entries")
            })
            .expect("oversized candidates should be treated as uncacheable, not fatal");

        assert!(!allocation.should_save);
        assert!(!allocation.should_retain);
        assert!(allocation.evictions.is_empty());
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.resident_tokens, 1000);
        assert!(cache.lookup("small").is_some());
    }

    #[test]
    fn small_ctx_smoke_test_scenario_records_without_eviction_loop() {
        // Regression for skippy-ci-smoke `prompt exact-prefix hit and
        // live-session reuse` (CI run 26193173851). The smoke test
        // ships SmolLM2-135M with `PROMPT_CTX_SIZE=768` and a 533-token
        // prompt. With a naive `n_ctx / 2 = 384` cap, the cap was
        // *smaller than a single prompt*, so the first record attempt
        // would call `evict_until_room_for` with `over_tokens`
        // permanently true on an empty cache and bail with
        // "no releasable entries".
        //
        // `ResidentCacheConfig::from_stage` derives the cap from the
        // model's `n_ctx` via `derive_max_resident_tokens`, which uses
        // a hard `MIN_CTX_FOR_CELL_CAP = 8192` floor: below that, the
        // cap is 0 (disabled) regardless of `min_tokens`. The smoke
        // test runs with `ctx_size = 768`, well below the floor, so
        // its derived cap is 0. This unit test pins that path: cap=0,
        // record a 533-token prompt against a 256-token min, no
        // eviction loop.
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 533, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 533, 100);
        assert_eq!(cache.stats().resident_tokens, 533);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn zero_token_budget_disables_the_check() {
        // max_resident_tokens = 0 means "unlimited" — legacy behavior.
        // 12 entries at 10k tokens each = 120k tokens, which the
        // previous unbounded cache happily accepted.
        let mut cache = ResidentPrefixCache::new(cfg(64, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        for i in 0..12 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 10_000, 100, |sid| {
                    dropped.push(sid);
                    Ok(())
                })
                .unwrap();
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 10_000, 100);
        }
        assert!(
            dropped.is_empty(),
            "zero token budget should not trigger evictions"
        );
        assert_eq!(cache.stats().entries, 12);
        assert_eq!(cache.stats().resident_tokens, 120_000);
    }

    #[test]
    fn evict_one_lru_evicts_least_recently_used() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        // Insert 3 entries.
        for i in 0..3 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 500, 100, |sid| {
                    dropped.push(sid);
                    Ok(())
                })
                .unwrap();
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 500, 100);
        }
        assert_eq!(cache.stats().entries, 3);
        assert_eq!(cache.stats().resident_tokens, 1500);

        // Touch page-1 and page-2 (LRU should now be page-0).
        cache.lookup("page-1");
        cache.lookup("page-2");

        let evicted = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("should have evicted one entry");

        assert_eq!(evicted.page_id, "page-0", "LRU should be page-0");
        assert_eq!(evicted.token_count, 500);
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().resident_tokens, 1000);
    }

    #[test]
    fn evict_one_lru_skips_borrowed_entries() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        let alloc0 = cache
            .allocate_for_record("page-0", 500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc0.seq_id, 500, 100);

        let alloc1 = cache
            .allocate_for_record("page-1", 500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-1".to_string(), alloc1.seq_id, 500, 100);

        // Borrow page-0 (the oldest entry).
        cache.acquire("page-0");

        let evicted = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("should have evicted one entry");

        // page-0 is borrowed, so LRU should skip it and evict page-1.
        assert_eq!(evicted.page_id, "page-1", "should skip borrowed entry");
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn evict_one_lru_empty_cache_returns_none() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let result = cache.evict_one_lru_entry(&mut |_| Ok(())).unwrap();
        assert!(result.is_none(), "empty cache should return None");
    }

    #[test]
    fn evict_one_lru_all_borrowed_returns_none() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        let alloc = cache
            .allocate_for_record("page-0", 500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 500, 100);
        cache.acquire("page-0");

        let result = cache.evict_one_lru_entry(&mut |_| Ok(())).unwrap();
        assert!(
            result.is_none(),
            "cache with only borrowed entries should return None"
        );
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn evict_one_lru_multiple_calls_drain_all_non_borrowed() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        let mut seq_ids = Vec::new();
        for i in 0..4 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 300, 50, |_| Ok(()))
                .unwrap();
            seq_ids.push(alloc.seq_id);
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 300, 50);
        }
        assert_eq!(cache.stats().entries, 4);

        // Touch two to shift LRU ordering.
        cache.lookup("page-2");
        cache.lookup("page-3");

        cache.acquire("page-2");

        let e1 = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("first eviction");
        assert_eq!(e1.page_id, "page-0");
        assert_eq!(cache.stats().entries, 3);

        let e2 = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("second eviction");
        assert_eq!(e2.page_id, "page-1");
        assert_eq!(cache.stats().entries, 2);

        let e3 = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("third eviction");
        assert_eq!(e3.page_id, "page-3");
        assert_eq!(cache.stats().entries, 1);

        let e4 = cache.evict_one_lru_entry(&mut |_| Ok(())).unwrap();
        assert!(e4.is_none(), "only borrowed remains");
        assert_eq!(cache.stats().entries, 1);

        assert!(dropped.contains(&seq_ids[0]), "page-0 seq_id dropped");
        assert!(dropped.contains(&seq_ids[1]), "page-1 seq_id dropped");
        assert!(dropped.contains(&seq_ids[3]), "page-3 seq_id dropped");
    }

    #[test]
    fn evict_one_lru_updates_internal_accounting() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 1000, 200, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 1000, 200);

        assert_eq!(cache.stats().resident_tokens, 1000);
        assert_eq!(cache.stats().estimated_bytes, 200);
        assert_eq!(cache.stats().entries, 1);

        let evicted = cache
            .evict_one_lru_entry(&mut |_| Ok(()))
            .unwrap()
            .expect("should evict");

        assert_eq!(cache.stats().resident_tokens, 0);
        assert_eq!(cache.stats().estimated_bytes, 0);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(evicted.token_count, 1000);
        assert_eq!(evicted.seq_id, alloc.seq_id);
        assert!(
            cache.free_seq_ids.contains(&alloc.seq_id),
            "evicted seq_id should be reusable"
        );
    }

    #[test]
    fn evict_one_lru_drop_failure_preserves_cache_state() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 1000, 200, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 1000, 200);

        let error = cache.evict_one_lru_entry(&mut |_| Err(anyhow::anyhow!("native drop failed")));

        assert!(error.is_err());
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 1000);
        assert_eq!(cache.stats().estimated_bytes, 200);
        assert!(cache.lookup("page-0").is_some());
        assert!(cache.free_seq_ids.is_empty());
    }

    #[test]
    fn allocate_for_record_drop_failure_preserves_existing_cache_state() {
        let mut cache = ResidentPrefixCache::new(cfg(1, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 1000, 200, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 1000, 200);

        let error = cache.allocate_for_record("page-1", 1000, 200, |_| {
            Err(anyhow::anyhow!("native drop failed"))
        });

        assert!(error.is_err());
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 1000);
        assert_eq!(cache.stats().estimated_bytes, 200);
        assert!(cache.lookup("page-0").is_some());
        assert!(cache.lookup("page-1").is_none());
        assert!(cache.free_seq_ids.is_empty());
    }

    #[test]
    fn evict_lru_until_tokens_evicts_multiple_entries_until_target() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut seq_ids = Vec::new();
        for (index, token_count) in [300, 400, 500].into_iter().enumerate() {
            let alloc = cache
                .allocate_for_record(&format!("page-{index}"), token_count, 100, |_| Ok(()))
                .unwrap();
            seq_ids.push(alloc.seq_id);
            cache.commit_record(format!("page-{index}"), alloc.seq_id, token_count, 100);
        }

        let mut dropped = Vec::new();
        let evictions = cache
            .evict_lru_until_tokens(700, &mut |seq_id| {
                dropped.push(seq_id);
                Ok(())
            })
            .unwrap();

        assert_eq!(evictions.len(), 2);
        assert_eq!(evictions[0].page_id, "page-0");
        assert_eq!(evictions[1].page_id, "page-1");
        assert_eq!(dropped, vec![seq_ids[0], seq_ids[1]]);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 500);
        assert!(cache.lookup("page-2").is_some());
    }

    #[test]
    fn evict_lru_until_tokens_stops_when_no_releasable_entries_remain() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        for (index, token_count) in [300, 400].into_iter().enumerate() {
            let alloc = cache
                .allocate_for_record(&format!("page-{index}"), token_count, 100, |_| Ok(()))
                .unwrap();
            cache.commit_record(format!("page-{index}"), alloc.seq_id, token_count, 100);
        }
        cache.acquire("page-1");

        let evictions = cache.evict_lru_until_tokens(1024, &mut |_| Ok(())).unwrap();

        assert_eq!(evictions.len(), 1);
        assert_eq!(evictions[0].page_id, "page-0");
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 400);
        assert!(cache.lookup("page-1").is_none());
    }

    #[test]
    fn evict_one_lru_seq_id_reused_on_subsequent_allocation() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 500, 100, |_| Ok(()))
            .unwrap();
        let orig_seq_id = alloc.seq_id;
        cache.commit_record("page-0".to_string(), orig_seq_id, 500, 100);

        cache.evict_one_lru_entry(&mut |_| Ok(())).unwrap();

        let alloc2 = cache
            .allocate_for_record("page-1", 500, 100, |_| Ok(()))
            .unwrap();
        assert_eq!(
            alloc2.seq_id, orig_seq_id,
            "should reuse evicted seq_id before allocating new one"
        );
    }

    #[test]
    fn evict_one_lru_then_recommit_same_page_id() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let alloc = cache
            .allocate_for_record("page-0", 500, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc.seq_id, 500, 100);

        cache.evict_one_lru_entry(&mut |_| Ok(())).unwrap();

        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().resident_tokens, 0);

        let alloc2 = cache
            .allocate_for_record("page-0", 800, 150, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc2.seq_id, 800, 150);

        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().resident_tokens, 800);
        assert_eq!(cache.stats().estimated_bytes, 150);
    }

    #[test]
    fn evict_one_lru_preserves_allocate_for_record_eviction_logic() {
        let mut cache = ResidentPrefixCache::new(cfg(4, 0, 4096));
        let mut dropped: Vec<i32> = Vec::new();

        for i in 0..4 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 500, 100, |_| Ok(()))
                .unwrap();
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 500, 100);
        }
        assert_eq!(cache.stats().entries, 4);

        cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("should evict");
        assert_eq!(cache.stats().entries, 3);

        let alloc = cache
            .allocate_for_record("page-4", 500, 100, |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap();
        assert!(
            alloc.evictions.is_empty(),
            "allocate after proactive eviction should not need more evictions"
        );
        cache.commit_record("page-4".to_string(), alloc.seq_id, 500, 100);
        assert_eq!(cache.stats().entries, 4);
    }

    #[test]
    fn evict_one_lru_after_release_shifts_lru_ordering() {
        let mut cache = ResidentPrefixCache::new(cfg(16, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();

        let alloc0 = cache
            .allocate_for_record("page-0", 500, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-0".to_string(), alloc0.seq_id, 500, 100);

        let alloc1 = cache
            .allocate_for_record("page-1", 500, 100, |_| Ok(()))
            .unwrap();
        cache.commit_record("page-1".to_string(), alloc1.seq_id, 500, 100);

        // Release bumps last_used, so page-0 becomes strictly newer.
        cache.acquire("page-0");
        cache.release("page-0");

        let evicted = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("should evict");
        assert_eq!(
            evicted.page_id, "page-1",
            "recently released entry should not be LRU immediately"
        );
    }

    #[test]
    fn evict_one_lru_from_cache_at_entry_cap_does_not_panic() {
        let mut cache = ResidentPrefixCache::new(cfg(4, 0, 0));
        let mut dropped: Vec<i32> = Vec::new();
        for i in 0..4 {
            let alloc = cache
                .allocate_for_record(&format!("page-{i}"), 500, 100, |_| Ok(()))
                .unwrap();
            cache.commit_record(format!("page-{i}"), alloc.seq_id, 500, 100);
        }
        assert_eq!(cache.stats().entries, 4);

        let evicted = cache
            .evict_one_lru_entry(&mut |sid| {
                dropped.push(sid);
                Ok(())
            })
            .unwrap()
            .expect("should evict at cap");
        assert_eq!(evicted.page_id, "page-0");
        assert_eq!(cache.stats().entries, 3);
    }
}
