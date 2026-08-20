use std::collections::{BTreeMap, HashMap};

use anyhow::Result;

use crate::{
    CacheBlobStore, CacheDedupeStats, ExactStatePayload, PrefixDiskTier, PrefixMissTracker,
};

#[derive(Debug)]
pub struct ExactStateCache<E> {
    max_entries: usize,
    max_bytes: u64,
    clock: u64,
    logical_bytes: u64,
    blobs: CacheBlobStore,
    entries: HashMap<String, ExactStateEntry<E>>,
    token_count_refs: BTreeMap<u64, usize>,
    /// Optional slow tier. When present, entries evicted from RAM are written
    /// here instead of being discarded, and lookups that miss in RAM fall
    /// through to it.
    disk: Option<PrefixDiskTier>,
    /// Per-entry metadata for disk-only archives (notably the KV page
    /// descriptor). Disk entries store raw bytes; the descriptor is what makes
    /// them importable, and it is small enough to keep in RAM alongside.
    disk_extras: HashMap<String, E>,
    misses: PrefixMissTracker,
}

#[derive(Debug, Clone)]
struct ExactStateEntry<E> {
    token_count: u64,
    logical_bytes: u64,
    last_used: u64,
    payload: ExactStatePayload,
    extra: E,
}

#[derive(Debug, Clone)]
pub struct ExactStateLookup<E> {
    pub page_id: String,
    pub token_count: u64,
    pub logical_bytes: u64,
    pub entries: usize,
    pub payload: ExactStatePayload,
    pub extra: E,
    /// True when the payload was served from the disk tier rather than RAM.
    pub from_disk: bool,
}

#[derive(Debug, Clone)]
pub struct ExactStateRecordOutcome {
    pub stored: bool,
    pub page_id: String,
    pub token_count: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub entries: usize,
    pub evicted_entries: usize,
    pub evicted_logical_bytes: u64,
    pub dedupe: CacheDedupeStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExactStateCacheStats {
    pub entries: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub block_count: usize,
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl<E: Clone + serde::Serialize> ExactStateCache<E> {
    pub fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries: max_entries.max(1),
            max_bytes,
            clock: 0,
            logical_bytes: 0,
            blobs: CacheBlobStore::default(),
            entries: HashMap::new(),
            token_count_refs: BTreeMap::new(),
            disk: None,
            disk_extras: HashMap::new(),
            misses: PrefixMissTracker::default(),
        }
    }

    /// Attach a disk tier so evicted prefixes survive eviction and restart.
    pub fn with_disk_tier(mut self, disk: PrefixDiskTier) -> Self {
        self.disk = Some(disk);
        self
    }

    pub fn disk_stats(&self) -> Option<crate::DiskTierStats> {
        self.disk.as_ref().map(PrefixDiskTier::stats)
    }

    pub fn miss_stats(&self) -> crate::PrefixMissStats {
        self.misses.stats()
    }

    /// Prefix lengths available in RAM or on disk, longest first.
    pub fn all_token_counts_at_most(&self, max_token_count: u64) -> Vec<u64> {
        let mut counts = self.token_counts_at_most(max_token_count);
        if let Some(disk) = self.disk.as_ref() {
            counts.extend(disk.token_counts_at_most(max_token_count));
            counts.sort_unstable_by(|left, right| right.cmp(left));
            counts.dedup();
        }
        counts
    }

    pub fn lookup(&mut self, page_id: &str) -> Option<ExactStateLookup<E>> {
        let hit = self.lookup_ram(page_id);
        if hit.is_some() {
            self.misses.note_hit(page_id);
        } else {
            self.misses.note_miss(page_id, now_secs());
        }
        hit
    }

    fn lookup_ram(&mut self, page_id: &str) -> Option<ExactStateLookup<E>> {
        self.clock = self.clock.saturating_add(1);
        let entries = self.entries.len();
        if let Some(entry) = self.entries.get_mut(page_id) {
            entry.last_used = self.clock;
            let hit = ExactStateLookup {
                page_id: page_id.to_string(),
                token_count: entry.token_count,
                logical_bytes: entry.logical_bytes,
                entries,
                payload: entry.payload.clone(),
                extra: entry.extra.clone(),
                from_disk: false,
            };
            return Some(hit);
        }
        None
    }

    /// Look up a prefix, falling back to the disk tier on a RAM miss.
    ///
    /// The RAM tier holds no metadata for a demoted entry, so `extra` (the KV
    /// page descriptor) must be supplied by the caller from the entry it is
    /// probing for. Without a descriptor a KV payload cannot be imported, so a
    /// disk hit is reported only when the caller can actually use it.
    ///
    /// A disk entry that fails verification is a hard error, not a miss:
    /// importing unverified KV bytes is silent numerical corruption.
    pub fn lookup_with_disk(
        &mut self,
        page_id: &str,
        payload_kind: crate::ExactStatePayloadKind,
        extra: impl FnOnce() -> E,
    ) -> Result<Option<ExactStateLookup<E>>>
    where
        E: serde::de::DeserializeOwned,
    {
        if let Some(hit) = self.lookup_ram(page_id) {
            self.misses.note_hit(page_id);
            return Ok(Some(hit));
        }
        let cached_extra = self.disk_extras.get(page_id).cloned();
        let Some(disk) = self.disk.as_mut() else {
            self.misses.note_miss(page_id, now_secs());
            return Ok(None);
        };
        let loaded = disk.load(page_id, payload_kind.as_str())?;
        // A failed verification quarantines the entry inside `load`, so prune
        // before deciding anything about the result.
        self.prune_disk_extras();
        let Some(load) = loaded else {
            self.misses.note_miss(page_id, now_secs());
            return Ok(None);
        };

        // Prefer this process's copy, then the persisted one, and only then
        // the caller's fallback. For a payload that is not self-describing the
        // persisted descriptor is what makes the bytes importable at all, so
        // ignoring it turned a restorable page into a silent miss.
        let extra = cached_extra
            .or_else(|| {
                load.extra
                    .clone()
                    .and_then(|value| serde_json::from_value::<E>(value).ok())
            })
            .unwrap_or_else(extra);
        let payload = ExactStatePayload::from_disk_components(payload_kind, load.components)?;
        self.misses.note_hit(page_id);
        Ok(Some(ExactStateLookup {
            page_id: page_id.to_string(),
            token_count: load.token_count,
            logical_bytes: payload.byte_len(),
            entries: self.entries.len(),
            payload,
            extra,
            from_disk: true,
        }))
    }

    /// Classify a lookup that failed because the stored page is incompatible
    /// with the current stage configuration.
    pub fn note_identity_mismatch(&mut self) {
        self.misses.note_identity_mismatch();
    }

    /// Mark an existing page as recently used without reconstructing or
    /// replacing its payload.
    ///
    /// Page identities include the complete prefix identity, so an existing
    /// entry is already the checkpoint the caller intends to record. This
    /// lets record paths avoid exporting and re-hashing the same state.
    pub fn touch(&mut self, page_id: &str) -> bool {
        self.clock = self.clock.saturating_add(1);
        if let Some(entry) = self.entries.get_mut(page_id) {
            entry.last_used = self.clock;
            return true;
        }
        self.disk.as_mut().is_some_and(|disk| disk.touch(page_id))
    }

    pub fn disk_contains(&self, page_id: &str) -> bool {
        self.disk
            .as_ref()
            .is_some_and(|disk| disk.contains(page_id))
    }

    /// Write a payload straight to the disk tier without making it resident.
    ///
    /// This is the archive path used by families whose reuse is resident
    /// rather than serialized: the RAM copy already exists as a pinned
    /// llama.cpp sequence, so putting a second copy in the RAM tier would
    /// double the memory cost for no benefit. `extra` is retained so a later
    /// restore has the page descriptor it needs to import the bytes.
    ///
    /// Returns whether the entry was written.
    pub fn store_on_disk(
        &mut self,
        page_id: &str,
        token_count: u64,
        kind: crate::ExactStatePayloadKind,
        components: &[&[u8]],
        extra: E,
    ) -> bool {
        let Some(disk) = self.disk.as_mut() else {
            return false;
        };
        // Persist the metadata with the bytes. Without it a restart keeps the
        // payload and loses the ability to import it.
        let encoded = serde_json::to_value(&extra).ok();
        if disk
            .store(page_id, token_count, kind.as_str(), components, encoded)
            .is_err()
        {
            return false;
        }
        let stored = disk.contains(page_id);
        if stored {
            self.disk_extras.insert(page_id.to_string(), extra);
        }
        self.prune_disk_extras();
        stored
    }

    /// Look up a page in the disk tier only, bypassing the RAM tier.
    pub fn lookup_disk_only(
        &mut self,
        page_id: &str,
        kind: crate::ExactStatePayloadKind,
    ) -> Result<Option<ExactStateLookup<E>>>
    where
        E: serde::de::DeserializeOwned,
    {
        let cached_extra = self.disk_extras.get(page_id).cloned();
        let Some(disk) = self.disk.as_mut() else {
            return Ok(None);
        };
        let Some(load) = disk.load(page_id, kind.as_str())? else {
            return Ok(None);
        };
        // Prefer the in-RAM copy; fall back to the persisted one so entries
        // written by a previous process are still importable.
        let extra = match cached_extra {
            Some(extra) => extra,
            None => match load
                .extra
                .clone()
                .and_then(|value| serde_json::from_value::<E>(value).ok())
            {
                Some(extra) => extra,
                // Bytes without usable metadata cannot be imported safely.
                None => return Ok(None),
            },
        };
        let payload = ExactStatePayload::from_disk_components(kind, load.components)?;
        self.misses.note_hit(page_id);
        Ok(Some(ExactStateLookup {
            page_id: page_id.to_string(),
            token_count: load.token_count,
            logical_bytes: payload.byte_len(),
            entries: self.entries.len(),
            payload,
            extra,
            from_disk: true,
        }))
    }

    pub fn token_counts_at_most(&self, max_token_count: u64) -> Vec<u64> {
        self.token_count_refs
            .range(..=max_token_count)
            .rev()
            .map(|(token_count, _)| *token_count)
            .collect()
    }

    pub fn record(
        &mut self,
        page_id: String,
        token_count: u64,
        payload: ExactStatePayload,
        extra: E,
    ) -> ExactStateRecordOutcome {
        self.clock = self.clock.saturating_add(1);
        if let Some(previous) = self.entries.remove(&page_id) {
            self.remove_entry(&page_id, previous);
        }

        let logical_bytes = payload.byte_len();
        let (payload, dedupe) = payload.dedupe_into(&mut self.blobs);
        self.logical_bytes = self.logical_bytes.saturating_add(logical_bytes);
        self.add_token_count(token_count);
        self.entries.insert(
            page_id.clone(),
            ExactStateEntry {
                token_count,
                logical_bytes,
                last_used: self.clock,
                payload,
                extra,
            },
        );

        self.misses.note_recorded(&page_id);
        let (mut evicted_entries, mut evicted_logical_bytes) = self.evict_until_within_limits();
        if self.max_bytes > 0
            && self.blobs.physical_bytes() > self.max_bytes
            && let Some(entry) = self.entries.remove(&page_id)
        {
            evicted_entries = evicted_entries.saturating_add(1);
            evicted_logical_bytes = evicted_logical_bytes.saturating_add(entry.logical_bytes);
            self.remove_entry(&page_id, entry);
        }
        let stored = self.entries.contains_key(&page_id);
        let stats = self.stats();
        ExactStateRecordOutcome {
            stored,
            page_id,
            token_count,
            logical_bytes,
            physical_bytes: stats.physical_bytes,
            entries: stats.entries,
            evicted_entries,
            evicted_logical_bytes,
            dedupe,
        }
    }

    pub fn stats(&self) -> ExactStateCacheStats {
        ExactStateCacheStats {
            entries: self.entries.len(),
            logical_bytes: self.logical_bytes,
            physical_bytes: self.blobs.physical_bytes(),
            block_count: self.blobs.block_count(),
            max_entries: self.max_entries,
            max_bytes: self.max_bytes,
        }
    }

    fn evict_until_within_limits(&mut self) -> (usize, u64) {
        let mut evicted_entries = 0usize;
        let mut evicted_logical_bytes = 0u64;
        loop {
            let over_entries = self.entries.len() > self.max_entries;
            let over_bytes = self.max_bytes > 0 && self.blobs.physical_bytes() > self.max_bytes;
            if !over_entries && !over_bytes {
                break;
            }
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(page_id, _)| page_id.clone())
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                evicted_entries = evicted_entries.saturating_add(1);
                evicted_logical_bytes = evicted_logical_bytes.saturating_add(entry.logical_bytes);
                self.remove_entry(&victim, entry);
            }
        }
        (evicted_entries, evicted_logical_bytes)
    }

    /// Remove an entry from RAM, demoting it to the disk tier first.
    ///
    /// Demotion happens before the blob references are released, because the
    /// payload must still be readable to be written out.
    ///
    /// A demotion failure must never fail the request that triggered the
    /// eviction: the disk tier is an optimisation, and a full or unwritable
    /// disk should degrade to today's behaviour (state is lost, next request
    /// recomputes) rather than surface an error on the serving path.
    fn remove_entry(&mut self, page_id: &str, entry: ExactStateEntry<E>) {
        self.demote_to_disk(page_id, &entry);
        self.logical_bytes = self.logical_bytes.saturating_sub(entry.logical_bytes);
        self.remove_token_count(entry.token_count);
        // Mapped payloads hold no blob references; `release_from` is a no-op
        // for them, so demoted-then-reinserted entries cannot double-release.
        entry.payload.release_from(&mut self.blobs);
        self.misses
            .note_evicted(page_id, entry.token_count, now_secs());
    }

    fn demote_to_disk(&mut self, page_id: &str, entry: &ExactStateEntry<E>) {
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        // Already on disk: this entry was promoted from it and never rewritten.
        if entry.payload.is_mapped() || disk.contains(page_id) {
            return;
        }
        let Ok(components) = entry.payload.disk_components() else {
            return;
        };
        let borrowed: Vec<&[u8]> = components.iter().map(|bytes| bytes.as_ref()).collect();
        // Persist the metadata with the bytes, exactly as `store_on_disk`
        // does. A recurrent KV payload is not self-describing: without its
        // page descriptor the bytes cannot be imported, so an entry demoted
        // with no metadata comes back after a restart as a permanent silent
        // miss -- disk consumed, nothing restorable.
        //
        // A unit metadata type encodes as JSON null, which round-trips as an
        // absent field; normalise it so absent and null mean the same thing
        // and the metadata checksum still matches.
        let encoded = serde_json::to_value(&entry.extra)
            .ok()
            .filter(|value| !value.is_null());
        let stored = disk
            .store(
                page_id,
                entry.token_count,
                entry.payload.kind().as_str(),
                &borrowed,
                encoded,
            )
            .is_ok()
            && disk.contains(page_id);
        if stored {
            self.disk_extras
                .insert(page_id.to_string(), entry.extra.clone());
        }
        self.prune_disk_extras();
    }

    /// Drop descriptors whose page is no longer on disk.
    ///
    /// `disk_extras` is only meaningful for a page the tier still holds: it is
    /// the descriptor that makes those bytes importable. The tier removes
    /// entries on its own schedule — LRU eviction in `enforce_budget`,
    /// `quarantine` for a page that fails verification — and those paths know
    /// nothing about this map. Without pruning, a long-lived process that
    /// churns the tier accumulates a descriptor per page it ever wrote,
    /// none of which can ever be used again.
    ///
    /// Called after each operation that can remove a disk entry rather than
    /// from the tier itself, because the tier is payload-agnostic and does not
    /// know the descriptor type.
    fn prune_disk_extras(&mut self) {
        let Some(disk) = self.disk.as_ref() else {
            self.disk_extras.clear();
            return;
        };
        if self.disk_extras.len() <= disk.entry_count() {
            return;
        }
        self.disk_extras.retain(|page_id, _| disk.contains(page_id));
    }

    fn add_token_count(&mut self, token_count: u64) {
        *self.token_count_refs.entry(token_count).or_default() += 1;
    }

    fn remove_token_count(&mut self, token_count: u64) {
        let Some(count) = self.token_count_refs.get_mut(&token_count) else {
            debug_assert!(false, "exact-state token-count index drifted");
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.token_count_refs.remove(&token_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ExactStatePayload, exact_state::ExactStateCache};

    #[test]
    fn exact_state_cache_evicts_lru_by_entry_cap() {
        let mut cache = ExactStateCache::new(1, 0);
        cache.record(
            "first".to_string(),
            2,
            ExactStatePayload::full_state(vec![1, 2]),
            (),
        );
        cache.record(
            "second".to_string(),
            2,
            ExactStatePayload::full_state(vec![3, 4]),
            (),
        );

        assert!(cache.lookup("first").is_none());
        assert!(cache.lookup("second").is_some());
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn touching_existing_page_refreshes_lru_without_replacing_payload() {
        let mut cache = ExactStateCache::new(2, 0);
        cache.record(
            "first".to_string(),
            2,
            ExactStatePayload::full_state(vec![1, 2]),
            (),
        );
        cache.record(
            "second".to_string(),
            2,
            ExactStatePayload::full_state(vec![3, 4]),
            (),
        );

        assert!(cache.touch("first"));
        assert!(!cache.touch("missing"));
        cache.record(
            "third".to_string(),
            2,
            ExactStatePayload::full_state(vec![5, 6]),
            (),
        );

        assert!(cache.lookup("first").is_some());
        assert!(cache.lookup("second").is_none());
        assert!(cache.lookup("third").is_some());
    }

    #[test]
    fn cached_token_counts_are_bounded_sorted_and_deduplicated() {
        let mut cache = ExactStateCache::new(4, 0);
        for (page_id, token_count) in [("a", 96), ("b", 160), ("c", 96), ("d", 224)] {
            cache.record(
                page_id.to_string(),
                token_count,
                ExactStatePayload::full_state(vec![1]),
                (),
            );
        }

        assert_eq!(cache.token_counts_at_most(200), vec![160, 96]);
    }

    #[test]
    fn cached_token_counts_track_replacement_and_eviction() {
        let mut cache = ExactStateCache::new(1, 0);
        cache.record(
            "first".to_string(),
            96,
            ExactStatePayload::full_state(vec![1]),
            (),
        );
        cache.record(
            "first".to_string(),
            160,
            ExactStatePayload::full_state(vec![2]),
            (),
        );
        assert_eq!(cache.token_counts_at_most(160), vec![160]);

        cache.record(
            "second".to_string(),
            224,
            ExactStatePayload::full_state(vec![3]),
            (),
        );
        assert_eq!(cache.token_counts_at_most(224), vec![224]);
    }

    #[test]
    fn exact_state_cache_releases_deduped_blocks_on_eviction() {
        let mut cache = ExactStateCache::new(1, 0);
        cache.record(
            "first".to_string(),
            8,
            ExactStatePayload::full_state(vec![7; 1024 * 1024]),
            (),
        );
        cache.record(
            "second".to_string(),
            8,
            ExactStatePayload::full_state(vec![7; 1024 * 1024]),
            (),
        );

        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().physical_bytes, 1024 * 1024);
        assert_eq!(cache.stats().block_count, 1);
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod disk_tier_integration_tests {
    use crate::{
        ExactStatePayload, ExactStatePayloadKind, PrefixDiskTier, PrefixMissReason,
        exact_state::ExactStateCache,
    };

    fn cache_with_disk(dir: &tempfile::TempDir, max_entries: usize) -> ExactStateCache<()> {
        let disk = PrefixDiskTier::open(dir.path(), 64 << 20).expect("open disk tier");
        ExactStateCache::new(max_entries, 0).with_disk_tier(disk)
    }

    /// A recurrent KV payload is not self-describing: without its page
    /// descriptor the bytes cannot be imported. A demoted entry that dropped
    /// its metadata came back after a restart as a permanent silent miss --
    /// disk consumed, nothing restorable.
    #[test]
    fn a_demoted_entry_keeps_the_metadata_needed_to_import_it() {
        #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Desc {
            layer_start: u32,
            layer_end: u32,
        }
        fn typed_cache(dir: &tempfile::TempDir) -> ExactStateCache<Desc> {
            let disk = PrefixDiskTier::open(dir.path(), 64 << 20).expect("open disk tier");
            // Capacity of one, so a second record demotes the first.
            ExactStateCache::new(1, 0).with_disk_tier(disk)
        }

        let dir = tempfile::tempdir().unwrap();
        let desc = Desc {
            layer_start: 0,
            layer_end: 35,
        };
        {
            let mut cache = typed_cache(&dir);
            cache.record(
                "page-a".to_string(),
                4096,
                ExactStatePayload::full_state(vec![7u8; 4096]),
                desc.clone(),
            );
            cache.record(
                "page-b".to_string(),
                64,
                ExactStatePayload::full_state(vec![0u8; 64]),
                desc.clone(),
            );
        }

        let mut restarted = typed_cache(&dir);
        let restored = restarted
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || Desc {
                layer_start: 999,
                layer_end: 999,
            })
            .unwrap()
            .expect("a demoted entry must survive a restart");
        assert_eq!(
            restored.extra, desc,
            "the descriptor must come back from disk, not from the caller fallback"
        );
    }

    /// The motivating scenario: a prefix is evicted from RAM under pressure
    /// and is still reusable afterwards instead of being lost.
    #[test]
    fn evicted_entry_is_demoted_to_disk_and_served_back() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        let bytes = vec![42u8; 8192];

        cache.record(
            "page-a".to_string(),
            2048,
            ExactStatePayload::full_state(bytes.clone()),
            (),
        );
        // Force page-a out of RAM.
        cache.record(
            "page-b".to_string(),
            2048,
            ExactStatePayload::full_state(vec![1u8; 8192]),
            (),
        );

        // Gone from RAM...
        assert!(cache.lookup("page-a").is_none());
        // ...but recoverable from disk, byte-for-byte.
        let restored = cache
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .expect("evicted entry should be served from the disk tier");

        assert!(restored.from_disk);
        assert_eq!(restored.token_count, 2048);
        assert_eq!(
            restored
                .payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .as_ref(),
            &bytes[..]
        );
    }

    /// Restores must borrow the mapping. Copying a multi-GB page defeats the
    /// entire point of using mmap.
    #[test]
    fn disk_restores_borrow_the_mapping_instead_of_copying() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        cache.record(
            "page-a".to_string(),
            64,
            ExactStatePayload::full_state(vec![7u8; 4096]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            64,
            ExactStatePayload::full_state(vec![8u8; 4096]),
            (),
        );

        let restored = cache
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .unwrap();

        assert!(restored.payload.is_mapped());
    }

    /// A payload promoted off disk holds no blob references, so re-evicting it
    /// must not release blocks it never acquired. Getting this wrong frees
    /// blocks still referenced by other entries.
    #[test]
    fn promote_then_evict_keeps_blob_accounting_correct() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        let shared = vec![5u8; 1024 * 1024];

        cache.record(
            "page-a".to_string(),
            64,
            ExactStatePayload::full_state(shared.clone()),
            (),
        );
        cache.record(
            "page-b".to_string(),
            64,
            ExactStatePayload::full_state(shared.clone()),
            (),
        );
        // page-a demoted; page-b resident and holding the shared block.
        let restored = cache
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .unwrap();
        assert!(restored.payload.is_mapped());

        // Re-record the mapped payload, evicting page-b, then evict it again.
        cache.record("page-a".to_string(), 64, restored.payload, ());
        cache.record(
            "page-c".to_string(),
            64,
            ExactStatePayload::full_state(vec![6u8; 1024 * 1024]),
            (),
        );

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        // Exactly one resident payload's worth of blocks is held.
        assert_eq!(stats.physical_bytes, 1024 * 1024);
        assert_eq!(stats.block_count, 1);
    }

    /// Reuse across a process restart is the point of the persistent index.
    #[test]
    fn prefixes_survive_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = vec![3u8; 4096];
        {
            let mut cache = cache_with_disk(&dir, 1);
            cache.record(
                "page-a".to_string(),
                4096,
                ExactStatePayload::full_state(bytes.clone()),
                (),
            );
            cache.record(
                "page-b".to_string(),
                64,
                ExactStatePayload::full_state(vec![0u8; 64]),
                (),
            );
        }

        let mut restarted = cache_with_disk(&dir, 1);
        let restored = restarted
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .expect("prefix should survive a restart");

        assert_eq!(
            restored
                .payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .as_ref(),
            &bytes[..]
        );
    }

    /// Disk lengths must be visible to the lookup ladder, or a demoted prefix
    /// is never probed for and the tier is dead weight.
    #[test]
    fn disk_prefix_lengths_are_visible_to_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        cache.record(
            "page-a".to_string(),
            2048,
            ExactStatePayload::full_state(vec![1u8; 512]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            1024,
            ExactStatePayload::full_state(vec![2u8; 512]),
            (),
        );

        // page-a is on disk, page-b in RAM; both lengths must be probed.
        let counts = cache.all_token_counts_at_most(4096);
        assert!(
            counts.contains(&2048),
            "disk length missing from {counts:?}"
        );
        assert!(counts.contains(&1024), "RAM length missing from {counts:?}");
    }

    /// Miss classification is the gate for the whole retention effort, so it
    /// must actually distinguish the two cases.
    #[test]
    fn miss_reasons_distinguish_eviction_from_never_seen() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        cache.record(
            "page-a".to_string(),
            64,
            ExactStatePayload::full_state(vec![1u8; 64]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            64,
            ExactStatePayload::full_state(vec![2u8; 64]),
            (),
        );

        cache.lookup("page-a");
        cache.lookup("page-never-seen");

        let stats = cache.miss_stats();
        assert_eq!(stats.misses_for(PrefixMissReason::EvictedRecently), 1);
        assert_eq!(stats.misses_for(PrefixMissReason::NeverSeen), 1);
    }

    /// Without a disk tier the cache must behave exactly as before.
    #[test]
    fn cache_without_a_disk_tier_is_unchanged() {
        let mut cache = ExactStateCache::<()>::new(1, 0);
        cache.record(
            "page-a".to_string(),
            64,
            ExactStatePayload::full_state(vec![1u8; 64]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            64,
            ExactStatePayload::full_state(vec![2u8; 64]),
            (),
        );

        assert!(cache.lookup("page-a").is_none());
        assert!(
            cache
                .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
                .unwrap()
                .is_none()
        );
        assert!(cache.disk_stats().is_none());
    }

    /// A KV-recurrent payload has two components and both must round-trip
    /// intact, in order.
    #[test]
    fn kv_recurrent_payloads_round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        let kv = vec![1u8; 2048];
        let recurrent = vec![2u8; 512];

        cache.record(
            "page-a".to_string(),
            256,
            ExactStatePayload::kv_recurrent(kv.clone(), recurrent.clone()),
            (),
        );
        cache.record(
            "page-b".to_string(),
            256,
            ExactStatePayload::full_state(vec![0u8; 64]),
            (),
        );

        let restored = cache
            .lookup_with_disk("page-a", ExactStatePayloadKind::KvRecurrent, || ())
            .unwrap()
            .unwrap();

        assert_eq!(
            restored.payload.kv_bytes().unwrap().unwrap().as_ref(),
            &kv[..]
        );
        assert_eq!(
            restored.payload.recurrent_state_bytes().unwrap().as_ref(),
            &recurrent[..]
        );
    }

    #[test]
    fn resident_kv_archives_round_trip_without_a_recurrent_component() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        let kv = vec![3u8; 2048];

        cache.store_on_disk(
            "page-a",
            256,
            ExactStatePayloadKind::ResidentKvArchive,
            &[&kv],
            (),
        );

        let restored = cache
            .lookup_disk_only("page-a", ExactStatePayloadKind::ResidentKvArchive)
            .unwrap()
            .unwrap();

        assert_eq!(
            restored.payload.kind(),
            ExactStatePayloadKind::ResidentKvArchive
        );
        assert!(restored.payload.is_mapped());
        assert_eq!(
            restored.payload.kv_bytes().unwrap().unwrap().as_ref(),
            &kv[..]
        );
        assert!(restored.payload.recurrent_state_bytes().is_err());
    }

    #[test]
    fn disk_hit_is_not_counted_as_a_ram_miss() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        cache.record(
            "page-a".to_string(),
            256,
            ExactStatePayload::full_state(vec![1u8; 2048]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            256,
            ExactStatePayload::full_state(vec![2u8; 2048]),
            (),
        );

        let restored = cache
            .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .unwrap();

        assert!(restored.from_disk);
        let stats = cache.miss_stats();
        assert_eq!(stats.total_misses(), 0);
        assert_eq!(stats.hits, 1);
    }

    /// `RecurrentOnly` had no disk coverage at all: every payload-kind test
    /// exercised `FullState`, `KvRecurrent` or `ResidentKvArchive`, so a
    /// single-component recurrent payload could have been broken on the disk
    /// path without any test noticing.
    #[test]
    fn recurrent_only_payloads_round_trip_through_the_disk_tier() {
        let dir = tempfile::tempdir().unwrap();
        let recurrent = vec![5u8; 3072];
        {
            let mut cache = cache_with_disk(&dir, 1);
            cache.record(
                "page-a".to_string(),
                1024,
                ExactStatePayload::recurrent_only(recurrent.clone()),
                (),
            );
            // Force the demotion of page-a.
            cache.record(
                "page-b".to_string(),
                64,
                ExactStatePayload::recurrent_only(vec![0u8; 64]),
                (),
            );
        }

        // Reopened process: the entry must still be there and still readable.
        let mut restarted = cache_with_disk(&dir, 1);
        let restored = restarted
            .lookup_with_disk("page-a", ExactStatePayloadKind::RecurrentOnly, || ())
            .unwrap()
            .expect("recurrent-only prefix should survive demotion and restart");

        assert!(restored.from_disk);
        assert_eq!(restored.token_count, 1024);
        assert_eq!(
            restored.payload.kind(),
            ExactStatePayloadKind::RecurrentOnly
        );
        assert!(restored.payload.is_mapped());
        assert_eq!(
            restored.payload.recurrent_state_bytes().unwrap().as_ref(),
            &recurrent[..]
        );
    }

    /// Every stored kind must reject every other kind. The single-direction
    /// test that existed left five of the six cross-kind pairs unchecked, and
    /// a mismatch that is *not* caught is a payload reinterpreted under the
    /// wrong layout -- silent numerical corruption, not a miss.
    #[test]
    fn every_payload_kind_rejects_a_lookup_for_a_different_kind() {
        use ExactStatePayloadKind::{FullState, KvRecurrent, RecurrentOnly, ResidentKvArchive};

        let all = [FullState, RecurrentOnly, KvRecurrent, ResidentKvArchive];
        for stored in all {
            for requested in all {
                if stored == requested {
                    continue;
                }
                let dir = tempfile::tempdir().unwrap();
                let mut cache = cache_with_disk(&dir, 1);
                let components: Vec<Vec<u8>> = (0..stored.disk_component_count())
                    .map(|index| vec![index as u8 + 1; 256])
                    .collect();
                let borrowed: Vec<&[u8]> =
                    components.iter().map(|bytes| bytes.as_slice()).collect();
                assert!(
                    cache.store_on_disk("page-a", 256, stored, &borrowed, ()),
                    "storing {stored:?} should succeed"
                );

                assert!(
                    cache.lookup_disk_only("page-a", requested).is_err(),
                    "stored {stored:?} must not be served as {requested:?}"
                );
            }
        }
    }

    /// A quarantined entry must stay rejected, not become a silent hit on the
    /// next probe once the mapping is cached.
    #[test]
    fn a_kind_mismatch_quarantines_the_entry_for_later_correct_lookups_too() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        assert!(cache.store_on_disk(
            "page-a",
            256,
            ExactStatePayloadKind::ResidentKvArchive,
            &[&[7u8; 256][..]],
            (),
        ));

        assert!(
            cache
                .lookup_disk_only("page-a", ExactStatePayloadKind::FullState)
                .is_err()
        );
        assert!(
            cache
                .lookup_disk_only("page-a", ExactStatePayloadKind::ResidentKvArchive)
                .unwrap()
                .is_none(),
            "a quarantined entry must not come back"
        );
    }

    /// Asking for the wrong payload kind must fail loudly rather than
    /// reinterpret the bytes.
    #[test]
    fn payload_kind_mismatch_on_restore_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = cache_with_disk(&dir, 1);
        cache.record(
            "page-a".to_string(),
            64,
            ExactStatePayload::kv_recurrent(vec![1u8; 64], vec![2u8; 64]),
            (),
        );
        cache.record(
            "page-b".to_string(),
            64,
            ExactStatePayload::full_state(vec![0u8; 64]),
            (),
        );

        assert!(
            cache
                .lookup_with_disk("page-a", ExactStatePayloadKind::FullState, || ())
                .is_err()
        );
    }
}
