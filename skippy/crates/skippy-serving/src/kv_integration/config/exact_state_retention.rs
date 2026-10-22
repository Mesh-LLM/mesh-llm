//! Bounded L1 retention of serialized exact-state checkpoints.

use std::sync::Mutex;

use anyhow::Result;
use skippy_cache::{CacheBlobStore, RadixEviction, UnifiedRadixCache};
use skippy_scheduler::{EvictableCacheEntry, rank_eviction_candidates};

use super::super::exact_state_diagnostics::emit_exact_state_diagnostic;
use super::super::{RadixExactEntry, RadixResidentEntry};

/// Evict the least expensive exact entries above a byte limit, retaining at
/// least `min_retained_entries`. Zero disables this limit. The numeric trace
/// lets a diagnostic distinguish a rejected demotion from later L1 eviction
/// without exposing token content or cache identity.
pub(super) fn evict_exact_entries_over(
    radix: &Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    limit_bytes: u64,
    min_retained_entries: usize,
    limit_kind: &'static str,
) -> Result<()> {
    if limit_bytes == 0 {
        return Ok(());
    }
    let floor = min_retained_entries.max(1);
    loop {
        let physical_bytes = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .physical_bytes();
        if physical_bytes <= limit_bytes {
            break;
        }
        let evicted = {
            let mut radix = radix
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (radix.stats().recurrent_entries > floor)
                .then(|| evict_lowest_recompute_value(&mut radix))
                .flatten()
        };
        let Some(evicted) = evicted else {
            break;
        };
        let event = if limit_kind == "hard" {
            "evicted_l1_hard"
        } else {
            "evicted_l1_soft"
        };
        emit_exact_state_diagnostic(
            event,
            &[
                ("position", evicted.tokens.len() as u64),
                ("logical_bytes", evicted.logical_bytes),
                ("physical_bytes", physical_bytes),
                ("limit_bytes", limit_bytes),
            ],
        );
        evicted.value.payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
    }
    Ok(())
}

/// Within one stage, saved prefill work grows with prefix length. Rank by
/// that work per logical snapshot byte, then recency, using the same policy
/// as resident KV capacity. Recheck physical bytes after each release because
/// deduplicated blocks may not free their full logical size.
fn evict_lowest_recompute_value(
    radix: &mut UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>,
) -> Option<RadixEviction<RadixExactEntry>> {
    let candidates = radix.recurrent_eviction_candidates();
    let mut ranked = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| EvictableCacheEntry {
            id: index.to_string(),
            units: candidate.logical_bytes,
            recompute_cost: u64::try_from(candidate.tokens.len()).unwrap_or(u64::MAX),
            last_used: candidate.last_used,
        })
        .collect::<Vec<_>>();
    rank_eviction_candidates(&mut ranked);
    ranked.into_iter().find_map(|entry| {
        let candidate = candidates.get(entry.id.parse::<usize>().ok()?)?;
        radix.evict_recurrent_candidate(&candidate.namespace, &candidate.tokens)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_integration::ExactStateExtra;
    use skippy_cache::ExactStatePayload;

    fn insert(
        radix: &Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>,
        blobs: &Mutex<CacheBlobStore>,
        tokens: &[i32],
        bytes: usize,
        seed: u8,
    ) {
        let (payload, _) = ExactStatePayload::full_state(vec![seed; bytes]).dedupe_into(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert_recurrent(
                "stage",
                tokens,
                bytes as u64,
                RadixExactEntry::new(seed.to_string(), payload, ExactStateExtra::default(), false),
            )
            .unwrap();
    }

    #[test]
    fn byte_pressure_preserves_expensive_old_continuations() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4096));
        let old_long = vec![1; 100];
        let new_long = vec![2; 101];
        insert(&radix, &blobs, &[1, 2], 6, 1);
        insert(&radix, &blobs, &old_long, 60, 2);
        insert(&radix, &blobs, &[2, 3, 4], 6, 3);
        insert(&radix, &blobs, &new_long, 60, 4);

        evict_exact_entries_over(&radix, &blobs, 120, 1, "soft").unwrap();

        assert_eq!(blobs.lock().unwrap().physical_bytes(), 120);
        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 2);
        assert!(radix.lookup_recurrent("stage", &old_long).is_some());
        assert!(radix.lookup_recurrent("stage", &new_long).is_some());
        assert!(radix.recurrent_exact("stage", &[1, 2]).is_none());
        assert!(radix.recurrent_exact("stage", &[2, 3, 4]).is_none());
    }

    #[test]
    fn byte_pressure_never_evicts_an_active_exact_state() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4096));
        let long = vec![1; 100];
        insert(&radix, &blobs, &long, 60, 1);
        insert(&radix, &blobs, &[2, 3], 6, 2);
        assert!(
            radix
                .lock()
                .unwrap()
                .acquire_recurrent("stage", &long)
                .is_some()
        );

        evict_exact_entries_over(&radix, &blobs, 5, 1, "hard").unwrap();

        assert_eq!(blobs.lock().unwrap().physical_bytes(), 60);
        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert!(radix.recurrent_exact("stage", &long).is_some());
        assert!(radix.recurrent_exact("stage", &[2, 3]).is_none());
        assert!(radix.release_recurrent("stage", &long));
    }
}
