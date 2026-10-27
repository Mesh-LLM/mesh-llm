//! Bounded L1 retention of serialized exact-state checkpoints.

use std::sync::Mutex;

use anyhow::Result;
use skippy_cache::{
    CacheBlobStore, ExactStatePayloadKind, RadixEviction, RadixEvictionCandidate, UnifiedRadixCache,
};
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

/// Within one stage, an older recurrent checkpoint already followed by a
/// newer, longer checkpoint on the same token path is a better eviction
/// candidate than the sole checkpoint on another branch. Rank those older
/// ancestors first, then use saved prefill work per logical snapshot byte and
/// recency within each group. Recheck physical bytes after each release because
/// deduplicated blocks may not free their full logical size.
fn evict_lowest_recompute_value(
    radix: &mut UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>,
) -> Option<RadixEviction<RadixExactEntry>> {
    let candidates = radix.recurrent_eviction_candidates();
    let superseded_prefixes = superseded_recurrent_prefixes(&candidates);
    let (mut superseded, mut other): (Vec<_>, Vec<_>) = candidates
        .iter()
        .enumerate()
        .partition(|(index, _)| superseded_prefixes[*index]);
    let ranked = [&mut superseded, &mut other].map(|group| {
        let mut ranked = group
            .iter()
            .map(|(index, candidate)| EvictableCacheEntry {
                id: index.to_string(),
                units: candidate.logical_bytes,
                recompute_cost: u64::try_from(candidate.tokens.len()).unwrap_or(u64::MAX),
                last_used: candidate.last_used,
            })
            .collect::<Vec<_>>();
        rank_eviction_candidates(&mut ranked);
        ranked
    });
    ranked.into_iter().flatten().find_map(|entry| {
        let candidate = candidates.get(entry.id.parse::<usize>().ok()?)?;
        radix.evict_recurrent_candidate(&candidate.namespace, &candidate.tokens)
    })
}

/// Sorted token paths keep every descendant adjacent to its ancestors. Close
/// each path once and propagate its newest use to the nearest retained prefix.
fn superseded_recurrent_prefixes(
    candidates: &[RadixEvictionCandidate<RadixExactEntry>],
) -> Vec<bool> {
    let mut order = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            (candidate.value.payload.kind() == ExactStatePayloadKind::KvRecurrent).then_some(index)
        })
        .collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        candidates[left]
            .namespace
            .cmp(&candidates[right].namespace)
            .then_with(|| candidates[left].tokens.cmp(&candidates[right].tokens))
    });
    let mut newest_use = candidates
        .iter()
        .map(|entry| entry.last_used)
        .collect::<Vec<_>>();
    let mut superseded = vec![false; candidates.len()];
    let mut path: Vec<usize> = Vec::new();
    for index in order {
        while path.last().is_some_and(|&ancestor| {
            candidates[ancestor].namespace != candidates[index].namespace
                || !candidates[index]
                    .tokens
                    .starts_with(&candidates[ancestor].tokens)
                || candidates[index].tokens.len() == candidates[ancestor].tokens.len()
        }) {
            close_recurrent_path(&mut path, candidates, &mut newest_use, &mut superseded);
        }
        path.push(index);
    }
    while !path.is_empty() {
        close_recurrent_path(&mut path, candidates, &mut newest_use, &mut superseded);
    }
    superseded
}

fn close_recurrent_path(
    path: &mut Vec<usize>,
    candidates: &[RadixEvictionCandidate<RadixExactEntry>],
    newest_use: &mut [u64],
    superseded: &mut [bool],
) {
    let Some(child) = path.pop() else { return };
    if let Some(&parent) = path.last() {
        superseded[parent] |= newest_use[child] >= candidates[parent].last_used;
        newest_use[parent] = newest_use[parent].max(newest_use[child]);
    }
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

    fn insert_recurrent(
        radix: &Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>,
        blobs: &Mutex<CacheBlobStore>,
        tokens: &[i32],
        bytes: usize,
        seed: u8,
    ) {
        let (payload, _) = ExactStatePayload::kv_recurrent(vec![seed; bytes - 1], vec![seed])
            .dedupe_into(&mut blobs.lock().unwrap());
        radix
            .lock()
            .unwrap()
            .insert_recurrent(
                "stage",
                tokens,
                bytes as u64,
                RadixExactEntry::new(seed.to_string(), payload, ExactStateExtra::default(), false),
            )
            .unwrap();
    }

    #[test]
    fn byte_pressure_keeps_a_distinct_recurrent_branch_anchor() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4096));
        let unique = vec![9; 40];
        let a_short = vec![1; 50];
        let a_long = vec![1; 100];
        let b_short = vec![2; 50];
        let b_long = vec![2; 100];
        for (tokens, bytes, seed) in [
            (&unique, 80, 1),
            (&a_short, 100, 2),
            (&a_long, 200, 3),
            (&b_short, 100, 4),
            (&b_long, 200, 5),
        ] {
            insert_recurrent(&radix, &blobs, tokens, bytes, seed);
        }

        evict_exact_entries_over(&radix, &blobs, 600, 4, "soft").unwrap();

        assert_eq!(blobs.lock().unwrap().physical_bytes(), 580);
        let mut radix = radix.lock().unwrap();
        assert!(radix.recurrent_exact("stage", &unique).is_some());
        assert!(radix.recurrent_exact("stage", &a_short).is_none());
        assert!(radix.recurrent_exact("stage", &a_long).is_some());
        assert!(radix.recurrent_exact("stage", &b_short).is_some());
        assert!(radix.recurrent_exact("stage", &b_long).is_some());
    }

    #[test]
    fn recently_reused_shorter_prefix_is_not_superseded() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4096));
        let unique = vec![9; 40];
        let short = vec![1; 50];
        let long = vec![1; 100];
        for (tokens, bytes, seed) in [(&unique, 80, 1), (&short, 100, 2), (&long, 200, 3)] {
            insert_recurrent(&radix, &blobs, tokens, bytes, seed);
        }

        let mut radix = radix.lock().unwrap();
        assert_eq!(
            radix
                .lookup_recurrent("stage", &short)
                .unwrap()
                .matched_tokens,
            50
        );
        let evicted = evict_lowest_recompute_value(&mut radix).unwrap();
        assert_eq!(evicted.tokens, unique);
        assert!(radix.recurrent_exact("stage", &short).is_some());
    }

    #[test]
    fn deep_descendant_recency_preserves_the_recent_middle_checkpoint() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4096));
        let short = vec![1; 25];
        let middle = vec![1; 50];
        let long = vec![1; 75];
        insert_recurrent(&radix, &blobs, &short, 80, 1);
        insert_recurrent(&radix, &blobs, &middle, 100, 2);
        radix
            .lock()
            .unwrap()
            .lookup_recurrent("stage", &short)
            .unwrap();
        insert_recurrent(&radix, &blobs, &long, 120, 3);
        radix
            .lock()
            .unwrap()
            .lookup_recurrent("stage", &middle)
            .unwrap();

        let candidates = radix.lock().unwrap().recurrent_eviction_candidates();
        let superseded = superseded_recurrent_prefixes(&candidates);
        let at = |tokens: &[i32]| {
            let index = candidates
                .iter()
                .position(|candidate| candidate.tokens == tokens)
                .unwrap();
            superseded[index]
        };
        assert!(
            at(&short),
            "the newer middle checkpoint covers the short path"
        );
        assert!(
            !at(&middle),
            "the long descendant predates the middle reuse"
        );
        assert!(
            !at(&long),
            "the terminal descendant has no longer checkpoint"
        );
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
