//! Exact-state working-set retention and its over-limit diagnostic.
//!
//! Exact snapshots are indivisible, and the byte limits are estimated from
//! attention KV metadata that cannot include architecture-specific recurrent
//! state, so one snapshot can legitimately exceed a limit. A working set is
//! held past the limits (never more than the configured entry limit) so
//! concurrent sessions on a stage stop evicting each other, then the hard
//! limit bounds the allowance: an automatic estimate keeps the working-set
//! floor, because the estimate structurally undercounts hybrid snapshots and
//! collapsing the catalog defeats the working set it exists to protect. An
//! explicitly configured limit is operator truth instead of an estimate, so
//! only the documented single indivisible snapshot may remain above it.
//! Retention past a nonzero hard limit is observable through one warning per
//! process.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use super::*;

/// Keep this many snapshots (subject to the configured entry limit) before a
/// byte limit evicts anything: falling to one entry turns the catalog into a
/// single-entry cache, where two concurrent sessions on the same stage evict
/// each other on every request.
pub(super) const EXACT_STATE_MIN_RETAINED_ENTRIES: usize = 4;

/// What the retention passes did to the catalog.
pub(super) struct RetentionOutcome {
    /// Entries both passes evicted, combined.
    pub(super) evicted_entries: usize,
    /// Physical bytes still retained after the passes.
    pub(super) retained_physical_bytes: u64,
    /// Entries the hard pass never drops: the automatic-estimate working set,
    /// or the single indivisible snapshot for an explicit limit.
    pub(super) hard_limit_floor_entries: usize,
}

impl RetentionOutcome {
    /// Whether the retention floor kept the catalog above a nonzero hard byte
    /// limit after eviction. The passes only stop at their floor, so this is
    /// exactly the condition where the floor prevents compliance.
    pub(super) fn floor_prevents_hard_limit_compliance(
        &self,
        limits: &ExactStateByteLimits,
    ) -> bool {
        limits.hard_bytes > 0 && self.retained_physical_bytes > limits.hard_bytes
    }
}

/// Evicts least-recently-used exact entries while the catalog holds more than
/// `limit_bytes`, never dropping below `min_retained_entries`. A zero limit
/// means the catalog has no byte ceiling and nothing is evicted. Returns the
/// number of entries evicted.
fn evict_exact_entries_over(
    radix: &Mutex<UnifiedRadixCache<super::super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    limit_bytes: u64,
    min_retained_entries: usize,
) -> Result<usize> {
    if limit_bytes == 0 {
        return Ok(0);
    }
    let floor = min_retained_entries.max(1);
    let mut evicted_total = 0usize;
    loop {
        let over_bytes = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .physical_bytes()
            > limit_bytes;
        let above_floor = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats()
            .recurrent_entries
            > floor;
        if !over_bytes || !above_floor {
            break;
        }
        let evicted = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evict_lru_recurrent();
        let Some(evicted) = evicted else {
            break;
        };
        evicted_total += 1;
        evicted.value.payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
    }
    Ok(evicted_total)
}

/// Holds the concurrent-session working set past both byte limits, then
/// applies the hard limit with the floor its provenance deserves.
pub(super) fn retain_working_set(
    radix: &Mutex<UnifiedRadixCache<super::super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    limits: &ExactStateByteLimits,
    max_entries: usize,
) -> Result<RetentionOutcome> {
    let working_set_floor = EXACT_STATE_MIN_RETAINED_ENTRIES.min(max_entries);
    // The soft cap is always the attention-derived estimate, so it keeps the
    // full working-set floor in both provenance modes.
    let soft_evicted =
        evict_exact_entries_over(radix, blobs, limits.soft_bytes, working_set_floor)?;
    let hard_limit_floor = if limits.hard_limit_explicit {
        // Operator-configured ceiling: only the documented single indivisible
        // snapshot may remain above it.
        1
    } else {
        working_set_floor
    };
    let hard_evicted = evict_exact_entries_over(radix, blobs, limits.hard_bytes, hard_limit_floor)?;
    let retained_physical_bytes = blobs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .physical_bytes();
    Ok(RetentionOutcome {
        evicted_entries: soft_evicted + hard_evicted,
        retained_physical_bytes,
        hard_limit_floor_entries: hard_limit_floor,
    })
}

static OVER_LIMIT_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

/// One warning per process when the retention floor keeps the catalog above a
/// nonzero hard byte limit: either the stage's snapshots exceed the
/// attention-derived estimate, or a configured ceiling is smaller than one
/// indivisible snapshot. Both are memory facts an operator must see rather
/// than silently absorb.
pub(super) fn warn_when_floor_prevents_hard_limit_compliance(
    outcome: &RetentionOutcome,
    limits: &ExactStateByteLimits,
) {
    if !outcome.floor_prevents_hard_limit_compliance(limits) {
        return;
    }
    if OVER_LIMIT_WARNING_EMITTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = skippy_events::diagnostics::emit(
        skippy_events::diagnostics::ServingDiagnostic::Warning {
            message: "exact-state retention floor keeps the catalog above its hard byte limit"
                .to_string(),
            context: Some(format!(
                "soft_bytes={} hard_bytes={} retained_physical_bytes={} hard_limit_floor_entries={} evicted_entries={}",
                limits.soft_bytes,
                limits.hard_bytes,
                outcome.retained_physical_bytes,
                outcome.hard_limit_floor_entries,
                outcome.evicted_entries
            )),
        },
    );
}
