//! Bounded L1 retention of serialized exact-state checkpoints.

use std::sync::Mutex;

use anyhow::Result;
use skippy_cache::{CacheBlobStore, UnifiedRadixCache};

use super::super::exact_state_diagnostics::emit_exact_state_diagnostic;
use super::super::{RadixExactEntry, RadixResidentEntry};

/// Evict least-recently-used exact entries above a byte limit, retaining at
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
        let above_floor = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats()
            .recurrent_entries
            > floor;
        if physical_bytes <= limit_bytes || !above_floor {
            break;
        }
        let evicted = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evict_lru_recurrent();
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
