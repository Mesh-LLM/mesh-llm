//! Disk retention for dense-attention families.
//!
//! # Why this file exists
//!
//! Dense families — Llama, Qwen3, DeepSeek, GLM4, Gemma, MiniMax — use the
//! `ResidentKv` payload, which keeps a prefix *resident* on a dedicated
//! llama.cpp sequence and reuses it in place. That is the fastest possible
//! reuse while the prefix fits, but it has no serialized form: when the
//! resident cache evicts an entry it calls `skippy_session_drop_sequence` and
//! the state is simply gone.
//!
//! Only `KvRecurrent` and `FullState` payloads flow through `ExactStateCache`,
//! so attaching a disk tier there alone would produce a feature that helps
//! hybrid/recurrent models and does nothing at all for the models people
//! actually run. This module closes that gap by giving dense prefixes a
//! serialized archive.
//!
//! # Why archive at record time, not at eviction time
//!
//! Eviction runs on the **decode hot path**: `evict_resident_prefix_for_tokens`
//! is called from binary execution to free KV cells before a decode batch.
//! Exporting hundreds of megabytes there would spike TTFT badly, and deferring
//! the export asynchronously means the llama.cpp sequence cannot be dropped
//! until the export completes — a lifecycle change that risks either
//! use-after-drop or a leaked cell that re-triggers the "failed to find a
//! memory slot" wedge that `max_resident_tokens` exists to prevent.
//!
//! Recording is the safe point. The prefix has just been prefilled, the
//! session is alive and quiescent, and the tokens are already known-good. The
//! archive is written once, and eviction stays exactly as cheap as it is
//! today. The cost is one export per newly recorded prefix, bounded by the
//! archive admission policy below.

use anyhow::Result;
use skippy_cache::ExactStatePayloadKind;

use crate::runtime_state::RuntimeState;

use super::{ExactStateExtra, KvStageIntegration, PrefillKvIdentity, StagePrefixCachePayload};

/// Only archive prefixes large enough that restoring beats recomputing.
///
/// Prefill is roughly quadratic in prefix length while a restore is linear in
/// bytes, so the disk tier wins by a wider margin the larger the prefix. Below
/// this floor the export and the write cost more than the prefill they would
/// save, so short prefixes stay RAM-only.
const MIN_ARCHIVE_TOKENS: u64 = 512;

impl KvStageIntegration {
    /// Whether dense prefixes should be archived to the disk tier.
    fn dense_archive_enabled(&self) -> bool {
        self.payload == StagePrefixCachePayload::ResidentKv
            && self
                .exact_states
                .lock()
                .expect("exact state cache lock poisoned")
                .disk_stats()
                .is_some()
    }

    /// Archive a freshly recorded dense prefix so it outlives resident
    /// eviction and process restart.
    ///
    /// Failures are deliberately swallowed into `Ok(())`: the archive is an
    /// optimisation, and neither a full disk nor a runtime that declines the
    /// export should fail the request that triggered it. The prefix simply
    /// stays RAM-only, which is exactly today's behaviour.
    pub fn archive_dense_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
    ) -> Result<bool> {
        if !self.dense_archive_enabled() {
            return Ok(false);
        }
        let token_count = identity.identity.token_count;
        if token_count < MIN_ARCHIVE_TOKENS.max(self.candidate_policy.min_tokens) {
            return Ok(false);
        }
        {
            let cache = self
                .exact_states
                .lock()
                .expect("exact state cache lock poisoned");
            if cache.disk_contains(&identity.page_id) {
                return Ok(false);
            }
        }

        // Export the exact token range this page id was computed over, so the
        // archived bytes and the identity always agree.
        let page = match runtime.export_kv_page(session_id, 0, token_count) {
            Ok(page) => page,
            Err(_) => return Ok(false),
        };
        let mut cache = self
            .exact_states
            .lock()
            .expect("exact state cache lock poisoned");
        Ok(cache.store_on_disk(
            &identity.page_id,
            token_count,
            ExactStatePayloadKind::ResidentKvArchive,
            &[&page.payload],
            ExactStateExtra {
                kv_desc: Some(page.desc),
            },
        ))
    }

    /// Restore a dense prefix from the disk tier after a resident-cache miss.
    ///
    /// Returns the number of tokens restored. The caller must treat this as a
    /// prefix restore exactly like a resident hit: the session now holds
    /// `token_count` tokens and only the divergent tail needs prefilling.
    pub fn restore_dense_prefix_from_disk(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<DenseDiskRestore>> {
        if !self.should_lookup() || !self.dense_archive_enabled() {
            return Ok(None);
        }
        for identity in identities {
            let restored = {
                let mut cache = self
                    .exact_states
                    .lock()
                    .expect("exact state cache lock poisoned");
                // A verification failure is a hard error inside the tier and
                // quarantines the entry; treat it here as a miss and continue
                // probing shorter candidates rather than failing the request.
                match cache
                    .lookup_disk_only(&identity.page_id, ExactStatePayloadKind::ResidentKvArchive)
                {
                    Ok(found) => found,
                    Err(_) => continue,
                }
            };
            let Some(restored) = restored else {
                continue;
            };
            let Some(desc) = restored.extra.kv_desc else {
                // Without the page descriptor the bytes cannot be imported.
                continue;
            };
            let Ok(Some(kv)) = restored.payload.kv_bytes() else {
                continue;
            };

            // The payload is checksummed, but the *metadata* describing where
            // those bytes belong is plain JSON in the index and is not. If the
            // descriptor's token range disagrees with the identity we looked
            // up, the runtime's `n_past` and the caller's restored-token count
            // diverge and the suffix prefill is applied at the wrong position
            // — silent numerical corruption on a path that looks verified.
            // Require exact agreement rather than trusting the index.
            let expected_tokens = identity.identity.token_count;
            if desc.token_start != 0
                || desc.token_count != expected_tokens
                || restored.token_count != expected_tokens
            {
                continue;
            }

            // Import borrows the mapped bytes directly; no copy is made.
            if runtime
                .import_kv_page(session_id, &desc, kv.as_ref())
                .is_err()
            {
                // A failed import can leave partially written KV cells behind
                // while the host still believes the session holds none.
                // Importing a different-length page on top of that would
                // compound the inconsistency, so stop probing entirely and let
                // the caller fall back to a full prefill.
                return Ok(None);
            }
            return Ok(Some(DenseDiskRestore {
                page_id: identity.page_id.clone(),
                token_count: restored.token_count,
            }));
        }
        Ok(None)
    }
}

/// A dense prefix served back from the disk tier.
#[derive(Debug, Clone)]
pub struct DenseDiskRestore {
    pub page_id: String,
    pub token_count: u64,
}
