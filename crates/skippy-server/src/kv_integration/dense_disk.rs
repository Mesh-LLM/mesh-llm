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

            // The token range agreeing is not sufficient. The descriptor also
            // tells the runtime how to *interpret* the bytes: layer range,
            // K/V ggml types, row strides. The tier checksums the descriptor
            // JSON as well as the payload (see `DiskEntry::extra_checksum`),
            // so a corrupted index cannot hand correctly-checksummed bytes to
            // the runtime under a wrong layout. Cross-check the one field
            // that must also agree with the bytes actually mapped, since that
            // relationship spans the two checksummed regions.
            if desc.payload_bytes != kv.as_ref().len() as u64 {
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

/// Picks which recorded candidate to archive, at most one per request.
///
/// The naive choices are both wrong:
///
/// - **Longest** is the request's own full length, including its unique tail.
///   Nothing else ever probes for it.
/// - **Lowest** is maximally shareable but tiny. Restoring 256 tokens of a
///   2129-token prompt saves 12% of the prefill -- indistinguishable from
///   noise, which is exactly what a split restart measured before this
///   existed.
///
/// The useful candidate is the **longest one strictly shorter than the full
/// prompt**: the largest stride-aligned prefix that excludes this request's
/// tail. For an agent workload that is the shared system-prompt-plus-tool-
/// schema bulk, so a restore covers nearly the whole prefill while still
/// matching a different session's divergent tail.
///
/// Archiving is capped at one page per request because each one is a full KV
/// export plus a synced write, and on the binary path that happens under the
/// runtime lock.
#[derive(Debug, Default)]
pub struct ArchiveCandidate {
    best: Option<(PrefillKvIdentity, usize)>,
}

impl ArchiveCandidate {
    /// Offer a freshly recorded candidate. Keeps the longest one that is
    /// strictly shorter than `full_len`; if every candidate is full-length
    /// (a short prompt with a single ladder entry), keeps that instead so
    /// small prompts still archive something.
    pub fn offer(&mut self, identity: &PrefillKvIdentity, token_count: usize, full_len: usize) {
        let partial = token_count < full_len;
        let better = match &self.best {
            None => true,
            Some((_, best_tokens)) => {
                let best_partial = *best_tokens < full_len;
                match (partial, best_partial) {
                    // Prefer any partial candidate over a full-length one.
                    (true, false) => true,
                    // Among partials, prefer the longest.
                    (true, true) => token_count > *best_tokens,
                    // Never displace a partial with a full-length candidate.
                    (false, true) => false,
                    (false, false) => token_count > *best_tokens,
                }
            }
        };
        if better {
            self.best = Some((identity.clone(), token_count));
        }
    }

    /// Take the selected candidate, if any.
    pub fn take(&mut self) -> Option<PrefillKvIdentity> {
        self.best.take().map(|(identity, _)| identity)
    }
}

#[cfg(test)]
mod archive_candidate_tests {
    use super::*;

    fn identity(tokens: u64) -> PrefillKvIdentity {
        PrefillKvIdentity {
            page_id: format!("page-{tokens}"),
            identity: crate::kv_proto::PageIdentity {
                token_count: tokens,
                ..Default::default()
            },
        }
    }

    /// The whole point: for an agent prompt the archived page must be the
    /// shared bulk, not the tiny floor candidate and not the unique tail.
    #[test]
    fn picks_the_largest_prefix_that_excludes_the_request_tail() {
        let full = 2129;
        let mut pick = ArchiveCandidate::default();
        // Ladder arrives longest-first, as the recorders emit it.
        for tokens in [2129usize, 2048, 1920, 1024, 512, 256] {
            pick.offer(&identity(tokens as u64), tokens, full);
        }
        let chosen = pick.take().expect("a candidate must be chosen");
        assert_eq!(
            chosen.identity.token_count, 2048,
            "must archive the shared bulk, not the tail (2129) or the floor (256)"
        );
    }

    /// A short prompt whose only candidate is its full length should still
    /// archive; otherwise small prompts silently never persist.
    #[test]
    fn falls_back_to_the_full_length_candidate_when_that_is_all_there_is() {
        let mut pick = ArchiveCandidate::default();
        pick.offer(&identity(512), 512, 512);
        assert_eq!(pick.take().expect("candidate").identity.token_count, 512);
    }

    #[test]
    fn offering_nothing_selects_nothing() {
        assert!(ArchiveCandidate::default().take().is_none());
    }
}
