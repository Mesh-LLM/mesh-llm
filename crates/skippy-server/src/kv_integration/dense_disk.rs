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
            && !self
                .dense_archive_unsupported
                .load(std::sync::atomic::Ordering::Relaxed)
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
    ) -> Result<DenseArchiveOutcome> {
        if !self.dense_archive_enabled() {
            return Ok(DenseArchiveOutcome::Skipped(DenseArchiveSkip::TierDisabled));
        }
        let token_count = identity.identity.token_count;
        if token_count < MIN_ARCHIVE_TOKENS.max(self.candidate_policy.min_tokens) {
            return Ok(DenseArchiveOutcome::Skipped(DenseArchiveSkip::TooShort));
        }
        {
            let cache = self
                .exact_states
                .lock()
                .expect("exact state cache lock poisoned");
            if cache.disk_contains(&identity.page_id) {
                return Ok(DenseArchiveOutcome::Skipped(
                    DenseArchiveSkip::AlreadyArchived,
                ));
            }
        }

        // Export the exact token range this page id was computed over, so the
        // archived bytes and the identity always agree.
        let export_timer = std::time::Instant::now();
        let page = match runtime.export_kv_page(session_id, 0, token_count) {
            Ok(page) => page,
            // Distinguished from a skip: the runtime declining to export is a
            // failure of the archive, not a policy decision, and collapsing
            // the two is exactly why a tier that never stores anything can
            // look healthy.
            Err(error) => {
                let reason = error.to_string();
                // An unsupported memory layout is a permanent property of this
                // stage, not a transient failure of this request. Latch it off
                // so we stop paying for an export attempt on every prefill,
                // and report it as a skip: nothing is broken, this model's
                // attention state simply cannot be described by one page.
                if is_unsupported_memory_layout(&reason) {
                    self.dense_archive_unsupported
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    return Ok(DenseArchiveOutcome::Skipped(
                        DenseArchiveSkip::UnsupportedMemoryLayout,
                    ));
                }
                // Carry the native reason. "The export failed" without a cause
                // is only marginally better than the silent bool this replaced:
                // it says something is wrong and nothing about what.
                return Ok(DenseArchiveOutcome::Failed(DenseArchiveFailure::Export {
                    reason,
                }));
            }
        };
        let export_ms = export_timer.elapsed().as_secs_f64() * 1000.0;
        let payload_bytes = page.payload.len() as u64;
        let write_timer = std::time::Instant::now();
        let stored = {
            let mut cache = self
                .exact_states
                .lock()
                .expect("exact state cache lock poisoned");
            cache.store_on_disk(
                &identity.page_id,
                token_count,
                ExactStatePayloadKind::ResidentKvArchive,
                &[&page.payload],
                ExactStateExtra {
                    kv_desc: Some(page.desc),
                },
            )
        };
        let write_ms = write_timer.elapsed().as_secs_f64() * 1000.0;
        if !stored {
            return Ok(DenseArchiveOutcome::Failed(DenseArchiveFailure::Write));
        }
        Ok(DenseArchiveOutcome::Archived {
            payload_bytes,
            export_ms,
            write_ms,
        })
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
            let verify_timer = std::time::Instant::now();
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

            let verify_ms = verify_timer.elapsed().as_secs_f64() * 1000.0;
            let payload_bytes = kv.as_ref().len() as u64;

            // Import borrows the mapped bytes directly; no copy is made.
            let import_timer = std::time::Instant::now();
            let import_failed = runtime
                .import_kv_page(session_id, &desc, kv.as_ref())
                .is_err();
            let import_ms = import_timer.elapsed().as_secs_f64() * 1000.0;
            if import_failed {
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
                verify_ms,
                import_ms,
                payload_bytes,
            }));
        }
        Ok(None)
    }
}

/// What happened to an attempted dense archive.
///
/// Previously this path returned `Result<bool>` and every caller matched
/// `Ok(true)`, so "policy declined to archive" and "the archive failed"
/// collapsed into the same silent `false`. Those need different responses --
/// one is normal, the other means the disk tier is not doing its job -- so
/// they are separate variants and both are reported.
#[derive(Debug, Clone, PartialEq)]
pub enum DenseArchiveOutcome {
    Archived {
        payload_bytes: u64,
        /// Exporting the KV page out of the live session.
        export_ms: f64,
        /// Serializing and writing the entry to the tier.
        write_ms: f64,
    },
    Skipped(DenseArchiveSkip),
    Failed(DenseArchiveFailure),
}

/// A deliberate decision not to archive. Normal, and not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseArchiveSkip {
    /// No disk tier configured, or this payload family does not archive.
    TierDisabled,
    /// Below the size floor where a restore beats recomputing.
    TooShort,
    /// The tier already holds this page.
    AlreadyArchived,
    /// The runtime's attention memory cannot be exported as one native KV
    /// page. Sliding-window models split attention across a full-context base
    /// cache and a window-bounded SWA cache; a single page with a single token
    /// range cannot represent both, so the native side declines. Resident
    /// reuse is unaffected.
    UnsupportedMemoryLayout,
}

/// An archive that was wanted but did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseArchiveFailure {
    /// The runtime declined to export the KV page.
    Export { reason: String },
    /// The tier refused or failed the write (budget, oversize, or IO).
    Write,
}

impl DenseArchiveOutcome {
    pub fn archived(&self) -> bool {
        matches!(self, Self::Archived { .. })
    }

    /// Write this outcome into a telemetry attribute map.
    ///
    /// Centralised because there are four archive call sites and they
    /// previously reported the same thing four slightly different ways -- or,
    /// on two of them, not at all.
    pub fn insert_attrs(
        &self,
        identity: &PrefillKvIdentity,
        attrs: &mut std::collections::BTreeMap<String, serde_json::Value>,
    ) {
        attrs.insert(
            "skippy.kv.archive_status".to_string(),
            serde_json::json!(self.status()),
        );
        attrs.insert(
            "skippy.kv.archive_candidate_tokens".to_string(),
            serde_json::json!(identity.identity.token_count),
        );
        if let Self::Archived {
            payload_bytes,
            export_ms,
            write_ms,
        } = self
        {
            attrs.insert(
                "skippy.kv.archived_tokens".to_string(),
                serde_json::json!(identity.identity.token_count),
            );
            attrs.insert(
                "skippy.kv.archive_bytes".to_string(),
                serde_json::json!(payload_bytes),
            );
            attrs.insert(
                "skippy.kv.archive_export_ms".to_string(),
                serde_json::json!(export_ms),
            );
            attrs.insert(
                "skippy.kv.archive_write_ms".to_string(),
                serde_json::json!(write_ms),
            );
        }
        if let Self::Failed(DenseArchiveFailure::Export { reason }) = self {
            attrs.insert(
                "skippy.kv.archive_error_class".to_string(),
                serde_json::json!(super::telemetry_error_class_from_message(reason)),
            );
        }
    }

    /// Stable label for telemetry.
    pub fn status(&self) -> &'static str {
        match self {
            Self::Archived { .. } => "archived",
            Self::Skipped(DenseArchiveSkip::TierDisabled) => "skipped_tier_disabled",
            Self::Skipped(DenseArchiveSkip::TooShort) => "skipped_too_short",
            Self::Skipped(DenseArchiveSkip::AlreadyArchived) => "skipped_already_archived",
            Self::Skipped(DenseArchiveSkip::UnsupportedMemoryLayout) => {
                "skipped_unsupported_memory"
            }
            Self::Failed(DenseArchiveFailure::Export { .. }) => "failed_export",
            Self::Failed(DenseArchiveFailure::Write) => "failed_write",
        }
    }
}

/// A dense prefix served back from the disk tier.
///
/// The two phase timings are carried out to telemetry because the first
/// measurement of a 25k-token restore spent ~18s inside this function, and
/// wall clock alone could not say whether that was checksum verification
/// (which faults every page in and hashes it) or the native import. Those
/// have completely different fixes, so the split is reported rather than
/// inferred.
#[derive(Debug, Clone)]
pub struct DenseDiskRestore {
    pub page_id: String,
    pub token_count: u64,
    /// Lookup, checksum verification, and page faulting.
    pub verify_ms: f64,
    /// `skippy_import_kv_page` alone.
    pub import_ms: f64,
    pub payload_bytes: u64,
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

/// Offer a record candidate to the archive selector.
///
/// Deliberately independent of the resident cache's admission decision: the
/// disk tier's cost model is bytes-and-a-write, the resident cache's is KV
/// cells, and a candidate rejected by one is routinely worth accepting in the
/// other. The only requirement here is that the runtime actually holds the
/// tokens the archive would export, which for a `token_start == 0` prefill
/// means the candidate cannot claim more tokens than this request carried.
///
/// `ArchiveCandidate` still applies the selection policy (prefer the longest
/// prefix strictly shorter than the full prompt), and `archive_dense_prefix`
/// still applies the size floor and dedupe check, so a wider offer does not
/// mean more writes -- it is still at most one archive per request.
pub fn offer_archive_candidate(
    archive_candidate: &mut ArchiveCandidate,
    identity: &PrefillKvIdentity,
    full_len: usize,
) {
    let Ok(token_count) = usize::try_from(identity.identity.token_count) else {
        return;
    };
    if token_count == 0 || token_count > full_len {
        return;
    }
    archive_candidate.offer(identity, token_count, full_len);
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

/// Whether a native export error means "this stage can never export a page",
/// as opposed to a failure specific to one request.
fn is_unsupported_memory_layout(reason: &str) -> bool {
    reason.contains("runtime memory type is not supported for native KV pages")
        || reason.contains("runtime has no attention KV cache")
        || reason.contains("runtime has no GLM-DSA MLA KV cache")
}

#[cfg(test)]
mod unsupported_memory_tests {
    use super::*;

    /// The exact strings the patched runtime emits when attention state cannot
    /// be expressed as one page. Asserted literally: these are a cross-language
    /// contract with `skippy_get_kv_cache`, and a silent drift here turns a
    /// permanent, expected limitation back into a per-prefill failure.
    #[test]
    fn native_unsupported_memory_errors_are_recognised() {
        for reason in [
            "Unsupported: runtime memory type is not supported for native KV pages",
            "Unsupported: runtime has no attention KV cache",
            "Unsupported: runtime has no GLM-DSA MLA KV cache",
        ] {
            assert!(
                is_unsupported_memory_layout(reason),
                "should latch archiving off for: {reason}"
            );
        }
    }

    /// A real, transient failure must stay a failure. Latching on one of these
    /// would silently disable the disk tier for the life of the process.
    #[test]
    fn ordinary_export_errors_do_not_latch() {
        for reason in [
            "RuntimeError: output buffer too small",
            "InvalidArgument: session is required",
            "runtime memory is unavailable",
        ] {
            assert!(
                !is_unsupported_memory_layout(reason),
                "should stay a retryable failure: {reason}"
            );
        }
    }

    #[test]
    fn unsupported_memory_reports_a_distinct_status() {
        let outcome = DenseArchiveOutcome::Skipped(DenseArchiveSkip::UnsupportedMemoryLayout);
        assert_eq!(outcome.status(), "skipped_unsupported_memory");
    }
}
