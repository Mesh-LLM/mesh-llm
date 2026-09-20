use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use skippy_cache::ExactStatePayload;

use crate::runtime_state::RuntimeState;

use super::{
    ExactStateExtra, ExactStateRecord, ExactStateRecordAdmission, ExactStateRestore,
    KvStageIntegration, PendingExactStateRecord, PrefillKvIdentity, StagePrefixCachePayload,
    records::add_reconstruct_stats,
};

/// Admission class for an exact-state capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureAdmission {
    /// Same-session continuation boundary (post-decode). Re-derivable only by
    /// replaying generation; competes for the reserved recorder slot.
    Continuation,
    /// Prefill-ladder boundary (shared checkpoint / final prefill state).
    /// Re-derivable by re-prefill; best-effort admission.
    BestEffort,
}

/// Total in-flight exact-state payloads per stage — exporting, queued in the
/// recorder channel, or held by the recorder worker. Sized as channel
/// capacity (1) plus one worker-held record.
pub const EXACT_STATE_ADMISSION_TOTAL: usize = 2;
/// Slots reserved for [`CaptureAdmission::Continuation`]: BestEffort captures
/// can never occupy more than `TOTAL - RESERVED` slots, so a continuation
/// boundary always finds a slot unless the whole budget is exhausted.
pub const EXACT_STATE_ADMISSION_RESERVED: usize = 1;

/// Move-only credit for one in-flight exact-state payload. Construction
/// reserves a slot atomically (CAS, so competing producers cannot
/// oversubscribe the budget); Drop releases it exactly once on every terminal
/// path — payload stored, skipped, errored, command dropped unexecuted, or
/// channel receiver dropped. The credit moves into
/// [`PendingExactStateRecord`] so the worker's completion releases it.
#[derive(Debug)]
pub struct ExactStateAdmissionCredit {
    class: CaptureAdmission,
    total: Arc<std::sync::atomic::AtomicUsize>,
    best_effort: Arc<std::sync::atomic::AtomicUsize>,
}

impl ExactStateAdmissionCredit {
    pub(crate) fn acquire(
        total: &Arc<std::sync::atomic::AtomicUsize>,
        best_effort: &Arc<std::sync::atomic::AtomicUsize>,
        class: CaptureAdmission,
    ) -> Option<Self> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        match class {
            CaptureAdmission::Continuation => {
                let mut observed = total.load(Acquire);
                loop {
                    if observed >= EXACT_STATE_ADMISSION_TOTAL {
                        return None;
                    }
                    match total.compare_exchange_weak(observed, observed + 1, AcqRel, Acquire) {
                        Ok(_) => {
                            return Some(Self {
                                class,
                                total: Arc::clone(total),
                                best_effort: Arc::clone(best_effort),
                            });
                        }
                        Err(actual) => observed = actual,
                    }
                }
            }
            CaptureAdmission::BestEffort => {
                let reserved_cap = EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED;
                let mut observed_best_effort = best_effort.load(Acquire);
                loop {
                    if observed_best_effort >= reserved_cap {
                        return None;
                    }
                    match best_effort.compare_exchange_weak(
                        observed_best_effort,
                        observed_best_effort + 1,
                        AcqRel,
                        Acquire,
                    ) {
                        Ok(_) => break,
                        Err(actual) => observed_best_effort = actual,
                    }
                }
                // Reserve the total slot; roll the BestEffort reservation back
                // if the total budget was exhausted concurrently.
                let mut observed_total = total.load(Acquire);
                loop {
                    if observed_total >= EXACT_STATE_ADMISSION_TOTAL {
                        best_effort.fetch_sub(1, AcqRel);
                        return None;
                    }
                    match total.compare_exchange_weak(
                        observed_total,
                        observed_total + 1,
                        AcqRel,
                        Acquire,
                    ) {
                        Ok(_) => {
                            return Some(Self {
                                class,
                                total: Arc::clone(total),
                                best_effort: Arc::clone(best_effort),
                            });
                        }
                        Err(actual) => observed_total = actual,
                    }
                }
            }
        }
    }
}

impl Drop for ExactStateAdmissionCredit {
    fn drop(&mut self) {
        let _ = self.total.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        if self.class == CaptureAdmission::BestEffort {
            let _ = self
                .best_effort
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

impl KvStageIntegration {
    pub fn restore_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        runtime.restore_transaction(session_id, |runtime| {
            self.restore_exact_state_inner(runtime, session_id, identities)
        })
    }

    fn restore_exact_state_inner(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        if !self.should_lookup() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        for identity in identities {
            let lookup_started = Instant::now();
            let (lookup, entries) = {
                let mut radix = self
                    .radix
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let lookup = radix.acquire_recurrent(&identity.namespace, &identity.token_ids);
                let entries = radix.stats().recurrent_entries;
                (lookup, entries)
            };
            #[cfg(test)]
            crate::frontend::capture_trace::log_lookup_candidate(
                &identity.namespace,
                &identity.token_ids,
                lookup.is_some(),
                lookup.as_ref().map(|matched| matched.stored_tokens.len()),
            );
            let Some(lookup) = lookup else {
                continue;
            };
            let lease = ExactStateLease {
                radix: std::sync::Arc::clone(&self.radix),
                namespace: identity.namespace.clone(),
                stored_tokens: lookup.stored_tokens.clone(),
            };
            let token_count = lookup.stored_tokens.len() as u64;
            let lookup_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
            let mut reconstruct_ms = 0.0;
            let mut reconstruct_bytes = 0u64;
            let mut reconstruct_blocks = 0usize;
            let mut kv_import_ms = 0.0;
            let mut recurrent_import_ms = 0.0;
            let mut deterministic_failure = false;
            let restore_result = (|| -> Result<bool> {
                match lookup.value.payload.kind().into() {
                    StagePrefixCachePayload::FullState => {
                        let (full_state, stats) = lookup
                            .value
                            .payload
                            .full_state_bytes_timed()
                            .context("reconstruct cached full-state payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?;
                        if full_state.is_empty() {
                            deterministic_failure = true;
                            return Err(anyhow::anyhow!("cached full-state payload is empty"));
                        }
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        let import_started = Instant::now();
                        runtime.import_full_state_for_token_count(
                            session_id,
                            full_state.as_ref(),
                            token_count,
                        )?;
                        kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                    }
                    StagePrefixCachePayload::KvRecurrent => {
                        if let Some((kv, stats)) = lookup
                            .value
                            .payload
                            .kv_bytes_timed()
                            .context("reconstruct cached KV payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?
                        {
                            add_reconstruct_stats(
                                &mut reconstruct_ms,
                                &mut reconstruct_bytes,
                                &mut reconstruct_blocks,
                                stats,
                            );
                            if let Some(desc) = lookup.value.extra.kv_desc.as_ref() {
                                desc.validate_payload(kv.len()).map_err(|error| {
                                    mark_deterministic_failure(&mut deterministic_failure, error)
                                })?;
                                if desc.token_start != 0 || desc.token_count != token_count {
                                    deterministic_failure = true;
                                    return Err(anyhow::anyhow!(
                                        "cached KV page token range mismatch for exact-state checkpoint"
                                    ));
                                }
                                let import_started = Instant::now();
                                runtime.import_kv_page(session_id, desc, kv.as_ref())?;
                                kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                            } else if !kv.is_empty() {
                                deterministic_failure = true;
                                return Err(anyhow::anyhow!(
                                    "cached KV payload is missing its descriptor"
                                ));
                            }
                        }
                        let (recurrent, stats) = lookup
                            .value
                            .payload
                            .recurrent_state_bytes_timed()
                            .context("reconstruct cached recurrent payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?;
                        if recurrent.is_empty() {
                            deterministic_failure = true;
                            return Err(anyhow::anyhow!("cached recurrent-state payload is empty"));
                        }
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        let import_started = Instant::now();
                        runtime.import_recurrent_state_for_token_count(
                            session_id,
                            recurrent.as_ref(),
                            token_count,
                        )?;
                        recurrent_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                    }
                    _ => return Ok(false),
                }
                Ok(true)
            })();
            let restored_payload = match restore_result {
                Ok(restored_payload) => restored_payload,
                Err(error) => {
                    drop(lease);
                    if deterministic_failure
                        && let Err(quarantine_error) = self.quarantine_exact_state_entry(
                            &identity.namespace,
                            &lookup.stored_tokens,
                            &lookup.value.page_id,
                        )
                    {
                        let _ =
                            mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
                                message: "Skippy exact-state quarantine failed".to_string(),
                                context: Some(format!(
                                    "page_id={} error={quarantine_error:#}",
                                    lookup.value.page_id
                                )),
                            });
                        return Err(error.context(format!(
                            "failed to fully quarantine corrupt exact-state entry: {quarantine_error:#}"
                        )));
                    }
                    return Err(error);
                }
            };
            if !restored_payload {
                drop(lease);
                continue;
            }
            let restored = ExactStateRestore {
                page_id: lookup.value.page_id,
                token_count: token_count as usize,
                payload_kind: lookup.value.payload.kind(),
                logical_bytes: lookup.logical_bytes,
                entries,
                reconstruct_ms,
                reconstruct_bytes,
                reconstruct_blocks,
                lookup_ms,
                kv_import_ms,
                recurrent_import_ms,
            };
            drop(lease);
            return Ok(Some(restored));
        }
        Ok(None)
    }

    fn quarantine_exact_state_entry(
        &self,
        namespace: &str,
        tokens: &[i32],
        page_id: &str,
    ) -> Result<bool> {
        let removed = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove_recurrent_if(namespace, tokens, |entry| entry.page_id == page_id);
        let Some(entry) = removed else {
            return Ok(false);
        };
        entry.payload.release_from(
            &mut self
                .exact_blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
        Ok(true)
    }

    pub fn record_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        admission: CaptureAdmission,
    ) -> Result<Option<ExactStateRecord>> {
        if !self.should_record() || !self.payload.is_exact_state() {
            #[cfg(test)]
            crate::frontend::capture_trace::POST_DECODE_NONE_SHOULD_RECORD
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Ok(None);
        }
        let token_count = identity.identity.token_count;
        if token_count < self.checkpoint_policy.min_tokens {
            #[cfg(test)]
            crate::frontend::capture_trace::POST_DECODE_NONE_MIN_TOKENS
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Ok(None);
        }
        // Admission: one credit per payload across its whole lifetime —
        // exporting, queued in the recorder channel, or held by the recorder
        // worker. Acquisition happens HERE, inside record_exact_state, on the
        // thread that already holds the runtime lock, immediately before the
        // export: a declined capture pays no lock-held export, and the credit
        // therefore bounds in-flight EXPORTED payloads — not scheduled capture
        // jobs or runtime-lock acquisitions. The credit moves into the
        // enqueued record and the worker releases it on every completion
        // path. Acquisition is a CAS reservation, so competing producers
        // cannot oversubscribe the budget.
        let Some(admission_credit) = ExactStateAdmissionCredit::acquire(
            &self.admission_outstanding,
            &self.admission_best_effort_outstanding,
            admission,
        ) else {
            #[cfg(test)]
            {
                crate::frontend::capture_trace::POST_DECODE_ADMISSION_DECLINED
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                crate::frontend::capture_trace::ANY_ADMISSION_DECLINED
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            return Ok(None);
        };
        if !self.try_begin_record(&identity.page_id) {
            #[cfg(test)]
            crate::frontend::capture_trace::POST_DECODE_NONE_BEGIN_RECORD
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Ok(None);
        }
        let already_recorded =
            match try_touch_exact_state(&self.radix, &identity.namespace, &identity.token_ids) {
                Ok(Some(already_recorded)) => already_recorded,
                Ok(None) => {
                    // Recording is optional. A background worker may hold this lock
                    // while hashing hundreds of MiB; never make inference wait for it.
                    #[cfg(test)]
                    crate::frontend::capture_trace::POST_DECODE_NONE_RADIX_BUSY
                        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    self.finish_record(&identity.page_id);
                    return Ok(None);
                }
                Err(error) => {
                    self.finish_record(&identity.page_id);
                    return Err(error);
                }
            };
        if already_recorded {
            #[cfg(test)]
            crate::frontend::capture_trace::POST_DECODE_NONE_ALREADY_RECORDED
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
        // Admission is handled by the acquired credit: BestEffort captures are
        // capped below the total budget, and Continuation (post-decode) has a
        // reserved slot, so the pre-export capacity decline is replaced by the
        // credit acquisition above. The channel try_send remains the final
        // bounded check: a full channel still declines loudly (DroppedFull).
        let exported = match self.payload {
            StagePrefixCachePayload::FullState => {
                runtime.export_full_state(session_id).map(|state| {
                    (
                        ExactStatePayload::full_state(state),
                        ExactStateExtra::default(),
                    )
                })
            }
            StagePrefixCachePayload::KvRecurrent => (|| {
                let kv = match runtime.export_kv_page(session_id, 0, token_count) {
                    Ok(kv) => Some(kv),
                    Err(error) if is_native_kv_unavailable(&error) => None,
                    Err(error) => return Err(error),
                };
                let recurrent = runtime.export_recurrent_state(session_id)?;
                Ok((
                    ExactStatePayload::kv_recurrent(
                        kv.as_ref().map(|kv| kv.payload.clone()).unwrap_or_default(),
                        recurrent,
                    ),
                    ExactStateExtra {
                        kv_desc: kv.as_ref().map(|kv| kv.desc.clone()),
                    },
                ))
            })(),
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                #[cfg(test)]
                crate::frontend::capture_trace::POST_DECODE_NONE_PAYLOAD_DISABLED
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                self.finish_record(&identity.page_id);
                return Ok(None);
            }
        };
        let (payload, extra) = match exported {
            Ok(exported) => exported,
            Err(error) => {
                self.finish_record(&identity.page_id);
                return Err(error);
            }
        };
        let payload_kind = payload.kind();
        let logical_bytes = payload.byte_len();
        match self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload,
            extra,
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids.clone(),
            admission_credit,
        }) {
            ExactStateRecordAdmission::Queued => {
                // Recording owns the radix/blob locks while it hashes a potentially
                // multi-hundred-MiB payload. Telemetry must not turn that background
                // work back into request latency by waiting for cache stats here.
                let entries = self
                    .radix
                    .try_lock()
                    .ok()
                    .map(|radix| radix.stats().recurrent_entries)
                    .unwrap_or_default();
                let physical_bytes = self
                    .exact_blobs
                    .try_lock()
                    .ok()
                    .map(|blobs| blobs.physical_bytes())
                    .unwrap_or_default();
                Ok(Some(ExactStateRecord {
                    page_id: identity.page_id.clone(),
                    token_count: token_count as usize,
                    payload_kind,
                    stored: false,
                    logical_bytes,
                    physical_bytes,
                    entries,
                    evicted_entries: 0,
                    evicted_logical_bytes: 0,
                    dedupe: Default::default(),
                }))
            }
            ExactStateRecordAdmission::DroppedFull | ExactStateRecordAdmission::WorkerStopped => {
                Ok(None)
            }
        }
    }
}

struct ExactStateLease {
    radix: std::sync::Arc<
        std::sync::Mutex<
            skippy_cache::UnifiedRadixCache<super::RadixResidentEntry, super::RadixExactEntry>,
        >,
    >,
    namespace: String,
    stored_tokens: Vec<i32>,
}

impl Drop for ExactStateLease {
    fn drop(&mut self) {
        let released = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .release_recurrent(&self.namespace, &self.stored_tokens);
        debug_assert!(released, "recurrent radix acquire/release must balance");
    }
}

fn try_touch_exact_state(
    radix: &std::sync::Mutex<
        skippy_cache::UnifiedRadixCache<super::RadixResidentEntry, super::RadixExactEntry>,
    >,
    namespace: &str,
    token_ids: &[i32],
) -> Result<Option<bool>> {
    match radix.try_lock() {
        Ok(mut radix) => Ok(Some(radix.recurrent_exact(namespace, token_ids).is_some())),
        Err(std::sync::TryLockError::WouldBlock) => Ok(None),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Ok(Some(
            poisoned
                .into_inner()
                .recurrent_exact(namespace, token_ids)
                .is_some(),
        )),
    }
}

fn is_native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
            || message.contains("no KV cache layers selected by layer range")
    })
}

fn mark_deterministic_failure(
    deterministic_failure: &mut bool,
    error: anyhow::Error,
) -> anyhow::Error {
    *deterministic_failure = true;
    error
}

impl StagePrefixCachePayload {
    pub(crate) fn is_exact_state(self) -> bool {
        matches!(self, Self::KvRecurrent | Self::FullState)
    }
}

impl From<skippy_cache::ExactStatePayloadKind> for StagePrefixCachePayload {
    fn from(kind: skippy_cache::ExactStatePayloadKind) -> Self {
        match kind {
            skippy_cache::ExactStatePayloadKind::FullState => Self::FullState,
            skippy_cache::ExactStatePayloadKind::KvRecurrent => Self::KvRecurrent,
            skippy_cache::ExactStatePayloadKind::RecurrentOnly => Self::Disabled,
        }
    }
}

impl From<StagePrefixCachePayload> for skippy_cache::ExactStatePayloadKind {
    fn from(payload: StagePrefixCachePayload) -> Self {
        match payload {
            StagePrefixCachePayload::FullState => Self::FullState,
            StagePrefixCachePayload::KvRecurrent => Self::KvRecurrent,
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                Self::FullState
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use skippy_cache::UnifiedRadixCache;

    use super::{is_native_kv_unavailable, try_touch_exact_state};

    type TestRadix = UnifiedRadixCache<
        crate::kv_integration::RadixResidentEntry,
        crate::kv_integration::RadixExactEntry,
    >;

    #[test]
    fn empty_kv_layer_range_is_an_unavailable_optional_kv_component() {
        let error = anyhow::anyhow!("RuntimeError: no KV cache layers selected by layer range");
        assert!(is_native_kv_unavailable(&error));
    }

    #[test]
    fn busy_exact_state_lock_skips_touch_without_waiting() {
        let cache = Arc::new(Mutex::new(TestRadix::new()));
        let locked = cache.clone();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = locked.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();

        let started = Instant::now();
        assert_eq!(
            try_touch_exact_state(&cache, "namespace", &[1]).unwrap(),
            None
        );
        assert!(started.elapsed() < Duration::from_millis(100));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
    }

    #[test]
    fn poisoned_exact_state_lock_recovers_without_panicking() {
        let cache = Arc::new(Mutex::new(TestRadix::new()));
        let poisoned = cache.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = poisoned.lock().unwrap();
                panic!("poison exact-state cache for test");
            })
            .join()
            .is_err()
        );

        assert_eq!(
            try_touch_exact_state(&cache, "namespace", &[1]).unwrap(),
            Some(false)
        );
    }
}

#[cfg(test)]
mod admission_credit_tests {
    use super::{
        CaptureAdmission, EXACT_STATE_ADMISSION_RESERVED, EXACT_STATE_ADMISSION_TOTAL,
        ExactStateAdmissionCredit,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
    }

    #[test]
    fn construction_and_drop_account_exactly_once() {
        let (total, best_effort) = counters();
        let guard = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(guard.is_some(), "fresh budget admits Continuation");
        assert_eq!(total.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(total.load(Ordering::Acquire), 0);
    }

    #[test]
    fn continuation_declines_when_total_budget_exhausted() {
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(first.is_some() && second.is_some(), "two slots for TOTAL=2");
        let third = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            third.is_none(),
            "third Continuation must decline at TOTAL=2"
        );
        assert_eq!(total.load(Ordering::Acquire), 2);
    }

    #[test]
    fn best_effort_declines_at_reserved_cap_even_with_budget_headroom() {
        let (total, best_effort) = counters();
        let first =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(first.is_some());
        let second =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            second.is_none(),
            "BestEffort cap is TOTAL - RESERVED = 1; second must decline"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 1);
    }

    #[test]
    fn continuation_admits_after_best_effort_under_reserved_share() {
        let (total, best_effort) = counters();
        let best =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(best.is_some(), "BestEffort takes its single slot");
        let cont = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            cont.is_some(),
            "Continuation must admit into the reserved share"
        );
    }

    #[test]
    fn concurrent_producers_never_oversubscribe_the_budget() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 25;
        // ONE shared budget for every producer: racing acquisitions hit the
        // SAME counters, classes are mixed per attempt (both CAS paths,
        // including the BestEffort rollback, race concurrently), credits are
        // RETAINED across a barrier, and rollback/drain are asserted exactly.
        let (total, best_effort) = counters();
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    start.wait();
                    for attempt in 0..ATTEMPTS {
                        let class = if (thread + attempt) % 2 == 0 {
                            CaptureAdmission::Continuation
                        } else {
                            CaptureAdmission::BestEffort
                        };
                        if let Some(credit) =
                            ExactStateAdmissionCredit::acquire(&total, &best_effort, class)
                        {
                            held.push((class, credit));
                        }
                    }
                    held
                })
            })
            .collect();
        let mut held: Vec<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        let held_best_effort = held
            .iter()
            .filter(|(class, _)| *class == CaptureAdmission::BestEffort)
            .count();
        // Retention bounds with mixed classes: Continuation fills at most
        // TOTAL slots, BestEffort at most TOTAL - RESERVED, and together they
        // cannot exceed the total budget.
        assert!(held.len() <= EXACT_STATE_ADMISSION_TOTAL);
        assert!(
            held_best_effort <= EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED,
            "BestEffort holds must respect the reserved share"
        );
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "retained credits must account for every outstanding slot"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "retained credits must account for the BestEffort counter"
        );

        // Top the budget up to exactly TOTAL held credits so the hammer below
        // runs against a deterministically exhausted budget.
        while held.len() < EXACT_STATE_ADMISSION_TOTAL {
            let credit = ExactStateAdmissionCredit::acquire(
                &total,
                &best_effort,
                CaptureAdmission::Continuation,
            )
            .expect("topping up an under-subscribed budget must succeed");
            held.push((CaptureAdmission::Continuation, credit));
        }

        // Rollback hammer: the total budget is exhausted by the held credits,
        // so every concurrent BestEffort attempt must decline, and the
        // rollback path must not leak a BestEffort slot (a missing rollback
        // would drift the counter above the held count).
        let hammer = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let hammer = Arc::clone(&hammer);
                std::thread::spawn(move || {
                    hammer.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "an exhausted budget must decline every BestEffort attempt"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "the rollback hammer must not change the total"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "the rollback hammer must not leak a BestEffort slot"
        );

        // Drain: dropping every held credit releases each slot exactly once,
        // and the budget recovers for new admissions.
        drop(held);
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        let recovery_continuation = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let recovery_best_effort =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            recovery_continuation.is_some() && recovery_best_effort.is_some(),
            "budget must recover after the drain"
        );
    }

    #[test]
    fn rollback_is_exercised_when_the_budget_is_held_by_continuations() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 50;
        // Deterministic rollback exercise: the budget STARTS held by exactly
        // TWO Continuation credits (total=2, best_effort=0), so the first
        // BestEffort attempts must pass the class cap and fail the total CAS
        // — the rollback path. Overlapping contenders can still decline at
        // the class cap while a rollback is in flight; the exact 2/0
        // assertions afterwards are what prove no rollback leaked. The
        // mixed-contention test above can retain a BestEffort credit, in
        // which case its hammer exits at the class cap throughout.
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("first Continuation admitted");
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("second Continuation admitted");
        assert_eq!(total.load(Ordering::Acquire), 2);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);

        let barrier = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "a continuation-exhausted budget must decline every BestEffort via rollback"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            2,
            "rollback leaves the total at the held credits"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            0,
            "rollback must not leak a BestEffort slot"
        );

        // Drain and recovery.
        drop((first, second));
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        assert!(
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort)
                .is_some(),
            "budget recovers after the drain"
        );
    }
}
