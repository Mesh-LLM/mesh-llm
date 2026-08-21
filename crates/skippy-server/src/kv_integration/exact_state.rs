use std::time::Instant;

use anyhow::{Context, Result};
use skippy_cache::ExactStatePayload;

use crate::runtime_state::RuntimeState;

use super::{
    ExactStateExtra, ExactStateRecord, ExactStateRecordAdmission, ExactStateRestore,
    KvStageIntegration, PendingExactStateRecord, PrefillKvIdentity, StagePrefixCachePayload,
    records::add_reconstruct_stats,
};

impl KvStageIntegration {
    pub fn restore_exact_state(
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
            let lookup = {
                let mut cache = self
                    .exact_states
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                cache.lookup(&identity.page_id)
            };
            let Some(lookup) = lookup else {
                continue;
            };
            let lookup_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
            let mut reconstruct_ms = 0.0;
            let mut reconstruct_bytes = 0u64;
            let mut reconstruct_blocks = 0usize;
            let mut kv_import_ms = 0.0;
            let mut recurrent_import_ms = 0.0;
            match lookup.payload.kind().into() {
                StagePrefixCachePayload::FullState => {
                    let (full_state, stats) = lookup
                        .payload
                        .full_state_bytes_timed()
                        .context("reconstruct cached full-state payload")?;
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
                        lookup.token_count,
                    )?;
                    kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                }
                StagePrefixCachePayload::KvRecurrent => {
                    if let Some((kv, stats)) = lookup
                        .payload
                        .kv_bytes_timed()
                        .context("reconstruct cached KV payload")?
                    {
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        if let Some(desc) = lookup.extra.kv_desc {
                            let import_started = Instant::now();
                            runtime.import_kv_page(session_id, &desc, kv.as_ref())?;
                            kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                        } else if !kv.is_empty() {
                            continue;
                        }
                    }
                    let (recurrent, stats) = lookup
                        .payload
                        .recurrent_state_bytes_timed()
                        .context("reconstruct cached recurrent payload")?;
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
                        lookup.token_count,
                    )?;
                    recurrent_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                }
                _ => continue,
            }
            return Ok(Some(ExactStateRestore {
                page_id: lookup.page_id,
                token_count: lookup.token_count as usize,
                payload_kind: lookup.payload.kind(),
                logical_bytes: lookup.logical_bytes,
                entries: lookup.entries,
                reconstruct_ms,
                reconstruct_bytes,
                reconstruct_blocks,
                lookup_ms,
                kv_import_ms,
                recurrent_import_ms,
            }));
        }
        Ok(None)
    }

    pub fn record_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
    ) -> Result<Option<ExactStateRecord>> {
        if !self.should_record() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        let token_count = identity.identity.token_count;
        if token_count < self.candidate_policy.min_tokens {
            return Ok(None);
        }
        if !self.try_begin_record(&identity.page_id) {
            return Ok(None);
        }
        let already_recorded = match try_touch_exact_state(&self.exact_states, &identity.page_id) {
            Ok(Some(already_recorded)) => already_recorded,
            Ok(None) => {
                // Recording is optional. A background worker may hold this lock
                // while hashing hundreds of MiB; never make inference wait for it.
                self.finish_record(&identity.page_id);
                return Ok(None);
            }
            Err(error) => {
                self.finish_record(&identity.page_id);
                return Err(error);
            }
        };
        if already_recorded {
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
        // Avoid paying a potentially multi-hundred-MiB runtime export when the
        // bounded worker queue is already occupied. Admission remains best-effort:
        // another producer may win the race before `try_send` below.
        if !self.has_exact_state_record_capacity() {
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
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
            token_count,
            payload,
            extra,
        }) {
            ExactStateRecordAdmission::Queued => {
                // Recording owns the exact-state mutex while it hashes a potentially
                // multi-hundred-MiB payload. Telemetry must not turn that background
                // work back into request latency by waiting for cache stats here.
                let stats = self
                    .exact_states
                    .try_lock()
                    .ok()
                    .map(|states| states.stats())
                    .unwrap_or_default();
                Ok(Some(ExactStateRecord {
                    page_id: identity.page_id.clone(),
                    token_count: token_count as usize,
                    payload_kind,
                    stored: false,
                    logical_bytes,
                    physical_bytes: stats.physical_bytes,
                    entries: stats.entries,
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

fn try_touch_exact_state(
    exact_states: &std::sync::Mutex<skippy_cache::ExactStateCache<ExactStateExtra>>,
    page_id: &str,
) -> Result<Option<bool>> {
    match exact_states.try_lock() {
        Ok(mut exact_states) => Ok(Some(exact_states.touch(page_id))),
        Err(std::sync::TryLockError::WouldBlock) => Ok(None),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => {
            Ok(Some(poisoned.into_inner().touch(page_id)))
        }
    }
}

fn is_native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
    })
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

    use skippy_cache::ExactStateCache;

    use super::{ExactStateExtra, try_touch_exact_state};

    #[test]
    fn busy_exact_state_lock_skips_touch_without_waiting() {
        let cache = Arc::new(Mutex::new(ExactStateCache::<ExactStateExtra>::new(1, 1024)));
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
        assert_eq!(try_touch_exact_state(&cache, "busy").unwrap(), None);
        assert!(started.elapsed() < Duration::from_millis(100));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
    }

    #[test]
    fn poisoned_exact_state_lock_recovers_without_panicking() {
        let cache = Arc::new(Mutex::new(ExactStateCache::<ExactStateExtra>::new(1, 1024)));
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
            try_touch_exact_state(&cache, "poisoned").unwrap(),
            Some(false)
        );
    }
}
