use anyhow::Result;

use crate::runtime_state::RuntimeState;

use super::{
    KvStageIntegration, PrefillKvIdentity, ResidentPrefixRecord, ResidentPrefixRestore,
    StagePrefixCachePayload,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentPrefixEviction {
    pub target_tokens: u64,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
}

impl KvStageIntegration {
    pub fn probe_resident_prefix(
        &self,
        identity: &PrefillKvIdentity,
    ) -> Option<ResidentPrefixRestore> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return None;
        }
        let lookup = {
            self.resident
                .lock()
                .expect("resident prefix cache lock poisoned")
                .lookup(&identity.page_id)
        }?;
        Some(ResidentPrefixRestore {
            page_id: identity.page_id.clone(),
            token_count: identity.identity.token_count as usize,
            seq_id: lookup.seq_id,
            entries: lookup.entries,
            borrowed: false,
        })
    }

    pub fn restore_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRestore>> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        for identity in identities {
            let token_count = identity
                .identity
                .token_count
                .try_into()
                .unwrap_or(usize::MAX)
                .min(token_ids.len());
            if token_count == 0 {
                continue;
            }
            if runtime.acquire_resident_prefix_lane(
                session_id,
                &identity.page_id,
                token_count as u64,
            )? {
                let entries = self
                    .resident
                    .lock()
                    .expect("resident prefix cache lock poisoned")
                    .stats()
                    .entries;
                return Ok(Some(ResidentPrefixRestore {
                    page_id: identity.page_id.clone(),
                    token_count,
                    seq_id: -1,
                    entries,
                    borrowed: true,
                }));
            }
            let lookup = {
                self.resident
                    .lock()
                    .expect("resident prefix cache lock poisoned")
                    .lookup(&identity.page_id)
            };
            let Some(lookup) = lookup else {
                continue;
            };
            runtime.restore_resident_prefix(
                session_id,
                lookup.seq_id,
                &token_ids[..token_count],
            )?;
            return Ok(Some(ResidentPrefixRestore {
                page_id: identity.page_id.clone(),
                token_count,
                seq_id: lookup.seq_id,
                entries: lookup.entries,
                borrowed: false,
            }));
        }
        Ok(None)
    }

    pub fn borrow_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRestore>> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        for identity in identities {
            let token_count = identity
                .identity
                .token_count
                .try_into()
                .unwrap_or(usize::MAX)
                .min(token_ids.len());
            if token_count == 0 {
                continue;
            }
            let lookup = {
                self.resident
                    .lock()
                    .expect("resident prefix cache lock poisoned")
                    .acquire(&identity.page_id)
            };
            let Some(lookup) = lookup else {
                continue;
            };
            if let Err(error) = runtime.borrow_resident_prefix_session(
                session_id,
                lookup.seq_id,
                &token_ids[..token_count],
            ) {
                self.release_resident_prefix(&identity.page_id);
                return Err(error);
            }
            return Ok(Some(ResidentPrefixRestore {
                page_id: identity.page_id.clone(),
                token_count,
                seq_id: lookup.seq_id,
                entries: lookup.entries,
                borrowed: true,
            }));
        }
        Ok(None)
    }

    /// Evict enough resident-prefix entries to release `target_tokens` KV
    /// cells, or all currently releasable entries.
    pub fn evict_resident_prefix_for_tokens(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        target_tokens: u64,
    ) -> Result<ResidentPrefixEviction> {
        if self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(ResidentPrefixEviction::default());
        }
        let mut cache = self
            .resident
            .lock()
            .expect("resident prefix cache lock poisoned");
        let mut drop_fn = |seq_id: i32| runtime.drop_resident_prefix_sequence(session_id, seq_id);
        let evictions = cache.evict_lru_until_tokens(target_tokens, &mut drop_fn)?;
        let evicted_tokens = evictions.iter().map(|eviction| eviction.token_count).sum();
        Ok(ResidentPrefixEviction {
            target_tokens,
            evicted_entries: evictions.len(),
            evicted_tokens,
        })
    }

    /// Evict only the resident-prefix cells needed to leave one native decode
    /// batch of headroom in the unified KV pool.
    ///
    /// The resident cache and active lanes share `n_ctx`. Treating `n_batch`
    /// as an unconditional eviction amount drains a healthy cache even when
    /// the pool already has ample room (and can erase every prefix when
    /// `n_batch` is larger than the resident working set). Account for both
    /// active-lane and resident-prefix occupancy first, then evict only the
    /// actual deficit. A zero pool size is the modelless/unknown-capacity
    /// fallback and conservatively reserves one complete decode batch.
    pub fn evict_resident_prefix_for_decode_batch(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
    ) -> Result<ResidentPrefixEviction> {
        let decode_batch_tokens = runtime.active_session_batch_size(session_id)? as u64;
        let active_session_tokens = runtime.session_stats().total_session_tokens;
        let resident_tokens = self
            .resident
            .lock()
            .expect("resident prefix cache lock poisoned")
            .stats()
            .resident_tokens;
        let target_tokens = proactive_resident_eviction_target(
            u64::from(runtime.kv_pool_tokens()),
            active_session_tokens,
            resident_tokens,
            decode_batch_tokens,
        );
        self.evict_resident_prefix_for_tokens(runtime, session_id, target_tokens)
    }

    pub fn release_resident_prefix(&self, page_id: &str) {
        self.resident
            .lock()
            .expect("resident prefix cache lock poisoned")
            .release(page_id);
    }

    pub fn record_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRecord>> {
        if !self.should_record() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        let token_count = identity
            .identity
            .token_count
            .try_into()
            .unwrap_or(usize::MAX)
            .min(token_ids.len());
        if token_count == 0 || (token_count as u64) < self.candidate_policy.min_tokens {
            return Ok(None);
        }
        let layer_count = identity
            .identity
            .layer_end
            .saturating_sub(identity.identity.layer_start)
            .max(1);
        let estimated_bytes = resident_estimated_bytes(token_count as u64, layer_count);
        let mut cache = self
            .resident
            .lock()
            .expect("resident prefix cache lock poisoned");
        let allocation = cache.allocate_for_record(
            &identity.page_id,
            token_count as u64,
            estimated_bytes,
            |seq_id| runtime.drop_resident_prefix_sequence(session_id, seq_id),
        )?;
        if !allocation.should_retain {
            return Ok(None);
        }
        if allocation.should_save {
            runtime.save_resident_prefix(session_id, allocation.seq_id, token_count as u64)?;
            cache.commit_record(
                identity.page_id.clone(),
                allocation.seq_id,
                token_count as u64,
                estimated_bytes,
            );
        }
        runtime.retain_resident_prefix_on_drop(
            session_id,
            identity.page_id.clone(),
            token_count as u64,
        )?;
        let stats = cache.stats();
        Ok(Some(ResidentPrefixRecord {
            page_id: identity.page_id.clone(),
            token_count,
            seq_id: allocation.seq_id,
            stored: allocation.should_save,
            evicted_entries: allocation.evictions.len(),
            evicted_tokens: allocation
                .evictions
                .iter()
                .map(|eviction| eviction.token_count)
                .sum(),
            entries: stats.entries,
            resident_tokens: stats.resident_tokens,
        }))
    }
}

fn proactive_resident_eviction_target(
    kv_pool_tokens: u64,
    active_session_tokens: u64,
    resident_tokens: u64,
    decode_batch_tokens: u64,
) -> u64 {
    if kv_pool_tokens == 0 {
        return decode_batch_tokens;
    }
    active_session_tokens
        .saturating_add(resident_tokens)
        .saturating_add(decode_batch_tokens)
        .saturating_sub(kv_pool_tokens)
}

fn resident_estimated_bytes(token_count: u64, layer_count: u32) -> u64 {
    token_count
        .saturating_mul(u64::from(layer_count))
        .saturating_mul(2)
}

#[cfg(test)]
mod proactive_eviction_tests {
    use super::proactive_resident_eviction_target;

    #[test]
    fn keeps_resident_prefixes_when_unified_pool_already_has_batch_headroom() {
        assert_eq!(
            proactive_resident_eviction_target(32_768, 776, 1_992, 2_048),
            0
        );
    }

    #[test]
    fn evicts_only_the_unified_pool_deficit() {
        assert_eq!(
            proactive_resident_eviction_target(8_192, 6_500, 1_500, 2_048),
            1_856
        );
    }

    #[test]
    fn unknown_pool_preserves_fixed_batch_fallback() {
        assert_eq!(
            proactive_resident_eviction_target(0, 776, 1_992, 2_048),
            2_048
        );
    }
}
