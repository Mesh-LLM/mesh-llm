use super::*;

impl KvStageIntegration {
    pub fn record_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRecord>> {
        let Some((token_count, layers)) =
            self.resident_capture_boundary(runtime, session_id, identity, token_ids)?
        else {
            return Ok(None);
        };
        let tokens = &identity.token_ids[..token_count];
        let mut radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut sequences = lock_resident_sequences(&self.resident_sequences);
        if let Some(existing) = radix.resident_exact(&identity.namespace, tokens) {
            return Ok(Some(resident_record(
                existing.value,
                false,
                radix.stats(),
                (0, 0),
            )));
        }
        let mut evicted = (0_usize, 0_u64);
        let seq_id = match sequences.allocate() {
            Ok(sequence) => sequence,
            Err(error) => {
                let removed = evict_one_resident(&mut radix, &mut sequences, |victim| {
                    self.release_native_prefix(runtime, session_id, victim)
                })?;
                let Some(removed) = removed else {
                    return Err(error);
                };
                evicted.0 += 1;
                evicted.1 = evicted.1.saturating_add(removed.value.token_count);
                sequences.allocate()?
            }
        };
        if let Err(error) = runtime.save_resident_prefix(session_id, seq_id, token_count as u64) {
            if let Err(release_error) = sequences.release(seq_id) {
                sequences.force_quarantine(seq_id);
                return Err(error).context(format!(
                    "quarantined resident sequence {seq_id} after release failed: {release_error:#}"
                ));
            }
            return Err(error);
        }
        let entry = RadixResidentEntry {
            page_id: identity.page_id.clone(),
            seq_id,
            token_count: token_count as u64,
            recompute_cost: (token_count as u64).saturating_mul(u64::from(layers)),
        };
        let candidate = skippy_cache::RadixEvictionCandidate {
            namespace: identity.namespace.clone(),
            tokens: tokens.to_vec(),
            logical_bytes: 0,
            last_used: 0,
            value: entry.clone(),
        };
        let bytes = match runtime.resident_prefix_bytes(session_id, seq_id) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.release_unpublished_prefix(runtime, session_id, &mut sequences, &candidate)?;
                return Err(error);
            }
        };
        let room = if bytes > self.resident_checkpoint_max_bytes {
            Ok(false)
        } else {
            make_record_room(
                self.resident_config,
                &mut radix,
                &mut sequences,
                bytes,
                &mut evicted,
                |victim| self.release_native_prefix(runtime, session_id, victim),
            )
        };
        if !matches!(room, Ok(true)) {
            self.release_unpublished_prefix(runtime, session_id, &mut sequences, &candidate)?;
            room?;
            return Ok(None);
        }
        let inserted = insert_saved_resident(
            &mut radix,
            &mut sequences,
            identity.namespace.clone(),
            tokens,
            bytes,
            entry.clone(),
            |seq| runtime.drop_resident_prefix_sequence(session_id, seq),
        )?;
        let entry = if inserted {
            entry
        } else {
            radix
                .resident_exact(&identity.namespace, tokens)
                .context("occupied resident radix entry disappeared after rejected insert")?
                .value
        };
        #[cfg(test)]
        if inserted {
            crate::frontend::capture_trace::log_stored_identity(&identity.namespace, tokens);
        }
        self.verify_resident_ownership(radix.stats().resident_entries, sequences.stats().0)?;
        Ok(Some(resident_record(
            entry,
            inserted,
            radix.stats(),
            evicted,
        )))
    }

    fn resident_capture_boundary(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        token_ids: &[i32],
    ) -> Result<Option<(usize, u32)>> {
        if !self.should_record() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        let requested = usize::try_from(identity.identity.token_count)
            .unwrap_or(usize::MAX)
            .min(token_ids.len());
        let layers = identity
            .identity
            .layer_end
            .saturating_sub(identity.identity.layer_start)
            .max(1);
        let checkpoint = self.resident_requires_checkpoint();
        let count = if checkpoint {
            if runtime.canonical_session_position(session_id)? != requested as u64 {
                return Ok(None);
            }
            requested
        } else {
            recordable_token_count(self.resident_config, requested)
        };
        if count == 0 || (count as u64) < self.checkpoint_policy.min_tokens {
            return Ok(None);
        }
        anyhow::ensure!(
            identity.token_ids.get(..count) == token_ids.get(..count),
            "resident prefix identity does not match the session token path"
        );
        Ok(Some((count, layers)))
    }
}

fn resident_record(
    entry: RadixResidentEntry,
    stored: bool,
    stats: skippy_cache::UnifiedRadixCacheStats,
    evicted: (usize, u64),
) -> ResidentPrefixRecord {
    ResidentPrefixRecord {
        page_id: entry.page_id,
        token_count: entry.token_count as usize,
        seq_id: entry.seq_id,
        stored,
        evicted_entries: evicted.0,
        evicted_tokens: evicted.1,
        entries: stats.resident_entries,
        resident_tokens: stats.resident_tokens,
    }
}

impl KvStageIntegration {
    fn release_unpublished_prefix(
        &self,
        runtime: &mut RuntimeState,
        session: &str,
        sequences: &mut ResidentSequencePool,
        candidate: &skippy_cache::RadixEvictionCandidate<RadixResidentEntry>,
    ) -> Result<()> {
        if let Err(error) = self.release_native_prefix(runtime, session, candidate) {
            sequences.quarantine(candidate.value.seq_id)?;
            return Err(error);
        }
        sequences.release(candidate.value.seq_id)
    }
}

fn make_record_room(
    config: skippy_cache::ResidentCacheConfig,
    radix: &mut skippy_cache::UnifiedRadixCache<RadixResidentEntry, super::super::RadixExactEntry>,
    sequences: &mut ResidentSequencePool,
    bytes: u64,
    evicted: &mut (usize, u64),
    mut drop_native: impl FnMut(&skippy_cache::RadixEvictionCandidate<RadixResidentEntry>) -> Result<()>,
) -> Result<bool> {
    while resident_index_over_capacity(config, radix.stats(), bytes) {
        let Some(removed) = evict_one_resident(radix, sequences, &mut drop_native)? else {
            return Ok(false);
        };
        evicted.0 = evicted.0.saturating_add(1);
        evicted.1 = evicted.1.saturating_add(removed.value.token_count);
    }
    Ok(true)
}

/// Dense KV can retain a shorter alias; recurrent checkpoints cannot be sliced.
fn recordable_token_count(config: skippy_cache::ResidentCacheConfig, requested: usize) -> usize {
    if config.max_resident_tokens > 0 {
        requested.min(usize::try_from(config.max_resident_tokens).unwrap_or(usize::MAX))
    } else {
        requested
    }
}

pub(super) fn resident_index_over_capacity(
    config: skippy_cache::ResidentCacheConfig,
    stats: skippy_cache::UnifiedRadixCacheStats,
    new_entry_bytes: u64,
) -> bool {
    stats.resident_entries.saturating_add(1) > config.max_entries
        || (config.max_bytes > 0
            && stats.resident_logical_bytes.saturating_add(new_entry_bytes) > config.max_bytes)
}

#[cfg(test)]
mod resident_record_cap_tests {
    use super::*;

    fn config(max_resident_tokens: u64, max_bytes: u64) -> skippy_cache::ResidentCacheConfig {
        skippy_cache::ResidentCacheConfig {
            max_entries: 64,
            max_bytes,
            min_tokens: 1,
            reserved_seq_count: 8,
            max_resident_tokens,
        }
    }

    #[test]
    fn prompt_above_the_cell_cap_records_the_cap_instead_of_nothing() {
        // mesh-llm#1358: n_ctx 28_672 -> cap 25_088. A 25_989-token prompt used
        // to be dropped entirely; it must be recorded at the cap.
        let ctx: u64 = 28_672;
        let cap = ctx - ctx / 8;
        assert_eq!(cap, 25_088);
        assert_eq!(recordable_token_count(config(cap, 0), 25_989), 25_088);
        assert_eq!(recordable_token_count(config(cap, 0), 24_000), 24_000);
    }

    #[test]
    fn unset_caps_keep_the_requested_prefix() {
        assert_eq!(recordable_token_count(config(0, 0), 40_000), 40_000);
    }

    #[test]
    fn host_budget_does_not_truncate_dense_token_identity() {
        // Host allocations are charged after capture; shared attention cells
        // are controlled by measured native pool occupancy.
        assert_eq!(recordable_token_count(config(0, 1), 40_000), 40_000);
    }

    #[test]
    fn clamped_entry_is_found_by_a_longer_request() {
        // Restore is a longest-prefix match, so an entry recorded at the cap
        // still serves a longer prompt: the identical re-send reuses the cap
        // and prefills only the remainder.
        let stored: Vec<i32> = (0..100).collect();
        let mut radix: skippy_cache::UnifiedRadixCache<
            RadixResidentEntry,
            crate::kv_integration::RadixExactEntry,
        > = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &stored,
                100,
                RadixResidentEntry {
                    page_id: "page".to_string(),
                    seq_id,
                    token_count: 100,
                    recompute_cost: 100,
                },
            )
            .unwrap();
        let longer: Vec<i32> = (0..140).collect();
        let hit = radix
            .peek_resident("stage", &longer)
            .expect("clamped prefix must still match a longer request");
        assert_eq!(hit.matched_tokens, 100);
    }
}

#[cfg(test)]
mod checkpoint_budget_tests {
    use super::*;

    #[test]
    fn native_host_bytes_evict_unpinned_owner_without_summing_shared_kv() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let first = sequences.allocate().unwrap();
        let second = sequences.allocate().unwrap();
        for (tokens, seq, bytes) in [(&[1, 2][..], first, 60), (&[1, 2, 3][..], second, 40)] {
            radix
                .insert_resident(
                    "model",
                    tokens,
                    bytes,
                    RadixResidentEntry {
                        page_id: seq.to_string(),
                        seq_id: seq,
                        token_count: tokens.len() as u64,
                        recompute_cost: 1,
                    },
                )
                .unwrap();
        }
        let pin = radix.acquire_resident_checkpoint("model", &[1, 2]).unwrap();
        let config = skippy_cache::ResidentCacheConfig {
            max_entries: 8,
            max_bytes: 100,
            min_tokens: 1,
            reserved_seq_count: 4,
            max_resident_tokens: 2,
        };
        let mut dropped = Vec::new();
        let mut evicted = (0, 0);
        assert!(
            make_record_room(
                config,
                &mut radix,
                &mut sequences,
                30,
                &mut evicted,
                |seq| {
                    dropped.push(seq.value.seq_id);
                    Ok(())
                }
            )
            .unwrap()
        );
        assert_eq!(dropped, [second]);
        assert_eq!(radix.stats().resident_logical_bytes, 60);
        assert!(radix.resident_exact("model", &[1, 2]).is_some());
        assert!(radix.release_resident("model", &pin.stored_tokens));
    }

    #[test]
    fn pinned_checkpoint_owners_cannot_be_discarded_to_satisfy_ram_budget() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let seq = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "model",
                &[1, 2],
                90,
                RadixResidentEntry {
                    page_id: "owner".into(),
                    seq_id: seq,
                    token_count: 2,
                    recompute_cost: 1,
                },
            )
            .unwrap();
        radix
            .acquire_resident_checkpoint("model", &[1, 2, 3])
            .unwrap();
        let config = skippy_cache::ResidentCacheConfig {
            max_entries: 8,
            max_bytes: 100,
            min_tokens: 1,
            reserved_seq_count: 4,
            max_resident_tokens: 0,
        };
        assert!(
            !make_record_room(
                config,
                &mut radix,
                &mut sequences,
                30,
                &mut (0, 0),
                |_| panic!("pinned checkpoint must not be dropped")
            )
            .unwrap()
        );
        assert_eq!(radix.stats().resident_entries, 1);
        assert_eq!(sequences.stats().0, 1);
    }
}
