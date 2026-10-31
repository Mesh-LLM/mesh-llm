//! Compare token metadata before choosing a warm alias or loading durable bytes.
use super::*;

pub(super) struct RestoreCandidate<'a> {
    pub(super) identity: &'a PrefillKvIdentity,
    pub(super) location: Option<skippy_cache::L3Location>,
}

pub(super) struct DurableRestoreLookup<'a> {
    pub(super) started: Instant,
    pub(super) location: Option<&'a skippy_cache::L3Location>,
    pub(super) shared_fallback: bool,
}

impl KvStageIntegration {
    pub(super) fn ordered_restore_candidates<'a>(
        &self,
        identities: &'a [PrefillKvIdentity],
        shared_durable_fallback: bool,
    ) -> Vec<RestoreCandidate<'a>> {
        if !self.records_resident_prefixes() {
            return identities
                .iter()
                .map(|identity| RestoreCandidate {
                    identity,
                    location: None,
                })
                .collect();
        }
        let resident = identities
            .iter()
            .filter_map(|identity| self.probe_resident_prefix(identity))
            .map(|hit| hit.token_count as u64)
            .max()
            .unwrap_or(0);
        let Some(l3) = self.l3.as_ref() else {
            return Vec::new();
        };
        let mut durable = identities
            .iter()
            .filter_map(|identity| {
                let tokens = if shared_durable_fallback {
                    self.durable_exact_lookup_token_ids(&identity.token_ids)
                } else {
                    &identity.token_ids
                };
                let location = l3
                    .locate_longest(&identity.namespace, tokens, MAX_PREFIX_PROBES)
                    .ok()??;
                preflight_l3_kv_location(&location).ok()?;
                (location.token_count > resident).then_some((
                    location.token_count,
                    RestoreCandidate {
                        identity,
                        location: Some(location),
                    },
                ))
            })
            .collect::<Vec<_>>();
        // Never load a candidate shorter than the native fallback. Ties stay
        // in shared memory. Actual fill still owns descriptor/cost admission.
        durable.sort_by_key(|(count, _)| std::cmp::Reverse(*count));
        durable
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_integration::RadixResidentEntry;
    use skippy_cache::{L3CacheManager, StoreLimits};
    use skippy_protocol::{MessageBase, StageConfig, StageKvCacheConfig, StageKvCacheMode};
    use skippy_runtime::{MemoryCacheCapabilities, ModelStateKind};

    #[test]
    fn committed_longer_l3_beats_short_ram_and_equal_ram_keeps_native_ownership() {
        let root = tempfile::tempdir().unwrap();
        let manager = L3CacheManager::acquire(root.path(), StoreLimits::new(1 << 20, 0)).unwrap();
        let config = StageConfig {
            layer_end: 1,
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                min_tokens: 1,
                max_entries: 8,
                max_bytes: 0,
                l2_max_bytes: 1024,
                exact_max_bytes: None,
                codec: Default::default(),
                shared_prefix_stride_tokens: 1,
                shared_prefix_record_limit: 0,
            }),
            ..StageConfig::default()
        };
        let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
            &config,
            Some(ModelStateKind::Dense),
            Some(MemoryCacheCapabilities {
                resident: true,
                resident_checkpoint: true,
                resident_attention: true,
                kv_recurrent: true,
                full_state: true,
            }),
            Some(manager),
            None,
        )
        .unwrap()
        .unwrap();
        let base: MessageBase = serde_json::from_value(serde_json::json!({
            "schema_version":1,"run_id":"run","request_id":"request","session_id":"session",
            "stage_id":"stage","stage_index":0,"topology_id":"topology"
        }))
        .unwrap();
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let identity = kv.prefill_identity(&config, &base, 0, &tokens);
        let desc = super::super::tests::native_desc();
        let payload = ExactStatePayload::kv_recurrent(vec![1; 256], vec![2; 16]);
        let l3 = kv.l3.as_ref().unwrap();
        l3.spill(
            &identity.namespace,
            &tokens,
            &payload,
            Some(serde_json::to_string(&desc).unwrap()),
            None,
        )
        .unwrap();
        let location = l3
            .locate_longest(&identity.namespace, &tokens, 64)
            .unwrap()
            .unwrap();
        assert_eq!(location.token_count, 8);
        for (count, seq, expected) in [(4, 8, false), (8, 9, true)] {
            kv.radix
                .lock()
                .unwrap()
                .insert_resident(
                    &identity.namespace,
                    &tokens[..count],
                    16,
                    RadixResidentEntry {
                        page_id: format!("ram-{count}"),
                        seq_id: seq,
                        token_count: count as u64,
                        recompute_cost: 1,
                    },
                )
                .unwrap();
            assert_eq!(
                kv.ordered_restore_candidates(std::slice::from_ref(&identity), false)
                    .is_empty(),
                expected
            );
        }
        assert_eq!(
            kv.radix.lock().unwrap().stats().resident_entries,
            2,
            "metadata selection must not evict the resident fallback"
        );
    }
}
