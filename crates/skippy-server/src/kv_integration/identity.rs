use skippy_cache::{
    NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION, prefix_identity_with_namespace,
};
use skippy_protocol::{MessageBase, StageConfig};

use crate::kv_proto::{KvCodec, PageIdentity, PageLayout};

use super::{KvStageIntegration, PrefillKvIdentity};

impl KvStageIntegration {
    pub fn prefill_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> PrefillKvIdentity {
        let prefix = prefix_identity_with_namespace(
            config,
            token_start,
            token_ids,
            base.chat_template_id.as_deref(),
        );
        let identity = PageIdentity {
            model_id: config.model_id.clone(),
            model_revision: "unknown".to_string(),
            runtime_abi_version: NATIVE_KV_RUNTIME_ABI_VERSION.to_string(),
            topology_id: config.topology_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            prefix_hash: prefix.prefix_hash.clone(),
            session_id: base.session_id.clone(),
            token_start,
            token_count: prefix.token_count,
            generation: 1,
            layout: PageLayout::LayerContiguous as i32,
            codec: KvCodec::Fp16 as i32,
            tokenizer_id: base
                .tokenizer_id
                .clone()
                .unwrap_or_else(|| config.model_id.clone()),
            chat_template_id: base.chat_template_id.clone().unwrap_or_default(),
            position_config_hash: format!("ctx:{}", config.ctx_size),
            kv_dtype: NATIVE_KV_DTYPE.to_string(),
        };
        PrefillKvIdentity {
            identity,
            page_id: prefix.page_id,
        }
    }

    pub fn lookup_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        let mut token_counts = self.lookup_candidate_token_counts(token_ids.len() as u64);
        if self.payload.is_exact_state() {
            token_counts.extend(
                self.exact_states
                    .lock()
                    .expect("exact state cache lock poisoned")
                    // Include lengths that survive only on disk, otherwise a
                    // demoted prefix is never probed for and cannot be hit.
                    .all_token_counts_at_most(token_ids.len() as u64),
            );
            token_counts.sort_unstable_by(|left, right| right.cmp(left));
            token_counts.dedup();
        }
        token_counts
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    pub fn record_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.record_candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    /// Pick a recurrent/full-state checkpoint below the near-tail page.
    ///
    /// The near-tail grid page can still include a short request-specific tail.
    /// One additional stride below it leaves room for that tail while keeping
    /// suffix prefill bounded for a changed-tail request.
    pub(crate) fn exact_shared_checkpoint_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Option<PrefillKvIdentity> {
        self.candidate_policy
            .candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .filter(|token_count| *token_count < token_ids.len() as u64)
            .nth(1)
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_integration::ExactStateExtra;
    use skippy_cache::ExactStatePayload;
    use skippy_protocol::{
        LoadMode, SCHEMA_VERSION, StageConfig, StageKvCacheConfig, StageKvCacheMode,
        StageKvCachePayload,
    };

    use super::*;

    fn test_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "org/model:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 4,
            ctx_size: 512,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: StageKvCachePayload::ResidentKv,
                max_entries: 8,
                max_bytes: 0,
                min_tokens: 64,
                shared_prefix_stride_tokens: 32,
                shared_prefix_record_limit: 2,
            }),
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn test_base() -> MessageBase {
        MessageBase {
            schema_version: SCHEMA_VERSION,
            run_id: "run".to_string(),
            request_id: "request".to_string(),
            session_id: "session".to_string(),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            topology_id: "topology".to_string(),
            model_id: Some("org/model:Q4_K_M".to_string()),
            tokenizer_id: None,
            chat_template_id: Some("template".to_string()),
            seq: Some(1),
        }
    }

    #[test]
    fn lookup_identities_use_full_longest_prefix_grid() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("cache enabled");
        let tokens = (0..160).collect::<Vec<_>>();

        let lookup_counts = kv
            .lookup_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        let record_counts = kv
            .record_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();

        assert_eq!(lookup_counts, vec![160, 128, 96, 64]);
        assert_eq!(record_counts, vec![160, 128]);
    }

    #[test]
    fn same_prefix_different_tail_identities_share_recorded_grid_page() {
        let config = StageConfig {
            ctx_size: 8192,
            kv_cache: Some(StageKvCacheConfig {
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
                ..test_config().kv_cache.expect("test cache config")
            }),
            ..test_config()
        };
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("cache enabled");
        let base = test_base();
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);

        let record_identities = kv.record_identities(&config, &base, 0, &recorded_tokens);
        let lookup_identities = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

        let record_counts = record_identities
            .iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        assert_eq!(record_counts, vec![2214, 2176]);

        let recorded_shared = record_identities
            .iter()
            .find(|identity| identity.identity.token_count == 2176)
            .expect("record identities should include shared grid prefix");
        let lookup_shared = lookup_identities
            .iter()
            .find(|identity| identity.identity.token_count == 2176)
            .expect("lookup identities should probe shared grid prefix");
        let recorded_exact = record_identities
            .iter()
            .find(|identity| identity.identity.token_count == 2214)
            .expect("record identities should include exact first prompt");
        let lookup_exact = lookup_identities
            .iter()
            .find(|identity| identity.identity.token_count == 2231)
            .expect("lookup identities should include exact second prompt");

        let checkpoint = kv
            .exact_shared_checkpoint_identity(&config, &base, 0, &recorded_tokens)
            .expect("exact-state shared checkpoint");
        assert_eq!(checkpoint.identity.token_count, 2048);
        let lookup_checkpoint = lookup_identities
            .iter()
            .find(|identity| identity.identity.token_count == 2048)
            .expect("lookup identities should probe checkpoint");
        assert_eq!(checkpoint.page_id, lookup_checkpoint.page_id);
        assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
        assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
    }

    #[test]
    fn exact_state_lookup_probes_cached_non_grid_prefix_length() {
        let config = StageConfig {
            ctx_size: 8192,
            kv_cache: Some(StageKvCacheConfig {
                payload: StageKvCachePayload::KvRecurrent,
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
                max_entries: 1,
                ..test_config().kv_cache.expect("test cache config")
            }),
            ..test_config()
        };
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("cache enabled");
        let base = test_base();
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let recorded = kv.prefill_identity(&config, &base, 0, &recorded_tokens);
        kv.exact_states
            .lock()
            .expect("exact state cache lock poisoned")
            .record(
                recorded.page_id.clone(),
                recorded.identity.token_count,
                ExactStatePayload::kv_recurrent(Vec::new(), vec![1]),
                ExactStateExtra::default(),
            );
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);

        let lookup = kv
            .lookup_identities(&config, &base, 0, &lookup_tokens)
            .into_iter()
            .find(|identity| identity.identity.token_count == 2214)
            .expect("lookup should probe the retained exact-state prefix length");

        assert_eq!(lookup.page_id, recorded.page_id);
    }
}
