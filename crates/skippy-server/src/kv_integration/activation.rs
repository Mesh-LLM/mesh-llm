use skippy_cache::activation_page_id;
use skippy_protocol::{MessageBase, StageConfig};
use skippy_runtime::ActivationFrame;

use super::{KvStageIntegration, ResidentActivationRecord, ResidentActivationRestore};

impl KvStageIntegration {
    pub fn restore_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
    ) -> Option<ResidentActivationRestore> {
        if !self.should_lookup() || token_ids.is_empty() {
            return None;
        }
        let identity = self.prefill_identity(config, base, token_start, token_ids);
        let page_id = activation_page_id(&identity.page_id, activation_width);
        let lookup = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned")
            .lookup(&page_id)?;
        Some(ResidentActivationRestore {
            identity,
            page_id,
            token_count: lookup.token_count as usize,
            payload_bytes: lookup.byte_size as usize,
            entries: lookup.entries,
            frame: lookup.frame,
        })
    }

    pub fn record_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
        frame: &ActivationFrame,
    ) -> Option<ResidentActivationRecord> {
        if !self.should_record() || token_ids.is_empty() {
            return None;
        }
        let token_count = token_ids.len() as u64;
        if token_count < self.candidate_policy.min_tokens || frame.payload.is_empty() {
            return None;
        }
        if u64::from(frame.desc.token_count) != token_count
            || frame.desc.payload_bytes != frame.payload.len() as u64
        {
            return None;
        }
        let identity = self.prefill_identity(config, base, token_start, token_ids);
        let page_id = activation_page_id(&identity.page_id, activation_width);
        let mut cache = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        let stored = cache.record(
            page_id.clone(),
            token_count,
            frame.payload.len() as u64,
            frame.clone(),
        );
        let stats = cache.stats();
        Some(ResidentActivationRecord {
            page_id,
            token_count: token_count as usize,
            payload_bytes: frame.payload.len(),
            evicted_entries: stored.evicted_entries,
            evicted_bytes: stored.evicted_bytes,
            entries: stats.entries,
            resident_bytes: stats.resident_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{
        LoadMode, SCHEMA_VERSION, StageConfig, StageKvCacheConfig, StageKvCacheMode,
        StageKvCachePayload,
    };
    use skippy_runtime::{
        ActivationDesc, ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout,
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
            ctx_size: 8192,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
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
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
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

    fn activation_frame(token_count: u32, payload_bytes: usize) -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                token_count,
                sequence_count: 1,
                payload_bytes: payload_bytes as u64,
                flags: 0,
            },
            payload: vec![7; payload_bytes],
        }
    }

    #[test]
    fn resident_activation_records_and_restores_exact_frame() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(tokens.len() as u32, 64);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert!(record.is_some());
        let restored = kv
            .restore_resident_activation(&config, &test_base(), 0, &tokens, 4096)
            .expect("activation frame should restore by exact identity");
        assert_eq!(restored.frame, frame);
    }

    #[test]
    fn resident_activation_rejects_mismatched_frame_token_count() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(128, 64);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert!(record.is_none());
        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &tokens, 4096)
                .is_none()
        );
    }

    #[test]
    fn resident_activation_stays_exact_for_shared_prefix_candidates() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("resident cache enabled");
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);
        let frame = activation_frame(recorded_tokens.len() as u32, 64);

        let record =
            kv.record_resident_activation(&config, &test_base(), 0, &recorded_tokens, 4096, &frame);

        assert!(record.is_some());
        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &lookup_tokens, 4096)
                .is_none()
        );
    }

    #[test]
    fn resident_activation_keys_include_activation_width() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(tokens.len() as u32, 64);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert!(record.is_some());
        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &tokens, 8192)
                .is_none()
        );
    }
}
