use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::NativeMtpDraft;
use crate::frontend::generation::ChainPrefixRestore;
use crate::frontend::generation::EmbeddedExecutionStats;
use crate::frontend::generation::EmbeddedFusedFirstDecode;
use crate::frontend::generation::EmbeddedStageZeroGeneration;
use crate::frontend::generation::MAX_EXACT_REPLAY_TOKENS;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::DecodeMessageArgs;
use crate::frontend::wire_messages::RestorePrefillDecodeMessageArgs;
use crate::frontend::wire_messages::embedded_decode_message;
use crate::frontend::wire_messages::embedded_prefix_cache_message;
use crate::frontend::wire_messages::embedded_restore_prefill_decode_message;
use crate::frontend::wire_messages::openai_stage_mask;
use crate::kv_integration::KvStageIntegration;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use anyhow::Context;
use anyhow::anyhow;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use skippy_protocol::MessageBase;
use skippy_protocol::SCHEMA_VERSION;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageSamplingConfig as WireSamplingConfig;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireActivationDType;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use skippy_runtime::ActivationFrame;
use skippy_runtime::SamplingConfig;
use std::collections::BTreeMap;
use std::net::TcpStream;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ChainPrefixCacheSavings {
    pub(super) hit_stage_count: u32,
    pub(super) stage0_activation_bytes_avoided: usize,
    pub(super) interstage_activation_bytes_avoided_estimate: usize,
}

pub(super) fn chain_prefix_cache_savings(
    stats: &StageReplyStats,
    restored_tokens: usize,
    wire_dtype: WireActivationDType,
    activation_width: i32,
) -> ChainPrefixCacheSavings {
    let hit_stage_count = prefix_cache_hit_stage_count(stats.kv_hit_stage_mask);
    let stage0_activation_bytes_avoided =
        estimated_activation_bytes(wire_dtype, restored_tokens, activation_width);
    let interstage_activation_bytes_avoided_estimate =
        stage0_activation_bytes_avoided.saturating_mul(hit_stage_count.saturating_sub(1) as usize);
    ChainPrefixCacheSavings {
        hit_stage_count,
        stage0_activation_bytes_avoided,
        interstage_activation_bytes_avoided_estimate,
    }
}

pub(super) fn insert_chain_prefix_cache_savings_attrs(
    attrs: &mut BTreeMap<String, Value>,
    savings: ChainPrefixCacheSavings,
) {
    attrs.insert(
        "skippy.kv.chain_cache_hit_stage_count".to_string(),
        json!(savings.hit_stage_count),
    );
    attrs.insert(
        "skippy.kv.chain_cache_stage0_activation_bytes_avoided".to_string(),
        json!(savings.stage0_activation_bytes_avoided),
    );
    attrs.insert(
        "skippy.kv.chain_cache_interstage_activation_bytes_avoided_estimate".to_string(),
        json!(savings.interstage_activation_bytes_avoided_estimate),
    );
}

fn prefix_cache_hit_stage_count(hit_stage_mask: i64) -> u32 {
    if hit_stage_mask <= 0 {
        return 0;
    }
    (hit_stage_mask as u64).count_ones()
}

fn estimated_activation_bytes(
    wire_dtype: WireActivationDType,
    token_count: usize,
    activation_width: i32,
) -> usize {
    let Ok(token_count) = i32::try_from(token_count) else {
        return 0;
    };
    skippy_protocol::binary::activation_wire_bytes(wire_dtype, token_count, activation_width)
        .unwrap_or(0)
}

pub(super) fn request_allows_exact_replay(request: &EmbeddedStageZeroGeneration<'_>) -> bool {
    request.draft.is_none() && request.sampling.temperature <= 0.0
}

fn exact_replay_cache_key(
    identity: &crate::kv_integration::PrefillKvIdentity,
    sampling: &SamplingConfig,
    chat_sampling_metadata: Option<&str>,
) -> String {
    use std::fmt::Write as _;

    let mut digest = Sha256::new();
    digest.update(b"skippy-exact-replay-v1");
    digest.update(identity.page_id.as_bytes());
    digest.update([u8::from(sampling.enabled)]);
    digest.update(sampling.seed.to_le_bytes());
    digest.update(sampling.temperature.to_bits().to_le_bytes());
    digest.update(sampling.top_p.to_bits().to_le_bytes());
    digest.update(sampling.top_k.to_le_bytes());
    digest.update(sampling.min_p.to_bits().to_le_bytes());
    digest.update(sampling.presence_penalty.to_bits().to_le_bytes());
    digest.update(sampling.frequency_penalty.to_bits().to_le_bytes());
    digest.update(sampling.repeat_penalty.to_bits().to_le_bytes());
    digest.update(sampling.penalty_last_n.to_le_bytes());
    for bias in &sampling.logit_bias {
        digest.update(b"logit-bias");
        digest.update(bias.token_id.to_le_bytes());
        digest.update(bias.bias.to_bits().to_le_bytes());
    }
    if let Some(metadata) = chat_sampling_metadata {
        digest.update(b"chat-sampling-metadata");
        digest.update(metadata.as_bytes());
    }
    let digest = digest.finalize();
    let mut suffix = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut suffix, "{byte:02x}");
    }
    format!("{}:replay:{suffix}", identity.page_id)
}

pub(super) fn stage0_prefill_record_identities(
    kv: &KvStageIntegration,
    config: &StageConfig,
    base: &MessageBase,
    token_start: u64,
    token_ids: &[i32],
) -> Vec<crate::kv_integration::PrefillKvIdentity> {
    kv.record_identities(config, base, token_start, token_ids)
}

pub(super) fn stage0_full_prefill_record_identities(
    kv: &KvStageIntegration,
    config: &StageConfig,
    base: &MessageBase,
    token_ids: &[i32],
) -> Vec<crate::kv_integration::PrefillKvIdentity> {
    stage0_prefill_record_identities(kv, config, base, 0, token_ids)
}

pub(super) struct EmbeddedReplayCheckpointRecord<'a> {
    pub(super) session_id: &'a str,
    pub(super) ids: &'a OpenAiGenerationIds,
    pub(super) prompt_token_ids: &'a [i32],
    pub(super) checkpoint_token_ids: &'a [i32],
    pub(super) predicted_tokens: &'a [i32],
    pub(super) predicted: i32,
    pub(super) sampling: &'a SamplingConfig,
    pub(super) chat_sampling_metadata: Option<&'a str>,
}

impl StageOpenAiBackend {
    pub(super) fn local_kv_message_base(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
    ) -> MessageBase {
        MessageBase {
            schema_version: SCHEMA_VERSION,
            run_id: self.config.run_id.clone(),
            request_id: ids.request_id_string(),
            session_id: session_id.to_string(),
            stage_id: "openai-local".to_string(),
            stage_index: self.config.stage_index,
            topology_id: self.config.topology_id.clone(),
            model_id: Some(self.config.model_id.clone()),
            tokenizer_id: None,
            chat_template_id: ids.cache.namespace(),
            seq: Some(ids.session_id),
        }
    }

    pub(super) fn evict_embedded_stage0_resident_prefix(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        target_tokens: Option<u64>,
    ) -> OpenAiResult<()> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(());
        };
        let eviction = (|| {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| anyhow!("runtime lock poisoned"))?;
            runtime
                .ensure_session_active(session_id)
                .context("activate embedded stage-0 session before resident-prefix eviction")?;
            if let Some(target_tokens) = target_tokens {
                kv.evict_resident_prefix_for_tokens(&mut runtime, session_id, target_tokens)
            } else {
                kv.evict_resident_prefix_for_decode_batch(&mut runtime, session_id)
            }
        })();
        let (status, error_kind, target_tokens, evicted_entries, evicted_tokens) = match &eviction {
            Ok(eviction) => (
                if eviction.evicted_entries > 0 {
                    "evicted"
                } else {
                    "noop"
                },
                None,
                eviction.target_tokens,
                eviction.evicted_entries,
                eviction.evicted_tokens,
            ),
            Err(error) => (
                "error",
                Some(proactive_eviction_error_kind(error)),
                target_tokens.unwrap_or_default(),
                0,
                0,
            ),
        };
        let mut attrs = self.openai_attrs(ids);
        attrs.extend(proactive_eviction_attrs(
            status,
            error_kind,
            target_tokens,
            evicted_entries,
            evicted_tokens,
        ));
        if error_kind.is_some() || evicted_entries > 0 || evicted_tokens > 0 {
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        } else {
            self.telemetry
                .emit_debug("stage.openai_kv_record_decision", attrs);
        }
        eviction
            .map(|_| ())
            .map_err(|error| openai_backend_error(error.context("evict embedded stage-0 KV")))
    }

    pub(super) fn restore_embedded_stage0_prefill(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
    ) -> OpenAiResult<Option<ActivationFrame>> {
        if token_start != 0 {
            return Ok(None);
        }
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        let base = self.local_kv_message_base(session_id, ids);
        let Some(activation) = kv.restore_resident_activation(
            &self.config,
            &base,
            token_start,
            token_ids,
            activation_width,
        ) else {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_activation_miss"),
            );
            attrs.insert("skippy.kv.token_start".to_string(), json!(token_start));
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            self.telemetry
                .emit("stage.openai_kv_lookup_decision", attrs);
            return Ok(None);
        };
        let restored = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            kv.restore_resident_prefix(
                &mut runtime,
                session_id,
                std::slice::from_ref(&activation.identity),
                token_ids,
            )
            .map_err(openai_backend_error)?
        };
        let Some(restored) = restored else {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_activation_hit_kv_miss"),
            );
            attrs.insert(
                "skippy.activation_cache.hit_page_id".to_string(),
                json!(activation.page_id),
            );
            self.telemetry
                .emit("stage.openai_kv_lookup_decision", attrs);
            return Ok(None);
        };
        if restored.token_count < token_ids.len() {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_activation_hit_prefix_short"),
            );
            attrs.insert(
                "skippy.kv.restored_tokens".to_string(),
                json!(restored.token_count),
            );
            self.telemetry
                .emit("stage.openai_kv_lookup_decision", attrs);
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("stage0_activation_resident_hit"),
        );
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restored.token_count),
        );
        attrs.insert(
            "skippy.kv.resident_lane_hit".to_string(),
            json!(restored.borrowed),
        );
        attrs.insert(
            "skippy.activation_cache.hit_page_id".to_string(),
            json!(activation.page_id),
        );
        attrs.insert(
            "skippy.activation_cache.payload_bytes".to_string(),
            json!(activation.payload_bytes),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        Ok(Some(activation.frame))
    }

    pub(super) fn record_embedded_stage0_prefill(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
        output: &ActivationFrame,
    ) -> OpenAiResult<()> {
        if token_start != 0 {
            return Ok(());
        }
        let Some(kv) = self.kv.as_ref() else {
            return Ok(());
        };
        let base = self.local_kv_message_base(session_id, ids);
        let identities =
            stage0_prefill_record_identities(kv, &self.config, &base, token_start, token_ids);
        let record_candidate_count = identities.len();
        let resident_records = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            identities
                .iter()
                .map(|identity| {
                    let token_count = identity
                        .identity
                        .token_count
                        .try_into()
                        .unwrap_or(usize::MAX)
                        .min(token_ids.len());
                    kv.record_resident_prefix(
                        &mut runtime,
                        session_id,
                        identity,
                        &token_ids[..token_count],
                    )
                    .map_err(openai_backend_error)
                })
                .collect::<OpenAiResult<Vec<_>>>()?
        };
        let activation_record = kv.record_resident_activation(
            &self.config,
            &base,
            token_start,
            token_ids,
            activation_width,
            output,
        );
        let mut recorded_any = false;
        for record in resident_records.into_iter().flatten() {
            recorded_any = true;
            let mut attrs = self.openai_attrs(ids);
            attrs.insert("skippy.kv.decision".to_string(), json!("stage0_record"));
            attrs.insert(
                "skippy.kv.record_candidates".to_string(),
                json!(record_candidate_count),
            );
            attrs.insert("skippy.kv.token_start".to_string(), json!(token_start));
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            attrs.insert(
                "skippy.kv.recorded_page_id".to_string(),
                json!(record.page_id),
            );
            attrs.insert(
                "skippy.kv.recorded_tokens".to_string(),
                json!(record.token_count),
            );
            attrs.insert(
                "skippy.kv.resident_seq_id".to_string(),
                json!(record.seq_id),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        }
        if let Some(record) = activation_record {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_activation_record"),
            );
            attrs.insert(
                "skippy.kv.record_candidates".to_string(),
                json!(record_candidate_count),
            );
            attrs.insert("skippy.kv.token_start".to_string(), json!(token_start));
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            attrs.insert(
                "skippy.activation_cache.recorded_page_id".to_string(),
                json!(record.page_id),
            );
            attrs.insert(
                "skippy.activation_cache.payload_bytes".to_string(),
                json!(record.payload_bytes),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        } else if !recorded_any {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert("skippy.kv.decision".to_string(), json!("stage0_record"));
            attrs.insert(
                "skippy.kv.record_candidates".to_string(),
                json!(record_candidate_count),
            );
            attrs.insert("skippy.kv.token_start".to_string(), json!(token_start));
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        }
        Ok(())
    }

    pub(super) fn record_embedded_stage0_full_prefill(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        token_ids: &[i32],
    ) -> OpenAiResult<bool> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(false);
        };
        if token_ids.is_empty() || !kv.should_record() {
            return Ok(false);
        }
        let base = self.local_kv_message_base(session_id, ids);
        let identities = stage0_full_prefill_record_identities(kv, &self.config, &base, token_ids);
        let record_candidate_count = identities.len();
        let records = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            identities
                .iter()
                .map(|identity| {
                    let token_count = identity
                        .identity
                        .token_count
                        .try_into()
                        .unwrap_or(usize::MAX)
                        .min(token_ids.len());
                    if token_count == token_ids.len() {
                        let _ = kv.record_exact_state(&mut runtime, session_id, identity);
                    }
                    kv.record_resident_prefix(
                        &mut runtime,
                        session_id,
                        identity,
                        &token_ids[..token_count],
                    )
                    .map_err(openai_backend_error)
                })
                .collect::<OpenAiResult<Vec<_>>>()?
        };
        let mut recorded_any = false;
        for record in records.into_iter().flatten() {
            recorded_any = true;
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_full_prefill_record"),
            );
            attrs.insert(
                "skippy.kv.record_candidates".to_string(),
                json!(record_candidate_count),
            );
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            attrs.insert(
                "skippy.kv.recorded_page_id".to_string(),
                json!(record.page_id),
            );
            attrs.insert(
                "skippy.kv.recorded_tokens".to_string(),
                json!(record.token_count),
            );
            attrs.insert(
                "skippy.kv.resident_seq_id".to_string(),
                json!(record.seq_id),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        }
        if !recorded_any {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("stage0_full_prefill_record"),
            );
            attrs.insert(
                "skippy.kv.record_candidates".to_string(),
                json!(record_candidate_count),
            );
            attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
        }
        Ok(recorded_any)
    }

    pub(super) fn record_embedded_stage0_full_prompt_first_token(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        token_ids: &[i32],
        predicted: i32,
    ) -> OpenAiResult<bool> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(false);
        };
        if token_ids.is_empty() || !kv.should_record() {
            return Ok(false);
        }
        let base = self.local_kv_message_base(session_id, ids);
        let identity = kv.prefill_identity(&self.config, &base, 0, token_ids);
        let recorded_state =
            self.record_embedded_stage0_full_prefill(session_id, ids, token_ids)?;
        let recorded_token = kv.record_cached_first_token(&identity, predicted);
        let mut attrs = self.openai_attrs(ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("stage0_full_prompt_first_token_record"),
        );
        attrs.insert("skippy.kv.token_count".to_string(), json!(token_ids.len()));
        attrs.insert("skippy.kv.predicted_token".to_string(), json!(predicted));
        attrs.insert(
            "skippy.kv.recorded_page_id".to_string(),
            json!(identity.page_id),
        );
        attrs.insert(
            "skippy.kv.recorded_state".to_string(),
            json!(recorded_state),
        );
        attrs.insert(
            "skippy.kv.recorded_first_token".to_string(),
            json!(recorded_token),
        );
        self.telemetry
            .emit("stage.openai_kv_record_decision", attrs);
        Ok(recorded_state || recorded_token)
    }

    pub(super) fn record_embedded_stage0_replay_checkpoint(
        &self,
        record: EmbeddedReplayCheckpointRecord<'_>,
    ) -> OpenAiResult<bool> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(false);
        };
        if record.prompt_token_ids.is_empty()
            || record.checkpoint_token_ids.is_empty()
            || record.predicted_tokens.len() >= MAX_EXACT_REPLAY_TOKENS
            || !kv.should_record()
        {
            return Ok(false);
        }
        let base = self.local_kv_message_base(record.session_id, record.ids);
        let prompt_identity = kv.prefill_identity(&self.config, &base, 0, record.prompt_token_ids);
        let replay_cache_key = exact_replay_cache_key(
            &prompt_identity,
            record.sampling,
            record.chat_sampling_metadata,
        );
        let recorded_state = self.record_embedded_stage0_full_prefill(
            record.session_id,
            record.ids,
            record.checkpoint_token_ids,
        )?;
        let recorded_replay = kv.record_cached_replay_tokens(
            &replay_cache_key,
            &prompt_identity,
            record.predicted_tokens,
            record.predicted,
            MAX_EXACT_REPLAY_TOKENS,
        );
        let mut attrs = self.openai_attrs(record.ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("stage0_exact_replay_record"),
        );
        attrs.insert(
            "skippy.kv.prompt_token_count".to_string(),
            json!(record.prompt_token_ids.len()),
        );
        attrs.insert(
            "skippy.kv.checkpoint_token_count".to_string(),
            json!(record.checkpoint_token_ids.len()),
        );
        attrs.insert(
            "skippy.kv.replay_token_count".to_string(),
            json!(recorded_replay.unwrap_or(record.predicted_tokens.len())),
        );
        attrs.insert(
            "skippy.kv.predicted_token".to_string(),
            json!(record.predicted),
        );
        attrs.insert(
            "skippy.kv.recorded_page_id".to_string(),
            json!(prompt_identity.page_id),
        );
        attrs.insert(
            "skippy.kv.recorded_state".to_string(),
            json!(recorded_state),
        );
        attrs.insert(
            "skippy.kv.recorded_replay".to_string(),
            json!(recorded_replay.is_some()),
        );
        self.telemetry
            .emit("stage.openai_kv_record_decision", attrs);
        Ok(recorded_state || recorded_replay.is_some())
    }

    pub(super) fn try_restore_embedded_split_exact_replay(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
    ) -> OpenAiResult<Option<EmbeddedFusedFirstDecode>> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        if request.prompt_token_ids.is_empty()
            || !kv.should_lookup()
            || !request_allows_exact_replay(request)
        {
            return Ok(None);
        }
        let timer = PhaseTimer::start();
        let base = self.local_kv_message_base(session_key, request.ids);
        let prompt_identity =
            kv.prefill_identity(request.config, &base, 0, request.prompt_token_ids);
        let replay_cache_key = exact_replay_cache_key(
            &prompt_identity,
            request.sampling,
            request.chat_sampling_metadata,
        );
        let replay_tokens =
            kv.lookup_cached_replay_tokens(&replay_cache_key, request.max_tokens as usize);
        if replay_tokens.len() < 2 {
            return Ok(None);
        }

        for replay_len in (2..=replay_tokens.len()).rev() {
            let mut checkpoint_tokens = request.prompt_token_ids.to_vec();
            checkpoint_tokens.extend_from_slice(&replay_tokens[..replay_len - 1]);
            let Some(restore) = self.try_restore_embedded_split_prefill(
                request,
                session_key,
                downstream,
                &checkpoint_tokens,
            )?
            else {
                continue;
            };
            if restore.restored_tokens < checkpoint_tokens.len() {
                continue;
            }
            let replay = replay_tokens[..replay_len].to_vec();
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("chain_exact_replay_hit"),
            );
            attrs.insert(
                "skippy.kv.prompt_token_count".to_string(),
                json!(request.prompt_token_ids.len()),
            );
            attrs.insert(
                "skippy.kv.checkpoint_token_count".to_string(),
                json!(checkpoint_tokens.len()),
            );
            attrs.insert(
                "skippy.kv.replay_token_count".to_string(),
                json!(replay.len()),
            );
            attrs.insert(
                "skippy.kv.restored_tokens".to_string(),
                json!(restore.restored_tokens),
            );
            attrs.insert(
                "skippy.kv.hit_page_id".to_string(),
                json!(prompt_identity.page_id),
            );
            attrs.insert(
                "skippy.kv.lookup_hits".to_string(),
                json!(restore.stats.kv_lookup_hits),
            );
            attrs.insert(
                "skippy.kv.hit_stage_mask".to_string(),
                json!(restore.stats.kv_hit_stage_mask),
            );
            insert_chain_prefix_cache_savings_attrs(
                &mut attrs,
                chain_prefix_cache_savings(
                    &restore.stats,
                    checkpoint_tokens.len(),
                    request.wire_dtype,
                    request.activation_width,
                ),
            );
            self.telemetry
                .emit("stage.openai_kv_lookup_decision", attrs);
            return Ok(Some(EmbeddedFusedFirstDecode {
                predicted: *replay.last().expect("checked replay length"),
                predicted_tokens: replay,
                native_mtp_draft: None,
                reply_stats: restore.stats,
                execution: EmbeddedExecutionStats::default(),
                elapsed_ms: timer.elapsed_ms(),
                token_phase: "exact-replay-cache",
                message_kind: "TryRestorePrefill",
            }));
        }
        Ok(None)
    }

    pub(super) fn try_restore_embedded_split_full_prompt_first_token(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
    ) -> OpenAiResult<Option<EmbeddedFusedFirstDecode>> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        if request.prompt_token_ids.is_empty() || !kv.should_lookup() {
            return Ok(None);
        }
        let timer = PhaseTimer::start();
        let base = self.local_kv_message_base(session_key, request.ids);
        let identity = kv.prefill_identity(request.config, &base, 0, request.prompt_token_ids);
        let Some(predicted) = kv.lookup_cached_first_token(&identity) else {
            return Ok(None);
        };
        let Some(restore) = self.try_restore_embedded_split_prefill(
            request,
            session_key,
            downstream,
            request.prompt_token_ids,
        )?
        else {
            return Ok(None);
        };
        if restore.restored_tokens < request.prompt_token_ids.len() {
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("chain_full_prompt_first_token_hit"),
        );
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restore.restored_tokens),
        );
        attrs.insert("skippy.kv.predicted_token".to_string(), json!(predicted));
        attrs.insert("skippy.kv.hit_page_id".to_string(), json!(identity.page_id));
        attrs.insert(
            "skippy.kv.lookup_hits".to_string(),
            json!(restore.stats.kv_lookup_hits),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(restore.stats.kv_hit_stage_mask),
        );
        insert_chain_prefix_cache_savings_attrs(
            &mut attrs,
            chain_prefix_cache_savings(
                &restore.stats,
                restore.restored_tokens,
                request.wire_dtype,
                request.activation_width,
            ),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        Ok(Some(EmbeddedFusedFirstDecode {
            predicted,
            predicted_tokens: vec![predicted],
            native_mtp_draft: None,
            reply_stats: restore.stats,
            execution: EmbeddedExecutionStats::default(),
            elapsed_ms: timer.elapsed_ms(),
            token_phase: "full-prompt-cache",
            message_kind: "TryRestorePrefill",
        }))
    }

    pub(super) fn try_restore_embedded_split_prefill(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
        prefill_tokens: &[i32],
    ) -> OpenAiResult<Option<ChainPrefixRestore>> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        if prefill_tokens.is_empty() || !kv.should_lookup() {
            return Ok(None);
        }
        let base = self.local_kv_message_base(session_key, request.ids);
        let identities = kv.lookup_identities(request.config, &base, 0, prefill_tokens);
        let mut restore_stats = StageReplyStats::default();
        let local_restore = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            match kv
                .restore_exact_state(&mut runtime, session_key, &identities)
                .map_err(openai_backend_error)?
            {
                Some(restored) => Some(restored.token_count),
                None => kv
                    .restore_resident_prefix(&mut runtime, session_key, &identities, prefill_tokens)
                    .map_err(openai_backend_error)?
                    .map(|restored| restored.token_count),
            }
        };
        let Some(local_restore) = local_restore else {
            return Ok(None);
        };
        if local_restore == 0 {
            return Ok(None);
        }
        let restored_tokens = local_restore.min(prefill_tokens.len());
        restore_stats.kv_lookup_hits += 1;
        restore_stats.kv_imported_pages += 1;
        restore_stats.kv_imported_tokens += restored_tokens as i64;
        restore_stats.kv_hit_stage_mask |= openai_stage_mask(request.config.stage_index);
        let restore = embedded_prefix_cache_message(
            WireMessageKind::TryRestorePrefill,
            request.wire_dtype,
            &prefill_tokens[..restored_tokens],
            request.ids.request_id,
            request.ids.session_id,
        )?;
        write_stage_message_conditioned(
            &mut *downstream,
            &restore,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_restore = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        if downstream_restore.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "expected prefix try-restore ACK from downstream, got {:?}",
                downstream_restore.kind
            )));
        }
        restore_stats.merge(downstream_restore.stats);
        if restore_stats.kv_lookup_errors > 0
            || restore_stats.kv_lookup_misses > 0
            || downstream_restore.stats.kv_lookup_hits == 0
        {
            self.drop_embedded_split_restore(request, session_key, downstream);
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("skippy.kv.decision".to_string(), json!("chain_restore_hit"));
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(restored_tokens),
        );
        attrs.insert(
            "skippy.kv.suffix_prefill_tokens".to_string(),
            json!(prefill_tokens.len().saturating_sub(restored_tokens)),
        );
        attrs.insert(
            "skippy.kv.lookup_hits".to_string(),
            json!(restore_stats.kv_lookup_hits),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(restore_stats.kv_hit_stage_mask),
        );
        insert_chain_prefix_cache_savings_attrs(
            &mut attrs,
            chain_prefix_cache_savings(
                &restore_stats,
                restored_tokens,
                request.wire_dtype,
                request.activation_width,
            ),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        Ok(Some(ChainPrefixRestore {
            restored_tokens,
            stats: restore_stats,
        }))
    }

    pub(super) fn try_restore_embedded_split_prefill_and_decode(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
        prefill_tokens: &[i32],
        current: i32,
        wire_sampling: Option<WireSamplingConfig>,
    ) -> OpenAiResult<Option<EmbeddedFusedFirstDecode>> {
        let Some(kv) = self.kv.as_ref() else {
            return Ok(None);
        };
        if prefill_tokens.is_empty() || !kv.should_lookup() {
            return Ok(None);
        }
        let timer = PhaseTimer::start();
        let base = self.local_kv_message_base(session_key, request.ids);
        let identity = kv.prefill_identity(request.config, &base, 0, prefill_tokens);
        let mut reply_stats = StageReplyStats::default();
        let local_restore = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            kv.restore_resident_prefix(
                &mut runtime,
                session_key,
                std::slice::from_ref(&identity),
                prefill_tokens,
            )
            .map_err(openai_backend_error)?
        };
        let Some(local_restore) = local_restore else {
            return Ok(None);
        };
        if local_restore.token_count < prefill_tokens.len() {
            return Ok(None);
        }
        reply_stats.kv_lookup_hits += 1;
        reply_stats.kv_imported_pages += 1;
        reply_stats.kv_imported_tokens += local_restore.token_count as i64;
        reply_stats.kv_hit_stage_mask |= openai_stage_mask(request.config.stage_index);

        let stage0_timer = PhaseTimer::start();
        let token_runtime_lock_wait_ms;
        let token_runtime_lock_hold_ms;
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            token_runtime_lock_wait_ms = lock_timer.elapsed_ms();
            let lock_hold_timer = PhaseTimer::start();
            if let Some(metadata) = request.chat_sampling_metadata {
                runtime
                    .configure_chat_sampling(
                        session_key,
                        metadata,
                        request.prompt_token_ids.len() as u64,
                        request.sampling.enabled.then_some(request.sampling),
                    )
                    .map_err(openai_backend_error)?;
            }
            let decode_message = embedded_decode_message(
                request.wire_dtype,
                DecodeMessageArgs {
                    request_id: request.ids.request_id,
                    session_id: request.ids.session_id,
                    prompt_token_count: request.prompt_token_ids.len(),
                    pos_start: prefill_tokens.len(),
                    decode_step: 0,
                    current,
                    sampling: wire_sampling.clone(),
                },
            )?;
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                &decode_message,
                &[current],
                None,
                BinaryStageExecutionOptions::new(
                    false,
                    stage_output_activation_capacity(
                        request.config,
                        decode_message.token_count,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?,
                    request.native_mtp_enabled,
                )
                .with_native_mtp_max_tokens(request.speculative.native_mtp.max_draft_tokens),
            )
            .map_err(openai_backend_error)?
            .2;
            token_runtime_lock_hold_ms = lock_hold_timer.elapsed_ms();
            output
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();

        let fused_message = embedded_restore_prefill_decode_message(
            request.wire_dtype,
            RestorePrefillDecodeMessageArgs {
                request_id: request.ids.request_id,
                session_id: request.ids.session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                pos_start: prefill_tokens.len(),
                decode_step: 0,
                prefix_tokens: prefill_tokens,
                current,
                sampling: wire_sampling,
                chat_sampling_metadata: request.chat_sampling_metadata,
            },
        )?;
        let forwarded = forwarded_stage_message_timed(
            request.config,
            &fused_message,
            &output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let forward_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let downstream_reply = super::embedded_execution::receive_embedded_stage_reply_one_of(
            downstream,
            request.prediction_return.as_ref(),
            &[WireReplyKind::PredictedToken, WireReplyKind::Ack],
        )?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        let downstream_missed = downstream_reply.kind != WireReplyKind::PredictedToken
            || downstream_reply.stats.kv_lookup_errors > 0
            || downstream_reply.stats.kv_lookup_misses > 0
            || downstream_reply.stats.kv_lookup_hits == 0;
        reply_stats.merge(downstream_reply.stats);
        if downstream_missed {
            self.drop_embedded_split_restore(request, session_key, downstream);
            return Ok(None);
        }
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!("chain_restore_decode_hit"),
        );
        attrs.insert(
            "skippy.kv.restored_tokens".to_string(),
            json!(prefill_tokens.len()),
        );
        attrs.insert(
            "skippy.kv.lookup_hits".to_string(),
            json!(reply_stats.kv_lookup_hits),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(reply_stats.kv_hit_stage_mask),
        );
        insert_chain_prefix_cache_savings_attrs(
            &mut attrs,
            chain_prefix_cache_savings(
                &reply_stats,
                prefill_tokens.len(),
                request.wire_dtype,
                request.activation_width,
            ),
        );
        self.telemetry
            .emit("stage.openai_kv_lookup_decision", attrs);
        self.record_embedded_stage0_full_prompt_first_token(
            session_key,
            request.ids,
            request.prompt_token_ids,
            downstream_reply.predicted,
        )?;
        Ok(Some(EmbeddedFusedFirstDecode {
            predicted: downstream_reply.predicted,
            predicted_tokens: vec![downstream_reply.predicted],
            native_mtp_draft: downstream_reply
                .native_mtp_draft
                .clone()
                .map(NativeMtpDraft::from_stage_draft),
            reply_stats,
            execution: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: token_runtime_lock_wait_ms,
                runtime_lock_hold_ms: token_runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms,
            },
            elapsed_ms: timer.elapsed_ms(),
            token_phase: "fused-restore",
            message_kind: "TryRestorePrefillDecode",
        }))
    }

    pub(super) fn drop_embedded_split_restore(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
        downstream: &mut TcpStream,
    ) {
        if let Ok(mut runtime) = self.runtime.lock() {
            let _ = runtime.drop_session_timed(session_key);
        }
        let stop = StageWireMessage::stop_with_identity(
            request.wire_dtype,
            request.ids.request_id,
            request.ids.session_id,
        );
        if write_stage_message_conditioned(
            &mut *downstream,
            &stop,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .is_ok()
        {
            let _ = recv_reply(&mut *downstream);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_prefix_cache_savings_counts_confirmed_stage_hits() {
        let stats = StageReplyStats {
            kv_hit_stage_mask: openai_stage_mask(0)
                | openai_stage_mask(1)
                | openai_stage_mask(2)
                | openai_stage_mask(3),
            ..Default::default()
        };

        let savings = chain_prefix_cache_savings(&stats, 256, WireActivationDType::F16, 5120);

        assert_eq!(savings.hit_stage_count, 4);
        assert_eq!(savings.stage0_activation_bytes_avoided, 2_621_440);
        assert_eq!(
            savings.interstage_activation_bytes_avoided_estimate,
            7_864_320
        );
    }

    #[test]
    fn chain_prefix_cache_savings_uses_wire_dtype() {
        let stats = StageReplyStats {
            kv_hit_stage_mask: openai_stage_mask(0) | openai_stage_mask(1),
            ..Default::default()
        };

        let q8 = chain_prefix_cache_savings(&stats, 256, WireActivationDType::Q8, 5120);
        let f32 = chain_prefix_cache_savings(&stats, 256, WireActivationDType::F32, 5120);

        assert_eq!(q8.stage0_activation_bytes_avoided, 1_311_744);
        assert_eq!(q8.interstage_activation_bytes_avoided_estimate, 1_311_744);
        assert_eq!(f32.stage0_activation_bytes_avoided, 5_242_880);
        assert_eq!(f32.interstage_activation_bytes_avoided_estimate, 5_242_880);
    }
}
