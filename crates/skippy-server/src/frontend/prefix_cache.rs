use super::*;

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
                false,
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
        let downstream_reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
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
