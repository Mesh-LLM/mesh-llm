use std::collections::VecDeque;

mod lifecycle;

use super::*;
use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::forwarded_stage_message;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::NativeMtpDecodeCounters;
use crate::frontend::NativeMtpDecodeOptions;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpDraftOrigin;
use crate::frontend::NativeMtpVerifier;
use crate::frontend::generation::EmbeddedStageZeroGeneration;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation::attrs_insert_prefill_chunk_policy;
use crate::frontend::generation::decode_token_phase;
use crate::frontend::prefill::PrefillChunkObservation;
use crate::frontend::prefill::drain_embedded_prefill_replies;
use crate::frontend::prefill::drain_one_embedded_prefill_reply;
use crate::frontend::request::wire_sampling_config;
use crate::frontend::speculative::OpenAiSpeculativeStats;
use crate::frontend::speculative::classify_verify_window;
use crate::frontend::speculative::propose_ngram_tokens;
use crate::frontend::speculative::repaired_commit_tokens;
use crate::frontend::speculative::verify_inputs_for_proposals;
use crate::frontend::util::ms_to_us;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::frontend::util::saturating_u32;
use crate::frontend::util::token_is_eog_with_runtime;
use crate::frontend::util::us_to_ms;
use crate::frontend::wire_messages::DecodeMessageArgs;
use crate::frontend::wire_messages::OpenAiPrefillChunk;
use crate::frontend::wire_messages::ReusableDecodeMessage;
use crate::frontend::wire_messages::ReusableDecodeMessageArgs;
use crate::frontend::wire_messages::VerifyWindowMessageArgs;
use crate::frontend::wire_messages::embedded_decode_message;
use crate::frontend::wire_messages::embedded_prefill_message;
use crate::frontend::wire_messages::embedded_verify_window_message;
use crate::frontend::wire_messages::generation_config_message;
use crate::telemetry::now_unix_nanos;
use lifecycle::{EmbeddedDecodeSummary, PipelinedCompositeWindow, decode_uses_context_sideband};
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_stage_zero_tokens(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        if request.config.downstream.is_none() {
            return self.generate_embedded_request_locally(request, on_token);
        }

        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let lane_pool = request
            .lane_pool
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("embedded stage 0 has no downstream lane pool"))?;
        let mut lane = lane_pool.checkout(request.ids)?;
        let mut direct_prediction_return_opened = false;
        if let Some(prediction_return) = request.prediction_return.as_ref() {
            match crate::binary_transport::direct_return::open_downstream_prediction_return_stream(
                request.config,
                request_id,
                session_id,
                request.wire_dtype,
            ) {
                Ok(stream) => {
                    prediction_return.attach_opened_stream(stream);
                    direct_prediction_return_opened = true;
                }
                Err(error) => {
                    eprintln!(
                        "direct prediction return upstream-opened sink unavailable: {error:#}"
                    );
                }
            }
        }
        let mut cache_stats = GenerationCacheStats::default();

        let result = (|| {
            let downstream = &mut lane.stream;
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_stage0_compute_ms = 0.0;
            let mut prefill_runtime_lock_wait_ms = 0.0;
            let mut prefill_runtime_lock_wait_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_hold_ms = 0.0;
            let mut prefill_runtime_lock_hold_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_acquires = 0usize;
            let mut prefill_runtime_sessions_before = None;
            let mut prefill_runtime_sessions_after = None;
            let mut prefill_forward_write_ms = 0.0;
            let mut prefill_output_activation_bytes = 0usize;
            let mut prefill_forward_activation_bytes = 0usize;
            let mut prefill_downstream_wait_ms = 0.0;
            let mut pending_prefill_replies = 0usize;
            let mut prefill_credit_wait_count = 0usize;
            let mut prefill_deferred_replies_drained = 0usize;
            let mut prefill_pending_replies_max = 0usize;
            let mut prefill_stage0_cache_hits = 0usize;
            let mut prefill_stage0_cache_misses = 0usize;
            let mut prefill_stage0_cache_errors = 0usize;
            let mut prefill_chain_cache_restored = false;
            let mut prefill_chain_restored_tokens = 0usize;
            let mut prefill_chain_cache_stats = StageReplyStats::default();
            let mut prefill_stage0_full_recorded = false;
            let mut fused_first_decode = None;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if let Some(seed) = lane_pool.prefill_transport_seed() {
                prefill_planner.observe(seed);
            }
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                if self.kv.is_some() {
                    cache_stats.status = "miss";
                }
                if request.max_tokens > 0 && request.draft.is_none() {
                    let current = *request
                        .prompt_token_ids
                        .last()
                        .expect("checked non-empty prompt");
                    if let Some(cached) = self.try_restore_embedded_split_exact_replay(
                        &request,
                        &session_key,
                        downstream,
                    )? {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = request
                            .prompt_token_ids
                            .len()
                            .saturating_add(cached.predicted_tokens.len().saturating_sub(1));
                        prefill_chain_cache_stats = cached.reply_stats;
                        cache_stats.cached_prompt_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.matched_prefix_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_exact_replay");
                        fused_first_decode = Some(cached);
                    } else if let Some(cached) = self
                        .try_restore_embedded_split_full_prompt_first_token(
                            &request,
                            &session_key,
                            downstream,
                        )?
                    {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = request.prompt_token_ids.len();
                        prefill_chain_cache_stats = cached.reply_stats;
                        cache_stats.cached_prompt_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.matched_prefix_tokens =
                            saturating_u32(request.prompt_token_ids.len());
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_full_prompt_first_token");
                        fused_first_decode = Some(cached);
                    } else if let Some(fused) = self.try_restore_embedded_split_prefill_and_decode(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                        current,
                        wire_sampling.clone(),
                    )? {
                        prefill_chain_cache_restored = true;
                        prefill_chain_restored_tokens = prefill_token_count;
                        prefill_chain_cache_stats = fused.reply_stats;
                        cache_stats.cached_prompt_tokens = saturating_u32(prefill_token_count);
                        cache_stats.matched_prefix_tokens = saturating_u32(prefill_token_count);
                        cache_stats.suffix_prefill_tokens = 0;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("chain_fused_exact_prefix");
                        fused_first_decode = Some(fused);
                    }
                }
                let split_prefill_restore = if prefill_chain_cache_restored {
                    None
                } else {
                    self.try_restore_embedded_split_prefill(
                        &request,
                        &session_key,
                        downstream,
                        prefill_tokens,
                    )?
                };
                if let Some(restore) = split_prefill_restore {
                    prefill_chain_restored_tokens = restore.restored_tokens;
                    prefill_chain_cache_restored =
                        prefill_chain_restored_tokens >= prefill_tokens.len();
                    prefill_chain_cache_stats = restore.stats;
                    cache_stats.cached_prompt_tokens =
                        saturating_u32(prefill_chain_restored_tokens);
                    cache_stats.matched_prefix_tokens =
                        saturating_u32(prefill_chain_restored_tokens);
                    cache_stats.suffix_prefill_tokens = saturating_u32(
                        prefill_tokens
                            .len()
                            .saturating_sub(prefill_chain_restored_tokens),
                    );
                    cache_stats.status = "hit";
                    cache_stats.hit_kind = Some("chain_prefix");
                }
                let mut pos_start = prefill_chain_restored_tokens.min(prefill_tokens.len());
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        drain_embedded_prefill_replies(
                            downstream,
                            &mut pending_prefill_replies,
                            &mut prefill_chain_cache_stats,
                        )?;
                        return Ok(());
                    }
                    let chunk_size =
                        prefill_planner.chunk_size_for(chunk_index, prefill_token_count);
                    let end = pos_start
                        .saturating_add(chunk_size)
                        .min(prefill_tokens.len());
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    let message = embedded_prefill_message(
                        request.wire_dtype,
                        OpenAiPrefillChunk {
                            seq_id: chunk_index,
                            pos_start,
                            prefill_token_count,
                            tokens: chunk,
                            request_id,
                            session_id,
                        },
                    )?;
                    let stage0_timer = PhaseTimer::start();
                    let pending_prefill_replies_before = pending_prefill_replies;
                    let mut output = self.restore_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                    )?;
                    if output.is_some() {
                        prefill_stage0_cache_hits += 1;
                    } else {
                        prefill_stage0_cache_misses += usize::from(pos_start == 0);
                    }
                    let output = match output.take() {
                        Some(output) => output,
                        None => {
                            let lock_timer = PhaseTimer::start();
                            let mut runtime = self
                                .runtime
                                .lock()
                                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                            let lock_wait_ms = lock_timer.elapsed_ms();
                            prefill_runtime_lock_wait_ms += lock_wait_ms;
                            prefill_runtime_lock_wait_max_ms =
                                prefill_runtime_lock_wait_max_ms.max(lock_wait_ms);
                            prefill_runtime_lock_acquires += 1;
                            let lock_hold_timer = PhaseTimer::start();
                            prefill_runtime_sessions_before
                                .get_or_insert_with(|| runtime.session_stats());
                            let output = run_binary_stage_message(
                                &mut runtime,
                                &session_key,
                                &message,
                                chunk,
                                None,
                                BinaryStageExecutionOptions::new(
                                    false,
                                    0,
                                    request.native_mtp_enabled,
                                )
                                .with_native_mtp_max_tokens(
                                    request.speculative.native_mtp.max_draft_tokens,
                                ),
                            )
                            .map_err(openai_backend_error)?
                            .2;
                            prefill_runtime_sessions_after = Some(runtime.session_stats());
                            let lock_hold_ms = lock_hold_timer.elapsed_ms();
                            prefill_runtime_lock_hold_ms += lock_hold_ms;
                            prefill_runtime_lock_hold_max_ms =
                                prefill_runtime_lock_hold_max_ms.max(lock_hold_ms);
                            output
                        }
                    };
                    if let Err(error) = self.record_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                        &output,
                    ) {
                        prefill_stage0_cache_errors += 1;
                        let mut attrs = self.openai_attrs(request.ids);
                        attrs.insert(
                            "skippy.kv.decision".to_string(),
                            json!("stage0_record_error"),
                        );
                        attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                        self.telemetry
                            .emit("stage.openai_kv_record_decision", attrs);
                    }
                    let chunk_stage0_compute_ms = stage0_timer.elapsed_ms();
                    prefill_stage0_compute_ms += chunk_stage0_compute_ms;
                    let forwarded = forwarded_stage_message(
                        request.config,
                        &message,
                        &output,
                        request.wire_dtype,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?;
                    prefill_output_activation_bytes =
                        prefill_output_activation_bytes.saturating_add(output.payload.len());
                    prefill_forward_activation_bytes =
                        prefill_forward_activation_bytes.saturating_add(forwarded.activation.len());
                    let write_timer = PhaseTimer::start();
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &forwarded,
                        request.wire_dtype,
                        request.downstream_wire_condition,
                    )
                    .map_err(openai_io_error)?;
                    let chunk_forward_write_ms = write_timer.elapsed_ms();
                    prefill_forward_write_ms += chunk_forward_write_ms;
                    let mut chunk_downstream_wait_ms = 0.0;
                    let mut chunk_deferred_replies_drained = 0usize;
                    let mut chunk_credit_wait_count = 0usize;
                    if request.prefill_reply_credit_limit == 0 {
                        let wait_timer = PhaseTimer::start();
                        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                        chunk_downstream_wait_ms = wait_timer.elapsed_ms();
                        if reply.kind != WireReplyKind::Ack {
                            return Err(OpenAiError::backend(format!(
                                "expected prefill ACK from downstream, got {:?}",
                                reply.kind
                            )));
                        }
                        prefill_chain_cache_stats.merge(reply.stats);
                    } else {
                        while pending_prefill_replies >= request.prefill_reply_credit_limit {
                            prefill_credit_wait_count = prefill_credit_wait_count.saturating_add(1);
                            chunk_credit_wait_count = chunk_credit_wait_count.saturating_add(1);
                            let drained = drain_one_embedded_prefill_reply(
                                downstream,
                                &mut pending_prefill_replies,
                                &mut prefill_chain_cache_stats,
                            )?;
                            prefill_deferred_replies_drained = prefill_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_deferred_replies_drained = chunk_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_downstream_wait_ms += drained.downstream_wait_ms;
                        }
                        pending_prefill_replies = pending_prefill_replies.saturating_add(1);
                        prefill_pending_replies_max =
                            prefill_pending_replies_max.max(pending_prefill_replies);
                    }
                    prefill_downstream_wait_ms += chunk_downstream_wait_ms;
                    prefill_planner.observe(PrefillChunkObservation {
                        compute_ms: chunk_stage0_compute_ms,
                        forward_write_ms: chunk_forward_write_ms,
                        downstream_wait_ms: chunk_downstream_wait_ms,
                    });
                    let mut chunk_attrs = self.openai_attrs(request.ids);
                    chunk_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("PrefillEmbd"));
                    chunk_attrs.insert("llama_stage.seq_id".to_string(), json!(chunk_index));
                    chunk_attrs.insert("llama_stage.pos_start".to_string(), json!(pos_start));
                    chunk_attrs.insert("llama_stage.token_count".to_string(), json!(chunk.len()));
                    chunk_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(chunk_stage0_compute_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(chunk_forward_write_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(chunk_downstream_wait_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.activation.len()),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_limit".to_string(),
                        json!(request.prefill_reply_credit_limit),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_before".to_string(),
                        json!(pending_prefill_replies_before),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_after".to_string(),
                        json!(pending_prefill_replies),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_wait_count".to_string(),
                        json!(chunk_credit_wait_count),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_deferred_replies_drained".to_string(),
                        json!(chunk_deferred_replies_drained),
                    );
                    self.telemetry.emit_debug_span(
                        "stage.openai_prefill_chunk",
                        chunk_attrs,
                        stage0_timer.start_unix_nanos,
                        now_unix_nanos() as u64,
                    );
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
                let drained = drain_embedded_prefill_replies(
                    downstream,
                    &mut pending_prefill_replies,
                    &mut prefill_chain_cache_stats,
                )?;
                prefill_deferred_replies_drained =
                    prefill_deferred_replies_drained.saturating_add(drained.drained_replies);
                prefill_downstream_wait_ms += drained.downstream_wait_ms;
                lane_pool.observe_prefill_transport(
                    &prefill_chain_cache_stats,
                    prefill_stage0_compute_ms,
                    prefill_chunks,
                );
                if !prefill_chain_cache_restored {
                    prefill_stage0_full_recorded = self.record_embedded_stage0_full_prefill(
                        &session_key,
                        request.ids,
                        prefill_tokens,
                    )?;
                }
            }
            let mut prefill_attrs = self.openai_attrs(request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            prefill_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(prefill_stage0_compute_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(prefill_runtime_lock_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(prefill_runtime_lock_wait_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(prefill_runtime_lock_hold_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(prefill_runtime_lock_hold_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(prefill_runtime_lock_acquires),
            );
            if let Some(stats) = prefill_runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = prefill_runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            prefill_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(prefill_forward_write_ms),
            );
            prefill_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(prefill_output_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(prefill_forward_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(prefill_downstream_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_write_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_write_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_wait_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_wait_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_total_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_total_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_stage_index".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_stage_index),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_observation_count".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_observation_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_limit".to_string(),
                json!(request.prefill_reply_credit_limit),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_max".to_string(),
                json!(prefill_pending_replies_max),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_after".to_string(),
                json!(pending_prefill_replies),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_wait_count".to_string(),
                json!(prefill_credit_wait_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_deferred_replies_drained".to_string(),
                json!(prefill_deferred_replies_drained),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_hits".to_string(),
                json!(prefill_stage0_cache_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_misses".to_string(),
                json!(prefill_stage0_cache_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_errors".to_string(),
                json!(prefill_stage0_cache_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_full_recorded".to_string(),
                json!(prefill_stage0_full_recorded),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_restored".to_string(),
                json!(prefill_chain_cache_restored),
            );
            prefill_attrs.insert(
                "skippy.kv.matched_prefix_tokens".to_string(),
                json!(prefill_chain_restored_tokens),
            );
            prefill_attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(prefill_token_count.saturating_sub(prefill_chain_restored_tokens)),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hits".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_misses".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_errors".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hit_stage_mask".to_string(),
                json!(prefill_chain_cache_stats.kv_hit_stage_mask),
            );
            super::prefix_cache::insert_chain_prefix_cache_savings_attrs(
                &mut prefill_attrs,
                super::prefix_cache::chain_prefix_cache_savings(
                    &prefill_chain_cache_stats,
                    prefill_chain_restored_tokens,
                    request.wire_dtype,
                    request.activation_width,
                ),
            );
            cache_stats.prompt_ms = prefill_timer.elapsed_ms();
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            let message = generation_config_message(
                request.wire_dtype,
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )?;
            write_stage_message_conditioned(
                &mut *downstream,
                &message,
                request.wire_dtype,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
            if reply.kind != WireReplyKind::Ack {
                return Err(OpenAiError::backend(format!(
                    "expected generation config ACK from downstream, got {:?}",
                    reply.kind
                )));
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_wait_max_ms = 0.0_f64;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_hold_max_ms = 0.0_f64;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_batch_size_max = 1usize;
            let mut decode_batch_wait_ms = 0.0;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_forward_activation_encode_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_downstream_wait_ms = 0.0;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut context_tokens = request.prompt_token_ids.to_vec();
            let mut exact_replay_tokens = Vec::new();
            let mut decode_message = ReusableDecodeMessage::new(
                request.wire_dtype,
                ReusableDecodeMessageArgs {
                    request_id,
                    session_id,
                    prompt_token_count: request.prompt_token_ids.len(),
                    base_pos_start: prefill_token_count,
                    sampling: wire_sampling.clone(),
                    sideband_capacity: skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
                },
            )?;
            let mut fused_reached_stop = false;
            let mut native_mtp = NativeMtpVerifier::default();
            let native_mtp_options = NativeMtpDecodeOptions::from_config(request.speculative);
            let mut native_mtp_counters = NativeMtpDecodeCounters::default();
            let mut native_mtp_reject_cooldown_remaining = 0usize;
            let mut native_mtp_suppress_cooldown_drafts_remaining = 0usize;
            let mut ngram_sidecar_controller = NgramSidecarController::new(
                native_mtp_options.ngram_initial_extension_tokens,
                native_mtp_options.ngram_max_proposal_tokens,
            );
            if let Some(mut fused) = fused_first_decode.take() {
                current = fused.predicted;
                let mut fused_native_mtp_draft = fused.native_mtp_draft.take();
                decode_stage0_compute_ms += fused.execution.stage0_compute_ms;
                decode_runtime_lock_wait_ms += fused.execution.runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(fused.execution.runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += fused.execution.runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(fused.execution.runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_forward_activation_encode_ms += fused.execution.activation_encode_ms;
                decode_output_activation_bytes = decode_output_activation_bytes
                    .saturating_add(fused.execution.output_activation_bytes);
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(fused.execution.forward_activation_bytes);
                decode_forward_write_ms += fused.execution.forward_write_ms;
                decode_downstream_wait_ms += fused.execution.downstream_wait_ms;
                for (index, token) in fused.predicted_tokens.iter().copied().enumerate() {
                    if decoded_tokens >= request.max_tokens as usize {
                        break;
                    }
                    current = token;
                    exact_replay_tokens.push(current);
                    context_tokens.push(current);
                    let native_mtp_decision = native_mtp.observe_target_token(
                        current,
                        if index == 0 {
                            ms_to_us(fused.execution.downstream_wait_ms)
                        } else {
                            0
                        },
                        if index == 0 {
                            fused_native_mtp_draft.take()
                        } else {
                            None
                        },
                        NativeMtpDraftOrigin::InitialSerial,
                    );
                    decoded_tokens += 1;
                    if self.telemetry.is_debug_enabled() {
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs.insert("llama_stage.decode_step".to_string(), json!(index));
                        token_attrs.insert(
                            "llama_stage.decode_token_phase".to_string(),
                            json!(fused.token_phase),
                        );
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!(fused.message_kind),
                        );
                        token_attrs.insert(
                            "llama_stage.elapsed_ms".to_string(),
                            json!(if index == 0 { fused.elapsed_ms } else { 0.0 }),
                        );
                        token_attrs.insert(
                            "llama_stage.cached_replay_token_index".to_string(),
                            json!(index),
                        );
                        token_attrs.insert(
                            "llama_stage.cached_replay_token_count".to_string(),
                            json!(fused.predicted_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.stage0_compute_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.runtime_lock_wait_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.runtime_lock_hold_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(if index == 0 {
                                fused.execution.output_activation_bytes
                            } else {
                                0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(if index == 0 {
                                fused.execution.forward_activation_bytes
                            } else {
                                0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.activation_encode_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.forward_write_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(if index == 0 {
                                fused.execution.downstream_wait_ms
                            } else {
                                0.0
                            }),
                        );
                        token_attrs
                            .insert("llama_stage.predicted_token".to_string(), json!(current));
                        token_attrs.insert(
                            "llama_stage.native_mtp.verification".to_string(),
                            json!(native_mtp_decision.label()),
                        );
                        self.telemetry
                            .emit_debug("stage.openai_decode_token", token_attrs);
                    }
                    if on_token(current)? == TokenControl::Stop {
                        fused_reached_stop = true;
                        break;
                    }
                }
            }
            let mut cached_ngram_proposer = CachedNgramProposer::from_config(request.speculative)?;
            let max_speculative_window = request.speculative_window.max(1);
            let mut adaptive_window = if request.adaptive_speculative_window {
                max_speculative_window.min(4)
            } else {
                max_speculative_window
            };
            let mut speculative_stats = OpenAiSpeculativeStats {
                adaptive_window_start: adaptive_window,
                adaptive_window_final: adaptive_window,
                adaptive_window_max: max_speculative_window,
                adaptive_window_min: if request.draft.is_some() || request.ngram_max > 0 {
                    adaptive_window
                } else {
                    0
                },
                adaptive_window_max_seen: adaptive_window,
                adaptive_window_enabled: request.adaptive_speculative_window,
                ..OpenAiSpeculativeStats::default()
            };
            let mut draft_guard = match request.draft.as_ref() {
                Some(draft) if request.speculative_window > 0 => {
                    let draft_reset_timer = PhaseTimer::start();
                    let mut draft = draft
                        .lock()
                        .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?;
                    draft
                        .reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.draft_model_path".to_string(),
                        json!(draft.path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(draft.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_draft_reset", draft_reset_timer, attrs);
                    Some(draft)
                }
                _ => None,
            };
            let mut verify_window_scheduler = VerifyWindowScheduler::new(
                VerifyWindowPipelineConfig::new(request.speculative.verify_window.pipeline_depth),
            );
            let composite_sidecar_enabled =
                native_mtp_options.ngram_hybrid && draft_guard.is_none();
            let native_mtp_verify_windows_enabled =
                (request.native_mtp_enabled || composite_sidecar_enabled) && draft_guard.is_none();
            let pipelined_decode_enabled =
                composite_sidecar_enabled && verify_window_scheduler.depth() > 1;
            if native_mtp_verify_windows_enabled && !direct_prediction_return_opened {
                return Err(OpenAiError::backend(
                    "native MTP verify windows require direct prediction return",
                ));
            }
            if native_mtp_verify_windows_enabled {
                verify_window_scheduler.mark_direct_prediction_return();
            }
            let mut pipelined_windows = VecDeque::new();
            let mut pipelined = None;
            let mut pipelined_current = current;
            let mut composite_proposal_buffer = None;
            let mut adaptive_verify_window = AdaptiveVerifyWindow::new(native_mtp_options);
            for decode_step in decoded_tokens as u32..request.max_tokens {
                if fused_reached_stop {
                    break;
                }
                if decoded_tokens >= request.max_tokens as usize {
                    break;
                }
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let token_timer = PhaseTimer::start();
                let native_mtp_remaining =
                    (request.max_tokens as usize).saturating_sub(decoded_tokens);
                let mut pipeline_seed = None;
                if pipelined_decode_enabled
                    && pipelined.is_none()
                    && pipelined_windows.is_empty()
                    && native_mtp_reject_cooldown_remaining == 0
                    && verify_window_scheduler.has_profitable_pipeline_width()
                {
                    let pending = request
                        .native_mtp_enabled
                        .then(|| native_mtp.take_pending_draft())
                        .flatten();
                    let native_mtp_origin = pending.as_ref().map(|draft| draft.origin);
                    let native_mtp_tokens = pending
                        .as_ref()
                        .map(|draft| {
                            draft
                                .tokens
                                .iter()
                                .copied()
                                .take(native_mtp_options.max_draft_tokens)
                                .take(native_mtp_remaining.saturating_sub(1))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let native_mtp_tokens =
                        if native_mtp_tokens.len() >= native_mtp_options.min_draft_tokens {
                            native_mtp_tokens.as_slice()
                        } else {
                            &[]
                        };
                    let proposal = CompositeProposalProvider::from_options(native_mtp_options)
                        .propose_with_ngram_extension(
                            native_mtp_tokens,
                            &context_tokens,
                            native_mtp_remaining,
                            ngram_sidecar_controller.extension_limit(
                                native_mtp_tokens,
                                native_mtp_remaining.saturating_sub(native_mtp_tokens.len()),
                            ),
                            cached_ngram_proposer.as_mut(),
                        )?;
                    match proposal.parallel_verify_width(
                        adaptive_verify_window.width(proposal.tokens().len()),
                        verify_window_scheduler.depth(),
                    ) {
                        Some(parallel_verify_width)
                            if verify_window_scheduler
                                .permit_pipeline_width(parallel_verify_width) =>
                        {
                            pipeline_seed = Some((
                                proposal,
                                if native_mtp_tokens.is_empty() {
                                    None
                                } else {
                                    native_mtp_origin
                                },
                                parallel_verify_width,
                            ));
                        }
                        _ => {
                            if let Some(pending) = pending {
                                native_mtp.restore_pending_draft(pending);
                            }
                        }
                    }
                }
                if native_mtp_verify_windows_enabled
                    && (!pipelined_decode_enabled
                        || (pipelined.is_none() && pipeline_seed.is_none()))
                    && native_mtp_reject_cooldown_remaining == 0
                    && native_mtp_remaining >= 2
                {
                    let pending_native_mtp_draft = (request.native_mtp_enabled
                        && composite_proposal_buffer.is_none())
                    .then(|| native_mtp.take_pending_draft())
                    .flatten();
                    match self.execute_native_mtp_verify_window(
                        &request,
                        downstream,
                        &session_key,
                        request_id,
                        session_id,
                        prefill_token_count,
                        &wire_sampling,
                        &native_mtp_options,
                        &mut verify_window_scheduler,
                        pending_native_mtp_draft,
                        &mut composite_proposal_buffer,
                        &mut cached_ngram_proposer,
                        &mut adaptive_verify_window,
                        &mut current,
                        decode_step,
                        &mut decoded_tokens,
                        &mut context_tokens,
                        &mut exact_replay_tokens,
                        &mut native_mtp,
                        &mut native_mtp_counters,
                        &mut native_mtp_reject_cooldown_remaining,
                        &mut native_mtp_suppress_cooldown_drafts_remaining,
                        &mut ngram_sidecar_controller,
                        &mut decode_stage0_compute_ms,
                        &mut decode_runtime_lock_wait_ms,
                        &mut decode_runtime_lock_wait_max_ms,
                        &mut decode_runtime_lock_hold_ms,
                        &mut decode_runtime_lock_hold_max_ms,
                        &mut decode_runtime_lock_acquires,
                        &mut decode_forward_activation_encode_ms,
                        &mut decode_output_activation_bytes,
                        &mut decode_forward_activation_bytes,
                        &mut decode_forward_write_ms,
                        &mut decode_downstream_wait_ms,
                        &mut on_token,
                    )? {
                        NativeMtpVerifyWindowControl::ReachedStop => break,
                        NativeMtpVerifyWindowControl::Continue => continue,
                        NativeMtpVerifyWindowControl::NoProposal => {}
                    }
                }
                if pipelined_decode_enabled {
                    if let Some((proposal, origin, parallel_verify_width)) = pipeline_seed {
                        pipelined_current = current;
                        pipelined = Some(CompositeProposalPipeline::new(
                            proposal,
                            origin,
                            parallel_verify_width,
                        ));
                    }
                    if let Some(pipeline) = pipelined.as_mut() {
                        while verify_window_scheduler.has_capacity()
                            && decoded_tokens
                                + pipelined_windows
                                    .iter()
                                    .map(|window: &PipelinedCompositeWindow| {
                                        window.input_tokens.len()
                                    })
                                    .sum::<usize>()
                                < request.max_tokens as usize
                        {
                            let Some(planned) = pipeline.next_window(
                                adaptive_verify_window.width(pipeline.candidate_len()),
                            ) else {
                                break;
                            };
                            let proposal_tokens = planned.proposal_tokens().to_vec();
                            let expected_free_target = planned.expected_free_target();
                            let native_mtp_token_count = planned.native_mtp_token_count();
                            let offset = pipelined_windows
                                .iter()
                                .map(|window: &PipelinedCompositeWindow| window.input_tokens.len())
                                .sum::<usize>();
                            let window = verify_window_scheduler.open(
                                prefill_token_count + decoded_tokens + offset,
                                decoded_tokens + offset,
                            )?;
                            let mut input_tokens = Vec::with_capacity(proposal_tokens.len() + 1);
                            input_tokens.push(pipelined_current);
                            input_tokens.extend_from_slice(&proposal_tokens);
                            let message = embedded_verify_window_message(
                                request.wire_dtype,
                                VerifyWindowMessageArgs {
                                    window_id: window.id,
                                    request_id,
                                    session_id,
                                    prompt_token_count: request.prompt_token_ids.len(),
                                    pos_start: window.base_position,
                                    decode_step: window.decode_step,
                                    tokens: &input_tokens,
                                    sampling: wire_sampling.clone(),
                                    checkpoint: false,
                                },
                            )?;
                            let dispatched = self.dispatch_embedded_stage_message(
                                &request,
                                downstream,
                                &session_key,
                                &message,
                                &input_tokens,
                            )?;
                            pipelined_windows.push_back(PipelinedCompositeWindow {
                                window,
                                input_tokens,
                                proposal_tokens,
                                expected_free_target,
                                native_mtp_token_count,
                                dispatched,
                            });
                            let Some(next_current) = expected_free_target else {
                                break;
                            };
                            pipelined_current = next_current;
                        }
                        if let Some(window) = pipelined_windows.pop_front() {
                            let verify = self.complete_dispatched_stage_message_direct(
                                &request,
                                downstream,
                                window.dispatched,
                                WireReplyKind::PredictedTokens,
                            )?;
                            let completed = verify_window_scheduler
                                .complete_next(verify.reply.window.window_id)?;
                            if completed != window.window {
                                return Err(OpenAiError::backend(
                                    "verify window scheduler lost FIFO state",
                                ));
                            }
                            let native_mtp_verify_decision = classify_native_mtp_verify_window(
                                &window.proposal_tokens,
                                &verify.reply.predicted_tokens,
                                decoded_tokens,
                                request.max_tokens as usize,
                                |token| token_is_eog_with_runtime(&self.runtime, token),
                            )?;
                            let fully_accepted_window = !native_mtp_verify_decision.rejected
                                && native_mtp_verify_decision.accepted_proposal_tokens
                                    == window.proposal_tokens.len();
                            let free_target_matches =
                                window.expected_free_target.is_none_or(|expected| {
                                    verify
                                        .reply
                                        .predicted_tokens
                                        .get(window.proposal_tokens.len())
                                        == Some(&expected)
                                });
                            let pipeline_continues = fully_accepted_window && free_target_matches;
                            if window.expected_free_target.is_some() {
                                verify_window_scheduler.observe_pipeline_profile(
                                    window.proposal_tokens.len(),
                                    pipeline_continues,
                                    verify.stats.stage0_compute_ms,
                                    verify.stats.downstream_wait_ms,
                                );
                            }
                            let accepted_candidate_tokens = native_mtp_verify_decision
                                .accepted_proposal_tokens
                                + usize::from(
                                    fully_accepted_window
                                        && window.expected_free_target.is_some()
                                        && free_target_matches,
                                );
                            if window.native_mtp_token_count > 0 {
                                let pipeline = pipelined.as_ref().expect("pipeline retained");
                                let span = native_mtp.observe_taken_draft_span(
                                    &window.proposal_tokens[..window.native_mtp_token_count],
                                    &verify.reply.predicted_tokens,
                                    ms_to_us(verify.elapsed_ms),
                                );
                                for index in 0..span.accepted_count + usize::from(span.rejected) {
                                    native_mtp_counters.observe_verify_window_verification(
                                        pipeline.origin().expect("native MTP candidate has origin"),
                                        index < span.accepted_count,
                                    );
                                }
                            }
                            speculative_stats.windows += 1;
                            speculative_stats.draft_tokens += window.proposal_tokens.len();
                            speculative_stats.primary_verify_requests += 1;
                            speculative_stats.primary_verify_tokens += window.input_tokens.len();
                            speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                            speculative_stats.primary_verify_stage0_compute_ms +=
                                verify.stats.stage0_compute_ms;
                            speculative_stats.primary_verify_runtime_lock_wait_ms +=
                                verify.stats.runtime_lock_wait_ms;
                            speculative_stats.primary_verify_runtime_lock_hold_ms +=
                                verify.stats.runtime_lock_hold_ms;
                            speculative_stats.primary_verify_activation_encode_ms +=
                                verify.stats.activation_encode_ms;
                            speculative_stats.primary_verify_forward_write_ms +=
                                verify.stats.forward_write_ms;
                            speculative_stats.primary_verify_downstream_wait_ms +=
                                verify.stats.downstream_wait_ms;
                            speculative_stats.primary_verify_output_activation_bytes =
                                speculative_stats
                                    .primary_verify_output_activation_bytes
                                    .saturating_add(verify.stats.output_activation_bytes);
                            speculative_stats.primary_verify_forward_activation_bytes =
                                speculative_stats
                                    .primary_verify_forward_activation_bytes
                                    .saturating_add(verify.stats.forward_activation_bytes);
                            decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                            decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                            decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                .max(verify.stats.runtime_lock_wait_ms);
                            decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                            decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                .max(verify.stats.runtime_lock_hold_ms);
                            decode_runtime_lock_acquires += 1;
                            decode_forward_activation_encode_ms +=
                                verify.stats.activation_encode_ms;
                            decode_output_activation_bytes = decode_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                            decode_forward_activation_bytes = decode_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                            decode_forward_write_ms += verify.stats.forward_write_ms;
                            decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                            if fully_accepted_window {
                                speculative_stats.accepted_tokens += accepted_candidate_tokens;
                                speculative_stats.full_accept_windows += 1;
                                let pipeline = pipelined.as_mut().expect("pipeline retained");
                                pipeline.set_next_draft(
                                    request.native_mtp_enabled,
                                    verify
                                        .reply
                                        .native_mtp_draft
                                        .clone()
                                        .map(NativeMtpDraft::from_stage_draft),
                                );
                            } else {
                                speculative_stats.rejected_tokens += 1;
                                speculative_stats.rejected_windows += 1;
                                speculative_stats.early_reject_windows += 1;
                                speculative_stats.first_reject_position_sum += 1;
                            }
                            pipelined
                                .as_mut()
                                .expect("pipeline retained")
                                .observe_accepted(accepted_candidate_tokens);
                            let mut reached_stop = false;
                            for token in verify
                                .reply
                                .predicted_tokens
                                .iter()
                                .copied()
                                .take(native_mtp_verify_decision.commit_count)
                            {
                                current = token;
                                decoded_tokens += 1;
                                exact_replay_tokens.push(current);
                                context_tokens.push(current);
                                if on_token(current)? == TokenControl::Stop
                                    || decoded_tokens >= request.max_tokens as usize
                                {
                                    reached_stop = true;
                                    break;
                                }
                            }
                            let previous_verify_width = adaptive_verify_window.current_tokens();
                            adaptive_verify_window.observe(pipeline_continues);
                            native_mtp_counters.observe_adaptive_verify_window(
                                window.proposal_tokens.len(),
                                previous_verify_width,
                                adaptive_verify_window.current_tokens(),
                            );
                            if !pipeline_continues || reached_stop {
                                let stale_count = pipelined_windows.len();
                                let stale_drain_timer = PhaseTimer::start();
                                while let Some(stale) = pipelined_windows.pop_front() {
                                    let stale_reply = self
                                        .complete_dispatched_stage_message_direct(
                                            &request,
                                            downstream,
                                            stale.dispatched,
                                            WireReplyKind::PredictedTokens,
                                        )?;
                                    verify_window_scheduler
                                        .complete_next(stale_reply.reply.window.window_id)?;
                                }
                                verify_window_scheduler.record_stale_discarded(
                                    stale_count,
                                    stale_drain_timer.elapsed_ms(),
                                );
                                let pipeline = pipelined.take().expect("pipeline retained");
                                if ngram_sidecar_controller.observe_tail_outcome(
                                    pipeline.proposal(),
                                    pipeline.accepted_tokens(),
                                    native_mtp_options.ngram_tail_backoff_proposals,
                                ) {
                                    native_mtp_counters.observe_ngram_tail_rejection();
                                }
                                native_mtp_counters.observe_hybrid_proposal(
                                    pipeline.proposal(),
                                    pipeline.accepted_tokens(),
                                );
                                native_mtp.clear_pending_draft();
                                if native_mtp_verify_decision.rejected
                                    && pipeline
                                        .proposal()
                                        .native_mtp_prefix_rejected(pipeline.accepted_tokens())
                                    && native_mtp_options.reject_cooldown_tokens > 0
                                {
                                    native_mtp_reject_cooldown_remaining =
                                        native_mtp_options.reject_cooldown_tokens;
                                    native_mtp_suppress_cooldown_drafts_remaining =
                                        native_mtp_options.suppress_cooldown_draft_limit;
                                }
                                if native_mtp_verify_decision.rejected || stale_count > 0 {
                                    let trim = self.trim_embedded_stage_session(
                                        &request,
                                        downstream,
                                        &session_key,
                                        request_id,
                                        session_id,
                                        prefill_token_count + decoded_tokens,
                                    )?;
                                    speculative_stats.recovery_ms += trim.elapsed_ms;
                                }
                                pipelined_current = current;
                                if reached_stop {
                                    break;
                                }
                            } else if pipelined_windows.is_empty()
                                && pipelined
                                    .as_ref()
                                    .is_some_and(|pipeline| !pipeline.has_remaining_candidates())
                            {
                                let mut pipeline = pipelined.take().expect("pipeline retained");
                                let next_draft_available = pipeline.next_draft().is_some();
                                ngram_sidecar_controller.observe_tail_outcome(
                                    pipeline.proposal(),
                                    pipeline.accepted_tokens(),
                                    native_mtp_options.ngram_tail_backoff_proposals,
                                );
                                native_mtp_counters.observe_hybrid_proposal(
                                    pipeline.proposal(),
                                    pipeline.accepted_tokens(),
                                );
                                native_mtp_counters.observe_verify_next_draft(
                                    next_draft_available,
                                    next_draft_available,
                                );
                                if let Some(next_draft) = pipeline.take_next_draft() {
                                    native_mtp.observe_next_draft(
                                        Some(next_draft),
                                        NativeMtpDraftOrigin::VerifyNext,
                                    );
                                }
                            }
                            if self.telemetry.is_debug_enabled() {
                                let mut attrs = self.openai_attrs(request.ids);
                                attrs.insert(
                                    "llama_stage.message_kind".to_string(),
                                    json!("VerifyWindow"),
                                );
                                attrs.insert(
                                    "llama_stage.spec.proposal_source".to_string(),
                                    json!("composite_mtp_ngram"),
                                );
                                attrs.insert(
                                    "llama_stage.verify_window_id".to_string(),
                                    json!(window.window.id),
                                );
                                attrs.insert(
                                    "llama_stage.verify_window.accepted".to_string(),
                                    json!(pipeline_continues),
                                );
                                attrs.insert(
                                    "llama_stage.verify_window.in_flight_after".to_string(),
                                    json!(verify_window_scheduler.in_flight_len()),
                                );
                                attrs.insert(
                                    "llama_stage.verify_window.stale_discarded".to_string(),
                                    json!(verify_window_scheduler.stale_discard_count()),
                                );
                                self.emit_openai_phase(
                                    "stage.openai_decode_verify_window",
                                    token_timer,
                                    attrs,
                                );
                            }
                            continue;
                        }
                    }
                }
                if draft_guard.is_some() || request.ngram_max > 0 {
                    let remaining = (request.max_tokens as usize).saturating_sub(decoded_tokens);
                    if remaining == 0 {
                        break;
                    }
                    let mut proposal_source = "none";
                    let proposal_limit = remaining.min(adaptive_window);
                    let propose_timer = PhaseTimer::start();
                    let mut draft_tokens = Vec::new();
                    if let (true, Some(draft)) =
                        (draft_tokens.is_empty(), draft_guard.as_deref_mut())
                    {
                        let proposal_limit = proposal_limit.min(draft.window);
                        draft_tokens = draft
                            .propose(current, proposal_limit)
                            .map_err(openai_backend_error)?;
                        if !draft_tokens.is_empty() {
                            proposal_source = "draft-model";
                        }
                    }
                    if let (true, Some(cache)) =
                        (draft_tokens.is_empty(), cached_ngram_proposer.as_mut())
                    {
                        draft_tokens = cache.propose(
                            &context_tokens,
                            &[],
                            proposal_limit.min(request.ngram_max),
                        )?;
                        if !draft_tokens.is_empty() {
                            proposal_source = "ngram-cache";
                        }
                    }
                    if draft_tokens.is_empty() && request.ngram_max > 0 {
                        draft_tokens = propose_ngram_tokens(
                            &context_tokens,
                            request.ngram_min,
                            proposal_limit.min(request.ngram_max),
                        )?;
                        if !draft_tokens.is_empty() {
                            proposal_source = "ngram";
                        }
                    }
                    let draft_propose_ms = propose_timer.elapsed_ms();
                    speculative_stats.draft_propose_ms += draft_propose_ms;
                    if !draft_tokens.is_empty() {
                        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
                        let message = embedded_verify_window_message(
                            request.wire_dtype,
                            VerifyWindowMessageArgs {
                                window_id: i32::try_from(decoded_tokens)
                                    .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?,
                                request_id,
                                session_id,
                                prompt_token_count: request.prompt_token_ids.len(),
                                pos_start: prefill_token_count + decoded_tokens,
                                decode_step: decoded_tokens,
                                tokens: &verify_inputs,
                                sampling: wire_sampling.clone(),
                                checkpoint: true,
                            },
                        )?;
                        let verify = self.execute_embedded_stage_message(
                            &request,
                            downstream,
                            &session_key,
                            &message,
                            &verify_inputs,
                            WireReplyKind::PredictedTokens,
                        )?;
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += draft_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += verify_inputs.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        speculative_stats.checkpoint_ms +=
                            us_to_ms(verify.reply.stats.checkpoint_total_us);
                        let decision = classify_verify_window(
                            &draft_tokens,
                            &verify.reply.predicted_tokens,
                            decoded_tokens,
                            request.max_tokens as usize,
                            |token| token_is_eog_with_runtime(&self.runtime, token),
                        )?;
                        speculative_stats.observe_verify_decision(
                            decision,
                            &mut adaptive_window,
                            request.adaptive_speculative_window,
                            max_speculative_window,
                        );
                        let mut commit_tokens =
                            verify.reply.predicted_tokens[..decision.commit_count].to_vec();
                        if decision.requires_repair() {
                            speculative_stats.recovery_restores += 1;
                            let restore = self.restore_embedded_stage_session(
                                &request,
                                downstream,
                                &session_key,
                                request_id,
                                session_id,
                            )?;
                            speculative_stats.recovery_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_ms += restore.elapsed_ms;
                            speculative_stats.recovery_restore_local_ms += restore.local_ms;
                            speculative_stats.recovery_restore_downstream_write_ms +=
                                restore.downstream_write_ms;
                            speculative_stats.recovery_restore_downstream_wait_ms +=
                                restore.downstream_wait_ms;
                            let repair_input_count = decision
                                .repair_input_count
                                .ok_or_else(|| OpenAiError::backend("missing repair count"))?;
                            if repair_input_count == 1 {
                                let repair_message = embedded_decode_message(
                                    request.wire_dtype,
                                    DecodeMessageArgs {
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        current,
                                        sampling: wire_sampling.clone(),
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    &[current],
                                    WireReplyKind::PredictedToken,
                                )?;
                                commit_tokens = vec![repair.reply.predicted];
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_decode_repairs += 1;
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_decode_elapsed_ms += repair.elapsed_ms;
                            } else {
                                let repair_inputs = &verify_inputs[..repair_input_count];
                                let repair_message = embedded_verify_window_message(
                                    request.wire_dtype,
                                    VerifyWindowMessageArgs {
                                        window_id: i32::try_from(decoded_tokens).map_err(|_| {
                                            OpenAiError::backend("decode step exceeds i32")
                                        })?,
                                        request_id,
                                        session_id,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                        pos_start: prefill_token_count + decoded_tokens,
                                        decode_step: decoded_tokens,
                                        tokens: repair_inputs,
                                        sampling: wire_sampling.clone(),
                                        checkpoint: false,
                                    },
                                )?;
                                let repair = self.execute_embedded_stage_message(
                                    &request,
                                    downstream,
                                    &session_key,
                                    &repair_message,
                                    repair_inputs,
                                    WireReplyKind::PredictedTokens,
                                )?;
                                commit_tokens = repaired_commit_tokens(
                                    &draft_tokens,
                                    decision.accepted_before_reject,
                                    repair_input_count,
                                    &repair.reply.predicted_tokens,
                                )?;
                                decode_stage0_compute_ms += repair.stats.stage0_compute_ms;
                                decode_runtime_lock_wait_ms += repair.stats.runtime_lock_wait_ms;
                                decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                    .max(repair.stats.runtime_lock_wait_ms);
                                decode_runtime_lock_hold_ms += repair.stats.runtime_lock_hold_ms;
                                decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                    .max(repair.stats.runtime_lock_hold_ms);
                                decode_runtime_lock_acquires += 1;
                                decode_forward_activation_encode_ms +=
                                    repair.stats.activation_encode_ms;
                                decode_output_activation_bytes = decode_output_activation_bytes
                                    .saturating_add(repair.stats.output_activation_bytes);
                                decode_forward_activation_bytes = decode_forward_activation_bytes
                                    .saturating_add(repair.stats.forward_activation_bytes);
                                decode_forward_write_ms += repair.stats.forward_write_ms;
                                decode_downstream_wait_ms += repair.stats.downstream_wait_ms;
                                speculative_stats.recovery_reverify_tokens += repair_inputs.len();
                                speculative_stats.recovery_ms += repair.elapsed_ms;
                                speculative_stats.recovery_reverify_elapsed_ms += repair.elapsed_ms;
                            }
                        }

                        let mut reached_stop = false;
                        for token in commit_tokens {
                            current = token;
                            decoded_tokens += 1;
                            context_tokens.push(current);
                            if on_token(current)? == TokenControl::Stop {
                                reached_stop = true;
                            }
                            if reached_stop || decoded_tokens >= request.max_tokens as usize {
                                break;
                            }
                        }
                        speculative_stats.adaptive_window_final = adaptive_window;
                        if proposal_source == "draft-model" && (decision.rejected() || reached_stop)
                        {
                            let draft_reset_timer = PhaseTimer::start();
                            if let Some(draft) = draft_guard.as_deref_mut() {
                                draft
                                    .reset_to_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                            }
                        }
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!("VerifyWindow"),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.windows".to_string(),
                            json!(speculative_stats.windows),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposed".to_string(),
                            json!(draft_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.accepted".to_string(),
                            json!(decision.accepted_before_reject),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.rejected".to_string(),
                            json!(decision.rejected()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.draft_propose_ms".to_string(),
                            json!(draft_propose_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_source".to_string(),
                            json!(proposal_source),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_limit".to_string(),
                            json!(proposal_limit),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_decode_verify_window",
                            token_timer,
                            token_attrs,
                        );
                        if reached_stop {
                            break;
                        }
                        continue;
                    }
                }
                let uses_context_sideband = decode_uses_context_sideband(
                    &context_tokens,
                    current,
                    skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
                );
                let records_replay_checkpoint = uses_context_sideband && context_tokens.len() > 1;
                let records_full_prompt_checkpoint = decode_step == 0
                    && uses_context_sideband
                    && context_tokens.len() == request.prompt_token_ids.len();
                let decode_step_index = usize::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds usize"))?;
                let message = if uses_context_sideband {
                    decode_message.update_with_tokens(
                        decode_step_index,
                        current,
                        &context_tokens,
                    )?
                } else {
                    decode_message.update(decode_step_index, current)?
                };
                let stage0_timer = PhaseTimer::start();
                let batch_outcome = self
                    .decode_frame_batcher
                    .decode(
                        &session_key,
                        current,
                        request.sampling.enabled.then_some(request.sampling),
                        None,
                    )
                    .map_err(openai_backend_error)?;
                let token_runtime_lock_wait_ms = batch_outcome.runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms = batch_outcome.runtime_lock_hold_ms;
                decode_runtime_lock_wait_ms += token_runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(token_runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_batch_size_max = decode_batch_size_max.max(batch_outcome.batch_size);
                decode_batch_wait_ms += batch_outcome.batch_wait_ms;
                let output = batch_outcome.output;
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    request.config,
                    message,
                    &output,
                    request.wire_dtype,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_forward_activation_encode_ms += forwarded.activation_encode_ms;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    request.wire_dtype,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = super::embedded_execution::receive_embedded_stage_reply(
                    downstream,
                    request.prediction_return.as_ref(),
                    WireReplyKind::PredictedToken,
                )?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if records_replay_checkpoint
                    && super::prefix_cache::request_allows_exact_replay(&request)
                {
                    self.record_embedded_stage0_replay_checkpoint(
                        super::prefix_cache::EmbeddedReplayCheckpointRecord {
                            session_id: &session_key,
                            ids: request.ids,
                            prompt_token_ids: request.prompt_token_ids,
                            checkpoint_token_ids: &context_tokens,
                            predicted_tokens: &exact_replay_tokens,
                            predicted: reply.predicted,
                            sampling: request.sampling,
                            chat_sampling_metadata: request.chat_sampling_metadata,
                        },
                    )?;
                } else if records_full_prompt_checkpoint {
                    self.record_embedded_stage0_full_prompt_first_token(
                        &session_key,
                        request.ids,
                        request.prompt_token_ids,
                        reply.predicted,
                    )?;
                }
                current = reply.predicted;
                let suppress_cooldown_draft_broad = native_mtp_options.suppress_cooldown_drafts
                    && native_mtp_reject_cooldown_remaining > 0;
                let suppress_cooldown_draft_limited = native_mtp_reject_cooldown_remaining > 0
                    && native_mtp_suppress_cooldown_drafts_remaining > 0;
                let suppress_cooldown_draft =
                    suppress_cooldown_draft_broad || suppress_cooldown_draft_limited;
                let native_mtp_draft = if suppress_cooldown_draft {
                    None
                } else {
                    reply
                        .native_mtp_draft
                        .clone()
                        .map(NativeMtpDraft::from_stage_draft)
                };
                if suppress_cooldown_draft {
                    native_mtp.clear_pending_draft();
                    native_mtp_counters.observe_suppressed_cooldown_draft();
                    native_mtp_suppress_cooldown_drafts_remaining =
                        native_mtp_suppress_cooldown_drafts_remaining.saturating_sub(1);
                }
                let native_mtp_decision = native_mtp.observe_target_token(
                    current,
                    ms_to_us(downstream_wait_ms),
                    native_mtp_draft,
                    if native_mtp_counters.verify_window_verification_count() == 0 {
                        NativeMtpDraftOrigin::InitialSerial
                    } else {
                        NativeMtpDraftOrigin::SerialAfterGap
                    },
                );
                native_mtp_reject_cooldown_remaining =
                    native_mtp_reject_cooldown_remaining.saturating_sub(1);
                decoded_tokens += 1;
                exact_replay_tokens.push(current);
                context_tokens.push(current);
                if self.telemetry.is_debug_enabled() {
                    let mut token_attrs = self.openai_attrs(request.ids);
                    token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                    token_attrs.insert(
                        "llama_stage.decode_token_phase".to_string(),
                        json!(decode_token_phase(decode_step)),
                    );
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(stage0_compute_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(token_runtime_lock_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(token_runtime_lock_hold_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_size".to_string(),
                        json!(batch_outcome.batch_size),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_wait_ms".to_string(),
                        json!(batch_outcome.batch_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.message.activation.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.activation_encode_ms".to_string(),
                        json!(forwarded.activation_encode_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(forward_write_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(downstream_wait_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                    token_attrs.insert(
                        "llama_stage.native_mtp.verification".to_string(),
                        json!(native_mtp_decision.label()),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
                        json!(native_mtp_options.suppress_cooldown_drafts),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
                        json!(native_mtp_options.suppress_cooldown_draft_limit),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.cooldown_draft_suppressed".to_string(),
                        json!(suppress_cooldown_draft),
                    );
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            if !pipelined_windows.is_empty() {
                let stale_count = pipelined_windows.len();
                let stale_drain_timer = PhaseTimer::start();
                while let Some(stale) = pipelined_windows.pop_front() {
                    let stale_reply = self.complete_dispatched_stage_message_direct(
                        &request,
                        downstream,
                        stale.dispatched,
                        WireReplyKind::PredictedTokens,
                    )?;
                    verify_window_scheduler.complete_next(stale_reply.reply.window.window_id)?;
                }
                verify_window_scheduler
                    .record_stale_discarded(stale_count, stale_drain_timer.elapsed_ms());
                let trim = self.trim_embedded_stage_session(
                    &request,
                    downstream,
                    &session_key,
                    request_id,
                    session_id,
                    prefill_token_count + decoded_tokens,
                )?;
                speculative_stats.recovery_ms += trim.elapsed_ms;
            }
            if let Some(pipeline) = pipelined.take() {
                native_mtp_counters
                    .observe_hybrid_proposal(pipeline.proposal(), pipeline.accepted_tokens());
                native_mtp.clear_pending_draft();
            }
            let native_mtp_stats = native_mtp.stats();
            self.record_embedded_decode_summary(
                &request,
                &mut cache_stats,
                decode_timer,
                EmbeddedDecodeSummary {
                    decoded_tokens,
                    stage0_compute_ms: decode_stage0_compute_ms,
                    runtime_lock_wait_ms: decode_runtime_lock_wait_ms,
                    runtime_lock_wait_max_ms: decode_runtime_lock_wait_max_ms,
                    runtime_lock_hold_ms: decode_runtime_lock_hold_ms,
                    runtime_lock_hold_max_ms: decode_runtime_lock_hold_max_ms,
                    runtime_lock_acquires: decode_runtime_lock_acquires,
                    decode_batch_size_max,
                    decode_batch_wait_ms,
                    forward_write_ms: decode_forward_write_ms,
                    activation_encode_ms: decode_forward_activation_encode_ms,
                    output_activation_bytes: decode_output_activation_bytes,
                    forward_activation_bytes: decode_forward_activation_bytes,
                    downstream_wait_ms: decode_downstream_wait_ms,
                    speculative_stats: &speculative_stats,
                    native_mtp_stats,
                    native_mtp_counters,
                    native_mtp_options,
                    verify_window_scheduler: &verify_window_scheduler,
                },
            );
            Ok(())
        })();

        self.finish_embedded_generation_session(&request, lane_pool, lane, &result, &session_key)?;
        result?;
        Ok(cache_stats)
    }
}
