use crate::frontend::generation::{
    EmbeddedStageZeroGeneration, GeneratedText, GenerationTokenLimit, InferenceBackendMode,
    InferenceGenerationIds, LocalGeneration, PhaseTimer, PreparedGenerationPrompt,
    PreparedTextPrompt, StageOpenAiBackend, TextGenerationCollector, emulation_generation_active,
};
use crate::frontend::util::{generation_stop_values, openai_backend_error};
use serde_json::json;
use skippy_inference_api::{InferenceError, InferenceResult, generation::GenerationRequest};
use skippy_runtime::{ModelWorkload, SamplingConfig};

pub(super) fn resident_capacity_target_tokens(prompt_token_count: usize) -> u64 {
    u64::try_from(prompt_token_count).unwrap_or(u64::MAX)
}

impl StageOpenAiBackend {
    pub(in crate::frontend) fn prepare_text_prompt(
        &self,
        prompt: &PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        ids: &InferenceGenerationIds,
    ) -> InferenceResult<PreparedTextPrompt> {
        let tokenize_timer = PhaseTimer::start();
        let token_ids = self.tokenize(&prompt.text)?;
        let mut tokenize_attrs = self.openai_attrs(ids);
        tokenize_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        tokenize_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(token_ids.len()),
        );
        self.emit_openai_phase("stage.openai_tokenize", tokenize_timer, tokenize_attrs);
        if token_ids.is_empty() {
            return Err(InferenceError::invalid_request("prompt produced no tokens"));
        }
        let max_tokens = max_tokens.resolve(token_ids.len(), self.ctx_size)?;
        Ok(PreparedTextPrompt {
            token_ids,
            max_tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::frontend) fn generate_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        prepared_text: Option<PreparedTextPrompt>,
        token_budget_stats: (usize, usize),
        stop: Option<&skippy_inference_api::StopSequence>,
        mut sampling: SamplingConfig,
        hook_request: Option<GenerationRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&skippy_inference_api::CancellationToken>,
        ids: InferenceGenerationIds,
        on_text_chunk: impl FnMut(&str) -> InferenceResult<()>,
    ) -> InferenceResult<GeneratedText> {
        let generation_timer = PhaseTimer::start();
        if cancellation.is_some_and(skippy_inference_api::CancellationToken::is_cancelled) {
            return Err(InferenceError::backend("request cancelled"));
        }
        if prompt.text.is_empty() {
            return Err(InferenceError::invalid_request(
                "request prompt/messages produced no text",
            ));
        }
        if prompt.has_media() {
            if sampling.top_logprobs.is_some() {
                return Err(InferenceError::unsupported(
                    "logprobs are not implemented yet for requests with media",
                ));
            }
            return self.generate_multimodal_text(
                prompt,
                max_tokens,
                stop,
                sampling,
                hook_request,
                hook_runtime,
                cancellation,
                ids,
                on_text_chunk,
            );
        }
        let stop_value_storage =
            generation_stop_values(stop, prompt.chat_parse_metadata.as_deref());
        let stop_values = stop_value_storage
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let PreparedTextPrompt {
            token_ids: prompt_token_ids,
            max_tokens,
        } = match prepared_text {
            Some(prepared) => prepared,
            None => self.prepare_text_prompt(&prompt, max_tokens, &ids)?,
        };
        let workload = self.model_workload()?;
        if workload == ModelWorkload::EncoderDecoder {
            if sampling.top_logprobs.is_some() {
                return Err(InferenceError::unsupported(
                    "logprobs are not implemented yet for encoder-decoder models",
                ));
            }
            let mut collector =
                TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk)?
                    .with_ignore_eos(sampling.ignore_eos);
            let cache_stats = self.generate_encoder_decoder_tokens(
                &prompt_token_ids,
                max_tokens,
                &sampling,
                hook_request.as_ref(),
                cancellation,
                &ids,
                |token| collector.push_token(token),
            )?;
            return collector.finish(prompt_token_ids.len(), cache_stats);
        }
        if workload != ModelWorkload::CausalGeneration {
            return Err(InferenceError::unsupported(format!(
                "model workload is {workload:?}; chat and completion endpoints require a generative model"
            )));
        }
        sampling.resolve_reasoning_budget(max_tokens);
        // This is an optional cache candidate. The already-rendered prompt is
        // valid even when its second, assistant-marker-free rendering cannot
        // be tokenized, so bypass the candidate rather than failing the chat
        // request.
        let recurrent_cache_prefix_token_ids = prompt
            .recurrent_cache_prefix_text
            .as_deref()
            .and_then(|prefix| match self.tokenize(prefix) {
                Ok(prefix) => Some(prefix),
                Err(_error) => {
                    let mut attrs = self.openai_attrs(&ids);
                    attrs.insert(
                        "skippy.kv.decision".to_string(),
                        json!("recurrent_prefix_candidate_skipped"),
                    );
                    attrs.insert("skippy.kv.error_class".to_string(), json!("tokenize_error"));
                    self.telemetry
                        .emit_debug("stage.openai_kv_record_decision", attrs);
                    None
                }
            })
            .filter(|prefix| {
                !prefix.is_empty()
                    && prefix.len() <= prompt_token_ids.len()
                    && prompt_token_ids.starts_with(prefix)
            });
        if cancellation.is_some_and(skippy_inference_api::CancellationToken::is_cancelled) {
            return Err(InferenceError::backend("request cancelled"));
        }
        if cancellation.is_some_and(skippy_inference_api::CancellationToken::is_cancelled) {
            return Err(InferenceError::backend("request cancelled"));
        }
        let token_admit_timer = PhaseTimer::start();
        let (reserved_tokens, active_reserved_tokens) = token_budget_stats;
        let mut token_admit_attrs = self.openai_attrs(&ids);
        token_admit_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(prompt_token_ids.len()),
        );
        token_admit_attrs.insert("llama_stage.max_tokens".to_string(), json!(max_tokens));
        token_admit_attrs.insert(
            "llama_stage.kv_reserved_tokens".to_string(),
            json!(reserved_tokens),
        );
        token_admit_attrs.insert(
            "llama_stage.kv_active_reserved_tokens".to_string(),
            json!(active_reserved_tokens),
        );
        token_admit_attrs.insert(
            "llama_stage.kv_capacity_tokens".to_string(),
            json!(self.generation_token_budget.capacity_tokens()),
        );
        self.emit_openai_phase(
            "stage.openai_generation_token_admit",
            token_admit_timer,
            token_admit_attrs,
        );
        let _resident_capacity_reservation = match self.kv.as_ref() {
            Some(kv) => kv
                .reserve_resident_capacity(
                    &ids.session_label,
                    // The resident allocator adds its own n_batch-sized free
                    // watermark for decode. Passing the outer reservation
                    // (prompt + decode headroom) would charge that headroom a
                    // second time and reject prompts that fit the shared pool.
                    resident_capacity_target_tokens(prompt_token_ids.len()),
                )
                .map_err(openai_backend_error)?,
            None => None,
        };
        let chat_sampling_metadata = prompt.chat_parse_metadata.as_deref();

        let emulation_active = emulation_generation_active(hook_request.as_ref(), &prompt);
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk)?
                .with_emulation_stop(emulation_active)
                .with_ignore_eos(sampling.ignore_eos);
        // The gate's verdict, applied once for this request rather than looked
        // up per token: a request that straddles a flip is correct either way,
        // since speculation is a throughput choice and never a correctness one.
        let speculating = self.gate_allows_speculation();
        let gated_speculative = self.gated_speculative();
        // Depth after the on/off verdict, and in that order: searching a budget
        // for a request the gate has just stood down would attribute the gate's
        // effect to the budget.
        let gated_speculative = crate::frontend::runahead_search::runahead_after_search(
            &gated_speculative,
            self.runahead_governor.as_deref(),
        );
        let speculation_summary = self.speculation_summary(
            &gated_speculative,
            speculating,
            sampling.top_logprobs.is_some(),
        );
        // Speculative windows, native MTP, and DFlash blocks sample or accept
        // tokens outside the one-token-per-sample path that records logprobs.
        let gated_speculative = if sampling.top_logprobs.is_some() {
            let mut stood_down = gated_speculative.into_owned();
            stood_down.ngram = None;
            stood_down.extension = None;
            stood_down.native_mtp.enabled = false;
            stood_down.dflash = None;
            std::borrow::Cow::Owned(stood_down)
        } else {
            gated_speculative
        };
        if sampling.top_logprobs.is_some()
            && !matches!(self.mode, InferenceBackendMode::LocalRuntime)
        {
            return Err(InferenceError::unsupported(
                "logprobs are not implemented yet for split (multi-stage) serving",
            ));
        }

        let cache_stats = match self.mode.clone() {
            InferenceBackendMode::LocalRuntime => self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: &prompt_token_ids,
                    recurrent_cache_prefix_token_ids: recurrent_cache_prefix_token_ids.as_deref(),
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    speculative: &gated_speculative,
                    native_mtp_enabled: self.config.native_mtp_enabled
                        && gated_speculative.native_mtp.enabled,
                    hook_request: hook_request.clone(),
                    hook_runtime: hook_runtime.clone(),
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            InferenceBackendMode::EmbeddedStageZero {
                config,
                prefill_chunk_policy,
                activation_width,
                downstream_wire_condition,
                prefill_reply_credit_limit,
                lane_pool,
                prediction_returns,
            } => {
                // A split DFlash target drafts on its final stage, whose drafts
                // stage 0 verifies like native MTP drafts.
                let split_dflash = crate::split_dflash::split_stage0_plan(&gated_speculative);
                let speculative = split_dflash.as_ref().unwrap_or(&gated_speculative);
                self.generate_embedded_stage_zero_tokens(
                    EmbeddedStageZeroGeneration {
                        config: &config,
                        prefill_chunk_policy: &prefill_chunk_policy,
                        activation_width,
                        downstream_wire_condition,
                        prefill_reply_credit_limit,
                        lane_pool,
                        prediction_return: prediction_returns
                            .as_ref()
                            .map(|hub| hub.register(ids.request_id, ids.session_id))
                            .transpose()
                            .map_err(openai_backend_error)?,
                        // Stood down with the proposers. Clearing `ngram` and
                        // `extension` while still handing over a draft runner
                        // leaves the classic serial draft loop able to propose,
                        // so the gate would not actually have turned speculation
                        // off.
                        draft: if speculating {
                            self.draft.clone()
                        } else {
                            None
                        },
                        speculative_window: self.speculative_window,
                        adaptive_speculative_window: self.adaptive_speculative_window,
                        speculative,
                        ngram_max: self.ngram_max,
                        native_mtp_enabled: split_dflash.is_some()
                            || (config.native_mtp_enabled && gated_speculative.native_mtp.enabled),
                        split_dflash_draft: split_dflash
                            .is_some()
                            .then(|| gated_speculative.dflash.as_ref())
                            .flatten()
                            .map(|dflash| dflash.draft_model_path.as_path()),
                        split_dflash_context_window: split_dflash
                            .is_some()
                            .then(|| gated_speculative.dflash.as_ref())
                            .flatten()
                            .and_then(|dflash| dflash.context_window),
                        split_dflash_downstream_bytes_per_token: split_dflash
                            .is_some()
                            .then(|| gated_speculative.dflash.as_ref())
                            .flatten()
                            .map_or(0, |dflash| {
                                crate::split_dflash::downstream_feature_bytes_per_token(
                                    dflash,
                                    config.layer_end,
                                    usize::try_from(activation_width).unwrap_or(0),
                                )
                            }),
                        split_dflash_health: Default::default(),
                        prompt_token_ids: &prompt_token_ids,
                        recurrent_cache_prefix_token_ids: recurrent_cache_prefix_token_ids
                            .as_deref(),
                        max_tokens,
                        sampling: &sampling,
                        chat_sampling_metadata,
                        hook_request,
                        hook_runtime,
                        cancellation,
                        ids: &ids,
                    },
                    |token| collector.push_token(token),
                )?
            }
        };

        let mut output = collector.finish(prompt_token_ids.len(), cache_stats)?;
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        summary_attrs.insert("skippy.kv.status".to_string(), json!(output.cache_status));
        summary_attrs.insert(
            "skippy.kv.cached_prompt_tokens".to_string(),
            json!(output.cached_prompt_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.matched_prefix_tokens".to_string(),
            json!(output.matched_prefix_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.suffix_prefill_tokens".to_string(),
            json!(output.suffix_prefill_tokens),
        );
        summary_attrs.insert(
            "skippy.kv.hit_kind".to_string(),
            json!(output.cache_hit_kind.unwrap_or("none")),
        );
        summary_attrs.insert(
            "llama_stage.detokenize_ms".to_string(),
            json!(output.detokenize_ms),
        );
        summary_attrs.insert(
            "llama_stage.text_emit_ms".to_string(),
            json!(output.text_emit_ms),
        );
        summary_attrs.insert(
            "llama_stage.eog_check_ms".to_string(),
            json!(output.eog_check_ms),
        );
        let total_ms = self.emit_openai_summary(
            "stage.openai_generation_summary",
            generation_timer,
            summary_attrs,
        );
        // A logprobs request ran with speculation stood down above: its decode
        // rate must not count toward the speculative window.
        self.record_speculation_outcome(&output, speculating && sampling.top_logprobs.is_none());
        if crate::generation_stats::observer_installed() {
            output.generation_stats = Some(self.generation_stats(
                &output,
                GenerationStatsInputs {
                    speculative_strategy: &gated_speculative.effective_strategy,
                    speculation_allowed: speculating && sampling.top_logprobs.is_none(),
                    speculation: speculation_summary,
                    request_shape: prompt.shape,
                    total_ms,
                },
            ));
        }
        Ok(output)
    }
}

/// The speculation plan a request ran under, as the stats observer reports it.
#[derive(Clone, Copy, Debug)]
struct SpeculationSummary {
    stand_down: Option<crate::generation_stats::SpeculationStandDown>,
    draft_token_cap: Option<usize>,
    verify_window_max_tokens: usize,
    runahead_budget_tokens: usize,
}

/// What [`StageOpenAiBackend::generation_stats`] needs beyond the output.
struct GenerationStatsInputs<'a> {
    speculative_strategy: &'a str,
    speculation_allowed: bool,
    speculation: SpeculationSummary,
    request_shape: crate::generation_stats::RequestShape,
    total_ms: f64,
}

/// Most draft tokens the plan's proposer offers per verify window.
///
/// Proposers are checked in the order the generation paths prefer them. A
/// DFlash draft without a stated cap uses its trained block, which this
/// frontend does not know, so it is `None` rather than a guess.
fn draft_token_cap(
    plan: &crate::frontend::SpeculativeDecodeConfig,
    draft_model_window: Option<usize>,
) -> Option<usize> {
    if let Some(dflash) = plan.dflash.as_ref() {
        return dflash.max_draft_tokens;
    }
    if plan.native_mtp.enabled {
        return Some(plan.native_mtp.max_draft_tokens);
    }
    if let Some(ngram) = plan.ngram.as_ref() {
        return Some(ngram.max_proposal_tokens);
    }
    draft_model_window
}

impl StageOpenAiBackend {
    /// Whether the resolved plan, before any gating, speculates at all.
    fn plan_speculates(&self) -> bool {
        let plan = &self.speculative;
        plan.dflash.is_some()
            || plan.native_mtp.enabled
            || plan.ngram.is_some()
            || plan.extension.is_some()
            || self.draft.is_some()
    }

    /// The speculation plan for one request, and why it stood down if it did.
    ///
    /// The cap and window come from the ungated plan, so a stood-down request
    /// still says what it would have run with; the run-ahead budget is the one
    /// the search settled on for this request.
    fn speculation_summary(
        &self,
        after_search: &crate::frontend::SpeculativeDecodeConfig,
        speculating: bool,
        logprobs: bool,
    ) -> SpeculationSummary {
        use crate::generation_stats::SpeculationStandDown;

        let stand_down = if !self.plan_speculates() {
            None
        } else if !speculating {
            let trialling = self
                .speculation_governor
                .as_ref()
                .is_some_and(|governor| governor.is_trialling());
            Some(if trialling {
                SpeculationStandDown::GateTrial
            } else {
                SpeculationStandDown::GateMeasuredLoss
            })
        } else {
            logprobs.then_some(SpeculationStandDown::Logprobs)
        };
        SpeculationSummary {
            stand_down,
            draft_token_cap: draft_token_cap(
                &self.speculative,
                self.draft.as_ref().map(|_| self.speculative_window),
            ),
            verify_window_max_tokens: self.speculative.verify_window.max_tokens,
            runahead_budget_tokens: after_search.verify_window.runahead_max_tokens,
        }
    }

    /// The observer's view of one finished generation: counts and timings only.
    fn generation_stats(
        &self,
        output: &GeneratedText,
        inputs: GenerationStatsInputs<'_>,
    ) -> crate::generation_stats::GenerationStats {
        let GenerationStatsInputs {
            speculative_strategy,
            speculation_allowed,
            speculation,
            request_shape,
            total_ms,
        } = inputs;
        let (split, downstream_lanes) = match &self.mode {
            InferenceBackendMode::LocalRuntime => (false, None),
            InferenceBackendMode::EmbeddedStageZero { lane_pool, .. } => {
                (true, lane_pool.as_ref().map(|pool| pool.capacity))
            }
        };
        let draft = output.speculative_stats.as_ref();
        let mtp = output.native_mtp_stats;
        let wire = output.activation_wire;
        crate::generation_stats::GenerationStats {
            model_id: self.model_id.clone(),
            split,
            downstream_lanes,
            concurrency_limit: self.generation_limit.current_limit(),
            context_size: self.ctx_size,
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            cached_prompt_tokens: output.cached_prompt_tokens,
            prefill_tokens: output.suffix_prefill_tokens,
            cache_hit_kind: output.cache_hit_kind.unwrap_or("none"),
            queue_wait_ms: output.queue_wait_ms,
            restore_ms: output.restore_ms,
            prefill_ms: output.prompt_ms,
            decode_ms: output.predicted_ms,
            total_ms,
            finish_reason: output.finish_reason.as_str(),
            speculative_strategy: speculative_strategy.to_owned(),
            speculation_allowed,
            speculation_stand_down: speculation.stand_down,
            draft_token_cap: speculation.draft_token_cap,
            verify_window_max_tokens: speculation.verify_window_max_tokens,
            runahead_budget_tokens: speculation.runahead_budget_tokens,
            draft_windows: draft.map_or(0, |stats| stats.windows as u64),
            draft_proposed_tokens: draft.map_or(0, |stats| stats.draft_tokens as u64),
            draft_accepted_tokens: draft.map_or(0, |stats| stats.accepted_tokens as u64),
            mtp_verifications: mtp.verification_count,
            mtp_proposed_tokens: mtp.drafted_tokens,
            mtp_accepted_tokens: mtp.accepted_tokens,
            activation_codec: wire.map(|wire| wire.codec),
            activation_prefill_bytes: wire.map_or(0, |wire| wire.prefill_bytes as u64),
            activation_decode_bytes: wire.map_or(0, |wire| wire.decode_bytes as u64),
            request_shape,
        }
    }
}
