use crate::frontend::admission::GenerationTokenBudgetRequest;
use crate::frontend::generation::{
    EmbeddedStageZeroGeneration, GENERATION_TOKEN_BUDGET_TIMEOUT, GeneratedText,
    GenerationTokenLimit, LocalGeneration, OpenAiBackendMode, OpenAiGenerationIds, PhaseTimer,
    PreparedGenerationPrompt, PreparedTextPrompt, StageOpenAiBackend, TextGenerationCollector,
    emulation_generation_active,
};
use crate::frontend::util::{generation_stop_values, openai_backend_error};
use openai_frontend::{ChatCompletionRequest, OpenAiError, OpenAiResult};
use serde_json::json;
use skippy_runtime::SamplingConfig;

impl StageOpenAiBackend {
    pub(in crate::frontend) fn prepare_text_prompt(
        &self,
        prompt: &PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        ids: &OpenAiGenerationIds,
    ) -> OpenAiResult<PreparedTextPrompt> {
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
            return Err(OpenAiError::invalid_request("prompt produced no tokens"));
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
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let generation_timer = PhaseTimer::start();
        if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
            return Err(OpenAiError::backend("request cancelled"));
        }
        if prompt.text.is_empty() {
            return Err(OpenAiError::invalid_request(
                "request prompt/messages produced no text",
            ));
        }
        if prompt.has_media() {
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
        if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let token_admit_timer = PhaseTimer::start();
        let token_budget_reservation = self.generation_token_budget.reserve_cancellable(
            GenerationTokenBudgetRequest::new(prompt_token_ids.len(), max_tokens),
            GENERATION_TOKEN_BUDGET_TIMEOUT,
            cancellation,
        )?;
        if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let mut token_admit_attrs = self.openai_attrs(&ids);
        token_admit_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(prompt_token_ids.len()),
        );
        token_admit_attrs.insert("llama_stage.max_tokens".to_string(), json!(max_tokens));
        token_admit_attrs.insert(
            "llama_stage.kv_reserved_tokens".to_string(),
            json!(token_budget_reservation.tokens()),
        );
        token_admit_attrs.insert(
            "llama_stage.kv_active_reserved_tokens".to_string(),
            json!(token_budget_reservation.active_tokens_after_reservation()),
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
        let chat_sampling_metadata = prompt.chat_parse_metadata.as_deref();

        let emulation_active = emulation_generation_active(hook_request.as_ref(), &prompt);
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk)?
                .with_emulation_stop(emulation_active)
                .with_ignore_eos(sampling.ignore_eos);
        let cache_stats = match self.mode.clone() {
            OpenAiBackendMode::LocalRuntime => self.generate_local_tokens(
                LocalGeneration {
                    prompt_token_ids: &prompt_token_ids,
                    recurrent_cache_prefix_token_ids: recurrent_cache_prefix_token_ids.as_deref(),
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    speculative: &self.speculative,
                    native_mtp_enabled: self.config.native_mtp_enabled
                        && self.speculative.native_mtp.enabled,
                    hook_request: hook_request.clone(),
                    hook_runtime: hook_runtime.clone(),
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
            OpenAiBackendMode::EmbeddedStageZero {
                config,
                prefill_chunk_policy,
                activation_width,
                downstream_wire_condition,
                prefill_reply_credit_limit,
                lane_pool,
                prediction_returns,
            } => self.generate_embedded_stage_zero_tokens(
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
                    draft: self.draft.clone(),
                    speculative_window: self.speculative_window,
                    adaptive_speculative_window: self.adaptive_speculative_window,
                    speculative: &self.speculative,
                    ngram_max: self.ngram_max,
                    native_mtp_enabled: config.native_mtp_enabled
                        && self.speculative.native_mtp.enabled,
                    prompt_token_ids: &prompt_token_ids,
                    max_tokens,
                    sampling: &sampling,
                    chat_sampling_metadata,
                    hook_request,
                    hook_runtime,
                    cancellation,
                    ids: &ids,
                },
                |token| collector.push_token(token),
            )?,
        };

        let output = collector.finish(prompt_token_ids.len(), cache_stats)?;
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
        self.emit_openai_summary(
            "stage.openai_generation_summary",
            generation_timer,
            summary_attrs,
        );
        Ok(output)
    }
}
