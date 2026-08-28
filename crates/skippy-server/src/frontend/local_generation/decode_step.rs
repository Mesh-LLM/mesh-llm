use std::time::Duration;

use openai_frontend::OpenAiResult;
use serde_json::json;

use crate::frontend::generation::{
    GenerationCacheStats, LocalGeneration, PhaseTimer, StageOpenAiBackend, TokenControl,
    decode_token_phase,
};
use crate::frontend::{NativeMtpDraft, NativeMtpDraftOrigin};

use super::token_generation::{DecodeState, decode_native_mtp};

impl StageOpenAiBackend {
    pub(in crate::frontend) fn generation_signal_window_tokens(&self) -> u32 {
        match &self.mode {
            crate::frontend::OpenAiBackendMode::EmbeddedStageZero { config, .. } => {
                config.generation_signal_window.unwrap_or(16)
            }
            crate::frontend::OpenAiBackendMode::LocalRuntime => 16,
        }
    }

    pub(super) fn decode_one_token(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<TokenControl> {
        let decode_step = state.decoded_tokens;
        let token_timer = PhaseTimer::start();
        let decode_call_timer = PhaseTimer::start();
        let (
            predicted,
            mut native_mtp_draft,
            token_batch_size,
            token_batch_wait_ms,
            token_runtime_lock_wait_ms,
            token_runtime_lock_hold_ms,
        ) = if request.native_mtp_enabled {
            let session_id = session_id.to_string();
            let sampling = request.sampling.enabled.then_some(request.sampling.clone());
            let max_draft_tokens = state.native_mtp_options.max_draft_tokens;
            let current = state.current;
            let outcome = self.iteration_scheduler.execute_runtime_timed(
                "native-mtp-decode",
                move |runtime| {
                    decode_native_mtp(
                        runtime,
                        &session_id,
                        current,
                        sampling.as_ref(),
                        max_draft_tokens,
                    )
                },
            )?;
            let (predicted, draft) = outcome.value;
            let native_mtp_draft = draft;
            let token_batch_size = 1;
            let token_batch_wait_ms = outcome.queue_wait_ms;
            let token_runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
            let token_runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
            (
                predicted,
                native_mtp_draft,
                token_batch_size,
                token_batch_wait_ms,
                token_runtime_lock_wait_ms,
                token_runtime_lock_hold_ms,
            )
        } else {
            let outcome = self.iteration_scheduler.execute_iteration(
                session_id,
                &[state.current],
                &[],
                request.sampling.enabled.then_some(request.sampling),
                true,
                skippy_runtime::IterationBatchPhase::Decode,
            )?;
            (
                outcome.predicted,
                None,
                outcome.batch_size,
                outcome.batch_wait_ms,
                outcome.runtime_lock_wait_ms,
                outcome.runtime_lock_hold_ms,
            )
        };
        state.current = predicted;
        if native_mtp_draft
            .as_ref()
            .is_some_and(|draft: &NativeMtpDraft| {
                draft.tokens.len() < state.native_mtp_options.min_draft_tokens
            })
        {
            native_mtp_draft = None;
        }
        let is_first_draft = state.decoded_tokens == 0;
        let draft_origin = if is_first_draft {
            NativeMtpDraftOrigin::InitialSerial
        } else {
            NativeMtpDraftOrigin::SerialAfterGap
        };
        let native_mtp_decision = request.native_mtp_enabled.then(|| {
            state
                .native_mtp
                .observe_target_token(state.current, 0, native_mtp_draft, draft_origin)
        });
        state.runtime_lock_wait_ms += token_runtime_lock_wait_ms;
        state.runtime_lock_wait_max_ms = state
            .runtime_lock_wait_max_ms
            .max(token_runtime_lock_wait_ms);
        state.runtime_lock_hold_ms += token_runtime_lock_hold_ms;
        state.runtime_lock_hold_max_ms = state
            .runtime_lock_hold_max_ms
            .max(token_runtime_lock_hold_ms);
        state.runtime_lock_acquires += 1;
        let token_decode_ms = if state.emit_token_debug {
            decode_call_timer.elapsed_ms()
        } else {
            0.0
        };
        let (token_signal, signal_window, token_signal_ms) = if state.generation_hooks_active {
            let signal_timer = PhaseTimer::start();
            let scheduler_session_id = session_id.to_string();
            let signal_window_tokens = self.generation_signal_window_tokens();
            let (sessions_before, token_signal, signal_window, sessions_after) = self
                .iteration_scheduler
                .execute_runtime("generation-hook-signals", move |runtime| {
                    let sessions_before = runtime.session_stats();
                    let token_signal = runtime.last_token_signal(&scheduler_session_id).ok();
                    let signal_window = runtime
                        .signal_window(&scheduler_session_id, signal_window_tokens)
                        .ok();
                    let sessions_after = runtime.session_stats();
                    Ok((sessions_before, token_signal, signal_window, sessions_after))
                })?;
            state.runtime_sessions_before.get_or_insert(sessions_before);
            state.runtime_sessions_after = Some(sessions_after);
            (token_signal, signal_window, signal_timer.elapsed_ms())
        } else {
            (None, None, 0.0)
        };
        let injected_current = if state.generation_hooks_active {
            self.maybe_run_generation_hooks(
                session_id,
                &mut state.hook_request,
                state.hook_runtime.as_ref(),
                state.decoded_tokens,
                &mut state.post_prefill_hook_checked,
                &mut state.last_mid_generation_hook_at,
                token_signal,
                signal_window,
            )?
        } else {
            None
        };
        if let Some(injected_current) = injected_current {
            state.current = injected_current;
            return Ok(TokenControl::Continue);
        }
        state.decoded_tokens += 1;
        state.generated_token_ids.push(state.current);
        if let Some(committed_token_ids) = state.linear_context_tokens.as_mut() {
            committed_token_ids.push(state.current);
        }
        if state.emit_token_debug {
            let mut token_attrs = self.openai_attrs(request.ids);
            token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
            token_attrs.insert(
                "llama_stage.decode_token_phase".to_string(),
                json!(decode_token_phase(
                    u32::try_from(decode_step).unwrap_or(u32::MAX)
                )),
            );
            token_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(token_timer.elapsed_ms()),
            );
            token_attrs.insert(
                "llama_stage.decode_call_ms".to_string(),
                json!(token_decode_ms),
            );
            token_attrs.insert(
                "llama_stage.decode_batch_size".to_string(),
                json!(token_batch_size),
            );
            token_attrs.insert(
                "llama_stage.decode_batch_wait_ms".to_string(),
                json!(token_batch_wait_ms),
            );
            token_attrs.insert("llama_stage.signal_ms".to_string(), json!(token_signal_ms));
            token_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(token_runtime_lock_wait_ms),
            );
            token_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(token_runtime_lock_hold_ms),
            );
            token_attrs.insert(
                "llama_stage.predicted_token".to_string(),
                json!(state.current),
            );
            if let Some(native_mtp_decision) = native_mtp_decision {
                token_attrs.insert(
                    "llama_stage.native_mtp.verification".to_string(),
                    json!(native_mtp_decision.label()),
                );
            }
            token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
            self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
        }
        emit_token(state.current)
    }

    pub(super) fn emit_decode_summary(
        &self,
        request: &LocalGeneration<'_>,
        state: &mut DecodeState,
        cache_stats: &mut GenerationCacheStats,
        decode_timer: PhaseTimer,
    ) -> OpenAiResult<Duration> {
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.decode_token_count".to_string(),
            json!(state.decoded_tokens),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(state.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_max_ms".to_string(),
            json!(state.runtime_lock_wait_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(state.runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_max_ms".to_string(),
            json!(state.runtime_lock_hold_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_acquires".to_string(),
            json!(state.runtime_lock_acquires),
        );
        if let Some(stats) = state.runtime_sessions_before.as_ref() {
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_before",
                stats,
            );
        }
        if let Some(stats) = state.runtime_sessions_after.as_ref() {
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                stats,
            );
        }
        request.speculative.insert_telemetry_attrs(&mut attrs);
        let native_mtp_stats = state.native_mtp.stats();
        cache_stats.native_mtp_stats = native_mtp_stats;
        let model_generation_elapsed = decode_timer.start_instant.elapsed();
        cache_stats.predicted_ms = model_generation_elapsed.as_secs_f64() * 1_000.0;
        native_mtp_stats.insert_attrs(&mut attrs);
        self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
        Ok(model_generation_elapsed)
    }
}
