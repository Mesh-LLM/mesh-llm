use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::binary::{StageWireMessage, WireReplyKind, recv_reply, write_stage_message};

use crate::frontend::{
    NativeMtpDecodeCounters, NativeMtpDecodeOptions, NativeMtpDecodeTelemetry, NativeMtpStats,
    decode_scheduler::{VerifyWindow, VerifyWindowScheduler},
    embedded_execution::DispatchedEmbeddedStage,
    generation::{
        EmbeddedStageZeroGeneration, GenerationCacheStats, LocalGeneration, PersistentStageLane,
        PersistentStageLanePool, PhaseTimer, StageOpenAiBackend, TokenControl,
    },
    speculative::OpenAiSpeculativeStats,
    util::openai_io_error,
};

pub(super) struct PipelinedCompositeWindow {
    pub(super) window: VerifyWindow,
    pub(super) input_tokens: Vec<i32>,
    pub(super) proposal_tokens: Vec<i32>,
    pub(super) expected_free_target: Option<i32>,
    pub(super) native_mtp_token_count: usize,
    pub(super) dispatched: DispatchedEmbeddedStage,
}

pub(super) struct EmbeddedDecodeSummary<'a> {
    pub(super) decoded_tokens: usize,
    pub(super) stage0_compute_ms: f64,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_wait_max_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_hold_max_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) decode_batch_size_max: usize,
    pub(super) decode_batch_wait_ms: f64,
    pub(super) forward_write_ms: f64,
    pub(super) activation_encode_ms: f64,
    pub(super) output_activation_bytes: usize,
    pub(super) forward_activation_bytes: usize,
    pub(super) downstream_wait_ms: f64,
    pub(super) speculative_stats: &'a OpenAiSpeculativeStats,
    pub(super) native_mtp_stats: NativeMtpStats,
    pub(super) native_mtp_counters: NativeMtpDecodeCounters,
    pub(super) native_mtp_options: NativeMtpDecodeOptions,
    pub(super) verify_window_scheduler: &'a VerifyWindowScheduler,
}

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_request_locally(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        self.generate_local_tokens(
            LocalGeneration {
                prompt_token_ids: request.prompt_token_ids,
                max_tokens: request.max_tokens,
                sampling: request.sampling,
                chat_sampling_metadata: request.chat_sampling_metadata,
                speculative: request.speculative,
                native_mtp_enabled: request.native_mtp_enabled,
                hook_request: request.hook_request,
                hook_runtime: request.hook_runtime,
                cancellation: request.cancellation,
                ids: request.ids,
            },
            on_token,
        )
    }

    pub(super) fn record_embedded_decode_summary(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        cache_stats: &mut GenerationCacheStats,
        decode_timer: PhaseTimer,
        summary: EmbeddedDecodeSummary<'_>,
    ) {
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.decode_token_count".to_string(),
            json!(summary.decoded_tokens),
        );
        attrs.insert(
            "llama_stage.stage0_compute_ms".to_string(),
            json!(summary.stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(summary.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_max_ms".to_string(),
            json!(summary.runtime_lock_wait_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(summary.runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_max_ms".to_string(),
            json!(summary.runtime_lock_hold_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_acquires".to_string(),
            json!(summary.runtime_lock_acquires),
        );
        attrs.insert(
            "llama_stage.decode_batch_size_max".to_string(),
            json!(summary.decode_batch_size_max),
        );
        attrs.insert(
            "llama_stage.decode_batch_wait_ms".to_string(),
            json!(summary.decode_batch_wait_ms),
        );
        attrs.insert(
            "llama_stage.forward_write_ms".to_string(),
            json!(summary.forward_write_ms),
        );
        attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(summary.activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.output_activation_bytes".to_string(),
            json!(summary.output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.forward_activation_bytes".to_string(),
            json!(summary.forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.downstream_wait_ms".to_string(),
            json!(summary.downstream_wait_ms),
        );
        request.speculative.insert_telemetry_attrs(&mut attrs);
        summary.speculative_stats.insert_attrs(&mut attrs);
        summary.native_mtp_stats.insert_attrs(&mut attrs);
        summary
            .native_mtp_counters
            .insert_summary_attrs(&mut attrs, summary.native_mtp_options);
        summary
            .verify_window_scheduler
            .insert_policy_telemetry_attrs(&mut attrs);

        cache_stats.native_mtp_stats = summary.native_mtp_stats;
        cache_stats.native_mtp_decode_telemetry = Some(NativeMtpDecodeTelemetry::new(
            summary.native_mtp_options,
            summary.native_mtp_counters,
        ));
        cache_stats.verify_window_pipeline_stats = Some(summary.verify_window_scheduler.stats());
        cache_stats.speculative_stats = Some(summary.speculative_stats.clone());
        cache_stats.predicted_ms = decode_timer.elapsed_ms();
        self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
    }

    pub(super) fn finish_embedded_generation_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        lane_pool: &PersistentStageLanePool,
        mut lane: PersistentStageLane,
        result: &OpenAiResult<()>,
        session_key: &str,
    ) -> OpenAiResult<()> {
        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(
                request.wire_dtype,
                request.ids.request_id,
                request.ids.session_id,
            ),
            request.wire_dtype,
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        self.drop_embedded_runtime_session(request, session_key);

        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (result, &stop_result) {
            (Ok(_), Ok(_)) => lane_pool.return_lane(lane),
            _ => lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            stop_result?;
        }
        Ok(())
    }

    fn drop_embedded_runtime_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
    ) {
        let lock_timer = PhaseTimer::start();
        let Ok(mut runtime) = self.runtime.lock() else {
            return;
        };
        let runtime_lock_wait_ms = lock_timer.elapsed_ms();
        let Ok(drop_stats) = runtime.drop_session_timed(session_key) else {
            return;
        };
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.session_reset_ms".to_string(),
            json!(drop_stats.reset_ms),
        );
        attrs.insert(
            "llama_stage.session_reset".to_string(),
            json!(drop_stats.reset_session),
        );
        attrs.insert(
            "llama_stage.lane_discarded".to_string(),
            json!(drop_stats.lane_discarded),
        );
        if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
            attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
        }
        Self::insert_runtime_session_stats(
            &mut attrs,
            "llama_stage.runtime_sessions_after",
            &drop_stats.stats_after,
        );
        self.telemetry
            .emit_debug("stage.openai_session_stop", attrs);
    }
}

pub(super) fn decode_uses_context_sideband(
    context_token_ids: &[i32],
    current: i32,
    sideband_capacity: usize,
) -> bool {
    context_token_ids.len() <= sideband_capacity
        && context_token_ids.last().copied() == Some(current)
}
