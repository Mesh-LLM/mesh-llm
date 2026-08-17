use crate::binary_transport::stage_execution::binary_message_request_id;
use crate::binary_transport::stage_execution::estimated_kv_tokens_after;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use serde_json::json;
use skippy_metrics::attr;
use skippy_metrics::metric;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;

#[derive(Default)]
pub(super) struct BinaryRequestSummary {
    request_id: Option<String>,
    prompt_index: i32,
    prompt_token_count: i32,
    pub(super) message_count: usize,
    prefill_message_count: usize,
    decode_message_count: usize,
    prefill_token_count: i64,
    decode_token_count: i64,
    compute_ms: f64,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
    upstream_reply_ms: f64,
    message_elapsed_ms: f64,
    input_activation_decode_ms: f64,
    forward_activation_encode_ms: f64,
    runtime_lock_hold_ms: f64,
    input_activation_bytes: usize,
    output_activation_bytes: usize,
    max_input_activation_bytes: usize,
    max_output_activation_bytes: usize,
    kv_tokens_after_max: i64,
    kv_token_layer_cells_max: i64,
    prefill_credit_limit: usize,
    prefill_credit_wait_count: usize,
    prefill_deferred_replies_drained: usize,
    prefill_pending_replies_max: usize,
    session_auto_align_count: usize,
    session_auto_align_ms: f64,
    session_auto_align_trimmed_tokens: u64,
    verify_window_count: usize,
    verify_window_session_auto_align_count: usize,
    verify_window_session_auto_align_ms: f64,
    verify_window_session_auto_align_trimmed_tokens: u64,
    verify_window_token_count: u64,
    verify_window_max_tokens: u64,
    verify_window_compute_ms: f64,
    verify_window_input_activation_decode_ms: f64,
    verify_window_runtime_lock_hold_ms: f64,
    verify_window_upstream_reply_ms: f64,
    verify_window_pre_compute_ms: f64,
    verify_window_post_compute_ms: f64,
    verify_window_pre_reply_ms: f64,
    verify_window_after_reply_ms: f64,
    verify_window_upstream_message_wait_ms: f64,
    reply_stats: StageReplyStats,
}

pub(super) struct BinaryMessageObservation<'a> {
    pub(super) config: &'a StageConfig,
    pub(super) message: &'a StageWireMessage,
    pub(super) reply_stats: StageReplyStats,
    pub(super) compute_ms: f64,
    pub(super) forward_write_ms: f64,
    pub(super) downstream_wait_ms: f64,
    pub(super) upstream_reply_ms: f64,
    pub(super) message_elapsed_ms: f64,
    pub(super) input_activation_decode_ms: f64,
    pub(super) forward_activation_encode_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) input_activation_bytes: usize,
    pub(super) output_activation_bytes: usize,
    pub(super) prefill_credit_limit: usize,
    pub(super) pending_prefill_replies_before: usize,
    pub(super) pending_prefill_replies_after: usize,
    pub(super) credit_wait_count: usize,
    pub(super) deferred_prefill_replies_drained: usize,
    pub(super) session_auto_align_count: usize,
    pub(super) session_auto_align_ms: f64,
    pub(super) session_auto_align_trimmed_tokens: u64,
    pub(super) verify_window_pre_compute_ms: f64,
    pub(super) verify_window_post_compute_ms: f64,
    pub(super) verify_window_pre_reply_ms: f64,
    pub(super) verify_window_after_reply_ms: f64,
    pub(super) upstream_message_wait_ms: f64,
}

impl BinaryRequestSummary {
    pub(super) fn observe(&mut self, observation: BinaryMessageObservation<'_>) {
        let message = observation.message;
        if self.message_count == 0 {
            self.request_id = Some(binary_message_request_id(message));
            self.prompt_index = message.state.seq_id;
            self.prompt_token_count = message.state.prompt_token_count;
        }

        self.message_count += 1;
        if message.kind.is_prefill() {
            self.prefill_message_count += 1;
            self.prefill_token_count += i64::from(message.token_count.max(0));
        } else if message.kind.requires_predicted_reply() {
            self.decode_message_count += 1;
            self.decode_token_count += i64::from(message.token_count.max(0));
        }

        self.compute_ms += observation.compute_ms;
        self.forward_write_ms += observation.forward_write_ms;
        self.downstream_wait_ms += observation.downstream_wait_ms;
        self.upstream_reply_ms += observation.upstream_reply_ms;
        self.message_elapsed_ms += observation.message_elapsed_ms;
        self.input_activation_decode_ms += observation.input_activation_decode_ms;
        self.forward_activation_encode_ms += observation.forward_activation_encode_ms;
        self.runtime_lock_hold_ms += observation.runtime_lock_hold_ms;
        self.input_activation_bytes += observation.input_activation_bytes;
        self.output_activation_bytes += observation.output_activation_bytes;
        self.max_input_activation_bytes = self
            .max_input_activation_bytes
            .max(observation.input_activation_bytes);
        self.max_output_activation_bytes = self
            .max_output_activation_bytes
            .max(observation.output_activation_bytes);

        let layer_count = i64::from(
            observation
                .config
                .layer_end
                .saturating_sub(observation.config.layer_start),
        );
        let kv_tokens_after = estimated_kv_tokens_after(message);
        self.kv_tokens_after_max = self.kv_tokens_after_max.max(kv_tokens_after);
        self.kv_token_layer_cells_max = self
            .kv_token_layer_cells_max
            .max(kv_tokens_after.saturating_mul(layer_count));
        self.prefill_credit_limit = observation.prefill_credit_limit;
        self.prefill_credit_wait_count += observation.credit_wait_count;
        self.prefill_deferred_replies_drained += observation.deferred_prefill_replies_drained;
        self.prefill_pending_replies_max = self
            .prefill_pending_replies_max
            .max(observation.pending_prefill_replies_before)
            .max(observation.pending_prefill_replies_after);
        self.session_auto_align_count += observation.session_auto_align_count;
        self.session_auto_align_ms += observation.session_auto_align_ms;
        self.session_auto_align_trimmed_tokens = self
            .session_auto_align_trimmed_tokens
            .saturating_add(observation.session_auto_align_trimmed_tokens);
        if message.kind == WireMessageKind::VerifyWindow {
            let token_count = message.token_count.max(0) as u64;
            self.verify_window_count += 1;
            self.verify_window_token_count =
                self.verify_window_token_count.saturating_add(token_count);
            self.verify_window_max_tokens = self.verify_window_max_tokens.max(token_count);
            self.verify_window_session_auto_align_count += observation.session_auto_align_count;
            self.verify_window_session_auto_align_ms += observation.session_auto_align_ms;
            self.verify_window_session_auto_align_trimmed_tokens = self
                .verify_window_session_auto_align_trimmed_tokens
                .saturating_add(observation.session_auto_align_trimmed_tokens);
            self.verify_window_compute_ms += observation.compute_ms;
            self.verify_window_input_activation_decode_ms += observation.input_activation_decode_ms;
            self.verify_window_runtime_lock_hold_ms += observation.runtime_lock_hold_ms;
            self.verify_window_upstream_reply_ms += observation.upstream_reply_ms;
            self.verify_window_pre_compute_ms += observation.verify_window_pre_compute_ms;
            self.verify_window_post_compute_ms += observation.verify_window_post_compute_ms;
            self.verify_window_pre_reply_ms += observation.verify_window_pre_reply_ms;
            self.verify_window_after_reply_ms += observation.verify_window_after_reply_ms;
            self.verify_window_upstream_message_wait_ms += observation.upstream_message_wait_ms;
        }
        self.reply_stats.merge(observation.reply_stats);
    }

    pub(super) fn emit(&self, telemetry: &Telemetry, config: &StageConfig, session_id: u64) {
        if self.message_count == 0 || !telemetry.is_enabled() {
            return;
        }
        let mut attrs = lifecycle_attrs(config);
        attrs.insert(attr::SESSION_ID.to_string(), json!(session_id.to_string()));
        if let Some(request_id) = self.request_id.as_ref() {
            attrs.insert(attr::REQUEST_ID.to_string(), json!(request_id));
        }
        attrs.insert(attr::PROMPT_INDEX.to_string(), json!(self.prompt_index));
        attrs.insert(
            attr::PROMPT_TOKEN_COUNT.to_string(),
            json!(self.prompt_token_count),
        );
        attrs.insert(
            "skippy.message_count".to_string(),
            json!(self.message_count),
        );
        attrs.insert(
            "skippy.prefill_message_count".to_string(),
            json!(self.prefill_message_count),
        );
        attrs.insert(
            "skippy.decode_message_count".to_string(),
            json!(self.decode_message_count),
        );
        attrs.insert(
            "skippy.prefill_token_count".to_string(),
            json!(self.prefill_token_count),
        );
        attrs.insert(
            "skippy.decode_token_count".to_string(),
            json!(self.decode_token_count),
        );
        attrs.insert("skippy.compute_ms".to_string(), json!(self.compute_ms));
        attrs.insert(
            "skippy.forward_write_ms".to_string(),
            json!(self.forward_write_ms),
        );
        attrs.insert(
            "skippy.downstream_wait_ms".to_string(),
            json!(self.downstream_wait_ms),
        );
        attrs.insert(
            "skippy.upstream_reply_ms".to_string(),
            json!(self.upstream_reply_ms),
        );
        attrs.insert(
            "skippy.message_elapsed_ms".to_string(),
            json!(self.message_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.input_activation_decode_ms".to_string(),
            json!(self.input_activation_decode_ms),
        );
        attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(self.forward_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(self.runtime_lock_hold_ms),
        );
        attrs.insert(
            "skippy.input_activation_bytes".to_string(),
            json!(self.input_activation_bytes),
        );
        attrs.insert(
            "skippy.output_activation_bytes".to_string(),
            json!(self.output_activation_bytes),
        );
        attrs.insert(
            "skippy.max_input_activation_bytes".to_string(),
            json!(self.max_input_activation_bytes),
        );
        attrs.insert(
            "skippy.max_output_activation_bytes".to_string(),
            json!(self.max_output_activation_bytes),
        );
        attrs.insert(
            "skippy.kv_tokens_after".to_string(),
            json!(self.kv_tokens_after_max),
        );
        attrs.insert(
            "skippy.kv_token_layer_cells".to_string(),
            json!(self.kv_token_layer_cells_max),
        );
        attrs.insert(
            "skippy.prefill_credit_limit".to_string(),
            json!(self.prefill_credit_limit),
        );
        attrs.insert(
            "skippy.prefill_credit_wait_count".to_string(),
            json!(self.prefill_credit_wait_count),
        );
        attrs.insert(
            "skippy.prefill_deferred_replies_drained".to_string(),
            json!(self.prefill_deferred_replies_drained),
        );
        attrs.insert(
            "skippy.prefill_pending_replies_max".to_string(),
            json!(self.prefill_pending_replies_max),
        );
        attrs.insert(
            "skippy.session_auto_align_count".to_string(),
            json!(self.session_auto_align_count),
        );
        attrs.insert(
            "skippy.session_auto_align_ms".to_string(),
            json!(self.session_auto_align_ms),
        );
        attrs.insert(
            "skippy.session_auto_align_trimmed_tokens".to_string(),
            json!(self.session_auto_align_trimmed_tokens),
        );
        if self.session_auto_align_count > 0 {
            attrs.insert(
                "skippy.session_auto_align_ms_avg".to_string(),
                json!(self.session_auto_align_ms / self.session_auto_align_count as f64),
            );
        }
        attrs.insert(
            "skippy.verify_window_count".to_string(),
            json!(self.verify_window_count),
        );
        attrs.insert(
            "skippy.verify_window_token_count".to_string(),
            json!(self.verify_window_token_count),
        );
        attrs.insert(
            "skippy.verify_window_max_tokens".to_string(),
            json!(self.verify_window_max_tokens),
        );
        attrs.insert(
            "skippy.verify_window_session_auto_align_count".to_string(),
            json!(self.verify_window_session_auto_align_count),
        );
        attrs.insert(
            "skippy.verify_window_session_auto_align_ms".to_string(),
            json!(self.verify_window_session_auto_align_ms),
        );
        attrs.insert(
            "skippy.verify_window_session_auto_align_trimmed_tokens".to_string(),
            json!(self.verify_window_session_auto_align_trimmed_tokens),
        );
        if self.verify_window_session_auto_align_count > 0 {
            attrs.insert(
                "skippy.verify_window_session_auto_align_ms_avg".to_string(),
                json!(
                    self.verify_window_session_auto_align_ms
                        / self.verify_window_session_auto_align_count as f64
                ),
            );
        }
        attrs.insert(
            "skippy.verify_window_pre_compute_ms".to_string(),
            json!(self.verify_window_pre_compute_ms),
        );
        attrs.insert(
            "skippy.verify_window_compute_ms".to_string(),
            json!(self.verify_window_compute_ms),
        );
        attrs.insert(
            "skippy.verify_window_input_activation_decode_ms".to_string(),
            json!(self.verify_window_input_activation_decode_ms),
        );
        attrs.insert(
            "skippy.verify_window_runtime_lock_hold_ms".to_string(),
            json!(self.verify_window_runtime_lock_hold_ms),
        );
        attrs.insert(
            "skippy.verify_window_upstream_reply_ms".to_string(),
            json!(self.verify_window_upstream_reply_ms),
        );
        attrs.insert(
            "skippy.verify_window_post_compute_ms".to_string(),
            json!(self.verify_window_post_compute_ms),
        );
        attrs.insert(
            "skippy.verify_window_pre_reply_ms".to_string(),
            json!(self.verify_window_pre_reply_ms),
        );
        attrs.insert(
            "skippy.verify_window_after_reply_ms".to_string(),
            json!(self.verify_window_after_reply_ms),
        );
        attrs.insert(
            "skippy.verify_window_upstream_message_wait_ms".to_string(),
            json!(self.verify_window_upstream_message_wait_ms),
        );
        if self.verify_window_count > 0 {
            let verify_window_count = self.verify_window_count as f64;
            attrs.insert(
                "skippy.verify_window_pre_compute_ms_avg".to_string(),
                json!(self.verify_window_pre_compute_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_compute_ms_avg".to_string(),
                json!(self.verify_window_compute_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_input_activation_decode_ms_avg".to_string(),
                json!(self.verify_window_input_activation_decode_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_runtime_lock_hold_ms_avg".to_string(),
                json!(self.verify_window_runtime_lock_hold_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_upstream_reply_ms_avg".to_string(),
                json!(self.verify_window_upstream_reply_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_tokens_avg".to_string(),
                json!(self.verify_window_token_count as f64 / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_post_compute_ms_avg".to_string(),
                json!(self.verify_window_post_compute_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_pre_reply_ms_avg".to_string(),
                json!(self.verify_window_pre_reply_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_after_reply_ms_avg".to_string(),
                json!(self.verify_window_after_reply_ms / verify_window_count),
            );
            attrs.insert(
                "skippy.verify_window_upstream_message_wait_ms_avg".to_string(),
                json!(self.verify_window_upstream_message_wait_ms / verify_window_count),
            );
        }
        let lookups = self.reply_stats.kv_lookup_hits + self.reply_stats.kv_lookup_misses;
        let hit_rate = if lookups > 0 {
            self.reply_stats.kv_lookup_hits as f64 / lookups as f64
        } else {
            0.0
        };
        attrs.insert(
            metric::KV_LOOKUP_REQUESTS.to_string(),
            json!(lookups.max(0)),
        );
        attrs.insert(
            metric::KV_LOOKUP_HITS.to_string(),
            json!(self.reply_stats.kv_lookup_hits),
        );
        attrs.insert(
            metric::KV_LOOKUP_MISSES.to_string(),
            json!(self.reply_stats.kv_lookup_misses),
        );
        attrs.insert("skippy.kv.lookup_hit_rate".to_string(), json!(hit_rate));
        attrs.insert(
            "skippy.kv.lookup_errors".to_string(),
            json!(self.reply_stats.kv_lookup_errors),
        );
        attrs.insert(
            metric::KV_IMPORTED_PAGES.to_string(),
            json!(self.reply_stats.kv_imported_pages),
        );
        attrs.insert(
            "skippy.kv.imported_tokens".to_string(),
            json!(self.reply_stats.kv_imported_tokens),
        );
        attrs.insert(
            metric::KV_COMMITTED_PAGES.to_string(),
            json!(self.reply_stats.kv_recorded_pages),
        );
        attrs.insert(
            "skippy.kv.recorded_bytes".to_string(),
            json!(self.reply_stats.kv_recorded_bytes),
        );
        attrs.insert(
            "skippy.kv.hit_stage_mask".to_string(),
            json!(self.reply_stats.kv_hit_stage_mask),
        );
        attrs.insert(
            "skippy.kv.record_stage_mask".to_string(),
            json!(self.reply_stats.kv_record_stage_mask),
        );
        telemetry.emit("stage.binary_request_summary", attrs);
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryMessageObservation, BinaryRequestSummary};
    use crate::binary_transport::stage_execution::prefix_cache_test_config;
    use skippy_protocol::StageConfig;
    use skippy_protocol::binary::{
        StageReplyStats, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    };

    #[test]
    fn request_summary_tracks_verify_window_compute_ms() {
        let config = prefix_cache_test_config();
        let mut summary = BinaryRequestSummary::default();
        let verify = test_message(WireMessageKind::VerifyWindow, 2);
        let decode = test_message(WireMessageKind::DecodeEmbd, 1);

        summary.observe(summary_observation(&config, &verify, 12.5));
        summary.observe(summary_observation(&config, &decode, 7.0));

        assert_eq!(summary.verify_window_count, 1);
        assert_eq!(summary.verify_window_token_count, 2);
        assert_eq!(summary.verify_window_max_tokens, 2);
        assert_eq!(summary.verify_window_compute_ms, 12.5);
        assert_eq!(summary.verify_window_input_activation_decode_ms, 1.25);
        assert_eq!(summary.verify_window_runtime_lock_hold_ms, 2.5);
        assert_eq!(summary.verify_window_upstream_reply_ms, 0.75);
        assert_eq!(summary.compute_ms, 19.5);
        assert_eq!(summary.input_activation_decode_ms, 2.5);
        assert_eq!(summary.runtime_lock_hold_ms, 5.0);
        assert_eq!(summary.upstream_reply_ms, 1.5);
    }

    #[test]
    fn request_summary_tracks_auto_align_totals() {
        let config = prefix_cache_test_config();
        let mut summary = BinaryRequestSummary::default();
        let verify = test_message(WireMessageKind::VerifyWindow, 2);
        let decode = test_message(WireMessageKind::DecodeEmbd, 1);

        let mut verify_observation = summary_observation(&config, &verify, 12.5);
        verify_observation.session_auto_align_count = 1;
        verify_observation.session_auto_align_ms = 0.75;
        verify_observation.session_auto_align_trimmed_tokens = 1;
        summary.observe(verify_observation);

        let mut decode_observation = summary_observation(&config, &decode, 7.0);
        decode_observation.session_auto_align_count = 1;
        decode_observation.session_auto_align_ms = 1.25;
        decode_observation.session_auto_align_trimmed_tokens = 2;
        summary.observe(decode_observation);

        assert_eq!(summary.session_auto_align_count, 2);
        assert_eq!(summary.session_auto_align_ms, 2.0);
        assert_eq!(summary.session_auto_align_trimmed_tokens, 3);
        assert_eq!(summary.verify_window_session_auto_align_count, 1);
        assert_eq!(summary.verify_window_session_auto_align_ms, 0.75);
        assert_eq!(summary.verify_window_session_auto_align_trimmed_tokens, 1);
    }
    fn test_message(kind: WireMessageKind, token_count: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count,
            state: StageStateHeader::new(kind, WireActivationDType::F16),
            request_id: 11,
            session_id: 13,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn summary_observation<'a>(
        config: &'a StageConfig,
        message: &'a StageWireMessage,
        compute_ms: f64,
    ) -> BinaryMessageObservation<'a> {
        BinaryMessageObservation {
            config,
            message,
            reply_stats: StageReplyStats::default(),
            compute_ms,
            forward_write_ms: 0.0,
            downstream_wait_ms: 0.0,
            upstream_reply_ms: 0.75,
            message_elapsed_ms: compute_ms,
            input_activation_decode_ms: 1.25,
            forward_activation_encode_ms: 0.0,
            runtime_lock_hold_ms: 2.5,
            input_activation_bytes: 0,
            output_activation_bytes: 0,
            prefill_credit_limit: 0,
            pending_prefill_replies_before: 0,
            pending_prefill_replies_after: 0,
            credit_wait_count: 0,
            deferred_prefill_replies_drained: 0,
            session_auto_align_count: 0,
            session_auto_align_ms: 0.0,
            session_auto_align_trimmed_tokens: 0,
            verify_window_pre_compute_ms: 0.25,
            verify_window_post_compute_ms: 0.5,
            verify_window_pre_reply_ms: 0.0,
            verify_window_after_reply_ms: 0.0,
            upstream_message_wait_ms: 0.0,
        }
    }
}
