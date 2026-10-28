//! Bounded, unsampled decision scores from the terminal stage.

use crate::binary_transport::binary_kv::{
    BinaryKvRecordResult, accumulate_shared_prefill_tokens, add_binary_record_stats,
    maybe_record_binary_full_prefill,
};
use crate::kv_integration::KvStageIntegration;
use crate::telemetry::Telemetry;
use anyhow::{Context, Result, ensure};
use skippy_protocol::binary::{
    StageReply, StageReplyStats, StageWireMessage, WireReplyKind, recv_reply,
};
use skippy_protocol::{StageConfig, StageTopology};
use std::net::TcpStream;
use std::sync::Arc;

use crate::binary_transport::stage_execution::input_activation_frame;
use crate::binary_transport::{
    WireCondition, forwarded_stage_message, write_stage_message_conditioned,
};
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::iteration_scheduler::readout::ReadoutIteration;

pub(super) struct ReadoutStage<'a> {
    pub config: &'a StageConfig,
    pub topology: Option<&'a StageTopology>,
    pub scheduler: &'a IterationScheduler,
    pub downstream: Option<&'a mut TcpStream>,
    pub wire_condition: WireCondition,
    pub activation_width: i32,
    pub kv: Option<&'a Arc<KvStageIntegration>>,
    pub telemetry: &'a Telemetry,
}

impl ReadoutStage<'_> {
    /// Execute or forward an unsampled readout and validate terminal score evidence.
    pub fn execute(
        &mut self,
        message: &mut StageWireMessage,
        session: &str,
        mut stats: StageReplyStats,
    ) -> Result<StageReply> {
        ensure!(
            message.state.matches_kind(message.kind),
            "invalid readout stage state"
        );
        ensure!(
            message.positions.is_empty(),
            "decision readout requires text positions"
        );
        ensure!(
            self.config.downstream.is_some() == self.downstream.is_some(),
            "readout downstream is unavailable"
        );
        let position = u64::try_from(message.pos_start).context("negative readout position")?;
        let input = input_activation_frame(self.config, self.topology, message)?;
        let mut outcomes = self
            .scheduler
            .execute_readout_wave(vec![ReadoutIteration {
                session_id: session,
                position,
                tokens: &message.tokens,
                selected_tokens: &message.readout_tokens,
                input,
                cancellation: None,
            }])
            .map_err(|error| anyhow::anyhow!("decision readout: {error}"))?;
        let outcome = outcomes
            .pop()
            .context("missing decision iteration output")?;
        let record = self.record_prefix(message, session)?;
        add_binary_record_stats(&mut stats, self.config, &record);
        if let Some(downstream) = self.downstream.as_mut() {
            let forwarded = forwarded_stage_message(
                self.config,
                message,
                &outcome.output,
                self.activation_width,
            )?;
            write_stage_message_conditioned(&mut **downstream, &forwarded, self.wire_condition)?;
            let mut reply = recv_reply(&mut **downstream)?;
            validate_readout_reply(&reply, message.readout_tokens.len())?;
            stats.merge(reply.stats);
            reply.stats = stats;
            return Ok(reply);
        }
        let reply = StageReply {
            kind: WireReplyKind::TokenScores,
            predicted: -1,
            predicted_tokens: Vec::new(),
            native_mtp_draft: None,
            window: Default::default(),
            stats,
            log_probabilities: outcome.log_probabilities,
        };
        validate_readout_reply(&reply, message.readout_tokens.len())?;
        Ok(reply)
    }

    /// Record a complete question prefix without caching teacher-forced continuations.
    fn record_prefix(
        &self,
        message: &StageWireMessage,
        session: &str,
    ) -> Result<BinaryKvRecordResult> {
        let Some(kv) = self.kv else {
            return Ok(BinaryKvRecordResult::default());
        };
        // Candidate continuations are teacher forced and are never part of the
        // shared question prefix. Record once at the full prompt boundary.
        if !kv.should_record() || message.pos_start >= message.state.prompt_token_count {
            return Ok(BinaryKvRecordResult::default());
        }
        accumulate_shared_prefill_tokens(
            &kv.split_prefill_tokens,
            session,
            message.pos_start as usize,
            &message.tokens,
        );
        let tokens = kv
            .split_prefill_tokens
            .lock()
            .map_err(|_| anyhow::anyhow!("decision prefix history lock poisoned"))?
            .get(session)
            .cloned()
            .unwrap_or_default();
        if tokens.len() != message.pos_start as usize + message.tokens.len()
            || tokens.len() != message.state.prompt_token_count as usize
        {
            return Ok(BinaryKvRecordResult::default());
        }
        let cache = kv.clone();
        let config = self.config.clone();
        let telemetry = self.telemetry.clone();
        let session = session.to_owned();
        let message = message.clone();
        self.scheduler
            .execute_runtime("decision-stage-prefix-record", move |runtime| {
                Ok(maybe_record_binary_full_prefill(
                    &config,
                    runtime,
                    Some(&cache),
                    &telemetry,
                    &session,
                    &message,
                    &tokens,
                ))
            })
            .map_err(|error| anyhow::anyhow!("record decision prefix: {error}"))
    }
}

/// Reject sampled tokens, incorrect score counts and invalid log probabilities.
pub(in crate::binary_transport) fn validate_readout_reply(
    reply: &StageReply,
    count: usize,
) -> Result<()> {
    ensure!(
        reply.kind == WireReplyKind::TokenScores,
        "expected selected-token scores from terminal stage"
    );
    ensure!(
        reply.predicted == -1
            && reply.predicted_tokens.is_empty()
            && reply.native_mtp_draft.is_none(),
        "readout returned sampled tokens"
    );
    ensure!(
        reply.log_probabilities.len() == count,
        "readout score cardinality mismatch"
    );
    ensure!(
        reply
            .log_probabilities
            .iter()
            .all(|score| score.is_finite() && *score <= 0.0),
        "invalid readout log probability"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real resident-cache runtime for exercising readout reply evidence.
    fn cache_runtime_config() -> Result<StageConfig> {
        skippy_runtime::test_support::require_native_runtime();
        let path =
            std::env::var("SKIPPY_DECISION_MODEL").context("SKIPPY_DECISION_MODEL is required")?;
        let layers = skippy_runtime::ModelInfo::open(&path)?
            .tensors()?
            .iter()
            .filter_map(|tensor| tensor.layer_index)
            .max()
            .context("model needs transformer layers")?
            + 1;
        Ok(StageConfig {
            model_id: "readout-record-evidence".into(),
            model_path: Some(path),
            stage_id: "stage-2".into(),
            stage_index: 2,
            layer_end: layers,
            ctx_size: 512,
            lane_count: 1,
            n_batch: Some(128),
            n_ubatch: Some(128),
            n_gpu_layers: 0,
            kv_offload: Some(false),
            op_offload: Some(false),
            selected_device: Some(skippy_protocol::StageDevice {
                backend_device: "CPU".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            native_mtp_enabled: false,
            load_mode: skippy_protocol::LoadMode::RuntimeSlice,
            kv_cache: Some(serde_json::from_value(serde_json::json!({
                "mode": "lookup-record", "min_tokens": 32,
                "shared_prefix_stride_tokens": 32, "shared_prefix_record_limit": 4,
                "max_entries": 8
            }))?),
            ..StageConfig::default()
        })
    }

    /// Readout wire replies report actual recording, only at the prompt boundary.
    #[test]
    #[ignore = "requires SKIPPY_DECISION_MODEL: a real resident-cache-capable GGUF"]
    fn readout_reply_preserves_prefix_recording_evidence() -> Result<()> {
        use crate::runtime_state::{
            load_runtime, loaded_memory_cache_capabilities, loaded_model_state_kind,
        };
        use skippy_protocol::binary::{StageStateHeader, WireMessageKind};

        let config = cache_runtime_config()?;
        let runtime = load_runtime(&config)?.context("loaded runtime")?;
        let kv = Arc::new(
            KvStageIntegration::from_loaded_model_with_l3_manager(
                &config,
                loaded_model_state_kind(Some(&runtime)),
                loaded_memory_cache_capabilities(Some(&runtime)),
                None,
                None,
            )?
            .context("resident prefix cache")?,
        );
        let telemetry = Telemetry::new(
            None,
            1,
            config.clone(),
            crate::telemetry::TelemetryLevel::Off,
        );
        let scheduler =
            IterationScheduler::new(runtime, &config, 1, true, None, telemetry.clone())?;
        let mut stage = ReadoutStage {
            config: &config,
            topology: None,
            scheduler: &scheduler,
            downstream: None,
            wire_condition: WireCondition::new(0.0, None)?,
            activation_width: 0,
            kv: Some(&kv),
            telemetry: &telemetry,
        };
        let mut message = StageWireMessage::stop_with_identity(11, 13);
        message.kind = WireMessageKind::PrefillReadout;
        message.state = StageStateHeader::new(message.kind);
        message.state.prompt_token_count = 64;
        message.pos_start = 0;
        message.token_count = 32;
        message.tokens = vec![1; 32];
        message.readout_tokens = vec![2];
        let partial = stage.execute(&mut message, "readout-record", Default::default())?;
        assert_eq!(partial.stats.kv_recorded_pages, 0);
        assert_eq!(partial.stats.kv_record_stage_mask, 0);

        message.pos_start = 32;
        let pending = StageReplyStats {
            kv_recorded_pages: 7,
            kv_record_stage_mask: 1,
            ..Default::default()
        };
        let recorded = stage.execute(&mut message, "readout-record", pending)?;
        assert!(recorded.stats.kv_recorded_pages > pending.kv_recorded_pages);
        assert_eq!(
            recorded.stats.kv_record_stage_mask,
            1 | (1 << config.stage_index)
        );

        message.pos_start = 64;
        message.token_count = 1;
        message.tokens.truncate(1);
        let continuation = stage.execute(&mut message, "readout-record", Default::default())?;
        assert_eq!(continuation.stats.kv_recorded_pages, 0);
        assert_eq!(continuation.stats.kv_record_stage_mask, 0);
        Ok(())
    }

    /// Construct a valid terminal score reply for evidence-validation tests.
    fn reply() -> StageReply {
        StageReply {
            kind: WireReplyKind::TokenScores,
            predicted: -1,
            predicted_tokens: Vec::new(),
            native_mtp_draft: None,
            window: Default::default(),
            stats: Default::default(),
            log_probabilities: vec![-1.0, -2.0],
        }
    }

    /// Terminal readout validation rejects sampling and malformed score evidence.
    #[test]
    fn terminal_evidence_rejects_sampling_wrong_cardinality_and_invalid_scores() {
        assert!(validate_readout_reply(&reply(), 2).is_ok());
        assert!(validate_readout_reply(&reply(), 1).is_err());
        for defect in 0..6 {
            let mut reply = reply();
            match defect {
                0 => reply.kind = WireReplyKind::Ack,
                1 => reply.predicted = 42,
                2 => reply.predicted_tokens.push(42),
                3 => reply.log_probabilities[0] = f32::NAN,
                4 => reply.log_probabilities[0] = f32::NEG_INFINITY,
                _ => reply.log_probabilities[0] = 0.5,
            }
            assert!(
                validate_readout_reply(&reply, 2).is_err(),
                "defect {defect}"
            );
        }
    }
}
