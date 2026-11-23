use std::{net::SocketAddr, path::PathBuf};

use skippy_protocol::{StageConfig, StageTopology};
use skippy_runtime::MtpSource;

use crate::{frontend::SpeculativeDecodeConfig, telemetry::TelemetryLevel};

use super::WireCondition;

#[derive(Clone)]
pub struct BinaryStageOptions {
    pub tuning: crate::settings::ServingTuning,
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub bind_addr: SocketAddr,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
    pub max_inflight: usize,
    pub reply_credit_limit: Option<usize>,
    pub async_prefill_forward: bool,
    pub downstream_wire_condition: WireCondition,
    pub downstream_connect_timeout_secs: u64,
    pub native_mtp_enabled: bool,
    /// Planned `throughput.last_stage_decode_batch`, delivered on the stage
    /// load. `None` means the coordinator said nothing, which stays unbatched.
    ///
    /// Only the final stage acts on it, and only without native MTP: the
    /// batched path produces no MTP drafts.
    pub last_stage_decode_batch: Option<bool>,
    /// Whether the iteration scheduler may serve multiple active lanes.
    ///
    /// Binary stages launched by the standalone CLI retain the historical
    /// enabled default. Mesh-launched stages receive the resolved value from
    /// the stage-control load request.
    pub continuous_batching: bool,
    pub openai: Option<EmbeddedOpenAiStageOptions>,
    /// Receives this stage's runtime compute time, for auto-balance
    /// split placement. `None` skips the accounting.
    pub compute_meter: Option<std::sync::Arc<crate::compute_meter::StageComputeMeter>>,
    /// Optional node-scoped durable cache manager supplied by the embedding host.
    pub l3_manager: Option<skippy_cache::L3CacheManager>,
    /// This stage's part in a split DFlash plan.
    pub split_dflash: Option<crate::split_dflash::SplitDFlashStage>,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiStageOptions {
    pub bind_addr: SocketAddr,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: usize,
    pub pipeline_decode_groups: Option<usize>,
    pub adaptive_generation_min_concurrency: Option<usize>,
    pub generation_queue_capacity: usize,
    pub generation_admission_timeout_secs: u64,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub prefill_adaptive_target_ms: f64,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub native_mtp_draft_model_path: Option<PathBuf>,
    pub native_mtp_max_tokens: usize,
    pub native_mtp_min_tokens: usize,
    pub speculative: SpeculativeDecodeConfig,
}

impl BinaryStageOptions {
    /// Size known plans before opening the stage. A transport-only stage has
    /// no proposal-window contract, so retain the historical two-input floor;
    /// larger incoming windows still use checkpoint/replay fallback.
    pub(super) fn recurrent_rollback_tokens(&self) -> u32 {
        let planned = self.openai.as_ref().map_or(2, |openai| {
            openai.speculative.recurrent_rollback_tokens(
                (self.resolved_mtp_source() != MtpSource::Disabled)
                    .then_some(openai.native_mtp_max_tokens),
                openai
                    .draft_model_path
                    .as_ref()
                    .map(|_| openai.speculative_window),
            )
        });
        planned.max(self.split_dflash.as_ref().map_or(
            0,
            crate::split_dflash::SplitDFlashStage::recurrent_rollback_tokens,
        ))
    }

    /// Resolve the MTP source from the admitted stage and optional OpenAI settings.
    pub fn resolved_mtp_source(&self) -> MtpSource {
        if !self.native_mtp_enabled || !self.config.native_mtp_enabled {
            return MtpSource::Disabled;
        }
        let Some(openai) = self.openai.as_ref() else {
            return MtpSource::Integrated;
        };
        if !openai.speculative.native_mtp.enabled {
            return MtpSource::Disabled;
        }
        if openai.native_mtp_draft_model_path.is_some() {
            MtpSource::External
        } else {
            MtpSource::Integrated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> BinaryStageOptions {
        BinaryStageOptions {
            tuning: Default::default(),
            config: StageConfig::default(),
            topology: None,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 0,
            telemetry_level: TelemetryLevel::Off,
            max_inflight: 1,
            reply_credit_limit: None,
            async_prefill_forward: false,
            downstream_wire_condition: WireCondition::new(0.0, None).unwrap(),
            downstream_connect_timeout_secs: 1,
            native_mtp_enabled: false,
            last_stage_decode_batch: None,
            continuous_batching: true,
            openai: None,
            compute_meter: None,
            l3_manager: None,
            split_dflash: None,
        }
    }

    #[test]
    fn unknown_binary_verifiers_keep_a_compatibility_floor() {
        let mut options = options();
        assert_eq!(options.recurrent_rollback_tokens(), 2);
        options.split_dflash = Some(crate::split_dflash::SplitDFlashStage {
            max_draft_tokens: Some(7),
            ..Default::default()
        });
        assert_eq!(options.recurrent_rollback_tokens(), 8);
    }
}
