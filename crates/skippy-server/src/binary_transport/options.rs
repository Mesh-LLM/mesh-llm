use std::{net::SocketAddr, path::PathBuf};

use skippy_protocol::{StageConfig, StageTopology};
use skippy_runtime::MtpSource;

use crate::{frontend::SpeculativeDecodeConfig, telemetry::TelemetryLevel};

use super::WireCondition;

#[derive(Clone)]
pub struct BinaryStageOptions {
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
    /// Whether the iteration scheduler may serve multiple active lanes.
    ///
    /// Binary stages launched by the standalone CLI retain the historical
    /// enabled default. Mesh-launched stages receive the resolved value from
    /// the stage-control load request.
    pub continuous_batching: bool,
    pub openai: Option<EmbeddedOpenAiStageOptions>,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiStageOptions {
    pub bind_addr: SocketAddr,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: usize,
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
