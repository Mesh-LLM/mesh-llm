use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result, bail};
use skippy_protocol::{StageConfig, StageTopology, binary::WireActivationDType};

use crate::{
    cli::ServeBinaryArgs, config::load_json, frontend::SpeculativeDecodeConfig,
    telemetry::TelemetryLevel,
};

use super::WireCondition;

#[derive(Clone)]
pub struct BinaryStageOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub bind_addr: SocketAddr,
    pub activation_width: i32,
    pub wire_dtype: WireActivationDType,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
    pub max_inflight: usize,
    pub reply_credit_limit: Option<usize>,
    pub async_prefill_forward: bool,
    pub downstream_wire_condition: WireCondition,
    pub downstream_connect_timeout_secs: u64,
    pub native_mtp_enabled: bool,
    pub openai: Option<EmbeddedOpenAiStageOptions>,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiStageOptions {
    pub bind_addr: SocketAddr,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub generation_concurrency: usize,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub native_mtp_max_tokens: usize,
    pub native_mtp_min_tokens: usize,
    pub speculative: SpeculativeDecodeConfig,
}

impl BinaryStageOptions {
    pub fn from_cli_args(args: ServeBinaryArgs) -> Result<Self> {
        if args.activation_width <= 0 {
            bail!("activation_width must be greater than zero");
        }
        if args.openai_generation_concurrency == 0 {
            bail!("--openai-generation-concurrency must be greater than zero");
        }
        if args.openai_prefill_chunk_size == 0 {
            bail!("--openai-prefill-chunk-size must be greater than zero");
        }
        let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
        let downstream_wire_condition =
            WireCondition::new(args.downstream_wire_delay_ms, args.downstream_wire_mbps)?;
        let config = load_json::<StageConfig>(&args.config)
            .with_context(|| format!("load stage config {}", args.config.display()))?;
        let topology = match args.topology.as_ref() {
            Some(path) => Some(
                load_json::<StageTopology>(path)
                    .with_context(|| format!("load topology {}", path.display()))?,
            ),
            None => None,
        };
        let bind_addr = args.bind_addr.unwrap_or(config.bind_addr.parse()?);
        let openai_speculative: SpeculativeDecodeConfig = args
            .openai_speculative_config
            .as_ref()
            .map(load_json)
            .transpose()
            .context("load --openai-speculative-config")?
            .unwrap_or_default();
        openai_speculative.validate()?;
        let openai = args
            .openai_bind_addr
            .map(|bind_addr| EmbeddedOpenAiStageOptions {
                bind_addr,
                model_id: args.openai_model_id,
                default_max_tokens: args.openai_default_max_tokens,
                generation_concurrency: args.openai_generation_concurrency,
                prefill_chunk_size: args.openai_prefill_chunk_size,
                prefill_chunk_policy: args.openai_prefill_chunk_policy,
                prefill_chunk_schedule: args.openai_prefill_chunk_schedule,
                prefill_adaptive_start: args.openai_prefill_adaptive_start,
                prefill_adaptive_step: args.openai_prefill_adaptive_step,
                prefill_adaptive_max: args.openai_prefill_adaptive_max,
                draft_model_path: args.openai_draft_model_path,
                speculative_window: args.openai_speculative_window,
                adaptive_speculative_window: args.openai_adaptive_speculative_window,
                draft_n_gpu_layers: args.openai_draft_n_gpu_layers,
                native_mtp_max_tokens: 3,
                native_mtp_min_tokens: 0,
                speculative: openai_speculative,
            });
        let native_mtp_enabled = config.native_mtp_enabled;
        Ok(Self {
            config,
            topology,
            bind_addr,
            activation_width: args.activation_width,
            wire_dtype,
            metrics_otlp_grpc: args.metrics_otlp_grpc,
            telemetry_queue_capacity: args.telemetry_queue_capacity,
            telemetry_level: args.telemetry_level,
            max_inflight: args.max_inflight,
            reply_credit_limit: args.reply_credit_limit,
            async_prefill_forward: args.async_prefill_forward || !args.no_async_prefill_forward,
            downstream_wire_condition,
            downstream_connect_timeout_secs: args.downstream_connect_timeout_secs,
            native_mtp_enabled,
            openai,
        })
    }
}

pub fn parse_wire_dtype(value: &str) -> Result<WireActivationDType> {
    match value {
        "fp32" | "f32" => Ok(WireActivationDType::F32),
        "fp16" | "f16" => Ok(WireActivationDType::F16),
        "q8" | "int8" | "i8" => Ok(WireActivationDType::Q8),
        _ => bail!("unsupported activation wire dtype {value}"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;
    use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};

    use super::*;
    use crate::{
        cli::{Cli, Command},
        frontend::{
            NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
            VerifyWindowConfig,
        },
    };

    fn stage_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 4,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn cache_composite_plan() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            requested_strategy: "mtp-cache".to_string(),
            effective_strategy: "native-mtp+ngram-cache".to_string(),
            native_mtp: NativeMtpProposalConfig {
                enabled: true,
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: 4,
                max_proposal_tokens: 6,
            }),
            extension: Some(NgramExtensionConfig { max_tokens: 6 }),
            verify_window: VerifyWindowConfig {
                min_tokens: 1,
                max_tokens: 6,
                pipeline_depth: 2,
            },
        }
    }

    #[test]
    fn typed_speculative_plan_reaches_embedded_stage_without_policy_merging() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let stage_path = dir.path().join("stage.json");
        let plan_path = dir.path().join("speculative.json");
        fs::write(
            &stage_path,
            serde_json::to_vec(&stage_config()).expect("serialize stage config"),
        )
        .expect("write stage config");
        let expected = cache_composite_plan();
        fs::write(
            &plan_path,
            serde_json::to_vec(&expected).expect("serialize speculative config"),
        )
        .expect("write speculative config");

        let cli = Cli::try_parse_from([
            "skippy-server",
            "serve-binary",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--activation-width",
            "2048",
            "--openai-bind-addr",
            "127.0.0.1:9337",
            "--openai-speculative-config",
            plan_path.to_str().expect("UTF-8 plan path"),
        ])
        .expect("parse binary stage CLI");
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };

        let options = BinaryStageOptions::from_cli_args(args).expect("resolve binary stage");
        let openai = options.openai.expect("embedded OpenAI configuration");

        assert!(options.native_mtp_enabled);
        assert_eq!(openai.speculative, expected);
    }

    #[test]
    fn cache_composite_plan_is_json_stable_for_stage_handoff() {
        let plan = cache_composite_plan();
        let json = serde_json::to_value(&plan).expect("serialize speculative plan");

        assert_eq!(json["ngram"]["min_ngram"], 2);
        assert_eq!(json["verify_window"]["pipeline_depth"], 2);
    }
}
