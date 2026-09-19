use crate::cli::{ServeArgs, ServeBinaryArgs};
use anyhow::{Context, Result, bail};
use skippy_config::load_json;
use skippy_protocol::{StageConfig, StageTopology};
use skippy_server::{
    binary_transport::{BinaryStageOptions, EmbeddedOpenAiStageOptions, WireCondition},
    frontend::SpeculativeDecodeConfig,
    http::StageHttpOptions,
};
pub fn stage_http_options(args: ServeArgs) -> Result<StageHttpOptions> {
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
    Ok(StageHttpOptions {
        config,
        topology,
        bind_addr,
        metrics_otlp_grpc: args.metrics_otlp_grpc,
        telemetry_queue_capacity: args.telemetry_queue_capacity,
        telemetry_level: args.telemetry_level.into(),
    })
}
pub fn binary_stage_options(args: ServeBinaryArgs) -> Result<BinaryStageOptions> {
    if args.openai_generation_concurrency == Some(0) {
        bail!("--openai-generation-concurrency must be greater than zero");
    }
    if args.openai_prefill_chunk_size == 0 {
        bail!("--openai-prefill-chunk-size must be greater than zero");
    }
    if !args.openai_prefill_adaptive_target_ms.is_finite()
        || args.openai_prefill_adaptive_target_ms <= 0.0
    {
        bail!("--openai-prefill-adaptive-target-ms must be finite and greater than zero");
    }
    let downstream_wire_condition = WireCondition::with_jitter(
        args.downstream_wire_delay_ms,
        args.downstream_wire_mbps,
        args.downstream_wire_jitter_ms,
        args.downstream_wire_stall_ms,
        args.downstream_wire_stall_p,
    )?;
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
    let openai_generation_concurrency = args
        .openai_generation_concurrency
        .unwrap_or_else(|| usize::try_from(config.lane_count).unwrap_or(usize::MAX));
    let openai_generation_queue_capacity =
        args.openai_generation_queue_capacity.unwrap_or_else(|| {
            skippy_server::frontend::default_generation_queue_capacity(
                openai_generation_concurrency,
            )
        });
    let adaptive_generation_min_concurrency =
        skippy_server::frontend::resolve_adaptive_generation_min_concurrency(
            args.openai_adaptive_generation_concurrency,
            args.openai_adaptive_generation_min_concurrency,
            openai_generation_concurrency,
            "--openai-adaptive-generation-min-concurrency",
        )?;
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
            generation_concurrency: openai_generation_concurrency,
            adaptive_generation_min_concurrency,
            generation_queue_capacity: openai_generation_queue_capacity,
            generation_admission_timeout_secs: args.openai_generation_admission_timeout_secs,
            prefill_chunk_size: args.openai_prefill_chunk_size,
            prefill_chunk_policy: args.openai_prefill_chunk_policy,
            prefill_chunk_schedule: args.openai_prefill_chunk_schedule,
            prefill_adaptive_start: args.openai_prefill_adaptive_start,
            prefill_adaptive_step: args.openai_prefill_adaptive_step,
            prefill_adaptive_max: args.openai_prefill_adaptive_max,
            prefill_adaptive_target_ms: args.openai_prefill_adaptive_target_ms,
            draft_model_path: args.openai_draft_model_path,
            speculative_window: args.openai_speculative_window,
            adaptive_speculative_window: args.openai_adaptive_speculative_window,
            draft_n_gpu_layers: args.openai_draft_n_gpu_layers,
            native_mtp_draft_model_path: args.openai_native_mtp_draft_model_path,
            native_mtp_max_tokens: 3,
            native_mtp_min_tokens: 0,
            speculative: openai_speculative,
        });
    let native_mtp_enabled = config.native_mtp_enabled;
    Ok(BinaryStageOptions {
        config,
        topology,
        bind_addr,
        metrics_otlp_grpc: args.metrics_otlp_grpc,
        telemetry_queue_capacity: args.telemetry_queue_capacity,
        telemetry_level: args.telemetry_level.into(),
        max_inflight: args.max_inflight,
        reply_credit_limit: args.reply_credit_limit,
        async_prefill_forward: args.async_prefill_forward || !args.no_async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs: args.downstream_connect_timeout_secs,
        native_mtp_enabled,
        continuous_batching: true,
        openai,
    })
}

pub fn local_openai_options(
    args: crate::cli::ServeOpenAiArgs,
) -> Result<skippy_api::serving::LocalOpenAiOptions> {
    if args.first_stage_addr.is_some() {
        bail!(
            "--first-stage-addr is no longer supported; direct prediction return requires embedded stage-0 OpenAI serving via serve-binary --openai-bind-addr"
        );
    }
    let config = crate::local_model::prepare_openai_stage(&args)?;
    let topology = args
        .topology
        .as_ref()
        .map(load_json)
        .transpose()
        .context("load topology")?;
    let speculative = args
        .speculative_config
        .as_ref()
        .map(load_json)
        .transpose()
        .context("load speculative config")?
        .unwrap_or_default();
    Ok(skippy_api::serving::LocalOpenAiOptions {
        config,
        topology,
        speculative,
        bind_addr: args.bind_addr,
        model_id: args.model_id,
        default_max_tokens: args.default_max_tokens,
        generation_concurrency: args.generation_concurrency,
        adaptive_generation_concurrency: args.adaptive_generation_concurrency,
        adaptive_generation_min_concurrency: args.adaptive_generation_min_concurrency,
        generation_queue_capacity: args.generation_queue_capacity,
        generation_admission_timeout_secs: args.generation_admission_timeout_secs,
        prefill_chunk_size: args.prefill_chunk_size,
        prefill_chunk_policy: args.prefill_chunk_policy,
        prefill_chunk_schedule: args.prefill_chunk_schedule,
        prefill_adaptive_start: args.prefill_adaptive_start,
        prefill_adaptive_step: args.prefill_adaptive_step,
        prefill_adaptive_max: args.prefill_adaptive_max,
        prefill_adaptive_target_ms: args.prefill_adaptive_target_ms,
        metrics_otlp_grpc: args.metrics_otlp_grpc,
        telemetry_queue_capacity: args.telemetry_queue_capacity,
        telemetry_level: args.telemetry_level.into(),
        openai_guardrails: args.openai_guardrails.into(),
    })
}

impl From<crate::cli::TelemetryLevel> for skippy_server::telemetry::TelemetryLevel {
    fn from(value: crate::cli::TelemetryLevel) -> Self {
        match value {
            crate::cli::TelemetryLevel::Off => Self::Off,
            crate::cli::TelemetryLevel::Summary => Self::Summary,
            crate::cli::TelemetryLevel::Debug => Self::Debug,
        }
    }
}
impl From<crate::cli::OpenAiGuardrailsCliMode> for skippy_server::frontend::OpenAiGuardrailsMode {
    fn from(value: crate::cli::OpenAiGuardrailsCliMode) -> Self {
        match value {
            crate::cli::OpenAiGuardrailsCliMode::Disabled => Self::Disabled,
            crate::cli::OpenAiGuardrailsCliMode::Metrics => Self::Metrics,
            crate::cli::OpenAiGuardrailsCliMode::Enforce => Self::Enforce,
        }
    }
}
impl From<crate::cli::NativeRuntimeArgs> for skippy_api::native_runtime::NativeRuntimeOptions {
    fn from(value: crate::cli::NativeRuntimeArgs) -> Self {
        Self {
            bundle_dirs: value.bundle_dirs,
            cache_dir: value.cache_dir,
            release: value.release,
            selection: value.selection,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;
    use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};

    use super::*;
    use crate::cli::{Cli, Command};
    use skippy_runtime::MtpSource;
    use skippy_server::frontend::{
        NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
        VerifyWindowConfig,
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
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
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
                runahead_max_tokens: 0,
            },
            ..SpeculativeDecodeConfig::default()
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
            "skippy",
            "serve-binary",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--openai-bind-addr",
            "127.0.0.1:9337",
            "--openai-speculative-config",
            plan_path.to_str().expect("UTF-8 plan path"),
        ])
        .expect("parse binary stage CLI");
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };

        let options = binary_stage_options(args).expect("resolve binary stage");
        assert_eq!(options.resolved_mtp_source(), MtpSource::Integrated);
        let openai = options.openai.expect("embedded OpenAI configuration");

        assert!(options.native_mtp_enabled);
        assert_eq!(openai.generation_concurrency, 1);
        assert_eq!(openai.generation_queue_capacity, 16);
        assert_eq!(openai.generation_admission_timeout_secs, 0);
        assert_eq!(openai.speculative, expected);
    }

    #[test]
    fn embedded_openai_admission_overrides_are_independent() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let stage_path = dir.path().join("stage.json");
        let mut config = stage_config();
        config.lane_count = 4;
        fs::write(
            &stage_path,
            serde_json::to_vec(&config).expect("serialize stage config"),
        )
        .expect("write stage config");

        let cli = Cli::try_parse_from([
            "skippy",
            "serve-binary",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--openai-bind-addr",
            "127.0.0.1:9337",
            "--openai-generation-concurrency",
            "2",
            "--openai-generation-queue-capacity",
            "33",
            "--openai-generation-admission-timeout-secs",
            "90",
        ])
        .expect("parse binary stage CLI");
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };
        let openai = binary_stage_options(args)
            .expect("resolve binary stage")
            .openai
            .expect("embedded OpenAI configuration");

        assert_eq!(openai.generation_concurrency, 2);
        assert_eq!(openai.generation_queue_capacity, 33);
        assert_eq!(openai.generation_admission_timeout_secs, 90);
    }

    #[test]
    fn native_mtp_sidecar_path_reaches_the_embedded_stage() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let stage_path = dir.path().join("stage.json");
        let sidecar_path = dir.path().join("sidecar-mtp.gguf");
        let plan_path = dir.path().join("speculative.json");
        fs::write(
            &stage_path,
            serde_json::to_vec(&stage_config()).expect("serialize stage config"),
        )
        .expect("write stage config");
        fs::write(&sidecar_path, b"gguf-stub").expect("write sidecar stub");
        fs::write(
            &plan_path,
            serde_json::to_vec(&cache_composite_plan()).expect("serialize speculative config"),
        )
        .expect("write speculative config");

        let cli = Cli::try_parse_from([
            "skippy",
            "serve-binary",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--openai-bind-addr",
            "127.0.0.1:9337",
            "--openai-speculative-config",
            plan_path.to_str().expect("UTF-8 speculative path"),
            "--openai-native-mtp-draft-model-path",
            sidecar_path.to_str().expect("UTF-8 sidecar path"),
        ])
        .expect("parse binary stage CLI");
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };

        let options = binary_stage_options(args).expect("resolve binary stage");
        assert_eq!(options.resolved_mtp_source(), MtpSource::External);
        let openai = options.openai.expect("embedded OpenAI configuration");

        // The sidecar attaches MTP heads to the served model; it must not be
        // opened as a standalone draft model.
        assert_eq!(
            openai.native_mtp_draft_model_path.as_deref(),
            Some(sidecar_path.as_path())
        );
        assert_eq!(openai.draft_model_path, None);
    }

    #[test]
    fn disabled_native_mtp_keeps_a_terminal_stage_out_of_integrated_mode() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let stage_path = dir.path().join("stage.json");
        let mut config = stage_config();
        config.native_mtp_enabled = false;
        fs::write(
            &stage_path,
            serde_json::to_vec(&config).expect("serialize stage config"),
        )
        .expect("write stage config");

        let cli = Cli::try_parse_from([
            "skippy",
            "serve-binary",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
        ])
        .expect("parse binary stage CLI");
        let Command::ServeBinary(args) = cli.command else {
            panic!("expected serve-binary command");
        };

        let options = binary_stage_options(args).expect("resolve binary stage");
        assert_eq!(options.resolved_mtp_source(), MtpSource::Disabled);
    }

    #[test]
    fn cache_composite_plan_is_json_stable_for_stage_handoff() {
        let plan = cache_composite_plan();
        let json = serde_json::to_value(&plan).expect("serialize speculative plan");

        assert_eq!(json["ngram"]["min_ngram"], 2);
        assert_eq!(json["verify_window"]["pipeline_depth"], 2);
    }
}
