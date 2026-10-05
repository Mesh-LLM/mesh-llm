use crate::cli::ServeBinaryArgs;
use anyhow::{Context, Result, bail};
use skippy_config::load_json;
use skippy_protocol::{StageConfig, StageTopology};
use skippy_serving::{
    binary_transport::{BinaryStageOptions, EmbeddedOpenAiStageOptions, WireCondition},
    frontend::SpeculativeDecodeConfig,
};
pub fn binary_stage_options(args: ServeBinaryArgs) -> Result<BinaryStageOptions> {
    if args.openai_generation_concurrency == Some(0) {
        bail!("--generation-concurrency must be greater than zero");
    }
    if args.openai_prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if !args.openai_prefill_adaptive_target_ms.is_finite()
        || args.openai_prefill_adaptive_target_ms <= 0.0
    {
        bail!("--prefill-adaptive-target-ms must be finite and greater than zero");
    }
    let downstream_wire_condition = WireCondition::with_jitter(
        args.downstream_wire_delay_ms,
        args.downstream_wire_mbps,
        args.downstream_wire_jitter_ms,
        args.downstream_wire_stall_ms,
        args.downstream_wire_stall_p,
    )?;
    let tuning = args.settings.tuning(args.settings.guardrail_mode()?)?;
    let mut config = load_json::<StageConfig>(&args.config)
        .with_context(|| format!("load stage config {}", args.config.display()))?;
    args.settings.apply_stage(&mut config)?;
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
            skippy_serving::frontend::default_generation_queue_capacity(
                openai_generation_concurrency,
            )
        });
    let adaptive_generation_min_concurrency =
        skippy_serving::frontend::resolve_adaptive_generation_min_concurrency(
            args.openai_adaptive_generation_concurrency,
            args.openai_adaptive_generation_min_concurrency,
            openai_generation_concurrency,
            "--adaptive-generation-min-concurrency",
        )?;
    let mut defaults = binary_frontend_defaults(&args, &config, openai_generation_concurrency);
    if args.openai_draft_model_path.is_some()
        && args.openai_speculative_config.is_none()
        && !args.settings.values.contains_key("native-mtp")
    {
        defaults.speculative.native_mtp.enabled = false;
    }
    let openai_speculative: SpeculativeDecodeConfig = args
        .openai_speculative_config
        .as_ref()
        .map(load_json)
        .transpose()
        .context("load --speculative-config")?
        .unwrap_or(defaults.speculative);
    let openai_speculative = args.settings.speculative(
        openai_speculative,
        args.openai_draft_model_path.is_some() || defaults.draft_model_path.is_some(),
    )?;
    openai_speculative.validate()?;
    let speculation_disabled = args.settings.has_speculative_overrides()
        && openai_speculative.effective_strategy == "disabled";
    config.native_mtp_enabled = openai_speculative.native_mtp.enabled;
    if openai_speculative.ngram_fallback_draft && args.openai_draft_model_path.is_none() {
        bail!("ngram_fallback_draft requires --draft-model-path");
    }
    let openai_bind_addr = if args.worker_only {
        None
    } else {
        args.openai_bind_addr.or_else(|| {
            (config.stage_index == 0)
                .then_some(args.api_bind_addr)
                .flatten()
        })
    };
    let adaptive_draft =
        args.openai_draft_model_path.is_some() || defaults.adaptive_speculative_window;
    let adaptive_speculative_window = args
        .settings
        .boolean("adaptive-speculative-window")?
        .unwrap_or(args.openai_adaptive_speculative_window || adaptive_draft);
    let openai = openai_bind_addr.map(|bind_addr| EmbeddedOpenAiStageOptions {
        bind_addr,
        model_id: args.openai_model_id,
        default_max_tokens: args.openai_default_max_tokens,
        generation_concurrency: openai_generation_concurrency,
        pipeline_decode_groups: tuning.pipeline_decode_groups,
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
        draft_model_path: args
            .openai_draft_model_path
            .or(defaults.draft_model_path)
            .filter(|_| !speculation_disabled),
        speculative_window: args.openai_speculative_window,
        adaptive_speculative_window,
        draft_n_gpu_layers: args.openai_draft_n_gpu_layers,
        native_mtp_draft_model_path: args
            .openai_native_mtp_draft_model_path
            .or(defaults.native_mtp_draft_model_path)
            .filter(|_| !speculation_disabled),
        native_mtp_max_tokens: openai_speculative.native_mtp.max_draft_tokens,
        native_mtp_min_tokens: openai_speculative.native_mtp.min_draft_tokens,
        speculative: openai_speculative,
    });
    let native_mtp_enabled = config.native_mtp_enabled;
    Ok(BinaryStageOptions {
        tuning: tuning.clone(),
        config,
        topology,
        bind_addr,
        metrics_otlp_grpc: args.metrics_otlp_grpc,
        telemetry_queue_capacity: args.telemetry_queue_capacity,
        telemetry_level: args.telemetry_level.into(),
        max_inflight: args.max_inflight,
        reply_credit_limit: args.reply_credit_limit,
        async_prefill_forward: args
            .settings
            .boolean("async-prefill-forward")?
            .unwrap_or(args.async_prefill_forward || !args.no_async_prefill_forward),
        downstream_wire_condition,
        downstream_connect_timeout_secs: args.downstream_connect_timeout_secs,
        native_mtp_enabled,
        continuous_batching: tuning
            .continuous_batching
            .unwrap_or(skippy_config::local_serving::CONTINUOUS_BATCHING),
        compute_meter: None,
        openai,
        l3_manager: None,
    })
}

fn binary_frontend_defaults(
    args: &ServeBinaryArgs,
    config: &StageConfig,
    generation_concurrency: usize,
) -> skippy_api::serving::InferenceOptions {
    let mut defaults = skippy_api::serving::InferenceOptions::embedded_stage_defaults(
        args.openai_model_id.clone(),
        args.openai_default_max_tokens,
        generation_concurrency,
        0,
        config.native_mtp_enabled,
    );
    if !args.worker_only
        && config.stage_index == 0
        && args.openai_speculative_config.is_none()
        && args.openai_draft_model_path.is_none()
        && args.openai_native_mtp_draft_model_path.is_none()
        && let Some(path) = config
            .source_model_path
            .as_deref()
            .or(config.model_path.as_deref())
    {
        skippy_api::speculative::apply_auto_speculation(&mut defaults, std::path::Path::new(path));
    }
    defaults
}

pub fn local_openai_options(
    args: crate::cli::ServeOpenAiArgs,
) -> Result<skippy_api::serving::LocalOpenAiOptions> {
    let mut config = crate::local_model::prepare_openai_stage(&args)?;
    let topology = args
        .topology
        .as_ref()
        .map(load_json)
        .transpose()
        .context("load topology")?;
    let speculative: Option<SpeculativeDecodeConfig> = args
        .speculative_config
        .as_ref()
        .map(load_json)
        .transpose()
        .context("load speculative config")?;
    let mut tuning = args.settings.tuning(args.openai_guardrails)?;
    let speculative = if args.settings.has_speculative_overrides() {
        let mut defaults = skippy_api::serving::InferenceOptions::embedded_stage_defaults(
            args.model_id.clone(),
            args.default_max_tokens,
            config.lane_count as usize,
            0,
            config.native_mtp_enabled,
        );
        if speculative.is_none()
            && tuning.draft_model_path.is_none()
            && tuning.native_mtp_draft_model_path.is_none()
            && let Some(path) = config
                .source_model_path
                .as_deref()
                .or(config.model_path.as_deref())
        {
            skippy_api::speculative::apply_auto_speculation(
                &mut defaults,
                std::path::Path::new(path),
            );
        }
        let has_draft = tuning.draft_model_path.is_some() || defaults.draft_model_path.is_some();
        if tuning.draft_model_path.is_some() {
            defaults.speculative_window = skippy_config::local_serving::DRAFT_MODEL_TOKENS;
        }
        if tuning.draft_model_path.is_some() && !args.settings.values.contains_key("native-mtp") {
            defaults.speculative.native_mtp.enabled = false;
        }
        tuning.draft_model_path = tuning.draft_model_path.or(defaults.draft_model_path);
        tuning.native_mtp_draft_model_path = tuning
            .native_mtp_draft_model_path
            .or(defaults.native_mtp_draft_model_path);
        tuning.speculative_window = tuning
            .speculative_window
            .or(Some(defaults.speculative_window));
        tuning.adaptive_speculative_window = tuning
            .adaptive_speculative_window
            .or(Some(defaults.adaptive_speculative_window));
        Some(
            args.settings
                .speculative(speculative.unwrap_or(defaults.speculative), has_draft)?,
        )
    } else {
        speculative
    };
    if let Some(plan) = speculative.as_ref() {
        config.native_mtp_enabled = plan.native_mtp.enabled;
        if args.settings.has_speculative_overrides() && plan.effective_strategy == "disabled" {
            tuning.draft_model_path = None;
            tuning.native_mtp_draft_model_path = None;
        }
    }

    let disk_cache = crate::disk_cache::from_public_settings(
        args.kv_cache_disk.as_deref(),
        args.kv_cache_disk_dir.clone(),
        args.kv_cache_min_free.as_deref(),
    )?;
    args.settings
        .validate_cache_dependencies(&config, disk_cache.is_some())?;
    Ok(skippy_api::serving::LocalOpenAiOptions {
        tuning,
        config,
        topology,
        speculative,
        bind_addr: args
            .bind_addr
            .unwrap_or_else(crate::serve::default_public_bind_addr),
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
        model_open_events: None,
        disk_cache,
    })
}

impl From<crate::cli::TelemetryLevel> for skippy_serving::telemetry::TelemetryLevel {
    fn from(value: crate::cli::TelemetryLevel) -> Self {
        match value {
            crate::cli::TelemetryLevel::Off => Self::Off,
            crate::cli::TelemetryLevel::Summary => Self::Summary,
            crate::cli::TelemetryLevel::Debug => Self::Debug,
        }
    }
}
impl From<crate::cli::OpenAiGuardrailsCliMode>
    for skippy_serving::frontend::InferenceGuardrailsMode
{
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
    use skippy_serving::frontend::{
        NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
        VerifyWindowConfig,
    };

    fn binary_args(cli: Cli) -> ServeBinaryArgs {
        let Command::Serve(mut args) = cli.command else {
            panic!("expected serve command");
        };
        crate::serve::apply_public_frontend_tuning(&args.public, &mut args.stage);
        args.stage.config = args.public.config.expect("stage config");
        args.stage.api_bind_addr = Some(
            args.public
                .bind_addr
                .unwrap_or_else(crate::serve::default_public_bind_addr),
        );
        args.stage
    }

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

    #[test]
    fn worker_stages_preserve_configured_speculation_without_sibling_discovery() {
        for (worker_only, stage_index) in [(true, 0), (false, 1)] {
            let directory = tempfile::tempdir().unwrap();
            let target = directory.path().join("model.gguf");
            // Unknown architecture metadata would disable frontend auto speculation.
            fs::write(directory.path().join("draft.gguf"), []).unwrap();
            let mut config = stage_config();
            config.stage_index = stage_index;
            config.model_path = Some(target.to_string_lossy().into_owned());
            let cli = Cli::try_parse_from(["skippy", "serve", "--model", "model.gguf"]).unwrap();
            let Command::Serve(mut args) = cli.command else {
                panic!("expected serve");
            };
            args.stage.worker_only = worker_only;
            let defaults = binary_frontend_defaults(&args.stage, &config, 1);
            assert!(defaults.native_mtp_enabled);
            assert!(defaults.speculative.native_mtp.enabled);
            assert!(defaults.draft_model_path.is_none());
            assert!(defaults.native_mtp_draft_model_path.is_none());
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
            "serve",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--stage-transport",
            "binary",
            "--bind-addr",
            "127.0.0.1:9337",
            "--speculative-config",
            plan_path.to_str().expect("UTF-8 plan path"),
        ])
        .expect("parse binary stage CLI");
        let options = binary_stage_options(binary_args(cli)).expect("resolve binary stage");
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
            "serve",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--stage-transport",
            "binary",
            "--bind-addr",
            "127.0.0.1:9337",
            "--generation-concurrency",
            "2",
            "--generation-queue-capacity",
            "33",
            "--generation-admission-timeout-secs",
            "90",
        ])
        .expect("parse binary stage CLI");
        let openai = binary_stage_options(binary_args(cli))
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
            "serve",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--stage-transport",
            "binary",
            "--bind-addr",
            "127.0.0.1:9337",
            "--speculative-config",
            plan_path.to_str().expect("UTF-8 speculative path"),
            "--native-mtp-draft-model-path",
            sidecar_path.to_str().expect("UTF-8 sidecar path"),
        ])
        .expect("parse binary stage CLI");
        let options = binary_stage_options(binary_args(cli)).expect("resolve binary stage");
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
            "serve",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--stage-transport",
            "binary",
        ])
        .expect("parse binary stage CLI");
        let options = binary_stage_options(binary_args(cli)).expect("resolve binary stage");
        assert_eq!(options.resolved_mtp_source(), MtpSource::Disabled);
    }

    #[test]
    fn cache_composite_plan_is_json_stable_for_stage_handoff() {
        let plan = cache_composite_plan();
        let json = serde_json::to_value(&plan).expect("serialize speculative plan");

        assert_eq!(json["ngram"]["min_ngram"], 2);
        assert_eq!(json["verify_window"]["pipeline_depth"], 2);
    }
    #[test]
    fn draft_fallback_requires_a_standalone_draft_model_path() {
        let dir = tempfile::tempdir().expect("create temp directory");
        let stage_path = dir.path().join("stage.json");
        let plan_path = dir.path().join("speculative.json");
        fs::write(
            &stage_path,
            serde_json::to_vec(&stage_config()).expect("serialize stage config"),
        )
        .expect("write stage config");
        let mut plan = cache_composite_plan();
        plan.ngram_fallback_draft = true;
        fs::write(
            &plan_path,
            serde_json::to_vec(&plan).expect("serialize speculative config"),
        )
        .expect("write speculative config");

        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            stage_path.to_str().expect("UTF-8 stage path"),
            "--stage-transport",
            "binary",
            "--bind-addr",
            "127.0.0.1:9337",
            "--speculative-config",
            plan_path.to_str().expect("UTF-8 speculative path"),
        ])
        .expect("parse binary stage CLI");
        let error = match binary_stage_options(binary_args(cli)) {
            Ok(_) => panic!("draft fallback without a draft model must fail"),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("requires --draft-model-path"));
    }
}
