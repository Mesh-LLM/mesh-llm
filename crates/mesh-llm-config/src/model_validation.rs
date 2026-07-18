use crate::diagnostic::{ConfigDiagnostic, DiagnosticResult, alias_diagnostic};
use crate::hardware_validation::{
    validate_gpu_assignment_constraints, validate_hardware, validate_throughput,
};
use crate::model::{
    AdvancedConfig, BoolOrAuto, ConfigPath, ConfigPathSegment, GpuAssignment, HardwareConfig,
    IntegerOrString, MeshConfig, ModelConfigDefaults, ModelConfigEntry, ModelFitConfig,
    MultimodalConfig, PrefixCacheConfig, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig,
    SkippyConfig, SpeculativeConfig, StringOrStringList, merge_hardware, merge_model_fit,
    merge_multimodal, merge_throughput,
};
use crate::validation_support::{
    looks_like_model_identifier, validate_allowed, validate_bool_or_auto, validate_hf_pair,
    validate_model_identifier, validate_non_empty, validate_non_negative_f64,
    validate_optional_enum, validate_optional_http_url, validate_optional_kv_cache_type,
    validate_optional_path, validate_optional_positive_u64, validate_optional_positive_usize,
    validate_optional_u32_range, validate_positive_f64, validate_probability, validate_string_list,
    validation_diagnostic,
};

pub(crate) fn validate_duplicate_model_entries(
    models: &[ModelConfigEntry],
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            if models[i].model == models[j].model
                && models[i].derived_profile() == models[j].derived_profile()
            {
                let profile_i = models[i].derived_profile();
                let profile_clause = if profile_i.is_empty() {
                    " and default profile".to_string()
                } else {
                    format!(" and profile=\"{profile_i}\"")
                };
                diagnostics.push(validation_diagnostic(
                    "models",
                    format!(
                        "duplicate model entry: models[{i}] and models[{j}] both have model=\"{}\"{profile_clause}",
                        models[i].model,
                    ),
                ));
            }
        }
    }
}

pub(crate) fn collect_legacy_draft_model_path_warnings(
    config: &MeshConfig,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Some(speculative) = config
        .defaults
        .as_ref()
        .and_then(|d| d.speculative.as_ref())
        .filter(|s| s.legacy_draft_model_path_used)
    {
        // Only warn when the value looks like a model identifier (a ':' that
        // sits after the last '/', as in `Org/Name:Q4_K_M`). Bare local paths
        // including Windows-style absolutes like `C:/models/draft.gguf` put
        // their ':' before the slash and are not identifiers, so they cannot
        // be migrated to draft_model without failing identifier validation.
        if speculative
            .draft_model
            .as_deref()
            .is_some_and(looks_like_model_identifier)
        {
            diagnostics.push(alias_diagnostic(
                ConfigPath::from_fields(["defaults", "speculative", "draft_model_path"]),
                ConfigPath::from_fields(["defaults", "speculative", "draft_model"]),
                "draft_model_path is deprecated; rename to draft_model",
            ));
        }
    }
    for (index, model) in config.models.iter().enumerate() {
        if let Some(speculative) = model
            .speculative
            .as_ref()
            .filter(|s| s.legacy_draft_model_path_used)
        {
            // Only warn when the value looks like a model identifier (a ':'
            // that sits after the last '/', as in `Org/Name:Q4_K_M`). Bare
            // local paths including Windows-style absolutes like
            // `C:/models/draft.gguf` cannot be migrated to draft_model
            // without failing identifier validation.
            if speculative
                .draft_model
                .as_deref()
                .is_some_and(looks_like_model_identifier)
            {
                let mut used_path =
                    ConfigPath::from_fields(["models", "speculative", "draft_model_path"]);
                used_path
                    .segments
                    .insert(1, ConfigPathSegment::Index { index });
                let mut canonical_path =
                    ConfigPath::from_fields(["models", "speculative", "draft_model"]);
                canonical_path
                    .segments
                    .insert(1, ConfigPathSegment::Index { index });
                diagnostics.push(alias_diagnostic(
                    used_path,
                    canonical_path,
                    "draft_model_path is deprecated; rename to draft_model",
                ));
            }
        }
    }
}

pub(crate) fn validate_model_defaults(
    defaults: &ModelConfigDefaults,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> DiagnosticResult {
    if let Some(model_fit) = &defaults.model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    if let Some(hardware) = &defaults.hardware {
        validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?;
        validate_gpu_assignment_constraints(
            Some(hardware),
            None,
            None,
            &format!("{base_path}.hardware.device"),
            gpu_assignment,
            false,
        )?;
    }
    if let Some(throughput) = &defaults.throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &defaults.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &defaults.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &defaults.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        defaults.hardware.as_ref(),
        defaults.multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &defaults.multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &defaults.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    Ok(())
}

pub(crate) fn validate_model_entry(
    model: &ModelConfigEntry,
    base_path: &str,
    gpu_assignment: GpuAssignment,
    defaults_hardware: Option<&HardwareConfig>,
) -> DiagnosticResult {
    let model_fit = merge_model_fit(
        model.model_fit.clone(),
        model.ctx_size,
        model.cache_type_k.clone(),
        model.cache_type_v.clone(),
        model.batch,
        model.ubatch,
        model.flash_attention,
    );
    let multimodal = merge_multimodal(model.multimodal.clone(), model.mmproj.clone());
    let hardware = merge_hardware(
        model.hardware.clone(),
        model.gpu_id.clone(),
        multimodal.as_ref().and_then(|config| config.mmproj.clone()),
        multimodal
            .as_ref()
            .and_then(|config| config.mmproj_offload.clone()),
    );
    let throughput = merge_throughput(model.throughput.clone(), model.parallel);

    if let Some(mmproj) = &model.mmproj {
        validate_non_empty(mmproj, &format!("{base_path}.multimodal.mmproj"))?;
    }
    if let Some(model_fit) = &model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    if let Some(hardware) = hardware.as_ref() {
        validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?;
    }
    if let Some(throughput) = &throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &model.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &model.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &model.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        hardware.as_ref(),
        multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &model.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    validate_gpu_assignment_constraints(
        hardware.as_ref(),
        defaults_hardware.and_then(|hardware| hardware.device.as_deref()),
        model
            .gpu_id_from_legacy_shim
            .then_some(model.gpu_id.as_deref())
            .flatten(),
        &format!("{base_path}.hardware.device"),
        gpu_assignment,
        true,
    )?;
    Ok(())
}

fn validate_model_fit(config: &ModelFitConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.ctx_size,
        &format!("{base_path}.ctx_size"),
        1,
        1_000_000,
    )?;
    validate_optional_u32_range(config.batch, &format!("{base_path}.batch"), 1, 10_000_000)?;
    validate_optional_u32_range(config.ubatch, &format!("{base_path}.ubatch"), 1, 10_000_000)?;
    if let (Some(batch), Some(ubatch)) = (config.batch, config.ubatch)
        && ubatch > batch
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.ubatch"),
            format!("{base_path}.ubatch must be less than or equal to {base_path}.batch"),
        ));
    }
    validate_optional_kv_cache_type(
        config.cache_type_k.as_deref(),
        &format!("{base_path}.cache_type_k"),
    )?;
    validate_optional_kv_cache_type(
        config.cache_type_v.as_deref(),
        &format!("{base_path}.cache_type_v"),
    )?;
    validate_optional_enum(
        config.kv_cache_policy.as_deref(),
        &["auto", "quality", "balanced", "saver"],
        &format!("{base_path}.kv_cache_policy"),
    )?;
    validate_bool_or_auto(
        config.kv_offload.as_ref(),
        &format!("{base_path}.kv_offload"),
    )?;
    validate_bool_or_auto(
        config.kv_unified.as_ref(),
        &format!("{base_path}.kv_unified"),
    )?;
    validate_bool_or_auto(
        config.prompt_cache.as_ref(),
        &format!("{base_path}.prompt_cache"),
    )?;
    validate_bool_or_auto(
        config.context_shift.as_ref(),
        &format!("{base_path}.context_shift"),
    )?;
    if let Some(cache_idle_slots) = config.cache_idle_slots
        && cache_idle_slots > 0
        && matches!(config.prompt_cache, Some(BoolOrAuto::Bool(false)))
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.cache_idle_slots"),
            format!("{base_path}.cache_idle_slots requires {base_path}.prompt_cache = true"),
        ));
    }
    if let Some(prefix_cache) = &config.prefix_cache {
        validate_prefix_cache(prefix_cache, &format!("{base_path}.prefix_cache"))?;
    }
    if let (Some(keep_tokens), Some(ctx_size)) = (config.keep_tokens, config.ctx_size)
        && keep_tokens > ctx_size
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.keep_tokens"),
            format!("{base_path}.keep_tokens must be less than or equal to {base_path}.ctx_size"),
        ));
    }
    validate_optional_u32_range(
        config.keep_tokens,
        &format!("{base_path}.keep_tokens"),
        1,
        1_000_000,
    )?;
    validate_optional_u32_range(
        config.checkpoint_interval,
        &format!("{base_path}.checkpoint_interval"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.checkpoint_count,
        &format!("{base_path}.checkpoint_count"),
        1,
        10_000_000,
    )?;
    validate_optional_path(
        config.lookup_cache_static.as_deref(),
        &format!("{base_path}.lookup_cache_static"),
    )?;
    validate_optional_path(
        config.lookup_cache_dynamic.as_deref(),
        &format!("{base_path}.lookup_cache_dynamic"),
    )?;
    Ok(())
}

fn validate_prefix_cache(config: &PrefixCacheConfig, base_path: &str) -> DiagnosticResult {
    if config.enabled == Some(false) {
        return Ok(());
    }
    if config.enabled == Some(true) {
        validate_optional_u32_range(
            config.max_entries,
            &format!("{base_path}.max_entries"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.min_tokens,
            &format!("{base_path}.min_tokens"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.shared_stride_tokens,
            &format!("{base_path}.shared_stride_tokens"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.shared_record_limit,
            &format!("{base_path}.shared_record_limit"),
            1,
            10_000_000,
        )?;
    }
    validate_optional_enum(
        config.payload_mode.as_deref(),
        &["resident-kv", "kv-recurrent", "full-state", "auto"],
        &format!("{base_path}.payload_mode"),
    )?;
    Ok(())
}

fn validate_skippy(config: &SkippyConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_path(
        config.stage_model_path.as_deref(),
        &format!("{base_path}.stage_model_path"),
    )?;
    validate_optional_enum(
        config.activation_wire_dtype.as_deref(),
        &["auto", "f16", "f32", "q8"],
        &format!("{base_path}.activation_wire_dtype"),
    )?;
    if config.openai_frontend_mode.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.openai_frontend_mode"),
            format!("{base_path}.openai_frontend_mode is documented-rejected and must not be set"),
        ));
    }
    validate_optional_positive_u64(
        config.lifecycle_startup_timeout_ms,
        &format!("{base_path}.lifecycle_startup_timeout_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_readiness_interval_ms,
        &format!("{base_path}.lifecycle_readiness_interval_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_health_interval_ms,
        &format!("{base_path}.lifecycle_health_interval_ms"),
    )?;
    validate_optional_enum(
        config.prefill_chunking.as_deref(),
        &["auto", "fixed", "schedule", "adaptive-ramp"],
        &format!("{base_path}.prefill_chunking"),
    )?;
    if let Some(schedule) = &config.prefill_chunk_schedule {
        validate_non_empty(schedule, &format!("{base_path}.prefill_chunk_schedule"))?;
        for item in schedule.split(',') {
            let trimmed = item.trim();
            if trimmed.is_empty()
                || trimmed
                    .parse::<u32>()
                    .ok()
                    .filter(|value| *value > 0)
                    .is_none()
            {
                return Err(validation_diagnostic(
                    &format!("{base_path}.prefill_chunk_schedule"),
                    format!(
                        "{base_path}.prefill_chunk_schedule must contain only comma-separated positive integers"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_speculative(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    if let Some(strategy) = config.strategy.as_deref() {
        validate_non_empty(strategy, &format!("{base_path}.strategy"))?;
    }
    validate_optional_enum(
        config.mode.as_deref(),
        &["auto", "disabled", "draft", "ngram"],
        &format!("{base_path}.mode"),
    )?;
    validate_model_identifier(
        config.draft_model.as_deref(),
        &format!("{base_path}.draft_model"),
        config.legacy_draft_model_path_used,
    )?;
    validate_hf_pair(
        config.draft_hf_repo.as_deref(),
        config.draft_hf_file.as_deref(),
        &format!("{base_path}.draft_hf_repo"),
        &format!("{base_path}.draft_hf_file"),
    )?;
    validate_optional_enum(
        config.draft_selection_policy.as_deref(),
        &["manual", "auto"],
        &format!("{base_path}.draft_selection_policy"),
    )?;
    validate_optional_enum(
        config.pairing_fault.as_deref(),
        &[
            "warn_disable",
            "fail-open",
            "fail-closed",
            "fail_open",
            "fail_closed",
        ],
        &format!("{base_path}.pairing_fault"),
    )?;
    validate_optional_u32_range(
        config.draft_min_tokens,
        &format!("{base_path}.draft_min_tokens"),
        0,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.draft_max_tokens,
        &format!("{base_path}.draft_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.draft_min_tokens, config.draft_max_tokens)
        && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_min_tokens"),
            format!(
                "{base_path}.draft_min_tokens must be less than or equal to {base_path}.draft_max_tokens"
            ),
        ));
    }
    validate_probability(
        config.draft_acceptance_threshold,
        &format!("{base_path}.draft_acceptance_threshold"),
    )?;
    validate_probability(
        config.draft_split_probability,
        &format!("{base_path}.draft_split_probability"),
    )?;
    if let Some(gpu_layers) = config.draft_gpu_layers
        && gpu_layers < -1
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_gpu_layers"),
            format!("{base_path}.draft_gpu_layers must be at least -1"),
        ));
    }
    validate_optional_positive_usize(config.draft_threads, &format!("{base_path}.draft_threads"))?;
    validate_optional_kv_cache_type(
        config.draft_cache_type_k.as_deref(),
        &format!("{base_path}.draft_cache_type_k"),
    )?;
    validate_optional_kv_cache_type(
        config.draft_cache_type_v.as_deref(),
        &format!("{base_path}.draft_cache_type_v"),
    )?;
    validate_optional_u32_range(
        config.ngram_min,
        &format!("{base_path}.ngram_min"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.ngram_max,
        &format!("{base_path}.ngram_max"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.ngram_min, config.ngram_max)
        && max < min
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.ngram_max"),
            format!("{base_path}.ngram_max must be greater than or equal to {base_path}.ngram_min"),
        ));
    }
    validate_bool_or_auto(
        config.spec_default.as_ref(),
        &format!("{base_path}.spec_default"),
    )?;
    if config.mode.as_deref() == Some("draft")
        && config.draft_model.is_none()
        && config.draft_hf_repo.is_none()
        && config.draft_selection_policy.is_none()
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_selection_policy"),
            format!(
                "{base_path}.draft_selection_policy must be set when {base_path}.mode = \"draft\" and no explicit draft model source is configured"
            ),
        ));
    }
    validate_speculative_proposer_controls(config, base_path)
}

fn validate_speculative_proposer_controls(
    config: &SpeculativeConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_optional_enum(
        config.ngram_proposer.as_deref(),
        &["simple", "cache"],
        &format!("{base_path}.ngram_proposer"),
    )?;
    validate_optional_u32_range(
        config.ngram_max_proposal_tokens,
        &format!("{base_path}.ngram_max_proposal_tokens"),
        1,
        10_000_000,
    )?;
    validate_extension_controls(config, base_path)?;
    validate_native_mtp_controls(config, base_path)?;
    validate_verify_window_controls(config, base_path)
}

fn validate_extension_controls(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.extension_initial_tokens,
        &format!("{base_path}.extension_initial_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.extension_max_tokens,
        &format!("{base_path}.extension_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(initial), Some(max)) =
        (config.extension_initial_tokens, config.extension_max_tokens)
        && initial > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.extension_initial_tokens"),
            format!(
                "{base_path}.extension_initial_tokens must be less than or equal to {base_path}.extension_max_tokens"
            ),
        ));
    }
    validate_optional_u32_range(
        config.extension_tail_backoff_proposals,
        &format!("{base_path}.extension_tail_backoff_proposals"),
        0,
        10_000_000,
    )
}

fn validate_native_mtp_controls(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.native_mtp_reject_cooldown_tokens,
        &format!("{base_path}.native_mtp_reject_cooldown_tokens"),
        0,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.native_mtp_suppress_cooldown_draft_limit,
        &format!("{base_path}.native_mtp_suppress_cooldown_draft_limit"),
        0,
        10_000_000,
    )
}

fn validate_verify_window_controls(
    config: &SpeculativeConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_optional_u32_range(
        config.verify_window_min_tokens,
        &format!("{base_path}.verify_window_min_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.verify_window_max_tokens,
        &format!("{base_path}.verify_window_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (
        config.verify_window_min_tokens,
        config.verify_window_max_tokens,
    ) && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.verify_window_min_tokens"),
            format!(
                "{base_path}.verify_window_min_tokens must be less than or equal to {base_path}.verify_window_max_tokens"
            ),
        ));
    }
    validate_optional_u32_range(
        config.verify_window_pipeline_depth,
        &format!("{base_path}.verify_window_pipeline_depth"),
        1,
        1_024,
    )
}

fn validate_request_defaults(config: &RequestDefaultsConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.max_tokens,
        &format!("{base_path}.max_tokens"),
        1,
        10_000_000,
    )?;
    if let Some(stop) = &config.stop {
        match stop {
            StringOrStringList::String(value) => {
                validate_non_empty(value, &format!("{base_path}.stop"))?
            }
            StringOrStringList::List(values) => {
                validate_string_list(values, &format!("{base_path}.stop"))?
            }
        }
    }
    validate_non_negative_f64(config.temperature, &format!("{base_path}.temperature"))?;
    validate_probability(config.top_p, &format!("{base_path}.top_p"))?;
    if let Some(top_k) = config.top_k
        && top_k < 0
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.top_k"),
            format!("{base_path}.top_k must be greater than or equal to 0"),
        ));
    }
    validate_probability(config.min_p, &format!("{base_path}.min_p"))?;
    validate_probability(config.typical_p, &format!("{base_path}.typical_p"))?;
    validate_non_negative_f64(config.top_nsigma, &format!("{base_path}.top_nsigma"))?;
    validate_non_negative_f64(
        config.dynatemp_range,
        &format!("{base_path}.dynatemp_range"),
    )?;
    validate_non_negative_f64(
        config.dynatemp_exponent,
        &format!("{base_path}.dynatemp_exponent"),
    )?;
    validate_non_negative_f64(
        config.repeat_penalty,
        &format!("{base_path}.repeat_penalty"),
    )?;
    if let Some(repeat_last_n) = config.repeat_last_n
        && repeat_last_n < -1
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.repeat_last_n"),
            format!("{base_path}.repeat_last_n must be greater than or equal to -1"),
        ));
    }
    validate_non_negative_f64(
        config.presence_penalty,
        &format!("{base_path}.presence_penalty"),
    )?;
    validate_non_negative_f64(
        config.frequency_penalty,
        &format!("{base_path}.frequency_penalty"),
    )?;
    if let Some(mode) = &config.mirostat_mode {
        match mode {
            IntegerOrString::Integer(value) if *value == 1 || *value == 2 => {}
            IntegerOrString::String(value) => validate_allowed(
                value,
                &["disabled", "1", "2"],
                &format!("{base_path}.mirostat_mode"),
            )?,
            _ => {
                return Err(validation_diagnostic(
                    &format!("{base_path}.mirostat_mode"),
                    format!("{base_path}.mirostat_mode must be one of: disabled, 1, 2"),
                ));
            }
        }
    }
    validate_positive_f64(
        config.mirostat_entropy,
        &format!("{base_path}.mirostat_entropy"),
    )?;
    validate_positive_f64(
        config.mirostat_learning_rate,
        &format!("{base_path}.mirostat_learning_rate"),
    )?;
    if let Some(samplers) = &config.samplers {
        validate_string_list(samplers, &format!("{base_path}.samplers"))?;
    }
    if config.backend_sampling.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.backend_sampling"),
            format!("{base_path}.backend_sampling is documented-rejected and must not be set"),
        ));
    }
    validate_optional_enum(
        config.reasoning_format.as_deref(),
        &["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
        &format!("{base_path}.reasoning_format"),
    )?;
    if let Some(reasoning_enabled) = &config.reasoning_enabled {
        match reasoning_enabled {
            ReasoningEnabled::Bool(_) => {}
            ReasoningEnabled::String(value) => validate_allowed(
                value,
                &["auto", "off", "on"],
                &format!("{base_path}.reasoning_enabled"),
            )?,
        }
    }
    if let Some(reasoning_budget) = &config.reasoning_budget {
        match reasoning_budget {
            ReasoningBudget::Integer(_) => {}
            ReasoningBudget::String(value) => validate_allowed(
                value,
                &["auto", "low", "medium", "high"],
                &format!("{base_path}.reasoning_budget"),
            )?,
        }
    }
    validate_optional_path(
        config.chat_template_file.as_deref(),
        &format!("{base_path}.chat_template_file"),
    )?;
    if config.grammar.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.grammar"),
            format!("{base_path}.grammar is documented-rejected and must not be set"),
        ));
    }
    if config.json_schema.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.json_schema"),
            format!("{base_path}.json_schema is documented-rejected and must not be set"),
        ));
    }
    if config.logprobs.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.logprobs"),
            format!("{base_path}.logprobs is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

fn validate_multimodal_pair(
    hardware: Option<&HardwareConfig>,
    multimodal: Option<&MultimodalConfig>,
    hardware_path: &str,
    multimodal_path: &str,
) -> DiagnosticResult {
    if let (Some(hardware), Some(multimodal)) = (hardware, multimodal) {
        if let (Some(hardware_mmproj), Some(multimodal_mmproj)) =
            (hardware.mmproj.as_deref(), multimodal.mmproj.as_deref())
            && hardware_mmproj != multimodal_mmproj
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj"),
                format!(
                    "{multimodal_path}.mmproj must match {hardware_path}.mmproj when both are set"
                ),
            ));
        }
        if let (Some(hardware_offload), Some(multimodal_offload)) = (
            hardware.mmproj_offload.as_ref(),
            multimodal.mmproj_offload.as_ref(),
        ) && hardware_offload != multimodal_offload
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj_offload"),
                format!(
                    "{multimodal_path}.mmproj_offload must match {hardware_path}.mmproj_offload when both are set"
                ),
            ));
        }
    }
    Ok(())
}

fn validate_multimodal(config: &MultimodalConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_path(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_optional_http_url(
        config.mmproj_url.as_deref(),
        &format!("{base_path}.mmproj_url"),
    )?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    validate_optional_u32_range(
        config.image_min_tokens,
        &format!("{base_path}.image_min_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.image_max_tokens,
        &format!("{base_path}.image_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.image_min_tokens, config.image_max_tokens)
        && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.image_min_tokens"),
            format!(
                "{base_path}.image_min_tokens must be less than or equal to {base_path}.image_max_tokens"
            ),
        ));
    }
    if config.embeddings.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.embeddings"),
            format!("{base_path}.embeddings is documented-rejected and must not be set"),
        ));
    }
    if config.reranking.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.reranking"),
            format!("{base_path}.reranking is documented-rejected and must not be set"),
        ));
    }
    if config.pooling.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.pooling"),
            format!("{base_path}.pooling is documented-rejected and must not be set"),
        ));
    }
    if config.vocoder.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.vocoder"),
            format!("{base_path}.vocoder is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

fn validate_advanced(config: &AdvancedConfig, base_path: &str) -> DiagnosticResult {
    if let Some(server) = &config.server {
        if server.host.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.host"),
                format!("{base_path}.server.host is documented-rejected and must not be set"),
            ));
        }
        if server.port.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.port"),
                format!("{base_path}.server.port is documented-rejected and must not be set"),
            ));
        }
        if server.reuse_port.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.reuse_port"),
                format!("{base_path}.server.reuse_port is documented-rejected and must not be set"),
            ));
        }
        if server.timeout.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.timeout"),
                format!("{base_path}.server.timeout is documented-rejected and must not be set"),
            ));
        }
        if server.metrics.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.metrics"),
                format!("{base_path}.server.metrics is documented-rejected and must not be set"),
            ));
        }
        if server.slots.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.slots"),
                format!("{base_path}.server.slots is documented-rejected and must not be set"),
            ));
        }
        if server.props.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.props"),
                format!("{base_path}.server.props is documented-rejected and must not be set"),
            ));
        }
        if server.api_prefix.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.api_prefix"),
                format!("{base_path}.server.api_prefix is documented-rejected and must not be set"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::legacy_validation_error_text;
    use crate::{MeshConfig, validate_config, validate_config_diagnostics};

    #[test]
    fn speculative_strategy_allows_package_declared_names() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
strategy = "mystery-oracle"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config)
            .expect("package strategy names are validated after package resolution");
        assert!(validate_config_diagnostics(&config).is_empty());
    }

    #[test]
    fn speculative_strategy_native_mtp_n1_alias_parses_as_mtp() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
strategy = "native-mtp-n1"
"#,
        )
        .expect("config should parse before validation");

        let strategy = config
            .defaults
            .as_ref()
            .and_then(|defaults| defaults.speculative.as_ref())
            .and_then(|speculative| speculative.strategy.as_deref());
        assert_eq!(strategy, Some("mtp"));
        validate_config(&config).expect("normalized speculative strategy should not fail");
        assert!(validate_config_diagnostics(&config).is_empty());
    }

    #[test]
    fn speculative_strategy_raw_name_is_deferred_to_package_resolution() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("native-mtp-n1".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        validate_config(&config)
            .expect("package strategy names are validated after package resolution");
        assert!(validate_config_diagnostics(&config).is_empty());
    }

    #[test]
    fn duplicate_model_with_same_profile_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
profile = "gaming"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
profile = "gaming"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("duplicate model entry"),
            "expected duplicate model error, got: {text}"
        );
        assert!(
            text.contains("models[0]"),
            "expected reference to models[0], got: {text}"
        );
        assert!(
            text.contains("models[1]"),
            "expected reference to models[1], got: {text}"
        );
    }

    #[test]
    fn duplicate_model_without_profile_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "my-model"

[[models]]
model = "my-model"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("duplicate model entry"),
            "expected duplicate model error, got: {text}"
        );
        assert!(
            text.contains("and default profile"),
            "expected 'and default profile' in error, got: {text}"
        );
    }

    #[test]
    fn draft_model_rejects_bare_path_without_colon() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("/models/draft.gguf".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must be a model identifier"),
            "expected identifier validation error, got: {text}"
        );
    }

    #[test]
    fn legacy_draft_model_path_skips_identifier_validation() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("/models/draft.gguf".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("must be a model identifier"),
            "expected no identifier error when legacy path used, got: {text}"
        );
    }

    #[test]
    fn draft_model_accepts_identifier_with_colon() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("Qwen/Qwen3-0.6B:Q4_K_M".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("must be a model identifier"),
            "expected no identifier error for valid identifier, got: {text}"
        );
    }

    #[test]
    fn legacy_draft_model_path_emits_migration_warning() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_some(),
            "expected legacy alias warning for draft_model_path, got diagnostics: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_bare_path_suppresses_migration_warning() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "/models/draft.gguf"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_none(),
            "bare path draft_model_path should not emit migration warning, got: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_windows_style_absolute_suppresses_migration_warning() {
        // The previous `contains(':')` heuristic falsely fired for
        // Windows-style absolute paths like `C:/models/draft.gguf` because
        // they contain a `:` after the drive letter. The fix requires the
        // colon quantization marker to follow the last `/`, so the path-like
        // value is no longer mistaken for an identifier.
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "C:/models/draft.gguf"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_none(),
            "Windows-style absolute path draft_model_path should not emit migration warning, got: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_rejects_nul_bytes() {
        // Legacy-path values should not bypass `validate_path_chars`. A NUL
        // byte inside a `draft_model_path` value must be rejected even when
        // `legacy_draft_model_path_used` is true.
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("bad\0path".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must not contain NUL bytes"),
            "expected NUL-byte rejection on legacy path, got: {text}"
        );
    }

    #[test]
    fn legacy_draft_model_path_rejects_control_characters() {
        // Control characters must also be rejected on legacy-path values.
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("bad\u{0001}path".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must not contain control characters"),
            "expected control-character rejection on legacy path, got: {text}"
        );
    }

    #[test]
    fn same_model_with_different_profiles_is_allowed() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
ctx_size = 4096

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
ctx_size = 8192
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("duplicate model entry"),
            "expected no duplicate error for different derived profiles, got: {text}"
        );
    }
}
