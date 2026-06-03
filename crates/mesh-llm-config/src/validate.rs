use crate::model::{merge_hardware, merge_model_fit, merge_multimodal, merge_throughput};
use crate::plugin_validation::validate_plugin_entries;
use crate::*;
use anyhow::{Result, bail};
use semver::{BuildMetadata, Version};

pub fn validate_config(config: &MeshConfig) -> Result<()> {
    if let Some(version) = config.version
        && version != 1
    {
        bail!("unsupported config version {version}; expected version = 1");
    }
    if let Some(bind) = config.owner_control.bind
        && bind.port() == 0
        && !bind.ip().is_loopback()
    {
        bail!("owner_control.bind must use a concrete port when binding a non-loopback address");
    }
    if let Some(advertise_addr) = config.owner_control.advertise_addr {
        if advertise_addr.port() == 0 {
            bail!("owner_control.advertise_addr must use a concrete port");
        }
        if advertise_addr.ip().is_unspecified() {
            bail!("owner_control.advertise_addr must not use an unspecified IP address");
        }
    }
    if let Some(parallel) = config.gpu.parallel
        && parallel < 1
    {
        bail!("gpu.parallel must be at least 1, got {parallel}");
    }
    validate_mesh_requirements_config(&config.mesh_requirements)?;
    validate_telemetry_config(&config.telemetry)?;
    validate_plugin_entries(&config.plugins)?;
    let defaults_hardware = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref());
    if let Some(defaults) = &config.defaults {
        validate_model_defaults(defaults, "defaults", config.gpu.assignment)?;
    }
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            bail!("models[{index}].model must not be empty");
        }
        validate_model_entry(
            model,
            &format!("models[{index}]"),
            config.gpu.assignment,
            defaults_hardware,
        )?;
    }
    Ok(())
}

fn validate_model_defaults(
    defaults: &ModelConfigDefaults,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    if let Some(model_fit) = &defaults.model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    if let Some(hardware) = &defaults.hardware {
        validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?;
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

fn validate_model_entry(
    model: &ModelConfigEntry,
    base_path: &str,
    gpu_assignment: GpuAssignment,
    defaults_hardware: Option<&HardwareConfig>,
) -> Result<()> {
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
    )?;
    Ok(())
}

fn validate_gpu_assignment_constraints(
    hardware: Option<&HardwareConfig>,
    inherited_device: Option<&str>,
    legacy_gpu_id: Option<&str>,
    device_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    if matches!(gpu_assignment, GpuAssignment::Auto) && legacy_gpu_id.is_some() {
        bail!("{device_path} must not be set when gpu.assignment = \"auto\"");
    }
    if matches!(gpu_assignment, GpuAssignment::Pinned) {
        match hardware
            .and_then(|config| config.device.as_deref())
            .or(inherited_device)
        {
            Some(device) if !device.trim().is_empty() && !device.eq_ignore_ascii_case("auto") => {}
            _ => {
                bail!(
                    "{device_path} must be set to a non-empty value when gpu.assignment = \"pinned\""
                );
            }
        }
    }
    Ok(())
}

fn validate_model_fit(config: &ModelFitConfig, base_path: &str) -> Result<()> {
    validate_optional_positive_u32(config.ctx_size, &format!("{base_path}.ctx_size"))?;
    validate_optional_positive_u32(config.batch, &format!("{base_path}.batch"))?;
    validate_optional_positive_u32(config.ubatch, &format!("{base_path}.ubatch"))?;
    if let (Some(batch), Some(ubatch)) = (config.batch, config.ubatch)
        && ubatch > batch
    {
        bail!("{base_path}.ubatch must be less than or equal to {base_path}.batch");
    }
    validate_optional_non_empty(
        config.cache_type_k.as_deref(),
        &format!("{base_path}.cache_type_k"),
    )?;
    validate_optional_non_empty(
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
        bail!("{base_path}.cache_idle_slots requires {base_path}.prompt_cache = true");
    }
    if let Some(prefix_cache) = &config.prefix_cache {
        validate_prefix_cache(prefix_cache, &format!("{base_path}.prefix_cache"))?;
    }
    if let (Some(keep_tokens), Some(ctx_size)) = (config.keep_tokens, config.ctx_size)
        && keep_tokens > ctx_size
    {
        bail!("{base_path}.keep_tokens must be less than or equal to {base_path}.ctx_size");
    }
    validate_optional_positive_u32(
        config.checkpoint_interval,
        &format!("{base_path}.checkpoint_interval"),
    )?;
    validate_optional_positive_u32(
        config.checkpoint_count,
        &format!("{base_path}.checkpoint_count"),
    )?;
    validate_optional_non_empty(
        config.lookup_cache_static.as_deref(),
        &format!("{base_path}.lookup_cache_static"),
    )?;
    validate_optional_non_empty(
        config.lookup_cache_dynamic.as_deref(),
        &format!("{base_path}.lookup_cache_dynamic"),
    )?;
    Ok(())
}

fn validate_prefix_cache(config: &PrefixCacheConfig, base_path: &str) -> Result<()> {
    if config.enabled == Some(false) {
        return Ok(());
    }
    if config.enabled == Some(true) {
        validate_optional_positive_u32(config.max_entries, &format!("{base_path}.max_entries"))?;
        validate_optional_positive_u32(config.min_tokens, &format!("{base_path}.min_tokens"))?;
        validate_optional_positive_u32(
            config.shared_stride_tokens,
            &format!("{base_path}.shared_stride_tokens"),
        )?;
        validate_optional_positive_u32(
            config.shared_record_limit,
            &format!("{base_path}.shared_record_limit"),
        )?;
    }
    validate_optional_enum(
        config.payload_mode.as_deref(),
        &["resident-kv", "kv-recurrent", "full-state", "auto"],
        &format!("{base_path}.payload_mode"),
    )?;
    Ok(())
}

fn validate_hardware(
    config: &HardwareConfig,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> Result<()> {
    if let Some(device) = &config.device {
        validate_non_empty(device, &format!("{base_path}.device"))?;
        if matches!(gpu_assignment, GpuAssignment::Pinned) && device.eq_ignore_ascii_case("auto") {
            bail!("{base_path}.device must not be \"auto\" when gpu.assignment = \"pinned\"");
        }
    }
    if let Some(gpu_layers) = &config.gpu_layers {
        match gpu_layers {
            IntegerOrString::Integer(value) if *value >= -1 && *value <= i64::from(i32::MAX) => {}
            IntegerOrString::Integer(value) if *value > i64::from(i32::MAX) => {
                bail!("{base_path}.gpu_layers must be at most {}", i32::MAX)
            }
            IntegerOrString::Integer(_) => bail!("{base_path}.gpu_layers must be at least -1"),
            IntegerOrString::String(value) => {
                validate_allowed(value, &["auto"], &format!("{base_path}.gpu_layers"))?
            }
        }
    }
    match (config.stage_layer_start, config.stage_layer_end) {
        (Some(start), Some(end)) if end <= start => {
            bail!("{base_path}.stage_layer_end must be greater than {base_path}.stage_layer_start");
        }
        (Some(_), None) => bail!(
            "{base_path}.stage_layer_end must be set when {base_path}.stage_layer_start is set"
        ),
        (None, Some(_)) => bail!(
            "{base_path}.stage_layer_start must be set when {base_path}.stage_layer_end is set"
        ),
        _ => {}
    }
    validate_optional_enum(
        config.placement.as_deref(),
        &["auto", "pooled", "separated"],
        &format!("{base_path}.placement"),
    )?;
    if let Some(tensor_split) = &config.tensor_split {
        match tensor_split {
            TensorSplitConfig::Ratios(ratios) => {
                for ratio in ratios {
                    if *ratio < 0.0 {
                        bail!("{base_path}.tensor_split must contain only non-negative ratios");
                    }
                }
            }
            TensorSplitConfig::String(value) => {
                validate_non_empty(value, &format!("{base_path}.tensor_split"))?
            }
        }
    }
    validate_optional_enum(
        config.split_mode.as_deref(),
        &["auto", "none", "layer", "row"],
        &format!("{base_path}.split_mode"),
    )?;
    if let Some(value) = &config.cpu_moe {
        validate_bool_or_auto(Some(value), &format!("{base_path}.cpu_moe"))?;
    }
    if config.rpc_backend.is_some() {
        bail!("{base_path}.rpc_backend is documented-rejected and must not be set");
    }
    if let Some(fit_context) = &config.fit_context {
        validate_bool_or_auto(Some(fit_context), &format!("{base_path}.fit_context"))?;
    }
    validate_non_negative_f64(
        config.safety_margin_gb,
        &format!("{base_path}.safety_margin_gb"),
    )?;
    validate_hf_pair(
        config.hf_repo.as_deref(),
        config.hf_file.as_deref(),
        &format!("{base_path}.hf_repo"),
        &format!("{base_path}.hf_file"),
    )?;
    validate_optional_non_empty(
        config.model_path.as_deref(),
        &format!("{base_path}.model_path"),
    )?;
    validate_optional_non_empty(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    validate_bool_or_auto(config.mmap.as_ref(), &format!("{base_path}.mmap"))?;
    validate_bool_or_auto(config.warmup.as_ref(), &format!("{base_path}.warmup"))?;
    validate_string_list(&config.lora_adapters, &format!("{base_path}.lora_adapters"))?;
    validate_string_list(
        &config.control_vectors,
        &format!("{base_path}.control_vectors"),
    )?;
    Ok(())
}

fn validate_throughput(config: &ThroughputConfig, base_path: &str) -> Result<()> {
    if let Some(parallel) = config.parallel
        && parallel < 1
    {
        bail!("{base_path}.parallel must be at least 1, got {parallel}");
    }
    validate_bool_or_auto(
        config.continuous_batching.as_ref(),
        &format!("{base_path}.continuous_batching"),
    )?;
    // `0` is a canonical auto/default sentinel for threads and threads_batch.
    if config.threads_http.is_some() {
        bail!("{base_path}.threads_http is documented-rejected and must not be set");
    }
    if let Some(BoolOrString::String(value)) = &config.poll {
        validate_allowed(
            value,
            &["auto", "busy", "sleep"],
            &format!("{base_path}.poll"),
        )?;
    }
    if let Some(cpu_affinity) = &config.cpu_affinity {
        match cpu_affinity {
            StringOrStringList::String(value) => {
                validate_non_empty(value, &format!("{base_path}.cpu_affinity"))?
            }
            StringOrStringList::List(values) => {
                validate_string_list(values, &format!("{base_path}.cpu_affinity"))?
            }
        }
    }
    validate_optional_non_empty(config.numa.as_deref(), &format!("{base_path}.numa"))?;
    if let Some(slot_prompt_similarity) = config.slot_prompt_similarity
        && slot_prompt_similarity < 0.0
    {
        bail!("{base_path}.slot_prompt_similarity must be non-negative");
    }
    if config.sleep_idle_seconds.is_some() {
        bail!("{base_path}.sleep_idle_seconds is documented-rejected and must not be set");
    }
    validate_optional_enum(
        config.tuning_profile.as_deref(),
        &["throughput", "balanced", "saver"],
        &format!("{base_path}.tuning_profile"),
    )?;
    Ok(())
}

fn validate_skippy(config: &SkippyConfig, base_path: &str) -> Result<()> {
    validate_optional_non_empty(
        config.stage_model_path.as_deref(),
        &format!("{base_path}.stage_model_path"),
    )?;
    validate_optional_non_empty(
        config.stage_role.as_deref(),
        &format!("{base_path}.stage_role"),
    )?;
    validate_optional_non_empty(
        config.stage_topology.as_deref(),
        &format!("{base_path}.stage_topology"),
    )?;
    validate_optional_enum(
        config.activation_wire_dtype.as_deref(),
        &["auto", "f16", "f32", "q8"],
        &format!("{base_path}.activation_wire_dtype"),
    )?;
    validate_optional_non_empty(
        config.binary_stage_transport.as_deref(),
        &format!("{base_path}.binary_stage_transport"),
    )?;
    if config.openai_frontend_mode.is_some() {
        bail!("{base_path}.openai_frontend_mode is documented-rejected and must not be set");
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
                bail!(
                    "{base_path}.prefill_chunk_schedule must contain only comma-separated positive integers"
                );
            }
        }
    }
    Ok(())
}

fn validate_speculative(config: &SpeculativeConfig, base_path: &str) -> Result<()> {
    validate_optional_enum(
        config.mode.as_deref(),
        &["auto", "disabled", "draft", "ngram"],
        &format!("{base_path}.mode"),
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
    validate_optional_positive_u32(
        config.draft_max_tokens,
        &format!("{base_path}.draft_max_tokens"),
    )?;
    if let (Some(min), Some(max)) = (config.draft_min_tokens, config.draft_max_tokens)
        && min > max
    {
        bail!(
            "{base_path}.draft_min_tokens must be less than or equal to {base_path}.draft_max_tokens"
        );
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
        bail!("{base_path}.draft_gpu_layers must be at least -1");
    }
    validate_optional_non_empty(
        config.draft_device.as_deref(),
        &format!("{base_path}.draft_device"),
    )?;
    validate_optional_positive_usize(config.draft_threads, &format!("{base_path}.draft_threads"))?;
    validate_optional_non_empty(
        config.draft_cache_type_k.as_deref(),
        &format!("{base_path}.draft_cache_type_k"),
    )?;
    validate_optional_non_empty(
        config.draft_cache_type_v.as_deref(),
        &format!("{base_path}.draft_cache_type_v"),
    )?;
    validate_optional_positive_u32(config.ngram_min, &format!("{base_path}.ngram_min"))?;
    validate_optional_positive_u32(config.ngram_max, &format!("{base_path}.ngram_max"))?;
    if let (Some(min), Some(max)) = (config.ngram_min, config.ngram_max)
        && max < min
    {
        bail!("{base_path}.ngram_max must be greater than or equal to {base_path}.ngram_min");
    }
    validate_bool_or_auto(
        config.spec_default.as_ref(),
        &format!("{base_path}.spec_default"),
    )?;
    if config.mode.as_deref() == Some("draft")
        && config.draft_model_path.is_none()
        && config.draft_hf_repo.is_none()
        && config.draft_selection_policy.is_none()
    {
        bail!(
            "{base_path}.draft_selection_policy must be set when {base_path}.mode = \"draft\" and no explicit draft model source is configured"
        );
    }
    Ok(())
}

fn validate_request_defaults(config: &RequestDefaultsConfig, base_path: &str) -> Result<()> {
    validate_optional_positive_u32(config.max_tokens, &format!("{base_path}.max_tokens"))?;
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
        bail!("{base_path}.top_k must be greater than or equal to 0");
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
        bail!("{base_path}.repeat_last_n must be greater than or equal to -1");
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
            _ => bail!("{base_path}.mirostat_mode must be one of: disabled, 1, 2"),
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
    validate_optional_non_empty(
        config.sampler_sequence.as_deref(),
        &format!("{base_path}.sampler_sequence"),
    )?;
    if config.backend_sampling.is_some() {
        bail!("{base_path}.backend_sampling is documented-rejected and must not be set");
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
    validate_optional_non_empty(
        config.chat_template.as_deref(),
        &format!("{base_path}.chat_template"),
    )?;
    validate_optional_non_empty(
        config.chat_template_file.as_deref(),
        &format!("{base_path}.chat_template_file"),
    )?;
    validate_optional_non_empty(
        config.system_prompt.as_deref(),
        &format!("{base_path}.system_prompt"),
    )?;
    if config.grammar.is_some() {
        bail!("{base_path}.grammar is documented-rejected and must not be set");
    }
    if config.json_schema.is_some() {
        bail!("{base_path}.json_schema is documented-rejected and must not be set");
    }
    if config.logprobs.is_some() {
        bail!("{base_path}.logprobs is documented-rejected and must not be set");
    }
    Ok(())
}

fn validate_multimodal_pair(
    hardware: Option<&HardwareConfig>,
    multimodal: Option<&MultimodalConfig>,
    hardware_path: &str,
    multimodal_path: &str,
) -> Result<()> {
    if let (Some(hardware), Some(multimodal)) = (hardware, multimodal) {
        if let (Some(hardware_mmproj), Some(multimodal_mmproj)) =
            (hardware.mmproj.as_deref(), multimodal.mmproj.as_deref())
            && hardware_mmproj != multimodal_mmproj
        {
            bail!("{multimodal_path}.mmproj must match {hardware_path}.mmproj when both are set");
        }
        if let (Some(hardware_offload), Some(multimodal_offload)) = (
            hardware.mmproj_offload.as_ref(),
            multimodal.mmproj_offload.as_ref(),
        ) && hardware_offload != multimodal_offload
        {
            bail!(
                "{multimodal_path}.mmproj_offload must match {hardware_path}.mmproj_offload when both are set"
            );
        }
    }
    Ok(())
}

fn validate_multimodal(config: &MultimodalConfig, base_path: &str) -> Result<()> {
    validate_optional_non_empty(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_optional_non_empty(
        config.mmproj_url.as_deref(),
        &format!("{base_path}.mmproj_url"),
    )?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    if let (Some(min), Some(max)) = (config.image_min_tokens, config.image_max_tokens)
        && min > max
    {
        bail!(
            "{base_path}.image_min_tokens must be less than or equal to {base_path}.image_max_tokens"
        );
    }
    if config.embeddings.is_some() {
        bail!("{base_path}.embeddings is documented-rejected and must not be set");
    }
    if config.reranking.is_some() {
        bail!("{base_path}.reranking is documented-rejected and must not be set");
    }
    if config.pooling.is_some() {
        bail!("{base_path}.pooling is documented-rejected and must not be set");
    }
    if config.vocoder.is_some() {
        bail!("{base_path}.vocoder is documented-rejected and must not be set");
    }
    Ok(())
}

fn validate_advanced(config: &AdvancedConfig, base_path: &str) -> Result<()> {
    if let Some(server) = &config.server {
        if server.host.is_some() {
            bail!("{base_path}.server.host is documented-rejected and must not be set");
        }
        if server.port.is_some() {
            bail!("{base_path}.server.port is documented-rejected and must not be set");
        }
        if server.reuse_port.is_some() {
            bail!("{base_path}.server.reuse_port is documented-rejected and must not be set");
        }
        if server.timeout.is_some() {
            bail!("{base_path}.server.timeout is documented-rejected and must not be set");
        }
        if server.metrics.is_some() {
            bail!("{base_path}.server.metrics is documented-rejected and must not be set");
        }
        if server.slots.is_some() {
            bail!("{base_path}.server.slots is documented-rejected and must not be set");
        }
        if server.props.is_some() {
            bail!("{base_path}.server.props is documented-rejected and must not be set");
        }
        if server.api_prefix.is_some() {
            bail!("{base_path}.server.api_prefix is documented-rejected and must not be set");
        }
        validate_optional_non_empty(
            server.alias.as_deref(),
            &format!("{base_path}.server.alias"),
        )?;
    }
    Ok(())
}

fn validate_optional_positive_u32(value: Option<u32>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_positive_u64(value: Option<u64>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_positive_usize(value: Option<usize>, path: &str) -> Result<()> {
    if value == Some(0) {
        bail!("{path} must be at least 1 when set");
    }
    Ok(())
}

fn validate_optional_non_empty(value: Option<&str>, path: &str) -> Result<()> {
    if let Some(value) = value {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

fn validate_non_empty(value: &str, path: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{path} must not be empty when set");
    }
    Ok(())
}

fn validate_optional_enum(value: Option<&str>, allowed: &[&str], path: &str) -> Result<()> {
    if let Some(value) = value {
        validate_allowed(value, allowed, path)?;
    }
    Ok(())
}

fn validate_allowed(value: &str, allowed: &[&str], path: &str) -> Result<()> {
    validate_non_empty(value, path)?;
    if !allowed
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        bail!("{path} must be one of: {}", allowed.join(", "));
    }
    Ok(())
}

fn validate_bool_or_auto(value: Option<&BoolOrAuto>, path: &str) -> Result<()> {
    if let Some(BoolOrAuto::String(value)) = value {
        validate_allowed(value, &["auto"], path)?;
    }
    Ok(())
}

fn validate_probability(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value
        && !(0.0..=1.0).contains(&value)
    {
        bail!("{path} must be between 0.0 and 1.0");
    }
    Ok(())
}

fn validate_non_negative_f64(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value
        && value < 0.0
    {
        bail!("{path} must be greater than or equal to 0.0");
    }
    Ok(())
}

fn validate_positive_f64(value: Option<f64>, path: &str) -> Result<()> {
    if let Some(value) = value
        && value <= 0.0
    {
        bail!("{path} must be greater than 0.0");
    }
    Ok(())
}

fn validate_hf_pair(
    repo: Option<&str>,
    file: Option<&str>,
    repo_path: &str,
    file_path: &str,
) -> Result<()> {
    validate_optional_non_empty(repo, repo_path)?;
    validate_optional_non_empty(file, file_path)?;
    match (repo, file) {
        (Some(_), None) => bail!("{file_path} must be set when {repo_path} is set"),
        (None, Some(_)) => bail!("{repo_path} must be set when {file_path} is set"),
        _ => Ok(()),
    }
}

fn validate_string_list(values: &[String], path: &str) -> Result<()> {
    for value in values {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

fn validate_mesh_requirements_config(config: &MeshRequirementsConfig) -> Result<()> {
    let min_node_version = config
        .min_node_version
        .as_deref()
        .map(parse_node_version)
        .transpose()?;
    let max_node_version = config
        .max_node_version
        .as_deref()
        .map(parse_node_version)
        .transpose()?;
    if let (Some(min), Some(max)) = (&min_node_version, &max_node_version)
        && version_precedence_cmp(min, max).is_gt()
    {
        bail!(
            "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version"
        );
    }

    if let (Some(min), Some(max)) = (config.min_protocol_version, config.max_protocol_version)
        && min > max
    {
        bail!(
            "mesh_requirements.min_protocol_version must be less than or equal to mesh_requirements.max_protocol_version"
        );
    }

    for signer_key in &config.release_signer_keys {
        validate_release_signer_key_shape(signer_key)?;
    }
    if config.require_release_attestation && config.release_signer_keys.is_empty() {
        bail!(
            "mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty; certified-build admission is not remote runtime attestation, so trust must be anchored in at least one release signer key"
        );
    }

    Ok(())
}

fn parse_node_version(raw: &str) -> Result<Version> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        bail!(
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)"
        );
    }
    let normalized = normalized
        .strip_prefix('v')
        .or_else(|| normalized.strip_prefix('V'))
        .unwrap_or(normalized);
    Version::parse(normalized).map_err(|_| {
        anyhow::anyhow!(
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)"
        )
    })
}

fn validate_release_signer_key_shape(raw: &str) -> Result<()> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        bail!("mesh_requirements.release_signer_keys entries must not be empty");
    }
    let Some(encoded) = normalized.strip_prefix("ed25519:") else {
        bail!(
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'"
        );
    };
    if encoded.len() != 64 || !encoded.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!(
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'"
        );
    }
    Ok(())
}

fn version_precedence_cmp(left: &Version, right: &Version) -> std::cmp::Ordering {
    let mut left = left.clone();
    let mut right = right.clone();
    left.build = BuildMetadata::EMPTY;
    right.build = BuildMetadata::EMPTY;
    left.cmp(&right)
}

fn validate_telemetry_config(config: &TelemetryConfig) -> Result<()> {
    if let Some(service_name) = &config.service_name
        && service_name.trim().is_empty()
    {
        bail!("telemetry.service_name must not be empty when set");
    }
    if let Some(endpoint) = &config.endpoint
        && endpoint.trim().is_empty()
    {
        bail!("telemetry.endpoint must not be empty when set");
    }
    if let Some(endpoint) = &config.metrics.endpoint
        && endpoint.trim().is_empty()
    {
        bail!("telemetry.metrics.endpoint must not be empty when set");
    }
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            bail!("telemetry.headers keys must not be empty");
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs
        && export_interval_secs < 1
    {
        bail!("telemetry.export_interval_secs must be at least 1");
    }
    if let Some(queue_size) = config.queue_size
        && queue_size < 1
    {
        bail!("telemetry.queue_size must be at least 1");
    }
    if config.prompt_shape_metrics {
        bail!("telemetry.prompt_shape_metrics is not supported yet and must remain false");
    }
    Ok(())
}
