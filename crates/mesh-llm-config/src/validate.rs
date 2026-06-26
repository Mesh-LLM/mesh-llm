pub(crate) use crate::diagnostic::DiagnosticResult;
pub use crate::diagnostic::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, alias_diagnostic, invalid_value_diagnostic,
    legacy_validation_error_text, rejected_field_diagnostic, unsupported_field_diagnostic,
};
use crate::model::{merge_hardware, merge_model_fit, merge_multimodal, merge_throughput};
use crate::plugin_validation::{
    PluginSchemaAvailability, validate_plugin_entries, validate_plugin_entries_strict,
};
use crate::*;
use anyhow::Result;
use semver::{BuildMetadata, Version};
use url::Url;

fn parsed_config_path(raw_path: &str) -> Option<ConfigPath> {
    ConfigPath::parse_rendered(raw_path).ok()
}

pub(crate) fn validation_diagnostic(
    raw_path: &str,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    let message = message.into();
    if let Some(diagnostic) = built_in_support_diagnostic(raw_path, message.clone()) {
        return diagnostic;
    }

    let mut diagnostic = ConfigDiagnostic::error(
        ConfigDiagnosticCode::InvalidValue,
        ConfigDiagnosticSource::Validation,
        message,
    );
    diagnostic.path = parsed_config_path(raw_path);
    diagnostic
}

fn validate_duplicate_model_entries(
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

pub fn validate_config_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();

    if let Some(version) = config.version
        && version != 1
    {
        diagnostics.push(validation_diagnostic(
            "version",
            format!("unsupported config version {version}; expected version = 1"),
        ));
    }
    if let Some(bind) = config.owner_control.bind
        && bind.port() == 0
        && !bind.ip().is_loopback()
    {
        diagnostics.push(validation_diagnostic(
            "owner_control.bind",
            "owner_control.bind must use a concrete port when binding a non-loopback address",
        ));
    }
    if let Some(advertise_addr) = config.owner_control.advertise_addr {
        match config.owner_control.bind {
            Some(bind) if bind.port() == 0 => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.bind",
                    "owner_control.bind must use a concrete port when owner_control.advertise_addr is set",
                ));
            }
            Some(bind) if bind.port() != advertise_addr.port() => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr must use the same port as owner_control.bind",
                ));
            }
            Some(_) => {}
            None => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening",
                ));
            }
        }
        if advertise_addr.port() == 0 {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must use a concrete port",
            ));
        }
        if advertise_addr.ip().is_unspecified() {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must not use an unspecified IP address",
            ));
        }
    }
    if let Some(parallel) = config.gpu.parallel
        && parallel < 1
    {
        diagnostics.push(validation_diagnostic(
            "gpu.parallel",
            format!("gpu.parallel must be at least 1, got {parallel}"),
        ));
    }
    if let Err(diagnostic) = validate_mesh_requirements_config(&config.mesh_requirements) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_telemetry_config(&config.telemetry) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_runtime_config(&config.runtime) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_plugin_entries(&config.plugins) {
        diagnostics.push(diagnostic);
    }
    let defaults_hardware = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref());
    if let Some(defaults) = &config.defaults
        && let Err(diagnostic) =
            validate_model_defaults(defaults, "defaults", config.gpu.assignment)
    {
        diagnostics.push(diagnostic);
    }
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            diagnostics.push(validation_diagnostic(
                &format!("models[{index}].model"),
                format!("models[{index}].model must not be empty"),
            ));
        }
        if let Err(diagnostic) = validate_model_entry(
            model,
            &format!("models[{index}]"),
            config.gpu.assignment,
            defaults_hardware,
        ) {
            diagnostics.push(diagnostic);
        }
    }

    validate_duplicate_model_entries(&config.models, &mut diagnostics);

    diagnostics
}

fn validate_runtime_config(config: &RuntimeConfig) -> DiagnosticResult {
    let mesh_version = config.native_runtime.mesh_version.as_deref();
    let skippy_abi = config.native_runtime.skippy_abi.as_deref();
    let selection = config.native_runtime.selection.as_deref();
    if mesh_version.is_none() && (skippy_abi.is_some() || selection.is_some()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime",
            "runtime.native_runtime override must set mesh_version when skippy_abi or selection is set",
        ));
    }
    if matches!(mesh_version, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.mesh_version",
            "runtime.native_runtime.mesh_version must not be empty",
        ));
    }
    if matches!(skippy_abi, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.skippy_abi",
            "runtime.native_runtime.skippy_abi must not be empty",
        ));
    }
    if matches!(selection, Some(value) if value.trim().is_empty()) {
        return Err(validation_diagnostic(
            "runtime.native_runtime.selection",
            "runtime.native_runtime.selection must not be empty",
        ));
    }
    Ok(())
}

pub fn validate_config_diagnostics_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Vec<ConfigDiagnostic>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let mut diagnostics = validate_config_diagnostics(config);
    diagnostics.extend(validate_plugin_entries_strict(
        &config.plugins,
        raw_toml,
        schema_for_plugin,
    ));
    diagnostics
}

pub fn canonical_builtin_diagnostic_path(raw_path: &str) -> Option<ConfigPath> {
    canonicalize_built_in_config_identifier(raw_path)
        .and_then(|path| ConfigPath::parse_rendered(&path).ok())
}

pub fn built_in_support_diagnostic(
    raw_path: &str,
    message: impl Into<String>,
) -> Option<ConfigDiagnostic> {
    let resolution = resolve_built_in_config_identifier(raw_path)?;
    let message = message.into();
    let mut diagnostic = match resolution.support {
        ConfigSupportState::Rejected => {
            rejected_field_diagnostic(resolution.canonical_path.clone(), message)
        }
        ConfigSupportState::Unsupported | ConfigSupportState::Unwired => {
            unsupported_field_diagnostic(resolution.canonical_path.clone(), message)
        }
        _ => invalid_value_diagnostic(resolution.canonical_path.clone(), message),
    };
    diagnostic.path = Some(resolution.requested_path);
    diagnostic.canonical_path = Some(resolution.canonical_path);
    diagnostic.schema_source = Some(ConfigDiagnosticSchemaSource::BuiltIn);
    Some(diagnostic)
}

pub fn validate_config(config: &MeshConfig) -> Result<()> {
    let diagnostics = validate_config_diagnostics(config);
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    }
}

pub fn validate_config_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Result<()>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let diagnostics =
        validate_config_diagnostics_with_plugin_schemas(config, raw_toml, schema_for_plugin);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
    if has_errors {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    } else {
        Ok(())
    }
}

fn validate_model_defaults(
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

fn validate_model_entry(
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

fn validate_gpu_assignment_constraints(
    hardware: Option<&HardwareConfig>,
    inherited_device: Option<&str>,
    legacy_gpu_id: Option<&str>,
    device_path: &str,
    gpu_assignment: GpuAssignment,
    require_pinned_device: bool,
) -> DiagnosticResult {
    if matches!(gpu_assignment, GpuAssignment::Auto) {
        let explicit_device = hardware
            .and_then(|config| config.device.as_deref())
            .is_some_and(|device| !device.trim().is_empty());
        if explicit_device || legacy_gpu_id.is_some() {
            return Err(validation_diagnostic(
                device_path,
                format!("{device_path} must not be set when gpu.assignment = \"auto\""),
            ));
        }
    }
    if require_pinned_device && matches!(gpu_assignment, GpuAssignment::Pinned) {
        match hardware
            .and_then(|config| config.device.as_deref())
            .or(inherited_device)
        {
            Some(device) if !device.trim().is_empty() && !device.eq_ignore_ascii_case("auto") => {}
            _ => {
                return Err(validation_diagnostic(
                    device_path,
                    format!(
                        "{device_path} must be set to a non-empty value when gpu.assignment = \"pinned\""
                    ),
                ));
            }
        }
    }
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

fn validate_hardware(
    config: &HardwareConfig,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> DiagnosticResult {
    if let Some(device) = &config.device {
        validate_non_empty(device, &format!("{base_path}.device"))?;
        if matches!(gpu_assignment, GpuAssignment::Pinned) && device.eq_ignore_ascii_case("auto") {
            return Err(validation_diagnostic(
                &format!("{base_path}.device"),
                format!("{base_path}.device must not be \"auto\" when gpu.assignment = \"pinned\""),
            ));
        }
    }
    if let Some(gpu_layers) = &config.gpu_layers {
        match gpu_layers {
            IntegerOrString::Integer(value) if *value >= -1 && *value <= i64::from(i32::MAX) => {}
            IntegerOrString::Integer(value) if *value > i64::from(i32::MAX) => {
                return Err(validation_diagnostic(
                    &format!("{base_path}.gpu_layers"),
                    format!("{base_path}.gpu_layers must be at most {}", i32::MAX),
                ));
            }
            IntegerOrString::Integer(_) => {
                return Err(validation_diagnostic(
                    &format!("{base_path}.gpu_layers"),
                    format!("{base_path}.gpu_layers must be at least -1"),
                ));
            }
            IntegerOrString::String(value) => {
                validate_allowed(value, &["auto"], &format!("{base_path}.gpu_layers"))?
            }
        }
    }
    match (config.stage_layer_start, config.stage_layer_end) {
        (Some(start), Some(end)) if end <= start => {
            return Err(validation_diagnostic(
                &format!("{base_path}.stage_layer_end"),
                format!(
                    "{base_path}.stage_layer_end must be greater than {base_path}.stage_layer_start"
                ),
            ));
        }
        (Some(_), None) => {
            return Err(validation_diagnostic(
                &format!("{base_path}.stage_layer_end"),
                format!(
                    "{base_path}.stage_layer_end must be set when {base_path}.stage_layer_start is set"
                ),
            ));
        }
        (None, Some(_)) => {
            return Err(validation_diagnostic(
                &format!("{base_path}.stage_layer_start"),
                format!(
                    "{base_path}.stage_layer_start must be set when {base_path}.stage_layer_end is set"
                ),
            ));
        }
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
                        return Err(validation_diagnostic(
                            &format!("{base_path}.tensor_split"),
                            format!(
                                "{base_path}.tensor_split must contain only non-negative ratios"
                            ),
                        ));
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
        return Err(validation_diagnostic(
            &format!("{base_path}.rpc_backend"),
            format!("{base_path}.rpc_backend is documented-rejected and must not be set"),
        ));
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
    validate_optional_path(
        config.model_path.as_deref(),
        &format!("{base_path}.model_path"),
    )?;
    validate_optional_path(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
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

fn validate_throughput(config: &ThroughputConfig, base_path: &str) -> DiagnosticResult {
    if let Some(parallel) = config.parallel
        && parallel < 1
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.parallel"),
            format!("{base_path}.parallel must be at least 1, got {parallel}"),
        ));
    }
    validate_bool_or_auto(
        config.continuous_batching.as_ref(),
        &format!("{base_path}.continuous_batching"),
    )?;
    // `0` is a canonical auto/default sentinel for threads and threads_batch.
    if config.threads_http.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.threads_http"),
            format!("{base_path}.threads_http is documented-rejected and must not be set"),
        ));
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

    if let Some(slot_prompt_similarity) = config.slot_prompt_similarity
        && slot_prompt_similarity < 0.0
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.slot_prompt_similarity"),
            format!("{base_path}.slot_prompt_similarity must be non-negative"),
        ));
    }
    if config.sleep_idle_seconds.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.sleep_idle_seconds"),
            format!("{base_path}.sleep_idle_seconds is documented-rejected and must not be set"),
        ));
    }
    validate_optional_enum(
        config.tuning_profile.as_deref(),
        &["throughput", "balanced", "saver"],
        &format!("{base_path}.tuning_profile"),
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
    validate_optional_enum(
        config.strategy.as_deref(),
        &["auto", "disabled", "native-mtp-n1"],
        &format!("{base_path}.strategy"),
    )?;
    validate_optional_enum(
        config.mode.as_deref(),
        &["auto", "disabled", "draft", "ngram"],
        &format!("{base_path}.mode"),
    )?;
    validate_optional_path(
        config.draft_model_path.as_deref(),
        &format!("{base_path}.draft_model_path"),
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
        1,
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
        && config.draft_model_path.is_none()
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
    Ok(())
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

fn validate_optional_u32_range(
    value: Option<u32>,
    path: &str,
    min: u32,
    max: u32,
) -> DiagnosticResult {
    if let Some(value) = value
        && (value < min || value > max)
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be between {min} and {max}, got {value}"),
        ));
    }
    Ok(())
}

fn validate_optional_positive_u64(value: Option<u64>, path: &str) -> DiagnosticResult {
    if value == Some(0) {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be at least 1 when set"),
        ));
    }
    Ok(())
}

fn validate_optional_positive_usize(value: Option<usize>, path: &str) -> DiagnosticResult {
    if value == Some(0) {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be at least 1 when set"),
        ));
    }
    Ok(())
}

fn validate_non_empty(value: &str, path: &str) -> DiagnosticResult {
    if value.trim().is_empty() {
        return Err(validation_diagnostic(
            path,
            format!("{path} must not be empty when set"),
        ));
    }
    Ok(())
}

fn validate_optional_enum(value: Option<&str>, allowed: &[&str], path: &str) -> DiagnosticResult {
    if let Some(value) = value {
        validate_allowed(value, allowed, path)?;
    }
    Ok(())
}

fn validate_optional_kv_cache_type(value: Option<&str>, path: &str) -> DiagnosticResult {
    validate_optional_enum(
        value,
        &[
            "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
        ],
        path,
    )
}

fn validate_allowed(value: &str, allowed: &[&str], path: &str) -> DiagnosticResult {
    validate_non_empty(value, path)?;
    if !allowed
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be one of: {}", allowed.join(", ")),
        ));
    }
    Ok(())
}

fn validate_bool_or_auto(value: Option<&BoolOrAuto>, path: &str) -> DiagnosticResult {
    if let Some(BoolOrAuto::String(value)) = value {
        validate_allowed(value, &["auto"], path)?;
    }
    Ok(())
}

fn validate_optional_http_url(value: Option<&str>, path: &str) -> DiagnosticResult {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            let url = Url::parse(trimmed).map_err(|_| {
                validation_diagnostic(
                    path,
                    format!("{path} must be a valid URL (http:// or https://)"),
                )
            })?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(validation_diagnostic(
                    path,
                    format!("{path} must use http:// or https:// scheme"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_probability(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && !(0.0..=1.0).contains(&value)
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be between 0.0 and 1.0"),
        ));
    }
    Ok(())
}

fn validate_non_negative_f64(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && value < 0.0
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be greater than or equal to 0.0"),
        ));
    }
    Ok(())
}

fn validate_positive_f64(value: Option<f64>, path: &str) -> DiagnosticResult {
    if let Some(value) = value
        && value <= 0.0
    {
        return Err(validation_diagnostic(
            path,
            format!("{path} must be greater than 0.0"),
        ));
    }
    Ok(())
}

fn validate_hf_pair(
    repo: Option<&str>,
    file: Option<&str>,
    repo_path: &str,
    file_path: &str,
) -> DiagnosticResult {
    let repo_present = repo.is_some_and(|v| !v.trim().is_empty());
    let file_present = file.is_some_and(|v| !v.trim().is_empty());
    match (repo_present, file_present) {
        (true, false) => Err(validation_diagnostic(
            file_path,
            format!("{file_path} must be set when {repo_path} is set"),
        )),
        (false, true) => Err(validation_diagnostic(
            repo_path,
            format!("{repo_path} must be set when {file_path} is set"),
        )),
        _ => Ok(()),
    }
}

fn validate_string_list(values: &[String], path: &str) -> DiagnosticResult {
    for value in values {
        validate_non_empty(value, path)?;
    }
    Ok(())
}

fn validate_optional_path(value: Option<&str>, path: &str) -> DiagnosticResult {
    if let Some(value) = value {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            validate_path_chars(trimmed, path)?;
        }
    }
    Ok(())
}

fn validate_path_chars(value: &str, path: &str) -> DiagnosticResult {
    if value.contains('\0') {
        return Err(validation_diagnostic(
            path,
            format!("{path} must not contain NUL bytes"),
        ));
    }
    for ch in value.chars() {
        if ch.is_control() {
            return Err(validation_diagnostic(
                path,
                format!("{path} must not contain control characters"),
            ));
        }
    }
    Ok(())
}

fn validate_mesh_requirements_config(config: &MeshRequirementsConfig) -> DiagnosticResult {
    let min_node_version = config
        .min_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.min_node_version"))
        .transpose()?;
    let max_node_version = config
        .max_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.max_node_version"))
        .transpose()?;
    if let (Some(min), Some(max)) = (&min_node_version, &max_node_version)
        && version_precedence_cmp(min, max).is_gt()
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_node_version",
            "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version",
        ));
    }

    if let (Some(min), Some(max)) = (config.min_protocol_version, config.max_protocol_version)
        && min > max
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_protocol_version",
            "mesh_requirements.min_protocol_version must be less than or equal to mesh_requirements.max_protocol_version",
        ));
    }

    for signer_key in &config.release_signer_keys {
        validate_release_signer_key_shape(signer_key, "mesh_requirements.release_signer_keys")?;
    }
    if config.require_release_attestation && config.release_signer_keys.is_empty() {
        return Err(validation_diagnostic(
            "mesh_requirements.require_release_attestation",
            "mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty; certified-build admission is not remote runtime attestation, so trust must be anchored in at least one release signer key",
        ));
    }

    Ok(())
}

fn parse_node_version(raw: &str, path: &str) -> std::result::Result<Version, ConfigDiagnostic> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)",
        ));
    }
    let normalized = normalized
        .strip_prefix('v')
        .or_else(|| normalized.strip_prefix('V'))
        .unwrap_or(normalized);
    Version::parse(normalized).map_err(|_| {
        validation_diagnostic(
            path,
            "mesh_requirements node version bounds must be valid semver strings (an optional leading 'v' is allowed)",
        )
    })
}

fn validate_release_signer_key_shape(raw: &str, path: &str) -> DiagnosticResult {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must not be empty",
        ));
    }
    let Some(encoded) = normalized.strip_prefix("ed25519:") else {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'",
        ));
    };
    if encoded.len() != 64 || !encoded.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(validation_diagnostic(
            path,
            "mesh_requirements.release_signer_keys entries must be of the form 'ed25519:<64-character-hex-public-key>'",
        ));
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

fn validate_telemetry_config(config: &TelemetryConfig) -> DiagnosticResult {
    if let Some(service_name) = &config.service_name {
        let trimmed = service_name.trim();
        if !trimmed.is_empty() {
            // Validate service name: alphanumeric, dash, underscore only
            if !trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(validation_diagnostic(
                    "telemetry.service_name",
                    "telemetry.service_name must contain only alphanumeric characters, dashes, and underscores",
                ));
            }
        }
    }
    validate_optional_http_url(config.endpoint.as_deref(), "telemetry.endpoint")?;
    validate_optional_http_url(
        config.metrics.endpoint.as_deref(),
        "telemetry.metrics.endpoint",
    )?;
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            return Err(validation_diagnostic(
                "telemetry.headers",
                "telemetry.headers keys must not be empty",
            ));
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs
        && export_interval_secs < 1
    {
        return Err(validation_diagnostic(
            "telemetry.export_interval_secs",
            "telemetry.export_interval_secs must be at least 1",
        ));
    }
    if let Some(queue_size) = config.queue_size
        && queue_size < 1
    {
        return Err(validation_diagnostic(
            "telemetry.queue_size",
            "telemetry.queue_size must be at least 1",
        ));
    }
    if config.prompt_shape_metrics {
        return Err(validation_diagnostic(
            "telemetry.prompt_shape_metrics",
            "telemetry.prompt_shape_metrics is not supported yet and must remain false",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn schema_diagnostic_constructors_preserve_paths_and_legacy_message() {
        let used_path = ConfigPath::from_fields(["models", "gpu_id"]);
        let canonical_path = ConfigPath::from_fields(["models", "hardware", "device"]);
        let diagnostic = alias_diagnostic(
            used_path.clone(),
            canonical_path.clone(),
            "legacy gpu_id alias resolved to models.hardware.device",
        )
        .with_help("Use models.hardware.device for new config writes.");

        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Warning);
        assert_eq!(diagnostic.code, ConfigDiagnosticCode::AliasApplied);
        assert_eq!(
            diagnostic.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(diagnostic.path, Some(used_path));
        assert_eq!(diagnostic.canonical_path, Some(canonical_path));
        assert_eq!(
            diagnostic.legacy_message(),
            "legacy gpu_id alias resolved to models.hardware.device"
        );
        assert_eq!(
            diagnostic.help.as_deref(),
            Some("Use models.hardware.device for new config writes.")
        );
    }

    #[test]
    fn schema_diagnostics_round_trip_via_toml() {
        let diagnostic = rejected_field_diagnostic(
            ConfigPath::from_fields(["defaults", "request_defaults", "json_schema"]),
            "defaults.request_defaults.json_schema is documented-rejected and must not be set",
        );

        let encoded = toml::to_string(&diagnostic).expect("diagnostic should serialize");
        let decoded: ConfigDiagnostic =
            toml::from_str(&encoded).expect("diagnostic should deserialize");

        assert_eq!(decoded, diagnostic);
    }

    #[test]
    fn schema_diagnostic_helpers_cover_validation_and_support_cases() {
        let invalid = invalid_value_diagnostic(
            ConfigPath::from_fields(["gpu", "parallel"]),
            "gpu.parallel must be at least 1, got 0",
        );
        let unsupported = unsupported_field_diagnostic(
            ConfigPath::from_fields(["runtime", "sleep_idle_seconds"]),
            "runtime.sleep_idle_seconds is not supported",
        );

        assert_eq!(invalid.code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(invalid.severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            invalid.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(unsupported.code, ConfigDiagnosticCode::UnsupportedField);
        assert_eq!(
            unsupported.canonical_path.as_ref().map(ConfigPath::render),
            Some("runtime.sleep_idle_seconds".to_string())
        );
    }

    #[test]
    fn canonical_path_aliases_use_stable_built_in_identifier() {
        assert_eq!(
            canonical_builtin_diagnostic_path("models[0].gpu_id")
                .as_ref()
                .map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );

        let diagnostic = built_in_support_diagnostic(
            "models[0].gpu_id",
            "legacy gpu_id should report the canonical device path",
        )
        .expect("legacy built-in alias should resolve");

        assert_eq!(
            diagnostic.path.as_ref().map(ConfigPath::render),
            Some("models[0].gpu_id".to_string())
        );
        assert_eq!(
            diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );
    }

    #[test]
    fn owner_control_advertise_addr_requires_matching_bind_port() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17002"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr must use the same port as owner_control.bind"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:0"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(legacy_validation_error_text(&diagnostics).contains(
            "owner_control.bind must use a concrete port when owner_control.advertise_addr is set"
        ));

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17001"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config).expect("matching bind and advertise ports should validate");
    }

    #[test]
    fn structured_diagnostics_report_canonical_path_for_alias_backed_invalid_input() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-4B-Q4_K_M"
gpu_id = "metal:0"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.canonical_path.as_ref().map(ConfigPath::render)
                    == Some("models.<model-ref>.hardware.device".to_string())
            })
            .expect("legacy gpu_id path should yield a canonical device diagnostic");

        assert_eq!(diagnostic.code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            diagnostic.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(
            diagnostic.path.as_ref().map(ConfigPath::render),
            Some("models[0].hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.message,
            "models[0].hardware.device must not be set when gpu.assignment = \"auto\""
        );
    }

    #[test]
    fn speculative_strategy_rejects_unknown_values() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
strategy = "mystery-oracle"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].path.as_ref().map(ConfigPath::render),
            Some("defaults.speculative.strategy".to_string())
        );
        assert!(
            diagnostics[0]
                .message
                .contains("defaults.speculative.strategy must be one of")
        );
    }

    #[test]
    fn legacy_validation_errors_derive_compatible_string_messages() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 0
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            legacy_validation_error_text(&diagnostics),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );

        let err =
            validate_config(&config).expect_err("legacy validation surface should still fail");
        assert_eq!(
            err.to_string(),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );
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
