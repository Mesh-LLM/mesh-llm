use crate::diagnostic::DiagnosticResult;
use crate::model::{
    BoolOrString, GpuAssignment, HardwareConfig, IntegerOrString, StringOrStringList,
    TensorSplitConfig, ThroughputConfig,
};
use crate::validation_support::{
    validate_allowed, validate_bool_or_auto, validate_hf_pair, validate_non_empty,
    validate_non_negative_f64, validate_optional_enum, validate_optional_path,
    validate_string_list, validation_diagnostic,
};

pub(crate) fn validate_gpu_assignment_constraints(
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

pub(crate) fn validate_hardware(
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

pub(crate) fn validate_throughput(config: &ThroughputConfig, base_path: &str) -> DiagnosticResult {
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

#[cfg(test)]
mod tests {
    use crate::{MeshConfig, validate_config, validate_config_diagnostics};

    include!("validate_gpu_tune_tests.rs");
}
