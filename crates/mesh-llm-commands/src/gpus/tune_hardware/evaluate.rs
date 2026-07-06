use super::device_request::{
    ConfiguredTuneDeviceRequest, EffectiveTuneHardware, effective_tune_hardware,
};
use super::mlock::{TuneMlockProbe, detect_mlock_probe, evaluate_mlock};
use super::types::{
    ConfiguredDeviceSource, EvaluatedTuneDevice, TuneDeviceSelectionSource, TuneDeviceTarget,
    TuneGpuTarget, TuneHardwareEvaluation, TuneHardwareEvaluationInput, TuneMemoryBudget,
    TuneMemorySource, display_list, is_pinnable_stable_id,
};
use crate::gpus::tune::{TuneDiagnostic, TuneDiagnosticCode, TuneDiagnosticSeverity, TuneField};
use mesh_llm_system::{
    hardware::{GpuFacts, HardwareSurvey, resolve_pinned_gpu_strict},
    vram::VramCapacity,
};

pub(crate) fn evaluate_tune_hardware(
    input: TuneHardwareEvaluationInput<'_>,
) -> Result<TuneHardwareEvaluation, TuneDiagnostic> {
    evaluate_tune_hardware_with_probe(input, detect_mlock_probe())
}

#[cfg(test)]
pub(crate) fn evaluate_tune_hardware_with_mlock_probe_for_tests(
    input: TuneHardwareEvaluationInput<'_>,
    mlock_probe: TuneMlockProbe,
) -> Result<TuneHardwareEvaluation, TuneDiagnostic> {
    evaluate_tune_hardware_with_probe(input, mlock_probe)
}

fn evaluate_tune_hardware_with_probe(
    input: TuneHardwareEvaluationInput<'_>,
    mlock_probe: TuneMlockProbe,
) -> Result<TuneHardwareEvaluation, TuneDiagnostic> {
    let effective_hardware = effective_tune_hardware(input.config, input.target);
    let evaluated_device = evaluate_device(&effective_hardware, input.survey)?;
    let memory = evaluate_memory_budget(&evaluated_device, input.survey);
    let mlock = evaluate_mlock(&memory, mlock_probe);
    Ok(TuneHardwareEvaluation {
        evaluated_device,
        memory,
        mlock,
    })
}

fn evaluate_device(
    effective_hardware: &EffectiveTuneHardware,
    survey: &HardwareSurvey,
) -> Result<EvaluatedTuneDevice, TuneDiagnostic> {
    if let Some(device_request) = &effective_hardware.device_request {
        let gpu = resolve_requested_gpu(device_request, survey)?;
        return Ok(EvaluatedTuneDevice {
            target: TuneDeviceTarget::Gpu(to_gpu_target(gpu)),
            source: TuneDeviceSelectionSource::from(device_request.source),
            report_only_main_gpu: effective_hardware.report_only_main_gpu,
        });
    }

    if let Some(gpu) = survey
        .gpus
        .iter()
        .filter(|gpu| gpu.backend_device.is_some())
        .max_by_key(|gpu| {
            let capacity = VramCapacity::new(gpu.vram_bytes, gpu.reserved_bytes);
            capacity.allocatable_bytes()
        })
    {
        return Ok(EvaluatedTuneDevice {
            target: TuneDeviceTarget::Gpu(to_gpu_target(gpu)),
            source: TuneDeviceSelectionSource::SurveyDefault,
            report_only_main_gpu: effective_hardware.report_only_main_gpu,
        });
    }

    Ok(EvaluatedTuneDevice {
        target: TuneDeviceTarget::Cpu,
        source: TuneDeviceSelectionSource::CpuSystemRamFallback,
        report_only_main_gpu: effective_hardware.report_only_main_gpu,
    })
}

fn evaluate_memory_budget(
    evaluated_device: &EvaluatedTuneDevice,
    survey: &HardwareSurvey,
) -> TuneMemoryBudget {
    match &evaluated_device.target {
        TuneDeviceTarget::Gpu(gpu) => gpu_memory_budget(gpu.index, survey),
        TuneDeviceTarget::Cpu => TuneMemoryBudget {
            source: TuneMemorySource::SystemRamFallback,
            total_bytes: survey.vram_bytes,
            reserved_bytes: None,
            allocatable_bytes: survey.vram_bytes,
        },
    }
}

fn gpu_memory_budget(index: usize, survey: &HardwareSurvey) -> TuneMemoryBudget {
    let selected_gpu = survey
        .gpus
        .iter()
        .find(|candidate| candidate.index == index)
        .expect("evaluated GPU must come from the survey");
    let capacity = VramCapacity::new(selected_gpu.vram_bytes, selected_gpu.reserved_bytes);
    TuneMemoryBudget {
        source: TuneMemorySource::EvaluatedGpuVram,
        total_bytes: selected_gpu.vram_bytes,
        reserved_bytes: selected_gpu.reserved_bytes,
        allocatable_bytes: capacity.allocatable_bytes(),
    }
}

fn resolve_requested_gpu<'a>(
    request: &ConfiguredTuneDeviceRequest,
    survey: &'a HardwareSurvey,
) -> Result<&'a GpuFacts, TuneDiagnostic> {
    match request.source {
        ConfiguredDeviceSource::LegacyGpuId => {
            resolve_requested_pinned_gpu(request, survey).map_err(|(_err, diagnostic)| diagnostic)
        }
        ConfiguredDeviceSource::ModelHardwareDevice
        | ConfiguredDeviceSource::DefaultsHardwareDevice => {
            resolve_pinned_with_backend_fallback(request, survey)
        }
    }
}

fn resolve_pinned_with_backend_fallback<'a>(
    request: &ConfiguredTuneDeviceRequest,
    survey: &'a HardwareSurvey,
) -> Result<&'a GpuFacts, TuneDiagnostic> {
    let (pinned_error, pinned_diagnostic) = match resolve_requested_pinned_gpu(request, survey) {
        Ok(gpu) => return Ok(gpu),
        Err(tuple) => tuple,
    };

    if !pinned_diagnostic_allows_backend_fallback(&pinned_error) {
        return Err(pinned_diagnostic);
    }

    resolve_backend_device(request, survey).map_err(|backend_diagnostic| {
        combine_backend_fallback_error(pinned_diagnostic, backend_diagnostic)
    })
}

fn pinned_diagnostic_allows_backend_fallback(
    error: &mesh_llm_system::hardware::PinnedGpuResolverError,
) -> bool {
    matches!(error, mesh_llm_system::hardware::PinnedGpuResolverError::NonPinnableConfiguredId { .. })
}

fn combine_backend_fallback_error(
    mut pinned_diagnostic: TuneDiagnostic,
    backend_diagnostic: TuneDiagnostic,
) -> TuneDiagnostic {
    pinned_diagnostic.message.push_str(&format!(
        "; backend fallback also failed: {}",
        backend_diagnostic.message
    ));
    pinned_diagnostic
}

fn resolve_requested_pinned_gpu<'a>(
    request: &ConfiguredTuneDeviceRequest,
    survey: &'a HardwareSurvey,
) -> Result<&'a GpuFacts, (mesh_llm_system::hardware::PinnedGpuResolverError, TuneDiagnostic)> {
    resolve_pinned_gpu_strict(Some(&request.requested_value), &survey.gpus)
        .map_err(|error| {
            let diagnostic = missing_configured_device(request, survey, &error.to_string());
            (error, diagnostic)
        })
}

fn resolve_backend_device<'a>(
    request: &ConfiguredTuneDeviceRequest,
    survey: &'a HardwareSurvey,
) -> Result<&'a GpuFacts, TuneDiagnostic> {
    let matches = survey
        .gpus
        .iter()
        .filter(|gpu| {
            gpu.backend_device.as_deref().is_some_and(|backend_device| {
                backend_device.eq_ignore_ascii_case(&request.requested_value)
            })
        })
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [gpu] => Ok(*gpu),
        [] => Err(missing_configured_device(
            request,
            survey,
            "requested backend device was not present in the survey",
        )),
        _ => Err(missing_configured_device(
            request,
            survey,
            "requested backend device matched multiple surveyed GPUs",
        )),
    }
}

fn missing_configured_device(
    request: &ConfiguredTuneDeviceRequest,
    survey: &HardwareSurvey,
    detail: &str,
) -> TuneDiagnostic {
    let available_backend_devices = survey
        .gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.clone())
        .collect::<Vec<_>>();
    let available_pinnable_ids = survey
        .gpus
        .iter()
        .filter_map(|gpu| gpu.stable_id.clone())
        .filter(|stable_id| is_pinnable_stable_id(stable_id))
        .collect::<Vec<_>>();
    let field_name = match request.source {
        ConfiguredDeviceSource::ModelHardwareDevice => "per-model hardware.device",
        ConfiguredDeviceSource::DefaultsHardwareDevice => "defaults.hardware.device",
        ConfiguredDeviceSource::LegacyGpuId => "legacy gpu_id",
    };

    TuneDiagnostic {
        severity: TuneDiagnosticSeverity::Error,
        code: TuneDiagnosticCode::MissingConfiguredDevice,
        field: Some(TuneField::Device),
        message: format!(
            "{field_name} `{}` could not be evaluated for tune planning: {detail}. Available backend devices: {}; available pinnable GPU IDs: {}.",
            request.requested_value,
            display_list(&available_backend_devices),
            display_list(&available_pinnable_ids),
        ),
    }
}

fn to_gpu_target(gpu: &GpuFacts) -> TuneGpuTarget {
    TuneGpuTarget {
        index: gpu.index,
        display_name: gpu.display_name.clone(),
        stable_id: gpu.stable_id.clone(),
        backend_device: gpu.backend_device.clone(),
    }
}
