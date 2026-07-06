use crate::gpus::tune_hardware::types::{TuneDeviceTarget, TuneHardwareEvaluation};
use crate::gpus::tune_resolver::{ResolvedTuneTarget, TuneTargetSelection};
use mesh_llm_config::{MeshConfig, ModelConfigEntry};
use mesh_llm_system::hardware::HardwareSurvey;

use super::*;

pub(crate) struct TuneRecommendationInput<'a> {
    pub(crate) apply_mode: TuneApplyMode,
    pub(crate) config: &'a MeshConfig,
    pub(crate) target: &'a ResolvedTuneTarget,
    pub(crate) metadata: &'a TuneGgufMetadata,
    pub(crate) hardware: &'a TuneHardwareEvaluation,
    pub(crate) survey: &'a HardwareSurvey,
}

pub(crate) fn build_tune_plan(input: TuneRecommendationInput<'_>) -> TunePlan {
    let model_entry = matched_model_entry(input.config, input.target);
    let defaults = input.config.defaults.as_ref();
    let fit = plan_fit(input.metadata, input.hardware, input.survey);
    let recommended_quant = recommended_kv_cache_quant(input.metadata.model_bytes);
    let recommended_cache_type = tune_kv_cache_type_from_quant(recommended_quant);

    let mut plan = TunePlan {
        target: plan_target(input.target, model_entry),
        apply_mode: input.apply_mode,
        field_statuses: Vec::new(),
        diagnostics: input.hardware.diagnostics(),
    };

    if fit.diagnostic.is_none() {
        push_kv_statuses(
            &mut plan,
            input.apply_mode,
            model_entry,
            defaults,
            recommended_cache_type,
        );
        push_flash_attention_status(
            &mut plan,
            input.apply_mode,
            model_entry,
            defaults,
            recommended_cache_type,
        );
        push_context_status(&mut plan, input.apply_mode, model_entry, defaults, &fit);
        push_batch_statuses(&mut plan, input.apply_mode, model_entry, defaults, &fit);
        push_gpu_layers_status(&mut plan, input.apply_mode, model_entry, defaults, &fit);
        push_fit_target_status(
            &mut plan,
            input.apply_mode,
            model_entry,
            defaults,
            input.hardware,
        );
        push_mmap_status(&mut plan, input.apply_mode, model_entry, defaults);
        push_mlock_status(
            &mut plan,
            input.apply_mode,
            model_entry,
            defaults,
            input.hardware,
        );
    }
    plan.field_statuses
        .push(input.hardware.device_field_status());
    push_cpu_moe_statuses(&mut plan, input.metadata);
    plan.field_statuses.push(unsupported_status(
        TuneField::TensorSplit,
        "tensor_split remains unsupported by the pinned runtime in v1",
    ));
    plan.field_statuses.push(unsupported_status(
        TuneField::Placement,
        "placement remains unsupported by the pinned runtime in v1",
    ));
    plan.field_statuses.push(TuneFieldStatus::Preserved {
        field: TuneField::Defaults,
        reason: "defaults.* remains preserve-only in v1 and is never rewritten by tune".to_string(),
    });
    if let Some(diagnostic) = fit.diagnostic {
        plan.diagnostics.push(diagnostic.clone());
        plan.field_statuses.push(TuneFieldStatus::Error {
            field: TuneField::CtxSize,
            diagnostic: diagnostic.clone(),
        });
        plan.field_statuses.push(TuneFieldStatus::Error {
            field: TuneField::GpuLayers,
            diagnostic,
        });
    }
    plan
}

#[derive(Clone)]
pub(crate) struct PlannedFit {
    pub(crate) ctx_size: u32,
    pub(crate) batch: u32,
    pub(crate) ubatch: u32,
    pub(crate) gpu_layers: TuneGpuLayersValue,
    pub(crate) diagnostic: Option<TuneDiagnostic>,
}

fn find_partial_gpu_layers_fit(
    metadata: &TuneGgufMetadata,
    layer_count: u32,
    selected_budget: u64,
    kv_bytes_per_token: u64,
    quant: model_artifact::gguf::GgufKvCacheQuant,
) -> Option<(TuneGpuLayersValue, u32)> {
    let bytes_per_layer = resident_model_bytes_for_layers(metadata.model_bytes, layer_count, 1);
    let max_layers = selected_budget
        .checked_div(bytes_per_layer)
        .map(|layers| layers.min(u64::from(layer_count)) as u32)
        .unwrap_or(0);
    for layers in (1..=max_layers).rev() {
        let resident = resident_model_bytes_for_layers(metadata.model_bytes, layer_count, layers);
        if !minimum_context_fits(resident, selected_budget, kv_bytes_per_token) {
            continue;
        }
        return Some((
            TuneGpuLayersValue::Count(layers),
            planned_context_length(&metadata.compact_meta, resident, selected_budget, quant),
        ));
    }
    None
}

fn plan_fit(
    metadata: &TuneGgufMetadata,
    hardware: &TuneHardwareEvaluation,
    survey: &HardwareSurvey,
) -> PlannedFit {
    let quant = recommended_kv_cache_quant(metadata.model_bytes);
    let kv_bytes_per_token = quant
        .kv_cache_bytes_per_token(&metadata.compact_meta)
        .unwrap_or_default();
    let selected_budget =
        derive_fit_target_mib(hardware.memory.allocatable_bytes).saturating_mul(1024 * 1024);
    let cpu_budget = derive_fit_target_mib(survey.vram_bytes).saturating_mul(1024 * 1024);
    let layer_count = metadata.compact_meta.layer_count.max(1);
    let cpu_can_fit = minimum_context_fits(metadata.model_bytes, cpu_budget, kv_bytes_per_token);

    let chosen = match &hardware.evaluated_device.target {
        TuneDeviceTarget::Cpu => {
            if cpu_can_fit {
                Some((
                    TuneGpuLayersValue::Count(0),
                    planned_context_length(
                        &metadata.compact_meta,
                        metadata.model_bytes,
                        cpu_budget,
                        quant,
                    ),
                ))
            } else {
                None
            }
        }
        TuneDeviceTarget::Gpu(_) => {
            if minimum_context_fits(metadata.model_bytes, selected_budget, kv_bytes_per_token) {
                Some((
                    TuneGpuLayersValue::All,
                    planned_context_length(
                        &metadata.compact_meta,
                        metadata.model_bytes,
                        selected_budget,
                        quant,
                    ),
                ))
            } else if !cpu_can_fit {
                None
            } else {
                find_partial_gpu_layers_fit(
                    metadata,
                    layer_count,
                    selected_budget,
                    kv_bytes_per_token,
                    quant,
                )
            }
        }
    };

    match chosen {
        Some((gpu_layers, ctx_size)) => PlannedFit {
            ctx_size,
            batch: recommended_batch(ctx_size),
            ubatch: recommended_ubatch(recommended_batch(ctx_size)),
            gpu_layers,
            diagnostic: None,
        },
        None => PlannedFit {
            ctx_size: 0,
            batch: 0,
            ubatch: 0,
            gpu_layers: TuneGpuLayersValue::Count(0),
            diagnostic: Some(insufficient_memory_diagnostic(
                &hardware.memory.source,
                selected_budget,
                metadata.model_bytes,
                kv_bytes_per_token,
            )),
        },
    }
}

pub(crate) fn matched_model_entry<'a>(
    config: &'a MeshConfig,
    target: &ResolvedTuneTarget,
) -> Option<&'a ModelConfigEntry> {
    config.models.get(target.config_matches.first()?.row_index)
}

fn plan_target(target: &ResolvedTuneTarget, model_entry: Option<&ModelConfigEntry>) -> TuneTarget {
    TuneTarget {
        requested: target.requested_input.clone(),
        resolved: Some(target.resolved_path.display().to_string()),
        config_model_ref: model_entry.map(|entry| entry.model.clone()).or_else(|| {
            matches!(target.selection, TuneTargetSelection::Configured)
                .then(|| target.canonical_model_ref.clone())
        }),
        derived_profile: model_entry.map(ModelConfigEntry::derived_profile),
    }
}
