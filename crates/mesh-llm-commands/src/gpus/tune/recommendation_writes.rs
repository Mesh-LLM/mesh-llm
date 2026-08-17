use mesh_llm_config::{ModelConfigDefaults, ModelConfigEntry};

use crate::gpus::tune_hardware::types::TuneHardwareEvaluation;

use super::*;

pub(crate) fn push_kv_statuses(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    recommended: TuneKvCacheType,
) {
    for (field, edit, current_value) in [
        (
            TuneField::CacheTypeK,
            TuneConfigEdit::SetModelFitCacheTypeK(recommended),
            existing_cache_type_k(model_entry, defaults),
        ),
        (
            TuneField::CacheTypeV,
            TuneConfigEdit::SetModelFitCacheTypeV(recommended),
            existing_cache_type_v(model_entry, defaults),
        ),
    ] {
        if let Some((value, source)) = current_value {
            if apply_mode != TuneApplyMode::ReplaceExisting {
                plan.field_statuses.push(TuneFieldStatus::Preserved {
                    field,
                    reason: preserve_reason(source, field),
                });
                continue;
            }
            if tune_kv_cache_type(&value).is_none() {
                let diagnostic = invalid_existing_value_diagnostic(field, &value);
                plan.diagnostics.push(diagnostic.clone());
                plan.field_statuses
                    .push(TuneFieldStatus::Error { field, diagnostic });
                continue;
            }
        }
        plan.field_statuses.push(TuneFieldStatus::Applied {
            recommendation: TuneRecommendation {
                field,
                value: TuneRecommendedValue::KvCacheType(recommended),
                rationale: format!("model-size KV policy recommends {:?}", recommended)
                    .to_lowercase(),
            },
            edit,
        });
    }
}

pub(crate) fn push_flash_attention_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    recommended_cache_type_v: TuneKvCacheType,
) {
    if let Some(source) = existing_flash_attention_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::FlashAttention,
            reason: preserve_reason(source, TuneField::FlashAttention),
        });
        return;
    }
    let recommended = effective_flash_attention(&recommended_cache_type_v);
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::FlashAttention,
            value: TuneRecommendedValue::FlashAttention(recommended),
            rationale: "non-f16 V-cache defaults to explicit flash attention for stable startup"
                .to_string(),
        },
        edit: TuneConfigEdit::SetModelFitFlashAttention(recommended),
    });
}

pub(crate) fn push_context_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    fit: &PlannedFit,
) {
    if fit.diagnostic.is_some() {
        return;
    }
    if let Some(source) = existing_ctx_size_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::CtxSize,
            reason: preserve_reason(source, TuneField::CtxSize),
        });
        return;
    }
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::CtxSize,
            value: TuneRecommendedValue::ContextSize(fit.ctx_size),
            rationale: "largest static context that fits the selected memory budget".to_string(),
        },
        edit: TuneConfigEdit::SetModelFitCtxSize(fit.ctx_size),
    });
}

pub(crate) fn push_batch_statuses(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    fit: &PlannedFit,
) {
    if fit.diagnostic.is_some() {
        return;
    }
    for (field, value, edit, source) in [
        (
            TuneField::Batch,
            fit.batch,
            TuneConfigEdit::SetModelFitBatch(fit.batch),
            existing_batch_source(model_entry, defaults),
        ),
        (
            TuneField::Ubatch,
            fit.ubatch,
            TuneConfigEdit::SetModelFitUbatch(fit.ubatch),
            existing_ubatch_source(model_entry, defaults),
        ),
    ] {
        if let Some(source) = source
            && apply_mode != TuneApplyMode::ReplaceExisting
        {
            plan.field_statuses.push(TuneFieldStatus::Preserved {
                field,
                reason: preserve_reason(source, field),
            });
            continue;
        }
        let recommendation_value = match field {
            TuneField::Batch => TuneRecommendedValue::Batch(value),
            TuneField::Ubatch => TuneRecommendedValue::Ubatch(value),
            _ => unreachable!(),
        };
        plan.field_statuses.push(TuneFieldStatus::Applied {
            recommendation: TuneRecommendation {
                field,
                value: recommendation_value,
                rationale: "conservative startup batch shape bounded by planned context"
                    .to_string(),
            },
            edit,
        });
    }
}

pub(crate) fn push_gpu_layers_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    fit: &PlannedFit,
) {
    if fit.diagnostic.is_some() {
        return;
    }
    if let Some(source) = existing_gpu_layers_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::GpuLayers,
            reason: preserve_reason(source, TuneField::GpuLayers),
        });
        return;
    }
    let rationale = match fit.gpu_layers {
        TuneGpuLayersValue::All => {
            "full model plus minimum KV budget fits safely on the evaluated device".to_string()
        }
        TuneGpuLayersValue::Count(count) => {
            format!("only {count} GPU layers fit safely after reserving KV budget")
        }
    };
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::GpuLayers,
            value: TuneRecommendedValue::GpuLayers(fit.gpu_layers),
            rationale,
        },
        edit: TuneConfigEdit::SetHardwareGpuLayers(fit.gpu_layers),
    });
}

pub(crate) fn push_fit_target_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    hardware: &TuneHardwareEvaluation,
) {
    if let Some(source) = existing_fit_target_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::FitTargetMib,
            reason: preserve_reason(source, TuneField::FitTargetMib),
        });
        return;
    }
    let fit_target_mib = derive_fit_target_mib(hardware.memory.allocatable_bytes);
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::FitTargetMib,
            value: TuneRecommendedValue::FitTargetMib(fit_target_mib),
            rationale: "allocatable memory after the existing 2 GiB safety margin".to_string(),
        },
        edit: TuneConfigEdit::SetHardwareFitTargetMib(fit_target_mib),
    });
}

pub(crate) fn push_mmap_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) {
    if let Some(source) = existing_mmap_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::Mmap,
            reason: preserve_reason(source, TuneField::Mmap),
        });
        return;
    }
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::Mmap,
            value: TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto),
            rationale: "keep runtime mmap default unless benchmark tune selects an explicit value"
                .to_string(),
        },
        edit: TuneConfigEdit::SetHardwareMmap(TuneBoolOrAutoValue::Auto),
    });
}

pub(crate) fn push_mlock_status(
    plan: &mut TunePlan,
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
    hardware: &TuneHardwareEvaluation,
) {
    if let Some(source) = existing_mlock_source(model_entry, defaults)
        && apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::Mlock,
            reason: preserve_reason(source, TuneField::Mlock),
        });
        return;
    }
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::Mlock,
            value: TuneRecommendedValue::Bool(hardware.mlock.available),
            rationale: hardware.mlock.reason.clone(),
        },
        edit: TuneConfigEdit::SetHardwareMlock(hardware.mlock.available),
    });
}
