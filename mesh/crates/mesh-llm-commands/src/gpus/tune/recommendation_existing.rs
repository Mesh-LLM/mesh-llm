use mesh_llm_config::{ModelConfigDefaults, ModelConfigEntry};

use super::*;

#[derive(Clone, Copy)]
pub(crate) enum ExistingValueSource {
    ModelNested,
    ModelLegacy,
    Defaults,
}

pub(crate) fn preserve_reason(source: ExistingValueSource, field: TuneField) -> String {
    let rendered = match (source, field) {
        (ExistingValueSource::ModelNested, TuneField::CacheTypeK) => {
            "models[].model_fit.cache_type_k"
        }
        (ExistingValueSource::ModelNested, TuneField::CacheTypeV) => {
            "models[].model_fit.cache_type_v"
        }
        (ExistingValueSource::ModelNested, TuneField::FlashAttention) => {
            "models[].model_fit.flash_attention"
        }
        (ExistingValueSource::ModelNested, TuneField::CtxSize) => "models[].model_fit.ctx_size",
        (ExistingValueSource::ModelNested, TuneField::Batch) => "models[].model_fit.batch",
        (ExistingValueSource::ModelNested, TuneField::Ubatch) => "models[].model_fit.ubatch",
        (ExistingValueSource::ModelNested, TuneField::GpuLayers) => "models[].hardware.gpu_layers",
        (ExistingValueSource::ModelNested, TuneField::FitTargetMib) => {
            "models[].hardware.fit_target_mib"
        }
        (ExistingValueSource::ModelNested, TuneField::Mmap) => "models[].hardware.mmap",
        (ExistingValueSource::ModelNested, TuneField::Mlock) => "models[].hardware.mlock",
        (ExistingValueSource::ModelLegacy, TuneField::CacheTypeK) => "models[].cache_type_k",
        (ExistingValueSource::ModelLegacy, TuneField::CacheTypeV) => "models[].cache_type_v",
        (ExistingValueSource::ModelLegacy, TuneField::FlashAttention) => "models[].flash_attention",
        (ExistingValueSource::ModelLegacy, TuneField::CtxSize) => "models[].ctx_size",
        (ExistingValueSource::ModelLegacy, TuneField::Batch) => "models[].batch",
        (ExistingValueSource::ModelLegacy, TuneField::Ubatch) => "models[].ubatch",
        (ExistingValueSource::Defaults, TuneField::CacheTypeK) => "defaults.model_fit.cache_type_k",
        (ExistingValueSource::Defaults, TuneField::CacheTypeV) => "defaults.model_fit.cache_type_v",
        (ExistingValueSource::Defaults, TuneField::FlashAttention) => {
            "defaults.model_fit.flash_attention"
        }
        (ExistingValueSource::Defaults, TuneField::CtxSize) => "defaults.model_fit.ctx_size",
        (ExistingValueSource::Defaults, TuneField::Batch) => "defaults.model_fit.batch",
        (ExistingValueSource::Defaults, TuneField::Ubatch) => "defaults.model_fit.ubatch",
        (ExistingValueSource::Defaults, TuneField::GpuLayers) => "defaults.hardware.gpu_layers",
        (ExistingValueSource::Defaults, TuneField::FitTargetMib) => {
            "defaults.hardware.fit_target_mib"
        }
        (ExistingValueSource::Defaults, TuneField::Mmap) => "defaults.hardware.mmap",
        (ExistingValueSource::Defaults, TuneField::Mlock) => "defaults.hardware.mlock",
        (_, _) => "existing tune setting",
    };
    format!("existing {rendered} remains authoritative")
}

pub(crate) fn existing_cache_type_k(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<(String, ExistingValueSource)> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.cache_type_k.clone())
        .map(|value| (value, ExistingValueSource::ModelNested))
        .or_else(|| {
            model_entry?
                .cache_type_k
                .clone()
                .map(|value| (value, ExistingValueSource::ModelLegacy))
        })
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .cache_type_k
                .clone()
                .map(|value| (value, ExistingValueSource::Defaults))
        })
}

pub(crate) fn existing_cache_type_v(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<(String, ExistingValueSource)> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.cache_type_v.clone())
        .map(|value| (value, ExistingValueSource::ModelNested))
        .or_else(|| {
            model_entry?
                .cache_type_v
                .clone()
                .map(|value| (value, ExistingValueSource::ModelLegacy))
        })
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .cache_type_v
                .clone()
                .map(|value| (value, ExistingValueSource::Defaults))
        })
}

pub(crate) fn existing_flash_attention_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.flash_attention)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            model_entry?
                .flash_attention
                .map(|_| ExistingValueSource::ModelLegacy)
        })
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .flash_attention
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_ctx_size_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.ctx_size)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            model_entry?
                .ctx_size
                .map(|_| ExistingValueSource::ModelLegacy)
        })
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .ctx_size
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_batch_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.batch)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| model_entry?.batch.map(|_| ExistingValueSource::ModelLegacy))
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .batch
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_ubatch_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.model_fit.as_ref())
        .and_then(|fit| fit.ubatch)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            model_entry?
                .ubatch
                .map(|_| ExistingValueSource::ModelLegacy)
        })
        .or_else(|| {
            defaults?
                .model_fit
                .as_ref()?
                .ubatch
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_gpu_layers_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| parse_gpu_layers_value(hardware.gpu_layers.as_ref()))
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            defaults?
                .hardware
                .as_ref()?
                .gpu_layers
                .as_ref()
                .and_then(|value| parse_gpu_layers_value(Some(value)))
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_fit_target_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.fit_target_mib)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            defaults?
                .hardware
                .as_ref()?
                .fit_target_mib
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_mmap_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.mmap.as_ref())
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            defaults?
                .hardware
                .as_ref()?
                .mmap
                .as_ref()
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn existing_mlock_source(
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&ModelConfigDefaults>,
) -> Option<ExistingValueSource> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.mlock)
        .map(|_| ExistingValueSource::ModelNested)
        .or_else(|| {
            defaults?
                .hardware
                .as_ref()?
                .mlock
                .map(|_| ExistingValueSource::Defaults)
        })
}

pub(crate) fn parse_gpu_layers_value(
    value: Option<&mesh_llm_config::IntegerOrString>,
) -> Option<i32> {
    match value? {
        mesh_llm_config::IntegerOrString::Integer(value) => i32::try_from(*value).ok(),
        mesh_llm_config::IntegerOrString::String(value) if value.eq_ignore_ascii_case("auto") => {
            Some(-1)
        }
        mesh_llm_config::IntegerOrString::String(value) => value.parse::<i32>().ok(),
    }
}
