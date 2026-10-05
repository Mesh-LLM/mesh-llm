use mesh_llm_config::{
    BoolOrAuto as OutputBoolOrAuto, FlashAttentionType as OutputFlashAttentionType,
    IntegerOrString as OutputIntegerOrString, ModelConfigDefaults as OutputModelConfigDefaults,
    ModelConfigEntry as OutputModelConfigEntry,
};

use super::*;

pub(crate) fn render_apply_mode(mode: TuneApplyMode) -> &'static str {
    match mode {
        TuneApplyMode::Review => "review",
        TuneApplyMode::ApplyMissing => "apply-missing",
        TuneApplyMode::ReplaceExisting => "replace-existing",
        TuneApplyMode::LaunchArgs => "launch-args",
    }
}

pub(crate) fn render_target_status(status: TuneTargetStatus) -> &'static str {
    match status {
        TuneTargetStatus::Ready => "ready",
        TuneTargetStatus::Written => "written",
        TuneTargetStatus::Skipped => "skipped",
        TuneTargetStatus::Failed => "failed",
    }
}

pub(crate) fn render_field_name(field: TuneField) -> &'static str {
    match field {
        TuneField::CacheTypeK => "cache_type_k",
        TuneField::CacheTypeV => "cache_type_v",
        TuneField::FlashAttention => "flash_attention",
        TuneField::CtxSize => "ctx_size",
        TuneField::Batch => "batch",
        TuneField::Ubatch => "ubatch",
        TuneField::GpuLayers => "gpu_layers",
        TuneField::FitTargetMib => "fit_target_mib",
        TuneField::Device => "device",
        TuneField::Mmap => "mmap",
        TuneField::Mlock => "mlock",
        TuneField::CpuMoe => "cpu_moe",
        TuneField::NCpuMoe => "n_cpu_moe",
        TuneField::TensorSplit => "tensor_split",
        TuneField::Placement => "placement",
        TuneField::Defaults => "defaults",
    }
}

pub(crate) fn render_recommended_value(value: &TuneRecommendedValue) -> String {
    match value {
        TuneRecommendedValue::KvCacheType(value) => match value {
            TuneKvCacheType::F16 => "f16".to_string(),
            TuneKvCacheType::Q8_0 => "q8_0".to_string(),
            TuneKvCacheType::Q4_0 => "q4_0".to_string(),
        },
        TuneRecommendedValue::FlashAttention(value) => match value {
            TuneFlashAttentionValue::Enabled => "enabled".to_string(),
            TuneFlashAttentionValue::Disabled => "disabled".to_string(),
        },
        TuneRecommendedValue::ContextSize(value) => value.to_string(),
        TuneRecommendedValue::Batch(value) => value.to_string(),
        TuneRecommendedValue::Ubatch(value) => value.to_string(),
        TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::All) => "all".to_string(),
        TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::Count(value)) => value.to_string(),
        TuneRecommendedValue::FitTargetMib(value) => value.to_string(),
        TuneRecommendedValue::Device(value) => value.clone(),
        TuneRecommendedValue::Bool(value) => value.to_string(),
        TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Enabled) => "enabled".to_string(),
        TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Disabled) => "disabled".to_string(),
        TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto) => "auto".to_string(),
    }
}

pub(crate) fn render_benchmark_candidate(candidate: &TuneBenchmarkCandidate) -> String {
    let mut s = format!(
        "ctx={} batch={} ubatch={} cache_k={} cache_v={} mmap={} mlock={} spec={}",
        candidate.ctx_size,
        candidate.batch,
        candidate.ubatch,
        render_cache_type(candidate.cache_type_k),
        render_cache_type(candidate.cache_type_v),
        render_benchmark_bool_or_auto(candidate.mmap),
        candidate.mlock,
        render_benchmark_speculative(&candidate.speculative),
    );
    if let Some(fa) = candidate.flash_attention {
        let fa_str = match fa {
            TuneFlashAttentionValue::Enabled => "enabled",
            TuneFlashAttentionValue::Disabled => "disabled",
        };
        s.push_str(&format!(" flash={fa_str}"));
    }
    s
}

fn render_cache_type(value: TuneKvCacheType) -> &'static str {
    match value {
        TuneKvCacheType::F16 => "f16",
        TuneKvCacheType::Q8_0 => "q8_0",
        TuneKvCacheType::Q4_0 => "q4_0",
    }
}

pub(crate) fn render_benchmark_speculative(
    speculative: &TuneBenchmarkSpeculativeCandidate,
) -> String {
    fn append_prob(suffix: &mut String, name: &str, value: Option<f64>) {
        if let Some(value) = value {
            suffix.push_str(&format!(":{name}={value:.6}"));
        }
    }
    match speculative {
        TuneBenchmarkSpeculativeCandidate::Disabled => "disabled".to_string(),
        TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => {
            let mut base = draft_model.as_ref().map_or_else(
                || format!("mtp:min={draft_min_tokens}:max={draft_max_tokens}"),
                |path| format!("mtp:path={path}:min={draft_min_tokens}:max={draft_max_tokens}"),
            );
            append_prob(&mut base, "accept", *draft_acceptance_threshold);
            append_prob(&mut base, "split", *draft_split_probability);
            base
        }
        TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model,
            draft_max_tokens,
            draft_min_tokens,
            draft_acceptance_threshold,
            draft_split_probability,
        } => {
            let mut base = match draft_min_tokens {
                Some(draft_min_tokens) => format!(
                    "draft:path={draft_model}:min={draft_min_tokens}:max={draft_max_tokens}"
                ),
                None => format!("draft:path={draft_model}:max={draft_max_tokens}"),
            };
            append_prob(&mut base, "accept", *draft_acceptance_threshold);
            append_prob(&mut base, "split", *draft_split_probability);
            base
        }
        TuneBenchmarkSpeculativeCandidate::MtpNgram {
            ngram_min,
            ngram_max,
        } => format!("mtp+ngram:min={ngram_min}:max={ngram_max}"),
    }
}

pub(crate) fn render_benchmark_bool_or_auto(value: TuneBoolOrAutoValue) -> &'static str {
    match value {
        TuneBoolOrAutoValue::Auto => "auto",
        TuneBoolOrAutoValue::Enabled => "enabled",
        TuneBoolOrAutoValue::Disabled => "disabled",
    }
}

pub(crate) fn preserved_value(
    field: TuneField,
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<TuneRecommendedValue> {
    match field {
        TuneField::CacheTypeK => existing_cache_type_k(model_entry, defaults)
            .and_then(|(value, _)| tune_kv_cache_type(&value))
            .map(TuneRecommendedValue::KvCacheType),
        TuneField::CacheTypeV => existing_cache_type_v(model_entry, defaults)
            .and_then(|(value, _)| tune_kv_cache_type(&value))
            .map(TuneRecommendedValue::KvCacheType),
        TuneField::FlashAttention => {
            let flash_attention = model_entry
                .and_then(|entry| entry.model_fit.as_ref())
                .and_then(|fit| fit.flash_attention)
                .or(model_entry.and_then(|entry| entry.flash_attention))
                .or_else(|| defaults?.model_fit.as_ref()?.flash_attention);
            flash_attention.map(render_flash_attention_value)
        }
        TuneField::CtxSize => preserved_model_fit_u32(model_entry, defaults, TuneField::CtxSize)
            .map(TuneRecommendedValue::ContextSize),
        TuneField::Batch => preserved_model_fit_u32(model_entry, defaults, TuneField::Batch)
            .map(TuneRecommendedValue::Batch),
        TuneField::Ubatch => preserved_model_fit_u32(model_entry, defaults, TuneField::Ubatch)
            .map(TuneRecommendedValue::Ubatch),
        TuneField::GpuLayers => preserved_gpu_layers(model_entry, defaults),
        TuneField::FitTargetMib => {
            preserved_fit_target_mib(model_entry, defaults).map(TuneRecommendedValue::FitTargetMib)
        }
        TuneField::Device => {
            preserved_device(model_entry, defaults).map(TuneRecommendedValue::Device)
        }
        TuneField::Mmap => {
            preserved_mmap(model_entry, defaults).map(TuneRecommendedValue::BoolOrAuto)
        }
        TuneField::Mlock => preserved_mlock(model_entry, defaults).map(TuneRecommendedValue::Bool),
        TuneField::CpuMoe
        | TuneField::NCpuMoe
        | TuneField::TensorSplit
        | TuneField::Placement
        | TuneField::Defaults => None,
    }
}

fn render_flash_attention_value(value: OutputFlashAttentionType) -> TuneRecommendedValue {
    match value {
        OutputFlashAttentionType::Enabled => {
            TuneRecommendedValue::FlashAttention(TuneFlashAttentionValue::Enabled)
        }
        OutputFlashAttentionType::Disabled => {
            TuneRecommendedValue::FlashAttention(TuneFlashAttentionValue::Disabled)
        }
        OutputFlashAttentionType::Auto => {
            TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto)
        }
    }
}

fn preserved_model_fit_u32(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
    field: TuneField,
) -> Option<u32> {
    match field {
        TuneField::CtxSize => model_entry
            .and_then(|entry| entry.model_fit.as_ref())
            .and_then(|fit| fit.ctx_size)
            .or(model_entry.and_then(|entry| entry.ctx_size))
            .or_else(|| defaults?.model_fit.as_ref()?.ctx_size),
        TuneField::Batch => model_entry
            .and_then(|entry| entry.model_fit.as_ref())
            .and_then(|fit| fit.batch)
            .or(model_entry.and_then(|entry| entry.batch))
            .or_else(|| defaults?.model_fit.as_ref()?.batch),
        TuneField::Ubatch => model_entry
            .and_then(|entry| entry.model_fit.as_ref())
            .and_then(|fit| fit.ubatch)
            .or(model_entry.and_then(|entry| entry.ubatch))
            .or_else(|| defaults?.model_fit.as_ref()?.ubatch),
        TuneField::CacheTypeK
        | TuneField::CacheTypeV
        | TuneField::FlashAttention
        | TuneField::GpuLayers
        | TuneField::FitTargetMib
        | TuneField::Device
        | TuneField::Mmap
        | TuneField::Mlock
        | TuneField::CpuMoe
        | TuneField::NCpuMoe
        | TuneField::TensorSplit
        | TuneField::Placement
        | TuneField::Defaults => None,
    }
}

fn preserved_gpu_layers(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<TuneRecommendedValue> {
    let gpu_layers = model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| parse_gpu_layers_value_for_output(hardware.gpu_layers.as_ref()))
        .or_else(|| {
            defaults?
                .hardware
                .as_ref()?
                .gpu_layers
                .as_ref()
                .and_then(|value| parse_gpu_layers_value_for_output(Some(value)))
        })?;
    if gpu_layers == -1 {
        return Some(TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::All));
    }
    u32::try_from(gpu_layers)
        .ok()
        .map(TuneGpuLayersValue::Count)
        .map(TuneRecommendedValue::GpuLayers)
}

fn preserved_fit_target_mib(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<u64> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.fit_target_mib)
        .or_else(|| defaults?.hardware.as_ref()?.fit_target_mib)
}

fn preserved_device(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<String> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.device.clone())
        .or_else(|| model_entry.and_then(|entry| entry.gpu_id.clone()))
        .or_else(|| defaults?.hardware.as_ref()?.device.clone())
}

fn preserved_mmap(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<TuneBoolOrAutoValue> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.mmap.as_ref())
        .or_else(|| defaults?.hardware.as_ref()?.mmap.as_ref())
        .and_then(render_bool_or_auto_value)
}

fn preserved_mlock(
    model_entry: Option<&OutputModelConfigEntry>,
    defaults: Option<&OutputModelConfigDefaults>,
) -> Option<bool> {
    model_entry
        .and_then(|entry| entry.hardware.as_ref())
        .and_then(|hardware| hardware.mlock)
        .or_else(|| defaults?.hardware.as_ref()?.mlock)
}

fn render_bool_or_auto_value(value: &OutputBoolOrAuto) -> Option<TuneBoolOrAutoValue> {
    match value {
        OutputBoolOrAuto::Bool(true) => Some(TuneBoolOrAutoValue::Enabled),
        OutputBoolOrAuto::Bool(false) => Some(TuneBoolOrAutoValue::Disabled),
        OutputBoolOrAuto::String(value) if value.eq_ignore_ascii_case("auto") => {
            Some(TuneBoolOrAutoValue::Auto)
        }
        OutputBoolOrAuto::String(_) => None,
    }
}

fn parse_gpu_layers_value_for_output(value: Option<&OutputIntegerOrString>) -> Option<i32> {
    match value? {
        OutputIntegerOrString::Integer(value) => i32::try_from(*value).ok(),
        OutputIntegerOrString::String(value) if value.eq_ignore_ascii_case("auto") => Some(-1),
        OutputIntegerOrString::String(value) => value.parse::<i32>().ok(),
    }
}
