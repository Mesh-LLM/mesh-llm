use std::path::PathBuf;

use anyhow::{Result, bail};

use super::super::{KvCachePolicy, family_policy_for_model_path};
use super::request_defaults::resolve_request_defaults;
use super::speculative::resolve_speculative_config;
use super::support::{
    KvMacroDefaults, ThroughputMacroDefaults, bool_or_auto_value, derive_fit_target_mib,
    effective_flash_attention, has_explicit_prefill_controls, kv_macro_defaults, parse_gpu_layers,
    parse_kv_offload_string, pick_owned, pick_string, pick_string_owned, pick_value,
    reject_unsupported_hardware_controls, reject_unsupported_model_fit_controls,
    resolve_bool_or_auto, resolve_field_string, resolve_field_value, resolve_prefix_cache,
    throughput_macro_defaults,
};
use super::types::{
    BUILTIN_BATCH, BUILTIN_CTX_SIZE, BUILTIN_PARALLEL, BUILTIN_PREFILL_CHUNK_SIZE,
    BUILTIN_SAFETY_MARGIN_GB, BUILTIN_UBATCH, ResolvedHardwareConfig, ResolvedModelFitConfig,
    ResolvedMultimodalConfig, ResolvedSkippyConfig, ResolvedSkippyExecutionConfig,
    ResolvedThroughputConfig, SkippyConfigResolveRequest,
};
use crate::plugin::{
    BoolOrAuto, ModelConfigDefaults, ModelConfigEntry, ModelFitConfig, ThroughputConfig,
};

#[cfg(test)]
pub(crate) fn resolve_skippy_config(
    request: SkippyConfigResolveRequest<'_>,
) -> Result<ResolvedSkippyConfig> {
    let config_model_id = request.model_id.to_string();
    resolve_skippy_config_for_selector(request, Some(&config_model_id))
}

pub(crate) fn resolve_skippy_config_for_selector(
    request: SkippyConfigResolveRequest<'_>,
    config_model_id: Option<&str>,
) -> Result<ResolvedSkippyConfig> {
    resolve_skippy_config_with_context(ResolverContext::new_for_selector(request, config_model_id))
}

fn resolve_skippy_config_with_context(
    context: ResolverContext<'_>,
) -> Result<ResolvedSkippyConfig> {
    validate_supported_model_fit_controls(&context)?;
    validate_supported_hardware_controls(&context)?;

    // Guard the size-tiered default so a model that cannot load quantised KV
    // (Flash Attention off, or a head_dim not divisible by the block size)
    // resolves to f16 instead of failing the context build. Explicit config /
    // family defaults still take precedence in resolve_cache_type_* below and
    // are intentionally not guarded here.
    let kv_policy = KvCachePolicy::for_model_size(context.request.model_bytes)
        .guarded_for_model(context.request.compact_meta);
    let hardware = resolve_hardware_config(&context)?;
    let family_policy = family_policy_for_model_path(
        &hardware.resolved_model_path,
        Some(context.request.model_id),
    );
    let model_fit = resolve_model_fit_config(&context, kv_policy, &family_policy)?;
    let throughput = resolve_throughput_config(&context);
    let skippy = resolve_execution_config(&context);
    let speculative = resolve_speculative_config(
        context
            .model_entry
            .and_then(|entry| entry.speculative.as_ref()),
        context
            .defaults
            .and_then(|value| value.speculative.as_ref()),
        context.request.model_id,
        &hardware.resolved_model_path,
        context.request.package_generation,
    )?;
    let resolved_request = resolve_request_defaults(
        context.defaults,
        context.model_entry,
        context.request.request_defaults,
    )?;
    let multimodal = resolve_multimodal_config(&context)?;

    Ok(ResolvedSkippyConfig {
        model_id: context.request.model_id.to_string(),
        model_path: context.request.model_path.to_path_buf(),
        model_fit,
        hardware,
        throughput,
        skippy,
        speculative,
        request_defaults: resolved_request,
        multimodal,
    })
}

fn resolve_multimodal_config(context: &ResolverContext<'_>) -> Result<ResolvedMultimodalConfig> {
    let model = context
        .model_entry
        .and_then(|entry| entry.multimodal.as_ref());
    let defaults = context.defaults.and_then(|value| value.multimodal.as_ref());
    let projector_use_gpu = resolve_bool_or_auto(
        model
            .and_then(|value| value.mmproj_offload.as_ref())
            .or_else(|| defaults.and_then(|value| value.mmproj_offload.as_ref())),
        "multimodal.mmproj_offload",
    )?;
    let policy = pick_string_owned(
        model.and_then(|value| value.glm_dsa_policy.as_deref()),
        defaults.and_then(|value| value.glm_dsa_policy.as_deref()),
        Some("auto"),
    );
    let glm_dsa_policy = match policy.as_str() {
        "auto" => skippy_protocol::GlmDsaPolicy::Auto,
        "v1" => skippy_protocol::GlmDsaPolicy::V1,
        other => bail!("multimodal.glm_dsa_policy must be \"auto\" or \"v1\", got {other:?}"),
    };
    Ok(ResolvedMultimodalConfig {
        projector_url: pick_owned(
            model.and_then(|value| value.mmproj_url.clone()),
            defaults.and_then(|value| value.mmproj_url.clone()),
        ),
        projector_use_gpu,
        media_marker: pick_owned(
            model.and_then(|value| value.media_marker.clone()),
            defaults.and_then(|value| value.media_marker.clone()),
        ),
        image_min_tokens: pick_owned(
            model.and_then(|value| value.image_min_tokens),
            defaults.and_then(|value| value.image_min_tokens),
        ),
        image_max_tokens: pick_owned(
            model.and_then(|value| value.image_max_tokens),
            defaults.and_then(|value| value.image_max_tokens),
        ),
        batch_max_tokens: pick_owned(
            model.and_then(|value| value.batch_max_tokens),
            defaults.and_then(|value| value.batch_max_tokens),
        ),
        glm_dsa_policy,
        generation_signal_window: pick_owned(
            model.and_then(|value| value.generation_signal_window),
            defaults.and_then(|value| value.generation_signal_window),
        ),
    })
}

struct ResolverContext<'a> {
    request: SkippyConfigResolveRequest<'a>,
    model_entry: Option<&'a ModelConfigEntry>,
    defaults: Option<&'a ModelConfigDefaults>,
    model_fit: Option<&'a ModelFitConfig>,
    global_model_fit: Option<&'a ModelFitConfig>,
    model_throughput: Option<&'a ThroughputConfig>,
    global_throughput: Option<&'a ThroughputConfig>,
}

impl<'a> ResolverContext<'a> {
    fn new_for_selector(
        request: SkippyConfigResolveRequest<'a>,
        config_model_id: Option<&str>,
    ) -> Self {
        let model_entry = config_model_id.and_then(|selector| {
            request
                .mesh_config
                .models
                .iter()
                .find(|entry| entry.model == selector)
        });
        Self::with_model_entry(request, model_entry)
    }

    fn with_model_entry(
        request: SkippyConfigResolveRequest<'a>,
        model_entry: Option<&'a ModelConfigEntry>,
    ) -> Self {
        let mesh_config = request.mesh_config;
        let defaults = mesh_config.defaults.as_ref();
        let model_fit = model_entry.and_then(|entry| entry.model_fit.as_ref());
        let global_model_fit = defaults.and_then(|value| value.model_fit.as_ref());
        let model_throughput = model_entry.and_then(|entry| entry.throughput.as_ref());
        let global_throughput = defaults.and_then(|value| value.throughput.as_ref());

        Self {
            request,
            model_entry,
            defaults,
            model_fit,
            global_model_fit,
            model_throughput,
            global_throughput,
        }
    }
}

fn validate_supported_model_fit_controls(context: &ResolverContext<'_>) -> Result<()> {
    reject_unsupported_model_fit_controls(context.model_fit, "models[].model_fit")?;
    reject_unsupported_model_fit_controls(context.global_model_fit, "defaults.model_fit")
}

fn validate_supported_hardware_controls(context: &ResolverContext<'_>) -> Result<()> {
    reject_unsupported_hardware_controls(
        context
            .model_entry
            .and_then(|entry| entry.hardware.as_ref()),
        "models[].hardware",
    )?;
    reject_unsupported_hardware_controls(
        context
            .defaults
            .and_then(|defaults| defaults.hardware.as_ref()),
        "defaults.hardware",
    )
}

fn resolve_model_fit_config(
    context: &ResolverContext<'_>,
    kv_policy: KvCachePolicy,
    family_policy: &super::super::family_policy::FamilyPolicy,
) -> Result<ResolvedModelFitConfig> {
    let kv = resolve_kv_defaults(context, kv_policy);
    let throughput = resolve_throughput_defaults(context);

    let ctx_size = pick_value(
        context.model_fit.and_then(|fit| fit.ctx_size),
        context.global_model_fit.and_then(|fit| fit.ctx_size),
        BUILTIN_CTX_SIZE,
    );
    let batch = resolve_field_value(
        context.model_fit.and_then(|fit| fit.batch),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        context.global_model_fit.and_then(|fit| fit.batch),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.batch),
        BUILTIN_BATCH,
    );
    let ubatch = resolve_field_value(
        context.model_fit.and_then(|fit| fit.ubatch),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        context.global_model_fit.and_then(|fit| fit.ubatch),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.ubatch),
        BUILTIN_UBATCH,
    );
    let cache_type_k = resolve_cache_type_k(context, &kv, kv_policy, family_policy);
    let cache_type_v = resolve_cache_type_v(context, &kv, kv_policy, family_policy);
    let kv_offload = resolve_kv_offload(context, &kv);
    let kv_offload_resolved = parse_kv_offload_string(&kv_offload);
    let kv_unified = resolve_kv_unified(context)?;
    let swa_full = pick_owned(
        context.model_fit.and_then(|fit| fit.swa_full),
        context.global_model_fit.and_then(|fit| fit.swa_full),
    );
    let cache_idle_slots = pick_owned(
        context.model_fit.and_then(|fit| fit.cache_idle_slots),
        context
            .global_model_fit
            .and_then(|fit| fit.cache_idle_slots),
    );
    let flash_attention = context
        .model_fit
        .and_then(|fit| fit.flash_attention)
        .or(context.global_model_fit.and_then(|fit| fit.flash_attention))
        .unwrap_or_else(|| effective_flash_attention(&cache_type_v));
    let prefix_cache = resolve_prefix_cache(context.model_fit, context.global_model_fit)?;

    Ok(ResolvedModelFitConfig {
        ctx_size,
        batch,
        ubatch,
        cache_type_k,
        cache_type_v,
        kv_cache_policy: kv.effective_policy,
        prefix_cache,
        kv_offload,
        kv_offload_resolved,
        kv_unified,
        swa_full,
        cache_idle_slots,
        flash_attention,
    })
}

fn resolve_kv_unified(context: &ResolverContext<'_>) -> Result<Option<bool>> {
    let model = resolve_bool_or_auto(
        context.model_fit.and_then(|fit| fit.kv_unified.as_ref()),
        "model_fit.kv_unified",
    )?;
    if model.is_some() {
        return Ok(model);
    }
    resolve_bool_or_auto(
        context
            .global_model_fit
            .and_then(|fit| fit.kv_unified.as_ref()),
        "defaults.model_fit.kv_unified",
    )
}

struct KvDefaults {
    effective_policy: String,
    model_macro: Option<KvMacroDefaults>,
    global_macro: Option<KvMacroDefaults>,
}

fn resolve_kv_defaults(context: &ResolverContext<'_>, kv_policy: KvCachePolicy) -> KvDefaults {
    let model_policy = context
        .model_fit
        .and_then(|fit| fit.kv_cache_policy.as_deref());
    let global_policy = context
        .global_model_fit
        .and_then(|fit| fit.kv_cache_policy.as_deref());
    let effective_policy = pick_string(model_policy, global_policy, Some("balanced"));

    KvDefaults {
        effective_policy: effective_policy.to_string(),
        model_macro: model_policy.map(|policy| kv_macro_defaults(policy, kv_policy)),
        global_macro: global_policy.map(|policy| kv_macro_defaults(policy, kv_policy)),
    }
}

fn guarded_family_default_kv_cache_type(
    context: &ResolverContext<'_>,
    family_policy: &super::super::family_policy::FamilyPolicy,
) -> Option<&'static str> {
    family_policy
        .default_kv_cache_type
        .and_then(|default| {
            crate::models::gguf::GgufKvCacheQuant::from_llama_args(default, default)
        })
        .map(|quant| {
            context
                .request
                .compact_meta
                .map(|meta| meta.compatible_default_kv_cache_quant(quant))
                .unwrap_or(quant)
        })
        .map(|quant| quant.k.as_llama_arg())
}

fn resolve_cache_type_k(
    context: &ResolverContext<'_>,
    kv: &KvDefaults,
    kv_policy: KvCachePolicy,
    family_policy: &super::super::family_policy::FamilyPolicy,
) -> String {
    if let Some(explicit) = context
        .model_fit
        .and_then(|fit| non_auto_string(fit.cache_type_k.as_deref()))
    {
        return explicit.to_string();
    }
    // Guard the family default against the model's quantised-KV compatibility
    // so an unloadable family default degrades to f16 instead of failing the
    // context build. Explicit config above and below stays unguarded.
    if let Some(family_default) = guarded_family_default_kv_cache_type(context, family_policy) {
        if let Some(explicit) = context
            .global_model_fit
            .and_then(|fit| non_auto_string(fit.cache_type_k.as_deref()))
        {
            return explicit.to_string();
        }
        return family_default.to_string();
    }
    resolve_field_string(
        None,
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        context
            .global_model_fit
            .and_then(|fit| non_auto_string(fit.cache_type_k.as_deref())),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_k.as_deref()),
        kv_policy.cache_type_k(),
    )
}

fn resolve_cache_type_v(
    context: &ResolverContext<'_>,
    kv: &KvDefaults,
    kv_policy: KvCachePolicy,
    family_policy: &super::super::family_policy::FamilyPolicy,
) -> String {
    if let Some(explicit) = context
        .model_fit
        .and_then(|fit| non_auto_string(fit.cache_type_v.as_deref()))
    {
        return explicit.to_string();
    }
    if let Some(family_default) = guarded_family_default_kv_cache_type(context, family_policy) {
        if let Some(explicit) = context
            .global_model_fit
            .and_then(|fit| non_auto_string(fit.cache_type_v.as_deref()))
        {
            return explicit.to_string();
        }
        return family_default.to_string();
    }
    resolve_field_string(
        None,
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        context
            .global_model_fit
            .and_then(|fit| non_auto_string(fit.cache_type_v.as_deref())),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.cache_type_v.as_deref()),
        kv_policy.cache_type_v(),
    )
}

fn non_auto_string(value: Option<&str>) -> Option<&str> {
    value.filter(|item| !item.eq_ignore_ascii_case("auto"))
}

fn resolve_kv_offload(context: &ResolverContext<'_>, kv: &KvDefaults) -> String {
    let model_kv_offload = context
        .model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);
    let global_kv_offload = context
        .global_model_fit
        .and_then(|fit| fit.kv_offload.as_ref())
        .map(bool_or_auto_value);

    resolve_field_string(
        model_kv_offload.as_deref(),
        kv.model_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        global_kv_offload.as_deref(),
        kv.global_macro
            .as_ref()
            .and_then(|defaults| defaults.kv_offload.as_deref()),
        "auto",
    )
}

fn resolve_hardware_config(context: &ResolverContext<'_>) -> Result<ResolvedHardwareConfig> {
    let model_hardware = context
        .model_entry
        .and_then(|entry| entry.hardware.as_ref());
    let global_hardware = context.defaults.and_then(|value| value.hardware.as_ref());

    let device = pick_owned(
        model_hardware.and_then(|hardware| hardware.device.clone()),
        global_hardware.and_then(|hardware| hardware.device.clone()),
    );
    let gpu_layers = parse_gpu_layers(
        model_hardware.and_then(|hardware| hardware.gpu_layers.as_ref()),
        global_hardware.and_then(|hardware| hardware.gpu_layers.as_ref()),
    )?
    .unwrap_or(-1);
    let mmap = resolve_mmap_override(
        model_hardware.and_then(|hardware| hardware.mmap.as_ref()),
        global_hardware.and_then(|hardware| hardware.mmap.as_ref()),
    )?;
    let mlock = pick_owned(
        model_hardware.and_then(|hardware| hardware.mlock),
        global_hardware.and_then(|hardware| hardware.mlock),
    )
    .unwrap_or(false);
    let repack = pick_owned(
        model_hardware.and_then(|hardware| hardware.repack),
        global_hardware.and_then(|hardware| hardware.repack),
    )
    .unwrap_or(false);
    let op_offload = pick_owned(
        model_hardware.and_then(|hardware| hardware.op_offload),
        global_hardware.and_then(|hardware| hardware.op_offload),
    );
    let no_host_buffer = pick_owned(
        model_hardware.and_then(|hardware| hardware.no_host_buffer),
        global_hardware.and_then(|hardware| hardware.no_host_buffer),
    )
    .unwrap_or(false);
    let check_tensors = pick_owned(
        model_hardware.and_then(|hardware| hardware.check_tensors),
        global_hardware.and_then(|hardware| hardware.check_tensors),
    )
    .unwrap_or(false);
    let direct_io = pick_owned(
        model_hardware.and_then(|hardware| hardware.direct_io),
        global_hardware.and_then(|hardware| hardware.direct_io),
    )
    .unwrap_or(false);
    let main_gpu = pick_owned(
        model_hardware.and_then(|hardware| hardware.main_gpu),
        global_hardware.and_then(|hardware| hardware.main_gpu),
    );
    let split_mode = resolve_split_mode(
        model_hardware.and_then(|hardware| hardware.split_mode.as_deref()),
        global_hardware.and_then(|hardware| hardware.split_mode.as_deref()),
    )?;
    let safety_margin_gb = pick_owned(
        model_hardware.and_then(|hardware| hardware.safety_margin_gb),
        global_hardware.and_then(|hardware| hardware.safety_margin_gb),
    )
    .unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    let fit_target_mib = pick_owned(
        model_hardware.and_then(|hardware| hardware.fit_target_mib),
        global_hardware.and_then(|hardware| hardware.fit_target_mib),
    )
    .or_else(|| derive_fit_target_mib(context.request.allocatable_memory_bytes, safety_margin_gb));
    let resolved_model_path = pick_owned(
        model_hardware.and_then(|hardware| hardware.model_path.clone()),
        global_hardware.and_then(|hardware| hardware.model_path.clone()),
    )
    .map(PathBuf::from)
    .unwrap_or_else(|| context.request.model_path.to_path_buf());
    let projector_path = resolve_projector_path(context);
    let stage_layer_start = pick_owned(
        model_hardware.and_then(|hardware| hardware.stage_layer_start),
        global_hardware.and_then(|hardware| hardware.stage_layer_start),
    );
    let stage_layer_end = pick_owned(
        model_hardware.and_then(|hardware| hardware.stage_layer_end),
        global_hardware.and_then(|hardware| hardware.stage_layer_end),
    );

    Ok(ResolvedHardwareConfig {
        device,
        gpu_layers,
        mmap,
        mlock,
        repack,
        op_offload,
        no_host_buffer,
        check_tensors,
        direct_io,
        main_gpu,
        split_mode,
        fit_target_mib,
        resolved_model_path,
        projector_path,
        stage_layer_start,
        stage_layer_end,
    })
}

fn resolve_mmap_override(
    model_mmap: Option<&BoolOrAuto>,
    global_mmap: Option<&BoolOrAuto>,
) -> Result<Option<bool>> {
    Ok(match model_mmap.or(global_mmap) {
        None => None,
        Some(BoolOrAuto::Bool(value)) => Some(*value),
        Some(BoolOrAuto::String(value)) if value.eq_ignore_ascii_case("auto") => None,
        Some(BoolOrAuto::String(_)) => bail!("hardware.mmap must be a boolean or \"auto\""),
    })
}

fn resolve_split_mode(
    model_split_mode: Option<&str>,
    global_split_mode: Option<&str>,
) -> Result<skippy_protocol::SplitMode> {
    match model_split_mode.or(global_split_mode) {
        None => Ok(skippy_protocol::SplitMode::Auto),
        Some(value) if value.eq_ignore_ascii_case("auto") => Ok(skippy_protocol::SplitMode::Auto),
        Some(value) if value.eq_ignore_ascii_case("none") => Ok(skippy_protocol::SplitMode::None),
        Some(value) if value.eq_ignore_ascii_case("layer") => Ok(skippy_protocol::SplitMode::Layer),
        Some(value) if value.eq_ignore_ascii_case("row") => Ok(skippy_protocol::SplitMode::Row),
        Some(value) if value.eq_ignore_ascii_case("tensor") => {
            Ok(skippy_protocol::SplitMode::Tensor)
        }
        Some(other) => bail!(
            "hardware.split_mode must be \"auto\", \"none\", \"layer\", \"row\", or \"tensor\", got {other:?}"
        ),
    }
}

fn resolve_projector_path(context: &ResolverContext<'_>) -> Option<PathBuf> {
    pick_owned(
        context
            .model_entry
            .and_then(|entry| entry.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                context
                    .model_entry
                    .and_then(|entry| entry.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
        context
            .defaults
            .and_then(|value| value.multimodal.as_ref())
            .and_then(|multimodal| multimodal.mmproj.clone())
            .or_else(|| {
                context
                    .defaults
                    .and_then(|value| value.hardware.as_ref())
                    .and_then(|hardware| hardware.mmproj.clone())
            }),
    )
    .map(PathBuf::from)
}

struct ThroughputDefaults {
    effective_profile: String,
    model_macro: Option<ThroughputMacroDefaults>,
    global_macro: Option<ThroughputMacroDefaults>,
}

fn resolve_throughput_defaults(context: &ResolverContext<'_>) -> ThroughputDefaults {
    let model_profile = context
        .model_throughput
        .and_then(|throughput| throughput.tuning_profile.as_deref());
    let global_profile = context
        .global_throughput
        .and_then(|throughput| throughput.tuning_profile.as_deref());
    let effective_profile = pick_string(model_profile, global_profile, Some("balanced"));

    ThroughputDefaults {
        effective_profile: effective_profile.to_string(),
        model_macro: model_profile.map(throughput_macro_defaults),
        global_macro: global_profile.map(throughput_macro_defaults),
    }
}

fn resolve_throughput_config(context: &ResolverContext<'_>) -> ResolvedThroughputConfig {
    let throughput = resolve_throughput_defaults(context);
    let parallel = resolve_field_value(
        context
            .model_throughput
            .and_then(|throughput| throughput.parallel),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        context
            .global_throughput
            .and_then(|throughput| throughput.parallel),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.parallel),
        BUILTIN_PARALLEL,
    );
    let continuous_batching = resolve_continuous_batching(context, &throughput);
    let threads = pick_owned(
        context
            .model_throughput
            .and_then(|throughput| throughput.threads),
        context
            .global_throughput
            .and_then(|throughput| throughput.threads),
    );
    let threads_batch = pick_owned(
        context
            .model_throughput
            .and_then(|throughput| throughput.threads_batch),
        context
            .global_throughput
            .and_then(|throughput| throughput.threads_batch),
    );

    ResolvedThroughputConfig {
        parallel,
        continuous_batching,
        threads,
        threads_batch,
        tuning_profile: throughput.effective_profile,
    }
}

fn resolve_continuous_batching(
    context: &ResolverContext<'_>,
    throughput: &ThroughputDefaults,
) -> String {
    let model_continuous_batching = context
        .model_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);
    let global_continuous_batching = context
        .global_throughput
        .and_then(|throughput| throughput.continuous_batching.as_ref())
        .map(bool_or_auto_value);

    resolve_field_string(
        model_continuous_batching.as_deref(),
        throughput
            .model_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        global_continuous_batching.as_deref(),
        throughput
            .global_macro
            .as_ref()
            .and_then(|defaults| defaults.continuous_batching.as_deref()),
        "auto",
    )
}

fn resolve_execution_config(context: &ResolverContext<'_>) -> ResolvedSkippyExecutionConfig {
    let model_skippy = context.model_entry.and_then(|entry| entry.skippy.as_ref());
    let global_skippy = context.defaults.and_then(|value| value.skippy.as_ref());

    let binary_stage_transport = pick_string_owned(
        model_skippy.and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        global_skippy.and_then(|skippy| skippy.binary_stage_transport.as_deref()),
        Some("auto"),
    );
    let prefill_chunking = pick_string_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunking.as_deref()),
        global_skippy.and_then(|skippy| skippy.prefill_chunking.as_deref()),
        Some("fixed"),
    );
    let prefill_chunk_size = pick_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunk_size),
        global_skippy.and_then(|skippy| skippy.prefill_chunk_size),
    )
    .map(|value| value as usize)
    .unwrap_or(BUILTIN_PREFILL_CHUNK_SIZE);
    let prefill_chunk_schedule = pick_owned(
        model_skippy.and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
        global_skippy.and_then(|skippy| skippy.prefill_chunk_schedule.clone()),
    );
    let prefill_controls_explicit = model_skippy.is_some_and(has_explicit_prefill_controls)
        || global_skippy.is_some_and(has_explicit_prefill_controls);

    ResolvedSkippyExecutionConfig {
        binary_stage_transport,
        prefill_chunking,
        prefill_chunk_size,
        prefill_chunk_schedule,
        prefill_controls_explicit,
        lifecycle_startup_timeout_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_startup_timeout_ms),
        ),
        lifecycle_readiness_interval_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_readiness_interval_ms),
        ),
        lifecycle_health_interval_ms: pick_owned(
            model_skippy.and_then(|skippy| skippy.lifecycle_health_interval_ms),
            global_skippy.and_then(|skippy| skippy.lifecycle_health_interval_ms),
        ),
    }
}
