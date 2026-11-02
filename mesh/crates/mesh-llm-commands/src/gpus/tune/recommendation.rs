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
    let placement = placement_constraints(input.apply_mode, model_entry, defaults);
    let fit = plan_fit(input.metadata, input.hardware, input.survey, placement);
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
    push_cpu_moe_statuses(
        &mut plan,
        input.metadata,
        CpuMoeTuneContext {
            apply_mode: input.apply_mode,
            model_entry,
            defaults,
            host_expert_blocks: fit
                .diagnostic
                .is_none()
                .then_some(fit.host_expert_blocks)
                .flatten(),
        },
    );
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
    /// Model blocks, from block 0, whose routed experts must load into host
    /// memory for every layer to stay on the accelerator.
    pub(crate) host_expert_blocks: Option<u32>,
    pub(crate) diagnostic: Option<TuneDiagnostic>,
}

/// Fit a mixture-of-experts model on the accelerator by moving the routed
/// experts of the fewest leading blocks into host memory.
///
/// The moved bytes are the exact per-block expert sums from the tensor
/// scan: models with dense leading blocks hold no expert bytes in their
/// first blocks, so an even spread would understate what stays resident.
/// Without per-block attribution there is no safe estimate, so no
/// host-expert recommendation is made. Host memory must hold every moved
/// expert.
fn find_host_expert_fit(
    metadata: &TuneGgufMetadata,
    layer_count: u32,
    selected_budget: u64,
    host_budget: u64,
    kv_bytes_per_token: u64,
    quant: skippy_model_artifact::gguf::GgufKvCacheQuant,
) -> Option<(u32, u32)> {
    let TuneTensorProfile::Exact(profile) = &metadata.tensor_profile else {
        return None;
    };
    if profile.expert_count == 0 || profile.expert_tensor_bytes == 0 {
        return None;
    }
    profile.expert_block_bytes.as_ref()?;
    (1..=layer_count).find_map(|blocks| {
        let moved = host_expert_bytes_for_blocks(metadata, blocks)?;
        if moved > host_budget {
            return None;
        }
        let resident = metadata.model_bytes.saturating_sub(moved);
        minimum_context_fits(resident, selected_budget, kv_bytes_per_token).then(|| {
            (
                blocks,
                planned_context_length(&metadata.compact_meta, resident, selected_budget, quant),
            )
        })
    })
}

fn find_partial_gpu_layers_fit(
    metadata: &TuneGgufMetadata,
    layer_count: u32,
    selected_budget: u64,
    kv_bytes_per_token: u64,
    quant: skippy_model_artifact::gguf::GgufKvCacheQuant,
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

/// Exact routed-expert bytes of the first `blocks` model blocks, from the
/// tensor scan's per-block attribution. `None` without attribution: an
/// even-spread guess is unsafe for models with dense leading blocks.
fn host_expert_bytes_for_blocks(metadata: &TuneGgufMetadata, blocks: u32) -> Option<u64> {
    let TuneTensorProfile::Exact(profile) = &metadata.tensor_profile else {
        return None;
    };
    let per_block = profile.expert_block_bytes.as_ref()?;
    Some(per_block.iter().take(blocks as usize).copied().sum())
}

/// Placement settings the written configuration keeps, which the fit must
/// respect instead of recommending around them.
#[derive(Clone, Copy, Default)]
pub(crate) struct PlacementConstraints {
    /// A preserved `cpu_moe = false` keeps every routed expert on the
    /// accelerator, so host-expert moves are not available to the plan.
    pub(crate) cpu_moe_false: bool,
    /// A preserved `cpu_moe = true` hosts every block's routed experts on
    /// the host at load time regardless of `n_cpu_moe`, so the plan must
    /// validate the all-block host allocation instead of a partial count.
    pub(crate) cpu_moe_true: bool,
    /// A preserved `n_cpu_moe` block count the written config will keep;
    /// `Some(0)` pins every expert to the accelerator like `cpu_moe = false`.
    pub(crate) kept_n_cpu_moe: Option<u32>,
}

/// Resolve the placement the written config will carry, mirroring
/// `push_cpu_moe_status` (a `cpu_moe` source is preserved in every mode)
/// and `push_n_cpu_moe_status` (`n_cpu_moe` is preserved unless replaced).
fn placement_constraints(
    apply_mode: TuneApplyMode,
    model_entry: Option<&ModelConfigEntry>,
    defaults: Option<&mesh_llm_config::ModelConfigDefaults>,
) -> PlacementConstraints {
    let model_hardware = model_entry.and_then(|entry| entry.hardware.as_ref());
    let cpu_moe = explicit_cpu_moe(model_hardware)
        .or_else(|| explicit_cpu_moe(defaults.and_then(|value| value.hardware.as_ref())));
    let n_cpu_moe = (apply_mode != TuneApplyMode::ReplaceExisting)
        .then(|| {
            model_hardware
                .and_then(|hardware| hardware.n_cpu_moe)
                .or_else(|| defaults?.hardware.as_ref()?.n_cpu_moe)
        })
        .flatten();
    PlacementConstraints {
        cpu_moe_false: cpu_moe == Some(false),
        cpu_moe_true: cpu_moe == Some(true),
        kept_n_cpu_moe: n_cpu_moe,
    }
}

/// Resolve an explicit `cpu_moe` boolean; `auto` leaves expert placement to
/// `n_cpu_moe`.
fn explicit_cpu_moe(hardware: Option<&mesh_llm_config::HardwareConfig>) -> Option<bool> {
    hardware
        .and_then(|hardware| hardware.cpu_moe.as_ref())
        .and_then(|value| match value {
            mesh_llm_config::BoolOrAuto::Bool(flag) => Some(*flag),
            mesh_llm_config::BoolOrAuto::String(_) => None,
        })
}

/// Check the fit with the operator's preserved host-expert block count:
/// those blocks' routed experts must fit in host memory and the remaining
/// resident bytes on the accelerator.
fn fit_with_kept_host_experts(
    metadata: &TuneGgufMetadata,
    blocks: u32,
    selected_budget: u64,
    host_budget: u64,
    kv_bytes_per_token: u64,
    quant: skippy_model_artifact::gguf::GgufKvCacheQuant,
) -> Option<u32> {
    let moved = host_expert_bytes_for_blocks(metadata, blocks)?;
    if moved > host_budget {
        return None;
    }
    let resident = metadata.model_bytes.saturating_sub(moved);
    minimum_context_fits(resident, selected_budget, kv_bytes_per_token)
        .then(|| planned_context_length(&metadata.compact_meta, resident, selected_budget, quant))
}

fn plan_fit(
    metadata: &TuneGgufMetadata,
    hardware: &TuneHardwareEvaluation,
    survey: &HardwareSurvey,
    placement: PlacementConstraints,
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

    let host_budget = survey
        .system_ram_bytes
        .map(|ram| derive_fit_target_mib(ram).saturating_mul(1024 * 1024))
        .unwrap_or(cpu_budget);
    let mut host_expert_blocks = None;
    let has_routed_experts = match &metadata.tensor_profile {
        TuneTensorProfile::Exact(profile) => {
            profile.expert_count > 0 && profile.expert_tensor_bytes > 0
        }
        _ => false,
    };
    // A preserved cpu_moe = true hosts every routed expert on the host at
    // load time regardless of n_cpu_moe, so the plan must validate the
    // all-block host allocation instead of a partial count; for dense
    // models the flag is inert.
    let all_blocks_host = placement.cpu_moe_true && has_routed_experts;
    let kept_n_cpu_moe = if all_blocks_host {
        Some(layer_count)
    } else {
        placement.kept_n_cpu_moe
    };
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
            if !all_blocks_host
                && minimum_context_fits(metadata.model_bytes, selected_budget, kv_bytes_per_token)
            {
                Some((
                    TuneGpuLayersValue::All,
                    planned_context_length(
                        &metadata.compact_meta,
                        metadata.model_bytes,
                        selected_budget,
                        quant,
                    ),
                ))
            } else if !placement.cpu_moe_false && kept_n_cpu_moe != Some(0) {
                if let Some(kept) = kept_n_cpu_moe {
                    // The operator's n_cpu_moe is preserved on write in every
                    // mode but ReplaceExisting, so the kept count is the only
                    // host-expert plan that can actually be persisted; when it
                    // does not fit, fall back to the CPU/partial-layer paths
                    // rather than recommending a different block count.
                    match fit_with_kept_host_experts(
                        metadata,
                        kept,
                        selected_budget,
                        host_budget,
                        kv_bytes_per_token,
                        quant,
                    ) {
                        Some(ctx_size) => {
                            host_expert_blocks = Some(kept);
                            Some((TuneGpuLayersValue::All, ctx_size))
                        }
                        // A pinned cpu_moe = true fixes the host placement:
                        // partial GPU layers cannot relieve it, so an
                        // unfittable all-block allocation is an error rather
                        // than a fallback plan.
                        None if !cpu_can_fit || all_blocks_host => None,
                        None => find_partial_gpu_layers_fit(
                            metadata,
                            layer_count,
                            selected_budget,
                            kv_bytes_per_token,
                            quant,
                        ),
                    }
                } else if let Some((blocks, ctx_size)) = find_host_expert_fit(
                    metadata,
                    layer_count,
                    selected_budget,
                    host_budget,
                    kv_bytes_per_token,
                    quant,
                ) {
                    host_expert_blocks = Some(blocks);
                    Some((TuneGpuLayersValue::All, ctx_size))
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
            host_expert_blocks,
            diagnostic: None,
        },
        None => PlannedFit {
            ctx_size: 0,
            batch: 0,
            ubatch: 0,
            gpu_layers: TuneGpuLayersValue::Count(0),
            host_expert_blocks: None,
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
