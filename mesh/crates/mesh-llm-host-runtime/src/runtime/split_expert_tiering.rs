//! Routed-expert tiering inputs for Mesh split planning: the operator's
//! offload policy, the package's per-layer expert bytes, and the expert
//! bytes each planned stage keeps in host memory.

use super::split_planning::{ExpertOffloadPolicy, RuntimeSliceStagePlan};
use crate::inference::skippy;
use skippy_coordinator::topology::host_expert_bytes_for_stage;

/// Whether the planner may move routed experts into host memory for this
/// model. An explicit `hardware.cpu_moe = false` keeps every expert on the
/// accelerator; `true`, `"auto"` and unset let the planner spill when the
/// node set cannot otherwise hold the model.
pub(crate) fn configured_expert_offload(
    mesh_config: &mesh_llm_config::MeshConfig,
    config_model_id: Option<&str>,
) -> ExpertOffloadPolicy {
    let model_hardware = config_model_id
        .and_then(|selector| {
            mesh_config
                .models
                .iter()
                .find(|model| model.model == selector)
        })
        .and_then(|model| model.hardware.as_ref());
    let default_hardware = mesh_config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref());
    let cpu_moe = model_hardware
        .and_then(|hardware| hardware.cpu_moe.as_ref())
        .or_else(|| default_hardware.and_then(|hardware| hardware.cpu_moe.as_ref()));
    match cpu_moe {
        Some(mesh_llm_config::BoolOrAuto::Bool(false)) => ExpertOffloadPolicy::Disabled,
        _ => ExpertOffloadPolicy::Auto,
    }
}

/// Routed-expert bytes per layer, only when the package accounts every layer
/// and the planner also has exact per-layer weights to subtract them from.
pub(super) fn package_layer_expert_weight_bytes(
    package: &skippy::SkippyPackageIdentity,
) -> Vec<u64> {
    if package.layer_expert_weight_bytes.len() == package.layer_count as usize
        && package.layer_weight_bytes.len() == package.layer_count as usize
    {
        return package.layer_expert_weight_bytes.clone();
    }
    Vec::new()
}

/// Routed-expert bytes a stage keeps in host memory under its planned
/// placement; zero when every expert stays on the accelerator.
pub(super) fn stage_host_expert_bytes(
    stage: &RuntimeSliceStagePlan,
    package: &skippy::SkippyPackageIdentity,
) -> u64 {
    let Some(host_layer_end) = stage.host_expert_layer_end else {
        return 0;
    };
    host_expert_bytes_for_stage(
        &package_layer_expert_weight_bytes(package),
        package.layer_count,
        stage.layer_start,
        host_layer_end,
    )
}
