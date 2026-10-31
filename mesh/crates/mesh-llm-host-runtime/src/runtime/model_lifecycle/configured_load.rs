//! Cold configured loads use the same source and device planning as eager startup.
use super::*;
use crate::runtime::startup_models;
use std::path::Path;

pub(super) struct ConfiguredLoadPlan {
    pub(super) plan: startup_models::StartupModelPlan,
    pub(super) capacity_bytes: u64,
}

/// Resolve projector and native sources before capacity admission, preserving CLI device precedence.
pub(super) async fn prepare(
    ctx: &RunAutoRuntimeLoopContext<'_>,
    model: &plugin::ModelConfigEntry,
) -> Result<ConfiguredLoadPlan> {
    let spec = crate::runtime::native_model_lifecycle::sources::startup_spec(
        ctx.config,
        ctx.options,
        model,
    )?;
    let mut plans =
        startup_models::resolve_startup_models(std::slice::from_ref(&spec), false).await?;
    startup_models::preflight_pinned_startup_models(
        ctx.config,
        std::slice::from_ref(&spec),
        &mut plans,
        ctx.options.llama_flavor,
        None,
    )?;
    let mut plan = plans.remove(0);
    // Logical IDs must not collapse onto the same filename-derived backend name.
    plan.declared_ref =
        startup_models::configured_server_alias(model, ctx.config.defaults.as_ref())
            .unwrap_or_else(|| model.model.clone());
    let capacity_bytes = plan
        .pinned_gpu
        .as_ref()
        .map(|gpu| gpu.allocatable_vram_bytes())
        .unwrap_or_else(|| ctx.node.local_runtime_capacity_bytes());
    Ok(ConfiguredLoadPlan {
        plan,
        capacity_bytes,
    })
}

/// Include explicitly resolved projector storage in the same device admission pool.
pub(super) fn model_and_projector_bytes(weights: u64, projector: Option<&Path>) -> Result<u64> {
    let projector = projector
        .map(std::fs::metadata)
        .transpose()?
        .map(|file| file.len())
        .unwrap_or(0);
    Ok(weights.saturating_add(projector))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configured_projector_is_included_in_device_admission() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [0u8; 123]).unwrap();
        assert_eq!(
            model_and_projector_bytes(1000, Some(file.path())).unwrap(),
            1123
        );
        assert_eq!(model_and_projector_bytes(1000, None).unwrap(), 1000);
        assert!(
            model_and_projector_bytes(1000, Some(Path::new("/missing/native-projector"))).is_err()
        );
    }
}
