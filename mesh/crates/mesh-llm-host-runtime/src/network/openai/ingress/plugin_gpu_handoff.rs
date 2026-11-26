//! Exclusive GPU handoff acquisition before forwarding to an `exclusive_gpu` plugin.
use super::*;
use crate::runtime::native_model_lifecycle::ExclusiveGpuHandoffLease;

/// Acquire an exclusive GPU handoff lease when `plugin_name` opts into `exclusive_gpu`.
/// Returns `Ok(None)` for standard plugins, or the failure reason string if handoff fails.
pub(super) async fn acquire_for_plugin(
    node: &mesh::Node,
    plugin_name: &str,
    model_name: &str,
) -> Result<Option<ExclusiveGpuHandoffLease>, &'static str> {
    if !node.plugin_exclusive_gpu_enabled(plugin_name).await {
        return Ok(None);
    }
    let Some(lifecycle) = node.native_model_lifecycle.get() else {
        return Ok(None);
    };
    lifecycle
        .acquire_exclusive_gpu_handoff()
        .await
        .map(Some)
        .map_err(|error| {
            tracing::warn!(
                plugin = %plugin_name,
                model = %model_name,
                %error,
                "exclusive_gpu handoff failed before forwarding to plugin endpoint"
            );
            "plugin_gpu_handoff_failed"
        })
}
