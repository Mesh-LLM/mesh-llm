//! Caller-owned transport replay policy for plugin operations.
use super::{PluginManager, ToolCallResult};
use anyhow::{Context, Result, bail};

/// Whether an operation can be repeated after an uncertain transport failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransportReplay {
    Allow,
    #[cfg(any(feature = "payments", test))]
    Never,
}

impl PluginManager {
    /// Invoke an operation with an explicit deadline. `None` waits until the
    /// plugin answers or its connection drops; the caller owns cancellation.
    pub async fn call_tool_with_timeout(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<ToolCallResult> {
        self.invoke_operation_with_replay(
            plugin_name,
            tool_name,
            arguments_json,
            timeout,
            TransportReplay::Allow,
        )
        .await
    }

    pub(crate) async fn invoke_operation_with_replay(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
        timeout: Option<std::time::Duration>,
        replay: TransportReplay,
    ) -> Result<ToolCallResult> {
        if self.is_test_bridge_enabled(plugin_name) {
            return self.call_tool(plugin_name, tool_name, arguments_json).await;
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin
            .call_tool_with_replay(tool_name, arguments_json, timeout, replay)
            .await
    }
}
