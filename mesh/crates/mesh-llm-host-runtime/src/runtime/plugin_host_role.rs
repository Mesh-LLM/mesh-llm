use crate::{mesh, plugin};
use std::time::Duration;

const WATCH_INTERVAL: Duration = Duration::from_secs(5);

/// Promotes this node to `NodeRole::Host` whenever a loaded plugin is
/// advertising at least one inference model, and releases the plugin's host
/// claim when that stops being true.
pub(super) fn spawn(node: mesh::Node, plugin_manager: plugin::PluginManager, http_port: u16) {
    tokio::spawn(async move {
        let mut plugin_claimed = false;
        loop {
            tokio::time::sleep(WATCH_INTERVAL).await;
            let has_plugin_models = match plugin_manager.inference_models().await {
                Ok(models) => !models.is_empty(),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "plugin host-role watcher: failed to read plugin inference models, skipping this tick"
                    );
                    continue;
                }
            };
            if has_plugin_models && !plugin_claimed {
                node.claim_host_role(mesh::HostRoleClaim::PluginInference, http_port)
                    .await;
                plugin_claimed = true;
            } else if !has_plugin_models && plugin_claimed {
                node.release_host_role(mesh::HostRoleClaim::PluginInference)
                    .await;
                plugin_claimed = false;
            }
        }
    });
}
