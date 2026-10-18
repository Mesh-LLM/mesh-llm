//! Periodic plugin health checks must not own the manager between rounds.
use super::{PluginManager, health};
use std::sync::{Arc, atomic::Ordering};

impl PluginManager {
    pub(super) fn start_supervisor(&self) {
        let manager = Arc::downgrade(&self.inner);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(
                health::HEALTH_CHECK_INTERVAL_SECS,
            ));
            loop {
                ticker.tick().await;
                let Some(inner) = manager.upgrade() else {
                    break;
                };
                if inner.shutting_down.load(Ordering::SeqCst) {
                    break;
                }
                // Release this strong reference before waiting for the next tick.
                // In-process runners can retain a Node and its payment ledger.
                let manager = PluginManager { inner };
                manager.supervise_plugins().await;
            }
        });
    }

    async fn supervise_plugins(&self) {
        for (plugin_name, plugin) in &self.inner.plugins {
            if self.inner.shutting_down.load(Ordering::SeqCst) {
                break;
            }
            self.supervise_plugin(plugin_name, plugin).await;
        }
    }

    async fn supervise_plugin(&self, plugin_name: &str, plugin: &super::runtime::ExternalPlugin) {
        if let Err(err) = plugin.supervise().await {
            tracing::warn!(plugin = %plugin.name(), error = %err, "Plugin supervision round failed");
        }
        if let Err(err) = self.refresh_plugin_endpoints(plugin_name).await {
            tracing::warn!(plugin = %plugin_name, error = %err, "Endpoint supervision round failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn idle_supervisor_does_not_retain_a_dropped_manager() {
        let manager = PluginManager::for_test_summaries(Vec::new());
        let weak = Arc::downgrade(&manager.inner);
        manager.start_supervisor();
        tokio::task::yield_now().await;
        manager.shutdown().await;
        drop(manager);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("supervisor retained the manager until its next health tick");
    }
}
