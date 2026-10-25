//! Bounded operator-facing lifecycle permission and health status.
use super::{PluginManager, PluginSummary};

impl PluginManager {
    pub(super) async fn project_exchange_status(&self, summaries: &mut [PluginSummary]) {
        for summary in summaries {
            let Some(manifest) = &mut summary.manifest else {
                continue;
            };
            if manifest.openai_exchange_body_access_requested.is_some() {
                if let Some(plugin) = self.inner.plugins.get(&summary.name) {
                    let live = plugin.manifest_snapshot().await;
                    self.refresh_exchange_permissions_status(
                        &summary.name,
                        live.as_ref()
                            .and_then(|manifest| manifest.openai_exchange_hook.as_deref()),
                    );
                }
                manifest.openai_exchange_status = Some(
                    if self.effective_exchange_grant(&summary.name).is_none() {
                        "not_granted"
                    } else {
                        self.exchange_health_status(&summary.name)
                    }
                    .to_owned(),
                );
            }
        }
    }
}
