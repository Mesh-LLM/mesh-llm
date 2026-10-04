//! Refresh security permissions after persisted owner-control configuration.

impl super::Node {
    pub(crate) async fn refresh_plugin_exchange_grants(&self) {
        let Some(manager) = self.plugin_manager().await else {
            return;
        };
        let config = self.config_state.lock().await.config().clone();
        manager.apply_exchange_grants(&config).await;
    }
}
