use super::Node;
use crate::runtime::config_state::{ApplyResult, ConfigApplyMode};
use std::collections::BTreeMap;
use std::sync::Arc;

impl Node {
    pub(crate) async fn set_plugin_web_ui_enabled(
        &self,
        plugin_name: &str,
        enabled: bool,
    ) -> anyhow::Result<()> {
        let config_state = Arc::clone(&self.config_state);
        let revision_tx = Arc::clone(&self.config_revision_tx);
        let plugin_name = plugin_name.to_string();
        let apply_result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let mut state = config_state.blocking_lock();
            let expected_revision = state.revision();
            let mut config = state.config().clone();
            match config
                .plugins
                .iter_mut()
                .find(|plugin| plugin.name == plugin_name)
            {
                Some(plugin) => plugin.web_ui_enabled = Some(enabled),
                None => config.plugins.push(crate::plugin::PluginConfigEntry {
                    name: plugin_name,
                    enabled: None,
                    web_ui_enabled: Some(enabled),
                    command: None,
                    args: Vec::new(),
                    url: None,
                    settings: BTreeMap::new(),
                    startup: crate::plugin::PluginStartupConfig::default(),
                }),
            }
            let result = state.apply(config, expected_revision);
            Ok((result, state.revision()))
        })
        .await
        .map_err(|error| anyhow::anyhow!("plugin web UI config task panicked: {error}"))??;

        match apply_result {
            (ApplyResult::Applied { apply_mode, .. }, revision) => {
                if apply_mode == ConfigApplyMode::Staged {
                    let _ = revision_tx.send(revision);
                }
                Ok(())
            }
            (
                ApplyResult::PersistedWithRevisionTrackingError {
                    revision, error, ..
                },
                _,
            ) => {
                let _ = revision_tx.send(revision);
                anyhow::bail!(error)
            }
            (ApplyResult::RevisionConflict { current_revision }, _) => {
                anyhow::bail!("config revision conflict at revision {current_revision}")
            }
            (ApplyResult::ValidationError { error, .. }, _)
            | (ApplyResult::PersistError(error), _) => anyhow::bail!(error),
        }
    }

    pub(crate) async fn plugin_settings(&self, plugin_name: &str) -> BTreeMap<String, toml::Value> {
        let state = self.config_state.lock().await;
        state
            .config()
            .plugins
            .iter()
            .find(|plugin| plugin.name == plugin_name)
            .map(|plugin| plugin.settings.clone())
            .unwrap_or_default()
    }

    pub(crate) async fn patch_plugin_settings(
        &self,
        plugin_name: &str,
        settings: BTreeMap<String, toml::Value>,
        unset: Vec<String>,
    ) -> anyhow::Result<BTreeMap<String, toml::Value>> {
        let config_state = Arc::clone(&self.config_state);
        let revision_tx = Arc::clone(&self.config_revision_tx);
        let plugin_name = plugin_name.to_string();
        let apply_result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let mut state = config_state.blocking_lock();
            let expected_revision = state.revision();
            let mut config = state.config().clone();
            let plugin = match config
                .plugins
                .iter_mut()
                .find(|plugin| plugin.name == plugin_name)
            {
                Some(plugin) => plugin,
                None => {
                    config.plugins.push(crate::plugin::PluginConfigEntry {
                        name: plugin_name.clone(),
                        enabled: None,
                        web_ui_enabled: None,
                        command: None,
                        args: Vec::new(),
                        url: None,
                        settings: BTreeMap::new(),
                        startup: crate::plugin::PluginStartupConfig::default(),
                    });
                    config.plugins.last_mut().expect("new plugin entry")
                }
            };
            for key in unset {
                plugin.settings.remove(&key);
            }
            plugin.settings.extend(settings);
            let updated_settings = plugin.settings.clone();
            let result = state.apply(config, expected_revision);
            Ok((result, state.revision(), updated_settings))
        })
        .await
        .map_err(|error| anyhow::anyhow!("plugin settings config task panicked: {error}"))??;

        match apply_result {
            (ApplyResult::Applied { apply_mode, .. }, revision, settings) => {
                if apply_mode == ConfigApplyMode::Staged {
                    let _ = revision_tx.send(revision);
                }
                Ok(settings)
            }
            (
                ApplyResult::PersistedWithRevisionTrackingError {
                    revision, error, ..
                },
                _,
                _,
            ) => {
                let _ = revision_tx.send(revision);
                anyhow::bail!(error)
            }
            (ApplyResult::RevisionConflict { current_revision }, _, _) => {
                anyhow::bail!("config revision conflict at revision {current_revision}")
            }
            (ApplyResult::ValidationError { error, .. }, _, _)
            | (ApplyResult::PersistError(error), _, _) => anyhow::bail!(error),
        }
    }
}
