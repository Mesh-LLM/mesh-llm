//! Refresh security permissions after persisted owner-control configuration.

use crate::plugin::PluginManager;
use crate::runtime::config_state::ConfigState;
use std::sync::Arc;
use tokio::sync::Mutex;

impl super::Node {
    pub(crate) async fn refresh_plugin_exchange_grants(&self) -> anyhow::Result<()> {
        let Some(manager) = self.plugin_manager().await else {
            return Ok(());
        };
        refresh_exchange_grants(Arc::clone(&self.config_state), manager).await
    }
}

async fn refresh_exchange_grants(
    config_state: Arc<Mutex<ConfigState>>,
    manager: PluginManager,
) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Handle::current();
    // The synchronous apply lock must never block a Tokio worker or cross an
    // async suspension. Its blocking owner runs publication on this runtime.
    tokio::task::spawn_blocking(move || {
        publish_exchange_grants(config_state, manager, runtime, || {});
    })
    .await
    .map_err(|error| anyhow::anyhow!("plugin grant refresh task panicked: {error}"))
}

fn publish_exchange_grants(
    config_state: Arc<Mutex<ConfigState>>,
    manager: PluginManager,
    runtime: tokio::runtime::Handle,
    after_snapshot: impl FnOnce(),
) {
    let apply_lock = config_state.blocking_lock().apply_serialization_lock();
    let _apply_guard = apply_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = config_state.blocking_lock().config().clone();
    // Lock order matches owner/config mutation: apply serialization, then a
    // short config snapshot. No config guard survives manager publication.
    after_snapshot();
    runtime.block_on(manager.apply_exchange_grants(&config));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{MeshConfig, PluginHostMode, ResolvedPlugins};

    #[tokio::test]
    async fn concurrent_refresh_cannot_restore_grants_after_revocation() {
        let directory = std::env::temp_dir().join(format!(
            "mesh-llm-owner-grant-refresh-{}",
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let config_path = directory.join("config.toml");
        let config = MeshConfig {
            plugins: vec![
                serde_json::from_value(serde_json::json!({
                    "name":"observer",
                    "openai_exchange_grant": {
                        "request_body":true,
                        "endpoints":["chat_completions"],
                        "phases":["request_received"],
                        "deadline_ms":100,
                        "max_body_bytes":1024,
                        "max_queue_bytes":1024,
                        "max_in_flight":1
                    }
                }))
                .unwrap(),
            ],
            ..Default::default()
        };
        std::fs::write(
            &config_path,
            crate::plugin::config_to_toml(&config).unwrap(),
        )
        .unwrap();
        let state = Arc::new(Mutex::new(ConfigState::load(&config_path).unwrap()));
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let manager = PluginManager::start(
            &ResolvedPlugins {
                externals: Vec::new(),
                inactive: Vec::new(),
            },
            PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            tx,
        )
        .await
        .unwrap();
        manager.apply_exchange_grants(&config).await;
        let (snapshot_tx, snapshot_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let old_state = Arc::clone(&state);
        let old_manager = manager.clone();
        let runtime = tokio::runtime::Handle::current();
        let old_refresh = tokio::task::spawn_blocking(move || {
            let unlocked_config = Arc::clone(&old_state);
            publish_exchange_grants(old_state, old_manager, runtime, || {
                assert!(unlocked_config.try_lock().is_ok());
                snapshot_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
        });
        snapshot_rx.await.unwrap();
        let apply_lock = state.lock().await.apply_serialization_lock();
        assert!(
            apply_lock.try_lock().is_err(),
            "snapshot/publication must exclude config commit"
        );
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let new_state = Arc::clone(&state);
        let new_manager = manager.clone();
        let revocation = tokio::spawn(async move {
            let (result, revision, _) = tokio::task::spawn_blocking(move || {
                started_tx.send(()).unwrap();
                super::super::apply_owner_control_config_with_persistence(
                    new_state,
                    MeshConfig::default(),
                    0,
                    |pending| pending.persist(),
                )
            })
            .await
            .unwrap();
            assert!(matches!(
                result,
                crate::runtime::config_state::ApplyResult::Applied { .. }
            ));
            assert_eq!(revision, 1);
            refresh_exchange_grants(Arc::clone(&state), new_manager)
                .await
                .unwrap();
        });
        started_rx.await.unwrap();
        release_tx.send(()).unwrap();
        old_refresh.await.unwrap();
        revocation.await.unwrap();
        assert!(manager.effective_exchange_grant("observer").is_none());
        manager.shutdown().await;
        std::fs::remove_dir_all(directory).ok();
    }
}
