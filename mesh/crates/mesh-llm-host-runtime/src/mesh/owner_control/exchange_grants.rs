//! Refresh security permissions after persisted owner-control configuration.

use crate::plugin::{MeshConfig, PluginManager};
use crate::runtime::config_state::{
    ApplyResult, ConfigPersistence, ConfigState, PendingConfigApply,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Commit and publish security permissions in the same cancellation-independent
/// blocking owner. Dropping the command's JoinHandle cannot skip publication.
pub(super) fn apply_with_exchange_grants(
    config_state: Arc<Mutex<ConfigState>>,
    config: MeshConfig,
    expected_revision: u64,
    persist: impl FnOnce(&PendingConfigApply) -> ConfigPersistence,
    manager_provider: impl FnOnce() -> Option<PluginManager>,
    runtime: tokio::runtime::Handle,
) -> (ApplyResult, u64, [u8; 32]) {
    let result = super::apply_owner_control_config_with_persistence(
        Arc::clone(&config_state),
        config,
        expected_revision,
        persist,
    );
    if matches!(
        &result.0,
        ApplyResult::Applied { .. }
            | ApplyResult::AppliedWithRestartRequired { .. }
            | ApplyResult::PersistedWithRevisionTrackingError { .. }
    ) {
        with_serialized_config(config_state, |config| {
            if let Some(manager) = manager_provider() {
                runtime.block_on(manager.apply_exchange_grants(&config));
            }
        });
    }
    result
}

impl super::Node {
    /// Production startup publishes the latest persisted grants and installs
    /// the manager atomically relative to owner applies and grant refreshes.
    pub(crate) async fn install_plugin_manager_with_exchange_grants(
        &self,
        manager: PluginManager,
    ) -> anyhow::Result<()> {
        let node = self.clone();
        let config_state = Arc::clone(&self.config_state);
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            with_serialized_config(config_state, |config| {
                runtime.block_on(async {
                    manager.apply_exchange_grants(&config).await;
                    node.set_plugin_manager(manager).await;
                });
            });
        })
        .await
        .map_err(|error| anyhow::anyhow!("plugin manager installation task panicked: {error}"))
    }
}

#[cfg(test)]
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

#[cfg(test)]
fn publish_exchange_grants(
    config_state: Arc<Mutex<ConfigState>>,
    manager: PluginManager,
    runtime: tokio::runtime::Handle,
    after_snapshot: impl FnOnce(),
) {
    with_serialized_config(config_state, |config| {
        after_snapshot();
        runtime.block_on(manager.apply_exchange_grants(&config));
    });
}

fn with_serialized_config(config_state: Arc<Mutex<ConfigState>>, publish: impl FnOnce(MeshConfig)) {
    let apply_lock = config_state.blocking_lock().apply_serialization_lock();
    let _apply_guard = apply_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = config_state.blocking_lock().config().clone();
    // Lock order matches owner/config mutation: apply serialization, then a
    // short config snapshot. No config guard survives manager publication.
    publish(config);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginHostMode, ResolvedPlugins};

    async fn granted_fixture() -> (std::path::PathBuf, Arc<Mutex<ConfigState>>, PluginManager) {
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
        (directory, state, manager)
    }

    #[tokio::test]
    async fn concurrent_refresh_cannot_restore_grants_after_revocation() {
        let (directory, state, manager) = granted_fixture().await;
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

    #[tokio::test]
    async fn cancelled_owner_command_still_reconciles_persisted_revocation() {
        let (directory, state, manager) = granted_fixture().await;
        let revision = manager.exchange_grant_revision("observer");
        let (persisted_tx, persisted_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (published_tx, published_rx) = tokio::sync::oneshot::channel();
        let apply_state = Arc::clone(&state);
        let apply_manager = manager.clone();
        let runtime = tokio::runtime::Handle::current();
        let waiter = tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                let result = apply_with_exchange_grants(
                    apply_state,
                    MeshConfig::default(),
                    0,
                    |pending| {
                        let persistence = pending.persist();
                        assert!(matches!(persistence, ConfigPersistence::Persisted));
                        persisted_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        persistence
                    },
                    || Some(apply_manager),
                    runtime,
                );
                published_tx.send(result).unwrap();
            })
            .await
            .unwrap();
        });
        persisted_rx.await.unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        release_tx.send(()).unwrap();
        let (result, applied_revision, _) = published_rx.await.unwrap();
        assert!(matches!(result, ApplyResult::Applied { .. }));
        assert_eq!(applied_revision, 1);
        assert_eq!(state.lock().await.revision(), 1);
        assert!(manager.effective_exchange_grant("observer").is_none());
        assert!(
            revision.has_changed().unwrap(),
            "in-flight observers must be revoked"
        );
        let persisted = ConfigState::load(&directory.join("config.toml")).unwrap();
        assert!(persisted.config().plugins.is_empty());
        manager.shutdown().await;
        std::fs::remove_dir_all(directory).ok();
    }

    async fn startup_fixture() -> (std::path::PathBuf, super::super::Node, PluginManager) {
        let (directory, state, manager) = granted_fixture().await;
        let mut node = super::super::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        node.config_state = state;
        assert!(node.plugin_manager().await.is_none());
        (directory, node, manager)
    }

    #[tokio::test]
    async fn startup_install_uses_revocation_persisted_before_manager_exists() {
        let (directory, node, manager) = startup_fixture().await;
        let state = Arc::clone(&node.config_state);
        let provider = node.clone();
        let runtime = tokio::runtime::Handle::current();
        let (result, _, _) = tokio::task::spawn_blocking(move || {
            apply_with_exchange_grants(
                state,
                MeshConfig::default(),
                0,
                PendingConfigApply::persist,
                || runtime.block_on(provider.plugin_manager()),
                runtime.clone(),
            )
        })
        .await
        .unwrap();
        assert!(matches!(result, ApplyResult::Applied { .. }));
        // This detached startup manager still has the original grant snapshot.
        assert!(manager.effective_exchange_grant("observer").is_some());
        node.install_plugin_manager_with_exchange_grants(manager.clone())
            .await
            .unwrap();
        assert!(manager.effective_exchange_grant("observer").is_none());
        assert!(
            node.plugin_manager()
                .await
                .unwrap()
                .effective_exchange_grant("observer")
                .is_none()
        );
        manager.shutdown().await;
        std::fs::remove_dir_all(directory).ok();
    }

    #[tokio::test]
    async fn owner_command_started_without_manager_revokes_manager_installed_before_commit() {
        let (directory, node, manager) = startup_fixture().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let command_node = node.clone();
        let runtime = tokio::runtime::Handle::current();
        let command = tokio::task::spawn_blocking(move || {
            assert!(runtime.block_on(command_node.plugin_manager()).is_none());
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            apply_with_exchange_grants(
                Arc::clone(&command_node.config_state),
                MeshConfig::default(),
                0,
                PendingConfigApply::persist,
                || runtime.block_on(command_node.plugin_manager()),
                runtime.clone(),
            )
        });
        started_rx.await.unwrap();
        node.install_plugin_manager_with_exchange_grants(manager.clone())
            .await
            .unwrap();
        assert!(manager.effective_exchange_grant("observer").is_some());
        release_tx.send(()).unwrap();
        let (result, revision, _) = command.await.unwrap();
        assert!(matches!(result, ApplyResult::Applied { .. }));
        assert_eq!(revision, 1);
        assert!(manager.effective_exchange_grant("observer").is_none());
        manager.shutdown().await;
        std::fs::remove_dir_all(directory).ok();
    }
}
