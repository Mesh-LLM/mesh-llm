//! This node's private node list: `[private] nodes` in its config file.
//!
//! `mesh-llm private add` and `remove` edit that list, and so can a hand edit
//! or a config apply. A running node notices the file change and reads the
//! list again, so no restart is needed. When the file cannot be read or does
//! not validate, the list is empty: private requests stay on this node until
//! the file is fixed.

use super::plugin_streams::endpoint_id_from_hex;
use iroh::EndpointId;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

/// How often a running node checks its config file for a changed list.
const RELOAD_INTERVAL: Duration = Duration::from_secs(2);

/// The nodes besides this one that may serve its private requests.
#[derive(Clone, Debug, Default)]
pub(crate) struct PrivateNodes(Arc<RwLock<Arc<HashSet<EndpointId>>>>);

impl PrivateNodes {
    pub(crate) fn of(ids: impl IntoIterator<Item = EndpointId>) -> Self {
        Self(Arc::new(RwLock::new(Arc::new(ids.into_iter().collect()))))
    }

    /// The list in `config`, kept current from the config file it came from
    /// when `reload` (a node with its own config file, not a scoped
    /// shared-serving membership).
    pub(crate) fn kept_current(
        config: &mesh_llm_config::MeshConfig,
        config_path: Option<&Path>,
        reload: bool,
    ) -> Self {
        let nodes = Self::of(private_nodes_in(config));
        if reload && let Ok(path) = crate::plugin::config_path(config_path) {
            spawn_private_nodes_reload(nodes.clone(), path);
        }
        nodes
    }

    pub(crate) fn contains(&self, id: &EndpointId) -> bool {
        self.current().contains(id)
    }

    pub(crate) fn current(&self) -> Arc<HashSet<EndpointId>> {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Replace the list; true when it changed.
    fn replace(&self, ids: HashSet<EndpointId>) -> bool {
        let mut current = self
            .0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if **current == ids {
            return false;
        }
        *current = Arc::new(ids);
        true
    }
}

/// The node ids in `config`'s `[private] nodes`. Validation rejects any other
/// entry before a config is used, so one that is not a node id is skipped.
pub(crate) fn private_nodes_in(config: &mesh_llm_config::MeshConfig) -> HashSet<EndpointId> {
    config
        .private
        .nodes
        .iter()
        .filter_map(|id| endpoint_id_from_hex(&id.trim().to_ascii_lowercase()))
        .collect()
}

/// The list in the config file at `path`; empty when it cannot be loaded.
fn load_private_nodes(path: &Path) -> HashSet<EndpointId> {
    if !path.exists() {
        return HashSet::new();
    }
    match mesh_llm_config::load_config(Some(path)) {
        Ok(config) => private_nodes_in(&config),
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "config file does not load; private requests stay on this node until it is fixed"
            );
            HashSet::new()
        }
    }
}

/// What identifies one version of the file: its modification time and size.
fn file_version(path: &Path) -> Option<(SystemTime, u64)> {
    let metadata = std::fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

/// Read the list again whenever the config file at `path` changes.
pub(crate) fn spawn_private_nodes_reload(nodes: PrivateNodes, path: PathBuf) {
    tokio::spawn(async move {
        // Nothing seen yet, so the first check reads the file again: an edit
        // made between the node's first read and this task starting counts.
        let mut seen = None;
        loop {
            tokio::time::sleep(RELOAD_INTERVAL).await;
            let version = file_version(&path);
            if version == seen {
                continue;
            }
            seen = version;
            let reload_path = path.clone();
            let Ok(ids) =
                tokio::task::spawn_blocking(move || load_private_nodes(&reload_path)).await
            else {
                continue;
            };
            let count = ids.len();
            if nodes.replace(ids) {
                tracing::info!(count, "private node list reloaded from the config file");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    fn write_config(path: &Path, ids: &[EndpointId]) {
        let nodes: Vec<String> = ids.iter().map(|id| format!("\"{id}\"")).collect();
        std::fs::write(path, format!("[private]\nnodes = [{}]\n", nodes.join(", "))).unwrap();
    }

    #[test]
    fn the_list_comes_from_the_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        assert!(load_private_nodes(&path).is_empty());
        write_config(&path, &[id(1), id(2)]);
        assert_eq!(load_private_nodes(&path), [id(1), id(2)].into());
    }

    #[test]
    fn a_config_file_that_does_not_load_lists_no_node() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[private]\nnodes = [\"studio\"]\n").unwrap();
        assert!(load_private_nodes(&path).is_empty());
    }

    #[tokio::test]
    async fn a_running_node_picks_up_an_edit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[id(1)]);
        let nodes = PrivateNodes::of(load_private_nodes(&path));
        spawn_private_nodes_reload(nodes.clone(), path.clone());

        write_config(&path, &[id(1), id(2), id(3)]);
        for _ in 0..300 {
            if nodes.current().len() == 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert_eq!(*nodes.current(), [id(1), id(2), id(3)].into());

        std::fs::remove_file(&path).unwrap();
        for _ in 0..300 {
            if nodes.current().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(nodes.current().is_empty());
    }
}
