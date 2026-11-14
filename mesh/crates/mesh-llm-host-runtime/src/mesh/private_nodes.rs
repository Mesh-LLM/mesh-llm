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
pub(crate) struct PrivateNodes {
    ids: Arc<RwLock<Arc<HashSet<EndpointId>>>>,
    /// The config file the list is kept current from, when there is one.
    file: Option<Arc<PathBuf>>,
    /// Held across each read of the file and the swap that follows, so an
    /// older read cannot replace a newer one.
    reload: Arc<std::sync::Mutex<()>>,
}

impl PrivateNodes {
    pub(crate) fn of(ids: impl IntoIterator<Item = EndpointId>) -> Self {
        Self {
            ids: Arc::new(RwLock::new(Arc::new(ids.into_iter().collect()))),
            file: None,
            reload: Arc::default(),
        }
    }

    /// The list in `config`, kept current from the config file it came from
    /// when `reload` (a node with its own config file, not a scoped
    /// shared-serving membership).
    pub(crate) fn kept_current(
        config: &mesh_llm_config::MeshConfig,
        config_path: Option<&Path>,
        reload: bool,
    ) -> Self {
        let mut nodes = Self::of(private_nodes_in(config));
        if reload && let Ok(path) = crate::plugin::config_path(config_path) {
            nodes.file = Some(Arc::new(path.clone()));
            spawn_private_nodes_reload(nodes.clone(), path);
        }
        nodes
    }

    /// Read the list from the config file now, rather than at the next check:
    /// a config apply does this before it reports the new list as live.
    pub(crate) fn reload_now(&self) {
        if let Some(path) = &self.file {
            self.reload_from(path);
        }
    }

    /// Read the list from `path` and install it; true when it changed.
    fn reload_from(&self, path: &Path) -> bool {
        let _reload = self
            .reload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.replace(load_private_nodes(path))
    }

    pub(crate) fn contains(&self, id: &EndpointId) -> bool {
        self.current().contains(id)
    }

    pub(crate) fn current(&self) -> Arc<HashSet<EndpointId>> {
        self.ids
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Replace the list; true when it changed.
    fn replace(&self, ids: HashSet<EndpointId>) -> bool {
        let mut current = self
            .ids
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
        // Nothing seen yet, so the first check reads the file again, even a
        // missing one: an edit or removal made between the node's first read
        // and this task starting counts.
        let mut seen: Option<Option<(SystemTime, u64)>> = None;
        loop {
            tokio::time::sleep(RELOAD_INTERVAL).await;
            let version = file_version(&path);
            if seen == Some(version) {
                continue;
            }
            seen = Some(version);
            let (reloading, reload_path) = (nodes.clone(), path.clone());
            let Ok(changed) =
                tokio::task::spawn_blocking(move || reloading.reload_from(&reload_path)).await
            else {
                continue;
            };
            if changed {
                let count = nodes.current().len();
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

    #[tokio::test]
    async fn a_file_removed_before_the_first_check_empties_the_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[id(1)]);
        let nodes = PrivateNodes::of(load_private_nodes(&path));
        std::fs::remove_file(&path).unwrap();
        spawn_private_nodes_reload(nodes.clone(), path.clone());

        for _ in 0..300 {
            if nodes.current().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(nodes.current().is_empty());
    }

    #[tokio::test]
    async fn reload_now_reads_the_file_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[id(1), id(2)]);
        let config = mesh_llm_config::load_config(Some(&path)).unwrap();
        let nodes = PrivateNodes::kept_current(&config, Some(&path), true);
        assert_eq!(*nodes.current(), [id(1), id(2)].into());

        write_config(&path, &[id(2)]);
        nodes.reload_now();
        assert_eq!(*nodes.current(), [id(2)].into());

        let unwatched = PrivateNodes::kept_current(&config, Some(&path), false);
        unwatched.reload_now();
        assert_eq!(*unwatched.current(), [id(1), id(2)].into());
    }

    #[test]
    fn an_older_background_read_cannot_win_over_reload_now() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        write_config(&path, &[id(1), id(2)]);
        let config = mesh_llm_config::load_config(Some(&path)).unwrap();
        let mut nodes = PrivateNodes::kept_current(&config, Some(&path), false);
        nodes.file = Some(Arc::new(path.clone()));

        // The background check has read the old list and not yet swapped it in.
        let background = nodes.reload.lock().unwrap();
        let old_read = load_private_nodes(&path);

        // An apply removes node 1 and reloads before reporting it live.
        write_config(&path, &[id(2)]);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let applying = nodes.clone();
        let apply = std::thread::spawn(move || {
            applying.reload_now();
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "reload_now must wait for the read in progress"
        );

        nodes.replace(old_read);
        drop(background);
        apply.join().unwrap();
        assert_eq!(*nodes.current(), [id(2)].into());
    }
}
