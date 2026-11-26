//! This node's private node list: `[private] nodes` in its config file.
//!
//! `mesh-llm private add` and `remove` edit that list, and so can a hand edit
//! or a config apply. A running node reads the list again when the file
//! changes (see [`super::config_reload`]). When the file cannot be read or
//! does not validate, the list is empty: private requests stay on this node
//! until the file is fixed.

use super::config_reload::{ReloadedSetting, SettingSpec, WhenUnloadable};
use super::plugin_streams::endpoint_id_from_hex;
use iroh::EndpointId;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

/// The nodes besides this one that may serve its private requests.
#[derive(Clone, Debug)]
pub(crate) struct PrivateNodes(ReloadedSetting<HashSet<EndpointId>>);

impl Default for PrivateNodes {
    fn default() -> Self {
        Self::of([])
    }
}

// The list only grants, so a broken file grants nothing.
const SETTING: SettingSpec<HashSet<EndpointId>> = SettingSpec {
    name: "private.nodes",
    read: private_nodes_in,
    when_unloadable: WhenUnloadable::Empty,
};

impl PrivateNodes {
    pub(crate) fn of(ids: impl IntoIterator<Item = EndpointId>) -> Self {
        Self(ReloadedSetting::fixed(SETTING, ids.into_iter().collect()))
    }

    /// The list in `config`, kept current from the config file it came from
    /// when `reload` (a node with its own config file, not a scoped
    /// shared-serving membership).
    pub(crate) fn kept_current(
        config: &mesh_llm_config::MeshConfig,
        config_path: Option<&Path>,
        reload: bool,
    ) -> Self {
        Self(ReloadedSetting::kept_current(
            SETTING,
            config,
            config_path,
            reload,
        ))
    }

    /// Read the list from the config file now, rather than at the next check:
    /// a config apply does this before it reports the new list as live.
    pub(crate) fn reload_now(&self) {
        self.0.reload_now();
    }

    pub(crate) fn contains(&self, id: &EndpointId) -> bool {
        self.current().contains(id)
    }

    pub(crate) fn current(&self) -> Arc<HashSet<EndpointId>> {
        self.0.current()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn id(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    #[test]
    fn the_list_comes_from_the_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            format!("[private]\nnodes = [\"{}\", \"{}\"]\n", id(1), id(2)),
        )
        .unwrap();
        let config = mesh_llm_config::load_config(Some(&path)).unwrap();
        let nodes = PrivateNodes::kept_current(&config, Some(&path), false);
        assert_eq!(*nodes.current(), [id(1), id(2)].into());
        assert!(nodes.contains(&id(1)));
        assert!(!nodes.contains(&id(3)));
    }
}
