/// `[private]`: the other nodes this node may send private requests to.
///
/// Each entry is a node id, 64 hex characters, as `mesh-llm auth status`
/// prints it. Empty means a private request never leaves this node.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct PrivateConfig {
    #[serde(default)]
    pub nodes: Vec<String>,
}

impl PrivateConfig {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The first entry that is not a node id, if any.
    pub fn invalid_node(&self) -> Option<&str> {
        self.nodes
            .iter()
            .map(String::as_str)
            .find(|id| !is_node_id(id))
    }
}

/// Whether `id` has the shape of a node id: 64 hex characters.
pub fn is_node_id(id: &str) -> bool {
    id.len() == 64 && id.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use crate::{config_to_toml, parse_config_toml};

    const ID: &str = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0";

    #[test]
    fn a_node_list_survives_a_save_and_reload() {
        let config = parse_config_toml(&format!("[private]\nnodes = [\"{ID}\"]\n")).unwrap();
        assert_eq!(config.private.nodes, [ID]);
        let saved = config_to_toml(&config).unwrap();
        assert_eq!(parse_config_toml(&saved).unwrap().private.nodes, [ID]);
    }

    #[test]
    fn an_empty_list_is_not_written() {
        let saved = config_to_toml(&parse_config_toml("").unwrap()).unwrap();
        assert!(!saved.contains("[private]"), "{saved}");
    }

    #[test]
    fn an_entry_that_is_not_a_node_id_is_rejected() {
        let error = parse_config_toml("[private]\nnodes = [\"studio\"]\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("private.nodes"), "{error}");
    }
}
