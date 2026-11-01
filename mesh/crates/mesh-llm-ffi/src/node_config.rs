//! Full native configuration constructor shared by platform SDKs.

use crate::{FfiError, MeshNodeHandle};
use mesh_llm_sdk::{MeshNodeBuilder, embedded_node::EmbeddedMeshNodeConfig};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Construct a node from the complete native configuration. Starting the handle
/// performs the same runtime validation as the Rust builder.
#[uniffi::export]
pub fn create_node_config_json(config_json: String) -> Result<Arc<MeshNodeHandle>, FfiError> {
    let config: EmbeddedMeshNodeConfig = serde_json::from_str(&config_json)
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(Arc::new(MeshNodeHandle {
        builder: MeshNodeBuilder::from_config(config),
        node: Mutex::new(None),
        streams: Arc::new(Mutex::new(HashMap::new())),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_config_constructor_preserves_settings_without_starting_a_node() {
        let handle = create_node_config_json(r#"{"mode":"client","network":{"region":"region-a","enumerate_host":false},"startup_timeout":{"secs":9,"nanos":0}}"#.into()).unwrap();
        let config = handle.builder.clone().build();
        assert_eq!(config.network.region.as_deref(), Some("region-a"));
        assert!(!config.network.enumerate_host);
        assert_eq!(config.startup_timeout.as_secs(), 9);
        assert!(handle.node.lock().unwrap().is_none());
        assert!(create_node_config_json(r#"{"network":{"typo":true}}"#.into()).is_err());
    }
}
