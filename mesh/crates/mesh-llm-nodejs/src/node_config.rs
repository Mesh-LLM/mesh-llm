//! Full native configuration factory for Node.js.

use super::{Node, to_napi_error};
use mesh_llm_sdk::{
    MeshNodeBuilder,
    embedded_node::{EmbeddedMeshNodeConfig, EmbeddedMeshNodeMode},
};
use napi::bindgen_prelude::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub(super) fn create_config(config_json: String) -> Result<Node> {
    let config: EmbeddedMeshNodeConfig =
        serde_json::from_str(&config_json).map_err(to_napi_error)?;
    let mode = match &config.mode {
        EmbeddedMeshNodeMode::Client => "client",
        EmbeddedMeshNodeMode::ServeOnly => "serve",
        EmbeddedMeshNodeMode::Serve => "combined",
    }
    .to_string();
    Ok(Node {
        builder: MeshNodeBuilder::from_config(config),
        node: tokio::sync::Mutex::new(None),
        mode,
        streams: Arc::new(Mutex::new(HashMap::new())),
    })
}
