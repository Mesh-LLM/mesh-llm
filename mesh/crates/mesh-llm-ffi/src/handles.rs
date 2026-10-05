use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::task::AbortHandle;

#[derive(uniffi::Object)]
pub struct MeshNodeHandle {
    pub(crate) builder: mesh_llm_sdk::MeshNodeBuilder,
    pub(crate) node: Mutex<Option<mesh_llm_sdk::MeshNode>>,
    pub(crate) streams: Arc<Mutex<HashMap<String, AbortHandle>>>,
}
