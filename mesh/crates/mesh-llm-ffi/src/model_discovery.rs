//! Lossless native model discovery bridge for platform SDKs.

use crate::errors::FfiError;
use crate::handles::MeshNodeHandle;
use crate::runtime_blocking::block_on;

#[uniffi::export]
impl MeshNodeHandle {
    /// Query all advertised workloads directly through the embedded node.
    pub fn mesh_models_json(&self, query_json: String) -> Result<String, FfiError> {
        let query = serde_json::from_str(&query_json)
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))?;
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        let page = block_on(node.models().list(query))
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))?;
        serde_json::to_string(&page).map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }

    /// Look up an exact public serving ID without applying LLM filtering.
    pub fn mesh_model_json(&self, id: String) -> Result<String, FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        let model = block_on(node.models().get(&id))
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))?;
        serde_json::to_string(&model).map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }
}
