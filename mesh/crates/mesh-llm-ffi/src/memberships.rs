//! Mesh membership controls through the native embedded node.

use crate::errors::FfiError;
use crate::handles::MeshNodeHandle;
use crate::runtime_blocking::block_on;

#[uniffi::export]
impl MeshNodeHandle {
    pub fn add_membership_json(&self, config_json: String) -> Result<String, FfiError> {
        let config = serde_json::from_str(&config_json)
            .map_err(|error| FfiError::JoinFailed(error.to_string()))?;
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        let status = block_on(node.add_membership(config))
            .map_err(|error| FfiError::JoinFailed(error.to_string()))?;
        serde_json::to_string(&status).map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn remove_membership(&self, name: String) -> Result<bool, FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        block_on(node.remove_membership(name))
            .map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn memberships_json(&self) -> Result<String, FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".to_string()))?;
        let statuses = block_on(node.memberships())
            .map_err(|error| FfiError::JoinFailed(error.to_string()))?;
        serde_json::to_string(&statuses).map_err(|error| FfiError::JoinFailed(error.to_string()))
    }
}
