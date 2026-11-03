//! Mesh membership controls through the native embedded node.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::{Node, to_napi_error};

#[napi]
impl Node {
    #[napi(js_name = "addMembershipJson")]
    pub async fn add_membership_json(&self, config_json: String) -> Result<String> {
        let config = serde_json::from_str(&config_json).map_err(to_napi_error)?;
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let status = node.add_membership(config).await.map_err(to_napi_error)?;
        serde_json::to_string(&status).map_err(to_napi_error)
    }

    #[napi(js_name = "removeMembership")]
    pub async fn remove_membership(&self, name: String) -> Result<bool> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        node.remove_membership(name).await.map_err(to_napi_error)
    }

    #[napi(js_name = "membershipsJson")]
    pub async fn memberships_json(&self) -> Result<String> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let statuses = node.memberships().await.map_err(to_napi_error)?;
        serde_json::to_string(&statuses).map_err(to_napi_error)
    }
}
