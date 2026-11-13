//! Native lifecycle subscriptions for Node.js.
use crate::{Node, to_napi_error};
use mesh_llm_sdk::embedded_node::status::StatusSubscription;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[napi]
pub struct StatusSubscriptionNative {
    subscription: Arc<StatusSubscription>,
}

#[napi]
impl Node {
    #[napi(js_name = "nativeStatusJson")]
    pub async fn native_status_json(&self) -> Result<String> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let status = node.native_status().await.map_err(to_napi_error)?;
        serde_json::to_string(&status).map_err(to_napi_error)
    }

    #[napi(js_name = "statusEvents")]
    pub async fn status_events(&self) -> Result<StatusSubscriptionNative> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        Ok(StatusSubscriptionNative {
            subscription: node.status_subscription().map_err(to_napi_error)?,
        })
    }
}

#[napi]
impl StatusSubscriptionNative {
    #[napi(js_name = "nextEventJson")]
    pub async fn next_event_json(&self) -> Result<Option<String>> {
        self.subscription
            .next()
            .await
            .map_err(to_napi_error)?
            .map(|status| serde_json::to_string(&status).map_err(to_napi_error))
            .transpose()
    }

    #[napi]
    pub async fn close(&self) {
        self.subscription.close().await;
    }
}
