//! Native lifecycle subscriptions shared by Swift, Kotlin and Python.
use crate::{FfiError, MeshNodeHandle, runtime_blocking::block_on};
use mesh_llm_sdk::embedded_node::status::StatusSubscription;
use std::sync::Arc;

#[derive(uniffi::Object)]
pub struct StatusSubscriptionNative {
    subscription: Arc<StatusSubscription>,
}

#[uniffi::export]
impl MeshNodeHandle {
    pub fn native_status_json(&self) -> Result<String, FfiError> {
        let node = self.node.lock().map_err(status_error)?;
        let node = node
            .as_ref()
            .ok_or_else(|| status_error("node is not running"))?;
        let status = block_on(node.native_status()).map_err(status_error)?;
        serde_json::to_string(&status).map_err(status_error)
    }

    pub fn status_events(&self) -> Result<Arc<StatusSubscriptionNative>, FfiError> {
        let node = self.node.lock().map_err(status_error)?;
        let node = node
            .as_ref()
            .ok_or_else(|| status_error("node is not running"))?;
        Ok(Arc::new(StatusSubscriptionNative {
            subscription: node.status_subscription().map_err(status_error)?,
        }))
    }
}

#[uniffi::export]
impl StatusSubscriptionNative {
    pub fn next_event_json(&self) -> Result<Option<String>, FfiError> {
        block_on(self.subscription.next())
            .map_err(status_error)?
            .map(|status| serde_json::to_string(&status).map_err(status_error))
            .transpose()
    }

    pub fn close_events(&self) {
        block_on(self.subscription.close());
    }
}

fn status_error(error: impl ToString) -> FfiError {
    FfiError::HostUnavailable(error.to_string())
}
