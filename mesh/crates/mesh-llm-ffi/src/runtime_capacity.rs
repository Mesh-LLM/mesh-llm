//! Lossless local runtime observations for UniFFI SDKs.

use crate::{FfiError, MeshNodeHandle};
use mesh_llm_sdk::capacity::LaneCapacity;

#[derive(Clone, Debug, PartialEq, Eq, uniffi::Record)]
pub struct RuntimeCapacityNative {
    pub execution_lanes: u64,
    pub context_tokens_per_lane: u64,
    pub context_pool_tokens: u64,
    pub captured_at_unix_ms: u64,
}

impl From<LaneCapacity> for RuntimeCapacityNative {
    fn from(value: LaneCapacity) -> Self {
        Self {
            execution_lanes: value.execution_lanes,
            context_tokens_per_lane: value.context_tokens_per_lane,
            context_pool_tokens: value.context_pool_tokens,
            captured_at_unix_ms: value.captured_at_unix_ms,
        }
    }
}

#[uniffi::export]
impl MeshNodeHandle {
    /// Current local capacity for an exact serving ID; unavailable is not zero.
    pub fn runtime_capacity(
        &self,
        model: String,
    ) -> Result<Option<RuntimeCapacityNative>, FfiError> {
        let node = self
            .node
            .lock()
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?;
        let node = node
            .as_ref()
            .ok_or_else(|| FfiError::HostUnavailable("node is not running".into()))?;
        Ok(node
            .runtime_capacity(&model)
            .map_err(|error| FfiError::HostUnavailable(error.to_string()))?
            .map(RuntimeCapacityNative::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_preserves_unsigned_capacity_and_refuses_stopped_nodes() {
        let capacity = RuntimeCapacityNative::from(LaneCapacity {
            execution_lanes: 2,
            context_tokens_per_lane: u64::MAX / 2,
            context_pool_tokens: u64::MAX - 1,
            captured_at_unix_ms: u64::MAX,
        });
        assert_eq!(capacity.execution_lanes, 2);
        assert_eq!(capacity.context_tokens_per_lane, u64::MAX / 2);
        assert_eq!(capacity.context_pool_tokens, u64::MAX - 1);
        assert_eq!(capacity.captured_at_unix_ms, u64::MAX);
        let node = crate::create_node_config_json("{}".into()).unwrap();
        assert!(matches!(
            node.runtime_capacity("model".into()),
            Err(FfiError::HostUnavailable(_))
        ));
    }
}
