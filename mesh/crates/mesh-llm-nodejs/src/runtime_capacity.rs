//! BigInt fields preserve native unsigned capacities without JS number rounding.

use super::{Node, to_napi_error};
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct RuntimeCapacity {
    pub execution_lanes: BigInt,
    pub context_tokens_per_lane: BigInt,
    pub context_pool_tokens: BigInt,
    pub captured_at_unix_ms: BigInt,
}

impl From<mesh_llm_sdk::capacity::RuntimeCapacity> for RuntimeCapacity {
    fn from(value: mesh_llm_sdk::capacity::RuntimeCapacity) -> Self {
        Self {
            execution_lanes: BigInt::from(value.execution_lanes as u64),
            context_tokens_per_lane: BigInt::from(value.context_tokens_per_lane),
            context_pool_tokens: BigInt::from(value.context_pool_tokens),
            captured_at_unix_ms: BigInt::from(value.captured_at_unix_ms),
        }
    }
}

#[napi]
impl Node {
    #[napi(js_name = "runtimeCapacity")]
    pub async fn runtime_capacity(&self, model: String) -> Result<Option<RuntimeCapacity>> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        node.runtime_capacity(&model)
            .map(|capacity| capacity.map(RuntimeCapacity::from))
            .map_err(to_napi_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stopped_node_reports_an_error_instead_of_absent_capacity() {
        let node = Node::create_config("{}".into()).unwrap();
        assert!(node.runtime_capacity("model".into()).await.is_err());
    }

    #[test]
    fn capacity_fields_keep_full_unsigned_precision() {
        let capacity = RuntimeCapacity::from(mesh_llm_sdk::capacity::RuntimeCapacity {
            execution_lanes: 2,
            context_tokens_per_lane: u64::MAX / 2,
            context_pool_tokens: u64::MAX - 1,
            captured_at_unix_ms: u64::MAX,
        });
        assert_eq!(capacity.execution_lanes.get_u64(), (false, 2, true));
        assert_eq!(
            capacity.context_tokens_per_lane.get_u64(),
            (false, u64::MAX / 2, true)
        );
        assert_eq!(
            capacity.context_pool_tokens.get_u64(),
            (false, u64::MAX - 1, true)
        );
        assert_eq!(
            capacity.captured_at_unix_ms.get_u64(),
            (false, u64::MAX, true)
        );
    }
}
