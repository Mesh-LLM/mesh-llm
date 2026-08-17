use super::capacity::{
    RuntimeCapacityLedger, RuntimeCapacityPool, RuntimeCapacityRequest, RuntimeCapacityReservation,
    runtime_model_required_bytes,
};
use super::{
    StartupPinnedGpuTarget, add_serving_assignment, advertise_model_ready,
    remove_serving_assignment, set_advertised_model_context,
    set_runtime_verified_served_model_capabilities, withdraw_advertised_model,
};
use crate::{mesh, models};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub(super) type RuntimeInstanceRegistry =
    Arc<tokio::sync::Mutex<HashMap<String, BTreeMap<String, Option<u32>>>>>;

pub(super) fn next_runtime_instance_id(next_sequence: &mut u64) -> String {
    let instance_id = format!("runtime-{}", *next_sequence);
    *next_sequence = next_sequence.saturating_add(1);
    instance_id
}

pub(super) fn runtime_capacity_pool(
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
) -> RuntimeCapacityPool {
    pinned_gpu
        .map(|gpu| RuntimeCapacityPool::PinnedGpu(gpu.stable_id.clone()))
        .unwrap_or(RuntimeCapacityPool::Node)
}

pub(super) fn runtime_capacity_request_for_model(
    instance_id: &str,
    model_name: &str,
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
    capacity_bytes: u64,
    model_bytes: u64,
) -> RuntimeCapacityRequest {
    RuntimeCapacityRequest {
        instance_id: instance_id.to_string(),
        model_name: model_name.to_string(),
        pool: runtime_capacity_pool(pinned_gpu),
        capacity_bytes,
        required_bytes: runtime_model_required_bytes(model_bytes),
    }
}

pub(super) fn reserve_runtime_capacity_for_model(
    ledger: &RuntimeCapacityLedger,
    instance_id: &str,
    model_name: &str,
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
    capacity_bytes: u64,
    model_bytes: u64,
) -> Result<RuntimeCapacityReservation> {
    ledger
        .reserve(runtime_capacity_request_for_model(
            instance_id,
            model_name,
            pinned_gpu,
            capacity_bytes,
            model_bytes,
        ))
        .map_err(Into::into)
}

pub(super) async fn register_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
    instance_id: &str,
    context_length: Option<u32>,
    capabilities: models::ModelCapabilities,
) {
    let (was_empty, context_changed, next_context) = {
        let mut guard = registry.lock().await;
        let instances = guard.entry(model_name.to_string()).or_default();
        let previous_context = runtime_registry_model_context(instances);
        let was_empty = instances.is_empty();
        instances.insert(instance_id.to_string(), context_length);
        let next_context = runtime_registry_model_context(instances);
        (was_empty, previous_context != next_context, next_context)
    };

    if context_changed {
        set_advertised_model_context(node, model_name, next_context).await;
    }
    if was_empty {
        add_serving_assignment(node, primary_model_name, model_name).await;
        set_runtime_verified_served_model_capabilities(
            node,
            primary_model_name,
            model_name,
            capabilities,
        )
        .await;
        advertise_model_ready(node, primary_model_name, model_name, "").await;
    }
}

pub(super) async fn unregister_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    model_name: &str,
    instance_id: &str,
) -> bool {
    let (removed, became_empty, context_changed, next_context) = {
        let mut guard = registry.lock().await;
        let Some(instances) = guard.get_mut(model_name) else {
            return false;
        };
        let previous_context = runtime_registry_model_context(instances);
        let removed = instances.remove(instance_id).is_some();
        let next_context = runtime_registry_model_context(instances);
        let became_empty = instances.is_empty();
        if became_empty {
            guard.remove(model_name);
        }
        (
            removed,
            became_empty,
            previous_context != next_context,
            next_context,
        )
    };

    if !removed {
        return false;
    }
    if became_empty {
        set_advertised_model_context(node, model_name, None).await;
        withdraw_advertised_model(node, model_name, "").await;
        remove_serving_assignment(node, model_name).await;
        true
    } else {
        if context_changed {
            set_advertised_model_context(node, model_name, next_context).await;
        }
        false
    }
}

pub(super) async fn runtime_registry_has_model(
    registry: &RuntimeInstanceRegistry,
    model_name: &str,
) -> bool {
    registry
        .lock()
        .await
        .get(model_name)
        .map(|instances| !instances.is_empty())
        .unwrap_or(false)
}

pub(super) fn runtime_registry_model_context(
    instances: &BTreeMap<String, Option<u32>>,
) -> Option<u32> {
    instances.values().filter_map(|context| *context).max()
}
