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

#[derive(Clone, Copy)]
pub(super) struct RuntimeCapacityTarget<'a> {
    pub(super) compact_meta: Option<&'a models::gguf::GgufCompactMeta>,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) device_override: Option<&'a str>,
    pub(super) node_capacity_bytes: u64,
    pub(super) host_ram_capacity_bytes: u64,
}

pub(super) struct RuntimeModelRegistration<'a> {
    pub primary_model_name: &'a str,
    pub model_name: &'a str,
    pub instance_id: &'a str,
    pub context_length: Option<u32>,
    pub capabilities: models::ModelCapabilities,
    pub workload_class: mesh::ModelWorkloadClass,
}

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

pub(super) fn runtime_capacity_pool_for_model(
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
    device_override: Option<&str>,
    host_ram_capacity_bytes: u64,
) -> RuntimeCapacityPool {
    if super::local_laya::is_laya(compact_meta) {
        if device_override.is_none_or(|device| device.eq_ignore_ascii_case("cpu")) {
            return if host_ram_capacity_bytes > 0 {
                RuntimeCapacityPool::HostRam
            } else {
                RuntimeCapacityPool::Node
            };
        }
        if pinned_gpu
            .zip(device_override)
            .is_some_and(|(gpu, device)| gpu.backend_device.eq_ignore_ascii_case(device))
        {
            return runtime_capacity_pool(pinned_gpu);
        }
        return RuntimeCapacityPool::Node;
    }

    runtime_capacity_pool(pinned_gpu)
}

pub(super) fn runtime_capacity_limit_for_model(target: RuntimeCapacityTarget<'_>) -> u64 {
    match runtime_capacity_pool_for_model(
        target.compact_meta,
        target.pinned_gpu,
        target.device_override,
        target.host_ram_capacity_bytes,
    ) {
        RuntimeCapacityPool::HostRam => target.host_ram_capacity_bytes,
        RuntimeCapacityPool::Node | RuntimeCapacityPool::PinnedGpu(_) => target.node_capacity_bytes,
    }
}

pub(super) fn runtime_capacity_request_for_model(
    instance_id: &str,
    model_name: &str,
    target: RuntimeCapacityTarget<'_>,
    model_bytes: u64,
) -> RuntimeCapacityRequest {
    let pool = runtime_capacity_pool_for_model(
        target.compact_meta,
        target.pinned_gpu,
        target.device_override,
        target.host_ram_capacity_bytes,
    );
    RuntimeCapacityRequest {
        instance_id: instance_id.to_string(),
        model_name: model_name.to_string(),
        capacity_bytes: runtime_capacity_limit_for_model(target),
        pool,
        required_bytes: runtime_model_required_bytes(model_bytes),
    }
}

pub(super) fn reserve_runtime_capacity_for_model(
    ledger: &RuntimeCapacityLedger,
    instance_id: &str,
    model_name: &str,
    target: RuntimeCapacityTarget<'_>,
    model_bytes: u64,
) -> Result<RuntimeCapacityReservation> {
    ledger
        .reserve(runtime_capacity_request_for_model(
            instance_id,
            model_name,
            target,
            model_bytes,
        ))
        .map_err(Into::into)
}

pub(super) async fn register_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    registration: RuntimeModelRegistration<'_>,
) {
    let RuntimeModelRegistration {
        primary_model_name,
        model_name,
        instance_id,
        context_length,
        capabilities,
        workload_class,
    } = registration;
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
            workload_class,
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
        crate::inference::skippy::forget_local_thinking(model_name);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(architecture: &str) -> models::gguf::GgufCompactMeta {
        models::gguf::GgufCompactMeta {
            architecture: architecture.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn laya_pool_uses_architecture_and_resolved_device() {
        let laya = metadata("laya");
        let llama_named_laya = metadata("llama");
        let gpu = StartupPinnedGpuTarget {
            index: 0,
            stable_id: "gpu-a".to_string(),
            backend_device: "CUDA0".to_string(),
            vram_bytes: 1_000,
            reserved_bytes: None,
        };

        assert_eq!(
            runtime_capacity_pool_for_model(Some(&laya), None, None, 800),
            RuntimeCapacityPool::HostRam
        );
        assert_eq!(
            runtime_capacity_limit_for_model(RuntimeCapacityTarget {
                compact_meta: Some(&laya),
                pinned_gpu: None,
                device_override: None,
                node_capacity_bytes: 1_000,
                host_ram_capacity_bytes: 800,
            }),
            800,
        );
        assert_eq!(
            runtime_capacity_pool_for_model(Some(&laya), None, Some("auto"), 800),
            RuntimeCapacityPool::Node,
            "the resolved accelerator target must use accelerator capacity"
        );
        assert_eq!(
            runtime_capacity_pool_for_model(Some(&laya), Some(&gpu), Some("CUDA0"), 800),
            RuntimeCapacityPool::PinnedGpu("gpu-a".to_string())
        );
        assert_eq!(
            runtime_capacity_pool_for_model(Some(&laya), Some(&gpu), Some("CPU"), 800),
            RuntimeCapacityPool::HostRam
        );
        assert_eq!(
            runtime_capacity_pool_for_model(Some(&llama_named_laya), None, None, 800),
            RuntimeCapacityPool::Node,
            "a display name cannot turn a non-Laya architecture into a Laya runtime"
        );
        let named_laya_request = runtime_capacity_request_for_model(
            "llama",
            "Laya-compatible display name",
            RuntimeCapacityTarget {
                compact_meta: Some(&llama_named_laya),
                pinned_gpu: None,
                device_override: None,
                node_capacity_bytes: 1_000,
                host_ram_capacity_bytes: 800,
            },
            700,
        );
        assert_eq!(named_laya_request.pool, RuntimeCapacityPool::Node);
        assert_eq!(named_laya_request.capacity_bytes, 1_000);
        assert_eq!(
            runtime_capacity_pool_for_model(Some(&laya), None, None, 0),
            RuntimeCapacityPool::Node,
            "missing host-RAM capacity remains unknown"
        );
    }

    #[test]
    fn laya_request_uses_host_ram_capacity_for_cpu_target() {
        let laya = metadata("laya");
        let request = runtime_capacity_request_for_model(
            "laya-cpu",
            "model-with-arbitrary-name",
            RuntimeCapacityTarget {
                compact_meta: Some(&laya),
                pinned_gpu: None,
                device_override: None,
                node_capacity_bytes: 4_000,
                host_ram_capacity_bytes: 800,
            },
            700,
        );

        assert_eq!(request.pool, RuntimeCapacityPool::HostRam);
        assert_eq!(request.capacity_bytes, 800);
    }
}
