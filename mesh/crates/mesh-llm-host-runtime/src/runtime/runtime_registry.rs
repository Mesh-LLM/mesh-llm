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
use skippy_inference_api::thinking::ThinkingControls;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub(super) type RuntimeInstanceRegistry =
    Arc<tokio::sync::Mutex<HashMap<String, BTreeMap<String, RuntimeInstanceMetadata>>>>;

pub(super) struct RuntimeInstanceMetadata {
    context_length: Option<u32>,
    thinking_efforts: Option<Vec<String>>,
}

pub(super) struct RuntimeModelRegistration<'a> {
    pub primary_model_name: &'a str,
    pub model_name: &'a str,
    pub instance_id: &'a str,
    pub context_length: Option<u32>,
    pub thinking: Option<&'a ThinkingControls>,
    pub backend: Option<Arc<dyn skippy_inference_api::InferenceBackend>>,
    pub port: u16,
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
    registration: RuntimeModelRegistration<'_>,
) {
    let RuntimeModelRegistration {
        primary_model_name,
        model_name,
        instance_id,
        context_length,
        thinking,
        backend,
        port,
        capabilities,
        workload_class,
    } = registration;
    node.register_native_backend(
        model_name,
        instance_id,
        port,
        backend,
        thinking.map(|controls| controls.efforts.clone()),
    )
    .await;
    let (was_empty, context_changed, next_context, next_efforts) = {
        let mut guard = registry.lock().await;
        let instances = guard.entry(model_name.to_string()).or_default();
        let previous_context = runtime_registry_model_context(instances);
        let was_empty = instances.is_empty();
        instances.insert(
            instance_id.to_string(),
            RuntimeInstanceMetadata {
                context_length,
                thinking_efforts: thinking.map(|controls| controls.efforts.clone()),
            },
        );
        let next_context = runtime_registry_model_context(instances);
        (
            was_empty,
            previous_context != next_context,
            next_context,
            model_thinking_efforts(instances),
        )
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
    }
    set_advertised_thinking_efforts(node, model_name, next_efforts).await;
    if was_empty {
        advertise_model_ready(node, primary_model_name, model_name, "").await;
    }
}

pub(super) async fn unregister_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    model_name: &str,
    instance_id: &str,
) -> bool {
    let (removed, became_empty, context_changed, next_context, next_efforts) = {
        let mut guard = registry.lock().await;
        let Some(instances) = guard.get_mut(model_name) else {
            return false;
        };
        let previous_context = runtime_registry_model_context(instances);
        let removed = instances.remove(instance_id).is_some();
        let next_context = runtime_registry_model_context(instances);
        let next_efforts = model_thinking_efforts(instances);
        let became_empty = instances.is_empty();
        if became_empty {
            guard.remove(model_name);
        }
        (
            removed,
            became_empty,
            previous_context != next_context,
            next_context,
            next_efforts,
        )
    };

    if !removed {
        return false;
    }
    node.unregister_native_backend(model_name, instance_id)
        .await;
    if became_empty {
        set_advertised_model_context(node, model_name, None).await;
        withdraw_advertised_model(node, model_name, "").await;
        remove_serving_assignment(node, model_name).await;
        crate::inference::skippy::forget_local_thinking(model_name);
        true
    } else {
        set_advertised_thinking_efforts(node, model_name, next_efforts).await;
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
    instances: &BTreeMap<String, RuntimeInstanceMetadata>,
) -> Option<u32> {
    instances
        .values()
        .filter_map(|instance| instance.context_length)
        .max()
}

/// A serving ID may select several loaded profiles. Advertise levels only when
/// the same observed list applies to every instance that can receive a request.
fn model_thinking_efforts(
    instances: &BTreeMap<String, RuntimeInstanceMetadata>,
) -> Option<Vec<String>> {
    let efforts = instances.values().next()?.thinking_efforts.as_ref()?;
    instances
        .values()
        .all(|instance| instance.thinking_efforts.as_ref() == Some(efforts))
        .then(|| efforts.clone())
}

async fn set_advertised_thinking_efforts(
    node: &mesh::Node,
    model_name: &str,
    efforts: Option<Vec<String>>,
) {
    node.update_served_model_descriptor(model_name, |existing| {
        let mut descriptor = existing.unwrap_or_else(|| mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                ..Default::default()
            },
            ..Default::default()
        });
        descriptor
            .metadata
            .get_or_insert_with(Default::default)
            .thinking_efforts = efforts;
        descriptor
    })
    .await;
    node.regossip().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn register(
        registry: &RuntimeInstanceRegistry,
        node: &mesh::Node,
        instance_id: &str,
        efforts: &[&str],
    ) {
        let thinking = ThinkingControls {
            efforts: efforts.iter().map(|effort| (*effort).to_string()).collect(),
            ..Default::default()
        };
        register_runtime_instance(
            registry,
            node,
            RuntimeModelRegistration {
                primary_model_name: "test-model",
                model_name: "test-model",
                instance_id,
                context_length: Some(8192),
                thinking: Some(&thinking),
                backend: None,
                port: 0,
                capabilities: Default::default(),
                workload_class: mesh::ModelWorkloadClass::CausalGeneration,
            },
        )
        .await;
    }

    async fn advertised_efforts(node: &mesh::Node) -> Option<Vec<String>> {
        node.served_model_descriptors()
            .await
            .into_iter()
            .find(|descriptor| descriptor.identity.model_name == "test-model")
            .and_then(|descriptor| descriptor.metadata)
            .and_then(|metadata| metadata.thinking_efforts)
    }

    #[tokio::test]
    async fn advertised_efforts_follow_registration_disagreement_unload_and_reload() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let registry = Arc::default();
        register(&registry, &node, "one", &["high", "low"]).await;
        assert_eq!(
            advertised_efforts(&node).await,
            Some(vec!["high".into(), "low".into()])
        );
        register(&registry, &node, "two", &["max"]).await;
        assert_eq!(advertised_efforts(&node).await, None);
        assert!(!unregister_runtime_instance(&registry, &node, "test-model", "two").await);
        assert_eq!(
            advertised_efforts(&node).await,
            Some(vec!["high".into(), "low".into()])
        );
        assert!(unregister_runtime_instance(&registry, &node, "test-model", "one").await);
        assert!(node.served_model_descriptors().await.is_empty());
        register(&registry, &node, "three", &[]).await;
        assert_eq!(advertised_efforts(&node).await, Some(vec![]));
    }

    fn observed(efforts: &[&str]) -> RuntimeInstanceMetadata {
        RuntimeInstanceMetadata {
            context_length: Some(8192),
            thinking_efforts: Some(efforts.iter().map(|effort| (*effort).into()).collect()),
        }
    }

    #[test]
    fn thinking_efforts_preserve_probe_order_and_unknown_presence() {
        let mut instances = BTreeMap::from([("one".into(), observed(&["high", "low"]))]);
        assert_eq!(
            model_thinking_efforts(&instances),
            Some(vec!["high".into(), "low".into()])
        );
        instances.insert("one".into(), observed(&[]));
        assert_eq!(model_thinking_efforts(&instances), Some(vec![]));
        instances.get_mut("one").unwrap().thinking_efforts = None;
        assert_eq!(model_thinking_efforts(&instances), None);
        instances.clear();
        assert_eq!(model_thinking_efforts(&instances), None);
    }

    #[test]
    fn thinking_efforts_require_agreement_across_loaded_instances() {
        let mut instances = BTreeMap::from([
            ("one".into(), observed(&["low", "high"])),
            ("two".into(), observed(&["low", "high"])),
        ]);
        assert!(model_thinking_efforts(&instances).is_some());
        instances.insert("two".into(), observed(&["max"]));
        assert_eq!(model_thinking_efforts(&instances), None);
        instances.get_mut("two").unwrap().thinking_efforts = None;
        assert_eq!(model_thinking_efforts(&instances), None);
        instances.remove("two");
        assert_eq!(
            model_thinking_efforts(&instances),
            Some(vec!["low".into(), "high".into()])
        );
    }
}
