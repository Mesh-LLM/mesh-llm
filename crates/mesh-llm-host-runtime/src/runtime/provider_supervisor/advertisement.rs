use super::*;
use crate::models;

pub(super) fn desired_provider_routes(
    model_id: &str,
    availability: &ProviderAvailability,
) -> Vec<String> {
    if !availability.available {
        return Vec::new();
    }
    let mut routes = vec![model_id.to_string()];
    routes.push(availability.versioned_model_id.clone());
    routes
}

pub(super) fn reconcile_provider_routes(
    availabilities: &[(String, ProviderAvailability)],
    routed_model_ids: &mut Vec<String>,
    port: u16,
    context: &ProviderSupervisorContext,
) {
    let desired = availabilities
        .iter()
        .flat_map(|(model_id, availability)| desired_provider_routes(model_id, availability))
        .collect::<Vec<_>>();
    for model_id in routed_model_ids.iter().filter(|id| !desired.contains(id)) {
        super::super::remove_runtime_local_target(&context.target_tx, model_id, port);
    }
    for model_id in desired.iter().filter(|id| !routed_model_ids.contains(id)) {
        super::super::add_runtime_local_target(&context.target_tx, model_id, port);
    }
    *routed_model_ids = desired;
}

pub(super) async fn publish_provider_state(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
    availability: &ProviderAvailability,
) {
    if !availability.available {
        tracing::debug!(
            reason = availability
                .unavailable_reason
                .as_deref()
                .unwrap_or("unknown"),
            "Apple provider model is currently unavailable"
        );
    }
    let status = if availability.available {
        "ready"
    } else {
        "unavailable"
    };
    upsert_provider_process(
        runtime,
        context,
        pid,
        port,
        status,
        availability.context_length,
    )
    .await;
}

pub(super) async fn publish_provider_unhealthy(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
) {
    upsert_provider_process(runtime, context, pid, port, "unhealthy", None).await;
}

async fn upsert_provider_process(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
    status: &str,
    context_length: Option<u32>,
) {
    let process_model_id = runtime
        .model_ids
        .first()
        .map(String::as_str)
        .unwrap_or(APPLE_MODEL_ID);
    let process = api::RuntimeProcessPayload {
        name: process_model_id.to_string(),
        instance_id: Some(provider_instance_id(process_model_id)),
        profile: String::new(),
        backend: runtime.runtime.manifest.runtime.provider_kind.clone(),
        status: status.to_string(),
        port,
        pid,
        slots: 1,
        context_length,
    };
    super::super::upsert_dashboard_process(&context.dashboard_processes, process.clone()).await;
    if let Some(console_state) = &context.console_state {
        console_state.upsert_local_process(process).await;
    }
}

pub(super) fn withdraw_provider_routes(
    routed_model_ids: &mut Vec<String>,
    port: u16,
    context: &ProviderSupervisorContext,
) {
    for model_id in routed_model_ids.drain(..) {
        super::super::remove_runtime_local_target(&context.target_tx, &model_id, port);
    }
}

pub(super) async fn reconcile_provider_advertisements(
    availabilities: &[(String, ProviderAvailability)],
    advertised_model_ids: &mut Vec<String>,
    context: &ProviderSupervisorContext,
) {
    let desired = availabilities
        .iter()
        .flat_map(|(model_id, availability)| desired_provider_routes(model_id, availability))
        .collect::<Vec<_>>();
    let previous = advertised_model_ids.clone();
    let mut changed = previous != desired;

    if previous != desired {
        reconcile_provider_model_names(&context.node, &previous, &desired).await;
    }
    for model_id in previous.iter().filter(|model| !desired.contains(model)) {
        changed |= context.node.remove_served_model_descriptor(model_id).await;
        changed |= context.node.remove_model_runtime_descriptor(model_id).await;
    }
    for model_id in &desired {
        let availability = availabilities
            .iter()
            .find(|(candidate, availability)| {
                candidate == model_id || availability.versioned_model_id == *model_id
            })
            .map(|(_, availability)| availability)
            .expect("desired provider route must have availability");
        context
            .node
            .upsert_served_model_descriptor(provider_served_model_descriptor(
                model_id,
                availability,
            ))
            .await;
        let runtime_descriptor = provider_runtime_descriptor(model_id, availability);
        let prev_descriptor = context.node.model_runtime_descriptor(model_id).await;
        changed |= stable_descriptor_changed(prev_descriptor.as_ref(), &runtime_descriptor);
        context
            .node
            .upsert_model_runtime_descriptor(runtime_descriptor)
            .await;
    }
    *advertised_model_ids = desired;
    if changed {
        context.node.regossip().await;
    }
}

async fn reconcile_provider_model_names(
    node: &mesh::Node,
    previous: &[String],
    desired: &[String],
) {
    let mut serving = node.serving_models().await;
    serving.retain(|model| !previous.contains(model));
    for model in desired {
        if !serving.contains(model) {
            serving.push(model.clone());
        }
    }
    node.set_serving_models(serving).await;

    let mut hosted = node.hosted_models().await;
    hosted.retain(|model| !previous.contains(model));
    for model in desired {
        if !hosted.contains(model) {
            hosted.push(model.clone());
        }
    }
    node.set_hosted_models(hosted).await;
}

fn provider_served_model_descriptor(
    model_id: &str,
    availability: &ProviderAvailability,
) -> mesh::ServedModelDescriptor {
    let capabilities = provider_model_capabilities(&availability.capabilities);
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_id.to_string(),
            canonical_ref: Some(availability.versioned_model_id.clone()),
            ..Default::default()
        },
        capabilities_known: true,
        capabilities,
        topology: None,
        metadata: Some(mesh::ServedModelMetadata {
            architecture: Some("apple_system".to_string()),
            native_context_length: availability.context_length,
            ..Default::default()
        }),
    }
}

fn provider_model_capabilities(capabilities: &[String]) -> models::ModelCapabilities {
    let supported = |name: &str| capabilities.iter().any(|value| value == name);
    let vision = supported("vision");
    models::ModelCapabilities {
        multimodal: vision,
        vision: if vision {
            models::CapabilityLevel::Supported
        } else {
            models::CapabilityLevel::None
        },
        reasoning: if supported("reasoning") {
            models::CapabilityLevel::Supported
        } else {
            models::CapabilityLevel::None
        },
        tool_use: if supported("tool_calling") {
            models::CapabilityLevel::Supported
        } else {
            models::CapabilityLevel::None
        },
        ..Default::default()
    }
}

fn provider_runtime_descriptor(
    model_id: &str,
    availability: &ProviderAvailability,
) -> mesh::ModelRuntimeDescriptor {
    mesh::ModelRuntimeDescriptor {
        model_name: model_id.to_string(),
        identity_hash: None,
        context_length: availability.context_length,
        ready: availability.available,
        provider_kind: Some(APPLE_PROVIDER_KIND.to_string()),
        model_version: Some(availability.model_version.clone()),
        max_concurrent_requests: Some(availability.max_concurrent_requests),
        active_requests: Some(availability.active_requests),
        queued_requests: Some(availability.queued_requests),
    }
}

fn stable_descriptor_changed(
    prev: Option<&mesh::ModelRuntimeDescriptor>,
    next: &mesh::ModelRuntimeDescriptor,
) -> bool {
    let Some(prev) = prev else {
        return true;
    };
    prev.model_name != next.model_name
        || prev.context_length != next.context_length
        || prev.ready != next.ready
        || prev.provider_kind != next.provider_kind
        || prev.model_version != next.model_version
        || prev.max_concurrent_requests != next.max_concurrent_requests
}

pub(super) async fn withdraw_provider_advertisement(
    advertised_model_ids: &mut Vec<String>,
    context: &ProviderSupervisorContext,
) {
    if advertised_model_ids.is_empty() {
        return;
    }
    let previous = std::mem::take(advertised_model_ids);
    reconcile_provider_model_names(&context.node, &previous, &[]).await;
    for model_id in previous {
        context.node.remove_served_model_descriptor(&model_id).await;
        context
            .node
            .remove_model_runtime_descriptor(&model_id)
            .await;
    }
    context.node.regossip().await;
}

pub(super) async fn remove_provider_process(
    context: &ProviderSupervisorContext,
    model_ids: &[String],
) {
    for model_id in model_ids {
        let instance_id = provider_instance_id(model_id);
        super::super::remove_dashboard_process(&context.dashboard_processes, &instance_id).await;
        if let Some(console_state) = &context.console_state {
            console_state.remove_local_process(&instance_id).await;
        }
    }
}

fn provider_instance_id(model_id: &str) -> String {
    format!("{PROVIDER_INSTANCE_PREFIX}{model_id}")
}
