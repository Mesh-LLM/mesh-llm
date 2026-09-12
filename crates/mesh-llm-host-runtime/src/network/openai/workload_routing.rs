//! Workload admission at the model and individual serving-target boundaries.
//!
//! Model-level admission is existential: any matching advertisement can make a
//! model discoverable. It never grants another peer that model's capabilities.
//! Both host and passive-client dispatch filter their concrete targets before
//! context ranking, affinity, reservations, and retry selection.

use crate::inference::election::InferenceTarget;
use crate::mesh::model_identity::descriptor_matches_routable_name;
use crate::mesh::{self, ModelWorkloadClass, ServedModelDescriptor};

#[cfg(test)]
mod tests;

pub(super) fn is_audio_upload_path(path: &str) -> bool {
    matches!(
        path.split('?').next().unwrap_or(path),
        "/v1/audio/transcriptions" | "/v1/audio/translations"
    )
}

pub(super) fn request_workload_class(path: &str) -> Option<ModelWorkloadClass> {
    match path.split('?').next().unwrap_or(path) {
        "/v1/chat/completions"
        | "/v1/completions"
        | "/v1/responses"
        | "/v1/audio/transcriptions"
        | "/v1/audio/translations" => Some(ModelWorkloadClass::CausalGeneration),
        "/v1/embeddings" => Some(ModelWorkloadClass::Embedding),
        "/v1/rerank" => Some(ModelWorkloadClass::Rerank),
        "/v1/audio/speech" => Some(ModelWorkloadClass::SpeechSynthesis),
        _ => None,
    }
}

/// Only generation requests reuse KV/session state. Metadata such as an
/// embeddings `user` field must not pin a stateless workload to one replica.
pub(super) fn supports_generation_affinity(path: &str) -> bool {
    matches!(
        path.split('?').next().unwrap_or(path),
        "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
    )
}

pub(super) fn affinity_body(
    request: &super::request_parse::BufferedHttpRequest,
) -> Option<&serde_json::Value> {
    supports_generation_affinity(&request.client_path)
        .then_some(request.body_json.as_ref())
        .flatten()
}

/// Prefer the elected targets only when at least one supports this request.
/// A stale/local incompatible copy must not shadow capable remote replicas.
pub(super) async fn ingress_candidates(
    node: &mesh::Node,
    model: &str,
    path: &str,
    targets: &crate::inference::election::ModelTargets,
) -> Vec<InferenceTarget> {
    let local = eligible_targets(node, model, path, &targets.candidates(model)).await;
    if local
        .iter()
        .any(|target| !matches!(target, InferenceTarget::None))
    {
        return local;
    }
    let remote = node
        .hosts_for_model(model)
        .await
        .into_iter()
        .map(InferenceTarget::Remote)
        .collect::<Vec<_>>();
    eligible_targets(node, model, path, &remote).await
}

fn class_is_compatible(
    requested: ModelWorkloadClass,
    advertised: Option<ModelWorkloadClass>,
) -> bool {
    match (requested, advertised) {
        (ModelWorkloadClass::CausalGeneration, None) => true,
        (
            ModelWorkloadClass::CausalGeneration,
            Some(ModelWorkloadClass::CausalGeneration | ModelWorkloadClass::EncoderDecoder),
        ) => true,
        (requested, Some(advertised)) => requested == advertised,
        (_, None) => false,
    }
}

pub(super) fn model_satisfies_workload_class(
    model: &str,
    requested: ModelWorkloadClass,
    descriptors: &[ServedModelDescriptor],
) -> bool {
    let mut matching = descriptors
        .iter()
        .filter(|descriptor| descriptor_matches_routable_name(descriptor, model))
        .peekable();
    if matching.peek().is_none() {
        return class_is_compatible(requested, None);
    }
    matching.any(|descriptor| {
        class_is_compatible(
            requested,
            descriptor
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.workload_class),
        )
    })
}

fn descriptor_supports_audio_upload(descriptor: &ServedModelDescriptor) -> bool {
    descriptor.capabilities_known
        && descriptor.capabilities.supports_audio_runtime()
        && matches!(
            descriptor
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.workload_class),
            Some(ModelWorkloadClass::CausalGeneration | ModelWorkloadClass::EncoderDecoder)
        )
}

pub(super) fn model_satisfies_request_workload(
    model: &str,
    workload: ModelWorkloadClass,
    path: &str,
    descriptors: &[ServedModelDescriptor],
) -> bool {
    if is_audio_upload_path(path) {
        descriptors.iter().any(|descriptor| {
            descriptor_matches_routable_name(descriptor, model)
                && descriptor_supports_audio_upload(descriptor)
        })
    } else {
        model_satisfies_workload_class(model, workload, descriptors)
    }
}

pub(super) fn descriptor_for_request<'a>(
    model: &str,
    path: &str,
    descriptors: &'a [ServedModelDescriptor],
) -> Option<&'a ServedModelDescriptor> {
    descriptors.iter().find(|descriptor| {
        descriptor_matches_routable_name(descriptor, model)
            && request_workload_class(path).is_none_or(|workload| {
                model_satisfies_request_workload(
                    model,
                    workload,
                    path,
                    std::slice::from_ref(*descriptor),
                )
            })
    })
}

pub(super) fn request_media(
    path: &str,
    body: Option<&serde_json::Value>,
) -> crate::network::router::MediaRequirements {
    if is_audio_upload_path(path) {
        crate::network::router::MediaRequirements {
            has_media: true,
            needs_audio: true,
            needs_vision: false,
        }
    } else {
        body.map_or_else(Default::default, crate::network::router::media_requirements)
    }
}

/// Pick metadata from a descriptor that supports this request, not whichever
/// peer happened to gossip the model name first.
pub(super) fn routing_candidates<'a>(
    node: &mesh::Node,
    models: &'a [String],
    path: &str,
    descriptors: &[ServedModelDescriptor],
) -> Vec<crate::network::router::RoutingCandidate<'a>> {
    let metrics = node.routing_metrics();
    models
        .iter()
        .filter(|model| {
            request_workload_class(path).is_none_or(|workload| {
                model_satisfies_request_workload(model, workload, path, descriptors)
            })
        })
        .map(|model| {
            let descriptor = descriptor_for_request(model, path, descriptors);
            let caps = descriptor.map_or_else(
                || super::routing_rank::capabilities_for_model(model, descriptors),
                |descriptor| descriptor.capabilities,
            );
            let (tps_hint, throughput_samples) = metrics
                .tps_for_model(model)
                .map(|(tps, samples)| (Some(tps), samples))
                .unwrap_or((None, 0));
            crate::network::router::RoutingCandidate {
                name: model,
                caps,
                parameter_count_b: descriptor
                    .and_then(|descriptor| descriptor.metadata.as_ref())
                    .and_then(|metadata| metadata.parameter_count_b),
                tps_hint,
                throughput_samples,
            }
        })
        .collect()
}

pub(super) async fn eligible_targets(
    node: &mesh::Node,
    model: &str,
    path: &str,
    candidates: &[InferenceTarget],
) -> Vec<InferenceTarget> {
    let Some(workload) = request_workload_class(path) else {
        return candidates.to_vec();
    };
    let local = node.served_model_descriptors().await;
    let state = node.state.lock().await;
    candidates
        .iter()
        .filter(|target| {
            let descriptors = match target {
                InferenceTarget::Local(_) => local.as_slice(),
                InferenceTarget::Remote(peer_id) => state
                    .peers
                    .get(peer_id)
                    .map_or(&[][..], |peer| peer.served_model_descriptors.as_slice()),
                InferenceTarget::None => return false,
            };
            model_satisfies_request_workload(model, workload, path, descriptors)
        })
        .cloned()
        .collect()
}

pub(super) async fn eligible_remote_hosts(
    node: &mesh::Node,
    model: &str,
    path: &str,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let candidates = hosts
        .iter()
        .copied()
        .map(InferenceTarget::Remote)
        .collect::<Vec<_>>();
    eligible_targets(node, model, path, &candidates)
        .await
        .into_iter()
        .filter_map(|target| match target {
            InferenceTarget::Remote(peer) => Some(peer),
            _ => None,
        })
        .collect()
}
