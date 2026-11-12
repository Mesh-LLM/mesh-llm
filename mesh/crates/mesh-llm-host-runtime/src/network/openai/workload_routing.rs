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

/// Recognize binary audio-to-text endpoints independently of query parameters.
pub(super) fn is_audio_upload_path(path: &str) -> bool {
    matches!(
        path.split('?').next().unwrap_or(path),
        "/v1/audio/transcriptions" | "/v1/audio/translations"
    )
}

/// Recognize document OCR uploads independently of query parameters.
pub(super) fn is_document_ocr_path(path: &str) -> bool {
    path.split('?').next().unwrap_or(path) == "/mesh/v1/documents/ocr"
}

/// Recognize the typed System One endpoints independently of query parameters.
/// `/v1/systemone` is the canonical Jev interface path; `/systemone` is the
/// legacy Skippy mount kept for compatibility.
pub(super) fn is_system_one_path(path: &str) -> bool {
    matches!(
        path.split('?').next().unwrap_or(path),
        "/v1/systemone" | "/systemone" | "/v1/decisions"
    )
}

/// Map inference endpoints to native workloads; control paths impose no class.
pub(super) fn request_workload_class(path: &str) -> Option<ModelWorkloadClass> {
    match path.split('?').next().unwrap_or(path) {
        "/v1/chat/completions"
        | "/v1/completions"
        | "/v1/responses"
        | "/v1/messages"
        | "/v1/audio/transcriptions"
        | "/v1/audio/translations" => Some(ModelWorkloadClass::CausalGeneration),
        "/v1/embeddings" => Some(ModelWorkloadClass::Embedding),
        "/v1/rerank" => Some(ModelWorkloadClass::Rerank),
        "/v1/audio/speech" => Some(ModelWorkloadClass::SpeechSynthesis),
        "/v1/realtime" => Some(ModelWorkloadClass::SpeechRecognition),
        "/mesh/v1/documents/ocr" => Some(ModelWorkloadClass::DocumentOcr),
        "/mesh/v1/images/similarity" => Some(ModelWorkloadClass::ImageTextSimilarity),
        "/mesh/v1/images/depth" => Some(ModelWorkloadClass::DepthEstimation),
        _ => None,
    }
}

/// Workload named in an admission error when an endpoint has no eligible model.
/// System One remains capability-gated rather than class-gated, but uses the
/// decision class to describe the failed request contract to clients.
pub(super) fn required_request_workload(path: &str) -> Option<ModelWorkloadClass> {
    request_workload_class(path)
        .or_else(|| is_system_one_path(path).then_some(ModelWorkloadClass::Decision))
}

/// Explain endpoint admission failures without leaking routing implementation details.
pub(super) fn unsupported_workload_message(path: &str, workload: ModelWorkloadClass) -> String {
    if is_system_one_path(path) {
        "no served model advertises System One support".to_string()
    } else if is_audio_upload_path(path) {
        "no served model advertises support for this audio-to-text endpoint".to_string()
    } else {
        format!("no served model advertises the required {workload:?} workload")
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

/// Expose identity hints only for endpoints whose generation state is reusable.
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

/// Every placement of `model` this node may dispatch to: its own eligible
/// targets first, then every eligible peer. Automatic routing draws its host
/// from this set so a request that arrived at a slow host can still land on a
/// faster peer in proportion to capacity; a named model keeps the local-first
/// [`ingress_candidates`] contract.
pub(super) async fn mesh_candidates(
    node: &mesh::Node,
    model: &str,
    path: &str,
    targets: &crate::inference::election::ModelTargets,
) -> Vec<InferenceTarget> {
    let mut candidates = eligible_targets(node, model, path, &targets.candidates(model))
        .await
        .into_iter()
        .filter(|target| !matches!(target, InferenceTarget::None))
        .collect::<Vec<_>>();
    let remote = node
        .hosts_for_model(model)
        .await
        .into_iter()
        .map(InferenceTarget::Remote)
        .filter(|target| !candidates.contains(target))
        .collect::<Vec<_>>();
    candidates.extend(eligible_targets(node, model, path, &remote).await);
    candidates
}

/// A peer advertises a model under its public id while this node serves the
/// same model under its runtime identity, so the automatic candidate list sees
/// one model twice: the identity (whose targets are local) and the public id
/// (whose targets are the peers). Drawing between the two names decided the
/// host before the host picker ran (lab, 2026-11-08: 10/10 identity picks went
/// local, 10/10 public-id picks went remote, two-choice never ran). Fold the
/// public id into the identity so the model is one candidate whose placements
/// cover every host.
/// The public ids in `models` that fold into a listed local identity without
/// losing a placement: every host reachable under the public id is also
/// reachable under the identity. A peer whose own runtime name differs from
/// this node's is reachable only under the public id (peer discovery projects
/// its runtime name to the public id, `mesh::peer_state::routable_models`),
/// so that public id stays a candidate of its own rather than losing the peer.
pub(super) async fn folded_public_aliases(
    node: &mesh::Node,
    models: &[String],
    descriptors: &[ServedModelDescriptor],
) -> Vec<String> {
    let mut folded = Vec::new();
    for name in models {
        if !is_public_alias_of_local_model(name, models, descriptors) {
            continue;
        }
        let hosts_under_public_id = node.hosts_for_model(name).await;
        let identities = descriptors.iter().filter_map(|descriptor| {
            (descriptor.identity.model_name != *name
                && descriptor_matches_routable_name(descriptor, name)
                && models.contains(&descriptor.identity.model_name))
            .then_some(descriptor.identity.model_name.as_str())
        });
        let mut hosts_under_identities = Vec::new();
        for identity in identities {
            hosts_under_identities.extend(node.hosts_for_model(identity).await);
        }
        if hosts_under_public_id
            .iter()
            .all(|host| hosts_under_identities.contains(host))
        {
            folded.push(name.clone());
        }
    }
    folded
}

pub(super) fn is_public_alias_of_local_model(
    name: &str,
    models: &[String],
    descriptors: &[ServedModelDescriptor],
) -> bool {
    if descriptors
        .iter()
        .any(|descriptor| descriptor.identity.model_name == name)
    {
        return false;
    }
    descriptors.iter().any(|descriptor| {
        descriptor_matches_routable_name(descriptor, name)
            && models.contains(&descriptor.identity.model_name)
    })
}

/// Preserve absent legacy chat metadata, but never authorize an unknown class.
fn class_is_compatible(
    requested: ModelWorkloadClass,
    advertised: Option<ModelWorkloadClass>,
) -> bool {
    match (requested, advertised) {
        (ModelWorkloadClass::Unknown, _) | (_, Some(ModelWorkloadClass::Unknown)) => false,
        (ModelWorkloadClass::CausalGeneration, None) => true,
        (
            ModelWorkloadClass::CausalGeneration,
            Some(ModelWorkloadClass::CausalGeneration | ModelWorkloadClass::EncoderDecoder),
        ) => true,
        (requested, Some(advertised)) => requested == advertised,
        (_, None) => false,
    }
}

/// Check discoverability across matching descriptors, not target eligibility.
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

/// Require verified audio and a known audio-to-text class on the same
/// descriptor: a generation model with an audio encoder, or a dedicated
/// speech-recognition model.
pub(crate) fn descriptor_supports_audio_upload(descriptor: &ServedModelDescriptor) -> bool {
    descriptor.capabilities_known
        && descriptor.capabilities.supports_audio_runtime()
        && matches!(
            descriptor
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.workload_class),
            Some(
                ModelWorkloadClass::CausalGeneration
                    | ModelWorkloadClass::EncoderDecoder
                    | ModelWorkloadClass::SpeechRecognition
            )
        )
}

/// System One is a runtime-verified endpoint capability, independent of the
/// model's primary workload class.
pub(crate) fn descriptor_supports_system_one(descriptor: &ServedModelDescriptor) -> bool {
    descriptor.capabilities_known && descriptor.capabilities.supports_system_one_runtime()
}

pub(super) fn model_satisfies_system_one(
    model: &str,
    descriptors: &[ServedModelDescriptor],
) -> bool {
    descriptors.iter().any(|descriptor| {
        descriptor_matches_routable_name(descriptor, model)
            && descriptor_supports_system_one(descriptor)
    })
}

/// Apply endpoint-specific admission, including stricter audio upload metadata.
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

/// Apply every endpoint-specific model requirement, including interfaces that
/// do not map one-to-one onto a primary workload class.
pub(super) fn model_satisfies_request(
    model: &str,
    path: &str,
    descriptors: &[ServedModelDescriptor],
) -> bool {
    if is_system_one_path(path) {
        model_satisfies_system_one(model, descriptors)
    } else {
        request_workload_class(path).is_none_or(|workload| {
            model_satisfies_request_workload(model, workload, path, descriptors)
        })
    }
}

/// Select one matching descriptor that independently supports the endpoint.
pub(super) fn descriptor_for_request<'a>(
    model: &str,
    path: &str,
    descriptors: &'a [ServedModelDescriptor],
) -> Option<&'a ServedModelDescriptor> {
    descriptors.iter().find(|descriptor| {
        descriptor_matches_routable_name(descriptor, model)
            && model_satisfies_request(model, path, std::slice::from_ref(*descriptor))
    })
}

/// The one descriptor that serves `model` for the endpoint at `path`, or
/// `None` when none does or more than one could have: a node can advertise a
/// native embedding model and a chat plugin under one name, and a fact read
/// from the wrong one would be attributed to a model that never ran. Exact
/// `model_name` matches win over public-ID aliases, as in routing.
pub(super) fn unambiguous_descriptor_for_request<'a>(
    model: &str,
    path: &str,
    descriptors: &'a [ServedModelDescriptor],
) -> Option<&'a ServedModelDescriptor> {
    let serves = |descriptor: &&ServedModelDescriptor| {
        model_satisfies_request(model, path, std::slice::from_ref(*descriptor))
    };
    let exact = descriptors
        .iter()
        .filter(|descriptor| descriptor.identity.model_name == model)
        .filter(serves)
        .collect::<Vec<_>>();
    let candidates = if exact.is_empty() {
        descriptors
            .iter()
            .filter(|descriptor| descriptor_matches_routable_name(descriptor, model))
            .filter(serves)
            .collect()
    } else {
        exact
    };
    match candidates.as_slice() {
        [only] => Some(*only),
        _ => None,
    }
}

/// Infer binary upload requirements without interpreting file bytes as JSON.
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
    } else if is_binary_task_path(path) {
        crate::network::router::MediaRequirements {
            has_media: true,
            needs_audio: false,
            needs_vision: false,
        }
    } else {
        body.map_or_else(Default::default, crate::network::router::media_requirements)
    }
}

/// Multipart tasks can select an automatic model without a JSON request body.
pub(super) fn is_binary_task_path(path: &str) -> bool {
    matches!(
        path.split('?').next(),
        Some("/mesh/v1/documents/ocr" | "/mesh/v1/images/similarity" | "/mesh/v1/images/depth")
    )
}

/// Pick metadata from a descriptor that supports this request, not whichever
/// peer happened to gossip the model name first. Capacity sums the measured
/// throughput of every routable placement (this node included) so the draw is
/// proportional to serving capacity, not to the number of distinct names.
pub(super) async fn routing_candidates<'a>(
    node: &mesh::Node,
    models: &'a [String],
    path: &str,
    descriptors: &[ServedModelDescriptor],
) -> Vec<crate::network::router::RoutingCandidate<'a>> {
    let placements = node.placements_for_models(models).await;
    let folded = folded_public_aliases(node, models, descriptors).await;
    let mut candidates = Vec::with_capacity(models.len());
    for model in models
        .iter()
        .filter(|model| !folded.contains(model))
        .filter(|model| model_satisfies_request(model, path, descriptors))
    {
        let descriptor = descriptor_for_request(model, path, descriptors);
        let caps = descriptor.map_or_else(
            || super::routing_rank::capabilities_for_model(model, descriptors),
            |descriptor| descriptor.capabilities,
        );
        let placement = placements.get(model.as_str()).cloned().unwrap_or_default();
        let capacity = model_capacity(node, model, &placement).await;
        candidates.push(crate::network::router::RoutingCandidate {
            name: model,
            caps,
            parameter_count_b: descriptor
                .and_then(|descriptor| descriptor.metadata.as_ref())
                .and_then(|metadata| metadata.parameter_count_b),
            replicas: placement.count(),
            capacity,
        });
    }
    candidates
}

/// Sum of per-placement draw weights for `model`. The local placement weighs
/// this host's own serving samples only (what it advertises to peers), so
/// requests this node routed to other hosts never speed up or slow down its
/// own weight; a remote placement weighs this node's own samples against that
/// host when it has enough, else the host's gossiped throughput hint, else the
/// neutral cold weight. A model with no visible placement keeps one neutral
/// placement so it is still drawn.
pub(super) async fn model_capacity(
    node: &mesh::Node,
    model: &str,
    placement: &crate::mesh::ModelPlacements,
) -> f64 {
    use crate::network::router::placement_weight;
    let weight_of = |rank: Option<super::routing_rank::TargetThroughputRank>| {
        placement_weight(
            rank.map(|rank| rank.avg_tokens_per_second_milli as f64 / 1000.0),
            rank.map(|rank| rank.throughput_samples).unwrap_or(0),
        )
    };
    let mut capacity = 0.0;
    if placement.local {
        capacity += weight_of(super::routing_rank::local_serving_throughput_rank(
            node, model,
        ));
    }
    for peer in &placement.remote {
        capacity +=
            weight_of(super::routing_rank::remote_target_throughput_rank(node, model, *peer).await);
    }
    if placement.local || !placement.remote.is_empty() {
        capacity
    } else {
        placement_weight(None, 0)
    }
}

/// Filter each candidate against its own advertisement before routing decisions.
pub(super) async fn eligible_targets(
    node: &mesh::Node,
    model: &str,
    path: &str,
    candidates: &[InferenceTarget],
) -> Vec<InferenceTarget> {
    if request_workload_class(path).is_none() && !is_system_one_path(path) {
        return candidates.to_vec();
    }
    // Local inference targets are native serving ports. Plugin descriptors are
    // advertised separately for peer discovery and dispatched by the plugin path.
    let local = node.served_model_descriptors().await;
    let peers = node.peers().await;
    let supported = node.supported_model_channels(&peers).await;
    let peers = peers
        .into_iter()
        .map(|peer| (peer.id, peer))
        .collect::<std::collections::HashMap<_, _>>();
    candidates
        .iter()
        .filter(|target| {
            let descriptors = match target {
                InferenceTarget::Local(_) => local.as_slice(),
                InferenceTarget::Remote(peer_id) => {
                    let Some(peer) = peers.get(peer_id) else {
                        return false;
                    };
                    if mesh::model_requires_unsupported_channel(peer, model, &supported) {
                        return false;
                    }
                    peer.served_model_descriptors.as_slice()
                }
                InferenceTarget::None => return false,
            };
            model_satisfies_request(model, path, descriptors)
        })
        .cloned()
        .collect()
}

/// Keep the targets whose own descriptor serves the request's image or audio
/// input. The model was chosen on capabilities pooled across every host, so a
/// copy of it that cannot see the input must not take the request. When no
/// target's descriptor says so, the list is returned unchanged rather than
/// failing a request the model-level check already admitted.
pub(super) async fn media_capable_targets(
    node: &mesh::Node,
    model: &str,
    path: &str,
    body: Option<&serde_json::Value>,
    candidates: Vec<InferenceTarget>,
) -> Vec<InferenceTarget> {
    let media = request_media(path, body);
    if !media.requires_runtime_modality() {
        return candidates;
    }
    let local = node.served_model_descriptors().await;
    let state = node.state.lock().await;
    let capable = candidates
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
            descriptor_for_request(model, path, descriptors).is_some_and(|descriptor| {
                crate::network::router::model_satisfies_media_requirements(
                    &descriptor.capabilities,
                    &media,
                )
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    drop(state);
    if capable.is_empty() {
        candidates
    } else {
        capable
    }
}

/// Restrict peer IDs using the same per-target workload policy as local ingress.
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
