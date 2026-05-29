//! Auto-route model admission helpers.
//!
//! Explicit model routing keeps an availability-preserving fallback when every
//! target is cooling. Auto routing can be stricter before it chooses a model:
//! if another model has a healthy target that can fit the request context, it
//! should use that model instead of spending an agent turn on a target we just
//! proved unhealthy or too small.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity::AffinityRouter;
use crate::network::router;

fn has_routable_candidate(candidates: &[election::InferenceTarget]) -> bool {
    candidates
        .iter()
        .any(|target| !matches!(target, election::InferenceTarget::None))
}

async fn target_context_satisfies_request(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    target: &election::InferenceTarget,
) -> bool {
    let Some(required_tokens) = required_tokens else {
        return !matches!(target, election::InferenceTarget::None);
    };
    let context_length = match target {
        election::InferenceTarget::Local(_) => node.local_model_context_length(model).await,
        election::InferenceTarget::Remote(peer_id) => {
            node.peer_model_context_length(*peer_id, model).await
        }
        election::InferenceTarget::None => return false,
    };
    context_length
        .map(|context| context >= required_tokens)
        .unwrap_or(true)
}

async fn context_compatible_targets(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    candidates: &[election::InferenceTarget],
) -> Vec<election::InferenceTarget> {
    let mut compatible = Vec::new();
    for candidate in candidates {
        if target_context_satisfies_request(node, model, required_tokens, candidate).await {
            compatible.push(candidate.clone());
        }
    }
    compatible
}

pub(crate) async fn model_has_eligible_target(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    candidates: &[election::InferenceTarget],
    affinity: &AffinityRouter,
) -> bool {
    let context_compatible =
        context_compatible_targets(node, model, required_tokens, candidates).await;
    if !has_routable_candidate(&context_compatible) {
        return false;
    }
    has_routable_candidate(&affinity.route_strict_eligible_candidates(model, &context_compatible))
}

pub(crate) async fn model_has_eligible_remote_host(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
) -> bool {
    let targets: Vec<election::InferenceTarget> = node
        .hosts_for_model(model)
        .await
        .into_iter()
        .map(election::InferenceTarget::Remote)
        .collect();
    model_has_eligible_target(node, model, required_tokens, &targets, affinity).await
}

pub(crate) fn pool_for_ready_models<'a>(
    available: &[router::RoutingCandidate<'a>],
    ready_models: &[&str],
) -> Vec<router::RoutingCandidate<'a>> {
    let ready = available
        .iter()
        .filter(|candidate| ready_models.contains(&candidate.name))
        .cloned()
        .collect::<Vec<_>>();
    if ready.is_empty() {
        available.to_vec()
    } else {
        ready
    }
}

pub(crate) async fn ready_remote_models<'a>(
    node: &mesh::Node,
    required_tokens: Option<u32>,
    available: &[router::RoutingCandidate<'a>],
    affinity: &AffinityRouter,
) -> Vec<&'a str> {
    let mut ready_models = Vec::new();
    for candidate in available {
        if model_has_eligible_remote_host(node, candidate.name, required_tokens, affinity).await {
            ready_models.push(candidate.name);
        }
    }
    ready_models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_route_pool_prefers_ready_models_when_any_ready() {
        let caps = crate::models::ModelCapabilities::default();
        let available = vec![
            router::RoutingCandidate::unscored("cooling-model", caps),
            router::RoutingCandidate::unscored("ready-model", caps),
        ];

        let pool = pool_for_ready_models(&available, &["ready-model"]);

        assert_eq!(pool.len(), 1);
        assert_eq!(pool[0].name, "ready-model");
    }

    #[test]
    fn auto_route_pool_preserves_availability_when_none_ready() {
        let caps = crate::models::ModelCapabilities::default();
        let available = vec![
            router::RoutingCandidate::unscored("cooling-a", caps),
            router::RoutingCandidate::unscored("cooling-b", caps),
        ];

        let pool = pool_for_ready_models(&available, &[]);

        assert_eq!(pool.len(), available.len());
        assert_eq!(pool[0].name, "cooling-a");
        assert_eq!(pool[1].name, "cooling-b");
    }
}
