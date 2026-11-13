//! Workload-compatible routing readiness, retaining local startup fallback.

use super::{model_has_eligible_target, model_is_locally_served};
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::Node;
use crate::network::affinity::AffinityRouter;
use crate::network::openai::{request_parse::BufferedHttpRequest, workload_routing};
use crate::network::request_trust;
use crate::network::router::RoutingCandidate;

/// Measure readiness without discarding temporarily unavailable models.
pub(crate) async fn ready_models<'a>(
    node: &Node,
    targets: Option<&ModelTargets>,
    request: &BufferedHttpRequest,
    available: &[RoutingCandidate<'a>],
    required_tokens: Option<u32>,
    affinity: &AffinityRouter,
) -> (Vec<RoutingCandidate<'a>>, Vec<&'a str>) {
    let mut admitted = Vec::new();
    let mut ready = Vec::new();
    for candidate in available {
        let model = candidate.name;
        if node.model_unavailable_due_to_channels(model).await {
            continue;
        }
        // Under a trust policy a model is ready only through the candidates
        // the policy permits, so automatic selection never picks one that
        // could then be served only by a candidate that does not qualify.
        let trust = request.trust;
        let path = &request.client_path;
        let candidates = match targets {
            Some(targets) => {
                workload_routing::ingress_candidates(node, model, path, targets, trust).await
            }
            None => {
                let hosts = node
                    .hosts_for_model(model)
                    .await
                    .into_iter()
                    .map(InferenceTarget::Remote)
                    .collect::<Vec<_>>();
                let eligible = workload_routing::eligible_targets(node, model, path, &hosts).await;
                request_trust::permitted(node, trust, model, path, eligible).await
            }
        };
        let locally_starting = targets.is_some()
            && !candidates
                .iter()
                .any(|target| !matches!(target, InferenceTarget::None))
            && model_is_locally_served(node, model).await
            && request_trust::permits_local_runtime(node, trust, model);
        admitted.push(candidate.clone());
        if locally_starting
            || model_has_eligible_target(
                node,
                model,
                required_tokens,
                &request.client_path,
                &candidates,
                affinity,
            )
            .await
        {
            ready.push(model);
        }
    }
    (admitted, ready)
}
