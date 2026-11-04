//! The request trust policy inside the attempt loop of a model route: the
//! policy is asked again as each candidate's turn comes, and gives its own
//! answer when it is why nothing was contacted.

use super::*;

/// Asks the request's trust policy about `target` again, as its turn comes:
/// a candidate that stopped qualifying since the pool was built is never
/// contacted, on a first attempt, a retry or a failover.
pub(super) async fn trust_policy_refuses(
    node: &mesh::Node,
    request: &BufferedHttpRequest,
    model: &str,
    target: &election::InferenceTarget,
    state: &mut RouteModelState,
) -> bool {
    let refused =
        !super::super::super::trust_policy::permits_attempt(node, request, Some(model), target)
            .await;
    state.refused_by_trust_policy |= refused;
    refused
}

/// Answers a request no candidate served: with the trust policy's own
/// answer when it is why nothing was contacted, else as an exhausted route.
pub(super) async fn finish_unserved_route_model_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &BufferedHttpRequest,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    if state.refused_by_trust_policy && state.attempts == 0 {
        record_route_model_unavailable(node, model, 0);
        let rejection = ModelRouteRejection::TrustPolicy(request.trust.unsatisfied(model));
        return rejection.respond(tcp_stream, route_observer).await;
    }
    exhaustion::finish_exhausted_route_model_request(
        node,
        tcp_stream,
        model,
        total_targets,
        state,
        route_observer,
    )
    .await
}
