//! The request trust policy inside the attempt loop of the mesh proxy, which
//! tunnels every request to a peer: the policy is asked again as each host's
//! turn comes, and a route on which it let nothing through ends with the
//! policy's own answer.

use super::*;

/// Asks the request's trust policy about `host` again, as its turn comes: a
/// host that stopped qualifying since the plan was made is never contacted.
pub(super) async fn host_refused(
    node: &mesh::Node,
    request: &BufferedHttpRequest,
    model: Option<&str>,
    host: iroh::EndpointId,
    state: &mut MeshAttemptState,
) -> bool {
    let candidate = election::InferenceTarget::Remote(host);
    let refused =
        !super::super::trust_policy::permits_attempt(node, request, model, &candidate).await;
    state.refused_by_trust_policy |= refused;
    refused
}

/// Answers a request whose every host the trust policy refused when its
/// turn came: the terminal outcome of the request.
pub(super) async fn answer_refused(
    stream: ClientStream,
    request: &BufferedHttpRequest,
    plan: &MeshRequestPlan,
    route_observer: OpenAiRouteObserver<'_>,
) -> crate::logging::TerminalOutcome {
    let model = plan.effective_model.as_deref().unwrap_or_default();
    let rejection = request.trust.unsatisfied(model);
    super::super::trust_policy::answer(stream, rejection, route_observer)
        .await
        .terminal_outcome()
}

/// The terminal outcome of a request no host could be planned for under its
/// trust policy.
pub(super) fn unplanned_terminal_outcome() -> crate::logging::TerminalOutcome {
    let code = crate::network::request_trust::UNSATISFIED_CODE;
    crate::logging::TerminalOutcome::Rejected(Some(code.into()))
}

/// Ends a route that served nothing: refused by the trust policy when it is
/// why no host was contacted, exhausted otherwise. A cached automatic model
/// whose every host failed is forgotten, and the request counts as
/// unavailable.
pub(super) fn unserved(
    node: &mesh::Node,
    stream: ClientStream,
    plan: &MeshRequestPlan,
    state: &MeshAttemptState,
    affinity: &AffinityRouter,
) -> MeshRouteResult {
    let effective_model = plan.effective_model.as_deref();
    if state.last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
        if let Some(key) = plan.auto_session_key {
            tracing::debug!(
                "auto: all hosts failed for cached model, forgetting session {key:016x}"
            );
            affinity.forget_auto_model(key);
        }
    }
    node.record_routed_request(
        effective_model,
        state.attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
    if state.refused_by_trust_policy && state.attempts == 0 {
        MeshRouteResult::RefusedByTrustPolicy(stream)
    } else {
        MeshRouteResult::Exhausted(stream)
    }
}
