//! Routing a request for a model under its trust policy
//! (`network/request_trust.rs`). The policy was decided when the request
//! was read (`network/openai/trust_policy.rs`); here it shapes the candidate
//! pool election picks from and fails the request closed when nothing
//! qualifies. This is the early answer, not the enforcement: every pool and
//! every attempt after it asks the policy again
//! (`workload_routing::ingress_candidates`, `resolve_remote_mesh_route`,
//! the attempt loop of `route_model_request`), so a candidate that shows up
//! later, or stops qualifying, is still refused.
use super::*;
use crate::network::openai::trust_policy;
use crate::network::request_trust::{self, Unsatisfied};

/// Where a request for a model asks to run: its routing headers with its
/// trust policy applied.
pub(super) struct RoutingAsk {
    pub(super) target: Option<iroh::EndpointId>,
    pub(super) excluded: Vec<iroh::EndpointId>,
    /// The candidate pool holding only the candidates the policy permits;
    /// `None` when the request runs under no policy.
    pub(super) trusted_targets: Option<election::ModelTargets>,
}

/// What [`routing_ask`] leaves the caller.
pub(super) enum Ask {
    Route(ClientStream, RoutingAsk),
    /// The request was answered: malformed headers, or a policy no
    /// candidate satisfies.
    Answered(proxy::RouteDispatchOutcome),
}

/// Parses the routing headers of a request for `model_name` and applies its
/// trust policy, answering the request when either rules it out.
pub(super) async fn routing_ask(
    tcp_stream: ClientStream,
    ctx: &IngressRouteContext<'_>,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> Ask {
    let parsed = parse_mesh_routing_headers(request)
        .and_then(|routing| twin_bracket_id(request).map(|_| routing));
    let (target, excluded) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => {
            let write = proxy::send_400_observed(tcp_stream, &message, route_observer).await;
            return Ask::Answered(response_outcome(400, write));
        }
    };
    match restrict(ctx, request, model_name, target, &excluded).await {
        Ok(trusted_targets) => Ask::Route(
            tcp_stream,
            RoutingAsk {
                target,
                excluded,
                trusted_targets,
            },
        ),
        Err(rejection) => {
            let model = Some(model_name);
            Ask::Answered(reject(tcp_stream, ctx, model, rejection, route_observer).await)
        }
    }
}

/// Applies the trust policy of `request` for `model_name` to the candidates
/// that serve it now: the candidate pool holding only those it permits, or
/// `None` when the request runs under no policy. An error when nothing the
/// request may still reach (its `x-mesh-target` and `x-mesh-exclude`
/// applied) qualifies.
async fn restrict(
    ctx: &IngressRouteContext<'_>,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    target: Option<iroh::EndpointId>,
    excluded: &[iroh::EndpointId],
) -> Result<Option<election::ModelTargets>, Unsatisfied> {
    let trust = request.trust;
    if !trust.requires_receipts() {
        return Ok(None);
    }
    if let Some(uncovered) = trust_policy::uncovered(request) {
        return Err(uncovered);
    }
    let path = &request.client_path;
    let mut targets = ctx.targets.clone();
    let mut local_qualifies = false;
    if let Some(candidates) = targets.targets.get_mut(model_name) {
        let elected = std::mem::take(candidates);
        *candidates = request_trust::permitted(ctx.node, trust, model_name, path, elected).await;
        local_qualifies = candidates
            .iter()
            .any(|candidate| matches!(candidate, election::InferenceTarget::Local(_)));
    }
    // The peers that qualify now, whether or not the election snapshot
    // lists them: routing reads them from mesh state too.
    let hosts = ctx.node.hosts_for_model(model_name).await;
    let mut peers = request_trust::permitted_hosts(ctx.node, trust, model_name, path, hosts).await;
    peers.retain(|peer| !excluded.contains(peer));
    let self_id = ctx.node.id();
    let reachable = match target {
        Some(pinned) if pinned == self_id => local_qualifies,
        Some(pinned) => peers.contains(&pinned),
        None => (local_qualifies && !excluded.contains(&self_id)) || !peers.is_empty(),
    };
    if !reachable {
        return Err(trust.unsatisfied(model_name));
    }
    Ok(Some(targets))
}

/// Answers a request its trust policy rejects. A policy no candidate
/// satisfies counts as a pre-routing rejection; one this node cannot read
/// is the client's malformed request, like any other 400.
pub(super) async fn reject(
    tcp_stream: ClientStream,
    ctx: &IngressRouteContext<'_>,
    model_name: Option<&str>,
    rejection: Unsatisfied,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if matches!(rejection, Unsatisfied::NoCandidate(_)) {
        record_pre_routing_rejection(ctx, model_name, 409).await;
    }
    trust_policy::answer(tcp_stream, rejection, route_observer).await
}

/// Answers a request its trust policy rejects before any route was chosen,
/// and releases what the request holds.
pub(super) async fn reject_unrouted(
    tcp_stream: ClientStream,
    ctx: &IngressRouteContext<'_>,
    request: &proxy::BufferedHttpRequest,
    rejection: Unsatisfied,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let model = request.model_name.as_deref();
    let outcome = reject(tcp_stream, ctx, model, rejection, route_observer).await;
    proxy::release_request_objects(ctx.node, &request.request_object_request_ids).await;
    outcome
}

/// Dispatches a request to the plugin endpoint that serves `model_name`.
/// Every plugin dispatch of ingress comes through here, and no plugin
/// proves a trust tier: a request whose policy requires receipts is
/// refused, whichever fallback led to the plugin.
pub(super) async fn try_route_plugin_model(
    ctx: &IngressRouteContext<'_>,
    tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if let Some(rejection) = trust_policy::unprovable(request, "a plugin endpoint") {
        return reject(tcp_stream, ctx, Some(model_name), rejection, route_observer).await;
    }
    route_plugin_model(ctx, tcp_stream, request, model_name, route_observer).await
}

impl<'a> IngressRouteContext<'a> {
    /// This context with `targets` as the candidate pool.
    pub(super) fn with_targets<'b>(
        &self,
        targets: &'b election::ModelTargets,
    ) -> IngressRouteContext<'b>
    where
        'a: 'b,
    {
        IngressRouteContext {
            targets,
            requested_by_node_id: self.requested_by_node_id.clone(),
            ..*self
        }
    }
}

#[cfg(test)]
#[path = "../ingress_tests/trust_routing.rs"]
mod tests;
