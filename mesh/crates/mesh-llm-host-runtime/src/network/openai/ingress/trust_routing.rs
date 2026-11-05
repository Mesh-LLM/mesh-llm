//! Routing a request under its trust policy (`network/request_trust.rs`):
//! the candidates the policy rules out join the request's exclusions, which
//! routing already enforces and fails closed on, and leave the candidate
//! pool election picks from.
use super::*;
use crate::network::request_trust::{self, Unsatisfied};

/// Where a request for a model asks to run: its routing headers with its
/// trust policy applied.
pub(super) struct RoutingAsk {
    pub(super) target: Option<iroh::EndpointId>,
    pub(super) excluded: Vec<iroh::EndpointId>,
    /// The candidate pool without the candidates the policy rules out;
    /// `None` when it rules out none.
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
    let (target, mut excluded) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => {
            let write = proxy::send_400_observed(tcp_stream, &message, route_observer).await;
            return Ask::Answered(response_outcome(400, write));
        }
    };
    match restrict(ctx, request, model_name, &mut excluded).await {
        Ok(trusted_targets) => Ask::Route(
            tcp_stream,
            RoutingAsk {
                target,
                excluded,
                trusted_targets,
            },
        ),
        Err(rejection) => {
            Ask::Answered(reject(tcp_stream, ctx, model_name, rejection, route_observer).await)
        }
    }
}

/// Applies the trust policy of `request` for `model_name`: extends
/// `excluded` with the candidates it rules out and returns the candidate
/// pool without them, or `None` when it rules out none. A request a peer
/// sent this node to serve is not routed here, so its policy is the
/// sender's business.
async fn restrict(
    ctx: &IngressRouteContext<'_>,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    excluded: &mut Vec<iroh::EndpointId>,
) -> Result<Option<election::ModelTargets>, Unsatisfied> {
    if ctx.requested_by_node_id.is_some() {
        return Ok(None);
    }
    let disqualified =
        request_trust::disqualified(ctx.node, &request.raw, &request.client_path, model_name)
            .await?;
    let Some(disqualified) = disqualified else {
        return Ok(None);
    };
    if disqualified.this_node {
        excluded.push(ctx.node.id());
    }
    excluded.extend(disqualified.peers.iter().copied());
    let mut targets = ctx.targets.clone();
    if let Some(candidates) = targets.targets.get_mut(model_name) {
        candidates.retain(|target| match target {
            election::InferenceTarget::Local(_) => !disqualified.this_node,
            election::InferenceTarget::Remote(peer) => !disqualified.peers.contains(peer),
            election::InferenceTarget::None => true,
        });
    }
    Ok(Some(targets))
}

/// Answers a request its trust policy rejects.
async fn reject(
    tcp_stream: ClientStream,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    rejection: Unsatisfied,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let message = match rejection {
        Unsatisfied::Invalid(message) => {
            record_pre_routing_rejection(ctx, Some(model_name), 400).await;
            return response_outcome(
                400,
                proxy::send_400_observed(tcp_stream, &message, route_observer).await,
            );
        }
        Unsatisfied::NoCandidate(message) => message,
    };
    let status = 409;
    record_pre_routing_rejection(ctx, Some(model_name), status).await;
    let code = request_trust::UNSATISFIED_CODE;
    let body = serde_json::json!({
        "error": {"message": message, "type": "invalid_request_error", "code": code}
    });
    response_outcome(
        status,
        proxy::send_json_with_status_and_headers_observed(
            tcp_stream,
            status,
            &body,
            &[],
            route_observer,
        )
        .await,
    )
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
