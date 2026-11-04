//! The request trust policy (`network/request_trust.rs`) at the OpenAI
//! ingress: where a request's policy is decided, and the one answer every
//! path gives a request it cannot serve under it.
//!
//! Each place that reads a client's request off a socket calls [`decide`]
//! before it dispatches anything: the API proxy for buffered requests and
//! for realtime sessions, and the mesh proxy that serves while this node's
//! own model loads. Everything after reads the decision off the request.

use super::client_stream::ClientStream;
use super::transport as proxy;
use crate::inference::election::InferenceTarget;
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::request_trust::{self, RequestTrust, Unsatisfied};

/// Decides the trust policy of `request` and records it on the request. A
/// request a peer assigned to this node is executed here and never routed,
/// so its policy was the sender's to apply and none is read from it.
pub(super) async fn decide(
    node: &mesh::Node,
    request: &mut proxy::BufferedHttpRequest,
    assigned_by_peer: bool,
) -> Result<(), Unsatisfied> {
    request.trust = if assigned_by_peer {
        RequestTrust::UNRESTRICTED
    } else {
        request_trust::decide(node, &request.raw).await?
    };
    Ok(())
}

/// Why `request` cannot run on its endpoint under its policy, when it
/// cannot: receipts cover chat completions only.
pub(super) fn uncovered(request: &proxy::BufferedHttpRequest) -> Option<Unsatisfied> {
    (request.trust.requires_receipts() && !request_trust::covers(&request.client_path))
        .then(request_trust::uncovered)
}

/// Whether the policy of `request` restricts who may serve it. A route that
/// asks the policy about no candidate (a plugin, a pipeline) is not taken
/// for such a request.
pub(super) fn restricts(request: &proxy::BufferedHttpRequest) -> bool {
    request.trust.requires_receipts()
}

/// Why a route that cannot prove the `receipts` tier (`what`: a plugin
/// endpoint, a virtual model, a pipeline) refuses `request`, when its
/// policy requires receipts.
pub(super) fn unprovable(request: &proxy::BufferedHttpRequest, what: &str) -> Option<Unsatisfied> {
    unprovable_under(request.trust, what)
}

/// [`unprovable`] for a route that is handed the policy without the request.
pub(super) fn unprovable_under(trust: RequestTrust, what: &str) -> Option<Unsatisfied> {
    trust
        .requires_receipts()
        .then(|| RequestTrust::unprovable(what))
}

/// Why nothing may serve `request` for `model`, when it runs under a policy:
/// the answer of a route left with no candidate that qualifies, whatever
/// else it would have said (still loading, excluded, not found).
pub(super) fn no_candidate(
    request: &proxy::BufferedHttpRequest,
    model: &str,
) -> Option<Unsatisfied> {
    restricts(request).then(|| request.trust.unsatisfied(model))
}

/// Whether `target` may serve `request` for `model` under its policy, as
/// the target's advertisement stands now. Asked before every attempt.
pub(super) async fn permits_attempt(
    node: &mesh::Node,
    request: &proxy::BufferedHttpRequest,
    model: Option<&str>,
    target: &InferenceTarget,
) -> bool {
    match model {
        Some(model) => {
            request_trust::permits(node, request.trust, model, &request.client_path, target).await
        }
        // nothing names the model a receipt would have to be for
        None => !request.trust.requires_receipts(),
    }
}

/// The peers of `hosts` that may serve `request` for `model` under its
/// policy, or why none may.
pub(super) async fn permitted_hosts(
    node: &mesh::Node,
    request: &proxy::BufferedHttpRequest,
    model: Option<&str>,
    hosts: Vec<iroh::EndpointId>,
) -> Result<Vec<iroh::EndpointId>, Unsatisfied> {
    if !request.trust.requires_receipts() {
        return Ok(hosts);
    }
    let Some(model) = model else {
        return Err(RequestTrust::unprovable("a request that names no model"));
    };
    let path = &request.client_path;
    let hosts = request_trust::permitted_hosts(node, request.trust, model, path, hosts).await;
    if hosts.is_empty() {
        return Err(request.trust.unsatisfied(model));
    }
    Ok(hosts)
}

/// The stream of a realtime session request its policy lets start, `None`
/// once a request it does not was answered. `decided` is what deciding the
/// policy came to. A live session has no receipts, and no lifecycle to end.
pub(super) async fn admit_session(
    stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    decided: Result<(), Unsatisfied>,
) -> Option<ClientStream> {
    match decided.err().or_else(|| uncovered(request)) {
        Some(rejection) => {
            let _ = answer(stream, rejection, OpenAiRouteObserver::default()).await;
            None
        }
        None => Some(stream),
    }
}

/// Answers a request the mesh proxy read and its trust policy rejects, ends
/// the request's lifecycle and releases what the request holds.
pub(super) async fn reject_and_finish(
    node: &mesh::Node,
    stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    rejection: Unsatisfied,
    mut lifecycle: crate::logging::OpenAiLifecycleAttachment,
) {
    let outcome = answer(stream, rejection, lifecycle.route_observer()).await;
    lifecycle.terminal(outcome.terminal_outcome());
    proxy::release_request_objects(node, &request.request_object_request_ids).await;
}

/// Answers a request its trust policy rejects: 400 for a policy this node
/// cannot read, 409 with the `trust_policy_unsatisfied` code for one no
/// candidate satisfies.
pub(super) async fn answer(
    stream: ClientStream,
    rejection: Unsatisfied,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let (status, written) = match rejection {
        Unsatisfied::Invalid(message) => (
            400,
            proxy::send_400_observed(stream, &message, route_observer).await,
        ),
        Unsatisfied::NoCandidate(message) => {
            let body = serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": request_trust::UNSATISFIED_CODE,
                }
            });
            (
                409,
                proxy::send_json_with_status_and_headers_observed(
                    stream,
                    409,
                    &body,
                    &[],
                    route_observer,
                )
                .await,
            )
        }
    };
    match written {
        Ok(()) => proxy::RouteDispatchOutcome::Responded(status),
        Err(_) => proxy::RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}
