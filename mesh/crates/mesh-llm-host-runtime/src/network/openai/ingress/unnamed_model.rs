//! Requests that name no model: the generic fallback to the first available
//! target.
use super::*;

/// Route a request that names no model to the first available target.
///
/// Nothing names the model a receipt would be for, and the fallback takes
/// whatever target comes first, so a request whose trust policy requires
/// receipts stops here.
pub(super) async fn route_unnamed_model_request(
    tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if let Some(rejection) = trust_policy::unprovable(request, "a request that names no model") {
        return trust_routing::reject(tcp_stream, ctx, None, rejection, route_observer).await;
    }
    let target = first_available_target(ctx.targets);
    let mut tcp_stream = tcp_stream;
    if let Some(outcome) = crate::network::openai::exchange_admission::admit_selected_route(
        ctx.node,
        &mut tcp_stream,
        request,
        None,
        "mesh",
        crate::network::openai::exchange_admission::SelectedRouteTarget::Mesh(&target),
        1,
    )
    .await
    {
        return outcome;
    }
    proxy::route_to_target(
        ctx.node.clone(),
        tcp_stream,
        None,
        target,
        &request.raw,
        proxy::RouteTargetContext {
            request_id: request.request_id,
            response_adapter: request.response_adapter,
            route_observer,
            trust: request.trust,
        },
    )
    .await
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}
