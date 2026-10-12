pub(crate) mod accept;
pub(crate) mod auto_route;
pub(crate) mod automatic;
pub(crate) mod client_stream;
mod forwarded_request;
pub(crate) mod ingress;
pub(crate) mod moa_gateway;
mod model_names;
mod parse_failure;
mod request_normalize;
pub(crate) mod request_parse;
mod response;
pub(crate) use response::send_503;
#[cfg(feature = "payments")]
pub(crate) use response::send_error;
pub(crate) mod response_adapter;
mod response_quality;
mod routing_rank;
pub(crate) mod runtime_events;
mod tool_call_ids;
pub(crate) mod transport;

mod payment_routing;

#[cfg(feature = "payments")]
pub(crate) use response::payment_recovery;

mod workload_routing;

#[cfg(all(test, feature = "payments"))]
pub(crate) use response::metered_payer::start as test_metered_start;

#[cfg(all(test, feature = "payments"))]
pub(crate) async fn test_payment_rank(
    node: &crate::mesh::Node,
    ordered: Vec<crate::inference::election::InferenceTarget>,
) -> (
    Result<bool, &'static str>,
    Vec<crate::inference::election::InferenceTarget>,
    usize,
) {
    let mut candidates = routing_rank::RankedCandidates {
        equivalent_prefix: ordered.len(),
        ordered,
    };
    let result = payment_routing::rank(node, "test", 1, 8, &mut candidates, None).await;
    (result, candidates.ordered, candidates.equivalent_prefix)
}
