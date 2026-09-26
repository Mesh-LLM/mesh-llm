mod body_reader;
mod cache_cost;
mod cancellation;
mod chat_stream;
mod common;
mod dispatch;
mod external_endpoint;
mod json_adaptation;
#[cfg(feature = "payments")]
mod model_prices;
mod models;
mod pipeline;
mod pipeline_adapter;
mod probe;
mod relay;
mod routing;
mod send;
mod stream_translation;

pub(super) use cache_cost::CacheCostObservation;
pub(crate) use common::PeerCapsuleIdSink;
pub(super) use common::{
    ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    attempt_outcome_for_result, completion_tokens_for_result, parse_token_usage_from_json_body,
    request_outcome_for_status, request_service_for_target, route_attempt_result_label,
    target_health_outcome_for_attempt,
};
pub(super) use external_endpoint::route_http_endpoint_attempt;
pub(crate) use models::send_models_list_with_descriptors;
pub(super) use pipeline::pipeline_proxy_local;
pub use pipeline::{PipelineCapsuleNonce, PipelineProxyResult};
pub(super) use routing::{route_local_attempt, route_remote_attempt};
#[cfg(feature = "payments")]
pub(crate) use send::send_error;
pub(crate) use send::{
    append_safe_header, is_valid_header_name, send_400, send_400_observed, send_409_observed,
    send_503, send_503_observed, send_error_observed, send_json_ok_with_headers,
    send_json_with_status_and_headers_observed,
};

#[cfg(feature = "payments")]
pub(crate) mod paid;
#[cfg(feature = "payments")]
mod paid_events;

#[cfg(feature = "payments")]
pub(crate) mod payment_recovery;
