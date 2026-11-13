//! Safe logging boundary for OpenAI request-reader failures.
//!
//! A malformed request becomes loggable only after complete, bounded headers
//! produced a canonical request ID and client path. Earlier failures write the
//! client error without inventing a request lifecycle or artifact.

use crate::network::openai::client_stream::ClientStream;

use super::request_parse::OpenAiRequestReadError;
use super::response::{send_400, send_400_observed, send_json_with_status_and_headers_observed};
use super::transport::RouteDispatchOutcome;
use crate::logging::OpenAiLifecycleAttachment;

fn attach_read_failure_logging(
    error: &OpenAiRequestReadError,
) -> Option<OpenAiLifecycleAttachment> {
    let context = error.context()?;
    let metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&context.client_path);
    Some(
        crate::logging_runtime_state()
            .map(|state| state.openai_ingress_attachment(context.request_id, metadata))
            .unwrap_or_else(OpenAiLifecycleAttachment::unowned),
    )
}

pub(crate) async fn send_read_failure(
    tcp_stream: ClientStream,
    error: &OpenAiRequestReadError,
) -> RouteDispatchOutcome {
    let mut lifecycle = attach_read_failure_logging(error);
    // Responses continuation needs its typed 404/validation errors. Other
    // protocols keep their own error envelopes (notably Anthropic Messages).
    let inference = error
        .context()
        .filter(|context| context.client_path.split('?').next() == Some("/v1/responses"))
        .and_then(|_| error.inference_error());
    let code = inference.map_or(400, |error| error.status().as_u16());
    let result = if let (Some(lifecycle), Some(inference)) = (lifecycle.as_ref(), inference) {
        send_json_with_status_and_headers_observed(
            tcp_stream,
            code,
            &serde_json::to_value(inference.body()).expect("InferenceError body serializes"),
            &[],
            lifecycle.route_observer(),
        )
        .await
    } else if let Some(lifecycle) = lifecycle.as_ref() {
        send_400_observed(tcp_stream, &error.to_string(), lifecycle.route_observer()).await
    } else {
        send_400(tcp_stream, &error.to_string()).await
    };
    let outcome = match result {
        Ok(()) => RouteDispatchOutcome::Responded(code),
        Err(_) => RouteDispatchOutcome::Dropped("response_write_failed"),
    };
    if let Some(lifecycle) = lifecycle.as_mut() {
        lifecycle.terminal(outcome.terminal_outcome());
    }
    outcome
}
