//! Safe logging boundary for OpenAI request-reader failures.
//!
//! A malformed request becomes loggable only after complete, bounded headers
//! produced a canonical request ID and client path. Earlier failures write the
//! client error without inventing a request lifecycle or artifact.

use crate::network::openai::client_stream::ClientStream;

use super::request_parse::OpenAiRequestReadError;
use super::response::{send_read_failure_error, send_read_failure_error_observed};
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
    let status = error.status();
    let message = error.to_string();
    let result = if let Some(lifecycle) = lifecycle.as_ref() {
        send_read_failure_error_observed(
            tcp_stream,
            status.status,
            &message,
            status.code.as_deref(),
            lifecycle.route_observer(),
        )
        .await
    } else {
        send_read_failure_error(tcp_stream, status.status, &message, status.code.as_deref()).await
    };
    let outcome = match result {
        Ok(()) => RouteDispatchOutcome::Responded(status.status),
        Err(_) => RouteDispatchOutcome::Dropped("response_write_failed"),
    };
    if let Some(lifecycle) = lifecycle.as_mut() {
        lifecycle.terminal(outcome.terminal_outcome());
    }
    outcome
}
