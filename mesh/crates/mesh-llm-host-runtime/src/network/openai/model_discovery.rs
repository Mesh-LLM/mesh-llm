//! HTTP projections of the shared mesh-scoped discovery service.

use super::client_stream::ClientStream;
use super::request_parse::{BufferedHttpRequest, header_values_from_raw};
use super::response::{
    model_error_response, models_response, send_json_with_status_and_headers_observed,
};
use super::transport::RouteDispatchOutcome;
use crate::logging::OpenAiRouteObserver;
use crate::{mesh, models::discovery};

/// These reads are handled locally rather than routed as model inference.
pub(crate) fn is_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    matches!(path, "/mesh/v1/models" | "/mesh/v1/models/lookup")
        || (method == "GET"
            && (matches!(path, "/v1/models" | "/models")
                || path.starts_with("/v1/models/")
                || path.starts_with("/models/")))
}

pub(crate) async fn respond(
    stream: ClientStream,
    node: &mesh::Node,
    request: &BufferedHttpRequest,
    observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    let (status, body) = if let Some(response) =
        discovery::http_response(node, &request.method, &request.path).await
    {
        response
    } else {
        compatibility_response(node, request).await
    };
    let headers = if status == 405 {
        vec![("Allow", "GET".to_string())]
    } else {
        Vec::new()
    };
    match send_json_with_status_and_headers_observed(stream, status, &body, &headers, observer)
        .await
    {
        Ok(()) => RouteDispatchOutcome::Responded(status),
        Err(_) => RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

async fn compatibility_response(
    node: &mesh::Node,
    request: &BufferedHttpRequest,
) -> (u16, serde_json::Value) {
    let anthropic = match header_values_from_raw(&request.raw, "anthropic-version") {
        Ok(values) => !values.is_empty(),
        Err(()) => return model_error_response(400, "invalid anthropic-version header", false),
    };
    match discovery::collect(node).await {
        Ok(models) => models_response(&models, &request.path, anthropic),
        Err(error) => {
            tracing::warn!(%error, "model discovery failed");
            model_error_response(503, "model discovery temporarily unavailable", anthropic)
        }
    }
}

#[cfg(test)]
mod tests;
