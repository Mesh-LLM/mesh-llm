//! Admission of the final JSON invocation before the virtual backend runs.
use super::*;
use crate::network::openai::response::prepared_dispatch::{PreparedSubdispatch, admit};

pub(super) fn prepare_invocation(
    request: serde_json::Value,
    candidates: Vec<VirtualModelCandidate>,
    response_adapter: proxy::ResponseAdapter,
) -> Result<(String, bool), String> {
    let requests_stream = request
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let invocation = VirtualModelInvocation {
        request,
        candidates,
        response_adapter: format!("{response_adapter:?}"),
    };
    serde_json::to_string(&invocation)
        .map(|encoded| (encoded, requests_stream))
        .map_err(|error| format!("failed to encode virtual model request: {error}"))
}

pub(super) async fn admit_invocation(
    node: &mesh::Node,
    mut stream: ClientStream,
    dispatch: PreparedSubdispatch<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<ClientStream, Box<proxy::RouteDispatchOutcome>> {
    if let Some(result) = admit(node, dispatch).await {
        let _ = stream.add_response_metadata(result.headers.clone());
        if let Some(status) = result.error_status() {
            let written = proxy::send_error_observed(
                stream,
                status,
                "OpenAI plugin policy rejected virtual model dispatch",
                route_observer,
            )
            .await;
            return Err(Box::new(if written.is_err() {
                proxy::RouteDispatchOutcome::Dropped("virtual_model_response_write_failed")
            } else if status == 403 {
                proxy::RouteDispatchOutcome::PolicyDenied
            } else {
                proxy::RouteDispatchOutcome::RequiredHookFailed
            }));
        }
    }
    Ok(stream)
}
