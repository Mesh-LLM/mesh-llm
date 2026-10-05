//! Selected-route plugin admission immediately before each backend attempt.
use super::{client_stream::ClientStream, transport};
use crate::mesh;
use serde_json::json;
use tokio::io::AsyncWriteExt;

pub(super) async fn admit_selected_route(
    node: &mesh::Node,
    stream: &mut ClientStream,
    request: &transport::BufferedHttpRequest,
    model: Option<&str>,
    provider: &str,
    target: &str,
    attempt: usize,
) -> Option<transport::RouteDispatchOutcome> {
    let manager = node.plugin_manager().await?;
    if !manager.has_exchange_hooks().await {
        return None;
    }
    let endpoint = exchange_endpoint(&request.client_path)?;
    let body = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|end| &request.raw[end + 4..])
        .unwrap_or_default();
    let mut event = crate::plugin::request_event(
        request.request_id.as_uuid().to_string(),
        endpoint,
        &request.method,
        &request.client_path,
        body,
        Default::default(),
        false,
    );
    event["phase"] = json!("backend_selected");
    event["observation_point"] = json!("backend_dispatch");
    event["model"] = json!(model);
    event["provider"] = json!(provider);
    event["target"] = json!(public_target(target));
    event["attempt"] = json!(attempt);
    event["effective_request_wire_digest"] = event["request_wire_digest"].take();
    event.as_object_mut().unwrap().remove("request_wire_digest");
    let result = match &request.exchange_observation_id {
        Some(id) => manager.selected_exchange_phase(id, event).await,
        None => manager.exchange_phase(event).await,
    };
    if let Err(error) = stream.add_response_metadata(result.headers.clone()) {
        tracing::warn!(%error, "plugin response metadata arrived after headers committed");
    }
    let status = result.error_status()?;
    Some(send_selected_route_denial(stream, status).await)
}

async fn send_selected_route_denial(
    stream: &mut ClientStream,
    status: u16,
) -> transport::RouteDispatchOutcome {
    let body = json!({"error":{"message":"OpenAI plugin admission rejected the selected route",
        "type": if status == 403 { "permission_error" } else { "service_unavailable" },
        "code": if status == 403 { "plugin_policy_denied" } else { "plugin_hook_unavailable" }}})
    .to_string();
    let bytes = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        if status == 403 {
            "Forbidden"
        } else {
            "Service Unavailable"
        },
        body.len()
    );
    match stream.write_all(bytes.as_bytes()).await {
        Ok(()) if status == 403 => transport::RouteDispatchOutcome::PolicyDenied,
        Ok(()) => transport::RouteDispatchOutcome::RequiredHookFailed,
        Err(_) => transport::RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

fn public_target(target: &str) -> String {
    let Ok(url) = reqwest::Url::parse(target) else {
        return target.to_owned();
    };
    url.origin().ascii_serialization()
}

fn exchange_endpoint(path: &str) -> Option<&'static str> {
    match path.split('?').next() {
        Some("/v1/chat/completions") => Some("chat_completions"),
        Some("/v1/completions") => Some("completions"),
        Some("/v1/responses") => Some("responses"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_metadata_never_exposes_url_credentials() {
        assert_eq!(
            public_target("https://operator:secret@example.com/v1?api_key=secret#private"),
            "https://example.com"
        );
        assert_eq!(public_target("local-native"), "local-native");
    }
}
