//! Selected-route plugin admission immediately before each backend attempt.
use super::{client_stream::ClientStream, transport};
use crate::mesh;
use serde_json::json;
use tokio::io::AsyncWriteExt;

pub(super) enum SelectedRouteTarget<'a> {
    Url(&'a str),
    MeshLabel(&'a str),
}

pub(super) async fn admit_selected_route(
    node: &mesh::Node,
    stream: &mut ClientStream,
    request: &transport::BufferedHttpRequest,
    model: Option<&str>,
    provider: &str,
    target: SelectedRouteTarget<'_>,
    attempt: usize,
) -> Option<transport::RouteDispatchOutcome> {
    let manager = node.plugin_manager().await?;
    if !manager.has_exchange_hooks().await {
        return None;
    }
    let event = match selected_route_event(request, model, provider, target, attempt) {
        Ok(event) => event?,
        Err(error) => {
            tracing::warn!(%error, "cannot observe effective OpenAI request entity");
            stream.record_exchange_outcome("internal_hook_failure");
            return Some(send_selected_route_denial(stream, 503).await);
        }
    };
    apply_selected_route_policy(&manager, request, stream, event).await
}

async fn apply_selected_route_policy(
    manager: &crate::plugin::PluginManager,
    request: &transport::BufferedHttpRequest,
    stream: &mut ClientStream,
    event: serde_json::Value,
) -> Option<transport::RouteDispatchOutcome> {
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

fn selected_route_event(
    request: &transport::BufferedHttpRequest,
    model: Option<&str>,
    provider: &str,
    target: SelectedRouteTarget<'_>,
    attempt: usize,
) -> anyhow::Result<Option<serde_json::Value>> {
    let Some(endpoint) = exchange_endpoint(&request.client_path) else {
        return Ok(None);
    };
    let body = request.effective_http_entity()?;
    let mut event = crate::plugin::request_event(
        request.request_id.as_uuid().to_string(),
        endpoint,
        &request.method,
        &request.client_path,
        &body,
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
    Ok(Some(event))
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

fn public_target(target: SelectedRouteTarget<'_>) -> String {
    let target = match target {
        SelectedRouteTarget::MeshLabel(label) => return label.to_owned(),
        SelectedRouteTarget::Url(url) => url,
    };
    let Ok(url) = reqwest::Url::parse(target) else {
        return "<invalid-url>".to_owned();
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
            public_target(SelectedRouteTarget::Url(
                "https://operator:secret@example.com/v1?api_key=secret#private"
            )),
            "https://example.com"
        );
        assert_eq!(
            public_target(SelectedRouteTarget::MeshLabel("local-native")),
            "local-native"
        );
    }

    #[tokio::test]
    async fn selected_event_redacts_malformed_urls_and_preserves_trusted_mesh_labels() {
        let (mut reader, mut writer) = tokio::io::duplex(1024);
        writer.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}").await.unwrap();
        let request = transport::read_http_request(&mut reader).await.unwrap();
        for url in [
            "https://operator:password@example.com:bad/private?api_key=secret#hidden",
            "https://operator:password@[invalid/private?api_key=secret#hidden",
        ] {
            let event = selected_route_event(
                &request,
                Some("model"),
                "plugin",
                SelectedRouteTarget::Url(url),
                1,
            )
            .unwrap()
            .unwrap();
            assert_eq!(event["phase"], "backend_selected");
            assert_eq!(event["target"], "<invalid-url>");
            let projected = serde_json::to_string(&event).unwrap();
            for sensitive in [
                "operator", "password", "private", "api_key", "secret", "hidden",
            ] {
                assert!(
                    !projected.contains(sensitive),
                    "route event leaked {sensitive}"
                );
            }
        }
        for label in ["Local(3131)", "Peer(abc)"] {
            let event = selected_route_event(
                &request,
                None,
                "mesh",
                SelectedRouteTarget::MeshLabel(label),
                2,
            )
            .unwrap()
            .unwrap();
            assert_eq!(event["target"], label);
        }
    }
}

#[cfg(test)]
#[path = "exchange_entity_tests.rs"]
mod entity_tests;
