//! Selected-route plugin admission immediately before each backend attempt.
use super::workload_routing::unambiguous_descriptor_for_request;
use super::{client_stream::ClientStream, transport};
use crate::inference::election::InferenceTarget;
use crate::mesh;
use serde_json::json;
use tokio::io::AsyncWriteExt;

pub(super) enum SelectedRouteTarget<'a> {
    Url(&'a str),
    /// A mesh election target. A remote peer additionally publishes its
    /// endpoint id as `target_peer_id`, the gateway-side counterpart of the
    /// serving-side `requested_by_node_id` (same lowercase-hex encoding), so a
    /// plugin can address the selected peer without parsing the `target` label.
    Mesh(&'a InferenceTarget),
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
    let model_identity = match (&target, model) {
        (SelectedRouteTarget::Mesh(target), Some(model)) => {
            selected_model_identity(node, target, model, &request.client_path).await
        }
        _ => None,
    };
    let event = match selected_route_event(
        request,
        model,
        provider,
        target,
        model_identity.as_ref(),
        attempt,
    ) {
        Ok(event) => event?,
        Err(error) => {
            tracing::warn!(%error, "cannot observe effective OpenAI request entity");
            stream.record_exchange_outcome("internal_hook_failure");
            return Some(send_selected_route_denial(stream, 503).await);
        }
    };
    apply_selected_route_policy(&manager, request, stream, event).await
}

/// The identity of `model` on the selected target, as this node knows it: its
/// own descriptor for a local target, the peer's gossiped descriptor for a
/// remote one. A peer descriptor never carries `weights_digest` (gossip strips
/// it; see `protocol/convert.rs`), so a remote target yields at most its
/// canonical ref here. Matches `model` under every name routing accepts for
/// it (the descriptor's `model_name`, then its public model ID) and only the
/// descriptor that serves the endpoint at `path`; with none or several
/// candidates it reports nothing. For a local target the weights digest is
/// kept only if it belongs to the selected runtime instance.
pub(super) async fn selected_model_identity(
    node: &mesh::Node,
    target: &InferenceTarget,
    model: &str,
    path: &str,
) -> Option<mesh::ServedModelIdentity> {
    let descriptors = match target {
        InferenceTarget::Local(_) => node.served_model_descriptors().await,
        InferenceTarget::Remote(peer_id) => {
            node.peers()
                .await
                .into_iter()
                .find(|peer| peer.id == *peer_id)?
                .served_model_descriptors
        }
        InferenceTarget::None => return None,
    };
    let mut identity = unambiguous_descriptor_for_request(model, path, &descriptors)?
        .identity
        .clone();
    if let InferenceTarget::Local(port) = target {
        node.scope_weights_digest_to_port(&mut identity, *port)
            .await;
    }
    Some(identity)
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
    model_identity: Option<&mesh::ServedModelIdentity>,
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
    event["target"] = json!(public_target(&target));
    if let Some(peer) = target_peer_id(&target) {
        event["target_peer_id"] = json!(peer);
    }
    if let Some(identity) = model_identity {
        if let Some(canonical_ref) = &identity.canonical_ref {
            event["model_canonical_ref"] = json!(canonical_ref);
        }
        if let Some(weights_digest) = &identity.weights_digest {
            event["weights_digest"] = json!(weights_digest);
        }
    }
    event["attempt"] = json!(attempt);
    event["effective_request_wire_digest"] = event["request_wire_digest"].take();
    event.as_object_mut().unwrap().remove("request_wire_digest");
    Ok(Some(event))
}

async fn send_selected_route_denial(
    stream: &mut ClientStream,
    status: u16,
) -> transport::RouteDispatchOutcome {
    let denied = matches!(status, 402 | 403);
    let body = json!({"error":{"message":"OpenAI plugin admission rejected the selected route",
        "type": if denied { "permission_error" } else { "service_unavailable" },
        "code": if denied { "plugin_policy_denied" } else { "plugin_hook_unavailable" }}})
    .to_string();
    let bytes = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        match status {
            402 => "Payment Required",
            403 => "Forbidden",
            _ => "Service Unavailable",
        },
        body.len()
    );
    match stream.write_all(bytes.as_bytes()).await {
        Ok(()) => transport::RouteDispatchOutcome::admission_rejected(status),
        Err(_) => transport::RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

/// Lowercase hex of the selected remote peer's endpoint id; absent for a local
/// backend, a URL backend, or a bare label.
fn target_peer_id(target: &SelectedRouteTarget<'_>) -> Option<String> {
    match target {
        SelectedRouteTarget::Mesh(InferenceTarget::Remote(peer)) => {
            Some(hex::encode(peer.as_bytes()))
        }
        _ => None,
    }
}

fn public_target(target: &SelectedRouteTarget<'_>) -> String {
    let target = match target {
        SelectedRouteTarget::Mesh(target) => return format!("{target:?}"),
        SelectedRouteTarget::Url(url) => *url,
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
            public_target(&SelectedRouteTarget::Url(
                "https://operator:secret@example.com/v1?api_key=secret#private"
            )),
            "https://example.com"
        );
        assert_eq!(
            public_target(&SelectedRouteTarget::Mesh(&InferenceTarget::Local(3131))),
            "Local(3131)"
        );
    }

    #[tokio::test]
    async fn selected_event_redacts_malformed_urls_and_labels_mesh_targets() {
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
                None,
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
        let peer = iroh::SecretKey::from_bytes(&[7; 32]).public();
        for target in [InferenceTarget::Local(3131), InferenceTarget::Remote(peer)] {
            let event = selected_route_event(
                &request,
                None,
                "mesh",
                SelectedRouteTarget::Mesh(&target),
                None,
                2,
            )
            .unwrap()
            .unwrap();
            assert_eq!(event["target"], format!("{target:?}"));
            assert!(event.get("model_canonical_ref").is_none());
            assert!(event.get("weights_digest").is_none());
            // Same encoding as the serving side's `requested_by_node_id`, so one
            // buyer-side key pairs with one seller-side key for the same exchange.
            match target {
                InferenceTarget::Remote(peer) => {
                    assert_eq!(event["target_peer_id"], hex::encode(peer.as_bytes()))
                }
                _ => assert!(event.get("target_peer_id").is_none()),
            }
        }
    }

    #[tokio::test]
    async fn selected_event_names_the_selected_model_where_known() {
        let (mut reader, mut writer) = tokio::io::duplex(1024);
        writer.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}").await.unwrap();
        let request = transport::read_http_request(&mut reader).await.unwrap();
        let target = InferenceTarget::Local(3131);
        let mut identity = mesh::ServedModelIdentity {
            model_name: "model".into(),
            canonical_ref: Some("org/repo@main/model.gguf".into()),
            weights_digest: Some("74a4da8c".into()),
            ..Default::default()
        };
        let event = selected_route_event(
            &request,
            Some("model"),
            "mesh",
            SelectedRouteTarget::Mesh(&target),
            Some(&identity),
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(event["model_canonical_ref"], "org/repo@main/model.gguf");
        assert_eq!(event["weights_digest"], "74a4da8c");

        // A peer's gossiped descriptor has no digest: the field is omitted,
        // never filled with a guess.
        identity.weights_digest = None;
        let event = selected_route_event(
            &request,
            Some("model"),
            "mesh",
            SelectedRouteTarget::Mesh(&target),
            Some(&identity),
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(event["model_canonical_ref"], "org/repo@main/model.gguf");
        assert!(event.get("weights_digest").is_none());
    }
}

#[cfg(test)]
#[path = "exchange_entity_tests.rs"]
mod entity_tests;
