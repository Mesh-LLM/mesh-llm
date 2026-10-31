//! Selected-route plugin admission immediately before each backend attempt.
use super::{client_stream::ClientStream, transport};
use crate::inference::election::InferenceTarget;
use crate::mesh;
use serde_json::json;
use tokio::io::AsyncWriteExt;

pub(super) enum SelectedRouteTarget<'a> {
    Url(&'a str),
    /// A mesh routing target, labelled by its debug form.
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
    let pricing = selected_route_pricing(node, &target, model).await;
    let event = match selected_route_event(request, model, provider, target, pricing, attempt) {
        Ok(event) => event?,
        Err(error) => {
            tracing::warn!(%error, "cannot observe effective OpenAI request entity");
            stream.record_exchange_outcome("internal_hook_failure");
            return Some(send_selected_route_denial(stream, 503).await);
        }
    };
    apply_selected_route_policy(&manager, request, stream, event).await
}

/// The selected remote mesh peer's payment offer for `model`, as routing ranks
/// it (`Node::peer_payment_offer`), in `Pricing`'s own JSON form. `None` for a
/// local target, an external URL, a request without a model, or a peer that
/// serves the model for free.
#[cfg(feature = "payments")]
async fn selected_route_pricing(
    node: &mesh::Node,
    target: &SelectedRouteTarget<'_>,
    model: Option<&str>,
) -> Option<serde_json::Value> {
    let (SelectedRouteTarget::Mesh(InferenceTarget::Remote(peer)), Some(model)) = (target, model)
    else {
        return None;
    };
    let pricing = node.peer_payment_offer(*peer, model).await?;
    Some(json!(pricing))
}

/// Without the payments feature no peer can advertise a price, and the host
/// does not list `openai_exchange.backend_pricing.v1`.
#[cfg(not(feature = "payments"))]
async fn selected_route_pricing(
    _node: &mesh::Node,
    _target: &SelectedRouteTarget<'_>,
    _model: Option<&str>,
) -> Option<serde_json::Value> {
    None
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
    pricing: Option<serde_json::Value>,
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
    if let Some(pricing) = pricing {
        event["pricing"] = pricing;
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
        SelectedRouteTarget::Mesh(target) => return format!("{target:?}"),
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
            public_target(SelectedRouteTarget::Mesh(&InferenceTarget::Local(3131))),
            "Local(3131)"
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
            assert!(event.get("pricing").is_none());
        }
    }

    async fn chat_request() -> transport::BufferedHttpRequest {
        let (mut reader, mut writer) = tokio::io::duplex(1024);
        writer.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}").await.unwrap();
        transport::read_http_request(&mut reader).await.unwrap()
    }

    #[tokio::test]
    async fn selected_event_carries_the_given_pricing_verbatim() {
        let request = chat_request().await;
        let peer = iroh::SecretKey::from_bytes(&[7; 32]).public();
        let pricing = json!({"input_msat_per_million": 10, "output_msat_per_million": 20});
        let event = selected_route_event(
            &request,
            Some("m"),
            "mesh",
            SelectedRouteTarget::Mesh(&InferenceTarget::Remote(peer)),
            Some(pricing.clone()),
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(event["pricing"], pricing);
    }

    /// A node that knows one peer serving `test`, priced by `offer` (free when
    /// `None`).
    #[cfg(feature = "payments")]
    async fn node_with_peer(
        offer: Option<mesh_llm_payments_types::pricing::Pricing>,
    ) -> anyhow::Result<(mesh::Node, mesh::Node)> {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let seller = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 0 }).await?;
        let mut announcement =
            seller.build_local_announcement(seller.snapshot_local_announcement_data().await);
        if let Some(offer) = offer {
            announcement.lightning_offers.insert("test".into(), offer);
        }
        node.add_peer_after_direct_requirements_validated(
            seller.id(),
            seller.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
        Ok((node, seller))
    }

    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn selected_remote_peer_pricing_is_its_offer_for_the_model() -> anyhow::Result<()> {
        use mesh_llm_payments_types::pricing::Pricing;
        for (cache_discount_percent, expected) in [
            (
                0,
                json!({"input_msat_per_million": 10, "output_msat_per_million": 20}),
            ),
            (
                25,
                json!({
                    "input_msat_per_million": 10,
                    "output_msat_per_million": 20,
                    "cache_discount_percent": 25,
                }),
            ),
        ] {
            let (node, seller) = node_with_peer(Some(Pricing {
                input_msat_per_million: 10,
                output_msat_per_million: 20,
                cache_discount_percent,
            }))
            .await?;
            let remote = InferenceTarget::Remote(seller.id());
            let target = SelectedRouteTarget::Mesh(&remote);
            let pricing = selected_route_pricing(&node, &target, Some("test")).await;
            assert_eq!(pricing.as_ref(), Some(&expected));
            let event = selected_route_event(
                &chat_request().await,
                Some("test"),
                "mesh",
                target,
                pricing,
                1,
            )?
            .unwrap();
            assert_eq!(event["pricing"], expected);
            // Another model on the same peer has no offer: a free route.
            let other = SelectedRouteTarget::Mesh(&remote);
            assert_eq!(
                selected_route_pricing(&node, &other, Some("other")).await,
                None
            );
            // Without a model there is nothing to price.
            assert_eq!(selected_route_pricing(&node, &other, None).await, None);
            node.endpoint.close().await;
            seller.endpoint.close().await;
        }
        Ok(())
    }

    #[cfg(not(feature = "payments"))]
    #[tokio::test]
    async fn without_payments_no_route_carries_pricing() -> anyhow::Result<()> {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let peer = InferenceTarget::Remote(iroh::SecretKey::from_bytes(&[7; 32]).public());
        let target = SelectedRouteTarget::Mesh(&peer);
        assert_eq!(
            selected_route_pricing(&node, &target, Some("test")).await,
            None
        );
        node.endpoint.close().await;
        Ok(())
    }

    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn local_external_and_free_routes_carry_no_pricing() -> anyhow::Result<()> {
        let (node, free_peer) = node_with_peer(None).await?;
        let free = InferenceTarget::Remote(free_peer.id());
        let local = InferenceTarget::Local(3131);
        for target in [
            SelectedRouteTarget::Mesh(&free),
            SelectedRouteTarget::Mesh(&local),
            SelectedRouteTarget::Mesh(&InferenceTarget::None),
            SelectedRouteTarget::Url("https://example.com/v1"),
        ] {
            assert_eq!(
                selected_route_pricing(&node, &target, Some("test")).await,
                None
            );
        }
        let event = selected_route_event(
            &chat_request().await,
            Some("test"),
            "mesh",
            SelectedRouteTarget::Mesh(&free),
            selected_route_pricing(&node, &SelectedRouteTarget::Mesh(&free), Some("test")).await,
            1,
        )?
        .unwrap();
        assert!(event.as_object().unwrap().get("pricing").is_none());
        node.endpoint.close().await;
        free_peer.endpoint.close().await;
        Ok(())
    }
}

#[cfg(test)]
#[path = "exchange_entity_tests.rs"]
mod entity_tests;
