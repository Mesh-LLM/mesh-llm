use anyhow::Result;
use mesh_llm_payments::{pricing::Pricing, service::PaymentService};
use tokio::io::{AsyncWriteExt, DuplexStream};

use super::{
    common::{ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult},
    routing::route_local_attempt_after_forward,
};
use crate::network::payments::request::PaidRequest;
use crate::{mesh::Node, network::openai::client_stream::ClientStream};

pub(super) async fn route(
    node: &Node,
    client: &mut ClientStream,
    peer: iroh::EndpointId,
    raw: &[u8],
    price: Pricing,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    if !is_local_origin(client) || !trusted_payment_headers(client.peer_addr().ok(), raw) {
        return payment_error(
            client,
            "only locally originated requests may spend this wallet",
        )
        .await;
    }
    let started = tokio::select! {
        result = super::metered_payer::start(node, peer, raw, price, logging.exchange_id) => result,
        _ = client.wait_for_response_disconnect() => return RouteAttemptResult::ClientDisconnected,
    };
    let (pipe, mut ready, cancel) = match started {
        Ok(pipe) => pipe,
        Err(error) if error.is::<PrePaymentTransportFailure>() => {
            return RouteAttemptResult::RetryableUnavailable;
        }
        Err(_) => return payment_error(client, "could not start paid inference").await,
    };
    let _cancel_on_drop = CancelOnDrop(cancel);
    tokio::select! {
        result = &mut ready => if result.is_err() { return payment_error(client, "payment was not authorized").await; },
        _ = client.wait_for_response_disconnect() => return RouteAttemptResult::ClientDisconnected,
    }
    let mut pipe = pipe;
    let result = route_local_attempt_after_forward(
        client,
        &mut pipe,
        0,
        logging.request_id,
        ResponseRetryPolicy::next_target_available(false),
        logging.response_adapter,
        logging.served_by,
        logging.peer_capsule_id,
        logging.route_observer,
    )
    .await;
    // Once a paid exchange starts, ordinary transport/quality retries must not
    // create a second bill. A failed attempt is terminal for this HTTP request.
    match result {
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableUnavailable
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_) => {
            payment_error(
                client,
                "paid inference interrupted; settlement remains recoverable",
            )
            .await
        }
        result => result,
    }
}

pub(super) fn is_local_origin(client: &ClientStream) -> bool {
    client.peer_addr().is_ok_and(|addr| {
        addr.ip().is_loopback() && !crate::network::tunnel::is_remote_bridge(addr)
    })
}

fn trusted_payment_headers(peer: Option<std::net::SocketAddr>, raw: &[u8]) -> bool {
    use crate::api::access::{is_trusted_local_request, request_host, request_origin};
    match (request_origin(raw), request_host(raw)) {
        (Ok(origin), Ok(host)) => is_trusted_local_request(peer, origin, host),
        _ => false,
    }
}

pub(super) async fn payment_error(client: &mut ClientStream, message: &str) -> RouteAttemptResult {
    let body =
        serde_json::json!({"error": {"message": message, "type": "payment_required"}}).to_string();
    let response = format!(
        "HTTP/1.1 402 Payment Required\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    if client.write_all(response.as_bytes()).await.is_err() {
        return RouteAttemptResult::ClientDisconnected;
    }
    RouteAttemptResult::Delivered {
        status_code: 402,
        usage: None,
        cache_cost: None,
        output_digests: Default::default(),
    }
}

struct CancelOnDrop(tokio::sync::watch::Sender<bool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(true);
    }
}

pub(super) type StartedExchange = (
    DuplexStream,
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::watch::Sender<bool>,
);

/// Constructed only before any payment-capable task exists.
#[derive(Debug)]
pub(super) struct PrePaymentTransportFailure;

impl std::fmt::Display for PrePaymentTransportFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider transport failed before payment handoff")
    }
}
impl std::error::Error for PrePaymentTransportFailure {}

pub(super) fn effective_intent(
    service: &PaymentService,
    request: &PaidRequest,
) -> Result<mesh_llm_payments::intent::PaymentIntent> {
    let profile = service.ledger.payment_intent()?;
    Ok(request
        .intent
        .as_ref()
        .map_or_else(|| profile.clone(), |request| profile.restrict(request)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn payments_cross_site_loopback_requests_are_denied_before_wallet_or_peer_access()
    -> Result<()> {
        for headers in [
            "Host: localhost\r\nOrigin: https://attacker.example\r\n",
            "Host: attacker.example\r\n",
            "Host: 127.0.0.1\r\nOrigin: null\r\n",
        ] {
            let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
            let mut application = tokio::net::TcpStream::connect(listener.local_addr()?).await?;
            let (socket, _) = listener.accept().await?;
            let mut client: ClientStream = socket.into();
            // A simple browser POST does not need application/json or a CORS preflight.
            let raw = format!(
                "POST /v1/completions HTTP/1.1\r\n{headers}Content-Type: text/plain\r\n\r\n{{\"model\":\"test\",\"prompt\":\"Hi\"}}"
            );
            let result = route(
                &node,
                &mut client,
                node.endpoint.id(),
                raw.as_bytes(),
                Pricing {
                    input_msat_per_million: 1,
                    output_msat_per_million: 1,
                    minimum_invoice_msat: 1,
                },
                RouteAttemptLoggingContext {
                    exchange_id: None,
                    request_id: Default::default(),
                    retry_policy: ResponseRetryPolicy::next_target_available(false),
                    response_adapter:
                        crate::network::openai::request_normalize::ResponseAdapter::None,
                    served_by: None,
                    peer_capsule_id: None,
                    route_observer: crate::logging::OpenAiRouteObserver::default(),
                },
            )
            .await;
            assert!(matches!(
                result,
                RouteAttemptResult::Delivered {
                    status_code: 402,
                    ..
                }
            ));
            assert!(node.payments.get().is_none());
            drop(client);
            let mut response = String::new();
            tokio::io::AsyncReadExt::read_to_string(&mut application, &mut response).await?;
            assert!(response.contains("only locally originated"));
            node.endpoint.close().await;
        }
        Ok(())
    }

    #[test]
    fn payments_native_and_trusted_local_browser_headers_are_accepted() {
        let peer = Some(([127, 0, 0, 1], 1234).into());
        assert!(trusted_payment_headers(
            peer,
            b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n"
        ));
        assert!(trusted_payment_headers(peer, b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://localhost:3131\r\n\r\n"));
        assert!(!trusted_payment_headers(
            peer,
            b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nOrigin: \xff\r\n\r\n"
        ));
    }
}
