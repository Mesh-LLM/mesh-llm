use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments::{
    ledger::{Charge, RequestTerms},
    pricing::{FEE_ALLOWANCE_MSAT, Pricing},
    service::PaymentService,
    wire::{self, Frame},
};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};

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
        result = start(node, peer, raw, price, logging.exchange_id) => result,
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
    }
}

struct CancelOnDrop(tokio::sync::watch::Sender<bool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let _ = self.0.send(true);
    }
}

type StartedExchange = (
    DuplexStream,
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::watch::Sender<bool>,
);

/// Constructed only before any payment-capable task exists.
#[derive(Debug)]
struct PrePaymentTransportFailure;

impl std::fmt::Display for PrePaymentTransportFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider transport failed before payment handoff")
    }
}
impl std::error::Error for PrePaymentTransportFailure {}

async fn start(
    node: &Node,
    peer: iroh::EndpointId,
    raw: &[u8],
    price: Pricing,
    exchange_id: Option<&str>,
) -> Result<StartedExchange> {
    let request = PaidRequest::parse(raw)?;
    let (mut send, mut recv) = node
        .open_http_tunnel(peer)
        .await
        .map_err(|_| PrePaymentTransportFailure)?;
    let service = node.payment_service().await?;
    ensure!(
        effective_intent(&service, &request)?.permits(&price, 0),
        "paid inference is excluded by spending policy or request restriction"
    );
    let id = uuid::Uuid::new_v4().to_string();
    send.write_all(wire::HTTP_UPGRADE)
        .await
        .map_err(|_| PrePaymentTransportFailure)?;
    wire::write(
        &mut send,
        &Frame::Request {
            id: id.clone(),
            model: request.model.clone(),
            pricing: price.clone(),
            http: request.backend_http(&id)?,
        },
    )
    .await
    .map_err(|_| PrePaymentTransportFailure)?;
    // Prefill/invoice receipt happens before spawning anything that can pay.
    let initial = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        read_initial_invoice(&mut recv),
    )
    .await
    .map_err(|_| PrePaymentTransportFailure)??;
    validate_initial_invoice(&service, &request, &price, &id, &initial)?;
    let (pipe, mut output) = tokio::io::duplex(64 * 1024);
    let (ready, wait_ready) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let evidence = exchange_id.map(|id| (node.clone(), id.to_owned()));
    tokio::spawn(async move {
        let result = exchange(
            service,
            peer,
            id,
            request,
            price,
            send,
            recv,
            initial,
            &mut output,
            ready,
            cancellation,
            evidence,
        )
        .await;
        if result.is_err() {
            // Static logging only: invoices, prompt contents, and hashes are
            // operator data and do not belong in ordinary runtime logs.
            tracing::warn!("paid inference exchange interrupted; durable settlement retained");
        }
    });
    Ok((pipe, wait_ready, cancel))
}

async fn read_initial_invoice(recv: &mut (impl AsyncRead + Unpin)) -> Result<Frame> {
    let frame = wire::read(recv).await.map_err(|error| {
        if error.downcast_ref::<std::io::Error>().is_some() {
            anyhow::Error::from(PrePaymentTransportFailure)
        } else {
            error
        }
    })?;
    // A provider-side prefill failure cannot have charged this payer: no
    // invoice has been accepted and no payment task has been spawned.
    if matches!(frame, Frame::Error { .. }) {
        return Err(PrePaymentTransportFailure.into());
    }
    Ok(frame)
}

fn validate_initial_invoice(
    service: &PaymentService,
    request: &PaidRequest,
    price: &Pricing,
    id: &str,
    initial: &Frame,
) -> Result<(u64, u64)> {
    let Frame::InputInvoice { terms, invoice } = initial else {
        bail!("expected input invoice");
    };
    ensure!(
        terms.id == id && terms.model == request.model && terms.pricing == *price,
        "payment terms mismatch"
    );
    ensure!(
        terms.input_tokens > 0 && terms.input_tokens <= 131_072,
        "invalid input count"
    );
    request.validate_output_allowance(terms.max_output_tokens)?;
    let input_amount = price.input_charge(terms.input_tokens)?;
    let total = input_amount
        .checked_add(price.output_charge(terms.max_output_tokens)?)
        .and_then(|n| n.checked_add(2 * FEE_ALLOWANCE_MSAT))
        .context("price overflow")?;
    ensure!(
        terms.max_total_msat == total && terms.expires_at_ms == invoice.expires_at_ms,
        "payment limit mismatch"
    );
    ensure!(
        effective_intent(service, request)?.permits(price, total),
        "paid inference is excluded by spending policy or request restriction"
    );
    invoice.validate_payment(input_amount, mesh_llm_payments::now_ms())?;
    ensure!(
        invoice.amount_msat == Some(input_amount),
        "fixed-amount inference invoice required"
    );
    Ok((input_amount, total))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn exchange(
    service: Arc<PaymentService>,
    peer: iroh::EndpointId,
    id: String,
    request: PaidRequest,
    price: Pricing,
    mut send: impl AsyncWrite + Unpin,
    mut recv: impl AsyncRead + Unpin,
    initial: Frame,
    output: &mut DuplexStream,
    ready: tokio::sync::oneshot::Sender<()>,
    mut cancellation: tokio::sync::watch::Receiver<bool>,
    evidence: Option<(Node, String)>,
) -> Result<()> {
    let (input_amount, total) =
        validate_initial_invoice(&service, &request, &price, &id, &initial)?;
    let Frame::InputInvoice { mut terms, invoice } = initial else {
        bail!("expected input invoice");
    };
    // Peer identity is from authenticated QUIC, never a peer-supplied field.
    terms.exchange_id = evidence.as_ref().map(|(_, id)| id.clone());
    terms.peer = peer.to_string();
    terms.payee = Some(invoice.payee.clone());
    tokio::select! {
        result = service.await_authorization(&terms) => result?,
        _ = cancellation.changed() => {
            let _ = service.ledger.cancel_unstarted(&id);
            let _ = wire::write(&mut send, &Frame::Cancel).await;
            bail!("application disconnected before approval");
        }
    }
    if !effective_intent(&service, &request)?.permits(&price, total) {
        service.ledger.fail_authorization_if_idle(&id)?;
        let _ = wire::write(&mut send, &Frame::Cancel).await;
        bail!("spending policy or request restriction changed before submission");
    }
    let observations =
        super::paid_events::Observations::for_exchange(evidence.as_ref(), &terms).await;
    observations.accepted(terms.max_total_msat);
    observations.invoice(0, &invoice);
    let mut accounted_msat = 0u64;
    // Start durable submission and terminal reconciliation, then read the
    // provider stream concurrently. The provider releases output only after
    // its own receiving wallet sees the payment arrive, so the payer's later
    // terminal observation must not become a second delivery gate.
    let payment_service = service.clone();
    let charge = Charge {
        request_id: id.clone(),
        segment: 0,
        invoice,
        amount_msat: input_amount,
        max_total_msat: input_amount + FEE_ALLOWANCE_MSAT,
    };
    let mut input_payment = tokio::spawn(async move { payment_service.pay_charge(&charge).await });
    let mut input_settled = false;
    let _ = ready.send(());
    let mut cancelled = false;
    let mut output_settled = false;
    loop {
        let reading = wire::read(&mut recv);
        tokio::pin!(reading);
        let frame = loop {
            tokio::select! {
                result = &mut input_payment, if !input_settled => {
                    let payment = result.context("input payment task failed")??;
                    accounted_msat = accounted_msat.saturating_add(payment.amount_msat).saturating_add(payment.fee_msat);
                    observations.settled(0, &payment);
                    input_settled = true;
                }
                frame = &mut reading => break frame?,
                _ = cancellation.changed(), if !cancelled => {
                    cancelled = true;
                    wire::write(&mut send, &Frame::Cancel).await?;
                }
            }
        };
        match frame {
            Frame::Output { bytes } => {
                ensure!(!output_settled, "output after final invoice");
                if !cancelled && output.write_all(&bytes).await.is_err() {
                    cancelled = true;
                    wire::write(&mut send, &Frame::Cancel).await?;
                }
            }
            Frame::OutputInvoice {
                request_id,
                tokens,
                invoice,
            } => {
                ensure!(
                    !output_settled && request_id == id,
                    "unexpected output invoice"
                );
                observations.invoice(1, &invoice);
                let payment = settle_output(&service, &terms, tokens, invoice).await?;
                accounted_msat = accounted_msat
                    .saturating_add(payment.amount_msat)
                    .saturating_add(payment.fee_msat);
                observations.settled(1, &payment);
                output_settled = true;
            }
            Frame::Complete => {
                if !input_settled {
                    let payment = input_payment.await.context("input payment task failed")??;
                    accounted_msat = accounted_msat
                        .saturating_add(payment.amount_msat)
                        .saturating_add(payment.fee_msat);
                    observations.settled(0, &payment);
                }
                service.ledger.finish(&id)?;
                observations.final_amount(accounted_msat);
                return Ok(());
            }
            _ => bail!("invalid payment exchange frame"),
        }
    }
}

pub(crate) async fn settle_output(
    service: &PaymentService,
    terms: &RequestTerms,
    tokens: u64,
    invoice: mesh_llm_payments::invoice::Invoice,
) -> Result<mesh_llm_payments::wallet::Transaction> {
    ensure!(
        tokens > 0 && tokens <= terms.max_output_tokens,
        "output token allowance exceeded"
    );
    ensure!(
        terms.payee.as_deref() == Some(invoice.payee.as_str()),
        "output invoice changed receiving wallet"
    );
    let amount_msat = terms.pricing.output_charge(tokens)?;
    ensure!(
        invoice.amount_msat == Some(amount_msat),
        "output invoice amount mismatch"
    );
    service
        .pay_charge(&Charge {
            request_id: terms.id.clone(),
            segment: 1,
            invoice,
            amount_msat,
            max_total_msat: amount_msat
                .checked_add(FEE_ALLOWANCE_MSAT)
                .context("fee overflow")?,
        })
        .await
}

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
    async fn initial_transport_drop_is_retryable_but_malformed_frame_is_terminal() -> Result<()> {
        let (writer, mut reader) = tokio::io::duplex(64);
        drop(writer);
        assert!(
            read_initial_invoice(&mut reader)
                .await
                .unwrap_err()
                .is::<PrePaymentTransportFailure>()
        );
        let (mut writer, mut reader) = tokio::io::duplex(64);
        writer.write_all(&[0, 0, 0, 1, b'!']).await?;
        assert!(
            !read_initial_invoice(&mut reader)
                .await
                .unwrap_err()
                .is::<PrePaymentTransportFailure>()
        );
        Ok(())
    }

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
