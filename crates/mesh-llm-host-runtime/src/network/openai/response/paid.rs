use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments_types::contract::{
    AuthorizeRequest, CancelRequest, CancelStage, Empty, IdRequest, PayInputRequest,
    SettleOutputRequest, ops,
};
use mesh_llm_payments_types::{
    RequestTerms,
    intent::PaymentIntent,
    pricing::Pricing,
    wire::{self, Frame},
};
use mesh_llm_wallet::provider::Transaction;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};

use super::{
    common::{ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult},
    routing::route_local_attempt_after_forward,
};
use crate::network::payments::{client::Payments, request::PaidRequest};
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
        output_digests: Default::default(),
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
    let payments = crate::network::payments::client::Payments::for_node(node).await?;
    ensure!(
        effective_intent(&payments, &request)
            .await?
            .permits(&price, 0),
        "paid inference is excluded by spending policy or request restriction"
    );
    // Read the balance for approval during the seller's prefill. If a
    // concurrent payment settles in between, the balance overstates the
    // funds, but the wallet still refuses any payment it cannot afford.
    let id = uuid::Uuid::new_v4().to_string();
    let _: Empty = payments
        .call(ops::PREFETCH, &IdRequest { id: id.clone() })
        .await?;
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
    if let Err(error) = validate_initial_invoice(&payments, &request, &price, &id, &initial).await {
        cancel(&payments, &id, CancelStage::Unstarted).await;
        return Err(error);
    }
    let (pipe, mut output) = tokio::io::duplex(64 * 1024);
    let (ready, wait_ready) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let evidence = exchange_id.map(|id| (node.clone(), id.to_owned()));
    let strike_node = node.clone();
    tokio::spawn(async move {
        let mut progress = ExchangeProgress::default();
        let result = exchange_tracked(
            payments,
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
            &mut progress,
        )
        .await;
        if result.is_err() && progress.paid_undelivered() {
            record_paid_undelivered(&strike_node, peer).await;
        }
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

async fn validate_initial_invoice(
    payments: &Payments,
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
    let total = price.request_cap_msat(input_amount, terms.max_output_tokens)?;
    ensure!(
        terms.max_total_msat == total && terms.expires_at_ms == invoice.expires_at_ms,
        "payment limit mismatch"
    );
    ensure!(
        effective_intent(payments, request)
            .await?
            .permits(price, total),
        "paid inference is excluded by spending policy or request restriction"
    );
    invoice.validate_payment(input_amount, mesh_llm_wallet::now_ms())?;
    ensure!(
        invoice.amount_msat == Some(input_amount),
        "fixed-amount inference invoice required"
    );
    Ok((input_amount, total))
}

/// What a paid exchange had done when it ended, for the payee blocklist.
#[derive(Debug, Default)]
pub(crate) struct ExchangeProgress {
    input_settled: bool,
    output_delivered: bool,
    cancelled: bool,
}

impl ExchangeProgress {
    /// Our input payment settled, the provider sent no output, and we did not
    /// cancel: the provider took the prefill charge and delivered nothing.
    pub(crate) fn paid_undelivered(&self) -> bool {
        self.input_settled && !self.output_delivered && !self.cancelled
    }
}

async fn record_paid_undelivered(node: &Node, peer: iroh::EndpointId) {
    let directory = node.config_state.lock().await.payment_directory();
    let payee = peer.to_string();
    let now = mesh_llm_wallet::now_ms();
    match tokio::task::spawn_blocking(move || {
        crate::network::payments::strikes::record_strike(&directory, &payee, now)
    })
    .await
    {
        Ok(Ok(true)) => {
            tracing::warn!("paid provider blocked after repeated paid-but-undelivered exchanges");
        }
        Ok(Ok(false)) => {}
        _ => tracing::warn!("could not persist paid-but-undelivered strike"),
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn exchange(
    payments: Payments,
    peer: iroh::EndpointId,
    id: String,
    request: PaidRequest,
    price: Pricing,
    send: impl AsyncWrite + Unpin,
    recv: impl AsyncRead + Unpin,
    initial: Frame,
    output: &mut DuplexStream,
    ready: tokio::sync::oneshot::Sender<()>,
    cancellation: tokio::sync::watch::Receiver<bool>,
    evidence: Option<(Node, String)>,
) -> Result<()> {
    exchange_tracked(
        payments,
        peer,
        id,
        request,
        price,
        send,
        recv,
        initial,
        output,
        ready,
        cancellation,
        evidence,
        &mut ExchangeProgress::default(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn exchange_tracked(
    payments: Payments,
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
    progress: &mut ExchangeProgress,
) -> Result<()> {
    let (_input_amount, total) =
        match validate_initial_invoice(&payments, &request, &price, &id, &initial).await {
            Ok(amounts) => amounts,
            Err(error) => {
                cancel(&payments, &id, CancelStage::Unstarted).await;
                return Err(error);
            }
        };
    let Frame::InputInvoice { mut terms, invoice } = initial else {
        bail!("expected input invoice");
    };
    // Peer identity is from authenticated QUIC, never a peer-supplied field.
    terms.exchange_id = evidence.as_ref().map(|(_, id)| id.clone());
    terms.peer = peer.to_string();
    terms.payee = Some(invoice.payee.clone());
    authorize_terms(
        &payments,
        &request,
        &price,
        total,
        &terms,
        &mut send,
        &mut cancellation,
    )
    .await?;
    let observations =
        super::paid_events::Observations::for_exchange(evidence.as_ref(), &terms).await;
    observations.accepted(terms.max_total_msat);
    observations.invoice(0, &invoice);
    let mut accounted_msat = 0u64;
    // Start durable submission and terminal reconciliation, then read the
    // provider stream concurrently. The provider releases output only after
    // its own receiving wallet sees the payment arrive, so the payer's later
    // terminal observation must not become a second delivery gate.
    let payment_client = payments.clone();
    let pay_input = PayInputRequest {
        terms: terms.clone(),
        invoice,
    };
    let mut input_payment = tokio::spawn(async move {
        payment_client
            .call::<_, Transaction>(ops::PAY_INPUT, &pay_input)
            .await
    });
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
                    progress.input_settled = true;
                }
                frame = &mut reading => break frame?,
                _ = cancellation.changed(), if !cancelled => {
                    cancelled = true;
                    progress.cancelled = true;
                    wire::write(&mut send, &Frame::Cancel).await?;
                }
            }
        };
        match frame {
            Frame::Output { bytes } => {
                ensure!(!output_settled, "output after final invoice");
                progress.output_delivered |= !bytes.is_empty();
                if !cancelled && output.write_all(&bytes).await.is_err() {
                    cancelled = true;
                    progress.cancelled = true;
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
                let payment = settle_output(&payments, &terms, tokens, invoice).await?;
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
                let _: Empty = payments
                    .call(ops::FINISH, &IdRequest { id: id.clone() })
                    .await?;
                observations.final_amount(accounted_msat);
                return Ok(());
            }
            _ => bail!("invalid payment exchange frame"),
        }
    }
}

/// Durably approve `terms`, then re-check spending policy before any payment.
async fn authorize_terms(
    payments: &Payments,
    request: &PaidRequest,
    price: &Pricing,
    total: u64,
    terms: &RequestTerms,
    send: &mut (impl AsyncWrite + Unpin),
    cancellation: &mut tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let authorize = AuthorizeRequest {
        terms: terms.clone(),
    };
    tokio::select! {
        result = payments.call::<_, Empty>(ops::AUTHORIZE, &authorize) => { result?; }
        _ = cancellation.changed() => {
            cancel(payments, &terms.id, CancelStage::Unstarted).await;
            let _ = wire::write(send, &Frame::Cancel).await;
            bail!("application disconnected before approval");
        }
    }
    if !effective_intent(payments, request)
        .await?
        .permits(price, total)
    {
        payments
            .call::<_, Empty>(
                ops::CANCEL,
                &CancelRequest {
                    id: terms.id.clone(),
                    stage: CancelStage::Authorization,
                },
            )
            .await?;
        let _ = wire::write(send, &Frame::Cancel).await;
        bail!("spending policy or request restriction changed before submission");
    }
    Ok(())
}

pub(crate) async fn settle_output(
    payments: &Payments,
    terms: &RequestTerms,
    tokens: u64,
    invoice: mesh_llm_wallet::invoice::Invoice,
) -> Result<Transaction> {
    payments
        .call(
            ops::SETTLE_OUTPUT,
            &SettleOutputRequest {
                terms: terms.clone(),
                tokens,
                invoice,
            },
        )
        .await
}

/// Best-effort release of a request that never started paying.
async fn cancel(payments: &Payments, id: &str, stage: CancelStage) {
    let _ = payments
        .call::<_, Empty>(
            ops::CANCEL,
            &CancelRequest {
                id: id.to_owned(),
                stage,
            },
        )
        .await;
}

pub(super) async fn effective_intent(
    payments: &Payments,
    request: &PaidRequest,
) -> Result<PaymentIntent> {
    let profile: PaymentIntent = payments.call(ops::PAYMENT_INTENT, &Empty {}).await?;
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
