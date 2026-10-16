//! Buyer side of the token-metered exchange, for locally originated paid
//! requests: admission at the request, deposit before any backend work,
//! unchanged streaming, settlement from the seller's reported usage. Every
//! ledger and wallet step goes through the `payments.v1` provider; the
//! buyer's own spending cap is local policy and never leaves this node.

mod exchange;

use anyhow::{Result, bail, ensure};
use mesh_llm_payments_types::contract::{
    CancelRequest, CancelStage, Empty, RoutingBudgetRequest, RoutingBudgetResponse, ops,
};
use mesh_llm_payments_types::metered_wire::{self as wire, AdmissionTerms, Frame};
use mesh_llm_payments_types::metering::validate_payment_floor;
use mesh_llm_payments_types::{RequestTerms, intent::PaymentIntent, pricing::Pricing};
use tokio::io::{AsyncRead, AsyncWriteExt, DuplexStream};

use super::{
    common::{ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult},
    routing::route_local_attempt_after_forward,
};
use crate::network::payments::{client::Payments, request::PaidRequest};
use crate::{mesh::Node, network::openai::client_stream::ClientStream};

pub(crate) use exchange::settle;

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
    let mut cancel_on_drop = CancelOnDrop(Some(cancel));
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
    // Only the application going away cancels the exchange. Every other
    // outcome means the exchange itself ended HTTP delivery (a full response,
    // or a settlement with no output), and it goes on to settle uncancelled
    // so a seller that delivered nothing is still held to it.
    if !matches!(result, RouteAttemptResult::ClientDisconnected) {
        cancel_on_drop.disarm();
    }
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
        addr.ip().is_loopback()
            && !crate::network::management_origin::is_untrusted_management_request(addr)
    })
}

fn trusted_payment_headers(peer: Option<std::net::SocketAddr>, raw: &[u8]) -> bool {
    use mesh_llm_control_api::access::{is_trusted_local_request, request_host, request_origin};
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

struct CancelOnDrop(Option<tokio::sync::watch::Sender<bool>>);
impl CancelOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancel) = &self.0 {
            let _ = cancel.send(true);
        }
    }
}

pub(crate) type StartedExchange = (
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

/// Seller admissions are read within this bound; a seller that does not
/// answer has not charged anything yet. Preserve the previous initial-invoice
/// allowance for wallet startup and queued admissions, without assuming a
/// typical provider latency. Client disconnect in `route` still cancels this wait.
const ADMISSION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// How long an accepted admission stays approved in the ledger.
const AUTHORIZATION_LIFETIME_MS: u64 = 3_600_000;

pub(crate) async fn start(
    node: &Node,
    peer: iroh::EndpointId,
    raw: &[u8],
    price: Pricing,
    exchange_id: Option<&str>,
) -> Result<StartedExchange> {
    let request = PaidRequest::parse(raw)?;
    let payments = Payments::for_node(node).await?;
    let cap = admission_eligibility(&payments, &request, &price).await?;
    let id = uuid::Uuid::new_v4().to_string();
    let (mut send, mut recv) = node
        .open_http_tunnel(peer)
        .await
        .map_err(|_| PrePaymentTransportFailure)?;
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
    // Nothing that can pay exists before the admission is validated.
    let initial = tokio::time::timeout(ADMISSION_TIMEOUT, read_admission(&mut recv))
        .await
        .map_err(|_| PrePaymentTransportFailure)??;
    let (terms, deposit) = match validate_admission(initial, &request, &price, &id) {
        Ok(admission) => admission,
        Err(error) => {
            cancel(&payments, &id, CancelStage::Unstarted).await;
            return Err(error);
        }
    };
    let (pipe, mut output) = tokio::io::duplex(64 * 1024);
    let (ready, wait_ready) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let evidence = exchange_id.map(|id| (node.clone(), id.to_owned()));
    let strike_node = node.clone();
    let classification = cancellation.clone();
    let accepted = exchange::Accepted {
        payments,
        local: RequestTerms {
            exchange_id: exchange_id.map(str::to_owned),
            id,
            peer: peer.to_string(),
            payee: deposit.as_ref().map(|invoice| invoice.payee.clone()),
            model: terms.model.clone(),
            pricing: price,
            max_output_tokens: u64::from(request.max_tokens),
            max_total_msat: cap,
            expires_at_ms: mesh_llm_wallet::now_ms() + AUTHORIZATION_LIFETIME_MS,
        },
        terms,
        deposit,
    };
    tokio::spawn(async move {
        let mut progress = ExchangeProgress::default();
        let result = exchange::run(
            accepted,
            send,
            recv,
            &mut output,
            ready,
            cancellation,
            evidence,
            &mut progress,
        )
        .await;
        // The exchange's own cancellation branch can lose a race with a
        // seller read error, so let the live watch value decide whether this
        // request was cancelled rather than whichever branch happened to run.
        progress.observe_cancellation(&classification);
        if result.is_err() && progress.paid_undelivered() {
            record_paid_undelivered(&strike_node, peer).await;
        }
        if result.is_err() {
            // Static logging only: invoices, prompt contents, and hashes are
            // operator data and do not belong in ordinary runtime logs.
            tracing::warn!("metered inference exchange interrupted; durable settlement retained");
        }
    });
    Ok((pipe, wait_ready, cancel))
}

/// Check paid eligibility and advertised rates before requesting admission.
/// The returned value preserves the legacy terms field, not a settlement cap.
/// Daily spending is checked again when the provider durably approves admission;
/// accepted work settles at the admitted rates without reserving budget.
async fn admission_eligibility(
    payments: &Payments,
    request: &PaidRequest,
    price: &Pricing,
) -> Result<u64> {
    let budget: RoutingBudgetResponse = payments
        .call(
            ops::ROUTING_BUDGET,
            &RoutingBudgetRequest {
                request_intent: request
                    .intent
                    .as_ref()
                    .map(serde_json::to_value)
                    .transpose()?,
            },
        )
        .await?;
    let PaymentIntent::AllowPaid { max_total_msat, .. } = budget.intent else {
        bail!("paid inference is excluded by spending policy or request restriction");
    };
    let cap = max_total_msat;
    ensure!(
        budget.available_msat > 0 && budget.intent.permits(price),
        "seller prices or remaining budget exceed the spending policy"
    );
    Ok(cap)
}

async fn read_admission(recv: &mut (impl AsyncRead + Unpin)) -> Result<Frame> {
    // Nothing is charged before admission, so any unreadable reply is
    // retryable. That includes an older seller that does not know this
    // protocol and answers with plain HTTP bytes.
    let frame = wire::read(recv).await.map_err(|error| {
        tracing::debug!(%error, "unreadable payment admission; trying another target");
        anyhow::Error::from(PrePaymentTransportFailure)
    })?;
    // A seller-side refusal or failure here cannot have charged this buyer:
    // no admission has been accepted and no payment task has been spawned.
    if matches!(frame, Frame::Error { .. }) {
        return Err(PrePaymentTransportFailure.into());
    }
    Ok(frame)
}

/// The admission must restate exactly what was asked, at the frozen price,
/// with a deposit that is either absent or the single defined share.
fn validate_admission(
    frame: Frame,
    request: &PaidRequest,
    price: &Pricing,
    id: &str,
) -> Result<(AdmissionTerms, Option<mesh_llm_wallet::invoice::Invoice>)> {
    let Frame::Admission { terms, invoice } = frame else {
        bail!("expected admission");
    };
    ensure!(
        terms.id == id && terms.model == request.model && terms.pricing == *price,
        "admission terms mismatch"
    );
    // The payment service also refuses a seller minimum above this node's own.
    validate_payment_floor(terms.min_payment_msat)?;
    ensure!(
        terms.deposit_msat == 0 || terms.deposit_msat == terms.min_payment_msat,
        "deposit mismatch"
    );
    match &invoice {
        Some(invoice) => {
            ensure!(
                terms.deposit_msat > 0 && invoice.amount_msat == Some(terms.deposit_msat),
                "deposit invoice mismatch"
            );
            invoice.validate_payment(terms.deposit_msat, mesh_llm_wallet::now_ms())?;
        }
        None => ensure!(terms.deposit_msat == 0, "deposit invoice missing"),
    }
    Ok((terms, invoice))
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

/// What a paid exchange had done when it ended, for the payee blocklist.
#[derive(Debug, Default, Clone)]
pub(crate) struct ExchangeProgress {
    pub(super) deposit_settled: bool,
    pub(super) output_delivered: bool,
    pub(super) cancelled: bool,
}

impl ExchangeProgress {
    /// Our deposit settled, the seller sent no output and we did not cancel:
    /// it took the deposit and delivered nothing, whether it
    /// hung up mid-exchange or closed the exchange with no output at all. An
    /// empty completion is indistinguishable from the second case here, which
    /// is a bounded false positive the strike threshold tames. What this must
    /// never blame the provider for is a failure after it delivered output, or
    /// after we cancelled it.
    pub(crate) fn paid_undelivered(&self) -> bool {
        self.deposit_settled && !self.output_delivered && !self.cancelled
    }

    /// Adopt the live cancellation state. A client disconnect and a provider
    /// read failure can race inside tokio's select, whose ready branches have no
    /// priority, so the read-error branch can return before the cancellation
    /// branch runs. Reading the watch value here keeps our own disconnect from
    /// becoming a strike against the provider.
    pub(crate) fn observe_cancellation(
        &mut self,
        cancellation: &tokio::sync::watch::Receiver<bool>,
    ) {
        self.cancelled |= *cancellation.borrow();
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn any_unreadable_admission_is_retryable() -> Result<()> {
        let (writer, mut reader) = tokio::io::duplex(64);
        drop(writer);
        assert!(
            read_admission(&mut reader)
                .await
                .unwrap_err()
                .is::<PrePaymentTransportFailure>()
        );
        // A malformed frame, and an older seller answering in plain HTTP
        // (parsed as a huge frame length), both fall back to another target.
        for reply in [
            &[0, 0, 0, 1, b'!'][..],
            b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n",
        ] {
            let (mut writer, mut reader) = tokio::io::duplex(256);
            writer.write_all(reply).await?;
            drop(writer);
            assert!(
                read_admission(&mut reader)
                    .await
                    .unwrap_err()
                    .is::<PrePaymentTransportFailure>()
            );
        }
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
                    cache_discount_percent: 0,
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

    #[test]
    fn a_strike_needs_a_settled_input_payment_and_no_delivered_output() {
        let settled = ExchangeProgress {
            deposit_settled: true,
            ..ExchangeProgress::default()
        };
        assert!(settled.paid_undelivered(), "took the deposit, no output");
        for progress in [
            // The provider sent output, so it delivered what we paid for even
            // if the exchange then failed on our side.
            ExchangeProgress {
                output_delivered: true,
                ..settled.clone()
            },
            // We cancelled, so the missing output is our own doing.
            ExchangeProgress {
                cancelled: true,
                ..settled.clone()
            },
            // The deposit never settled: this is a pre-payment failure,
            // which the existing target-health cooldown already covers.
            ExchangeProgress::default(),
        ] {
            assert!(!progress.paid_undelivered(), "{progress:?}");
        }
    }

    #[test]
    fn a_cancellation_that_arrives_after_the_read_error_still_exempts_the_provider() {
        let (cancel, cancellation) = tokio::sync::watch::channel(false);
        let mut progress = ExchangeProgress {
            deposit_settled: true,
            ..ExchangeProgress::default()
        };
        // The read-error branch won the race and returned before the
        // cancellation branch ran; the watch value still exempts the provider.
        cancel.send(true).expect("the receiver is alive");
        progress.observe_cancellation(&cancellation);
        assert!(!progress.paid_undelivered());
    }
}
