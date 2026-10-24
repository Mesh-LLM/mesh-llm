//! Seller side of the token-metered exchange: admission at the request,
//! deposit before any backend work, unchanged streaming, settlement from the
//! backend's reported usage. Every ledger and wallet step goes through the
//! `payments.v1` provider; this module owns the transport, the backend and
//! delivery.

mod exchange_events;

use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments_types::contract::{
    Empty, FinalBill, RequestIdRequest, ServeAdmitRequest, ServeAdmitResponse,
    ServeRecoverResponse, ServeSettleRequest, SettledReceived, ops,
};
use mesh_llm_payments_types::metered_wire::{self as wire, AdmissionTerms, Frame};
use mesh_llm_payments_types::metering::MeteredUsage;
use mesh_llm_payments_types::response_usage::{ResponseEvidence, ResponseUsage};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

use super::{backend::Backend, client::Payments, request::PaidRequest};
use crate::{inference::election::ModelTargets, mesh::Node};
use exchange_events::DeliveredCapture;

/// How long the backend may go without producing a byte.
const BACKEND_STALL: Duration = Duration::from_secs(300);

pub(crate) async fn serve(
    node: Node,
    peer: iroh::EndpointId,
    mut reader: impl AsyncRead + Unpin,
    mut writer: impl AsyncWrite + Unpin,
    targets: ModelTargets,
) -> Result<()> {
    let result = serve_inner(&node, peer, &mut reader, &mut writer, &targets).await;
    if let Err(error) = &result {
        tracing::warn!(buyer = %peer, error = %format!("{error:#}"), "paid exchange failed");
        // Avoid reflecting request contents or wallet-provider errors to peers.
        let _ = wire::write(
            &mut writer,
            &Frame::Error {
                message: "metered exchange interrupted; recover by request ID".into(),
            },
        )
        .await;
    }
    result
}

/// One admitted request, from admission to settlement.
struct Session {
    payments: Payments,
    terms: AdmissionTerms,
    request: PaidRequest,
    backend: Backend,
    deposit: Option<mesh_llm_wallet::invoice::Invoice>,
    observations: super::lifecycle::Observations,
    exchange_id: Option<String>,
    /// This connection received the successful once-only dispatch claim.
    /// A shared ledger's dispatched flag cannot establish local ownership.
    owns_dispatch: bool,
}

async fn serve_inner(
    node: &Node,
    peer: iroh::EndpointId,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    targets: &ModelTargets,
) -> Result<()> {
    let initial = tokio::time::timeout(Duration::from_secs(10), wire::read(reader)).await??;
    let payments = Payments::for_node(node).await?;
    let payer = peer;
    let peer = peer.to_string();
    if let Frame::Recover { id } = initial {
        return recover(&payments, &id, writer).await;
    }
    let admission = match admit(node, &payments, peer.clone(), initial, targets).await {
        Ok(admission) => admission,
        Err(error) => {
            // Without this line a refused buyer only sees its own 503.
            tracing::warn!(buyer = %peer, error = %format!("{error:#}"), "refused paid admission");
            let _ = wire::write(
                writer,
                &Frame::Error {
                    message: format!("admission refused: {error:#}"),
                },
            )
            .await;
            return Err(error);
        }
    };
    let mut session = match admission {
        Admission::Admitted(session) => session,
        Admission::Refused(reason) => {
            return wire::write(
                writer,
                &Frame::Error {
                    message: reason.message().into(),
                },
            )
            .await;
        }
        // The buyer replayed a request this seller already holds: answer
        // from the ledger, never a second dispatch.
        Admission::Replayed(id) => return recover(&payments, &id, writer).await,
    };
    let result = serve_admitted(node, &mut session, payer, reader, writer).await;
    if result.is_err() && session.owns_dispatch {
        // Only this connection's successful claim permits cleanup. A losing
        // replay or a pre-claim failure must not abandon another stream's work.
        let _: Result<Empty> = session
            .payments
            .call(ops::SERVE_ABANDON, &request_id(&session.terms.id))
            .await;
    }
    result
}

/// The name this seller priced the requested model under. Offers are
/// advertised under the priced (often runtime) name and buyers address the
/// model by its public ID, exactly as `peer_offer_for_model` resolves it, so
/// the seller sells under either: the name it showed on the mesh is the name
/// it charges for, without the operator having to price both.
fn priced_model<'a>(
    prices: &std::collections::BTreeMap<String, mesh_llm_payments_types::pricing::Pricing>,
    requested: &'a str,
    backend: &'a str,
) -> Option<&'a str> {
    [requested, backend]
        .into_iter()
        .find(|name| prices.contains_key(*name))
}

enum Admission {
    Admitted(Box<Session>),
    Refused(super::refusal::Refusal),
    Replayed(String),
}

async fn admit(
    node: &Node,
    payments: &Payments,
    peer: String,
    frame: Frame,
    targets: &ModelTargets,
) -> Result<Admission> {
    let Frame::Request {
        id,
        model,
        pricing,
        http,
    } = frame
    else {
        bail!("expected metered request");
    };
    uuid::Uuid::parse_str(&id).context("invalid request ID")?;
    let mut request = PaidRequest::parse(&http)?;
    ensure!(request.model == model, "model mismatch");
    // Prices and offers use the public model ID; the local backend is
    // registered, and must be addressed, under its internal name.
    let backend_model =
        crate::network::openai::ingress::served_model_for_public_id(node, targets, &model).await;
    let backend = Backend::resolve(node, targets, &backend_model, &model).await?;
    request.use_backend_model(&backend_model);
    let prices = node.advertised_payment_offers().await?;
    let priced = priced_model(&prices, &model, &backend_model)
        .with_context(|| format!("this seller has no price for {model}"))?
        .to_owned();
    let admitted: ServeAdmitResponse = payments
        .call(
            ops::SERVE_ADMIT,
            &ServeAdmitRequest {
                billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                request_id: id.clone(),
                buyer: peer.clone(),
                model: priced.clone(),
                pricing: pricing.clone(),
            },
        )
        .await?;
    let (deposit, deposit_msat, min_payment_msat) = match admitted {
        ServeAdmitResponse::Blocked { reason } => {
            tracing::warn!(buyer = %peer, model = %priced, %reason, "refused paid admission");
            return Ok(Admission::Refused(super::refusal::Refusal::DebtBlocked));
        }
        ServeAdmitResponse::Replayed { request_id } => {
            return Ok(Admission::Replayed(request_id));
        }
        ServeAdmitResponse::Admitted {
            min_payment_msat, ..
        } => (None, 0, min_payment_msat),
        ServeAdmitResponse::Deposit {
            invoice,
            min_payment_msat,
            ..
        } => {
            let amount = invoice.amount_msat.context("amountless deposit invoice")?;
            (Some(invoice), amount, min_payment_msat)
        }
    };
    let terms = AdmissionTerms {
        id,
        model,
        pricing,
        deposit_msat,
        min_payment_msat,
    };
    let (lifecycle, exchange_id) = provider_observers(node).await;
    let observations = lifecycle.observe(&terms);
    Ok(Admission::Admitted(Box::new(Session {
        payments: payments.clone(),
        terms,
        request,
        backend,
        owns_dispatch: deposit.is_none(),
        deposit,
        observations,
        exchange_id,
    })))
}

/// The provider lifecycle for this serving request, and the one fresh ID
/// naming the request on both of the provider's channels,
/// `payment.lifecycle.v1` and `openai.exchange.v1`, so a plugin subscribed to
/// both joins them exactly. The ID is minted only when a plugin subscribes to
/// either, and is never the private request (recovery) ID.
async fn provider_observers(node: &Node) -> (super::lifecycle::ProviderLifecycle, Option<String>) {
    let lifecycle = super::lifecycle::ProviderLifecycle::for_node(node).await;
    let subscribed = lifecycle.is_subscribed()
        || crate::network::openai::paid_exchange::PaidServedExchange::is_subscribed(node).await;
    let exchange_id = subscribed.then(|| uuid::Uuid::new_v4().to_string());
    (lifecycle.named(exchange_id.clone()), exchange_id)
}

async fn serve_admitted(
    node: &Node,
    session: &mut Session,
    payer: iroh::EndpointId,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    wire::write(
        writer,
        &Frame::Admission {
            terms: session.terms.clone(),
            invoice: session.deposit.clone(),
        },
    )
    .await?;
    let accepted = tokio::time::timeout(Duration::from_secs(60), wire::read(reader)).await??;
    ensure!(
        matches!(accepted, Frame::Accepted {}),
        "admission cancelled"
    );
    if confirm_dispatch(session).await.is_err() {
        // A duplicate may have won the claim while we awaited the deposit.
        // Even an uncertain provider reply grants no local ownership: recover
        // the durable outcome without dispatching or abandoning shared state.
        return recover(&session.payments, &session.terms.id, writer).await;
    }
    session.owns_dispatch = true;
    let outcome = dispatch(node, session, payer, reader, writer).await;
    let bill = match outcome {
        Some(usage) => {
            let bill: FinalBill = session
                .payments
                .call(
                    ops::SERVE_SETTLE,
                    &ServeSettleRequest {
                        request_id: session.terms.id.clone(),
                        usage,
                    },
                )
                .await?;
            session.observations.delivered(usage.completion_tokens);
            Some(bill)
        }
        None => {
            let _: Empty = session
                .payments
                .call(ops::SERVE_ABANDON, &request_id(&session.terms.id))
                .await?;
            None
        }
    };
    settle(session, bill, writer).await
}

/// Wait for the deposit (if any) to settle and claim the once-only dispatch.
/// Nothing runs on the backend before this returns.
async fn confirm_dispatch(session: &Session) -> Result<()> {
    let Some(invoice) = &session.deposit else {
        // Admitted without a deposit: dispatch was claimed at admission.
        return Ok(());
    };
    session.observations.invoice(0, invoice);
    let settled: SettledReceived = session
        .payments
        .call(ops::SERVE_DEPOSIT_CONFIRMED, &request_id(&session.terms.id))
        .await?;
    session.observations.received(0, invoice, &settled);
    Ok(())
}

/// Run the backend and forward its response. `None` abandons the request:
/// the backend reported no usage, failed, or the buyer cancelled.
async fn dispatch(
    node: &Node,
    session: &Session,
    payer: iroh::EndpointId,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
) -> Option<MeteredUsage> {
    let instance = match session.backend.local_port() {
        Some(port) => match node.begin_runtime_instance_request(port).await {
            Ok(guard) => guard,
            Err(_) => return None,
        },
        None => None,
    };
    let request_digest =
        crate::plugin::openai_exchange::request_body_digest(&session.request.body, None);
    let exchange = crate::network::openai::paid_exchange::PaidServedExchange::begin(
        node,
        &session.terms.model,
        session.exchange_id.as_deref(),
    )
    .await;
    let mut delivered = exchange.as_ref().map(|_| DeliveredCapture::default());
    let usage = match session
        .backend
        .connect(&session.request, &session.terms.id)
        .await
    {
        Ok(backend) => forward_backend(backend, reader, writer, delivered.as_mut()).await,
        Err(_) => None,
    };
    drop(instance);
    exchange_events::end_exchange(
        node,
        exchange.zip(delivered),
        exchange_events::paid_response_adapter(&session.request),
        usage.is_some(),
        &session.terms.model,
        request_digest.as_deref(),
        payer,
    )
    .await;
    usage
}

/// Forward the backend's response to the buyer, observing usage on the way.
/// The backend is observed independently of buyer transport completion: a
/// buyer that goes away mid-response still owes nothing (abandoned), and a
/// complete response is billed even when the last frame did not land.
async fn forward_backend(
    mut backend: impl AsyncRead + Unpin,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    mut delivered: Option<&mut DeliveredCapture>,
) -> Option<MeteredUsage> {
    let mut usage = ResponseUsage::default();
    let mut buffer = [0_u8; 16 * 1024];
    let cancelled = wait_for_cancel_or_invalid_frame(reader);
    tokio::pin!(cancelled);
    loop {
        let read = tokio::select! {
            read = tokio::time::timeout(BACKEND_STALL, backend.read(&mut buffer)) => read,
            _ = &mut cancelled => return reported(usage.interrupted_evidence()),
        };
        let count = match read {
            Ok(Ok(0)) => break,
            Ok(Ok(count)) => count,
            _ => return reported(usage.interrupted_evidence()),
        };
        usage.observe(&buffer[..count]);
        if let Some(capture) = delivered.as_deref_mut() {
            capture.push(&buffer[..count]);
        }
        let frame = Frame::Output {
            bytes: buffer[..count].to_vec(),
        };
        if wire::write(writer, &frame).await.is_err() {
            return reported(usage.interrupted_evidence());
        }
    }
    usage.finish();
    reported(usage.evidence())
}

fn reported(evidence: ResponseEvidence) -> Option<MeteredUsage> {
    match evidence {
        ResponseEvidence::Reported(usage) => Some(usage),
        ResponseEvidence::BackendError | ResponseEvidence::Missing => None,
    }
}

/// A clean send-half EOF is not cancellation: the buyer may still consume
/// the response. Invalid/truncated frames terminate conservatively, rather
/// than pretending the buyer sent Cancel. The response writer detects loss
/// of the receive half independently.
async fn wait_for_cancel_or_invalid_frame(reader: &mut (impl AsyncRead + Unpin)) -> Result<()> {
    let mut prefix = [0_u8; 1];
    match reader.read(&mut prefix).await {
        Ok(0) => std::future::pending::<Result<()>>().await,
        Ok(_) => {
            let mut framed = std::io::Cursor::new(prefix).chain(reader);
            ensure!(
                matches!(wire::read(&mut framed).await?, Frame::Cancel {}),
                "unexpected buyer frame during inference"
            );
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

/// Send the settlement, wait for the final invoice to be received and close.
async fn settle(
    session: &Session,
    bill: Option<FinalBill>,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    let frame = settlement_frame(&session.terms.id, bill.as_ref());
    wire::write(writer, &frame).await?;
    if let Some(bill) = bill {
        session.observations.final_amount(bill.bill_msat);
        if let Some(invoice) = bill.invoice {
            session.observations.invoice(1, &invoice);
            // Bounded by FINAL_SETTLEMENT_WAIT in the payments service. A
            // final that does not land in time stays an unpaid strike and a
            // late payment is recorded by reconciliation, so the exchange
            // completes either way.
            let settled: Result<SettledReceived> = session
                .payments
                .call(
                    ops::SETTLE_RECEIVED,
                    &mesh_llm_payments_types::contract::InvoiceRequest {
                        invoice: invoice.clone(),
                    },
                )
                .await;
            match settled {
                Ok(settled) => session.observations.received(1, &invoice, &settled),
                Err(error) => tracing::debug!(
                    request = %session.terms.id,
                    error = %format!("{error:#}"),
                    "final invoice not settled before completing; left to reconciliation"
                ),
            }
        }
    }
    wire::write(writer, &Frame::Complete {}).await
}

fn settlement_frame(id: &str, bill: Option<&FinalBill>) -> Frame {
    match bill {
        Some(bill) => Frame::Settlement {
            request_id: id.to_owned(),
            usage: Some(bill.usage),
            bill_msat: Some(bill.bill_msat),
            deposit_applied_msat: bill.deposit_msat.min(bill.bill_msat),
            invoice: bill.invoice.clone(),
        },
        None => Frame::Settlement {
            request_id: id.to_owned(),
            usage: None,
            bill_msat: None,
            deposit_applied_msat: 0,
            invoice: None,
        },
    }
}

/// A buyer refetches the outcome of a request after a disconnect or restart.
/// Never dispatches; a request still in progress answers `Pending`.
pub(super) async fn recover(
    payments: &Payments,
    id: &str,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    uuid::Uuid::parse_str(id)?;
    let response: ServeRecoverResponse = payments.call(ops::SERVE_RECOVER, &request_id(id)).await?;
    let bill = match response {
        ServeRecoverResponse::Pending => return wire::write(writer, &Frame::Pending {}).await,
        ServeRecoverResponse::Abandoned => None,
        ServeRecoverResponse::Billed { bill } => Some(bill),
    };
    // Recovery never waits for the payment: the buyer pays inside its own
    // settlement step and periodic reconciliation records it. Waiting here
    // held a seller task per Recover pass until invoice expiry.
    wire::write(writer, &settlement_frame(id, bill.as_ref())).await?;
    wire::write(writer, &Frame::Complete {}).await
}

fn request_id(id: &str) -> RequestIdRequest {
    RequestIdRequest {
        request_id: id.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_seller_sells_under_the_public_id_of_a_model_priced_by_runtime_name() {
        let price = mesh_llm_payments_types::pricing::Pricing {
            input_msat_per_million: 500,
            output_msat_per_million: 1500,
            cache_discount_percent: 0,
        };
        let runtime = "local-gguf/sha256-631aa3b9";
        let public = "bartowski/google_gemma-3-27b-it-GGUF@main:Q4_K_M";
        let by_runtime = [(runtime.to_owned(), price.clone())].into();
        assert_eq!(
            super::priced_model(&by_runtime, public, runtime),
            Some(runtime)
        );
        assert_eq!(
            super::priced_model(&by_runtime, runtime, runtime),
            Some(runtime)
        );
        let by_public = [(public.to_owned(), price)].into();
        assert_eq!(
            super::priced_model(&by_public, public, runtime),
            Some(public)
        );
        assert_eq!(super::priced_model(&by_public, "other", "other"), None);
    }

    use super::*;
    use std::task::Poll;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn send_half_eof_is_not_cancel_but_explicit_cancel_is() -> Result<()> {
        let mut eof = tokio::io::empty();
        let watcher = wait_for_cancel_or_invalid_frame(&mut eof);
        tokio::pin!(watcher);
        std::future::poll_fn(|cx| {
            assert!(matches!(
                std::future::Future::poll(watcher.as_mut(), cx),
                Poll::Pending
            ));
            Poll::Ready(())
        })
        .await;
        let mut bytes = Vec::new();
        wire::write(&mut bytes, &Frame::Cancel {}).await?;
        wait_for_cancel_or_invalid_frame(&mut bytes.as_slice()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn truncated_and_unexpected_buyer_frames_are_errors_not_cancel() -> Result<()> {
        let mut bytes = Vec::new();
        wire::write(&mut bytes, &Frame::Accepted {}).await?;
        assert!(
            wait_for_cancel_or_invalid_frame(&mut bytes.as_slice())
                .await
                .is_err()
        );
        let (mut send, mut recv) = tokio::io::duplex(8);
        send.write_all(&[0]).await?;
        send.shutdown().await?;
        assert!(wait_for_cancel_or_invalid_frame(&mut recv).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn a_complete_response_is_billed_even_when_the_buyer_stopped_reading() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 44\r\n\r\n{\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}";
        let (mut buyer_send, mut buyer_recv) = tokio::io::duplex(1 << 16);
        buyer_send.shutdown().await.unwrap();
        let mut sink = Vec::new();
        let usage = forward_backend(&response[..], &mut buyer_recv, &mut sink, None).await;
        assert_eq!(
            usage,
            Some(MeteredUsage {
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
            })
        );
    }

    #[tokio::test]
    async fn a_backend_that_reports_no_usage_abandons() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        let (_buyer_send, mut buyer_recv) = tokio::io::duplex(16);
        let mut sink = Vec::new();
        assert_eq!(
            forward_backend(&response[..], &mut buyer_recv, &mut sink, None).await,
            None
        );
    }

    #[tokio::test]
    async fn a_buyer_cancel_abandons_a_stalled_backend() {
        let (_backend_send, backend_recv) = tokio::io::duplex(16);
        let (mut buyer_send, mut buyer_recv) = tokio::io::duplex(64);
        wire::write(&mut buyer_send, &Frame::Cancel {})
            .await
            .unwrap();
        let mut sink = Vec::new();
        assert_eq!(
            forward_backend(backend_recv, &mut buyer_recv, &mut sink, None).await,
            None
        );
    }

    #[test]
    fn an_abandoned_settlement_carries_no_bill_and_a_billed_one_applies_the_deposit() {
        let Frame::Settlement {
            usage,
            bill_msat,
            deposit_applied_msat,
            invoice,
            ..
        } = settlement_frame("id", None)
        else {
            panic!("expected settlement");
        };
        assert!(usage.is_none() && bill_msat.is_none() && invoice.is_none());
        assert_eq!(deposit_applied_msat, 0);
        let bill = FinalBill {
            usage: MeteredUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
            },
            bill_msat: 10,
            deposit_msat: 30,
            remaining_msat: 0,
            settled: true,
            invoice: None,
        };
        let Frame::Settlement {
            deposit_applied_msat,
            ..
        } = settlement_frame("id", Some(&bill))
        else {
            panic!("expected settlement");
        };
        // A deposit exceeding the bill is applied only up to the bill.
        assert_eq!(deposit_applied_msat, 10);
    }
}
