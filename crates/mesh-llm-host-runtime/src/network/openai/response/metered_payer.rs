//! Local spending authority for the backend-neutral v2 exchange.
use crate::{mesh::Node, network::payments::request::PaidRequest};
use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments::{
    ledger::{Charge, RequestTerms},
    pricing::{Pricing, fee_allowance_msat, payment_cap_msat},
    service::PaymentService,
};
use mesh_llm_payments_types::{
    metered_wire::{self as wire, AdmissionTerms, Frame},
    metering::PromptEstimate,
};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};

pub(crate) async fn start(
    node: &Node,
    peer: iroh::EndpointId,
    raw: &[u8],
    pricing: Pricing,
    exchange_id: Option<&str>,
) -> Result<super::paid::StartedExchange> {
    let request = PaidRequest::parse(raw)?;
    let estimate = PromptEstimate::from_request(&request.body)?;
    ensure!(!estimate.has_media, "paid media unsupported");
    let output = u64::from(
        request
            .max_tokens
            .context("metered inference requires explicit max_tokens")?,
    );
    let service = node.payment_service().await?;
    let intent = super::paid::effective_intent(&service, &request)?;
    let mesh_llm_payments::intent::PaymentIntent::AllowPaid { max_total_msat, .. } = intent else {
        bail!("paid policy disabled")
    };
    let available = service.ledger.available_budget(
        service.wallet().await?.balance().await?.spendable_msat,
        mesh_llm_payments::now_ms(),
    )?;
    let max_total_msat = max_total_msat.min(available);
    let estimated_cost = pricing
        .input_charge(estimate.tokens)?
        .checked_add(pricing.output_charge(output)?)
        .context("cost overflow")?;
    let minimum = payment_cap_msat(estimated_cost)?
        .checked_add(fee_allowance_msat(estimated_cost)?)
        .context("cap overflow")?;
    ensure!(
        intent.permits(&pricing, minimum) && minimum <= max_total_msat,
        "estimated exposure exceeds buyer cap"
    );
    let id = uuid::Uuid::new_v4().to_string();
    let (mut send, mut recv) = node
        .open_http_tunnel(peer)
        .await
        .map_err(|_| super::paid::PrePaymentTransportFailure)?;
    send.write_all(wire::HTTP_UPGRADE)
        .await
        .map_err(|_| super::paid::PrePaymentTransportFailure)?;
    wire::write(
        &mut send,
        &Frame::Request {
            id: id.clone(),
            model: request.model.clone(),
            pricing: pricing.clone(),
            max_total_msat,
            http: request.backend_http(&id)?,
        },
    )
    .await?;
    // No payment-capable task before validating the admission. Never downgrade.
    let initial = tokio::time::timeout(Duration::from_secs(30), wire::read(&mut recv)).await??;
    if matches!(initial, Frame::Error { .. }) {
        return Err(super::paid::PrePaymentTransportFailure.into());
    }
    let Frame::Admission { terms, invoice } = initial else {
        bail!("provider did not accept metered v2")
    };
    validate_admission(
        &terms,
        invoice.as_ref(),
        &request,
        &pricing,
        &id,
        max_total_msat,
    )?;
    let payer_terms = RequestTerms {
        exchange_id: exchange_id.map(str::to_owned),
        id: id.clone(),
        peer: peer.to_string(),
        payee: invoice.as_ref().map(|i| i.payee.clone()),
        model: terms.model.clone(),
        pricing: pricing.clone(),
        input_tokens: terms.prompt_estimate,
        max_output_tokens: terms.max_output_tokens,
        max_total_msat,
        expires_at_ms: mesh_llm_payments::now_ms() + 3_600_000,
    };
    service.save_metered_authorization(&terms)?;
    service.await_authorization(&payer_terms).await?;
    let observations = super::paid_events::Observations::for_exchange(
        Some(&(node.clone(), exchange_id.unwrap_or_default().to_owned())),
        &payer_terms,
    )
    .await;
    observations.accepted(max_total_msat);
    let (pipe, mut output) = tokio::io::duplex(64 * 1024);
    let (ready, wait_ready) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let strike_node = node.clone();
    let payee = peer;
    tokio::spawn(async move {
        let mut progress = ExchangeProgress::default();
        let result = exchange(
            Exchange {
                service,
                payer: payer_terms,
                terms,
                deposit: invoice,
                observations,
                usage_filter: (!request.client_include_usage)
                    .then(super::metered_usage_filter::UsageFilter::default),
            },
            send,
            recv,
            &mut output,
            ready,
            cancellation,
            &mut progress,
        )
        .await;
        if result.is_err() && progress.paid_undelivered().await {
            record_paid_undelivered(&strike_node, payee).await;
        }
        if result.is_err() {
            tracing::warn!("metered inference interrupted; durable authorization retained");
        }
    });
    Ok((pipe, wait_ready, cancel))
}

fn validate_admission(
    terms: &AdmissionTerms,
    invoice: Option<&mesh_llm_payments::invoice::Invoice>,
    request: &PaidRequest,
    price: &Pricing,
    id: &str,
    cap: u64,
) -> Result<()> {
    let estimate = PromptEstimate::from_request(&request.body)?;
    ensure!(
        terms.id == id
            && terms.model == request.model
            && terms.pricing == *price
            && terms.prompt_estimate == estimate.tokens
            && terms.max_output_tokens
                == u64::from(request.max_tokens.context("missing output limit")?)
            && terms.max_total_msat == cap,
        "admission terms mismatch"
    );
    let deposit =
        u64::try_from((u128::from(price.input_charge(estimate.tokens)?) * 3).div_ceil(10))?;
    ensure!(
        terms.deposit_msat == 0 || terms.deposit_msat == deposit,
        "deposit mismatch"
    );
    match invoice {
        Some(invoice) => {
            ensure!(
                terms.deposit_msat > 0 && invoice.amount_msat == Some(terms.deposit_msat),
                "deposit invoice mismatch"
            );
            invoice.validate_payment(terms.deposit_msat, mesh_llm_payments::now_ms())?;
        }
        None => ensure!(terms.deposit_msat == 0, "deposit invoice missing"),
    }
    Ok(())
}

/// What a metered exchange had done when it ended, for the payee blocklist.
#[derive(Default)]
pub(super) struct ExchangeProgress {
    deposit: Option<tokio::task::JoinHandle<Result<()>>>,
    output_delivered: bool,
    cancelled: bool,
}

impl ExchangeProgress {
    /// The deposit is confirmed settled, the provider sent no output, and we
    /// did not cancel: the provider took payment and delivered nothing. An
    /// unsettled, failed or uncertain deposit is never proof of nondelivery.
    pub(super) async fn paid_undelivered(&mut self) -> bool {
        if self.output_delivered || self.cancelled {
            return false;
        }
        match self.deposit.take() {
            Some(payment) => matches!(payment.await, Ok(Ok(()))),
            None => false,
        }
    }
}

async fn record_paid_undelivered(node: &Node, peer: iroh::EndpointId) {
    let directory = node.config_state.lock().await.payment_directory();
    let payee = peer.to_string();
    let now = mesh_llm_payments::now_ms();
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

struct Exchange {
    service: Arc<PaymentService>,
    payer: RequestTerms,
    terms: AdmissionTerms,
    deposit: Option<mesh_llm_payments::invoice::Invoice>,
    observations: super::paid_events::Observations,
    /// Present when the caller did not request streamed usage.
    usage_filter: Option<super::metered_usage_filter::UsageFilter>,
}

async fn exchange(
    exchange: Exchange,
    mut send: impl AsyncWrite + Unpin,
    mut recv: impl AsyncRead + Unpin,
    output: &mut DuplexStream,
    ready: tokio::sync::oneshot::Sender<()>,
    mut cancellation: tokio::sync::watch::Receiver<bool>,
    progress: &mut ExchangeProgress,
) -> Result<()> {
    let Exchange {
        service,
        payer,
        terms,
        deposit,
        observations,
        mut usage_filter,
    } = exchange;
    if *cancellation.borrow() {
        service.ledger.cancel_unstarted(&terms.id)?;
        wire::write(&mut send, &Frame::Cancel {}).await?;
        bail!("client disconnected before dispatch");
    }
    // Acceptance permits a deposit submission, not dispatch. The provider waits
    // for terminal incoming settlement before it starts any backend work.
    wire::write(&mut send, &Frame::Accepted {}).await?;
    let input_payment = deposit.map(|invoice| {
        let service = service.clone();
        let payer = payer.clone();
        tokio::spawn(async move {
            pay(
                &service,
                &payer,
                0,
                terms.deposit_msat,
                invoice,
                &super::paid_events::Observations::default(),
            )
            .await
        })
    });
    let _ = ready.send(());
    progress.deposit = input_payment;
    let mut cancelled = false;
    let mut finalized = false;
    loop {
        let reading = wire::read(&mut recv);
        tokio::pin!(reading);
        let frame = tokio::select! {
            frame = &mut reading => frame?,
            _ = cancellation.changed(), if !cancelled => {
                cancelled = true;
                progress.cancelled = true;
                wire::write(&mut send, &Frame::Cancel {}).await?;
                reading.await?
            }
        };
        match frame {
            Frame::Output { bytes } => {
                ensure!(!finalized, "output after settlement");
                progress.output_delivered |= !bytes.is_empty();
                let bytes = match usage_filter.as_mut() {
                    Some(filter) => filter.push(&bytes),
                    None => bytes,
                };
                if !cancelled && output.write_all(&bytes).await.is_err() {
                    cancelled = true;
                    progress.cancelled = true;
                    wire::write(&mut send, &Frame::Cancel {}).await?;
                }
            }
            settlement @ Frame::Settlement { .. } => {
                ensure!(!finalized, "duplicate settlement");
                // HTTP delivery ends independently of final wallet completion.
                flush_filter(&mut usage_filter, output, cancelled).await;
                output.shutdown().await?;
                settle(&service, &payer, &terms, settlement, &observations).await?;
                finalized = true;
            }
            Frame::Complete {} => {
                ensure!(finalized, "completion before settlement");
                if let Some(payment) = progress.deposit.take() {
                    payment.await??;
                }
                service.metered_complete(&terms.id).await?;
                return Ok(());
            }
            Frame::Pending {} => {
                // Unknown usage keeps authorization.
                flush_filter(&mut usage_filter, output, cancelled).await;
                return Ok(());
            }
            _ => bail!("invalid metered provider frame"),
        }
    }
}

async fn flush_filter(
    filter: &mut Option<super::metered_usage_filter::UsageFilter>,
    output: &mut DuplexStream,
    cancelled: bool,
) {
    if let Some(filter) = filter.as_mut() {
        let rest = filter.finish();
        if !cancelled && !rest.is_empty() {
            let _ = output.write_all(&rest).await;
        }
    }
}

pub(super) async fn settle(
    service: &PaymentService,
    payer: &RequestTerms,
    terms: &AdmissionTerms,
    frame: Frame,
    observations: &super::paid_events::Observations,
) -> Result<()> {
    let Frame::Settlement {
        request_id,
        usage,
        bill_msat,
        deposit_applied_msat,
        credit_applied_msat,
        invoice,
        ..
    } = frame
    else {
        bail!("expected settlement")
    };
    ensure!(
        request_id == terms.id && payer.id == terms.id,
        "settlement ID mismatch"
    );
    let Some(usage) = usage else {
        ensure!(
            bill_msat.is_none()
                && invoice.is_none()
                && credit_applied_msat == 0
                && deposit_applied_msat == 0,
            "unknown usage cannot create charge"
        );
        return Ok(());
    };
    ensure!(
        usage.completion_tokens <= terms.max_output_tokens,
        "output allowance exceeded"
    );
    let bill = terms
        .pricing
        .input_charge(usage.prompt_tokens)?
        .checked_add(terms.pricing.output_charge(usage.completion_tokens)?)
        .context("bill overflow")?;
    ensure!(bill_msat == Some(bill), "backend bill mismatch");
    ensure!(
        deposit_applied_msat <= terms.deposit_msat && deposit_applied_msat <= bill,
        "invalid deposit application"
    );
    let owed = bill - deposit_applied_msat;
    ensure!(
        credit_applied_msat <= owed && credit_applied_msat <= terms.credit_msat,
        "unapproved credit application"
    );
    let remaining = owed - credit_applied_msat;
    match invoice {
        Some(invoice) => {
            ensure!(
                remaining > 0 && invoice.amount_msat == Some(remaining),
                "final invoice amount mismatch"
            );
            pay(service, payer, 1, remaining, invoice, observations).await?;
        }
        None => ensure!(remaining == 0, "final invoice missing"),
    }
    observations.final_amount(bill);
    Ok(())
}

async fn pay(
    service: &PaymentService,
    terms: &RequestTerms,
    segment: u32,
    amount: u64,
    invoice: mesh_llm_payments::invoice::Invoice,
    observations: &super::paid_events::Observations,
) -> Result<()> {
    ensure!(
        terms
            .payee
            .as_ref()
            .is_none_or(|payee| payee == &invoice.payee),
        "receiving wallet changed"
    );
    invoice.validate_payment(amount, mesh_llm_payments::now_ms())?;
    ensure!(
        invoice.amount_msat == Some(amount),
        "invoice amount mismatch"
    );
    service.metered_pin_payee(&terms.id, &invoice.payee).await?;
    observations.invoice(segment, &invoice);
    let payment = service
        .metered_pay_charge(&Charge {
            request_id: terms.id.clone(),
            segment,
            invoice,
            amount_msat: amount,
            max_total_msat: payment_cap_msat(amount)?,
        })
        .await?;
    observations.settled(segment, &payment);
    Ok(())
}

pub(super) async fn recover(
    node: &Node,
    service: &PaymentService,
    payer: &RequestTerms,
    terms: &AdmissionTerms,
    peer: iroh::EndpointId,
) -> Result<()> {
    let (mut send, mut recv) = node.open_http_tunnel(peer).await?;
    send.write_all(wire::HTTP_UPGRADE).await?;
    wire::write(
        &mut send,
        &Frame::Recover {
            id: terms.id.clone(),
        },
    )
    .await?;
    let mut settled = false;
    loop {
        match wire::read(&mut recv).await? {
            settlement @ Frame::Settlement { .. } => {
                settle(
                    service,
                    payer,
                    terms,
                    settlement,
                    &super::paid_events::Observations::default(),
                )
                .await?;
                settled = true;
            }
            Frame::Complete {} => {
                ensure!(settled, "completion before settlement");
                service.metered_complete(&terms.id).await?;
                return Ok(());
            }
            Frame::Pending {} => return Ok(()),
            _ => bail!("invalid metered recovery frame"),
        }
    }
}

#[cfg(test)]
mod strike_tests {
    use super::ExchangeProgress;

    fn progress(deposit: Option<anyhow::Result<()>>) -> ExchangeProgress {
        ExchangeProgress {
            deposit: deposit.map(|result| tokio::spawn(async move { result })),
            ..ExchangeProgress::default()
        }
    }

    #[tokio::test]
    async fn settled_deposit_without_output_is_a_strike() {
        assert!(progress(Some(Ok(()))).paid_undelivered().await);
    }

    #[tokio::test]
    async fn unsettled_or_absent_deposit_is_not_proof() {
        assert!(!progress(Some(Err(anyhow::anyhow!("uncertain")))).paid_undelivered().await);
        assert!(!progress(None).paid_undelivered().await);
    }

    #[tokio::test]
    async fn cancellation_or_any_output_is_not_a_strike() {
        let mut cancelled = progress(Some(Ok(())));
        cancelled.cancelled = true;
        assert!(!cancelled.paid_undelivered().await);
        let mut delivered = progress(Some(Ok(())));
        delivered.output_delivered = true;
        assert!(!delivered.paid_undelivered().await);
    }
}
