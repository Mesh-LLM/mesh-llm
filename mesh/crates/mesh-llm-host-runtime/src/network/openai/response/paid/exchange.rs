//! The accepted metered exchange: approve, pay the deposit while the seller
//! waits for it, relay output, check the settlement and pay the final
//! invoice. Runs detached from the HTTP response so wallet completion never
//! depends on the application still listening.

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments_types::RequestTerms;
use mesh_llm_payments_types::contract::{
    AcceptAdmissionRequest, CancelStage, Empty, IdRequest, PayDepositRequest, SettleFinalRequest,
    SettleFinalResponse, ops,
};
use mesh_llm_payments_types::metered_wire::{self as wire, AdmissionTerms, Frame};
use mesh_llm_wallet::invoice::Invoice;
use mesh_llm_wallet::provider::Transaction;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};

use super::ExchangeProgress;
use crate::mesh::Node;
use crate::network::openai::response::metered_usage_filter::UsageFilter;
use crate::network::payments::{client::Payments, lifecycle::Observations};

/// A validated admission, not yet approved or paid.
pub(super) struct Accepted {
    pub(super) payments: Payments,
    /// The buyer's local terms: its cap, never sent to the seller.
    pub(super) local: RequestTerms,
    pub(super) terms: AdmissionTerms,
    pub(super) deposit: Option<Invoice>,
    /// Present when the caller did not ask for streamed usage.
    pub(super) usage_filter: Option<UsageFilter>,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    accepted: Accepted,
    mut send: impl AsyncWrite + Unpin,
    mut recv: impl AsyncRead + Unpin,
    output: &mut DuplexStream,
    ready: tokio::sync::oneshot::Sender<()>,
    mut cancellation: tokio::sync::watch::Receiver<bool>,
    evidence: Option<(Node, String)>,
    progress: &mut ExchangeProgress,
) -> Result<()> {
    let Accepted {
        payments,
        local,
        terms,
        deposit,
        mut usage_filter,
    } = accepted;
    approve(&payments, &local, &terms, &mut send, &mut cancellation).await?;
    let observations = Observations::for_exchange(evidence.as_ref(), &terms).await;
    observations.accepted(local.max_total_msat);
    // Acceptance permits a deposit submission, not dispatch: the seller waits
    // for its receiving wallet before it starts any backend work.
    wire::write(&mut send, &Frame::Accepted {}).await?;
    let mut deposit_payment = deposit.map(|invoice| {
        observations.invoice(0, &invoice);
        let payments = payments.clone();
        let request = PayDepositRequest {
            request_id: terms.id.clone(),
            invoice,
        };
        tokio::spawn(async move {
            payments
                .call::<_, Transaction>(ops::PAY_DEPOSIT, &request)
                .await
        })
    });
    let _ = ready.send(());
    let mut deposit_pending = deposit_payment.is_some();
    let mut watching = true;
    let mut cancelled = false;
    let mut settled = false;
    loop {
        let reading = wire::read(&mut recv);
        tokio::pin!(reading);
        let frame = loop {
            tokio::select! {
                result = async {
                    match deposit_payment.as_mut() {
                        Some(task) => task.await,
                        None => std::future::pending().await,
                    }
                }, if deposit_pending => {
                    deposit_pending = false;
                    let payment = result.context("deposit payment task failed")??;
                    observations.settled(0, &payment);
                    progress.deposit_settled = true;
                }
                frame = &mut reading => break frame?,
                changed = cancellation.changed(), if watching => {
                    // A dropped sender is the application side finishing
                    // after the response ended, not a cancellation.
                    watching = false;
                    if changed.is_ok() && *cancellation.borrow() {
                        cancelled = true;
                        progress.cancelled = true;
                        wire::write(&mut send, &Frame::Cancel {}).await?;
                    }
                }
            }
        };
        match frame {
            Frame::Output { bytes } => {
                ensure!(!settled, "output after settlement");
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
                ensure!(!settled, "duplicate settlement");
                // HTTP delivery ends independently of final wallet completion.
                flush_filter(&mut usage_filter, output, cancelled).await;
                let _ = output.shutdown().await;
                settle(&payments, &terms, settlement, &observations).await?;
                settled = true;
            }
            Frame::Complete {} => {
                ensure!(settled, "completion before settlement");
                if deposit_pending && let Some(payment) = deposit_payment.take() {
                    let payment = payment.await.context("deposit payment task failed")??;
                    observations.settled(0, &payment);
                    progress.deposit_settled = true;
                }
                let _: Empty = payments
                    .call(
                        ops::FINISH,
                        &IdRequest {
                            id: terms.id.clone(),
                        },
                    )
                    .await?;
                // Closing the exchange does not make it a delivery: a seller
                // that ends it having sent no output kept the deposit and
                // returned nothing, which is the abuse the blocklist is for.
                ensure!(
                    cancelled || progress.output_delivered,
                    "seller completed the exchange without delivering output"
                );
                return Ok(());
            }
            _ => bail!("invalid metered seller frame"),
        }
    }
}

/// Durably approve the admission against policy and balance.
async fn approve(
    payments: &Payments,
    local: &RequestTerms,
    terms: &AdmissionTerms,
    send: &mut (impl AsyncWrite + Unpin),
    cancellation: &mut tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let accept = AcceptAdmissionRequest {
        terms: local.clone(),
        admission: terms.clone(),
    };
    tokio::select! {
        result = payments.call::<_, Empty>(ops::ACCEPT_ADMISSION, &accept) => { result?; }
        _ = cancellation.changed() => {
            super::cancel(payments, &terms.id, CancelStage::Unstarted).await;
            let _ = wire::write(send, &Frame::Cancel {}).await;
            bail!("application disconnected before approval");
        }
    }
    Ok(())
}

async fn flush_filter(
    filter: &mut Option<UsageFilter>,
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

/// Hand the seller's settlement to the provider, which checks it against the
/// stored admission and pays the final invoice, if any.
pub(crate) async fn settle(
    payments: &Payments,
    terms: &AdmissionTerms,
    frame: Frame,
    observations: &Observations,
) -> Result<()> {
    let Frame::Settlement {
        request_id,
        usage,
        bill_msat,
        deposit_applied_msat,
        invoice,
    } = frame
    else {
        bail!("expected settlement");
    };
    ensure!(request_id == terms.id, "settlement ID mismatch");
    if let Some(invoice) = &invoice {
        observations.invoice(1, invoice);
    }
    let settled: SettleFinalResponse = payments
        .call(
            ops::SETTLE_FINAL,
            &SettleFinalRequest {
                request_id,
                usage,
                bill_msat,
                deposit_applied_msat,
                invoice,
            },
        )
        .await?;
    if let Some(payment) = &settled.payment {
        observations.settled(1, payment);
    }
    observations.final_amount(settled.bill_msat);
    Ok(())
}
