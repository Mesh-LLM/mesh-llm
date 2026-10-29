//! Seller-side coarse operations behind `payments.v1`: token-metered billing.
//! The host keeps the transport, the backend and delivery; these own every
//! ledger and wallet step of admitting, depositing, settling and recovering a
//! paid request. No inference or delivery gate lives here.

use std::time::Duration;

use anyhow::{Context, Result, ensure};
use mesh_llm_payments_types::contract::{
    FinalBill, ServeAdmitRequest, ServeAdmitResponse, ServeRecoverResponse, ServeSettleRequest,
    SettledReceived,
};

use crate::invoice::Invoice;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredSettlement, MeteredStatus};
use crate::ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind};
use crate::lifetimes::{DEPOSIT_INVOICE_EXPIRY_SECS, FINAL_INVOICE_EXPIRY_SECS};
use crate::service::{Arrival, PaymentService};
use crate::wallet::{PaymentStatus, Transaction};

/// Bound on the in-memory good-standing cache. Trust is an optimization, not
/// a durable registry; money and debt stay in the ledger.
const STANDING_CACHE: usize = 1024;

/// How long admission waits for a buyer's in-flight final payments to land
/// before counting them as strikes. Well inside the buyer's admission timeout.
#[cfg(not(test))]
const STRIKE_SETTLEMENT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
#[cfg(test)]
const STRIKE_SETTLEMENT_WAIT: std::time::Duration = std::time::Duration::from_millis(300);
const STRIKE_SETTLEMENT_POLL: std::time::Duration = std::time::Duration::from_millis(100);

impl PaymentService {
    /// Unpaid final invoices for `buyer`, after giving ones still in flight a
    /// short chance to settle. A buyer sending requests back to back pays each
    /// final a second or two after its response, so the newest finals are
    /// normally unpaid only because Lightning has not finished yet. Refusing
    /// at that moment would turn ordinary sequential traffic into spurious
    /// refusals. A buyer that really stopped paying is still refused once the
    /// wait ends.
    async fn settled_strikes(&self, buyer: &str) -> Result<u64> {
        let deadline = tokio::time::Instant::now() + STRIKE_SETTLEMENT_WAIT;
        loop {
            let strikes = self.ledger.metered_unpaid_finals(buyer)?;
            if strikes < crate::ledger::metered::STRIKE_LIMIT
                || tokio::time::Instant::now() >= deadline
            {
                return Ok(strikes);
            }
            tokio::time::sleep(STRIKE_SETTLEMENT_POLL).await;
        }
    }

    /// Admit one request at the seller's current prices. Buyers at the strike
    /// limit are refused; buyers in good standing are admitted without a
    /// deposit and dispatch is claimed here; everyone else is quoted the
    /// operator's flat deposit. Replaying the same terms before
    /// dispatch returns the same answer; a request already dispatched or
    /// finalized is reported as replayed: never a second dispatch.
    pub async fn serve_admit(&self, request: ServeAdmitRequest) -> Result<ServeAdmitResponse> {
        let ServeAdmitRequest {
            billing_revision: _,
            request_id,
            buyer,
            model,
            pricing,
        } = request;
        if let Some(owner) = self.ledger.metered_owner(&request_id)? {
            ensure!(owner == buyer, "request belongs to another buyer");
            if self.ledger.metered_past_admission(&request_id)? {
                return Ok(ServeAdmitResponse::Replayed { request_id });
            }
        }
        ensure!(
            self.ledger.pricing()?.get(&model) == Some(&pricing),
            "seller prices changed"
        );
        let strikes = self.settled_strikes(&buyer).await?;
        if strikes >= crate::ledger::metered::STRIKE_LIMIT {
            return Ok(ServeAdmitResponse::Blocked {
                reason: format!("buyer has {strikes} unpaid final invoices"),
            });
        }
        let require_deposit = !self.metered_trusted(&buyer).await?;
        let deposit = self.ledger.metered_admit(&MeteredRequest {
            id: &request_id,
            peer: &buyer,
            model: &model,
            pricing: &pricing,
            require_deposit,
        })?;
        // The minimum recorded at admission, so a replay restates the same
        // terms the buyer saved even if the operator has changed it since.
        let min_payment_msat = self.ledger.metered_min_payment_msat(&request_id)?;
        if deposit == 0 {
            ensure!(
                self.ledger.metered_claim_dispatch(&request_id, &buyer)?,
                "request already dispatched"
            );
            return Ok(ServeAdmitResponse::Admitted {
                request_id,
                min_payment_msat,
            });
        }
        let invoice = self
            .metered_invoice(&request_id, &buyer, MeteredInvoiceKind::Deposit, deposit)
            .await?
            .context("deposit invoice missing")?;
        Ok(ServeAdmitResponse::Deposit {
            request_id,
            invoice,
            min_payment_msat,
        })
    }

    /// Wait for the admitted deposit to settle in the receiving wallet, record
    /// it paid and claim the once-only dispatch. Bounded by the deposit
    /// invoice's lifetime.
    pub async fn serve_deposit_confirmed(&self, request_id: &str) -> Result<SettledReceived> {
        let receipt = self
            .ledger
            .metered_invoice(request_id, MeteredInvoiceKind::Deposit)?
            .context("no deposit was requested for this request")?;
        let payment = self.settle_received(&receipt.invoice).await?;
        ensure!(
            self.ledger
                .metered_claim_dispatch(request_id, &receipt.peer)?,
            "request already dispatched"
        );
        Ok(settled(&payment))
    }

    /// Bill the backend-reported usage at the admitted rates, apply the paid
    /// deposit and issue the final invoice for any remainder. Idempotent:
    /// a retry returns the stored bill and the same invoice.
    pub async fn serve_settle(&self, request: ServeSettleRequest) -> Result<FinalBill> {
        let ServeSettleRequest { request_id, usage } = request;
        let settlement = self
            .ledger
            .metered_finalize(&request_id, MeteredOutcome::Usage(usage))?;
        self.final_bill(&request_id, settlement).await
    }

    /// Close a request without usage: the deposit stays received, no bill, no
    /// invoice, no debt, no strike. Terminal and idempotent.
    pub fn serve_abandon(&self, request_id: &str) -> Result<()> {
        self.ledger
            .metered_finalize(request_id, MeteredOutcome::Abandoned)?;
        Ok(())
    }

    /// A buyer refetches the outcome of a request after a disconnect or
    /// restart. Refreshes payment facts from the wallet; never dispatches.
    pub async fn serve_recover(&self, request_id: &str) -> Result<ServeRecoverResponse> {
        self.ledger
            .metered_owner(request_id)?
            .context("unknown metered request")?;
        self.metered_refresh(request_id).await?;
        let settlement = self.ledger.metered_settlement(request_id)?;
        Ok(match settlement.status {
            MeteredStatus::Admitted => ServeRecoverResponse::Pending,
            MeteredStatus::Abandoned => ServeRecoverResponse::Abandoned,
            MeteredStatus::Billed | MeteredStatus::Settled => ServeRecoverResponse::Billed {
                bill: self.final_bill(request_id, settlement).await?,
            },
        })
    }

    /// Wait for `invoice` to settle and record it received; returns the
    /// receiving wallet's record of the payment.
    pub async fn settle_received(&self, invoice: &Invoice) -> Result<Transaction> {
        let payment = self.wait_received(invoice).await?;
        self.record_received(&invoice.payment_hash).await?;
        Ok(payment)
    }

    /// [`Self::settle_received`] that gives up after `deadline`. Giving up
    /// records nothing: the invoice stays `unpaid` (a strike for a final) and
    /// periodic reconciliation still records a payment that lands later.
    pub async fn settle_received_within(
        &self,
        invoice: &Invoice,
        deadline: Duration,
    ) -> Result<Transaction> {
        let payment = self.wait_received_within(invoice, deadline).await?;
        self.record_received(&invoice.payment_hash).await?;
        Ok(payment)
    }

    /// Record a metered invoice paid; a paid final invoice puts its buyer in
    /// good standing for the next admission.
    async fn record_received(&self, hash: &str) -> Result<()> {
        if let Some(receipt) = self.ledger.mark_metered_received(hash)?
            && receipt.kind == MeteredInvoiceKind::Final
            && self.ledger.metered_settlement(&receipt.request_id)?.status == MeteredStatus::Settled
        {
            self.note_settled(&receipt.peer).await;
        }
        Ok(())
    }

    pub async fn arrival(&self, invoice: &Invoice, deadline: Duration) -> Result<bool> {
        Ok(self.wait_arrival(invoice, deadline).await? == Arrival::Claiming)
    }

    async fn final_bill(
        &self,
        request_id: &str,
        settlement: MeteredSettlement,
    ) -> Result<FinalBill> {
        let usage = settlement.usage.context("request was abandoned")?;
        let bill_msat = settlement.bill_msat.context("request was abandoned")?;
        let peer = self
            .ledger
            .metered_owner(request_id)?
            .context("unknown metered request")?;
        let invoice = if settlement.remaining_msat > 0 {
            self.metered_invoice(
                request_id,
                &peer,
                MeteredInvoiceKind::Final,
                settlement.remaining_msat,
            )
            .await?
        } else {
            None
        };
        let settled = self.ledger.metered_settlement(request_id)?.status == MeteredStatus::Settled;
        if settled {
            self.note_settled(&peer).await;
        }
        Ok(FinalBill {
            usage,
            bill_msat,
            deposit_msat: settlement.deposit_paid_msat,
            remaining_msat: settlement.remaining_msat,
            settled,
            invoice,
        })
    }

    /// Serialize invoice creation and reuse the exact persisted invoice on
    /// retry, so a request never carries two invoices of one kind.
    async fn metered_invoice(
        &self,
        request_id: &str,
        peer: &str,
        kind: MeteredInvoiceKind,
        amount: u64,
    ) -> Result<Option<Invoice>> {
        if amount == 0 {
            return Ok(None);
        }
        let _guard = self.invoice_lock.lock().await;
        if let Some(receipt) = self.ledger.metered_invoice(request_id, kind)? {
            ensure!(
                receipt.peer == peer && receipt.invoice.amount_msat == Some(amount),
                "invoice terms changed"
            );
            return Ok(Some(receipt.invoice));
        }
        let expiry = match kind {
            MeteredInvoiceKind::Deposit => DEPOSIT_INVOICE_EXPIRY_SECS,
            MeteredInvoiceKind::Final => FINAL_INVOICE_EXPIRY_SECS,
        };
        let invoice = self
            .wallet()
            .await?
            .create_invoice(Some(amount), expiry)
            .await?;
        ensure!(
            invoice.amount_msat == Some(amount),
            "wallet changed invoice amount"
        );
        self.ledger.record_metered_invoice(&MeteredInvoice {
            request_id: request_id.into(),
            peer: peer.into(),
            kind,
            invoice: invoice.clone(),
            paid: false,
        })?;
        Ok(Some(invoice))
    }

    /// Refresh incoming facts for one request from the wallet. Observes only;
    /// never treats expiry as settlement or failure.
    pub async fn metered_refresh(&self, request_id: &str) -> Result<()> {
        for receipt in self.ledger.metered_invoices(Some(request_id))? {
            if !receipt.paid {
                self.refresh_invoice(&receipt).await?;
            }
        }
        Ok(())
    }

    /// Refresh every unpaid metered invoice. Used by periodic reconciliation
    /// so a payment observed late still clears a strike.
    pub async fn metered_refresh_all(&self) -> Result<()> {
        let mut first_error = None;
        for receipt in self.ledger.unpaid_metered_invoices()? {
            if let Err(error) = self.refresh_invoice(&receipt).await {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    async fn refresh_invoice(&self, receipt: &MeteredInvoice) -> Result<()> {
        let Some(payment) = self
            .wallet()
            .await?
            .lookup(&receipt.invoice.payment_hash)
            .await?
        else {
            return Ok(());
        };
        crate::service::validate_payment_update(&payment, &receipt.invoice.payment_hash, true)?;
        if payment.status == PaymentStatus::Succeeded {
            ensure!(
                crate::service::incoming_covers(
                    &payment,
                    receipt.invoice.amount_msat.context("amountless invoice")?,
                ),
                "incoming underpayment"
            );
            self.record_received(&receipt.invoice.payment_hash).await?;
        }
        Ok(())
    }

    /// Whether `peer` may be admitted without a deposit: it has settled a
    /// request in this process's lifetime. Outstanding finals are enforced
    /// separately by the admission strike limit, not by making a peer cold.
    /// Seller-owned and disposable; debt itself is durable.
    pub async fn metered_trusted(&self, peer: &str) -> Result<bool> {
        Ok(self.metered_standing.lock().await.contains(peer))
    }

    async fn note_settled(&self, peer: &str) {
        let mut standing = self.metered_standing.lock().await;
        if standing.len() >= STANDING_CACHE {
            standing.clear();
        }
        standing.insert(peer.to_owned());
    }
}

fn settled(payment: &Transaction) -> SettledReceived {
    // A wallet that reports no amount reports 0; never claim a 0 credit.
    SettledReceived {
        credited_msat: (payment.amount_msat > 0).then_some(payment.amount_msat),
        fee_msat: Some(payment.fee_msat),
    }
}
