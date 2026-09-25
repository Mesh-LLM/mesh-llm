//! Seller-side coarse operations behind `payments.v1`. The host keeps the
//! transport, the decode gate and delivery; these own every ledger and wallet
//! step of serving a paid request.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use mesh_llm_payments_types::contract::{
    InputInvoiceResponse, OutputInvoice, ServeBeginRequest, ServeInputInvoiceRequest,
    ServeRecoverResponse,
};

use crate::invoice::Invoice;
use crate::ledger::{RequestTerms, receivables::Receivable};
use crate::lifetimes::INPUT_INVOICE_EXPIRY_SECS;
use crate::service::{Arrival, PaymentService};
use crate::wallet::PaymentStatus;

impl PaymentService {
    /// Check the offered price, wake the receiving wallet, wait out the
    /// peer's prior debt, then durably open serving.
    pub async fn serve_begin(self: &Arc<Self>, request: ServeBeginRequest) -> Result<()> {
        ensure!(
            self.ledger.pricing()?.get(&request.model) == Some(&request.pricing),
            "seller prices changed"
        );
        // Wake the receiving wallet during prefill so the input invoice
        // doesn't wait on a cold start.
        drop(self.prefetch_balance());
        self.await_prior_settlement(
            &request.peer,
            Duration::from_millis(request.prior_settlement_ms),
        )
        .await?;
        self.ledger.begin_serving(
            &request.id,
            &request.peer,
            &request.pricing,
            request.max_output,
        )
    }

    /// Fix the backend's output allowance, then create and durably record the
    /// input invoice.
    pub async fn serve_input_invoice(
        &self,
        request: ServeInputInvoiceRequest,
    ) -> Result<InputInvoiceResponse> {
        let ServeInputInvoiceRequest {
            id,
            peer,
            model,
            pricing,
            input_tokens,
            max_output_tokens,
        } = request;
        ensure!(
            input_tokens > 0 && input_tokens <= 131_072,
            "paid input limit exceeded"
        );
        self.ledger
            .resolve_serving_output_allowance(&id, max_output_tokens)?;
        let amount = pricing.input_charge(input_tokens)?;
        let max_total_msat = pricing.request_cap_msat(amount, max_output_tokens)?;
        let invoice = self
            .wallet()
            .await?
            .create_invoice(Some(amount), INPUT_INVOICE_EXPIRY_SECS)
            .await?;
        self.ledger.record_receivable(&Receivable {
            request_id: id.clone(),
            peer: peer.clone(),
            segment: 0,
            invoice: invoice.clone(),
            tokens: input_tokens,
            paid: false,
        })?;
        Ok(InputInvoiceResponse {
            terms: RequestTerms {
                exchange_id: None,
                id,
                peer,
                payee: Some(invoice.payee.clone()),
                model,
                pricing,
                input_tokens,
                max_output_tokens,
                max_total_msat,
                expires_at_ms: invoice.expires_at_ms,
            },
            invoice,
        })
    }

    /// Wait for `invoice` to settle and record it received.
    pub async fn settle_received(&self, invoice: &Invoice) -> Result<()> {
        self.wait_received(invoice).await?;
        self.ledger.mark_received(&invoice.payment_hash)
    }

    pub async fn arrival(&self, invoice: &Invoice, deadline: Duration) -> Result<bool> {
        Ok(self.wait_arrival(invoice, deadline).await? == Arrival::Claiming)
    }

    pub async fn output_invoice(&self, id: &str) -> Result<Option<OutputInvoice>> {
        Ok(self
            .output_receivable(id)
            .await?
            .map(|receipt| OutputInvoice {
                tokens: receipt.tokens,
                invoice: receipt.invoice,
            }))
    }

    /// A completed HTTP body can precede its trailing output payment. Wait
    /// before starting another backend, without forgiving debt or granting
    /// additional credit.
    pub async fn await_prior_settlement(&self, peer: &str, deadline: Duration) -> Result<()> {
        tokio::time::timeout(deadline, async {
            loop {
                if !self.ledger.has_outstanding_payment(peer)? {
                    return Ok(());
                }
                self.refresh_receivables(peer).await?;
                if !self.ledger.has_outstanding_payment(peer)? {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .context("prior payment settlement deadline exceeded")?
    }

    async fn refresh_receivables(&self, peer: &str) -> Result<()> {
        // Records paid input, or lapses abandoned zero-delivery input, so a
        // slow or interrupted payment does not keep this peer blocked.
        for id in self.ledger.unpaid_input_requests(peer)? {
            self.input_received(&id).await?;
        }
        let unpaid = self.ledger.unpaid_invoices(peer)?;
        if unpaid.is_empty() {
            return Ok(());
        }
        let wallet = self.wallet().await?;
        for invoice in unpaid {
            if wallet
                .lookup(&invoice.payment_hash)
                .await?
                .is_some_and(|p| {
                    p.inbound
                        && p.payment_hash.as_deref() == Some(invoice.payment_hash.as_str())
                        && p.status == PaymentStatus::Succeeded
                })
            {
                self.ledger.mark_received(&invoice.payment_hash)?;
            }
        }
        Ok(())
    }

    /// The random request ID is a bearer recovery capability, allowing
    /// recovery when the client's ephemeral mesh identity changes. The caller
    /// validates the ID's shape.
    pub async fn serve_recover(&self, id: &str) -> Result<ServeRecoverResponse> {
        let (_, _, _, finished) = self.ledger.serving_account(id)?;
        if !finished {
            return Ok(ServeRecoverResponse::Pending);
        }
        let receipts = self.ledger.receivables(Some(id))?;
        if receipts.iter().any(|r| r.segment == 0) && !self.input_received(id).await? {
            return Ok(ServeRecoverResponse::Pending);
        }
        Ok(ServeRecoverResponse::Complete {
            output: self.output_invoice(id).await?,
        })
    }
}
