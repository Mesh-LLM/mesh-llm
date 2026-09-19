use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use mesh_llm_payments::{
    ledger::{RequestTerms, receivables::Receivable},
    pricing::{FEE_ALLOWANCE_MSAT, Pricing},
    service::PaymentService,
    wire::Frame,
};
use skippy_server::frontend::generation_gate::GenerationGate;
use tokio::sync::mpsc;

pub(super) struct InvoiceGate {
    pub service: Arc<PaymentService>,
    pub request_id: String,
    pub peer: String,
    pub model: String,
    pub pricing: Pricing,
    pub max_tokens: Option<u32>,
    pub events: mpsc::UnboundedSender<Frame>,
    pub runtime: tokio::runtime::Handle,
    pub authorized: AtomicBool,
    pub cancelled: AtomicBool,
    pub output_tokens: AtomicU64,
}

impl InvoiceGate {
    async fn authorize(&self, input: usize, output: u32) -> Result<()> {
        ensure!(
            output > 0 && output <= self.max_tokens.unwrap_or(u32::MAX),
            "backend exceeded output allowance"
        );
        ensure!(input > 0 && input <= 131_072, "paid input limit exceeded");
        self.service
            .ledger
            .resolve_serving_output_allowance(&self.request_id, u64::from(output))?;
        let amount = self.pricing.input_charge(input as u64)?;
        let max_total_msat = amount
            .checked_add(self.pricing.output_charge(u64::from(output))?)
            .and_then(|total| total.checked_add(2 * FEE_ALLOWANCE_MSAT))
            .context("request price overflow")?;
        let invoice = self
            .service
            .wallet()
            .await?
            .create_invoice(Some(amount))
            .await?;
        self.service.ledger.record_receivable(&Receivable {
            request_id: self.request_id.clone(),
            peer: self.peer.clone(),
            segment: 0,
            invoice: invoice.clone(),
            tokens: input as u64,
            paid: false,
        })?;
        self.events
            .send(Frame::InputInvoice {
                terms: RequestTerms {
                    id: self.request_id.clone(),
                    peer: self.peer.clone(),
                    payee: Some(invoice.payee.clone()),
                    model: self.model.clone(),
                    pricing: self.pricing.clone(),
                    input_tokens: input as u64,
                    max_output_tokens: u64::from(output),
                    max_total_msat,
                    expires_at_ms: invoice.expires_at_ms,
                },
                invoice: invoice.clone(),
            })
            .context("payment transport closed")?;
        tokio::select! {
            received = self.service.wait_received(&invoice) => received?,
            _ = async {
                while !self.cancelled.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => anyhow::bail!("request cancelled"),
        }
        self.service.ledger.mark_received(&invoice.payment_hash)?;
        ensure!(!self.cancelled.load(Ordering::Acquire), "request cancelled");
        self.authorized.store(true, Ordering::Release);
        Ok(())
    }
}

impl GenerationGate for InvoiceGate {
    fn after_prefill(&self, input: usize, output: u32) -> openai_frontend::OpenAiResult<()> {
        self.runtime
            .block_on(self.authorize(input, output))
            .map_err(|_| {
                openai_frontend::OpenAiError::backend("inference payment was not authorized")
            })
    }

    fn before_token(&self) -> openai_frontend::OpenAiResult<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(openai_frontend::OpenAiError::backend(
                "paid request cancelled",
            ));
        }
        Ok(())
    }

    fn committed_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Acquire)
    }

    fn committed_token(&self) -> openai_frontend::OpenAiResult<()> {
        self.output_tokens.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}
