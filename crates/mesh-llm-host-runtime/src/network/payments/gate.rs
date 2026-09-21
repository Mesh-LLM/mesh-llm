use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use mesh_llm_payments::{
    ledger::{RequestTerms, receivables::Receivable},
    pricing::{FEE_ALLOWANCE_MSAT, Pricing},
    service::{Arrival, PaymentService},
    wire::Frame,
};
use skippy_server::frontend::generation_gate::GenerationGate;
use tokio::sync::mpsc;

pub(super) enum GateEvent {
    InputInvoice(Box<Frame>),
    Opened,
    Failed,
}

pub(super) struct InvoiceGate {
    pub service: Arc<PaymentService>,
    pub request_id: String,
    pub peer: String,
    pub model: String,
    pub pricing: Pricing,
    pub max_tokens: Option<u32>,
    pub events: mpsc::UnboundedSender<GateEvent>,
    pub runtime: tokio::runtime::Handle,
    pub authorized: Arc<AtomicBool>,
    pub cancelled: Arc<AtomicBool>,
    pub started: AtomicBool,
    pub output_tokens: AtomicU64,
    /// Settles the input payment in the ledger once the receiver observes a
    /// completed payment. Started when output delivery opens, awaited before
    /// the request completes.
    pub input_settlement: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
}

impl InvoiceGate {
    /// Wait for the input payment the delivery gate opened on to settle.
    /// Between `claiming` and this point the provider carries the risk.
    pub(super) async fn await_input_settlement(&self) -> Result<()> {
        let Some(handle) = self.input_settlement.lock().await.take() else {
            return Ok(());
        };
        handle.await.context("input settlement task failed")?
    }

    fn prepare_authorization(&self, input: usize, output: u32) -> Result<Authorization> {
        ensure!(
            !self.started.swap(true, Ordering::AcqRel),
            "payment authorization already started"
        );
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
        Ok(Authorization {
            service: self.service.clone(),
            request_id: self.request_id.clone(),
            peer: self.peer.clone(),
            model: self.model.clone(),
            pricing: self.pricing.clone(),
            input: input as u64,
            output,
            amount,
            max_total_msat,
            events: self.events.clone(),
            authorized: self.authorized.clone(),
            cancelled: self.cancelled.clone(),
            input_settlement: self.input_settlement.clone(),
            stalled: std::time::Instant::now(),
        })
    }

    fn spawn_authorization(&self, authorization: Authorization) {
        self.runtime.spawn(async move {
            // Prefill has completed and decode now runs concurrently. This
            // span is the payment-added delay before buffered output may be
            // released, not a decode stall.
            let result = authorization.authorize().await;
            let opened_on = result
                .as_ref()
                .ok()
                .map(|arrival: &Arrival| arrival.as_str());
            let opened = result.is_ok();
            tracing::debug!(
                target: "mesh_llm::payments::timing",
                phase = "delivery_gate_wait",
                ms = authorization.stalled.elapsed().as_millis() as u64,
                opened,
                opened_on,
                "delivery gate"
            );
            let event = if opened {
                GateEvent::Opened
            } else {
                GateEvent::Failed
            };
            if authorization.events.send(event).is_err() {
                authorization.cancelled.store(true, Ordering::Release);
            }
        });
    }
}

struct Authorization {
    service: Arc<PaymentService>,
    request_id: String,
    peer: String,
    model: String,
    pricing: Pricing,
    input: u64,
    output: u32,
    amount: u64,
    max_total_msat: u64,
    events: mpsc::UnboundedSender<GateEvent>,
    authorized: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    input_settlement: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
    stalled: std::time::Instant,
}

impl Authorization {
    async fn authorize(&self) -> Result<Arrival> {
        let invoice = self
            .service
            .wallet()
            .await?
            .create_invoice(Some(self.amount))
            .await?;
        tracing::debug!(
            target: "mesh_llm::payments::timing",
            phase = "invoice_created",
            ms = self.stalled.elapsed().as_millis() as u64,
            "receiver invoice"
        );
        self.service.ledger.record_receivable(&Receivable {
            request_id: self.request_id.clone(),
            peer: self.peer.clone(),
            segment: 0,
            invoice: invoice.clone(),
            tokens: self.input,
            paid: false,
        })?;
        self.events
            .send(GateEvent::InputInvoice(Box::new(Frame::InputInvoice {
                terms: RequestTerms {
                    exchange_id: None,
                    id: self.request_id.clone(),
                    peer: self.peer.clone(),
                    payee: Some(invoice.payee.clone()),
                    model: self.model.clone(),
                    pricing: self.pricing.clone(),
                    input_tokens: self.input,
                    max_output_tokens: u64::from(self.output),
                    max_total_msat: self.max_total_msat,
                    expires_at_ms: invoice.expires_at_ms,
                },
                invoice: invoice.clone(),
            })))
            .context("payment transport closed")?;
        // Open output delivery on the earliest receiver-side evidence that
        // the HTLC arrived. Settlement is still recorded only on a completed
        // payment, by the task spawned below and awaited before the request
        // finishes.
        let arrival = tokio::select! {
            claiming = self.service.wait_arrival(&invoice) => claiming?,
            _ = async {
                while !self.cancelled.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => anyhow::bail!("request cancelled"),
        };
        let service = self.service.clone();
        let settlement = tokio::spawn(async move {
            service.wait_received(&invoice).await?;
            service.ledger.mark_received(&invoice.payment_hash)?;
            anyhow::Ok(())
        });
        *self.input_settlement.lock().await = Some(settlement);
        ensure!(!self.cancelled.load(Ordering::Acquire), "request cancelled");
        self.authorized.store(true, Ordering::Release);
        Ok(arrival)
    }
}

impl GenerationGate for InvoiceGate {
    fn after_prefill(&self, input: usize, output: u32) -> openai_frontend::OpenAiResult<()> {
        let authorization = self.prepare_authorization(input, output).map_err(|_| {
            openai_frontend::OpenAiError::backend("inference payment was not authorized")
        })?;
        self.spawn_authorization(authorization);
        Ok(())
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
