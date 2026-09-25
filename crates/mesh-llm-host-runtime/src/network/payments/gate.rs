use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use mesh_llm_payments_types::{
    contract::{
        ArrivalResponse, Empty, IdRequest, InputInvoiceResponse, InvoiceRequest,
        RecordDeliveredRequest, ServeInputInvoiceRequest, ops,
    },
    lifetimes::{INPUT_ARRIVAL_WAIT, PRE_PAYMENT_OUTPUT_TOKENS},
    pricing::Pricing,
    wire::Frame,
};

use super::client::Payments;
use skippy_server::frontend::generation_gate::GenerationGate;
use tokio::sync::mpsc;

pub(super) enum GateEvent {
    InputInvoice(Box<Frame>),
    Opened,
    Failed,
}

pub(super) struct InvoiceGate {
    pub payments: Payments,
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
    /// Delivered-token watermark observed by the host, and the part of it
    /// already written to the ledger. Writes are batched by the server.
    pub delivered_tokens: AtomicU64,
    pub flushed_tokens: AtomicU64,
    /// Expiry of the input invoice once created (0 before). The decode pause
    /// is bounded by this, not by when the pause began.
    pub invoice_expires_at_ms: Arc<AtomicU64>,
    /// Settles the input payment in the ledger once the receiver observes a
    /// completed payment. Started when output delivery opens, awaited before
    /// the request completes.
    pub input_settlement: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
}

const CLOSE_SERVING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

impl InvoiceGate {
    /// Wait for the input payment the delivery gate opened on to settle.
    /// Between `claiming` and this point the provider carries the risk.
    /// Write the delivered-token watermark to the ledger if it moved.
    pub(super) async fn flush_delivered(&self) -> Result<()> {
        let tokens = self.delivered_tokens.load(Ordering::Acquire);
        if tokens <= self.flushed_tokens.load(Ordering::Acquire) {
            return Ok(());
        }
        let _: Empty = self
            .payments
            .call(
                ops::RECORD_DELIVERED,
                &RecordDeliveredRequest {
                    id: self.request_id.clone(),
                    tokens,
                },
            )
            .await?;
        self.flushed_tokens.fetch_max(tokens, Ordering::AcqRel);
        Ok(())
    }

    /// Flush the watermark, then close serving accounting. Idempotent.
    /// Bounded so a stuck engine or wallet cannot hold the response open;
    /// on timeout the request fails and startup recovery reconciles it.
    pub(super) async fn close_serving(&self) -> Result<()> {
        tokio::time::timeout(CLOSE_SERVING_TIMEOUT, self.close_serving_inner())
            .await
            .map_err(|_| anyhow::anyhow!("closing serving accounting timed out"))?
    }

    async fn close_serving_inner(&self) -> Result<()> {
        let flushed = self.flush_delivered().await;
        let _: Empty = self
            .payments
            .call(
                ops::SERVE_FINISH,
                &IdRequest {
                    id: self.request_id.clone(),
                },
            )
            .await?;
        flushed
    }

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
        // The output allowance is fixed durably by the engine when it issues
        // the input invoice; a refusal there fails authorization.
        Ok(Authorization {
            payments: self.payments.clone(),
            request_id: self.request_id.clone(),
            peer: self.peer.clone(),
            model: self.model.clone(),
            pricing: self.pricing.clone(),
            input: input as u64,
            output,
            events: self.events.clone(),
            authorized: self.authorized.clone(),
            cancelled: self.cancelled.clone(),
            invoice_expires_at_ms: self.invoice_expires_at_ms.clone(),
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
            let opened_on = result.as_ref().ok().map(|arrival: &ArrivalResponse| {
                if arrival.claiming {
                    "claiming"
                } else {
                    "terminal"
                }
            });
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
    payments: Payments,
    request_id: String,
    peer: String,
    model: String,
    pricing: Pricing,
    input: u64,
    output: u32,
    events: mpsc::UnboundedSender<GateEvent>,
    authorized: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    invoice_expires_at_ms: Arc<AtomicU64>,
    input_settlement: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<()>>>>>,
    stalled: std::time::Instant,
}

impl Authorization {
    async fn authorize(&self) -> Result<ArrivalResponse> {
        let InputInvoiceResponse { terms, invoice } = self
            .payments
            .call(
                ops::SERVE_INPUT_INVOICE,
                &ServeInputInvoiceRequest {
                    id: self.request_id.clone(),
                    peer: self.peer.clone(),
                    model: self.model.clone(),
                    pricing: self.pricing.clone(),
                    input_tokens: self.input,
                    max_output_tokens: u64::from(self.output),
                },
            )
            .await?;
        self.invoice_expires_at_ms
            .store(invoice.expires_at_ms, Ordering::Release);
        tracing::debug!(
            target: "mesh_llm::payments::timing",
            phase = "invoice_created",
            ms = self.stalled.elapsed().as_millis() as u64,
            "receiver invoice"
        );
        self.events
            .send(GateEvent::InputInvoice(Box::new(Frame::InputInvoice {
                terms,
                invoice: invoice.clone(),
            })))
            .context("payment transport closed")?;
        // Open output delivery on the earliest receiver-side evidence that
        // the HTLC arrived. Settlement is still recorded only on a completed
        // payment, by the task spawned below and awaited before the request
        // finishes. The wait lasts exactly as long as the invoice is payable:
        // giving up any earlier would leave a window where this node still
        // claims a late HTLC after the buffered output has been discarded.
        let wait = InvoiceRequest {
            invoice: invoice.clone(),
        };
        let arrival = tokio::select! {
            arrival = self.payments.call::<_, ArrivalResponse>(ops::AWAIT_ARRIVAL, &wait) => arrival?,
            _ = async {
                while !self.cancelled.load(Ordering::Acquire) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            } => anyhow::bail!("request cancelled"),
        };
        let payments = self.payments.clone();
        let settlement = tokio::spawn(async move {
            let _: Empty = payments
                .call(ops::SETTLE_RECEIVED, &InvoiceRequest { invoice })
                .await?;
            anyhow::Ok(())
        });
        *self.input_settlement.lock().await = Some(settlement);
        ensure!(!self.cancelled.load(Ordering::Acquire), "request cancelled");
        self.authorized.store(true, Ordering::Release);
        Ok(arrival)
    }
}

/// Beyond the arrival wait so the authorization task, which owns the
/// deadline, always resolves the pause first; this bound only guards against a
/// gate that nothing will ever resolve.
const PRE_PAYMENT_PAUSE_SLACK: std::time::Duration = std::time::Duration::from_secs(5);
const PRE_PAYMENT_PAUSE_POLL: std::time::Duration = std::time::Duration::from_millis(5);

impl InvoiceGate {
    /// Runs before each output token is consumed (on the direct decode path,
    /// the generation thread; on the default scheduler path, the token
    /// consumer, so the scheduler may keep decoding behind this pause).
    /// Up to `cap` tokens pass ahead of the input payment; after that it
    /// pauses here rather than filling the delivery buffer, because a full
    /// buffer stalls the backend stream and its receiver-stall timeout would
    /// cancel generation while the payment is still in flight. The pause ends
    /// when the payment arrives (continue) or the request is cancelled, which
    /// is also how authorization failure and invoice expiry reach this thread:
    /// the serving task drops its guard and sets `cancelled`.
    fn wait_for_decode_allowance(
        &self,
        cap: u64,
        max_pause: std::time::Duration,
    ) -> openai_frontend::OpenAiResult<()> {
        let cancelled = || {
            Err(openai_frontend::OpenAiError::backend(
                "paid request cancelled",
            ))
        };
        let mut paused_since = None;
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return cancelled();
            }
            if self.authorized.load(Ordering::Acquire)
                || self.output_tokens.load(Ordering::Acquire) < cap
            {
                return Ok(());
            }
            let started = *paused_since.get_or_insert_with(std::time::Instant::now);
            // Once the invoice exists the pause is bounded by its expiry, so a
            // slow invoice creation cannot shorten a payer's window. Before
            // that, `max_pause` from the pause start bounds a stuck creation.
            let expires = self.invoice_expires_at_ms.load(Ordering::Acquire);
            let gave_up = if expires == 0 {
                started.elapsed() >= max_pause
            } else {
                mesh_llm_wallet::now_ms()
                    > expires.saturating_add(PRE_PAYMENT_PAUSE_SLACK.as_millis() as u64)
            };
            if gave_up {
                self.cancelled.store(true, Ordering::Release);
                return Err(openai_frontend::OpenAiError::backend(
                    "input payment did not arrive",
                ));
            }
            std::thread::sleep(PRE_PAYMENT_PAUSE_POLL);
        }
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
        self.wait_for_decode_allowance(
            PRE_PAYMENT_OUTPUT_TOKENS,
            INPUT_ARRIVAL_WAIT + PRE_PAYMENT_PAUSE_SLACK,
        )
    }

    fn committed_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Acquire)
    }

    fn committed_token(&self) -> openai_frontend::OpenAiResult<()> {
        self.output_tokens.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    async fn gate() -> Arc<InvoiceGate> {
        let (events, _receiver) = mpsc::unbounded_channel();
        Arc::new(InvoiceGate {
            payments: Payments::unavailable_for_tests().await.unwrap(),
            request_id: uuid::Uuid::new_v4().to_string(),
            peer: "peer".into(),
            model: "test".into(),
            pricing: Pricing {
                input_msat_per_million: 1,
                output_msat_per_million: 1,
                minimum_invoice_msat: 1,
            },
            max_tokens: Some(64),
            events,
            runtime: tokio::runtime::Handle::current(),
            authorized: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            started: AtomicBool::new(false),
            output_tokens: AtomicU64::new(0),
            delivered_tokens: AtomicU64::new(0),
            flushed_tokens: AtomicU64::new(0),
            invoice_expires_at_ms: Arc::new(AtomicU64::new(0)),
            input_settlement: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    /// Emulate the backend's decode loop on a blocking thread: every token
    /// asks the gate first, then commits. Returns tokens committed and the
    /// loop's outcome.
    fn decode(gate: &InvoiceGate, tokens: u64, cap: u64, max_pause: Duration) -> (u64, bool) {
        for _ in 0..tokens {
            if gate.wait_for_decode_allowance(cap, max_pause).is_err() {
                return (gate.committed_tokens(), false);
            }
            gate.committed_token().unwrap();
        }
        (gate.committed_tokens(), true)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn decode_runs_ahead_to_the_cap_then_pauses_until_paid() {
        let gate = gate().await;
        let decoding = gate.clone();
        let task =
            tokio::task::spawn_blocking(move || decode(&decoding, 20, 8, Duration::from_secs(30)));
        // Decode reaches the cap and holds there: paused, not failed.
        let deadline = Instant::now() + Duration::from_secs(5);
        while gate.committed_tokens() < 8 {
            assert!(Instant::now() < deadline, "decode never reached the cap");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(gate.committed_tokens(), 8, "decode passed the cap unpaid");
        assert!(!task.is_finished(), "a paused request must not end");
        assert!(!gate.cancelled.load(Ordering::Acquire));
        // Late payment: decode resumes and completes the whole allowance.
        gate.authorized.store(true, Ordering::Release);
        assert_eq!(task.await.unwrap(), (20, true));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_ends_a_paused_decode() {
        // Authorization failure and invoice expiry reach the generation
        // thread this way: the serving task's guard sets `cancelled`.
        let gate = gate().await;
        let decoding = gate.clone();
        let task =
            tokio::task::spawn_blocking(move || decode(&decoding, 20, 4, Duration::from_secs(30)));
        while gate.committed_tokens() < 4 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        gate.cancelled.store(true, Ordering::Release);
        let started = Instant::now();
        assert_eq!(task.await.unwrap(), (4, false));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unresolved_pause_gives_up_and_cancels() {
        let gate = gate().await;
        let decoding = gate.clone();
        let started = Instant::now();
        let outcome = tokio::task::spawn_blocking(move || {
            decode(&decoding, 20, 3, Duration::from_millis(150))
        })
        .await
        .unwrap();
        assert_eq!(outcome, (3, false));
        assert!(started.elapsed() >= Duration::from_millis(150));
        assert!(gate.cancelled.load(Ordering::Acquire));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_invoice_creation_does_not_shorten_the_payment_window() {
        // Invoice created after the pause began, still payable for a while:
        // the pause must outlive the pre-invoice bound, then resume on payment.
        let gate = gate().await;
        let decoding = gate.clone();
        let task = tokio::task::spawn_blocking(move || {
            decode(&decoding, 20, 3, Duration::from_millis(100))
        });
        while gate.committed_tokens() < 3 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        gate.invoice_expires_at_ms
            .store(mesh_llm_wallet::now_ms() + 60_000, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            !task.is_finished(),
            "pause gave up while the invoice was payable"
        );
        assert!(!gate.cancelled.load(Ordering::Acquire));
        gate.authorized.store(true, Ordering::Release);
        assert_eq!(task.await.unwrap(), (20, true));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_gives_up_after_the_invoice_deadline() {
        let gate = gate().await;
        let slack = PRE_PAYMENT_PAUSE_SLACK.as_millis() as u64;
        gate.invoice_expires_at_ms.store(
            mesh_llm_wallet::now_ms().saturating_sub(slack) + 150,
            Ordering::Release,
        );
        let decoding = gate.clone();
        let started = Instant::now();
        let outcome = tokio::task::spawn_blocking(move || {
            decode(&decoding, 20, 3, Duration::from_secs(3600))
        })
        .await
        .unwrap();
        assert_eq!(outcome, (3, false));
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(gate.cancelled.load(Ordering::Acquire));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paid_before_the_cap_never_pauses() {
        let gate = gate().await;
        gate.authorized.store(true, Ordering::Release);
        let decoding = gate.clone();
        let started = Instant::now();
        let outcome =
            tokio::task::spawn_blocking(move || decode(&decoding, 50, 2, Duration::from_millis(1)))
                .await
                .unwrap();
        assert_eq!(outcome, (50, true));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn production_gate_pauses_well_inside_the_buffer_and_invoice_lifetime() {
        assert_eq!(
            mesh_llm_payments_types::lifetimes::INPUT_INVOICE_EXPIRY_SECS,
            60
        );
        assert_eq!(INPUT_ARRIVAL_WAIT, Duration::from_secs(60));
    }
}
