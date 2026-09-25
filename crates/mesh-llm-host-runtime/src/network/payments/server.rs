use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments_types::contract::{
    Empty, IdRequest, InvoiceRequest, OutputReceivableResponse, ServeBeginRequest,
    ServeRecoverResponse, ops,
};
use mesh_llm_payments_types::wire::{self, Frame};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use super::{
    client::Payments,
    gate::{GateEvent, InvoiceGate},
    request::PaidRequest,
};
use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::Node,
};

pub(crate) async fn serve(
    node: Node,
    peer: iroh::EndpointId,
    mut reader: impl AsyncRead + Unpin,
    mut writer: impl AsyncWrite + Unpin,
    targets: ModelTargets,
) -> Result<()> {
    let result = serve_inner(&node, peer, &mut reader, &mut writer, &targets).await;
    if result.is_err() {
        // Avoid reflecting request contents or wallet-provider errors to peers.
        let _ = wire::write(
            &mut writer,
            &Frame::Error {
                message: "paid inference failed; recover pending invoices by request ID".into(),
            },
        )
        .await;
    }
    result
}

async fn serve_inner(
    node: &Node,
    peer: iroh::EndpointId,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    targets: &ModelTargets,
) -> Result<()> {
    let frame = tokio::time::timeout(Duration::from_secs(10), wire::read(reader)).await??;
    let payments = Payments::for_node(node).await?;
    if let Frame::Recover { id } = frame {
        return recover(&payments, &id, writer).await;
    }
    let Frame::Request {
        id,
        model,
        pricing,
        http,
    } = frame
    else {
        bail!("expected inference request");
    };
    uuid::Uuid::parse_str(&id).context("invalid request ID")?;
    let mut request = PaidRequest::parse(&http)?;
    ensure!(request.model == model, "model mismatch");
    // Prices and offers use the public model ID; the local backend is
    // registered, and must be addressed, under its internal name.
    let backend_model =
        crate::network::openai::ingress::served_model_for_public_id(node, targets, &model).await;
    request.use_backend_model(&backend_model);
    let port = targets
        .candidates(&backend_model)
        .iter()
        .find_map(|target| match target {
            InferenceTarget::Local(port) => Some(*port),
            _ => None,
        })
        .context("paid model must be served locally")?;
    // Checks the seller's current price, wakes the receiving wallet during
    // prefill, and waits out this peer's prior debt before opening serving.
    let peer = peer.to_string();
    let _: Empty = payments
        .call(
            ops::SERVE_BEGIN,
            &ServeBeginRequest {
                id: id.clone(),
                peer: peer.clone(),
                model: model.clone(),
                pricing: pricing.clone(),
                max_output: u64::from(request.max_tokens.unwrap_or(u32::MAX)),
                prior_settlement_ms: PRIOR_SETTLEMENT_WAIT.as_millis() as u64,
            },
        )
        .await?;
    let _instance = node.begin_runtime_instance_request(port).await?;
    let (events, mut receiver) = mpsc::unbounded_channel();
    let gate = Arc::new(InvoiceGate {
        payments: payments.clone(),
        request_id: id.clone(),
        peer: peer.clone(),
        model,
        pricing: pricing.clone(),
        max_tokens: request.max_tokens,
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
    });
    let _serving_guard = ServingGuard { gate: gate.clone() };
    let generated = generate(reader, writer, port, &request, &gate, &mut receiver).await;
    // Close serving on every path, before anything else can observe this
    // peer, so an interrupted request's delivered output counts as debt.
    let closed = gate.close_serving().await;
    let transport_alive = generated?;
    closed?;
    // Generation is over. Release the runtime's in-flight slot (the gate
    // registration ended with generation) before waiting on the payer's wallet: those waits can last
    // up to the output invoice lifetime, and the debt they settle is already
    // durable, so recovery finishes it if this task ends first.
    drop(_instance);
    // Delivery opened on receiver-side HTLC arrival; the input payment must
    // still be recorded as settled before this request is complete.
    gate.await_input_settlement().await?;
    let output = if gate.authorized.load(Ordering::Acquire) {
        payments
            .call::<_, OutputReceivableResponse>(
                ops::OUTPUT_RECEIVABLE,
                &IdRequest { id: id.clone() },
            )
            .await?
            .output
    } else {
        None
    };
    if let Some(receipt) = output {
        if transport_alive {
            wire::write(
                writer,
                &Frame::OutputInvoice {
                    request_id: id.clone(),
                    tokens: receipt.tokens,
                    invoice: receipt.invoice.clone(),
                },
            )
            .await?;
        }
        let _: Empty = payments
            .call(
                ops::SETTLE_RECEIVED,
                &InvoiceRequest {
                    invoice: receipt.invoice,
                },
            )
            .await?;
    }
    if transport_alive {
        wire::write(writer, &Frame::Complete).await?;
    }
    Ok(())
}

/// Runs the backend under the registered payment gate until its output is
/// delivered or the payer goes away; the registration ends with generation.
async fn generate(
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    port: u16,
    request: &PaidRequest,
    gate: &Arc<InvoiceGate>,
    receiver: &mut mpsc::UnboundedReceiver<GateEvent>,
) -> Result<bool> {
    let backend_id = uuid::Uuid::new_v4();
    let _registration =
        skippy_server::frontend::generation_gate::register(*backend_id.as_bytes(), gate.clone())
            .map_err(|_| anyhow::anyhow!("could not install payment gate"))?;
    let mut backend = TcpStream::connect(("127.0.0.1", port)).await?;
    backend
        .write_all(&request.backend_http(&backend_id.to_string())?)
        .await?;
    stream_output(reader, writer, &mut backend, gate, receiver).await
}

async fn stream_output(
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl AsyncWrite + Unpin),
    backend: &mut TcpStream,
    gate: &InvoiceGate,
    events: &mut mpsc::UnboundedReceiver<GateEvent>,
) -> Result<bool> {
    // Decode runs as soon as prefill completes. Drain the backend into a
    // bounded buffer, but do not release even HTTP headers until the provider's
    // receiving wallet sees `claiming` or a terminal fallback. The payment gate
    // pauses token delivery after `PRE_PAYMENT_OUTPUT_TOKENS`, which keeps this buffer
    // from filling while unpaid. The byte cap is only a backstop: if it is
    // ever reached, reads stop, the backend's stream stalls, and its
    // receiver-stall timeout cancels generation (it does not pause decode).
    let mut buffer = vec![0; 16 * 1024];
    let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
    let mut pending_bytes = 0_usize;
    let mut gate_open = false;
    let mut invoice_sent = false;
    let mut backend_eof = false;
    let mut delivery = super::delivery::DeliveryUsage::default();
    let invoice_timeout = tokio::time::sleep(Duration::from_secs(300));
    tokio::pin!(invoice_timeout);
    // Keep partial cancellation-frame bytes across backend output reads.
    let incoming = wire::read(reader);
    tokio::pin!(incoming);
    loop {
        if gate_open {
            while let Some(bytes) = pending.pop_front() {
                pending_bytes -= bytes.len();
                if !deliver_output(writer, gate, &mut delivery, bytes).await? {
                    return Ok(false);
                }
            }
            if backend_eof {
                break;
            }
        }
        let read_capacity = if gate_open {
            buffer.len()
        } else {
            buffer
                .len()
                .min(MAX_BUFFERED_OUTPUT_BYTES.saturating_sub(pending_bytes))
        };
        tokio::select! {
            _ = &mut invoice_timeout, if !invoice_sent => {
                bail!("generation did not reach payment gate");
            }
            event = events.recv(), if !gate_open => {
                match event.context("payment gate closed")? {
                    GateEvent::InputInvoice(invoice) => {
                        ensure!(!invoice_sent, "duplicate input invoice");
                        if wire::write(writer, invoice.as_ref()).await.is_err() {
                            gate.cancelled.store(true, Ordering::Release);
                            return Ok(false);
                        }
                        invoice_sent = true;
                    }
                    GateEvent::Opened => {
                        ensure!(invoice_sent, "payment gate opened before invoice");
                        gate_open = true;
                    }
                    GateEvent::Failed => bail!("input payment was not authorized"),
                }
            }
            incoming = &mut incoming => {
                let _ = incoming;
                gate.cancelled.store(true, Ordering::Release);
                break;
            }
            read = backend.read(&mut buffer[..read_capacity]), if !backend_eof && read_capacity > 0 => {
                let count = read?;
                if count == 0 {
                    // No authorization means the backend never produced a
                    // first token (e.g. prefill was rejected); nothing will be
                    // invoiced, so fail now instead of waiting for the invoice.
                    ensure!(
                        gate.started.load(Ordering::Acquire),
                        "backend ended before payment authorization started"
                    );
                    backend_eof = true;
                } else if gate_open {
                    if !deliver_output(writer, gate, &mut delivery, buffer[..count].to_vec()).await? {
                        return Ok(false);
                    }
                } else {
                    pending_bytes += count;
                    pending.push_back(buffer[..count].to_vec());
                }
            }
        }
    }
    Ok(true)
}

/// Backstop for bytes buffered before payment. Sized at ~2 KiB per token of
/// `PRE_PAYMENT_OUTPUT_TOKENS`, several times a streamed chat-completion
/// chunk, so the token pause is always reached first.
const MAX_BUFFERED_OUTPUT_BYTES: usize = 1024 * 1024;
const _: () = assert!(
    MAX_BUFFERED_OUTPUT_BYTES as u64
        >= 2048 * mesh_llm_payments_types::lifetimes::PRE_PAYMENT_OUTPUT_TOKENS
);

async fn deliver_output(
    writer: &mut (impl AsyncWrite + Unpin),
    gate: &InvoiceGate,
    delivery: &mut super::delivery::DeliveryUsage,
    bytes: Vec<u8>,
) -> Result<bool> {
    let frame = Frame::Output { bytes };
    if wire::write(writer, &frame).await.is_err() {
        gate.cancelled.store(true, Ordering::Release);
        return Ok(false);
    }
    let Frame::Output { bytes } = frame else {
        unreachable!("output frame changed before delivery accounting");
    };
    let delivered_tokens = delivery.observe(&bytes)?;
    gate.delivered_tokens
        .fetch_max(delivered_tokens, Ordering::AcqRel);
    if delivered_tokens.saturating_sub(gate.flushed_tokens.load(Ordering::Acquire))
        >= DELIVERED_FLUSH_TOKENS
    {
        gate.flush_delivered().await?;
    }
    Ok(true)
}

/// How long a new request waits for the same peer's prior debt to settle.
const PRIOR_SETTLEMENT_WAIT: Duration = Duration::from_secs(30);

/// Delivered-token watermark writes are batched: the ledger is raised once at
/// least this many tokens accumulate, and the final count is written
/// atomically with the serving close. A crash can only under-record (never
/// over-record): the loss is bounded by one batch plus the last output frame,
/// which for coalesced SSE or a nonstreaming response can hold many tokens.
const DELIVERED_FLUSH_TOKENS: u64 = 32;

pub(super) async fn recover(
    payments: &Payments,
    id: &str,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    uuid::Uuid::parse_str(id)?;
    let response: ServeRecoverResponse = payments
        .call(ops::SERVE_RECOVER, &IdRequest { id: id.into() })
        .await?;
    let ServeRecoverResponse::Complete { output } = response else {
        return wire::write(writer, &Frame::Pending).await;
    };
    if let Some(receipt) = output {
        wire::write(
            writer,
            &Frame::OutputInvoice {
                request_id: id.into(),
                tokens: receipt.tokens,
                invoice: receipt.invoice,
            },
        )
        .await?;
    }
    wire::write(writer, &Frame::Complete).await
}

struct ServingGuard {
    gate: Arc<InvoiceGate>,
}
impl Drop for ServingGuard {
    fn drop(&mut self) {
        self.gate.cancelled.store(true, Ordering::Release);
        // Backstop for a cancelled serving task, which never reaches the
        // awaited close; every other path has already closed serving.
        let gate = self.gate.clone();
        self.gate.runtime.spawn(async move {
            let _ = gate.close_serving().await;
        });
    }
}
