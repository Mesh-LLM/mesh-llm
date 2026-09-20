use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_payments::{
    service::PaymentService,
    wire::{self, Frame},
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use super::{
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
    let service = node.payment_service().await?;
    if let Frame::Recover { id } = frame {
        return recover(&service, &id, writer).await;
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
    let request = PaidRequest::parse(&http)?;
    ensure!(request.model == model, "model mismatch");
    ensure!(
        service.ledger.pricing()?.get(&model) == Some(&pricing),
        "seller prices changed"
    );
    let port = targets
        .candidates(&model)
        .iter()
        .find_map(|target| match target {
            InferenceTarget::Local(port) => Some(*port),
            _ => None,
        })
        .context("paid model must be served locally")?;
    let peer = peer.to_string();
    refresh_receivables(&service, &peer).await?;
    service.ledger.begin_serving(
        &id,
        &peer,
        &pricing,
        u64::from(request.max_tokens.unwrap_or(u32::MAX)),
    )?;
    let _instance = node.begin_runtime_instance_request(port).await?;
    let (events, mut receiver) = mpsc::unbounded_channel();
    let gate = Arc::new(InvoiceGate {
        service: service.clone(),
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
        input_settlement: Arc::new(tokio::sync::Mutex::new(None)),
    });
    let _serving_guard = ServingGuard { gate: gate.clone() };
    let backend_id = uuid::Uuid::new_v4();
    let _registration =
        skippy_server::frontend::generation_gate::register(*backend_id.as_bytes(), gate.clone())
            .map_err(|_| anyhow::anyhow!("could not install payment gate"))?;
    let mut backend = TcpStream::connect(("127.0.0.1", port)).await?;
    backend
        .write_all(&request.backend_http(&backend_id.to_string())?)
        .await?;
    let transport_alive = stream_output(reader, writer, &mut backend, &gate, &mut receiver).await?;
    drop(backend);
    service.ledger.finish_serving(&id)?;
    // Delivery opened on receiver-side HTLC arrival; the input payment must
    // still be recorded as settled before this request is complete.
    gate.await_input_settlement().await?;
    if gate.authorized.load(Ordering::Acquire)
        && let Some(receipt) = service.output_receivable(&id).await?
    {
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
        service.wait_received(&receipt.invoice).await?;
        service
            .ledger
            .mark_received(&receipt.invoice.payment_hash)?;
    }
    if transport_alive {
        wire::write(writer, &Frame::Complete).await?;
    }
    Ok(())
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
    // receiving wallet sees `claiming` or a terminal fallback. Reaching the
    // cap applies TCP backpressure to decode rather than growing without bound.
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

const MAX_BUFFERED_OUTPUT_BYTES: usize = 256 * 1024;

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
    gate.service
        .ledger
        .record_delivered_tokens(&gate.request_id, delivered_tokens)?;
    Ok(true)
}

async fn refresh_receivables(service: &PaymentService, peer: &str) -> Result<()> {
    let unpaid = service.ledger.unpaid_invoices(peer)?;
    if unpaid.is_empty() {
        return Ok(());
    }
    let wallet = service.wallet().await?;
    for invoice in unpaid {
        if wallet
            .lookup(&invoice.payment_hash)
            .await?
            .is_some_and(|p| {
                p.inbound
                    && p.payment_hash.as_deref() == Some(invoice.payment_hash.as_str())
                    && p.status == mesh_llm_payments::wallet::PaymentStatus::Succeeded
            })
        {
            service.ledger.mark_received(&invoice.payment_hash)?;
        }
    }
    Ok(())
}

async fn recover(
    service: &PaymentService,
    id: &str,
    writer: &mut (impl AsyncWrite + Unpin),
) -> Result<()> {
    uuid::Uuid::parse_str(id)?;
    // The random request ID is a bearer recovery capability, allowing recovery
    // when the client's ephemeral mesh identity changes after restart.
    let (_, _, _, finished) = service.ledger.serving_account(id)?;
    if !finished {
        return wire::write(writer, &Frame::Pending).await;
    }
    if let Some(receipt) = service.output_receivable(id).await? {
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
        let _ = self
            .gate
            .service
            .ledger
            .finish_serving(&self.gate.request_id);
    }
}
