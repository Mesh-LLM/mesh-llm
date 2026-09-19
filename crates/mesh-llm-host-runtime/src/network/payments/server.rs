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

use super::{gate::InvoiceGate, request::PaidRequest};
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
        authorized: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
        output_tokens: AtomicU64::new(0),
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
    let invoice = tokio::time::timeout(Duration::from_secs(300), receiver.recv())
        .await?
        .context("generation did not reach payment gate")?;
    wire::write(writer, &invoice).await?;
    let transport_alive = stream_output(reader, writer, &mut backend, &gate).await?;
    drop(backend);
    service.ledger.finish_serving(&id)?;
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
) -> Result<bool> {
    // The backend is suspended until the authoritative receiving wallet sees
    // settlement. Headers may arrive early; the payer holds them until paid.
    let mut buffer = vec![0; 16 * 1024];
    let mut transport_alive = true;
    let mut delivery = super::delivery::DeliveryUsage::default();
    // Keep partial cancellation-frame bytes across backend output reads.
    let incoming = wire::read(reader);
    tokio::pin!(incoming);
    loop {
        tokio::select! {
            incoming = &mut incoming => {
                let _ = incoming;
                gate.cancelled.store(true, Ordering::Release);
                break;
            }
            read = backend.read(&mut buffer) => {
                let count = read?;
                if count == 0 { break; }
                if wire::write(writer, &Frame::Output { bytes: buffer[..count].to_vec() }).await.is_err() {
                    transport_alive = false;
                    gate.cancelled.store(true, Ordering::Release);
                    break;
                }
                let delivered_tokens = delivery.observe(&buffer[..count])?;
                gate.service.ledger.record_delivered_tokens(&gate.request_id, delivered_tokens)?;
            }
        }
    }
    Ok(transport_alive)
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
