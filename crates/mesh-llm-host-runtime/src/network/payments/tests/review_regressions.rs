use super::*;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, ReadBuf};

struct ObservedReader<R> {
    reader: R,
    remaining: usize,
    consumed: Option<tokio::sync::oneshot::Sender<()>>,
}

impl<R: AsyncRead + Unpin> AsyncRead for ObservedReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let result = Pin::new(&mut self.reader).poll_read(cx, buf);
        self.remaining = self.remaining.saturating_sub(buf.filled().len() - before);
        if self.remaining == 0
            && let Some(consumed) = self.consumed.take()
        {
            let _ = consumed.send(());
        }
        result
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_cancellation_mid_frame_still_settles_output() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(10), fragmented_exchange()).await?
}

async fn fragmented_exchange() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let seller = TestWallet {
        owner: 1,
        network: network.clone(),
    };
    let service = Arc::new(PaymentService::with_provider(
        directory.path(),
        Arc::new(TestWallet {
            owner: 2,
            network: network.clone(),
        }),
    )?);
    allow_paid(&service)?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(10_000),
    })?;
    let node = Node::new_for_tests(NodeRole::Client).await?;
    let peer = node.endpoint.id();
    let id = uuid::Uuid::new_v4().to_string();
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    let invoice = seller.create_invoice(Some(40), 3600).await?;
    let input = Frame::InputInvoice {
        terms: mesh_llm_payments::ledger::RequestTerms {
            exchange_id: None,
            id: id.clone(),
            peer: peer.to_string(),
            payee: Some(invoice.payee.clone()),
            model: "test".into(),
            max_total_msat: price.request_cap_msat(price.input_charge(40)?, 8)?,
            pricing: price.clone(),
            input_tokens: 40,
            max_output_tokens: 8,
            expires_at_ms: invoice.expires_at_ms,
        },
        invoice: invoice.clone(),
    };
    let output_frame = serde_json::to_vec(&Frame::Output {
        bytes: b"fragmented output".to_vec(),
    })?;
    let (payer_stream, seller_stream) = tokio::io::duplex(8192);
    let (payer_read, payer_write) = tokio::io::split(payer_stream);
    let (mut seller_read, mut seller_write) = tokio::io::split(seller_stream);
    let (consumed, partial_consumed) = tokio::sync::oneshot::channel();
    let reader = ObservedReader {
        reader: payer_read,
        remaining: 4 + serde_json::to_vec(&input)?.len() + 4 + 2,
        consumed: Some(consumed),
    };
    let request = super::super::request::PaidRequest::parse(b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}")?;
    let (mut output, mut application) = tokio::io::duplex(1024);
    let (ready, _wait_ready) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let payer_service = service.clone();
    let request_id = id.clone();
    let exchange = tokio::spawn(async move {
        let mut reader = reader;
        let initial = wire::read(&mut reader).await?;
        let balance = payer_service.prefetch_balance();
        crate::network::openai::test_payment_exchange(
            payer_service,
            peer,
            request_id,
            request,
            price,
            payer_write,
            reader,
            initial,
            balance,
            &mut output,
            ready,
            cancellation,
            None,
        )
        .await
    });
    wire::write(&mut seller_write, &input).await?;
    seller.wait_for_payment(&invoice.payment_hash).await?;
    seller_write.write_u32(output_frame.len() as u32).await?;
    seller_write.write_all(&output_frame[..2]).await?;
    partial_consumed.await?;
    cancel.send(true)?;
    settle_after_cancel(
        &seller,
        &mut seller_read,
        &mut seller_write,
        &output_frame[2..],
        &id,
    )
    .await?;
    exchange.await??;
    let mut discarded = Vec::new();
    application.read_to_end(&mut discarded).await?;
    assert!(discarded.is_empty());
    assert_eq!(network.payments.load(Ordering::SeqCst), 2);
    assert_eq!(
        service.ledger.request_state(&id)?.as_deref(),
        Some("completed")
    );
    node.endpoint.close().await;
    Ok(())
}

async fn settle_after_cancel(
    seller: &TestWallet,
    reader: &mut (impl AsyncRead + Unpin),
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    tail: &[u8],
    id: &str,
) -> Result<()> {
    assert!(matches!(wire::read(reader).await?, Frame::Cancel));
    writer.write_all(tail).await?;
    let invoice = seller.create_invoice(Some(3), 3600).await?;
    wire::write(
        writer,
        &Frame::OutputInvoice {
            request_id: id.into(),
            tokens: 3,
            invoice: invoice.clone(),
        },
    )
    .await?;
    seller.wait_for_payment(&invoice.payment_hash).await?;
    wire::write(writer, &Frame::Complete).await?;
    Ok(())
}
