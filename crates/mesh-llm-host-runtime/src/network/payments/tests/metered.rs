//! The real v2 payer/provider over authenticated QUIC and streamed TCP HTTP.
use super::*;
use mesh_llm_payments_types::metered_wire as v2;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn first_deposit_then_trusted_stream_without_native_gate() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), streamed_pair()).await?
}

async fn streamed_pair() -> Result<()> {
    let seller_dir = tempfile::tempdir()?;
    let buyer_dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let seller = Arc::new(PaymentService::with_provider(
        seller_dir.path(),
        Arc::new(TestWallet {
            owner: 1,
            network: network.clone(),
        }),
    )?);
    let buyer = Arc::new(PaymentService::with_provider(
        buyer_dir.path(),
        Arc::new(TestWallet {
            owner: 2,
            network: network.clone(),
        }),
    )?);
    allow_paid(&buyer)?;
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    seller.ledger.set_pricing("test", Some(&price))?;
    let provider = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
    provider
        .payments
        .set(seller.clone())
        .map_err(|_| anyhow::anyhow!("already set"))?;
    let payer = Node::new_for_tests(NodeRole::Client).await?;
    payer
        .payments
        .set(buyer.clone())
        .map_err(|_| anyhow::anyhow!("already set"))?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let backend_network = network.clone();
    let backend = tokio::spawn(stream_backend(listener, backend_network));
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert("test".into(), vec![InferenceTarget::Local(port)]);
    let serving_node = provider.clone();
    let serving = tokio::spawn(serve_pair(serving_node, targets));
    let announcement =
        provider.build_local_announcement(provider.snapshot_local_announcement_data().await);
    payer
        .add_peer_after_direct_requirements_validated(
            provider.id(),
            provider.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
    let connection = payer
        .endpoint
        .connect(provider.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    payer
        .state
        .lock()
        .await
        .connections
        .insert(provider.id(), connection);
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"stream\":true,\"max_tokens\":8}";
    verify_exchanges(&payer, &provider, &buyer, &seller, &network, raw, &price).await?;
    backend.await??;
    let _conn = serving.await??;
    payer.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

async fn stream_backend(
    listener: tokio::net::TcpListener,
    backend_network: Arc<Network>,
) -> Result<()> {
    for _ in 0..2 {
        let (mut stream, _) = listener.accept().await?;
        let mut raw = [0; 8192];
        let count = stream.read(&mut raw).await?;
        let text = std::str::from_utf8(&raw[..count])?;
        assert!(text.contains("\"include_usage\":true"));
        assert!(
            backend_network.payments.load(Ordering::SeqCst) > 0,
            "dispatch before deposit"
        );
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"hello\"}]}\n\n").await?;
        tokio::task::yield_now().await;
        stream.write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n").await?;
        stream.shutdown().await?;
    }
    Ok(())
}

async fn serve_pair(
    serving_node: Node,
    targets: ModelTargets,
) -> Result<iroh::endpoint::Connection> {
    let conn = serving_node.endpoint.accept().await.unwrap().await?;
    for _ in 0..2 {
        let (send, mut recv) = loop {
            let (send, mut recv) = conn.accept_bi().await?;
            if recv.read_u8().await? == crate::protocol::STREAM_TUNNEL_HTTP {
                break (send, recv);
            }
        };
        let mut prefix = vec![0; v2::HTTP_UPGRADE.len()];
        recv.read_exact(&mut prefix).await?;
        assert_eq!(prefix, v2::HTTP_UPGRADE);
        super::super::metered::serve(
            serving_node.clone(),
            conn.remote_id(),
            recv,
            send,
            targets.clone(),
        )
        .await?;
    }
    Ok(conn)
}

async fn verify_exchanges(
    payer: &Node,
    provider: &Node,
    buyer: &PaymentService,
    seller: &PaymentService,
    network: &Network,
    raw: &[u8],
    price: &Pricing,
) -> Result<()> {
    // The caller omits include_usage first, then opts in. Usage stays hidden
    // unless requested, while settlement still bills backend-reported usage.
    let opted_in = String::from_utf8(raw.to_vec())?.replace(
        "\"stream\":true",
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}",
    );
    for completed in 1..=2 {
        let request = if completed == 1 {
            raw
        } else {
            opted_in.as_bytes()
        };
        let (mut pipe, ready, _cancel) = crate::network::openai::test_metered_start(
            payer,
            provider.id(),
            request,
            price.clone(),
            None,
        )
        .await?;
        ready.await?;
        let mut response = String::new();
        pipe.read_to_string(&mut response).await?;
        assert!(response.contains("hello"));
        assert!(response.contains("data: [DONE]"));
        assert_eq!(response.contains("\"usage\""), completed == 2, "{response}");
        loop {
            if buyer
                .ledger
                .requests()?
                .iter()
                .filter(|r| r.state == "completed")
                .count()
                == completed
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            network.payments.load(Ordering::SeqCst),
            if completed == 1 { 2 } else { 3 }
        );
        assert_eq!(
            seller
                .ledger
                .metered_unpaid_finals(&payer.id().to_string())?,
            0
        );
    }
    Ok(())
}

/// Deterministic failure after accepting a prefix, not a timing-dependent TCP reset.
struct PartialSubmission {
    accepted: Vec<u8>,
}

impl tokio::io::AsyncWrite for PartialSubmission {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        bytes: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        if self.accepted.is_empty() {
            let count = bytes.len().min(7);
            self.accepted.extend_from_slice(&bytes[..count]);
            std::task::Poll::Ready(Ok(count))
        } else {
            std::task::Poll::Ready(Err(std::io::ErrorKind::BrokenPipe.into()))
        }
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn partial_backend_submission_abandons_without_bill_and_keeps_dispatch_claim() -> Result<()> {
    use mesh_llm_payments::ledger::metered::{MeteredRequest, MeteredStatus};
    let dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let wallet = Arc::new(TestWallet { owner: 1, network });
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    let deposit_msat = service.ledger.metered_admit(&MeteredRequest {
        id: "partial",
        peer: "buyer",
        pricing: &price,
        estimated_input_tokens: 1000,
        require_deposit: true,
    })?;
    let invoice = service
        .metered_invoice(
            "partial",
            "buyer",
            mesh_llm_payments::ledger::metered_invoices::MeteredInvoiceKind::Deposit,
            deposit_msat,
        )
        .await?
        .unwrap();
    // Authoritative incoming wallet fact before dispatch.
    wallet
        .network
        .entries
        .lock()
        .unwrap()
        .get_mut(&invoice.payment_hash)
        .unwrap()
        .1
        .status = PaymentStatus::Succeeded;
    service.metered_refresh("partial").await?;
    assert!(service.ledger.metered_claim_dispatch("partial", "buyer")?);
    let mut backend = PartialSubmission {
        accepted: Vec::new(),
    };
    let failure = super::super::metered_backend::submit_request(
        &mut backend,
        b"POST /v1/completions HTTP/1.1\r\n\r\n{}",
    )
    .await
    .unwrap_err();
    assert_eq!(backend.accepted, b"POST /v");
    service
        .ledger
        .metered_finalize("partial", failure.outcome())?;
    let settlement = service.ledger.metered_settlement("partial")?;
    assert_eq!(settlement.status, MeteredStatus::Abandoned);
    assert_eq!(settlement.deposit_paid_msat, deposit_msat);
    assert_eq!(settlement.bill_msat, None);
    assert_eq!(service.ledger.metered_unpaid_finals("buyer")?, 0);
    assert!(!service.ledger.metered_claim_dispatch("partial", "buyer")?);
    drop(service);
    let restarted = PaymentService::with_provider(dir.path(), wallet)?;
    restarted.metered_refresh("partial").await?;
    assert!(
        !restarted
            .ledger
            .metered_claim_dispatch("partial", "buyer")?
    );
    assert!(
        restarted
            .metered_final_invoice("partial", "buyer")
            .await?
            .is_none()
    );
    Ok(())
}

#[tokio::test]
async fn connection_failure_before_request_writing_abandons() -> Result<()> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let request = super::super::request::PaidRequest::parse(
        b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"test\",\"max_tokens\":8}",
    )?;
    let failure = match super::super::metered_backend::Backend::Local(port)
        .connect(&request, "id")
        .await
    {
        Err(failure) => failure,
        Ok(_) => anyhow::bail!("unexpected connection"),
    };
    assert!(matches!(
        failure.outcome(),
        mesh_llm_payments::ledger::metered::MeteredOutcome::Abandoned
    ));
    Ok(())
}
