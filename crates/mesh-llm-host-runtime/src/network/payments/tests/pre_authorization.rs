use super::*;

// A backend that rejects the request before generation (e.g. a prompt the
// scheduler refuses) never reaches payment authorization. The payer must get
// a prompt error, not wait for an invoice that will never be issued.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_backend_rejection_before_authorization_fails_promptly_without_invoice()
-> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), rejected_exchange()).await?
}

async fn rejecting_backend(listener: tokio::net::TcpListener) -> Result<()> {
    let (mut stream, _) = listener.accept().await?;
    let mut raw = vec![0; 16 * 1024];
    let _ = stream.read(&mut raw).await?;
    stream
        .write_all(
            b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        )
        .await?;
    stream.shutdown().await?;
    Ok(())
}

async fn rejected_exchange() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let service = Arc::new(PaymentService::with_provider(
        dir.path(),
        Arc::new(TestWallet {
            owner: 1,
            network: network.clone(),
        }),
    )?);
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    service.ledger.set_pricing("test", Some(&price))?;
    let provider = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
    provider
        .payments
        .set(service.clone())
        .map_err(|_| anyhow::anyhow!("already initialized"))?;
    crate::network::payments::node_ext::attach_payments_plugin(&provider).await?;
    let caller = Node::new_for_tests(NodeRole::Client).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "test".into(),
        vec![InferenceTarget::Local(listener.local_addr()?.port())],
    );
    let backend = tokio::spawn(rejecting_backend(listener));
    let serving_node = provider.clone();
    let serving = tokio::spawn(async move {
        let conn = serving_node.endpoint.accept().await.unwrap().await?;
        let (send, recv) = conn.accept_bi().await?;
        let result =
            super::super::server::serve(serving_node, conn.remote_id(), recv, send, targets).await;
        Ok::<_, anyhow::Error>((conn, result))
    });
    let conn = caller
        .endpoint
        .connect(provider.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    let id = uuid::Uuid::new_v4().to_string();
    let request = super::super::request::PaidRequest::parse(b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}")?;
    wire::write(
        &mut send,
        &Frame::Request {
            id: id.clone(),
            model: "test".into(),
            pricing: price.clone(),
            http: request.backend_http(&id)?,
        },
    )
    .await?;
    let started = std::time::Instant::now();
    match wire::read(&mut recv).await? {
        Frame::Error { .. } => {}
        Frame::InputInvoice { .. } => anyhow::bail!("invoiced a request that never generated"),
        Frame::Output { .. } => anyhow::bail!("unpaid backend output was released"),
        _ => anyhow::bail!("unexpected frame after backend rejection"),
    }
    assert!(started.elapsed() < Duration::from_secs(5));
    let (_serving_connection, result) = serving.await??;
    assert!(result.is_err());
    backend.await??;
    assert!(service.output_receivable(&id).await?.is_none());
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    assert!(network.entries.lock().unwrap().is_empty());
    caller.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}
