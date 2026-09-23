use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_expired_input_discards_buffered_output_without_output_debt() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), expired_exchange()).await?
}

async fn expired_backend(listener: tokio::net::TcpListener) -> Result<()> {
    let (mut stream, _) = listener.accept().await?;
    let mut raw = vec![0; 16 * 1024];
    let count = stream.read(&mut raw).await?;
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    request.parse(&raw[..count])?;
    let id = request
        .headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("x-request-id"))
        .unwrap();
    let id = uuid::Uuid::parse_str(std::str::from_utf8(id.value)?)?;
    let gate = skippy_server::frontend::generation_gate::registered_for_test(*id.as_bytes())
        .unwrap()
        .unwrap();
    tokio::task::spawn_blocking(move || {
        gate.after_prefill(40, 8).unwrap();
        gate.before_token().unwrap();
        gate.committed_token().unwrap();
        assert_eq!(gate.committed_tokens(), 1);
    })
    .await?;
    stream
        .write_all(
            b"HTTP/1.1 402 Payment Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await?;
    stream.shutdown().await?;
    Ok(())
}

async fn expired_exchange() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    network.invoice_expiry_seconds.store(2, Ordering::SeqCst);
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
    let caller = Node::new_for_tests(NodeRole::Client).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "test".into(),
        vec![InferenceTarget::Local(listener.local_addr()?.port())],
    );
    let backend = tokio::spawn(expired_backend(listener));
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
    let Frame::InputInvoice { invoice, .. } = wire::read(&mut recv).await? else {
        anyhow::bail!("expected invoice")
    };
    match wire::read(&mut recv).await? {
        Frame::Error { .. } => {}
        Frame::Output { .. } => anyhow::bail!("unpaid buffered output was released"),
        _ => anyhow::bail!("unexpected frame after unpaid prefill"),
    }
    let (_serving_connection, result) = serving.await??;
    assert!(result.is_err());
    backend.await??;
    assert!(mesh_llm_payments::now_ms() >= invoice.expires_at_ms);
    let (_, _, tokens, finished) = service.ledger.serving_account(&id)?;
    assert!(finished);
    assert_eq!(tokens, 0);
    assert!(service.output_receivable(&id).await?.is_none());
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    assert_lapses_after_grace(
        &service,
        &caller.endpoint.id().to_string(),
        &id,
        &invoice,
        &price,
    )?;
    caller.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

// Right after expiry the payment may still be being observed, so the peer
// stays blocked; past the grace, a zero-delivery request lapses and the buyer
// is admitted again.
fn assert_lapses_after_grace(
    service: &mesh_llm_payments::service::PaymentService,
    caller_id: &str,
    id: &str,
    invoice: &mesh_llm_payments::invoice::Invoice,
    price: &mesh_llm_payments::pricing::Pricing,
) -> Result<()> {
    assert!(service.ledger.has_outstanding_payment(caller_id)?);
    let grace = mesh_llm_payments::lifetimes::INPUT_LAPSE_GRACE.as_millis() as u64;
    assert!(
        service
            .ledger
            .lapse_abandoned_input(id, invoice, invoice.expires_at_ms + grace)?
    );
    assert!(!service.ledger.has_outstanding_payment(caller_id)?);
    service
        .ledger
        .begin_serving("another", caller_id, price, 8)?;
    Ok(())
}
