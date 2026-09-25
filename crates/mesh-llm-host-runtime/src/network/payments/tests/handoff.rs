use super::*;
use mesh_llm_payments::ledger::RequestTerms;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payment_handoff_preserves_correlation_and_never_retries_after_submission() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), handed_off(false, false)).await?
}

async fn handed_off(hold: bool, fallback: bool) -> Result<()> {
    let (_dir, network, service, payer) = payer_fixture(hold).await?;
    let (provider, price) = advertised_provider(&payer).await?;
    let (listener, mut application, mut client) = client_socket().await?;
    let attempts = Arc::new(AtomicUsize::new(0));
    let server = tokio::spawn(fake_paid_provider(
        provider.clone(),
        network.clone(),
        price,
        service.clone(),
        hold,
        fallback.then(|| attempts.clone()),
    ));
    cache_connection(&payer, &provider).await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"hi\",\"max_tokens\":8}";
    let mut failing = None;
    let retryable = if fallback {
        let (first, task) = failing_provider(&payer, attempts.clone()).await?;
        let result = crate::network::openai::transport::test_paid_multi_target(
            payer.clone(),
            client,
            vec![first.id(), provider.id()],
        )
        .await;
        failing = Some((first, task));
        client = tokio::net::TcpStream::connect(listener.local_addr()?)
            .await?
            .into();
        matches!(
            result,
            crate::network::openai::transport::RouteDispatchOutcome::Dropped(_)
        )
    } else {
        crate::network::openai::transport::test_paid_target_attempt(
            &payer,
            &mut client,
            provider.id(),
            raw,
            "wrapper-exchange",
        )
        .await
    };
    verify_handoff(&service, &network, retryable, fallback, hold).await?;
    drop(client);
    let mut response = String::new();
    application.read_to_string(&mut response).await?;
    assert!(response.contains("402"));
    server.await??;
    if let Some((first, task)) = failing {
        first.endpoint.close().await;
        task.await??;
    }
    payer.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

async fn cache_connection(payer: &Node, provider: &Node) -> Result<()> {
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
    Ok(())
}

async fn client_socket() -> Result<(
    tokio::net::TcpListener,
    tokio::net::TcpStream,
    crate::network::openai::client_stream::ClientStream,
)> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let application = tokio::net::TcpStream::connect(listener.local_addr()?).await?;
    let (client, _) = listener.accept().await?;
    Ok((listener, application, client.into()))
}

async fn payer_fixture(
    hold: bool,
) -> Result<(tempfile::TempDir, Arc<Network>, Arc<PaymentService>, Node)> {
    let dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    network.hold_payments.store(hold, Ordering::SeqCst);
    let service = Arc::new(PaymentService::with_provider(
        dir.path(),
        Arc::new(TestWallet {
            owner: 2,
            network: network.clone(),
        }),
    )?);
    allow_paid(&service)?;
    let payer = Node::new_for_tests(NodeRole::Client).await?;
    payer
        .payments
        .set(service.clone())
        .map_err(|_| anyhow::anyhow!("already set"))?;
    crate::network::payments::node_ext::attach_payments_plugin(&payer).await?;
    Ok((dir, network, service, payer))
}

async fn advertised_provider(payer: &Node) -> Result<(Node, Pricing)> {
    let provider = Node::new_for_tests(NodeRole::Client).await?;
    provider.set_models(vec!["test".into()]).await;
    provider.set_serving_models(vec!["test".into()]).await;
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    let mut announcement =
        provider.build_local_announcement(provider.snapshot_local_announcement_data().await);
    announcement
        .lightning_offers
        .insert("test".into(), price.clone());
    payer
        .add_peer_after_direct_requirements_validated(
            provider.id(),
            provider.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
    Ok((provider, price))
}

async fn verify_handoff(
    service: &PaymentService,
    network: &Network,
    retryable: bool,
    fallback: bool,
    hold: bool,
) -> Result<()> {
    assert!(
        !retryable,
        "post-handoff drop must not reach normal retries"
    );
    assert!(
        !service.ledger.requests()?.is_empty(),
        "routing did not reach payment handoff"
    );
    assert_eq!(
        service.ledger.requests()?[0].terms.exchange_id.as_deref(),
        Some(if fallback {
            "multi-provider-exchange"
        } else {
            "wrapper-exchange"
        })
    );
    if hold {
        assert_eq!(network.payments.load(Ordering::SeqCst), 0);
        assert_eq!(service.ledger.pending_charges()?.len(), 1);
        network.hold_payments.store(false, Ordering::SeqCst);
        network.payment_release.notify_waiters();
    }
    while network.payments.load(Ordering::SeqCst) != 1 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    Ok(())
}

type FailingProviderTask = tokio::task::JoinHandle<Result<()>>;

async fn failing_provider(
    payer: &Node,
    attempts: Arc<AtomicUsize>,
) -> Result<(Node, FailingProviderTask)> {
    let first = Node::new_for_tests(NodeRole::Client).await?;
    let mut announcement =
        first.build_local_announcement(first.snapshot_local_announcement_data().await);
    announcement.lightning_offers.insert(
        "test".into(),
        Pricing {
            input_msat_per_million: 500_000,
            output_msat_per_million: 500_000,
            minimum_invoice_msat: 1,
        },
    );
    payer
        .add_peer_after_direct_requirements_validated(
            first.id(),
            first.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
    let failing_node = first.clone();
    let task = tokio::spawn(async move {
        let conn = failing_node.endpoint.accept().await.unwrap().await?;
        while let Ok((mut send, mut recv)) = accept_payment_stream(&conn).await {
            let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
            recv.read_exact(&mut prefix).await?;
            let _ = wire::read(&mut recv).await?;
            attempts.fetch_add(1, Ordering::SeqCst);
            wire::write(
                &mut send,
                &Frame::Error {
                    message: "prefill unavailable".into(),
                },
            )
            .await?;
            send.finish()?;
        }
        Ok::<_, anyhow::Error>(())
    });
    let conn = payer
        .endpoint
        .connect(first.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    payer
        .state
        .lock()
        .await
        .connections
        .insert(first.id(), conn);
    Ok((first, task))
}

async fn fake_paid_provider(
    provider: Node,
    network: Arc<Network>,
    price: Pricing,
    service: Arc<PaymentService>,
    hold: bool,
    prior_attempts: Option<Arc<AtomicUsize>>,
) -> Result<()> {
    let connection = provider.endpoint.accept().await.unwrap().await?;
    let (mut send, mut recv) = accept_payment_stream(&connection).await?;
    let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
    recv.read_exact(&mut prefix).await?;
    anyhow::ensure!(
        prefix == wire::HTTP_UPGRADE,
        "unexpected tunnel prefix: {:?}",
        String::from_utf8_lossy(&prefix)
    );
    let Frame::Request { id, .. } = wire::read(&mut recv).await? else {
        anyhow::bail!("request expected")
    };
    if let Some(attempts) = prior_attempts {
        assert!(
            attempts.load(Ordering::SeqCst) > 0,
            "failing provider must be attempted first"
        );
    }
    let seller = TestWallet {
        owner: 1,
        network: network.clone(),
    };
    let invoice = seller.create_invoice(Some(1), 3600).await?;
    let terms = RequestTerms {
        exchange_id: None,
        id,
        peer: String::new(),
        payee: None,
        model: "test".into(),
        max_total_msat: price.request_cap_msat(price.input_charge(1)?, 8)?,
        pricing: price,
        input_tokens: 1,
        max_output_tokens: 8,
        expires_at_ms: invoice.expires_at_ms,
    };
    wire::write(&mut send, &Frame::InputInvoice { terms, invoice }).await?;
    while if hold {
        service.ledger.pending_charges()?.is_empty()
    } else {
        network.payments.load(Ordering::SeqCst) == 0
    } {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    send.finish()?;
    let _ = send.stopped().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payment_inflight_at_provider_drop_is_terminal_and_retains_reservation() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), handed_off(true, false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefill_failure_uses_existing_router_to_reach_second_paid_provider() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), handed_off(false, true)).await?
}

async fn accept_payment_stream(
    connection: &iroh::endpoint::Connection,
) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
    loop {
        let (send, mut recv) = connection.accept_bi().await?;
        let kind = recv.read_u8().await?;
        if kind == crate::protocol::STREAM_TUNNEL_HTTP {
            return Ok((send, recv));
        }
        // Peer announcements may open a gossip stream before the HTTP tunnel.
        drop(send);
        drop(recv);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_ignores_alternates_and_original_return_completes_existing_request() -> Result<()>
{
    tokio::time::timeout(Duration::from_secs(15), recover_original()).await?
}

async fn recover_original() -> Result<()> {
    let (_dir, network, service, payer) = payer_fixture(false).await?;
    let original = Node::new_for_tests(NodeRole::Client).await?;
    let (alternate, price) = advertised_provider(&payer).await?;
    let terms = RequestTerms {
        exchange_id: None,
        id: "recover-original".into(),
        peer: original.id().to_string(),
        payee: None,
        model: "test".into(),
        max_total_msat: price.request_cap_msat(price.input_charge(1)?, 8)?,
        pricing: price,
        input_tokens: 1,
        max_output_tokens: 8,
        expires_at_ms: u64::MAX,
    };
    service.await_authorization(&terms).await?;
    crate::network::openai::payment_recovery::recover(&payer).await?;
    assert_eq!(
        service.ledger.request_state(&terms.id)?.as_deref(),
        Some("approved")
    );
    assert_eq!(
        service.ledger.policy_status(mesh_llm_payments::now_ms())?["reserved_msat"],
        terms.max_total_msat
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), alternate.endpoint.accept())
            .await
            .is_err()
    );
    let announcement =
        original.build_local_announcement(original.snapshot_local_announcement_data().await);
    payer
        .add_peer_after_direct_requirements_validated(
            original.id(),
            original.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
    let accepting = original.clone();
    let server = tokio::spawn(async move {
        let connection = accepting.endpoint.accept().await.unwrap().await?;
        let (mut send, mut recv) = accept_payment_stream(&connection).await?;
        let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
        recv.read_exact(&mut prefix).await?;
        assert!(
            matches!(wire::read(&mut recv).await?, Frame::Recover { id } if id == "recover-original")
        );
        wire::write(&mut send, &Frame::Complete).await?;
        send.finish()?;
        let _ = send.stopped().await;
        Ok::<_, anyhow::Error>(())
    });
    cache_connection(&payer, &original).await?;
    crate::network::openai::payment_recovery::recover(&payer).await?;
    assert_eq!(
        service.ledger.request_state(&terms.id)?.as_deref(),
        Some("completed")
    );
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    server.await??;
    payer.endpoint.close().await;
    original.endpoint.close().await;
    alternate.endpoint.close().await;
    Ok(())
}
