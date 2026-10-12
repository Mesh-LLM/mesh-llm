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
        let id = text
            .lines()
            .find_map(|l| l.strip_prefix("x-request-id: "))
            .unwrap();
        let id = uuid::Uuid::parse_str(id.trim())?;
        assert!(
            skippy_server::frontend::generation_gate::registered_for_test(*id.as_bytes())
                .unwrap()
                .is_none()
        );
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"hello\"}]}\n\n").await?;
        tokio::task::yield_now().await;
        stream.write_all(b"data: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n").await?;
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
    for completed in 1..=2 {
        let (mut pipe, ready, _cancel) = crate::network::openai::test_metered_start(
            payer,
            provider.id(),
            raw,
            price.clone(),
            None,
        )
        .await?;
        ready.await?;
        let mut response = String::new();
        pipe.read_to_string(&mut response).await?;
        assert!(response.contains("hello"));
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
        assert_eq!(seller.ledger.metered_exposure(&payer.id().to_string())?, 0);
    }
    Ok(())
}
