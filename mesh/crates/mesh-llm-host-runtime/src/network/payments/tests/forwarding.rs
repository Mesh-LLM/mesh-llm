use super::*;
use crate::network::{affinity::AffinityRouter, tunnel::Manager};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_three_nodes_cannot_spend_forwarding_wallet() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), remote_request(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_three_nodes_cannot_spend_through_legacy_bridge() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(15), remote_request(true)).await?
}

async fn remote_request(legacy: bool) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
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
        daily_budget_msat: Some(100_000),
    })?;
    let caller = Node::new_for_tests(NodeRole::Client).await?;
    let relay = Node::new_for_tests(NodeRole::Client).await?;
    let provider = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
    relay
        .payments
        .set(service.clone())
        .map_err(|_| anyhow::anyhow!("already initialized"))?;
    crate::network::payments::node_ext::attach_payments_plugin(&relay).await?;
    provider.set_models(vec!["test".into()]).await;
    provider.set_serving_models(vec!["test".into()]).await;
    let mut announcement =
        provider.build_local_announcement(provider.snapshot_local_announcement_data().await);
    announcement.lightning_offers.insert(
        "test".into(),
        Pricing {
            input_msat_per_million: 1_000_000,
            output_msat_per_million: 1_000_000,
            minimum_invoice_msat: 1,
        },
    );
    assert!(
        relay
            .add_peer_after_direct_requirements_validated(
                provider.endpoint.id(),
                provider.endpoint.addr(),
                &announcement,
                Some(1),
            )
            .await
    );
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "test".into(),
        vec![InferenceTarget::Remote(provider.endpoint.id())],
    );
    let (targets_tx, targets_rx) = tokio::sync::watch::channel(targets);
    let (_legacy_tx, legacy_rx) = tokio::sync::mpsc::channel(1);
    let (http_tx, http_rx) = tokio::sync::mpsc::channel(1);
    let (_stage_tx, stage_rx) = tokio::sync::mpsc::channel(1);
    let manager = Manager::start(relay.clone(), legacy_rx, http_rx, stage_rx).await?;
    let unused_backend = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    manager.set_http_port(unused_backend.local_addr()?.port());
    if !legacy {
        manager.set_http_ingress(targets_rx, AffinityRouter::default());
    }
    let (response, accepted_connection) = send_remote_request(&caller, &relay, http_tx).await?;
    assert_forwarding_denied(&response, legacy, &service, &network, &unused_backend).await?;
    drop(accepted_connection);
    drop(targets_tx);
    caller.endpoint.close().await;
    relay.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

type InboundHttp = (
    iroh::EndpointId,
    iroh::endpoint::SendStream,
    iroh::endpoint::RecvStream,
);

async fn send_remote_request(
    caller: &Node,
    relay: &Node,
    http_tx: tokio::sync::mpsc::Sender<InboundHttp>,
) -> Result<(String, iroh::endpoint::Connection)> {
    let accepting = relay.clone();
    let accepted = tokio::spawn(async move {
        let conn = accepting.endpoint.accept().await.unwrap().await?;
        let (send, recv) = conn.accept_bi().await?;
        http_tx.send((conn.remote_id(), send, recv)).await?;
        Ok::<_, anyhow::Error>(conn)
    });
    let conn = caller
        .endpoint
        .connect(relay.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    let body = r#"{"model":"test","prompt":"Hi","max_tokens":8}"#;
    // Spoofed loopback forwarding headers must not override authenticated QUIC origin.
    send.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nX-Forwarded-For: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await?;
    let response = String::from_utf8(recv.read_to_end(16 * 1024).await?)?;
    Ok((response, accepted.await??))
}

async fn assert_forwarding_denied(
    response: &str,
    legacy: bool,
    service: &PaymentService,
    network: &Network,
    unused_backend: &tokio::net::TcpListener,
) -> Result<()> {
    assert!(response.starts_with("HTTP/1.1 402"), "{response}");
    let message = if legacy {
        "payment-capable peer required"
    } else {
        "only locally originated requests may spend this wallet"
    };
    assert!(response.contains(message), "{response}");
    assert!(service.ledger.requests()?.is_empty());
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), unused_backend.accept())
            .await
            .is_err()
    );
    Ok(())
}
