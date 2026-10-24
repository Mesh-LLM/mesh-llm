//! Exercise the legacy refusal over actual authenticated QUIC, not only prefix matching.
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_upgrades_are_refused_before_admission_or_backend_dispatch() -> Result<()> {
    for version in [1, 2] {
        tokio::time::timeout(Duration::from_secs(20), refuse_legacy(version)).await??;
    }
    Ok(())
}

async fn refuse_legacy(version: u8) -> Result<()> {
    let network = Arc::new(Network::default());
    let pair = Pair::new(network.clone()).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let (targets, receiver) =
        tokio::sync::watch::channel(local_targets(listener.local_addr()?.port()));
    let node = pair.seller_node.clone();
    let legacy = format!(
        "POST /mesh/payment/v{version} HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n"
    );
    let prefix_len = legacy.len();
    let serving = tokio::spawn(async move {
        let connection = node.endpoint.accept().await.unwrap().await?;
        let (send, mut recv) = accept_tunnel(&connection).await?;
        let mut prefix = vec![0; prefix_len];
        recv.read_exact(&mut prefix).await?;
        assert!(matches!(
            crate::network::payments::intercept_inbound(
                &node,
                connection.remote_id(),
                Some(&receiver),
                prefix,
                recv,
                send,
            )
            .await?,
            crate::network::payments::Inbound::Handled
        ));
        Ok::<_, anyhow::Error>(connection)
    });
    let connection = pair
        .buyer_node
        .endpoint
        .connect(pair.seller_node.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    send.write_u8(crate::protocol::STREAM_TUNNEL_HTTP).await?;
    send.write_all(legacy.as_bytes()).await?;
    let frame = wire::read(&mut recv).await?;
    assert!(
        matches!(frame, Frame::Error { message } if message.starts_with("payment_refused:protocol_version;"))
    );
    let _server_connection = serving.await??;
    assert!(pair.seller.ledger.metered_owner("r")?.is_none());
    assert!(pair.seller.ledger.metered_invoices(None)?.is_empty());
    assert!(pair.buyer.ledger.requests()?.is_empty());
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    assert!(
        network.entries.lock().unwrap().is_empty(),
        "no invoice issued"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err(),
        "backend must not be dispatched"
    );
    drop(targets);
    pair.close().await;
    Ok(())
}
