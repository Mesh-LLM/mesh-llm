//! Recovery by request ID: a settled request's settlement is replayed and
//! its final invoice paid; an unknown request is refused; a request still
//! in progress answers Pending; nothing is ever dispatched twice.
use super::*;
use mesh_llm_payments_types::contract::{
    IdRequest, ReconcileResponse, RequestIdRequest, ServeRecoverResponse, ops,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_buyer_recovers_a_settled_request_and_pays_its_final_invoice() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), recovered()).await?
}

async fn recovered() -> Result<()> {
    let network = Arc::new(Network::default());
    let pair = Pair::new(network.clone()).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let backend = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_backend_request(&mut stream).await?;
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 52\r\nConnection: close\r\n\r\n{\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":3}}").await?;
        stream.shutdown().await?;
        Ok::<_, anyhow::Error>(())
    });
    // Hold outgoing payments: the deposit never lands, so the seller never
    // dispatches, and the buyer's exchange is interrupted before settlement.
    // The second exchange is the buyer's recovery of the first.
    let serving = pair.serve(local_targets(port), 2);
    pair.connect().await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}";
    let response = pair.request(raw, None).await?;
    assert!(response.contains("completion_tokens"), "{response}");
    pair.wait_completed(1).await?;
    assert_eq!(network.payments.load(Ordering::SeqCst), 2);
    let id = pair.buyer.ledger.requests()?[0].terms.id.clone();
    assert_seller_answers_recovery_from_its_ledger(&pair, &id).await?;
    assert_buyer_replays_a_settled_request(&pair, &id).await?;
    assert_eq!(network.payments.load(Ordering::SeqCst), 2);
    backend.await??;
    let _connection = serving.await??;
    pair.close().await;
    Ok(())
}

/// The seller answers a recovery probe from its ledger. The final invoice is
/// already paid, so the bill is settled; unknown requests are refused.
async fn assert_seller_answers_recovery_from_its_ledger(pair: &Pair, id: &str) -> Result<()> {
    let payments = super::super::client::Payments::for_node(&pair.seller_node).await?;
    let recovered: ServeRecoverResponse = payments
        .call(
            ops::SERVE_RECOVER,
            &RequestIdRequest {
                request_id: id.to_owned(),
            },
        )
        .await?;
    let ServeRecoverResponse::Billed { bill } = recovered else {
        panic!("expected a bill, got {recovered:?}");
    };
    assert_eq!(bill.bill_msat, 43);
    assert!(bill.settled);
    let unknown = payments
        .call::<_, ServeRecoverResponse>(
            ops::SERVE_RECOVER,
            &RequestIdRequest {
                request_id: uuid::Uuid::new_v4().to_string(),
            },
        )
        .await;
    assert!(unknown.is_err(), "unknown requests are refused");
    Ok(())
}

/// The buyer re-runs recovery over the wire for a finished request: the
/// seller replays the settlement, nothing is dispatched or paid again, and a
/// finished request needs no recovery pass.
async fn assert_buyer_replays_a_settled_request(pair: &Pair, id: &str) -> Result<()> {
    let payments = super::super::client::Payments::for_node(&pair.buyer_node).await?;
    let reconciled: ReconcileResponse = payments
        .call(ops::RECONCILE, &mesh_llm_payments_types::contract::Empty {})
        .await?;
    assert!(
        reconciled.approved.is_empty(),
        "finished requests need no recovery"
    );
    let (mut send, mut recv) = pair
        .buyer_node
        .open_http_tunnel(pair.seller_node.id())
        .await?;
    send.write_all(wire::HTTP_UPGRADE).await?;
    wire::write(&mut send, &Frame::Recover { id: id.to_owned() }).await?;
    let Frame::Settlement {
        request_id,
        bill_msat,
        ..
    } = wire::read(&mut recv).await?
    else {
        panic!("expected settlement replay");
    };
    assert_eq!((request_id.as_str(), bill_msat), (id, Some(43)));
    assert!(matches!(wire::read(&mut recv).await?, Frame::Complete {}));
    let _: mesh_llm_payments_types::contract::Empty = payments
        .call(ops::FINISH, &IdRequest { id: id.to_owned() })
        .await?;
    Ok(())
}
