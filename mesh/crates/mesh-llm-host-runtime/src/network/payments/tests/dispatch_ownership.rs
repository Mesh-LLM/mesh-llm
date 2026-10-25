//! Concurrent admission replays must never abandon the dispatch winner.
//! In-memory streams exercise the real host and payments provider without
//! timing sleeps, external wallets, or a model runtime.
use super::*;
use mesh_llm_payments_types::contract::ServeRecoverResponse;
use mesh_llm_payments_types::metered_wire::AdmissionTerms;

struct Exchange {
    stream: tokio::io::DuplexStream,
    serving: tokio::task::JoinHandle<Result<()>>,
    terms: AdmissionTerms,
    invoice: Option<Invoice>,
}

impl Exchange {
    async fn admit(pair: &Pair, port: u16, id: &str) -> Result<Self> {
        let (mut stream, server) = tokio::io::duplex(64 * 1024);
        let node = pair.seller_node.clone();
        let peer = pair.buyer_node.id();
        let serving = tokio::spawn(async move {
            let (reader, writer) = tokio::io::split(server);
            super::super::server::serve(node, peer, reader, writer, local_targets(port)).await
        });
        wire::write(
            &mut stream,
            &Frame::Request {
                id: id.into(),
                model: "test".into(),
                pricing: price(),
                http: b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}".to_vec(),
            },
        )
        .await?;
        let Frame::Admission { terms, invoice } = wire::read(&mut stream).await? else {
            anyhow::bail!("expected admission");
        };
        Ok(Self {
            stream,
            serving,
            terms,
            invoice,
        })
    }

    async fn cancel(mut self) -> Result<()> {
        wire::write(&mut self.stream, &Frame::Cancel {}).await?;
        assert!(matches!(
            wire::read(&mut self.stream).await?,
            Frame::Error { .. }
        ));
        assert!(self.serving.await?.is_err());
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn losing_replay_and_preclaim_cancel_preserve_the_winners_final_invoice() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), concurrent_replays()).await?
}

async fn concurrent_replays() -> Result<()> {
    let network = Arc::new(Network::default());
    let pair = Pair::new(network.clone()).await?;
    let wallet = TestWallet {
        owner: 2,
        network: network.clone(),
    };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let id = uuid::Uuid::new_v4().to_string();
    let (mut winner, loser, cancelled) = deposit_replays(&pair, port, &id, &wallet).await?;
    wire::write(&mut winner.stream, &Frame::Accepted {}).await?;
    // Accepting the backend request proves A owns dispatch. Keep its response
    // blocked until both the pre-claim failure and losing claim have finished.
    let (mut backend, _) = listener.accept().await?;
    read_backend_request(&mut backend).await?;
    interrupt_replays(&pair, &id, loser, cancelled).await?;
    backend.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":3}}").await?;
    backend.shutdown().await?;
    let hash = pay_final(&mut winner.stream, &wallet, &id).await?;
    winner.serving.await??;
    assert_recovered_bill(&pair, &id, &hash).await?;
    assert_eq!(network.payments.load(Ordering::SeqCst), 2);
    assert!(
        std::future::poll_fn(|cx| {
            std::task::Poll::Ready(listener.poll_accept(cx).is_pending())
        })
        .await,
        "duplicate backend dispatch"
    );
    trusted_cancellation(&pair, port).await?;
    drop(winner.stream);

    pair.close().await;
    Ok(())
}

async fn deposit_replays(
    pair: &Pair,
    port: u16,
    id: &str,
    wallet: &TestWallet,
) -> Result<(Exchange, Exchange, Exchange)> {
    let winner = Exchange::admit(pair, port, id).await?;
    let loser = Exchange::admit(pair, port, id).await?;
    let cancelled = Exchange::admit(pair, port, id).await?;
    let deposit = winner.invoice.as_ref().unwrap();
    assert_eq!(
        loser.invoice.as_ref().unwrap().payment_hash,
        deposit.payment_hash
    );
    assert_eq!(
        cancelled.invoice.as_ref().unwrap().payment_hash,
        deposit.payment_hash
    );
    assert_eq!(winner.terms.id, loser.terms.id);
    wallet.pay(deposit, 1000, 1010).await?;
    Ok((winner, loser, cancelled))
}

/// Both failures run while the winner is waiting for backend output.
async fn interrupt_replays(
    pair: &Pair,
    id: &str,
    mut loser: Exchange,
    cancelled: Exchange,
) -> Result<()> {
    cancelled.cancel().await?;
    assert!(matches!(
        pair.seller.serve_recover(id).await?,
        ServeRecoverResponse::Pending
    ));
    wire::write(&mut loser.stream, &Frame::Accepted {}).await?;
    assert!(matches!(
        wire::read(&mut loser.stream).await?,
        Frame::Pending {}
    ));
    loser.serving.await??;
    assert!(matches!(
        pair.seller.serve_recover(id).await?,
        ServeRecoverResponse::Pending
    ));
    Ok(())
}

async fn pay_final(
    stream: &mut tokio::io::DuplexStream,
    wallet: &TestWallet,
    id: &str,
) -> Result<String> {
    let mut output = Vec::new();
    loop {
        match wire::read(stream).await? {
            Frame::Output { bytes } => output.extend(bytes),
            Frame::Settlement {
                request_id,
                usage,
                bill_msat,
                invoice,
                ..
            } => {
                assert_eq!(request_id, id);
                assert_eq!(bill_msat, Some(4300));
                assert!(usage.is_some());
                assert!(String::from_utf8(output)?.contains("completion_tokens"));
                let invoice = invoice.expect("winner must still issue a final invoice");
                assert_eq!(invoice.amount_msat, Some(3300));
                wallet.pay(&invoice, 3300, 3310).await?;
                assert!(matches!(wire::read(stream).await?, Frame::Complete {}));
                return Ok(invoice.payment_hash);
            }
            frame => anyhow::bail!("unexpected winner frame: {frame:?}"),
        }
    }
}

async fn assert_recovered_bill(pair: &Pair, id: &str, hash: &str) -> Result<()> {
    let ServeRecoverResponse::Billed { bill } = pair.seller.serve_recover(id).await? else {
        anyhow::bail!("winner was abandoned");
    };
    assert!(bill.settled);
    assert_eq!(bill.bill_msat, 4300);
    assert_eq!(bill.invoice.unwrap().payment_hash, hash);
    Ok(())
}

/// A trusted admission owns dispatch already: failing before Accepted must
/// still abandon its own row, unlike a deposit admission with no claim yet.
async fn trusted_cancellation(pair: &Pair, port: u16) -> Result<()> {
    let id = uuid::Uuid::new_v4().to_string();
    let trusted = Exchange::admit(pair, port, &id).await?;
    assert!(trusted.invoice.is_none());
    trusted.cancel().await?;
    assert!(matches!(
        pair.seller.serve_recover(&id).await?,
        ServeRecoverResponse::Abandoned
    ));
    Ok(())
}
