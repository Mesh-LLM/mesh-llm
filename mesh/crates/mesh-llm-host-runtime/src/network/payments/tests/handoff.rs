//! The HTTP route's hand-off to the metered exchange: once the buyer has
//! accepted an admission, ordinary retries never create a second bill, the
//! application sees a 402 when the seller fails after that, a seller that keeps
//! the deposit without output is struck, and a seller that refuses admission
//! lets the router reach the next paid seller.
use super::*;
use mesh_llm_payments_types::metered_wire::AdmissionTerms;
use mesh_llm_payments_types::metering::PAYMENT_FLOOR_MSAT;

/// How the fake seller ends a handed-off exchange.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SellerEnding {
    /// Hangs up mid-exchange, which the buyer sees as EOF.
    Drop,
    /// Settles as abandoned and closes with `Complete`, having sent no output.
    EmptyComplete,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payment_handoff_preserves_correlation_and_never_retries_after_acceptance() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        handed_off(false, false, SellerEnding::Drop),
    )
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seller_that_completes_without_output_is_struck() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        handed_off(false, false, SellerEnding::EmptyComplete),
    )
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deposit_inflight_at_seller_drop_is_terminal_and_retains_reservation() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        handed_off(true, false, SellerEnding::Drop),
    )
    .await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admission_refusal_uses_existing_router_to_reach_second_paid_seller() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        handed_off(false, true, SellerEnding::Drop),
    )
    .await?
}

async fn handed_off(hold: bool, fallback: bool, ending: SellerEnding) -> Result<()> {
    let network = Arc::new(Network::default());
    network.hold_payments.store(hold, Ordering::SeqCst);
    let pair = Pair::new(network.clone()).await?;
    let (listener, mut application, mut client) = client_socket().await?;
    let attempts = Arc::new(AtomicUsize::new(0));
    let server = tokio::spawn(fake_seller(
        pair.seller_node.clone(),
        network.clone(),
        pair.buyer.clone(),
        hold,
        ending,
        fallback.then(|| attempts.clone()),
    ));
    pair.connect().await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"hi\",\"max_tokens\":8}";
    let mut refusing = None;
    let retryable = if fallback {
        let (first, task) = refusing_seller(&pair.buyer_node, attempts.clone()).await?;
        let result = crate::network::openai::transport::test_paid_multi_target(
            pair.buyer_node.clone(),
            client,
            vec![first.id(), pair.seller_node.id()],
        )
        .await;
        refusing = Some((first, task));
        client = tokio::net::TcpStream::connect(listener.local_addr()?)
            .await?
            .into();
        matches!(
            result,
            crate::network::openai::transport::RouteDispatchOutcome::Dropped(_)
        )
    } else {
        crate::network::openai::transport::test_paid_target_attempt(
            &pair.buyer_node,
            &mut client,
            pair.seller_node.id(),
            raw,
            "wrapper-exchange",
        )
        .await
    };
    verify_handoff(&pair.buyer, &network, retryable, fallback, hold).await?;
    // The seller took the deposit and closed without output: that is a
    // paid-but-undelivered strike against it. With `hold`, the deposit only
    // settles after the drop, so the exchange never observed it: no strike.
    //
    // Asserted while this client is still connected, because dropping it first
    // would cancel the request from our side and legitimately exempt the
    // seller — the test would then be measuring its own timing.
    if !hold {
        assert_paid_undelivered_strike(&pair.buyer_node, &pair.seller_node).await?;
    }
    drop(client);
    let mut response = String::new();
    application.read_to_string(&mut response).await?;
    assert!(response.contains("402"), "{response}");
    server.await??;
    if let Some((first, task)) = refusing {
        first.endpoint.close().await;
        task.await??;
    }
    pair.close().await;
    Ok(())
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

async fn assert_paid_undelivered_strike(buyer: &Node, seller: &Node) -> Result<()> {
    let directory = buyer.config_state.lock().await.payment_directory();
    let now = mesh_llm_wallet::now_ms();
    use crate::network::payments::strikes::{PayeeStrikes, STRIKE_THRESHOLD};
    let mut recorded = false;
    for _ in 0..200 {
        let mut probe = PayeeStrikes::load(&directory);
        // One real strike plus THRESHOLD-1 synthetic ones reaches a block
        // only if the real one was recorded for this seller.
        for _ in 1..STRIKE_THRESHOLD {
            probe.record(&seller.id().to_string(), now);
        }
        if probe.is_blocked(&seller.id().to_string(), now) {
            recorded = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(recorded, "paid-but-undelivered strike was not recorded");
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

type SellerTask = tokio::task::JoinHandle<Result<()>>;

/// A paid seller that refuses every admission with an `Error` frame, as a
/// seller whose backend is unavailable does, before any payment state exists.
async fn refusing_seller(buyer: &Node, attempts: Arc<AtomicUsize>) -> Result<(Node, SellerTask)> {
    let first = Node::new_for_tests(NodeRole::Client).await?;
    let mut announcement =
        first.build_local_announcement(first.snapshot_local_announcement_data().await);
    announcement.lightning_offers.insert(
        "test".into(),
        Pricing {
            input_msat_per_million: 500_000,
            output_msat_per_million: 500_000,
            cache_discount_percent: 0,
        },
    );
    buyer
        .add_peer_after_direct_requirements_validated(
            first.id(),
            first.endpoint.addr(),
            &announcement,
            Some(1),
        )
        .await;
    let refusing_node = first.clone();
    let task = tokio::spawn(async move {
        let conn = refusing_node.endpoint.accept().await.unwrap().await?;
        while let Ok((mut send, mut recv)) = accept_tunnel(&conn).await {
            let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
            recv.read_exact(&mut prefix).await?;
            let _ = wire::read(&mut recv).await?;
            attempts.fetch_add(1, Ordering::SeqCst);
            wire::write(
                &mut send,
                &Frame::Error {
                    message: "admission refused".into(),
                },
            )
            .await?;
            send.finish()?;
        }
        Ok::<_, anyhow::Error>(())
    });
    let conn = buyer
        .endpoint
        .connect(first.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    buyer
        .state
        .lock()
        .await
        .connections
        .insert(first.id(), conn);
    Ok((first, task))
}

/// A seller speaking the metered wire by hand: admits with a deposit, waits
/// for it, then ends the exchange as `ending` says without any output.
async fn fake_seller(
    seller: Node,
    network: Arc<Network>,
    buyer: Arc<PaymentService>,
    hold: bool,
    ending: SellerEnding,
    prior_attempts: Option<Arc<AtomicUsize>>,
) -> Result<()> {
    let connection = seller.endpoint.accept().await.unwrap().await?;
    let (mut send, mut recv) = accept_tunnel(&connection).await?;
    let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
    recv.read_exact(&mut prefix).await?;
    anyhow::ensure!(
        prefix == wire::HTTP_UPGRADE,
        "unexpected tunnel prefix: {:?}",
        String::from_utf8_lossy(&prefix)
    );
    let Frame::Request {
        id, pricing, http, ..
    } = wire::read(&mut recv).await?
    else {
        anyhow::bail!("request expected")
    };
    if let Some(attempts) = prior_attempts {
        assert!(
            attempts.load(Ordering::SeqCst) > 0,
            "refusing seller must be attempted first"
        );
    }
    crate::network::payments::request::PaidRequest::parse(&http)?;
    let deposit = PAYMENT_FLOOR_MSAT;
    let wallet = TestWallet {
        owner: 1,
        network: network.clone(),
    };
    let invoice = wallet.create_invoice(Some(deposit), 3600).await?;
    // The deposit invoice's hash keys the charge in the buyer's ledger.
    let hash = invoice.payment_hash.clone();
    wire::write(
        &mut send,
        &Frame::Admission {
            terms: AdmissionTerms {
                id: id.clone(),
                model: "test".into(),
                pricing,
                deposit_msat: deposit,
                min_payment_msat: PAYMENT_FLOOR_MSAT,
            },
            invoice: Some(invoice),
        },
    )
    .await?;
    anyhow::ensure!(matches!(wire::read(&mut recv).await?, Frame::Accepted {}));
    wait_for_deposit(&buyer, &hash, hold).await?;
    if ending == SellerEnding::EmptyComplete {
        wire::write(
            &mut send,
            &Frame::Settlement {
                request_id: id,
                usage: None,
                bill_msat: None,
                deposit_applied_msat: 0,
                invoice: None,
            },
        )
        .await?;
        wire::write(&mut send, &Frame::Complete {}).await?;
    }
    send.finish()?;
    let _ = send.stopped().await;
    Ok(())
}

/// With `hold`, wait only for the deposit to be prepared. Otherwise hang up
/// only once the buyer's own ledger has settled the deposit, and then let the
/// exchange observe that settlement: closing as soon as the wallet reports the
/// send decides the strike on a scheduling race, and the seller dropping
/// before the buyer observes settlement is the documented no-strike gap.
async fn wait_for_deposit(buyer: &PaymentService, hash: &str, hold: bool) -> Result<()> {
    if hold {
        while buyer.ledger.pending_charges()?.is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        return Ok(());
    }
    while buyer.ledger.charge_state(hash)?.as_deref() != Some("succeeded") {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    Ok(())
}
