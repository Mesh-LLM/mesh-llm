//! Real payer and confirmed simulated deposit, failing provider, then paid selection.
use super::*;
use crate::network::openai::{test_metered_start, test_payment_rank};
use anyhow::Context;
use mesh_llm_payments_types::{metered_wire as v2, metering::PromptEstimate};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn third_paid_nondelivery_excludes_provider_from_next_selection() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), qualifying_failures()).await?
}

async fn qualifying_failures() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let buyer_dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let buyer = Arc::new(PaymentService::with_provider(
        buyer_dir.path(),
        Arc::new(TestWallet {
            owner: 2,
            network: network.clone(),
        }),
    )?);
    buyer.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(1_000_000),
    })?;
    let wallet = Arc::new(TestWallet {
        owner: 1,
        network: network.clone(),
    });
    let provider = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
    let payer = Node::new_for_tests(NodeRole::Client).await?;
    *payer.config_state.lock().await =
        crate::runtime::config_state::ConfigState::load(&directory.path().join("config.toml"))?;
    payer
        .payments
        .set(buyer.clone())
        .map_err(|_| anyhow::anyhow!("already set"))?;
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
    let (finished, mut failures) = tokio::sync::mpsc::channel(1);
    let serving = tokio::spawn(failing_provider(provider.clone(), wallet, finished));
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
    for failure in 1..=3 {
        assert_paid_selection(&payer, provider.id(), false).await?;
        consume_failed_exchange(&payer, provider.id(), raw, &price).await?;
        failures.recv().await.context("provider ended early")?;
        await_persisted_failure(&payer, provider.id(), failure).await?;
        // Undelivered exchanges retain their authorizations; fund the next
        // attempt rather than clearing or forgiving that uncertainty.
        network.top_up_msat.fetch_add(100_000, Ordering::SeqCst);
        assert_paid_selection(&payer, provider.id(), failure == 3).await?;
    }
    let _connection = serving.await??;
    assert_eq!(buyer.ledger.requests()?.len(), 3);
    payer.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

async fn consume_failed_exchange(
    payer: &Node,
    provider: iroh::EndpointId,
    raw: &[u8],
    price: &Pricing,
) -> Result<()> {
    let (mut pipe, ready, _cancel) =
        test_metered_start(payer, provider, raw, price.clone(), None).await?;
    ready.await?;
    let mut bytes = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), pipe.read_to_end(&mut bytes))
        .await
        .context("output did not close")??;
    assert!(bytes.is_empty());
    Ok(())
}

async fn await_persisted_failure(
    payer: &Node,
    provider: iroh::EndpointId,
    failure: usize,
) -> Result<()> {
    // Output EOF precedes the detached strike write. Wait for that exact
    // persisted outcome, not a fixed delay or an assumed task ordering.
    let path = payer
        .config_state
        .lock()
        .await
        .payment_directory()
        .join("payee_strikes.json");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(bytes) = std::fs::read(&path) {
                let state: serde_json::Value = serde_json::from_slice(&bytes)?;
                let entry = &state["payees"][provider.to_string()];
                let count = entry["strikes"].as_array().map_or(0, Vec::len);
                if count == failure || entry["blocked_until_ms"].is_number() {
                    return Ok::<_, anyhow::Error>(());
                }
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .context("strike did not persist")?
}

async fn assert_paid_selection(
    payer: &Node,
    provider: iroh::EndpointId,
    blocked: bool,
) -> Result<()> {
    let (result, ordered, _) =
        test_payment_rank(payer, vec![InferenceTarget::Remote(provider)]).await;
    if blocked {
        assert!(result.unwrap_err().contains("blocked"));
        assert!(ordered.is_empty());
        let free = iroh::SecretKey::generate().public();
        let (result, ordered, prefix) = test_payment_rank(
            payer,
            vec![
                InferenceTarget::Remote(provider),
                InferenceTarget::Remote(free),
            ],
        )
        .await;
        assert!(!result.unwrap());
        assert_eq!(ordered, vec![InferenceTarget::Remote(free)]);
        assert_eq!(prefix, 1);
    } else {
        assert!(result.unwrap());
        assert_eq!(ordered, vec![InferenceTarget::Remote(provider)]);
    }
    Ok(())
}

async fn failing_provider(
    node: Node,
    wallet: Arc<TestWallet>,
    finished: tokio::sync::mpsc::Sender<()>,
) -> Result<iroh::endpoint::Connection> {
    let connection = node
        .endpoint
        .accept()
        .await
        .context("no connection")?
        .await?;
    for _ in 0..3 {
        let (mut send, mut recv) = loop {
            let (send, mut recv) = connection.accept_bi().await?;
            if recv.read_u8().await? == crate::protocol::STREAM_TUNNEL_HTTP {
                break (send, recv);
            }
        };
        let mut prefix = vec![0; v2::HTTP_UPGRADE.len()];
        recv.read_exact(&mut prefix).await?;
        assert_eq!(prefix, v2::HTTP_UPGRADE);
        let v2::Frame::Request {
            id,
            model,
            pricing,
            max_total_msat,
            http,
        } = v2::read(&mut recv).await?
        else {
            anyhow::bail!("expected request")
        };
        let request = super::super::request::PaidRequest::parse(&http)?;
        let estimate = PromptEstimate::from_request(&request.body)?;
        let deposit =
            u64::try_from((u128::from(pricing.input_charge(estimate.tokens)?) * 3).div_ceil(10))?;
        let invoice = wallet.create_invoice(Some(deposit), 3600).await?;
        v2::write(
            &mut send,
            &v2::Frame::Admission {
                terms: v2::AdmissionTerms {
                    id,
                    model,
                    pricing,
                    max_total_msat,
                    prompt_estimate: estimate.tokens,
                    max_output_tokens: u64::from(request.max_tokens.unwrap()),
                    deposit_msat: deposit,
                    credit_msat: 0,
                },
                invoice: Some(invoice.clone()),
            },
        )
        .await?;
        assert!(matches!(v2::read(&mut recv).await?, v2::Frame::Accepted {}));
        assert_eq!(
            wallet.wait_for_payment(&invoice.payment_hash).await?.status,
            PaymentStatus::Succeeded
        );
        // Deliberate protocol failure after settled money, with no Output.
        v2::write(
            &mut send,
            &v2::Frame::Error {
                message: "paid but not delivered".into(),
            },
        )
        .await?;
        finished.send(()).await?;
    }
    Ok(connection)
}
