//! Opt-in workflow qualification against real native serving and plugin registration.
use super::*;
use mesh_llm_payments_types::metered_wire as v2;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires task-owned Skippy server; MESH_METERING_LIVE_PORT and MODEL"]
async fn native_serving_first_and_repeat_settlement() -> Result<()> {
    let port = std::env::var("MESH_METERING_LIVE_PORT")?.parse()?;
    let model = std::env::var("MESH_METERING_LIVE_MODEL")?;
    tokio::time::timeout(Duration::from_secs(90), qualify(port, &model, None)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Python3; launches manifest-registered external inference fixture"]
async fn registered_plugin_first_and_repeat_settlement() -> Result<()> {
    // Fresh peers each round exercise admission and trusted repeat, rather
    // than only repeating requests against an already-established trust cache.
    for _ in 0..5 {
        tokio::time::timeout(Duration::from_secs(40), plugin_workflow()).await??;
    }
    Ok(())
}

async fn plugin_workflow() -> Result<()> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/network/payments/tests/fixtures/metered_plugin.py");
    let mode = crate::plugin::PluginHostMode {
        mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
    };
    // Use the normal external-process lifecycle, without starting unrelated
    // auto-installed or bundled plugins from the operator's machine.
    let specs = crate::plugin::ResolvedPlugins {
        externals: vec![crate::plugin::ExternalPluginSpec {
            name: "metering-fixture".into(),
            command: "python3".into(),
            args: vec![fixture.display().to_string(), port.to_string()],
            url: None,
            env: Default::default(),
            startup: Default::default(),
            web_ui_enabled: None,
            installed_metadata: None,
        }],
        inactive: Vec::new(),
    };
    let (tx, _rx) = tokio::sync::mpsc::channel(16);
    let manager = crate::plugin::PluginManager::start(&specs, mode, tx).await?;
    let result = qualify(port, "test", Some(manager.clone())).await;
    manager.shutdown().await;
    result
}

async fn qualify(
    port: u16,
    model: &str,
    plugin: Option<crate::plugin::PluginManager>,
) -> Result<()> {
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
    seller.ledger.set_pricing(model, Some(&price))?;
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
    let mut targets = ModelTargets::default();
    if let Some(manager) = plugin {
        let route = manager.inference_endpoint_for_model(model).await?.unwrap();
        assert_eq!(route.plugin_name, "metering-fixture");
        assert_eq!(route.address, format!("http://127.0.0.1:{port}/adapter/v1"));
        *provider.plugin_manager.lock().await = Some(manager);
    } else {
        targets
            .targets
            .insert(model.into(), vec![InferenceTarget::Local(port)]);
    }
    let serving = tokio::spawn(serve_pair(provider.clone(), targets));
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
    let body = serde_json::json!({
        "model": model, "messages": [{"role":"user", "content":"Say hello."}],
        "stream": true, "max_tokens": 8,
    });
    let raw = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{body}");
    let result = tokio::time::timeout(
        Duration::from_secs(30),
        exchanges(
            &payer,
            &provider,
            &buyer,
            &seller,
            &network,
            raw.as_bytes(),
            &price,
        ),
    )
    .await
    .unwrap_or_else(|error| {
        Err(anyhow::anyhow!(
            "exchange timeout: {error}; buyer={:?}; seller={:?}; buyer_policy={:?}; unpaid_finals={:?}",
            buyer
                .ledger
                .requests()
                .and_then(|rows| Ok(serde_json::to_string(&rows)?)),
            seller
                .ledger
                .receivables(None)
                .and_then(|rows| Ok(serde_json::to_string(&rows)?)),
            buyer.ledger.policy_status(mesh_llm_payments::now_ms()),
            seller.ledger.metered_unpaid_finals(&payer.id().to_string())
        ))
    });
    // Keep the accepted QUIC connection alive until the payer consumes Complete.
    let _connection = if result.is_err() {
        serving.abort();
        None
    } else {
        Some(serving.await??)
    };
    payer.endpoint.close().await;
    provider.endpoint.close().await;
    result
}

async fn serve_pair(node: Node, targets: ModelTargets) -> Result<iroh::endpoint::Connection> {
    let conn = node.endpoint.accept().await.unwrap().await?;
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
        super::super::metered::serve(node.clone(), conn.remote_id(), recv, send, targets.clone())
            .await?;
    }
    Ok(conn)
}

async fn exchanges(
    payer: &Node,
    provider: &Node,
    buyer: &PaymentService,
    seller: &PaymentService,
    network: &Network,
    raw: &[u8],
    price: &Pricing,
) -> Result<()> {
    let parsed = crate::network::payments::request::PaidRequest::parse(raw)?;
    for completed in 1..=2 {
        let mut body = parsed.body.clone();
        // Parsed backend bodies already opt in; construct fresh caller intent.
        body.as_object_mut().unwrap().remove("stream_options");
        let include_usage = completed == 2;
        if include_usage {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }
        let request =
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{body}");
        let (mut pipe, ready, _cancel) = crate::network::openai::test_metered_start(
            payer,
            provider.id(),
            request.as_bytes(),
            price.clone(),
            None,
        )
        .await?;
        ready.await?;
        let mut response = String::new();
        pipe.read_to_string(&mut response).await?;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("[DONE]"), "{response}");
        assert_eq!(response.contains("\"usage\""), include_usage, "{response}");
        wait_for_completion(buyer, seller, completed).await?;
        assert_eq!(
            network.payments.load(Ordering::SeqCst),
            if completed == 1 { 2 } else { 3 }
        );
        assert_eq!(
            seller
                .ledger
                .metered_unpaid_finals(&payer.id().to_string())?,
            0
        );
        assert_eq!(
            buyer.ledger.policy_status(mesh_llm_payments::now_ms())?["reserved_msat"],
            0
        );
        assert!(seller.ledger.receivables(None)?.iter().all(|row| row.paid));
        assert_settled_requests(buyer, seller, completed)?;
    }
    Ok(())
}

async fn wait_for_completion(
    buyer: &PaymentService,
    seller: &PaymentService,
    completed: usize,
) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if buyer
                .ledger
                .requests()?
                .iter()
                .filter(|r| r.state == "completed")
                .count()
                == completed
            {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .map_err(|error| {
        anyhow::anyhow!(
            "buyer completion {completed}: {error}; requests={:?}; receivables={:?}",
            buyer
                .ledger
                .requests()
                .and_then(|rows| Ok(serde_json::to_string(&rows)?)),
            seller
                .ledger
                .receivables(None)
                .and_then(|rows| Ok(serde_json::to_string(&rows)?))
        )
    })??;
    Ok(())
}

fn assert_settled_requests(
    buyer: &PaymentService,
    seller: &PaymentService,
    completed: usize,
) -> Result<()> {
    let rows = buyer.ledger.requests()?;
    assert_eq!(rows.len(), completed);
    for row in rows {
        let settlement = seller.ledger.metered_settlement(&row.terms.id)?;
        assert_eq!(row.state, "completed");
        assert_eq!(
            settlement.status,
            mesh_llm_payments::ledger::metered::MeteredStatus::Settled
        );
        let usage = settlement.usage.expect("authoritative backend usage");
        assert!(usage.prompt_tokens > 0);
        assert!(usage.completion_tokens > 0);
        assert_eq!(
            settlement.bill_msat,
            Some(usage.prompt_tokens + usage.completion_tokens)
        );
    }
    Ok(())
}
