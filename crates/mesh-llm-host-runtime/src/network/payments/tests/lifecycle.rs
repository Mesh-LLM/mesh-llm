use super::*;

use serde_json::Value;

// Both sides of one paid exchange observe it: the payer through its ingress
// task, the provider at the seller-op call sites in the serving path. They
// share no exchange ID, so they join on the invoices' payment hashes. On each
// side, the lifecycle events carry the same exchange ID as that side's
// `openai.exchange.v1` events.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payer_and_provider_each_observe_one_paid_exchange() -> Result<()> {
    let network = Arc::new(Network::default());
    network.record_lifecycle.store(true, Ordering::SeqCst);
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange_on(network.clone(), false, false, Some(8), 8),
    )
    .await??;
    let provider = settled_events(&network.provider_lifecycle, 6).await?;
    let payer = settled_events(&network.payer_lifecycle, 6).await?;

    // Every phase exactly once per side, in serving order where the serving
    // path orders them (input settlement is recorded by its own task).
    assert_eq!(
        sorted_phases(&provider),
        sorted(&[
            "delivered",
            "input_invoice_issued",
            "input_settlement_observed",
            "output_invoice_issued",
            "output_settlement_observed",
            "terms_accepted",
        ])
    );
    assert_eq!(
        sorted_phases(&payer),
        sorted(&[
            "final_accounted",
            "input_invoice_issued",
            "input_settlement_observed",
            "output_invoice_issued",
            "output_settlement_observed",
            "terms_accepted",
        ])
    );
    assert_before(&provider, "terms_accepted", "input_invoice_issued");
    assert_before(&provider, "input_invoice_issued", "delivered");
    assert_before(&provider, "delivered", "output_invoice_issued");
    assert_before(
        &provider,
        "output_invoice_issued",
        "output_settlement_observed",
    );

    for event in &provider {
        assert_eq!(event["role"], "provider");
    }
    for event in &payer {
        assert_eq!(event["role"], "payer");
        assert_eq!(event["exchange_id"], "host-evidence-id");
    }
    let provider_exchange = provider[0]["exchange_id"].as_str().unwrap();
    assert!(
        provider
            .iter()
            .all(|e| e["exchange_id"] == provider_exchange)
    );
    assert_ne!(provider_exchange, "host-evidence-id");
    // The provider's two channels join exactly: its effective and terminal
    // exchange events carry the lifecycle events' exchange ID.
    let exchange_ids = network.provider_exchange_ids.lock().unwrap().clone();
    assert_eq!(
        exchange_ids.len(),
        2,
        "effective + terminal: {exchange_ids:?}"
    );
    assert!(exchange_ids.iter().all(|id| id == provider_exchange));

    // The join: each invoice's payment hash, and the amount on it, is the
    // same as each side saw it, and each side's settlement names that hash.
    for (invoice, settlement) in [
        ("input_invoice_issued", "input_settlement_observed"),
        ("output_invoice_issued", "output_settlement_observed"),
    ] {
        let hash = &phase(&provider, invoice)["payment_hash"];
        assert!(hash.is_string());
        assert_eq!(&phase(&payer, invoice)["payment_hash"], hash);
        assert_eq!(&phase(&provider, settlement)["payment_hash"], hash);
        assert_eq!(&phase(&payer, settlement)["payment_hash"], hash);
        assert_eq!(
            phase(&provider, invoice)["amount_msat"],
            phase(&payer, invoice)["amount_msat"]
        );
        assert_eq!(phase(&provider, invoice)["source"], "provider_asserted");
        assert_eq!(phase(&provider, settlement)["source"], "wallet_reported");
        assert_eq!(phase(&provider, settlement)["settlement"], "terminal");
    }

    // Both sides accepted the same terms: the provider priced them and the
    // payer approved them, so the cap on each acceptance is identical.
    let accepted = phase(&provider, "terms_accepted");
    assert_eq!(accepted["source"], "provider_asserted");
    assert_eq!(
        accepted["amount_msat"],
        phase(&payer, "terms_accepted")["amount_msat"]
    );

    // The delivered watermark is the provider's own count, written once at close.
    let delivered = phase(&provider, "delivered");
    assert_eq!(delivered["tokens"], 3);
    assert_eq!(delivered["source"], "provider_asserted");

    // Nothing private leaves either side.
    for event in provider.iter().chain(&payer) {
        let json = event.to_string();
        for secret in ["preimage", "bolt11", "test output", "Hi"] {
            assert!(!json.contains(secret), "{secret} in {json}");
        }
    }
    Ok(())
}

/// The events a recorder holds once `count` have arrived and delivery has
/// gone quiet; publication is asynchronous and best-effort.
async fn settled_events(events: &Arc<Mutex<Vec<Value>>>, count: usize) -> Result<Vec<Value>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while events.lock().unwrap().len() < count {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "only {:?}",
            events.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    Ok(events.lock().unwrap().clone())
}

fn phase<'a>(events: &'a [Value], name: &str) -> &'a Value {
    events
        .iter()
        .find(|event| event["phase"] == name)
        .unwrap_or_else(|| panic!("no {name} in {events:?}"))
}

fn assert_before(events: &[Value], first: &str, second: &str) {
    let at = |name: &str| events.iter().position(|event| event["phase"] == name);
    assert!(
        at(first) < at(second),
        "{first} not before {second}: {events:?}"
    );
}

fn sorted_phases(events: &[Value]) -> Vec<String> {
    let mut phases: Vec<String> = events
        .iter()
        .map(|event| event["phase"].as_str().unwrap_or_default().to_owned())
        .collect();
    phases.sort();
    phases
}

fn sorted(phases: &[&str]) -> Vec<String> {
    let mut phases: Vec<String> = phases.iter().map(|phase| (*phase).to_owned()).collect();
    phases.sort();
    phases
}
