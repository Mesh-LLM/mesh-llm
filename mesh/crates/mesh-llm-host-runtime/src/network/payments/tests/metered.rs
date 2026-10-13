//! The real buyer and seller over authenticated QUIC against a streamed HTTP
//! backend: deposit on the first request, trusted admission on the second,
//! settlement from the backend's reported usage, usage hidden from callers
//! that did not ask for it.
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn first_request_deposits_then_trusted_buyer_streams_without_one() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), streamed_pair(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payer_and_provider_each_observe_the_lifecycle() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), streamed_pair(true)).await?
}

async fn streamed_pair(record_lifecycle: bool) -> Result<()> {
    let network = Arc::new(Network::default());
    network
        .record_lifecycle
        .store(record_lifecycle, Ordering::SeqCst);
    let pair = Pair::new(network.clone()).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let backend = tokio::spawn(stream_backend(listener, network.clone(), 2));
    let serving = pair.serve(local_targets(port), 2);
    pair.connect().await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"stream\":true,\"max_tokens\":8}";
    // The caller omits include_usage first, then opts in. Usage stays hidden
    // unless requested, while settlement still bills backend-reported usage.
    let opted_in = String::from_utf8(raw.to_vec())?.replace(
        "\"stream\":true",
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}",
    );
    for completed in 1..=2 {
        let request: &[u8] = if completed == 1 {
            raw
        } else {
            opted_in.as_bytes()
        };
        let response = pair.request(request, Some("host-evidence-id")).await?;
        assert!(response.contains("hello"), "{response}");
        assert!(response.contains("data: [DONE]"), "{response}");
        assert_eq!(response.contains("\"usage\""), completed == 2, "{response}");
        pair.wait_completed(completed).await?;
        // First request: deposit + final. Second: trusted, final only.
        assert_eq!(
            network.payments.load(Ordering::SeqCst),
            if completed == 1 { 2 } else { 3 }
        );
        assert_eq!(
            pair.seller
                .ledger
                .metered_unpaid_finals(&pair.buyer_node.id().to_string())?,
            0
        );
    }
    let requests = pair.buyer.ledger.requests()?;
    assert!(requests.iter().all(|r| r.state == "completed"));
    // 40 prompt + 3 completion tokens at 1 msat each, plus 10 msat fee per
    // payment: two payments for the deposited request, one for the trusted.
    let spent: Vec<u64> = requests.iter().map(|r| r.spent_msat).collect();
    assert_eq!(spent.iter().sum::<u64>(), 43 * 2 + 30);
    assert_eq!(
        requests[0].terms.exchange_id.as_deref(),
        Some("host-evidence-id")
    );
    backend.await??;
    let _connection = serving.await??;
    if record_lifecycle {
        assert_lifecycle(&network).await;
    }
    pair.close().await;
    Ok(())
}

async fn stream_backend(
    listener: tokio::net::TcpListener,
    network: Arc<Network>,
    count: usize,
) -> Result<()> {
    for _ in 0..count {
        let (mut stream, _) = listener.accept().await?;
        let body = read_backend_request(&mut stream).await?;
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["max_tokens"], 8);
        assert!(
            network.payments.load(Ordering::SeqCst) > 0,
            "dispatch before deposit"
        );
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"hello\"}]}\n\n").await?;
        tokio::task::yield_now().await;
        stream.write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n").await?;
        stream.shutdown().await?;
    }
    Ok(())
}

/// Both sides publish `payment.lifecycle.v1` for the deposited request:
/// the buyer joins on the host's exchange id, the seller on its own fresh
/// id that also names its `openai.exchange.v1` events.
async fn assert_lifecycle(network: &Network) {
    let payer = wait_for_events(&network.payer_lifecycle, "final_accounted", 2).await;
    let provider = wait_for_events(&network.provider_lifecycle, "final_accounted", 2).await;
    let first = |events: &[serde_json::Value]| -> serde_json::Value {
        events
            .iter()
            .find(|e| e["phase"] == "input_settlement_observed" && e["segment"] == 0)
            .cloned()
            .expect("deposit settlement observed")
    };
    let payer_deposit = first(&payer);
    let provider_deposit = first(&provider);
    assert_eq!(payer_deposit["exchange_id"], "host-evidence-id");
    assert_eq!(payer_deposit["role"], "payer");
    assert_eq!(provider_deposit["role"], "provider");
    assert_ne!(provider_deposit["exchange_id"], "host-evidence-id");
    assert_eq!(
        payer_deposit["payment_hash"],
        provider_deposit["payment_hash"]
    );
    assert_eq!(
        payer
            .iter()
            .filter(|e| e["phase"] == "terms_accepted")
            .count(),
        2
    );
    assert!(
        provider
            .iter()
            .any(|e| e["phase"] == "delivered" && e["tokens"] == 3)
    );
}

async fn wait_for_events(
    events: &Arc<Mutex<Vec<serde_json::Value>>>,
    phase: &str,
    count: usize,
) -> Vec<serde_json::Value> {
    for _ in 0..1000 {
        let snapshot = events.lock().unwrap().clone();
        if snapshot.iter().filter(|e| e["phase"] == phase).count() >= count {
            return snapshot;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("missing {count} x {phase}: {:#?}", events.lock().unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backend_without_usage_abandons_the_request_without_a_bill() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), abandoned()).await?
}

async fn abandoned() -> Result<()> {
    let network = Arc::new(Network::default());
    let pair = Pair::new(network.clone()).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let backend = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await?;
        read_backend_request(&mut stream).await?;
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            .await?;
        stream.shutdown().await?;
        Ok::<_, anyhow::Error>(())
    });
    let serving = pair.serve(local_targets(port), 1);
    pair.connect().await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}";
    let response = pair.request(raw, None).await?;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    pair.wait_completed(1).await?;
    // Only the deposit moved: no final invoice, no debt, no strike.
    assert_eq!(network.payments.load(Ordering::SeqCst), 1);
    let requests = pair.buyer.ledger.requests()?;
    assert_eq!(requests[0].state, "completed");
    assert_eq!(
        pair.seller
            .ledger
            .metered_unpaid_finals(&pair.buyer_node.id().to_string())?,
        0
    );
    backend.await??;
    let _connection = serving.await??;
    pair.close().await;
    Ok(())
}
