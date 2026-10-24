//! Explicit refusals are observable before any payment authority exists.
use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_refusal_is_observed_without_payment() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), async {
        let network = Arc::new(Network::default());
        network.record_lifecycle.store(true, Ordering::SeqCst);
        let pair = Pair::new(network.clone()).await?;
        super::super::lifecycle::refused(&pair.buyer_node, None, super::super::refusal::Refusal::DebtBlocked).await;
        assert!(network.payer_lifecycle.lock().unwrap().is_empty());
        let node = pair.seller_node.clone();
        let seller = tokio::spawn(async move {
            let connection = node.endpoint.accept().await.unwrap().await?;
            for message in [
                "payment_refused:protocol_version",
                "this seller requires token-metered payment protocol v3; upgrade mesh-llm",
                "payment_refused:debt_blocked",
                "payment_refused:unknown; private-provider-error",
            ] {
                let (mut send, mut recv) = accept_tunnel(&connection).await?;
                let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
                recv.read_exact(&mut prefix).await?;
                assert!(matches!(wire::read(&mut recv).await?, Frame::Request { .. }));
                wire::write(&mut send, &Frame::Error { message: message.into() }).await?;
                send.finish()?;
            }
            drop(node);
            Ok::<_, anyhow::Error>(connection)
        });
        pair.connect().await?;
        for (index, reason) in ["protocol_version", "protocol_version", "debt_blocked", "admission_refused"].into_iter().enumerate() {
            assert!(pair.request(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"private-prompt\",\"max_tokens\":8}", Some("exchange-refused")).await.is_err());
            loop {
                if network.payer_lifecycle.lock().unwrap().get(index).is_some_and(|event| event["reason"] == reason) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        let _connection = seller.await??;
        let events = network.payer_lifecycle.lock().unwrap().clone();
        assert_eq!(events.len(), 4);
        for event in &events {
            assert_eq!(event["exchange_id"], "exchange-refused");
            assert_eq!(event["phase"], "refused");
            assert_eq!(event["role"], "payer");
            assert_eq!(event["source"], "payer_asserted");
            assert_eq!(event.as_object().unwrap().len(), 6);
            let mut blank = event.clone();
            blank["event_ref"] = "".into();
            assert_eq!(event["event_ref"], crate::plugin::openai_exchange::request_body_digest(&blank, None).unwrap());
        }
        assert_eq!(events[0]["event_ref"], events[1]["event_ref"]);
        assert_ne!(events[0]["event_ref"], events[2]["event_ref"]);
        assert!(pair.buyer.ledger.requests()?.is_empty());
        assert_eq!(network.payments.load(Ordering::SeqCst), 0);
        assert!(network.entries.lock().unwrap().is_empty());
        pair.close().await;
        Ok(())
    }).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpaid_final_limit_reports_debt_without_dispatch() -> Result<()> {
    use mesh_llm_payments::ledger::metered::{MeteredOutcome, MeteredRequest, STRIKE_LIMIT};
    use mesh_llm_payments::ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind};
    use mesh_llm_payments_types::metering::MeteredUsage;

    tokio::time::timeout(Duration::from_secs(20), async {
        let network = Arc::new(Network::default());
        network.record_lifecycle.store(true, Ordering::SeqCst);
        let pair = Pair::new(network.clone()).await?;
        let peer = pair.buyer_node.id().to_string();
        let wallet = TestWallet { owner: 1, network: network.clone() };
        for n in 0..STRIKE_LIMIT {
            let id = format!("private-debt-{n}");
            pair.seller.ledger.metered_admit(&MeteredRequest {
                id: &id, peer: &peer, pricing: &price(), require_deposit: false,
            })?;
            pair.seller.ledger.metered_finalize(&id, MeteredOutcome::Usage(MeteredUsage {
                prompt_tokens: 50, completion_tokens: 50, cached_tokens: 0,
            }))?;
            pair.seller.ledger.record_metered_invoice(&MeteredInvoice {
                request_id: id, peer: peer.clone(), kind: MeteredInvoiceKind::Final,
                invoice: wallet.create_invoice(Some(10_000), 3600).await?, paid: false,
            })?;
        }
        let invoices = network.entries.lock().unwrap().len();
        let backend = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        let serving = pair.serve(local_targets(backend.local_addr()?.port()), 1);
        pair.connect().await?;
        assert!(pair.request(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}", Some("exchange-debt")).await.is_err());
        let _connection = serving.await??;
        loop {
            if !network.payer_lifecycle.lock().unwrap().is_empty() { break; }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let events = network.payer_lifecycle.lock().unwrap().clone();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["reason"], "debt_blocked");
        assert_eq!(events[0]["exchange_id"], "exchange-debt");
        assert_eq!(events[0].as_object().unwrap().len(), 6);
        assert_eq!(network.entries.lock().unwrap().len(), invoices);
        assert_eq!(network.payments.load(Ordering::SeqCst), 0);
        assert!(pair.buyer.ledger.requests()?.is_empty());
        assert!(tokio::time::timeout(Duration::from_millis(50), backend.accept()).await.is_err());
        pair.close().await;
        Ok(())
    }).await?
}
