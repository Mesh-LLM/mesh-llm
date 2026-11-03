//! Pending admission exercises a slow wallet, explicit retries and cancelled hook futures.
use super::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn event(peer: &str, observation: &str) -> Value {
    serde_json::json!({"endpoint":"chat_completions","phase":"backend_selected",
        "target_peer_id":peer,"model":"model","observation_id":observation,
        "exchange_id":observation,"body":{"max_tokens":4}})
}

async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn slow_admission_keeps_one_payment_and_requires_explicit_retry_before_inference() {
    let buyer_dir = tempfile::tempdir().unwrap();
    let seller_dir = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    chain.pending.store(true, Ordering::SeqCst);
    let buyer = engine(
        buyer_dir.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        chain.clone(),
    );
    let seller = engine(
        seller_dir.path(),
        &"bb".repeat(32),
        &"aa".repeat(32),
        true,
        chain.clone(),
    );
    let request = event(&seller.node, "first-observation");
    let hook = {
        let buyer = buyer.clone();
        tokio::spawn(async move {
            buyer
                .buyer_admission(&request, Duration::from_millis(20))
                .await
        })
    };
    until(|| !buyer.outbox.lock().unwrap().is_empty()).await;
    let processing = {
        let buyer = buyer.clone();
        let seller = seller.clone();
        tokio::spawn(async move { pump(&buyer, &seller).await })
    };
    until(|| !chain.attempts.lock().unwrap().is_empty()).await;
    let denied = hook.await.unwrap().unwrap();
    assert_eq!(denied.status, Some(402));
    let pending = buyer.ledger.active().unwrap().pop().unwrap();
    assert_eq!(pending.state, "negotiating");
    assert!(pending.exchange_id.is_none());
    assert_eq!(pending.quote.as_ref().unwrap().first_credit, 3);
    assert!(pending.quote.as_ref().unwrap().expires_at >= now() + 3500);
    assert!(!seller.ledger.earned(&buyer.node).unwrap());
    assert!(buyer.cancel(&seller.node, &pending.id).is_err());
    assert_eq!(buyer.ledger.active().unwrap()[0].id, pending.id);

    buyer
        .hook(serde_json::json!({"phase":"exchange_finished","observation_id":"first-observation"}))
        .await
        .unwrap();
    assert!(buyer.ledger.get(&pending.id).unwrap().usage.is_none());
    assert_eq!(
        buyer
            .buyer_admission(
                &event(&seller.node, "still-pending"),
                Duration::from_millis(1)
            )
            .await
            .unwrap()
            .status,
        Some(402)
    );
    assert_eq!(buyer.ledger.active().unwrap()[0].id, pending.id);
    assert_eq!(chain.payments.lock().unwrap().len(), 1);

    chain.pending.store(false, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(2), processing)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    pump(&buyer, &seller).await.unwrap();
    assert_eq!(buyer.ledger.get(&pending.id).unwrap().state, "ready");
    assert!(buyer.ledger.get(&pending.id).unwrap().exchange_id.is_none());
    assert_eq!(chain.payments.lock().unwrap().len(), 1);
    assert_eq!(
        buyer
            .buyer_admission(
                &event(&seller.node, "explicit-retry"),
                Duration::from_millis(1)
            )
            .await
            .unwrap()
            .decision,
        OpenAiAdmissionDecision::Allow
    );
    assert_eq!(
        buyer
            .ledger
            .get(&pending.id)
            .unwrap()
            .exchange_id
            .as_deref(),
        Some("explicit-retry")
    );
    assert_eq!(
        buyer
            .buyer_admission(
                &event(&seller.node, "another-inference"),
                Duration::from_millis(1)
            )
            .await
            .unwrap()
            .decision,
        OpenAiAdmissionDecision::Deny
    );

    assert_eq!(seller.hook(serde_json::json!({"endpoint":"chat_completions","phase":"request_received",
        "requested_by_node_id":buyer.node,"model":"model","observation_id":"seller-retry","body":{"max_tokens":4}})).await.unwrap().decision, OpenAiAdmissionDecision::Allow);
    seller
        .hook(
            serde_json::json!({"phase":"exchange_finished","observation_id":"seller-retry",
        "usage":{"prompt_tokens":1,"completion_tokens":4}}),
        )
        .await
        .unwrap();
    pump(&buyer, &seller).await.unwrap();
    pump(&buyer, &seller).await.unwrap();
    assert!(buyer.ledger.active().unwrap().is_empty());
    assert!(seller.ledger.active().unwrap().is_empty());
    let mut amounts = chain
        .attempts
        .lock()
        .unwrap()
        .values()
        .map(|a| a.request.obligation.amount)
        .collect::<Vec<_>>();
    amounts.sort_unstable();
    assert_eq!(amounts, vec![2, 3]);
}

#[tokio::test]
async fn concurrent_hooks_cannot_steal_an_admission_and_task_abort_detaches_it() {
    let directory = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let buyer = engine(
        directory.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        chain.clone(),
    );
    let peer = "bb".repeat(32);
    let request = event(&peer, "waiting");
    let hook = {
        let buyer = buyer.clone();
        let request = request.clone();
        tokio::spawn(async move {
            buyer
                .buyer_admission(&request, Duration::from_secs(10))
                .await
        })
    };
    until(|| !buyer.ledger.active().unwrap().is_empty()).await;
    let id = buyer.ledger.active().unwrap()[0].id.clone();
    for observation in ["waiting", "parallel"] {
        assert_eq!(
            buyer
                .buyer_admission(&event(&peer, observation), Duration::from_millis(1))
                .await
                .unwrap()
                .decision,
            OpenAiAdmissionDecision::Deny
        );
    }
    let mut another_model = event(&peer, "another-model");
    another_model["model"] = serde_json::json!("different-model");
    assert_eq!(
        buyer
            .buyer_admission(&another_model, Duration::from_millis(1))
            .await
            .unwrap()
            .decision,
        OpenAiAdmissionDecision::Deny
    );
    assert_eq!(
        buyer.ledger.get(&id).unwrap().exchange_id.as_deref(),
        Some("waiting")
    );
    hook.abort();
    assert!(hook.await.unwrap_err().is_cancelled());
    assert!(buyer.ledger.get(&id).unwrap().exchange_id.is_none());
    assert_eq!(
        buyer
            .buyer_admission(&event(&peer, "retry"), Duration::from_millis(1))
            .await
            .unwrap()
            .status,
        Some(402)
    );
    assert_eq!(buyer.ledger.active().unwrap()[0].id, id);
    assert!(chain.payments.lock().unwrap().is_empty());
    let ids = buyer
        .outbox
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(_, message)| match message {
            Message::Hello { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![id.clone(), id]);
}

#[tokio::test]
async fn expired_pending_quote_is_never_replaced_with_a_new_charge() {
    let buyer_dir = tempfile::tempdir().unwrap();
    let seller_dir = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let buyer = engine(
        buyer_dir.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        chain.clone(),
    );
    let seller = engine(
        seller_dir.path(),
        &"bb".repeat(32),
        &"aa".repeat(32),
        true,
        chain.clone(),
    );
    buyer
        .buyer_admission(&event(&seller.node, "pending"), Duration::from_millis(1))
        .await
        .unwrap();
    pump(&buyer, &seller).await.unwrap();
    pump(&buyer, &seller).await.unwrap();
    let id = buyer.ledger.active().unwrap()[0].id.clone();
    buyer
        .ledger
        .update(&id, |exchange| {
            exchange.quote.as_mut().unwrap().expires_at = now() - 1;
            Ok(())
        })
        .unwrap();
    let rejected = buyer
        .buyer_admission(
            &event(&seller.node, "expired-retry"),
            Duration::from_millis(1),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status, Some(402));
    assert!(rejected.reason.unwrap().contains("expired"));
    assert_eq!(buyer.ledger.active().unwrap()[0].id, id);
    assert_eq!(chain.payments.lock().unwrap().len(), 1);
    assert!(buyer.outbox.lock().unwrap().is_empty());
}

fn payable(buyer: &Engine, amount: u64) -> PaymentRequest {
    payable_for_peer(buyer, amount, &"bb".repeat(32))
}

fn payable_for_peer(buyer: &Engine, amount: u64, peer: &str) -> PaymentRequest {
    let id = uuid::Uuid::new_v4().to_string();
    buyer
        .ledger
        .insert(&Exchange {
            id: id.clone(),
            peer: peer.into(),
            model: "model".into(),
            payer: buyer.wallet.info.address.clone(),
            seller: false,
            exchange_id: Some("payment".into()),
            state: "running".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        })
        .unwrap();
    PaymentRequest::new(Obligation {
        version: 1,
        id: format!("{id}/usage"),
        purpose: Purpose::Usage,
        buyer_peer: buyer.node.clone(),
        seller_peer: peer.into(),
        request_id: id,
        model: "model".into(),
        offer_id: "offer".into(),
        asset: buyer.wallet.info.asset.clone(),
        payer: buyer.wallet.info.address.clone(),
        recipient: format!("0x{}", "22".repeat(20)),
        amount,
        expires_at: now() + 300,
    })
    .unwrap()
}

#[test]
fn token_budget_adds_fees_only_when_the_declared_fee_asset_matches() {
    for separate in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut buyer = engine(
            directory.path(),
            &"aa".repeat(32),
            &"bb".repeat(32),
            false,
            Arc::new(Chain::default()),
        );
        let engine = Arc::get_mut(&mut buyer).unwrap();
        engine.config.buying.max_fee = 7;
        engine.config.buying.daily_budget = if separate { 10 } else { 17 };
        if separate {
            engine.wallet.info.fee_asset = Some(Asset {
                token: "native:fee-token".into(),
                ..engine.wallet.info.asset.clone()
            });
        }
        let payment = payable(engine, 10);
        let authorized = engine
            .authorize_payment(&payment.obligation.request_id, &payment)
            .unwrap();
        assert_eq!(authorized.max_total, if separate { 10 } else { 17 });
        assert_eq!(authorized.max_fee, 7);
        assert!(
            engine
                .ledger
                .reserve_budget("over-budget", 1, engine.config.buying.daily_budget)
                .is_err()
        );
        let repeated = engine
            .authorize_payment(&payment.obligation.request_id, &payment)
            .unwrap();
        assert_eq!(repeated.max_total, authorized.max_total);
    }
}

#[test]
fn separate_fee_asset_accepts_zero_demo_cap_without_combining_units() {
    let directory = tempfile::tempdir().unwrap();
    let mut buyer = engine(
        directory.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        Arc::new(Chain::default()),
    );
    let engine = Arc::get_mut(&mut buyer).unwrap();
    engine.wallet.info.fee_asset = Some(Asset {
        token: "native:fee-token".into(),
        ..engine.wallet.info.asset.clone()
    });
    engine.config.buying.max_fee = 0;
    let payment = payable(engine, 10);
    let authorized = engine
        .authorize_payment(&payment.obligation.request_id, &payment)
        .unwrap();
    assert_eq!(authorized.max_total, 10);
    assert_eq!(authorized.max_fee, 0);
}

#[tokio::test]
async fn background_reconciliation_does_not_wait_for_finality_or_skip_other_peers() {
    let directory = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    chain.pending.store(true, Ordering::SeqCst);
    let mut buyer = engine(
        directory.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        chain.clone(),
    );
    Arc::get_mut(&mut buyer)
        .unwrap()
        .config
        .buying
        .peers
        .push("cc".repeat(32));
    let mut ids = Vec::new();
    for (index, peer) in ["bb".repeat(32), "cc".repeat(32)].into_iter().enumerate() {
        let payment = payable_for_peer(&buyer, 10, &peer);
        let id = payment.obligation.request_id.clone();
        buyer
            .ledger
            .update(&id, |exchange| {
                exchange.peer = peer.clone();
                exchange.state = "billed".into();
                exchange.usage_payment = Some(payment.clone());
                Ok(())
            })
            .unwrap();
        buyer.authorize_payment(&id, &payment).unwrap();
        let finalized = index == 1;
        chain.attempts.lock().unwrap().insert(
            payment.obligation.id.clone(),
            PaymentAttempt {
                payment_id: payment.obligation.id.clone(),
                request: payment.clone(),
                state: if finalized {
                    PaymentState::Finalized
                } else {
                    PaymentState::Submitted
                },
                tx_hashes: vec![Chain::receipt(&payment).tx_hash],
                receipt: finalized.then(|| Chain::receipt(&payment)),
                error: None,
            },
        );
        ids.push((id, payment.obligation.id));
    }
    tokio::time::timeout(Duration::from_millis(100), buyer.retry())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(chain.lookups.load(Ordering::SeqCst), 2);
    assert!(buyer.ledger.get(&ids[0].0).unwrap().usage_receipt.is_none());
    assert_eq!(buyer.ledger.get(&ids[1].0).unwrap().state, "paid");
    assert_eq!(
        buyer.outbox.lock().unwrap().pop().unwrap().0,
        "cc".repeat(32)
    );
    assert!(chain.payments.lock().unwrap().is_empty());

    // A terminal error from the first peer is reported only after the second
    // peer's existing proof has also been retransmitted.
    chain
        .attempts
        .lock()
        .unwrap()
        .get_mut(&ids[0].1)
        .unwrap()
        .state = PaymentState::Held;
    assert!(buyer.retry().await.is_err());
    assert_eq!(
        buyer.outbox.lock().unwrap().pop().unwrap().0,
        "cc".repeat(32)
    );
    assert!(chain.payments.lock().unwrap().is_empty());
}

#[tokio::test]
async fn only_explicit_resubmit_recovers_a_journaled_prepare_before_broadcast_gap() {
    let buyer_dir = tempfile::tempdir().unwrap();
    let seller_dir = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let buyer = engine(
        buyer_dir.path(),
        &"aa".repeat(32),
        &"bb".repeat(32),
        false,
        chain.clone(),
    );
    let seller = engine(
        seller_dir.path(),
        &"bb".repeat(32),
        &"aa".repeat(32),
        true,
        chain.clone(),
    );
    let id = uuid::Uuid::new_v4().to_string();
    seller
        .message(
            &buyer.node,
            Message::Hello {
                id: id.clone(),
                model: "model".into(),
                payer: buyer.wallet.info.address.clone(),
            },
        )
        .await
        .unwrap();
    let quote = seller.ledger.get(&id).unwrap().quote.unwrap();
    let payment = quote.admission.as_ref().unwrap().clone();
    buyer
        .ledger
        .insert(&Exchange {
            id: id.clone(),
            peer: seller.node.clone(),
            model: "model".into(),
            payer: buyer.wallet.info.address.clone(),
            seller: false,
            exchange_id: None,
            state: "negotiating".into(),
            quote: Some(quote),
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        })
        .unwrap();
    let authorized = buyer.authorize_payment(&id, &payment).unwrap();
    // A persisted signed attempt exists, but the process died before broadcast.
    chain.attempts.lock().unwrap().insert(
        payment.obligation.id.clone(),
        PaymentAttempt {
            payment_id: payment.obligation.id.clone(),
            request: payment.clone(),
            state: PaymentState::Prepared,
            tx_hashes: vec![Chain::receipt(&payment).tx_hash],
            receipt: None,
            error: None,
        },
    );
    tokio::time::timeout(Duration::from_millis(100), buyer.retry())
        .await
        .unwrap()
        .unwrap();
    assert!(chain.lookups.load(Ordering::SeqCst) > 0);
    assert!(chain.payments.lock().unwrap().is_empty());
    assert_eq!(
        chain
            .attempts
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .state,
        PaymentState::Prepared
    );

    buyer.resubmit(&id).await.unwrap();
    {
        let calls = chain.payments.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].payment_id, authorized.payment_id);
        assert_eq!(calls[0].request, authorized.request);
        assert_eq!(calls[0].max_total, authorized.max_total);
        assert_eq!(calls[0].max_fee, authorized.max_fee);
    }
    assert_eq!(chain.attempts.lock().unwrap().len(), 1);
    pump(&buyer, &seller).await.unwrap();
    assert_eq!(buyer.ledger.get(&id).unwrap().state, "ready");
    buyer.resubmit(&id).await.unwrap();
    assert_eq!(chain.payments.lock().unwrap().len(), 1);
}
