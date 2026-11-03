use super::*;
use crate::{
    binding::WalletTransport,
    config::{Buying, Price},
};
use std::collections::BTreeMap;
use std::sync::Mutex;
use wallet_contract::{Asset, VerifyResult, WalletInfo};

#[derive(Default)]
struct Chain {
    attempts: Mutex<BTreeMap<String, PaymentAttempt>>,
    payments: Mutex<Vec<PayRequest>>,
    lookups: std::sync::atomic::AtomicUsize,
    pending: std::sync::atomic::AtomicBool,
    lose_reply: std::sync::atomic::AtomicBool,
    reject_before_journal: std::sync::atomic::AtomicBool,
}
impl Chain {
    fn receipt(request: &PaymentRequest) -> Receipt {
        Receipt {
            tx_hash: format!("0x{}", request.memo.trim_start_matches("0x")),
            block_hash: "0xfixture".into(),
            block_number: 10,
            transfer_log_index: 0,
            payer: request.obligation.payer.clone(),
            recipient: request.obligation.recipient.clone(),
            amount: request.obligation.amount,
            memo: request.memo.clone(),
            finalized: true,
        }
    }
}
impl WalletTransport for Chain {
    fn execute<'a>(
        &'a self,
        operation: &'a str,
        input: Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match operation {
                ops::PAY => {
                    let pay: PayRequest = serde_json::from_value(input)?;
                    pay.request.validate()?;
                    self.payments.lock().unwrap().push(pay.clone());
                    if self
                        .reject_before_journal
                        .swap(false, std::sync::atomic::Ordering::SeqCst)
                    {
                        bail!("fixture rejected payment before signing");
                    }
                    let mut attempts = self.attempts.lock().unwrap();
                    if let Some(old) = attempts.get_mut(&pay.payment_id) {
                        ensure!(old.request == pay.request, "intent changed");
                        if old.state == PaymentState::Prepared
                            && !self.pending.load(std::sync::atomic::Ordering::SeqCst)
                        {
                            old.state = PaymentState::Finalized;
                            old.receipt = Some(Self::receipt(&old.request));
                        }
                        return Ok(serde_json::to_value(old)?);
                    }
                    let receipt = Self::receipt(&pay.request);
                    let pending = self.pending.load(std::sync::atomic::Ordering::SeqCst);
                    let attempt = PaymentAttempt {
                        payment_id: pay.payment_id.clone(),
                        request: pay.request,
                        state: if pending {
                            PaymentState::Submitted
                        } else {
                            PaymentState::Finalized
                        },
                        tx_hashes: vec![receipt.tx_hash.clone()],
                        receipt: if pending { None } else { Some(receipt) },
                        error: None,
                    };
                    attempts.insert(pay.payment_id, attempt.clone());
                    if self
                        .lose_reply
                        .swap(false, std::sync::atomic::Ordering::SeqCst)
                    {
                        bail!("fixture lost wallet reply after durable submission");
                    }
                    Ok(serde_json::to_value(attempt)?)
                }
                ops::VERIFY => {
                    let verify: VerifyRequest = serde_json::from_value(input)?;
                    let attempts = self.attempts.lock().unwrap();
                    let attempt = attempts
                        .get(&verify.request.obligation.id)
                        .context("payment is absent")?;
                    ensure!(
                        attempt.request == verify.request
                            && attempt.tx_hashes.contains(&verify.tx_hash),
                        "wrong delivery evidence"
                    );
                    Ok(serde_json::to_value(VerifyResult {
                        state: attempt.state.clone(),
                        receipt: attempt.receipt.clone(),
                        reason: None,
                    })?)
                }
                ops::LOOKUP => {
                    self.lookups
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let lookup: LookupRequest = serde_json::from_value(input)?;
                    let mut attempts = self.attempts.lock().unwrap();
                    let attempt = attempts
                        .get_mut(&lookup.payment_id)
                        .context("unknown intent")?;
                    if attempt.state == PaymentState::Submitted
                        && !self.pending.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        attempt.state = PaymentState::Finalized;
                        attempt.receipt = Some(Self::receipt(&attempt.request));
                    }
                    Ok(serde_json::to_value(attempt)?)
                }
                _ => bail!("unexpected fixture wallet operation"),
            }
        })
    }
}
fn engine(
    temp: &std::path::Path,
    node: &str,
    peer: &str,
    seller: bool,
    chain: Arc<Chain>,
) -> Arc<Engine> {
    let info = WalletInfo {
        wallet_id: node.into(),
        address: format!("0x{}", (if seller { "22" } else { "11" }).repeat(20)),
        asset: Asset::default(),
        method: wallet_contract::METHOD.into(),
        can_send: true,
        fee_asset: None,
    };
    let price = Price {
        input_per_million: 1_000_000,
        output_per_million: 1_000_000,
    };
    let config = Config {
        data_dir: temp.into(),
        wallet_binding: "wallet".into(),
        prices: if seller {
            BTreeMap::from([("model".into(), price.clone())])
        } else {
            Default::default()
        },
        admission_fee: 3,
        buying: Buying {
            enabled: !seller,
            peers: vec![peer.into()],
            maximum_price: price,
            max_payment: 100,
            max_fee: 0,
            daily_budget: 1000,
        },
    };
    let ledger = Ledger::open(temp).unwrap();
    ledger.pin_node(node).unwrap();
    ledger.pin(&info).unwrap();
    ledger.detach_buyer_admissions().unwrap();
    Arc::new(Engine {
        config,
        ledger,
        wallet: Wallet::fixture(info, chain),
        node: node.into(),
        context: None,
        changed: Default::default(),
        peers: Default::default(),
        ingress: Default::default(),
        outbox: Default::default(),
    })
}
async fn pump(buyer: &Engine, seller: &Engine) -> Result<()> {
    let messages: Vec<_> = buyer.outbox.lock().unwrap().drain(..).collect();
    for (peer, message) in messages {
        ensure!(peer == seller.node, "wrong seller");
        seller.message(&buyer.node, message).await?;
    }
    let messages: Vec<_> = seller.outbox.lock().unwrap().drain(..).collect();
    for (peer, message) in messages {
        ensure!(peer == buyer.node, "wrong buyer");
        buyer.message(&seller.node, message).await?;
    }
    Ok(())
}
async fn infer(buyer: Arc<Engine>, seller: Arc<Engine>, counter: u64) -> Result<()> {
    let buyer_event = serde_json::json!({"endpoint":"chat_completions","phase":"backend_selected","target_peer_id":seller.node,"model":"model","exchange_id":format!("buyer-{counter}"),"body":{"max_tokens":4}});
    let buy = buyer.clone();
    let task = tokio::spawn(async move { buy.hook(buyer_event).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !task.is_finished() {
            pump(&buyer, &seller).await?;
            tokio::task::yield_now().await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    ensure!(
        task.await??.decision == OpenAiAdmissionDecision::Allow,
        "buyer was denied"
    );
    let serving = serde_json::json!({"endpoint":"chat_completions","phase":"request_received","requested_by_node_id":buyer.node,"model":"model","exchange_id":format!("seller-{counter}"),"body":{"max_tokens":4}});
    ensure!(
        seller.hook(serving).await?.decision == OpenAiAdmissionDecision::Allow,
        "seller was denied"
    );
    seller.hook(serde_json::json!({"phase":"exchange_finished","exchange_id":format!("seller-{counter}"),"usage":{"prompt_tokens":1,"completion_tokens":4}})).await?;
    pump(&buyer, &seller).await?;
    pump(&buyer, &seller).await?;
    Ok(())
}
#[tokio::test]
async fn authenticated_admission_decode_and_exact_postpaid_roundtrip() {
    let buyer_dir = tempfile::tempdir().unwrap();
    let seller_dir = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let buyer_node = "aa".repeat(32);
    let seller_node = "bb".repeat(32);
    let buyer = engine(
        buyer_dir.path(),
        &buyer_node,
        &seller_node,
        false,
        chain.clone(),
    );
    let seller = engine(
        seller_dir.path(),
        &seller_node,
        &buyer_node,
        true,
        chain.clone(),
    );
    infer(buyer.clone(), seller.clone(), 1).await.unwrap();
    assert!(buyer.ledger.active().unwrap().is_empty());
    assert!(seller.ledger.active().unwrap().is_empty());
    assert!(seller.ledger.earned(&buyer_node).unwrap());
    infer(buyer.clone(), seller.clone(), 2).await.unwrap();
    let amounts: Vec<_> = chain
        .attempts
        .lock()
        .unwrap()
        .values()
        .map(|x| x.request.obligation.amount)
        .collect();
    assert_eq!(amounts.len(), 3);
    assert_eq!(amounts.iter().sum::<u64>(), 10);
    assert!(amounts.contains(&3) && amounts.contains(&2) && amounts.contains(&5));
    // Delayed admission messages and duplicate bills never reopen settled work.
    let settled = buyer
        .ledger
        .get(
            &chain
                .attempts
                .lock()
                .unwrap()
                .values()
                .next()
                .unwrap()
                .request
                .obligation
                .request_id,
        )
        .unwrap();
    buyer
        .message(
            &seller.node,
            Message::Ready {
                id: settled.id.clone(),
            },
        )
        .await
        .unwrap();
    assert_eq!(buyer.ledger.get(&settled.id).unwrap().state, "settled");
    let usage = settled.usage.unwrap();
    buyer
        .message(
            &seller.node,
            Message::Bill {
                id: settled.id.clone(),
                input: usage.0,
                output: usage.1,
                payment: settled.usage_payment.map(Box::new),
            },
        )
        .await
        .unwrap();
    pump(&buyer, &seller).await.unwrap();
    assert_eq!(buyer.ledger.get(&settled.id).unwrap().state, "settled");
    assert!(buyer.ledger.active().unwrap().is_empty());
    assert_eq!(chain.attempts.lock().unwrap().len(), 3);
}
#[tokio::test]
async fn admission_ticket_cannot_be_consumed_by_another_peer_or_concurrently() {
    let temp = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let seller = engine(temp.path(), &"bb".repeat(32), &"aa".repeat(32), true, chain);
    let request = serde_json::json!({"phase":"request_received","requested_by_node_id":"cc".repeat(32),"model":"model","exchange_id":"wrong","body":{"max_tokens":4}});
    assert_eq!(
        seller.hook(request).await.unwrap().decision,
        OpenAiAdmissionDecision::Deny
    );
    assert!(bounded_request(&serde_json::json!({"body":{}})).is_err());
    assert!(bounded_request(&serde_json::json!({"body":{"max_tokens":4097}})).is_err());
}

#[tokio::test]
async fn uncertain_admission_is_recovered_by_lookup_without_a_second_submission() {
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
    buyer
        .ledger
        .insert(&Exchange {
            id: id.clone(),
            peer: seller.node.clone(),
            model: "model".into(),
            payer: buyer.wallet.info.address.clone(),
            seller: false,
            exchange_id: Some("local-request".into()),
            state: "negotiating".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        })
        .unwrap();
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
    let messages: Vec<_> = seller.outbox.lock().unwrap().drain(..).collect();
    chain
        .lose_reply
        .store(true, std::sync::atomic::Ordering::SeqCst);
    for (_, message) in messages {
        assert!(buyer.message(&seller.node, message).await.is_err());
    }
    assert_eq!(chain.attempts.lock().unwrap().len(), 1);
    assert!(buyer.ledger.get(&id).unwrap().admission_receipt.is_none());
    buyer.retry().await.unwrap();
    pump(&buyer, &seller).await.unwrap();
    assert_eq!(buyer.ledger.get(&id).unwrap().state, "ready");
    assert!(seller.ledger.earned(&buyer.node).unwrap());
    assert_eq!(chain.attempts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn rejects_rehashed_payments_for_another_asset_or_expiry() {
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
        chain,
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
    let exchange = Exchange {
        id,
        peer: seller.node.clone(),
        model: "model".into(),
        payer: buyer.wallet.info.address.clone(),
        seller: false,
        exchange_id: None,
        state: "negotiating".into(),
        quote: None,
        usage: None,
        usage_payment: None,
        admission_receipt: None,
        usage_receipt: None,
        error: None,
    };
    let payment = quote.admission.clone().unwrap();
    buyer
        .validate_payment(&exchange, &quote, &payment, Purpose::Admission, 3)
        .unwrap();
    let mut changed = payment.obligation.clone();
    changed.asset.chain_id += 1;
    assert!(
        buyer
            .validate_payment(
                &exchange,
                &quote,
                &PaymentRequest::new(changed).unwrap(),
                Purpose::Admission,
                3
            )
            .is_err()
    );
    let mut changed = payment.obligation.clone();
    changed.expires_at = now() - 1;
    assert!(
        buyer
            .validate_payment(
                &exchange,
                &quote,
                &PaymentRequest::new(changed).unwrap(),
                Purpose::Admission,
                3
            )
            .is_err()
    );
    let mut changed = payment.obligation;
    changed.expires_at += 1;
    assert!(
        buyer
            .validate_payment(
                &exchange,
                &quote,
                &PaymentRequest::new(changed).unwrap(),
                Purpose::Admission,
                3
            )
            .is_err()
    );
    let mut identity = buyer.wallet.info.clone();
    identity.method = "other-verifier.v1".into();
    assert!(buyer.ledger.pin(&identity).is_err());
}

#[tokio::test]
async fn operator_resubmits_only_the_same_intent_after_prejournal_failure() {
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
    buyer
        .ledger
        .insert(&Exchange {
            id: id.clone(),
            peer: seller.node.clone(),
            model: "model".into(),
            payer: buyer.wallet.info.address.clone(),
            seller: false,
            exchange_id: Some("local-request".into()),
            state: "negotiating".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        })
        .unwrap();
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
    let messages: Vec<_> = seller.outbox.lock().unwrap().drain(..).collect();
    chain
        .reject_before_journal
        .store(true, std::sync::atomic::Ordering::SeqCst);
    for (_, message) in messages {
        assert!(buyer.message(&seller.node, message).await.is_err());
    }
    assert!(chain.attempts.lock().unwrap().is_empty());
    assert!(buyer.retry().await.is_err());
    assert!(chain.attempts.lock().unwrap().is_empty());
    buyer.resubmit(&id).await.unwrap();
    pump(&buyer, &seller).await.unwrap();
    buyer.resubmit(&id).await.unwrap();
    pump(&buyer, &seller).await.unwrap();
    assert_eq!(chain.attempts.lock().unwrap().len(), 1);
    assert_eq!(buyer.ledger.get(&id).unwrap().state, "ready");
}

#[tokio::test]
async fn unsupported_paid_endpoints_are_denied_before_any_transfer() {
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
    for endpoint in ["responses", "completions"] {
        let buying = serde_json::json!({"phase":"backend_selected","endpoint":endpoint,"target_peer_id":seller.node,"model":"model","exchange_id":"unsupported-buyer","body":{"max_tokens":4}});
        let decision = buyer.hook(buying).await.unwrap();
        assert_eq!(decision.decision, OpenAiAdmissionDecision::Deny);
        assert_eq!(decision.status, Some(402));
        let serving = serde_json::json!({"phase":"request_received","endpoint":endpoint,"requested_by_node_id":buyer.node,"model":"model","exchange_id":"unsupported-seller","body":{"max_tokens":4}});
        assert_eq!(
            seller.hook(serving.clone()).await.unwrap().decision,
            OpenAiAdmissionDecision::Deny
        );
        let mut free = serving.clone();
        free["model"] = serde_json::json!("unpriced");
        assert_eq!(
            seller.hook(free).await.unwrap().decision,
            OpenAiAdmissionDecision::Abstain
        );
        let mut local = serving;
        local
            .as_object_mut()
            .unwrap()
            .remove("requested_by_node_id");
        assert_eq!(
            seller.hook(local).await.unwrap().decision,
            OpenAiAdmissionDecision::Abstain
        );
        let observation = format!("alias-{endpoint}");
        let alias = serde_json::json!({"phase":"request_received","endpoint":endpoint,"model":"auto","requested_by_node_id":buyer.node,
            "exchange_id":observation,"observation_id":observation,"body":{"max_tokens":4}});
        assert_eq!(
            seller.hook(alias).await.unwrap().decision,
            OpenAiAdmissionDecision::Abstain
        );
        let selected = serde_json::json!({"phase":"backend_selected","model":"model","observation_id":observation,"exchange_id":observation,"target":"local"});
        let rejected = seller.hook(selected).await.unwrap();
        assert_eq!(rejected.status, Some(402));
        assert!(rejected.reason.unwrap().contains("chat_completions only"));
    }
    assert!(chain.attempts.lock().unwrap().is_empty());
    assert!(buyer.outbox.lock().unwrap().is_empty());
    assert!(seller.outbox.lock().unwrap().is_empty());
}

#[tokio::test]
async fn resolving_an_unpaid_alias_to_a_priced_model_is_denied() {
    let temp = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let peer = "aa".repeat(32);
    let seller = engine(temp.path(), &"bb".repeat(32), &peer, true, chain.clone());
    let ingress = serde_json::json!({"phase":"request_received","endpoint":"chat_completions","model":"auto","requested_by_node_id":peer,
        "exchange_id":"peer-asserted","observation_id":"host-observation","body":{"max_tokens":4}});
    assert_eq!(
        seller.hook(ingress).await.unwrap().decision,
        OpenAiAdmissionDecision::Abstain
    );
    let selected = serde_json::json!({"phase":"backend_selected","model":"model","target":"local","exchange_id":"peer-asserted","observation_id":"host-observation"});
    let denied = seller.hook(selected).await.unwrap();
    assert_eq!(denied.decision, OpenAiAdmissionDecision::Deny);
    assert_eq!(denied.status, Some(402));
    assert!(chain.attempts.lock().unwrap().is_empty());
    assert!(seller.outbox.lock().unwrap().is_empty());
    seller.hook(serde_json::json!({"phase":"exchange_finished","exchange_id":"peer-asserted","observation_id":"host-observation"})).await.unwrap();
    assert!(seller.ingress.lock().unwrap().is_empty());
}

#[tokio::test]
async fn resolved_alias_consumes_only_its_authenticated_peers_ready_ticket() {
    let temp = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let peer = "aa".repeat(32);
    let mut seller = engine(temp.path(), &"bb".repeat(32), &peer, true, chain.clone());
    Arc::get_mut(&mut seller).unwrap().config.admission_fee = 0;
    seller
        .message(
            &peer,
            Message::Hello {
                id: uuid::Uuid::new_v4().to_string(),
                model: "model".into(),
                payer: format!("0x{}", "11".repeat(20)),
            },
        )
        .await
        .unwrap();
    for (observation, source) in [("other-peer", "cc".repeat(32)), ("paid-peer", peer.clone())] {
        seller.hook(serde_json::json!({"phase":"request_received","endpoint":"chat_completions","model":"mesh","requested_by_node_id":source,
            "exchange_id":"same-peer-asserted-id","observation_id":observation,"body":{"max_tokens":4}})).await.unwrap();
    }
    let selected = |observation: &str| {
        serde_json::json!({"phase":"backend_selected","model":"model","target":"local",
        "exchange_id":"same-peer-asserted-id","observation_id":observation})
    };
    assert_eq!(
        seller.hook(selected("other-peer")).await.unwrap().decision,
        OpenAiAdmissionDecision::Deny
    );
    assert_eq!(
        seller.hook(selected("paid-peer")).await.unwrap().decision,
        OpenAiAdmissionDecision::Allow
    );
    assert_eq!(
        seller.hook(selected("paid-peer")).await.unwrap().decision,
        OpenAiAdmissionDecision::Allow
    );
    assert_eq!(
        seller.ledger.active().unwrap()[0].exchange_id.as_deref(),
        Some("paid-peer")
    );
    assert!(chain.attempts.lock().unwrap().is_empty());
}

#[tokio::test]
async fn free_and_local_resolved_models_do_not_enter_payment_flow() {
    let temp = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let peer = "aa".repeat(32);
    let seller = engine(temp.path(), &"bb".repeat(32), &peer, true, chain.clone());
    seller.hook(serde_json::json!({"phase":"request_received","endpoint":"chat_completions","model":"free","requested_by_node_id":peer,
        "exchange_id":"free","observation_id":"free-observation","body":{"max_tokens":4}})).await.unwrap();
    let free = serde_json::json!({"phase":"backend_selected","model":"free","target":"local","exchange_id":"free","observation_id":"free-observation"});
    assert_eq!(
        seller.hook(free).await.unwrap().decision,
        OpenAiAdmissionDecision::Abstain
    );
    let local = serde_json::json!({"phase":"backend_selected","model":"model","target":"local","exchange_id":"local","observation_id":"local-observation","body":{"max_tokens":4}});
    assert_eq!(
        seller.hook(local).await.unwrap().decision,
        OpenAiAdmissionDecision::Abstain
    );
    assert!(chain.attempts.lock().unwrap().is_empty());
    assert!(seller.ledger.active().unwrap().is_empty());
    assert!(seller.outbox.lock().unwrap().is_empty());
}

#[tokio::test]
async fn remote_tracking_capacity_fails_closed_and_terminal_releases_it() {
    let temp = tempfile::tempdir().unwrap();
    let chain = Arc::new(Chain::default());
    let peer = "aa".repeat(32);
    let seller = engine(temp.path(), &"bb".repeat(32), &peer, true, chain);
    for index in 0..256 {
        let event = serde_json::json!({"phase":"request_received","model":"free","requested_by_node_id":peer,"exchange_id":format!("{index}"),"body":{"max_tokens":4}});
        assert_eq!(
            seller.hook(event).await.unwrap().decision,
            OpenAiAdmissionDecision::Abstain
        );
    }
    let next = serde_json::json!({"phase":"request_received","model":"free","requested_by_node_id":peer,"exchange_id":"next","body":{"max_tokens":4}});
    assert_eq!(
        seller.hook(next.clone()).await.unwrap().decision,
        OpenAiAdmissionDecision::Deny
    );
    seller
        .hook(serde_json::json!({"phase":"exchange_finished","exchange_id":"0"}))
        .await
        .unwrap();
    assert_eq!(
        seller.hook(next).await.unwrap().decision,
        OpenAiAdmissionDecision::Abstain
    );
}

#[path = "engine_admission_tests.rs"]
mod pending;
