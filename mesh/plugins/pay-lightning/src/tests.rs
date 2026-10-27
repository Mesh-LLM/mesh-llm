use super::*;
use async_trait::async_trait;
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
use mesh_llm_payments::ledger::{ApprovalMode, Policy};
use mesh_llm_wallet::provider::*;
use std::sync::{
    Mutex,
    atomic::{AtomicU8, AtomicUsize, Ordering},
};

#[derive(Default)]
struct Network {
    next: AtomicU8,
    received: Mutex<BTreeMap<String, Transaction>>,
    sends: AtomicUsize,
}
struct Wallet(Arc<Network>, bool);
pub(crate) fn invoice(number: u8, amount: u64) -> Invoice {
    let secret = SecretKey::from_slice(&[7; 32]).unwrap();
    let raw = InvoiceBuilder::new(Currency::Bitcoin)
        .description("offline fixture".into())
        .payment_hash(PaymentHash([number; 32]))
        .payment_secret(PaymentSecret([42; 32]))
        .current_timestamp()
        .expiry_time(std::time::Duration::from_secs(3600))
        .min_final_cltv_expiry_delta(144)
        .amount_milli_satoshis(amount)
        .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
        .unwrap()
        .to_string();
    Invoice::parse(&raw).unwrap()
}
#[async_trait]
impl WalletProvider for Wallet {
    async fn balance(&self) -> Result<Balance> {
        Ok(Balance {
            spendable_msat: 1_000_000,
        })
    }
    async fn transactions(&self, _: usize) -> Result<Vec<Transaction>> {
        Ok(vec![])
    }
    async fn create_invoice(&self, amount: Option<u64>, _: u32) -> Result<Invoice> {
        Ok(invoice(
            self.0.next.fetch_add(1, Ordering::SeqCst) + 1,
            amount.unwrap(),
        ))
    }
    async fn pay(
        &self,
        invoice: &Invoice,
        amount: u64,
        _: u64,
    ) -> std::result::Result<Transaction, PayError> {
        self.0.sends.fetch_add(1, Ordering::SeqCst);
        let payment = Transaction {
            id: invoice.payment_hash.clone(),
            payment_hash: Some(invoice.payment_hash.clone()),
            inbound: false,
            amount_msat: amount,
            fee_msat: 0,
            status: PaymentStatus::Succeeded,
            claiming: false,
            status_msg: None,
            created_at_ms: mesh_llm_wallet::now_ms(),
            settled_at_ms: Some(mesh_llm_wallet::now_ms()),
        };
        let mut incoming = payment.clone();
        incoming.inbound = true;
        self.0
            .received
            .lock()
            .unwrap()
            .insert(invoice.payment_hash.clone(), incoming);
        Ok(payment)
    }
    async fn lookup(&self, hash: &str) -> Result<Option<Transaction>> {
        Ok(self
            .0
            .received
            .lock()
            .unwrap()
            .get(hash)
            .cloned()
            .map(|mut payment| {
                payment.inbound = self.1;
                payment
            }))
    }
    async fn wait_for_payment(&self, hash: &str) -> Result<Transaction> {
        self.lookup(hash)
            .await?
            .context("fixture payment not received")
    }
}
struct Fixture {
    seller: Engine,
    buyer: Engine,
    network: Arc<Network>,
    _seller_dir: tempfile::TempDir,
    _buyer_dir: tempfile::TempDir,
}
fn price() -> Pricing {
    Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        cache_discount_percent: 50,
    }
}
impl Fixture {
    fn new() -> Self {
        let network = Arc::new(Network::default());
        let seller_dir = tempfile::tempdir().unwrap();
        let buyer_dir = tempfile::tempdir().unwrap();
        let seller_service = Arc::new(
            PaymentService::with_provider(
                seller_dir.path(),
                Arc::new(Wallet(network.clone(), true)),
            )
            .unwrap(),
        );
        seller_service
            .ledger
            .set_pricing("model", Some(&price()))
            .unwrap();
        let buyer_service = Arc::new(
            PaymentService::with_provider(
                buyer_dir.path(),
                Arc::new(Wallet(network.clone(), false)),
            )
            .unwrap(),
        );
        buyer_service
            .ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(100_000),
            })
            .unwrap();
        let seller = Engine::new(seller_service, "706179", 0, seller_dir.path()).unwrap();
        let mut buyer = Engine::new(buyer_service, "706179", 100_000, buyer_dir.path()).unwrap();
        buyer.buying.insert("model".into(), price());
        buyer.buy_peers.insert("seller".into());
        Self {
            seller,
            buyer,
            network,
            _seller_dir: seller_dir,
            _buyer_dir: buyer_dir,
        }
    }
    async fn prepare(&self) -> String {
        let admission = self
            .seller
            .channel("buyer", Message::Quote)
            .await
            .unwrap()
            .remove(0);
        let Message::Admission { terms, .. } = &admission else {
            panic!("not admission")
        };
        let id = terms.id.clone();
        let paid = self
            .buyer
            .channel("seller", admission)
            .await
            .unwrap()
            .remove(0);
        self.seller.channel("buyer", paid).await.unwrap();
        id
    }
}
fn gate(exchange: &str, peer: &str) -> Value {
    json!({"phase":"request_received","observation_point":"serving_host_ingress","model":"model","requested_by_node_id":peer,"exchange_id":exchange})
}
fn terminal(exchange: &str, usage: Value) -> Value {
    json!({"phase":"exchange_finished","ingress_observation_point":"serving_host_ingress","exchange_id":exchange,"execution_outcome":"success","status":200,"usage":usage})
}

#[tokio::test]
async fn cold_deposit_gate_final_recovery_and_warm_admission() {
    let f = Fixture::new();
    assert_eq!(
        f.seller.gate(&gate("cold", "buyer")).unwrap().status,
        Some(402)
    );
    let id = f.prepare().await;
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 1);
    assert_eq!(
        f.seller.gate(&gate("x", "attacker")).unwrap().status,
        Some(402)
    );
    assert_eq!(
        f.seller.gate(&gate("x", "buyer")).unwrap().decision,
        Decision::Allow
    );
    assert_eq!(
        f.seller.gate(&gate("y", "buyer")).unwrap().status,
        Some(402)
    );
    let (_,settlement)=f.seller.terminal(&terminal("x",json!({"prompt_tokens":2000,"completion_tokens":1000,"prompt_tokens_details":{"cached_tokens":1000}}))).await.unwrap().unwrap();
    let Message::Settlement { bill, .. } = &settlement else {
        panic!("not bill")
    };
    assert_eq!(bill.bill_msat, 2500);
    assert_eq!(bill.remaining_msat, 1500);
    assert!(
        f.seller
            .channel("attacker", Message::Recover { id: id.clone() })
            .await
            .is_err()
    );
    let recovered = f
        .seller
        .channel("buyer", Message::Recover { id: id.clone() })
        .await
        .unwrap()
        .remove(0);
    assert_eq!(
        serde_json::to_value(&settlement).unwrap(),
        serde_json::to_value(&recovered).unwrap()
    );
    let paid = f
        .buyer
        .channel("seller", settlement)
        .await
        .unwrap()
        .remove(0);
    let ready = f.seller.channel("buyer", paid).await.unwrap();
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 2);
    assert!(f.seller.service.metered_trusted("buyer").await.unwrap());
    let warm = ready
        .into_iter()
        .find(|m| matches!(m, Message::Admission { .. }))
        .unwrap();
    let Message::Admission { terms, invoice } = &warm else {
        unreachable!()
    };
    assert_eq!(terms.deposit_msat, 0);
    assert!(invoice.is_none());
    let paid = f.buyer.channel("seller", warm).await.unwrap().remove(0);
    f.seller.channel("buyer", paid).await.unwrap();
    assert_eq!(
        f.seller.gate(&gate("warm", "buyer")).unwrap().decision,
        Decision::Allow
    );
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn absent_usage_abandons_without_bill_or_strike() {
    let f = Fixture::new();
    let id = f.prepare().await;
    f.seller.gate(&gate("x", "buyer")).unwrap();
    let (_, message) = f
        .seller
        .terminal(&terminal("x", Value::Null))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(message, Message::Abandoned { .. }));
    assert_eq!(
        f.seller
            .service
            .ledger
            .metered_unpaid_finals("buyer")
            .unwrap(),
        0
    );
    assert!(matches!(
        f.seller
            .channel("buyer", Message::Recover { id })
            .await
            .unwrap()[0],
        Message::Abandoned { .. }
    ));
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn unsolicited_seller_or_excess_rates_never_pay() {
    let f = Fixture::new();
    let admission = f
        .seller
        .channel("buyer", Message::Quote)
        .await
        .unwrap()
        .remove(0);
    assert!(f.buyer.channel("attacker", admission).await.is_err());
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn forged_final_bill_and_wrong_sender_are_rejected() {
    let f = Fixture::new();
    let id = f.prepare().await;
    f.seller.gate(&gate("x", "buyer")).unwrap();
    let (_, message) = f
        .seller
        .terminal(&terminal(
            "x",
            json!({"prompt_tokens":2000,"completion_tokens":1000}),
        ))
        .await
        .unwrap()
        .unwrap();
    let Message::Settlement { mut bill, .. } = message else {
        unreachable!()
    };
    bill.bill_msat += 1;
    assert!(
        f.buyer
            .channel("seller", Message::Settlement { id, bill })
            .await
            .is_err()
    );
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 1);
}
#[test]
fn local_gateway_and_unpriced_models_are_not_gated() {
    let f = Fixture::new();
    let mut event = gate("x", "buyer");
    event["observation_point"] = json!("gateway_ingress");
    assert_eq!(f.seller.gate(&event).unwrap().decision, Decision::Abstain);
    event = gate("x", "buyer");
    event["model"] = json!("free");
    assert_eq!(f.seller.gate(&event).unwrap().decision, Decision::Abstain);
}

#[test]
fn normalized_cached_usage_and_partial_usage_are_not_confused() {
    let event = terminal(
        "x",
        json!({"prompt_tokens":10,"completion_tokens":5,"cached_prompt_tokens":6}),
    );
    assert_eq!(reported_usage(&event).unwrap().cached_tokens, 6);
    assert!(reported_usage(&terminal("x", json!({"prompt_tokens":10}))).is_none());
    assert!(
        reported_usage(&terminal(
            "x",
            json!({"prompt_tokens":10,"completion_tokens":5,"cached_prompt_tokens":11})
        ))
        .is_none()
    );
}
#[tokio::test]
async fn duplicate_admission_and_paid_replay_do_not_buy_or_dispatch_twice() {
    let f = Fixture::new();
    let admission = f
        .seller
        .channel("buyer", Message::Quote)
        .await
        .unwrap()
        .remove(0);
    let duplicate: Message =
        serde_json::from_value(serde_json::to_value(&admission).unwrap()).unwrap();
    let paid = f
        .buyer
        .channel("seller", admission)
        .await
        .unwrap()
        .remove(0);
    let Message::Paid { id } = &paid else {
        unreachable!()
    };
    let id = id.clone();
    f.seller.channel("buyer", paid).await.unwrap();
    f.buyer.channel("seller", duplicate).await.unwrap();
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 1);
    f.seller.gate(&gate("x", "buyer")).unwrap();
    f.seller
        .channel("buyer", Message::Paid { id })
        .await
        .unwrap();
    assert_eq!(
        f.seller.gate(&gate("y", "buyer")).unwrap().status,
        Some(402)
    );
}
#[tokio::test]
async fn rates_above_local_ceiling_do_not_pay() {
    let f = Fixture::new();
    let Message::Admission { mut terms, invoice } = f
        .seller
        .channel("buyer", Message::Quote)
        .await
        .unwrap()
        .remove(0)
    else {
        unreachable!()
    };
    terms.pricing.output_msat_per_million += 1;
    assert!(
        f.buyer
            .channel("seller", Message::Admission { terms, invoice })
            .await
            .is_err()
    );
    assert_eq!(f.network.sends.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn repeated_quote_returns_same_unconsumed_invoice() {
    let f = Fixture::new();
    let first = f.seller.channel("buyer", Message::Quote).await.unwrap();
    let second = f.seller.channel("buyer", Message::Quote).await.unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(second).unwrap()
    );
    assert_eq!(f.network.next.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn peer_reconnect_recovers_debt_before_new_quote() {
    let f = Fixture::new();
    let id = f.prepare().await;
    let messages = f.buyer.peer_up("seller").await.unwrap();
    assert!(matches!(&messages[0],Message::Recover{id:recovered} if recovered==&id));
    assert!(matches!(messages.last().unwrap(), Message::Quote));
    assert!(f.buyer.peer_up("attacker").await.unwrap().is_empty());
}

#[tokio::test]
async fn gateway_terminal_cannot_settle_sellers_ticket() {
    let f = Fixture::new();
    f.prepare().await;
    f.seller.gate(&gate("x", "buyer")).unwrap();
    let mut event = terminal("x", json!({"prompt_tokens":9000,"completion_tokens":9000}));
    event["ingress_observation_point"] = json!("gateway_ingress");
    assert!(f.seller.terminal(&event).await.unwrap().is_none());
}
