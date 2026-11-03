use super::*;
use crate::{
    bridge::Worker,
    wallet::{Settings, Wallet, asset, fee_asset},
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use wallet_contract::{
    Obligation, PayRequest, PaymentRequest, PaymentState, Purpose, Receipt, VerifyResult,
    WalletInfo,
};

struct Fixture {
    directory: PathBuf,
    prepared: Mutex<usize>,
    broadcasts: Mutex<Vec<Value>>,
    finalized: Mutex<bool>,
    mismatch: Mutex<bool>,
}

const PAYER: &str = "0x1111111111111111111111111111111111111111";
const RECIPIENT: &str = "0x2222222222222222222222222222222222222222";

fn request(id: &str) -> PayRequest {
    let obligation = Obligation {
        version: 1,
        id: id.into(),
        purpose: Purpose::Usage,
        buyer_peer: "aa".repeat(32),
        seller_peer: "bb".repeat(32),
        request_id: "fixture-request".into(),
        model: "fixture-model".into(),
        offer_id: "fixture-offer".into(),
        asset: asset(),
        payer: PAYER.into(),
        recipient: RECIPIENT.into(),
        amount: 17,
        expires_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 300,
    };
    PayRequest {
        payment_id: id.into(),
        request: PaymentRequest::new(obligation).unwrap(),
        max_fee: 0,
        max_total: 17,
    }
}

fn info() -> WalletInfo {
    WalletInfo {
        wallet_id: "deterministic-offline-fixture".into(),
        address: PAYER.into(),
        asset: asset(),
        method: wallet::METHOD.into(),
        can_send: true,
        fee_asset: Some(fee_asset()),
    }
}

fn setup() -> (tempfile::TempDir, Settings, Arc<Fixture>) {
    let directory = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let settings = Settings {
        data_dir: directory.path().into(),
        ..Default::default()
    };
    let fixture = Arc::new(Fixture {
        directory: directory.path().into(),
        prepared: Mutex::new(0),
        broadcasts: Mutex::new(Vec::new()),
        finalized: Mutex::new(false),
        mismatch: Mutex::new(false),
    });
    (directory, settings, fixture)
}

#[async_trait]
impl Worker for Fixture {
    async fn call(&self, operation: &str, input: Value) -> Result<Value> {
        match operation {
            "prepare" => {
                *self.prepared.lock().unwrap() += 1;
                let authorized: PayRequest = serde_json::from_value(input)?;
                Ok(json!({"signed_transaction":"0x02aabbcc",
                    "tx_hash":format!("0x{}", "11".repeat(32)),"expires_at":authorized.request.obligation.expires_at*1000,"fee":"20000000000000","nonce":"7"}))
            }
            "broadcast" => {
                // A separate connection sees the durable commit before broadcast.
                let db = rusqlite::Connection::open(self.directory.join("payments.sqlite3"))?;
                let signed: String =
                    db.query_row("SELECT signed FROM attempts", [], |row| row.get(0))?;
                assert_eq!(
                    serde_json::from_str::<Value>(&signed)?,
                    input["signed_transaction"]
                );
                self.broadcasts.lock().unwrap().push(input);
                // Simulate an ambiguous response after submission, not a proven failure.
                anyhow::bail!("fixture response lost after broadcast")
            }
            "verify" => {
                let request: VerifyRequest = serde_json::from_value(input)?;
                if !*self.finalized.lock().unwrap() {
                    return Ok(serde_json::to_value(VerifyResult {
                        state: PaymentState::Uncertain,
                        receipt: None,
                        reason: Some("awaiting finalized block".into()),
                    })?);
                }
                let receipt = Receipt {
                    tx_hash: request.tx_hash,
                    block_hash: format!("0x{}", "22".repeat(32)),
                    block_number: 100,
                    transfer_log_index: 0,
                    payer: PAYER.into(),
                    recipient: RECIPIENT.into(),
                    amount: if *self.mismatch.lock().unwrap() {
                        18
                    } else {
                        17
                    },
                    memo: request.request.memo,
                    finalized: true,
                };
                Ok(serde_json::to_value(VerifyResult {
                    state: PaymentState::Finalized,
                    receipt: Some(receipt),
                    reason: None,
                })?)
            }
            _ => anyhow::bail!("unexpected offline worker operation"),
        }
    }
}

#[tokio::test]
async fn journal_precedes_broadcast_and_restart_reconciles_without_another_signature() {
    let (_directory, settings, fixture) = setup();
    let wallet = Wallet::with_worker(settings.clone(), info(), fixture.clone()).unwrap();
    let payment = request("payment-one");
    let attempt = wallet.pay(payment.clone()).await.unwrap();
    assert_eq!(attempt.state, PaymentState::Uncertain);
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
    assert_eq!(wallet.journal.reserved().unwrap(), 17); // ETH wei are never counted as USDT.
    drop(wallet);
    *fixture.finalized.lock().unwrap() = true;
    let restarted = Wallet::with_worker(settings, info(), fixture.clone()).unwrap();
    let attempt = restarted.lookup("payment-one").await.unwrap();
    assert_eq!(attempt.state, PaymentState::Finalized);
    assert_eq!(restarted.journal.reserved().unwrap(), 0);
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
    assert_eq!(fixture.broadcasts.lock().unwrap().len(), 1);
    let repeated = restarted.pay(payment).await.unwrap();
    assert_eq!(repeated.tx_hashes, attempt.tx_hashes);
    assert_eq!(fixture.broadcasts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn duplicate_pay_rebroadcasts_only_same_signed_payload_and_changed_terms_are_rejected() {
    let (_directory, settings, fixture) = setup();
    let wallet = Wallet::with_worker(settings, info(), fixture.clone()).unwrap();
    let payment = request("same-id");
    wallet.pay(payment.clone()).await.unwrap();
    wallet.pay(payment.clone()).await.unwrap();
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
    {
        let broadcasts = fixture.broadcasts.lock().unwrap();
        assert_eq!(broadcasts.len(), 2);
        assert_eq!(broadcasts[0], broadcasts[1]);
    }
    let mut altered = payment;
    altered.request.obligation.amount = 18;
    altered.request.memo = altered.request.obligation.memo().unwrap();
    assert!(wallet.pay(altered).await.is_err());
    assert!(wallet.pay(request("another-id")).await.is_err());
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
}

#[tokio::test]
async fn expired_absence_stays_uncertain_and_never_releases_or_replaces_a_payment() {
    let (_directory, settings, fixture) = setup();
    let wallet = Wallet::with_worker(settings.clone(), info(), fixture.clone()).unwrap();
    wallet.pay(request("expired")).await.unwrap();
    let db = rusqlite::Connection::open(fixture.directory.join("payments.sqlite3")).unwrap();
    db.execute("UPDATE attempts SET expires_at='1'", [])
        .unwrap();
    let original = wallet.journal.get("expired").unwrap().unwrap().authorized;
    let pending = wallet.pay(original).await.unwrap();
    assert_eq!(pending.state, PaymentState::Uncertain);
    assert!(pending.error.unwrap().contains("remains valid"));
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
    assert_eq!(fixture.broadcasts.lock().unwrap().len(), 1);
    assert_eq!(wallet.journal.reserved().unwrap(), 17);
    assert!(!wallet.journal.unresolved().unwrap().is_empty());
    // Passing a local deadline does not invalidate a signed Ethereum transfer.
    // It can still become finalized later and must reconcile by its original hash.
    *fixture.finalized.lock().unwrap() = true;
    drop(wallet);
    let restarted = Wallet::with_worker(settings, info(), fixture.clone()).unwrap();
    let delivered = restarted.lookup("expired").await.unwrap();
    assert_eq!(delivered.state, PaymentState::Finalized);
    assert_eq!(restarted.journal.reserved().unwrap(), 0);
    assert_eq!(*fixture.prepared.lock().unwrap(), 1);
    assert_eq!(fixture.broadcasts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn mismatched_finalized_receipt_is_rejected_and_provider_is_binding_only() {
    let (_directory, settings, fixture) = setup();
    *fixture.finalized.lock().unwrap() = true;
    *fixture.mismatch.lock().unwrap() = true;
    let wallet = Wallet::with_worker(settings, info(), fixture).unwrap();
    let request = VerifyRequest {
        request: request("receipt").request,
        tx_hash: format!("0x{}", "11".repeat(32)),
    };
    assert!(wallet.verify(request).await.is_err());
    let manifest = server::manifest();
    assert!(manifest.operations.is_empty());
    assert!(
        manifest
            .capabilities
            .iter()
            .any(|capability| capability == wallet_contract::CAPABILITY)
    );
}

#[test]
fn journal_excludes_a_second_process_and_another_wallet_identity() {
    let (_directory, settings, _fixture) = setup();
    let journal = journal::Journal::open(&settings.data_dir, &info()).unwrap();
    assert!(journal::Journal::open(&settings.data_dir, &info()).is_err());
    drop(journal);
    let mut other = info();
    other.address = RECIPIENT.into();
    assert!(journal::Journal::open(&settings.data_dir, &other).is_err());
}

#[tokio::test]
async fn rust_bridge_reads_an_offline_profile_and_stays_synchronized_after_a_worker_refusal() {
    let (directory, mut settings, _fixture) = setup();
    settings.data_dir = directory.path().canonicalize().unwrap();
    let profile = serde_json::json!({"version":1,"wallet_id":"offline-bridge-fixture",
        "address":PAYER,"chain_id":wallet::CHAIN_ID,"token":wallet::USDT});
    let path = settings.data_dir.join("wallet.json");
    std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    // This profile has no encrypted seed, no account creation and no RPC endpoint.
    let worker = bridge::Bridge::spawn(&settings).await.unwrap();
    let read = bridge::invoke(worker.as_ref(), "profile", settings.worker_settings())
        .await
        .unwrap();
    assert_eq!(read, profile);
    assert!(
        bridge::invoke(worker.as_ref(), "unsupported", json!({}))
            .await
            .is_err()
    );
    let repeated = bridge::invoke(worker.as_ref(), "profile", settings.worker_settings())
        .await
        .unwrap();
    assert_eq!(repeated, profile);
}
