//! Deterministic, offline fixtures. This well-known test key must never be funded.
use crate::{
    journal::Journal,
    keys::Profile,
    rpc::address,
    wallet::{Settings, Wallet},
};
use alloy::{
    consensus::transaction::SignerRecoverable,
    eips::eip2718::{Decodable2718, Encodable2718},
    network::{EthereumWallet, NetworkTransactionBuilder, TransactionBuilder},
    primitives::{B256, U256, keccak256},
    signers::local::PrivateKeySigner,
    sol_types::SolCall,
};
use std::num::NonZeroU64;
use tempo_alloy::{
    contracts::precompiles::ITIP20, primitives::TempoTxEnvelope, rpc::TempoTransactionRequest,
};
use wallet_contract::{
    Asset, CHAIN_ID, OUSD, Obligation, PayRequest, PaymentAttempt, PaymentRequest, PaymentState,
    Purpose,
};

fn private_tempdir() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    directory
}

fn fixture_request(payer: String) -> PayRequest {
    let request = PaymentRequest::new(Obligation {
        version: 1,
        id: "obligation-fixture".into(),
        purpose: Purpose::Usage,
        buyer_peer: "buyer".into(),
        seller_peer: "seller".into(),
        request_id: "request-fixture".into(),
        model: "fixture".into(),
        offer_id: "offer-fixture".into(),
        asset: Asset::default(),
        payer,
        recipient: "0x2222222222222222222222222222222222222222".into(),
        amount: 7,
        expires_at: 4_000_000_000,
    })
    .unwrap();
    PayRequest {
        payment_id: "payment-fixture".into(),
        request,
        max_fee: 100,
        max_total: 107,
    }
}

async fn signed_fixture() -> (PayRequest, Vec<u8>) {
    let signer = PrivateKeySigner::from_bytes(&B256::repeat_byte(1)).unwrap();
    let payment = fixture_request(signer.address().to_string());
    let data = ITIP20::transferWithMemoCall {
        to: address(&payment.request.obligation.recipient).unwrap(),
        amount: U256::from(7),
        memo: payment.request.memo.parse().unwrap(),
    }
    .abi_encode();
    let request = TempoTransactionRequest::default()
        .with_chain_id(CHAIN_ID)
        .with_from(signer.address())
        .with_to(address(OUSD).unwrap())
        .with_value(U256::ZERO)
        .with_input(data.clone())
        .with_nonce(9)
        .with_nonce_key(U256::ZERO)
        .with_fee_token(address(OUSD).unwrap())
        .with_valid_before(NonZeroU64::new(payment.request.obligation.expires_at).unwrap())
        .with_gas_limit(60_000)
        .with_max_fee_per_gas(1_200_000_000)
        .with_max_priority_fee_per_gas(0);
    let envelope = request
        .build(&EthereumWallet::from(signer.clone()))
        .await
        .unwrap();
    let raw = envelope.encoded_2718();
    assert_eq!(raw[0], 0x76);
    let mut input = raw.as_slice();
    let decoded = TempoTxEnvelope::decode_2718(&mut input).unwrap();
    assert!(input.is_empty());
    assert_eq!(decoded.recover_signer().unwrap(), signer.address());
    let TempoTxEnvelope::AA(decoded) = decoded else {
        panic!("expected native Tempo transaction");
    };
    assert_eq!(*decoded.hash(), keccak256(&raw));
    let tx = decoded.tx();
    assert_eq!(tx.chain_id, CHAIN_ID);
    assert_eq!(tx.nonce, 9);
    assert_eq!(tx.nonce_key, U256::ZERO);
    assert_eq!(tx.fee_token, Some(address(OUSD).unwrap()));
    assert_eq!(
        tx.valid_before.unwrap().get(),
        payment.request.obligation.expires_at
    );
    assert_eq!(tx.calls.len(), 1);
    assert_eq!(tx.calls[0].input.as_ref(), data.as_slice());
    assert!(tx.fee_payer_signature.is_none());
    assert!(tx.key_authorization.is_none());
    assert!(tx.tempo_authorization_list.is_empty());
    (payment, raw)
}

fn attempt(payment: &PayRequest, raw: &[u8]) -> PaymentAttempt {
    PaymentAttempt {
        payment_id: payment.payment_id.clone(),
        request: payment.request.clone(),
        state: PaymentState::Prepared,
        tx_hashes: vec![keccak256(raw).to_string()],
        receipt: None,
        error: None,
    }
}

#[tokio::test]
async fn native_signature_and_journal_survive_restart_before_broadcast() {
    let (payment, raw) = signed_fixture().await;
    let dir = private_tempdir();
    let journal = Journal::open(dir.path(), "fixture", &payment.request.obligation.payer).unwrap();
    journal
        .insert(&payment, &attempt(&payment, &raw), 9, &raw, 79)
        .unwrap();
    assert_eq!(journal.reserved().unwrap(), 79);
    drop(journal); // Simulated crash boundary: signed and committed, no network submission.
    let journal = Journal::open(dir.path(), "fixture", &payment.request.obligation.payer).unwrap();
    let stored = journal.get(&payment.payment_id).unwrap().unwrap();
    assert_eq!(stored.raw, raw);
    assert_eq!(stored.nonce, 9);
    assert_eq!(stored.attempt.state, PaymentState::Prepared);
    assert_eq!(
        stored.attempt.tx_hashes[0],
        keccak256(&stored.raw).to_string()
    );
    assert_eq!(stored.request.request, payment.request);
    assert_eq!(journal.reserved().unwrap(), 79);
    let mut second = payment.clone();
    second.payment_id = "second-payment".into();
    assert!(
        journal
            .insert(&second, &attempt(&second, &raw), 9, &raw, 79)
            .is_err()
    );
    let mut expired = stored.attempt;
    expired.state = PaymentState::Expired;
    journal.update(&expired).unwrap();
    assert_eq!(journal.reserved().unwrap(), 0);
    journal
        .insert(&second, &attempt(&second, &raw), 9, &raw, 79)
        .unwrap();
    assert_eq!(journal.list().unwrap().len(), 2);
}

#[tokio::test]
async fn reused_payment_id_with_changed_economics_is_refused_without_rpc() {
    let (payment, raw) = signed_fixture().await;
    let dir = private_tempdir();
    let profile = Profile {
        wallet_id: "fixture".into(),
        address: payment.request.obligation.payer.clone(),
        chain_id: CHAIN_ID,
        token: OUSD.into(),
        key_file: "fixture-keystore-not-unlocked.json".into(),
    };
    std::fs::write(
        dir.path().join("wallet.json"),
        serde_json::to_vec(&profile).unwrap(),
    )
    .unwrap();
    let wallet = Wallet::load(Settings {
        data_dir: dir.path().into(),
        rpc_url: "http://127.0.0.1:1".into(),
        password_file: None,
        ..Default::default()
    })
    .unwrap();
    wallet
        .journal
        .insert(&payment, &attempt(&payment, &raw), 9, &raw, 79)
        .unwrap();
    let mut changed = payment;
    changed.request.obligation.amount = 8;
    changed.request = PaymentRequest::new(changed.request.obligation).unwrap();
    let error = wallet.pay(changed).await.unwrap_err().to_string();
    assert!(error.contains("different immutable economic fields"));
    assert_eq!(wallet.journal.list().unwrap().len(), 1);
    let mut self_payment = fixture_request(wallet.profile.address.clone());
    self_payment.request.obligation.recipient =
        format!("0x{}", &wallet.profile.address[2..].to_ascii_uppercase());
    self_payment.request = PaymentRequest::new(self_payment.request.obligation).unwrap();
    assert!(
        wallet
            .pay(self_payment)
            .await
            .unwrap_err()
            .to_string()
            .contains("self-payment")
    );
}

#[test]
fn locked_wallet_never_auto_creates_a_key() {
    let dir = private_tempdir();
    assert!(
        Wallet::load(Settings {
            data_dir: dir.path().into(),
            ..Default::default()
        })
        .is_err()
    );
    assert!(!dir.path().join("wallet.json").exists());
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn wallet_operations_are_available_only_through_explicit_bindings() {
    let manifest = crate::server::manifest();
    assert!(manifest.operations.is_empty());
    assert!(
        manifest
            .capabilities
            .iter()
            .any(|capability| capability == wallet_contract::CAPABILITY)
    );
}
