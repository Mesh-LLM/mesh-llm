use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};

use crate::{
    invoice::Invoice,
    ledger::{ApprovalMode, Charge, Ledger, Policy, RequestTerms},
    pricing::Pricing,
    service::{Arrival, PaymentService},
    wallet::{Balance, PayError, PaymentStatus, Transaction, WalletProvider},
};

mod blocking;
mod failure_boundaries;
mod payment_notifications;
mod provisioning;
mod recovery_boundaries;
mod review_regressions;

fn invoice(number: u8, amount: u64) -> Invoice {
    invoice_with_expiry(number, amount, 3600)
}

fn invoice_with_expiry(number: u8, amount: u64, seconds: u64) -> Invoice {
    let secret = SecretKey::from_slice(&[7; 32]).unwrap();
    let bolt11 = InvoiceBuilder::new(Currency::Bitcoin)
        .description("test".into())
        .payment_hash(PaymentHash([number; 32]))
        .payment_secret(PaymentSecret([42; 32]))
        .current_timestamp()
        .expiry_time(std::time::Duration::from_secs(seconds))
        .min_final_cltv_expiry_delta(144)
        .amount_milli_satoshis(amount)
        .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
        .unwrap()
        .to_string();
    Invoice::parse(&bolt11).unwrap()
}

fn terms(id: &str, cap: u64) -> RequestTerms {
    RequestTerms {
        exchange_id: None,
        id: id.into(),
        peer: "peer".into(),
        payee: None,
        model: "test".into(),
        pricing: Pricing {
            input_msat_per_million: 1000,
            output_msat_per_million: 1000,
            minimum_invoice_msat: 1,
        },
        input_tokens: 100,
        max_output_tokens: 100,
        max_total_msat: cap,
        expires_at_ms: crate::now_ms() + 3_600_000,
    }
}

fn charge(id: &str, segment: u32, number: u8, amount: u64, cap: u64) -> Charge {
    Charge {
        request_id: id.into(),
        segment,
        invoice: invoice(number, amount),
        amount_msat: amount,
        max_total_msat: cap,
    }
}

#[derive(Default)]
struct MockWallet {
    payments: Mutex<HashMap<String, Transaction>>,
    calls: AtomicUsize,
    lose_response: AtomicBool,
    pending: AtomicBool,
    lookup_unavailable: AtomicBool,
    updates: tokio::sync::Notify,
    lookups: AtomicUsize,
    waits: AtomicUsize,
    arrival_waits: AtomicUsize,
    settle_during_lookup: AtomicBool,
    reject_submission: AtomicBool,
    terminal_failure: AtomicBool,
    hide_payments: AtomicBool,
    invoice_unavailable: AtomicBool,
    /// Expiry the service asked for on each `create_invoice`.
    invoice_expiries: Mutex<Vec<u32>>,
}

#[async_trait]
impl WalletProvider for MockWallet {
    async fn balance(&self) -> Result<Balance> {
        Ok(Balance {
            spendable_msat: 100_000,
        })
    }
    async fn transactions(&self, _: usize) -> Result<Vec<Transaction>> {
        Ok(self.payments.lock().unwrap().values().cloned().collect())
    }
    async fn create_invoice(&self, amount: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        anyhow::ensure!(
            !self.invoice_unavailable.load(Ordering::SeqCst),
            "invoice service unavailable"
        );
        self.invoice_expiries.lock().unwrap().push(expiry_secs);
        Ok(invoice_with_expiry(
            200,
            amount.unwrap_or(1000),
            u64::from(expiry_secs),
        ))
    }
    async fn lookup(&self, hash: &str) -> Result<Option<Transaction>> {
        self.lookups.fetch_add(1, Ordering::SeqCst);
        anyhow::ensure!(
            !self.lookup_unavailable.load(Ordering::SeqCst),
            "wallet status unavailable"
        );
        if self.hide_payments.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let observed = self.payments.lock().unwrap().get(hash).cloned();
        if self.settle_during_lookup.swap(false, Ordering::SeqCst) {
            {
                let mut payments = self.payments.lock().unwrap();
                let payment = payments.get_mut(hash).unwrap();
                payment.status = PaymentStatus::Succeeded;
                payment.settled_at_ms = Some(crate::now_ms());
            }
            self.updates.notify_waiters();
        }
        Ok(observed)
    }
    async fn wait_for_payment(&self, hash: &str) -> Result<Transaction> {
        self.waits.fetch_add(1, Ordering::SeqCst);
        loop {
            // Register before lookup so an event during the lookup wakes us.
            let changed = self.updates.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(payment) = self.lookup(hash).await?
                && payment.status != PaymentStatus::Pending
            {
                return Ok(payment);
            }
            changed.await;
        }
    }
    async fn wait_for_arrival(&self, hash: &str) -> Result<Transaction> {
        self.arrival_waits.fetch_add(1, Ordering::SeqCst);
        loop {
            let changed = self.updates.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(payment) = self.lookup(hash).await?
                && (payment.status != PaymentStatus::Pending || payment.is_claiming())
            {
                return Ok(payment);
            }
            changed.await;
        }
    }
    async fn pay(&self, invoice: &Invoice, amount: u64, cap: u64) -> Result<Transaction, PayError> {
        assert!(amount + 10 <= cap);
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.reject_submission.load(Ordering::SeqCst) {
            return Err(PayError::NotSubmitted(anyhow::anyhow!(
                "preflight rejected"
            )));
        }
        let pending = self.pending.load(Ordering::SeqCst);
        let payment = Transaction {
            id: invoice.payment_hash.clone(),
            payment_hash: Some(invoice.payment_hash.clone()),
            inbound: false,
            amount_msat: amount,
            fee_msat: 10,
            status: if self.terminal_failure.load(Ordering::SeqCst) {
                PaymentStatus::Failed
            } else if pending {
                PaymentStatus::Pending
            } else {
                PaymentStatus::Succeeded
            },
            claiming: false,
            status_msg: None,
            created_at_ms: crate::now_ms(),
            settled_at_ms: (!pending).then(crate::now_ms),
        };
        self.payments
            .lock()
            .unwrap()
            .insert(invoice.payment_hash.clone(), payment.clone());
        self.updates.notify_waiters();
        if self.lose_response.swap(false, Ordering::SeqCst) {
            return Err(PayError::Uncertain(anyhow::anyhow!(
                "response lost after submission"
            )));
        }
        Ok(payment)
    }
}

#[tokio::test]
async fn lost_payment_response_recovers_after_restart_without_double_spend() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_response.store(true, Ordering::SeqCst);
    let request = terms("one", 1000);
    let charge = charge("one", 0, 1, 100, 200);
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        service
            .ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(100_000),
            })
            .unwrap();
        service.ledger.propose(&request).unwrap();
        service.approve("one").await.unwrap();
        assert!(service.pay_charge(&charge).await.is_err());
        assert_eq!(service.ledger.pending_charges().unwrap().len(), 1);
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    service.reconcile_pending().await.unwrap();
    service.pay_charge(&charge).await.unwrap();
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.requests().unwrap()[0].spent_msat, 110);
    assert!(service.ledger.pending_charges().unwrap().is_empty());
}

#[tokio::test]
async fn a_fresh_charge_pays_without_a_lookup() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    service
        .ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(1000),
        })
        .unwrap();
    service
        .await_authorization(&terms("one", 1000))
        .await
        .unwrap();
    service
        .pay_charge(&charge("one", 0, 1, 100, 200))
        .await
        .unwrap();
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), 0);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn output_payment_uses_the_original_authorization_and_actual_fees() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    service
        .ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(1000),
        })
        .unwrap();
    service
        .await_authorization(&terms("one", 1000))
        .await
        .unwrap();
    service
        .pay_charge(&charge("one", 0, 1, 100, 200))
        .await
        .unwrap();
    service
        .pay_charge(&charge("one", 1, 2, 400, 500))
        .await
        .unwrap();
    service.ledger.finish("one").unwrap();
    assert_eq!(service.ledger.requests().unwrap()[0].spent_msat, 520);
    assert_eq!(
        service
            .ledger
            .available_budget(100_000, crate::now_ms())
            .unwrap(),
        480
    );
}

#[test]
fn duplicate_segments_and_hashes_cannot_change_the_debit() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(dir.path()).unwrap();
    ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(100_000),
        })
        .unwrap();
    ledger.propose(&terms("one", 1000)).unwrap();
    ledger.approve("one", 100_000, crate::now_ms()).unwrap();
    let original = charge("one", 0, 1, 100, 200);
    assert!(ledger.prepare_charge(&original).unwrap());
    assert!(!ledger.prepare_charge(&original).unwrap());
    let mut changed = original.clone();
    changed.max_total_msat += 1;
    assert!(ledger.prepare_charge(&changed).is_err());
    changed = original.clone();
    changed.segment = 1;
    assert!(ledger.prepare_charge(&changed).is_err());
    assert!(
        ledger
            .prepare_charge(&charge("one", 1, 2, 900, 1000))
            .is_err()
    );
}

#[test]
fn invoices_reject_peer_metadata_substitution_and_amount_changes() {
    let mut invoice = invoice(1, 100);
    assert!(invoice.validate_payment(101, crate::now_ms()).is_err());
    assert!(
        invoice
            .validate_payment(100, invoice.expires_at_ms)
            .is_err()
    );
    invoice.payment_hash = "00".repeat(32);
    assert!(invoice.validate_payment(100, crate::now_ms()).is_err());
}

#[test]
fn committed_output_debt_survives_process_restart() {
    let dir = tempfile::tempdir().unwrap();
    let price = terms("one", 1000).pricing;
    {
        let service =
            PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default())).unwrap();
        service
            .ledger
            .begin_serving("one", "peer", &price, 2)
            .unwrap();
        service.ledger.record_delivered_tokens("one", 2).unwrap();
        service.ledger.record_delivered_tokens("one", 2).unwrap();
        assert!(service.ledger.record_delivered_tokens("one", 3).is_err());
    }
    let service =
        PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default())).unwrap();
    let (_, _, tokens, finished) = service.ledger.serving_account("one").unwrap();
    assert_eq!(tokens, 2);
    assert!(finished);
    assert!(service.ledger.record_delivered_tokens("one", 3).is_err());
}

#[tokio::test]
async fn oversized_wire_frame_is_rejected_before_allocation() {
    use tokio::io::AsyncWriteExt;
    let (mut writer, mut reader) = tokio::io::duplex(64);
    writer.write_u32(u32::MAX).await.unwrap();
    assert!(crate::wire::read(&mut reader).await.is_err());
}

#[tokio::test]
async fn one_policy_command_enables_paid_use_and_free_only_stops_new_work() -> Result<()> {
    use crate::control::ControlCommand;
    use crate::intent::PaymentIntent;
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let read = service
        .control(ControlCommand::Policy { value: None })
        .await?;
    assert_eq!(read["mode"], "free_only");
    assert!(
        service
            .await_authorization(&terms("off", 1000))
            .await
            .is_err()
    );
    assert!(service.ledger.requests()?.is_empty());
    service
        .control(ControlCommand::Policy {
            value: Some(Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(1000),
            }),
        })
        .await?;
    assert!(matches!(
        service.ledger.payment_intent()?,
        PaymentIntent::AllowPaid { .. }
    ));
    service.await_authorization(&terms("paid", 700)).await?;
    assert!(
        service
            .await_authorization(&terms("over", 400))
            .await
            .is_err()
    );
    let status = service
        .control(ControlCommand::Policy { value: None })
        .await?;
    assert_eq!(status["reserved_msat"], 700);
    assert_eq!(status["remaining_daily_budget_msat"], 300);
    service
        .control(ControlCommand::Policy {
            value: Some(Policy::default()),
        })
        .await?;
    assert!(
        service
            .await_authorization(&terms("off-again", 100))
            .await
            .is_err()
    );
    // Previously authorized settlement survives disabling new paid inference.
    service.pay_charge(&charge("paid", 0, 90, 100, 200)).await?;
    service.ledger.finish("paid")?;
    let status = service
        .control(ControlCommand::Policy { value: None })
        .await?;
    assert_eq!(status["spent_today_msat"], 110);
    assert_eq!(status["reserved_msat"], 0);
    assert_eq!(status["remaining_daily_budget_msat"], 0);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    Ok(())
}
