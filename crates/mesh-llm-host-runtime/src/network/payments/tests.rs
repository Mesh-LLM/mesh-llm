use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
use mesh_llm_payments::{
    invoice::Invoice,
    ledger::{ApprovalMode, Policy},
    pricing::Pricing,
    service::PaymentService,
    wallet::{Balance, PayError, PaymentStatus, Transaction, WalletProvider},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::{Node, NodeRole},
};

mod forwarding;
mod metered;
mod metered_acceptance;
mod metered_live;

#[derive(Default)]
struct Network {
    entries: Mutex<HashMap<String, (u8, Transaction)>>,
    next: AtomicUsize,
    payments: AtomicUsize,
    invoice_expiry_seconds: AtomicUsize,
    updates: tokio::sync::Notify,
    hold_payments: AtomicBool,
    payment_release: tokio::sync::Notify,
}

struct TestWallet {
    owner: u8,
    network: Arc<Network>,
}

#[async_trait]
impl WalletProvider for TestWallet {
    async fn balance(&self) -> Result<Balance> {
        Ok(Balance {
            spendable_msat: 100_000,
        })
    }
    async fn transactions(&self, _: usize) -> Result<Vec<Transaction>> {
        Ok(Vec::new())
    }
    async fn create_invoice(&self, amount: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        let number = self.network.next.fetch_add(1, Ordering::SeqCst) + 1;
        let key = SecretKey::from_slice(&[7; 32])?;
        // The simulated wallet honours the host's expiry unless a test forces
        // a shorter one to exercise the expiry paths quickly.
        let expiry = match self.network.invoice_expiry_seconds.load(Ordering::SeqCst) {
            0 => u64::from(expiry_secs),
            seconds => seconds as u64,
        };
        let invoice = InvoiceBuilder::new(Currency::Bitcoin)
            .description("simulated wallet".into())
            .payment_hash(PaymentHash([number as u8; 32]))
            .payment_secret(PaymentSecret([42; 32]))
            .current_timestamp()
            .expiry_time(Duration::from_secs(expiry))
            .min_final_cltv_expiry_delta(144)
            .amount_milli_satoshis(amount.unwrap_or(1))
            .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &key))?;
        let invoice = Invoice::parse(&invoice.to_string())?;
        self.network.entries.lock().unwrap().insert(
            invoice.payment_hash.clone(),
            (
                self.owner,
                Transaction {
                    id: invoice.payment_hash.clone(),
                    payment_hash: Some(invoice.payment_hash.clone()),
                    inbound: true,
                    amount_msat: amount.unwrap_or(1),
                    fee_msat: 10,
                    status: PaymentStatus::Pending,
                    claiming: false,
                    status_msg: None,
                    created_at_ms: mesh_llm_payments::now_ms(),
                    settled_at_ms: None,
                },
            ),
        );
        Ok(invoice)
    }
    async fn lookup(&self, hash: &str) -> Result<Option<Transaction>> {
        Ok(self
            .network
            .entries
            .lock()
            .unwrap()
            .get(hash)
            .and_then(|(owner, payment)| {
                if *owner != self.owner && payment.status == PaymentStatus::Pending {
                    return None;
                }
                let mut payment = payment.clone();
                payment.inbound = *owner == self.owner;
                Some(payment)
            }))
    }
    async fn pay(&self, invoice: &Invoice, amount: u64, cap: u64) -> Result<Transaction, PayError> {
        invoice.validate_payment(amount, mesh_llm_payments::now_ms())?;
        assert!(amount + 10 <= cap);
        while self.network.hold_payments.load(Ordering::SeqCst) {
            let released = self.network.payment_release.notified();
            tokio::pin!(released);
            if !self.network.hold_payments.load(Ordering::SeqCst) {
                break;
            }
            released.await;
        }
        {
            let mut entries = self.network.entries.lock().unwrap();
            let (owner, payment) = entries.get_mut(&invoice.payment_hash).unwrap();
            assert_ne!(*owner, self.owner);
            assert_eq!(payment.status, PaymentStatus::Pending);
            payment.status = PaymentStatus::Succeeded;
            payment.settled_at_ms = Some(mesh_llm_payments::now_ms());
        }
        self.network.payments.fetch_add(1, Ordering::SeqCst);
        self.network.updates.notify_waiters();
        Ok(self.lookup(&invoice.payment_hash).await?.unwrap())
    }

    async fn wait_for_payment(&self, hash: &str) -> Result<Transaction> {
        loop {
            let changed = self.network.updates.notified();
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
}

fn allow_paid(service: &PaymentService) -> Result<()> {
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })
}
