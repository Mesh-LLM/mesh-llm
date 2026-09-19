use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use tokio::sync::{Mutex, OnceCell};

use crate::invoice::Invoice;
use crate::ledger::{ApprovalMode, Charge, Ledger, RequestTerms};
use crate::wallet::{PayError, PaymentStatus, Transaction, WalletProvider};

/// Owns wallet I/O and durable authorization for a single data directory.
pub struct PaymentService {
    pub ledger: Ledger,
    directory: PathBuf,
    wallet: OnceCell<Arc<dyn WalletProvider>>,
    payment_lock: Mutex<()>,
    receivable_lock: Mutex<()>,
    _process_lock: std::fs::File,
}

impl PaymentService {
    pub fn open(directory: &Path) -> Result<Self> {
        let ledger = Ledger::open(directory)?;
        let process_lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("service.lock"))?;
        fs2::FileExt::try_lock_exclusive(&process_lock)
            .context("payment service is already running; use its local API")?;
        ledger.close_interrupted_serving()?;
        ledger.finalize_terminal_requests()?;
        Ok(Self {
            ledger,
            directory: directory.to_owned(),
            wallet: OnceCell::new(),
            payment_lock: Mutex::new(()),
            receivable_lock: Mutex::new(()),
            _process_lock: process_lock,
        })
    }

    pub fn with_provider(directory: &Path, wallet: Arc<dyn WalletProvider>) -> Result<Self> {
        let mut service = Self::open(directory)?;
        service.wallet = OnceCell::new_with(Some(wallet));
        Ok(service)
    }

    pub fn has_wallet(&self) -> bool {
        self.wallet.get().is_some() || self.directory.join("lexe/seedphrase.txt").exists()
    }

    pub async fn wallet(&self) -> Result<&Arc<dyn WalletProvider>> {
        self.wallet
            .get_or_try_init(|| async {
                #[cfg(feature = "lexe")]
                {
                    crate::open_wallet(&self.directory.join("lexe")).await
                }
                #[cfg(not(feature = "lexe"))]
                {
                    let _ = &self.directory;
                    bail!("no wallet provider configured")
                }
            })
            .await
    }

    pub async fn approve(&self, id: &str) -> Result<()> {
        let balance = self.wallet().await?.balance().await?;
        self.ledger
            .approve(id, balance.spendable_msat, crate::now_ms())
    }

    pub async fn await_authorization(&self, terms: &RequestTerms) -> Result<()> {
        self.ledger.propose(terms)?;
        if self.ledger.policy()?.mode == ApprovalMode::Automatic {
            self.approve(&terms.id).await?;
        }
        loop {
            ensure!(
                crate::now_ms() < terms.expires_at_ms,
                "input invoice expired while awaiting approval"
            );
            let state = self
                .ledger
                .request_state(&terms.id)?
                .context("payment request disappeared")?;
            match state.as_str() {
                "approved" => return Ok(()),
                "pending" => tokio::time::sleep(Duration::from_millis(250)).await,
                _ => bail!("payment request was rejected or closed"),
            }
        }
    }

    pub async fn pay_charge(&self, charge: &Charge) -> Result<Transaction> {
        if let Err(error) = self.ledger.prepare_charge(charge) {
            self.ledger.fail_authorization_if_idle(&charge.request_id)?;
            return Err(error);
        }
        self.settle_charge(charge).await
    }

    async fn settle_charge(&self, charge: &Charge) -> Result<Transaction> {
        let wallet = self.wallet().await?;
        let mut payment = self.start_or_observe_charge(charge).await?;
        validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
        if payment.status == PaymentStatus::Pending {
            // The provider owns notification transport. Keep the reservation if
            // observation is interrupted; outgoing HTLCs may outlive the invoice.
            payment = wallet
                .wait_for_payment(&charge.invoice.payment_hash)
                .await?;
        }
        validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
        ensure!(
            payment.status != PaymentStatus::Pending,
            "wallet returned a nonterminal payment update"
        );
        self.ledger.reconcile(&payment, crate::now_ms())?;
        ensure!(
            payment.status == PaymentStatus::Succeeded,
            "Lightning payment failed"
        );
        Ok(payment)
    }

    async fn start_or_observe_charge(&self, charge: &Charge) -> Result<Transaction> {
        let _guard = self.payment_lock.lock().await;
        let state = self
            .ledger
            .charge_state(&charge.invoice.payment_hash)?
            .context("payment intent missing")?;
        ensure!(
            state != "failed",
            "Lightning payment failed; authorization closed"
        );
        let wallet = self.wallet().await?;
        if let Some(payment) = wallet.lookup(&charge.invoice.payment_hash).await? {
            validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
            return Ok(payment);
        }
        ensure!(
            state == "prepared",
            "payment outcome uncertain; awaiting authoritative wallet status"
        );
        if let Err(error) = charge
            .invoice
            .validate_payment(charge.amount_msat, crate::now_ms())
        {
            self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
            return Err(error);
        }
        self.ledger.begin_submission(&charge.invoice.payment_hash)?;
        match wallet
            .pay(&charge.invoice, charge.amount_msat, charge.max_total_msat)
            .await
        {
            Ok(payment) => {
                validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
                Ok(payment)
            }
            Err(PayError::NotSubmitted(error)) => {
                self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
                Err(error.context("payment was not submitted"))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Recover each charge independently. Never resubmit an uncertain attempt,
    /// even if the wallet has not indexed it yet or the invoice has expired.
    pub async fn reconcile_pending(&self) -> Result<()> {
        self.ledger.finalize_terminal_requests()?;
        let mut first_error = None;
        for charge in self.ledger.pending_charges()? {
            let result = match self.start_or_observe_charge(&charge).await {
                Ok(payment) => self.ledger.reconcile(&payment, crate::now_ms()),
                Err(error) => Err(error),
            };
            if let Err(error) = result {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub async fn recover_output_debt(&self) -> Result<()> {
        let mut first_error = None;
        for id in self.ledger.uninvoiced_output()? {
            if let Err(error) = self.output_receivable(&id).await {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub async fn output_receivable(
        &self,
        id: &str,
    ) -> Result<Option<crate::ledger::receivables::Receivable>> {
        let _guard = self.receivable_lock.lock().await;
        if let Some(receipt) = self
            .ledger
            .receivables(Some(id))?
            .into_iter()
            .find(|r| r.segment == 1)
        {
            return Ok(Some(receipt));
        }
        let (peer, pricing, tokens, finished) = self.ledger.serving_account(id)?;
        ensure!(finished, "generation is still active");
        if tokens == 0 {
            return Ok(None);
        }
        let invoice = self
            .wallet()
            .await?
            .create_invoice(Some(pricing.output_charge(tokens)?))
            .await?;
        let receipt = crate::ledger::receivables::Receivable {
            request_id: id.into(),
            peer,
            segment: 1,
            invoice,
            tokens,
            paid: false,
        };
        self.ledger.record_receivable(&receipt)?;
        Ok(Some(receipt))
    }

    pub async fn wait_received(&self, invoice: &Invoice) -> Result<()> {
        let wallet = self.wallet().await?;
        let remaining = invoice.expires_at_ms.saturating_sub(crate::now_ms());
        let payment = if remaining == 0 {
            // Expiry ends an unpaid wait, but must not hide an existing receipt.
            let payment = wallet
                .lookup(&invoice.payment_hash)
                .await?
                .context("payment invoice expired")?;
            ensure!(
                payment.status == PaymentStatus::Succeeded,
                "payment invoice expired"
            );
            payment
        } else {
            tokio::time::timeout(
                Duration::from_millis(remaining),
                wallet.wait_for_payment(&invoice.payment_hash),
            )
            .await
            .context("payment invoice expired")??
        };
        validate_payment_update(&payment, &invoice.payment_hash, true)?;
        ensure!(
            payment.status == PaymentStatus::Succeeded,
            "incoming payment did not succeed"
        );
        Ok(())
    }
}

fn validate_payment_update(payment: &Transaction, hash: &str, inbound: bool) -> Result<()> {
    ensure!(
        payment.payment_hash.as_deref() == Some(hash),
        "wallet returned a different payment hash"
    );
    ensure!(payment.inbound == inbound, "payment has wrong direction");
    Ok(())
}
