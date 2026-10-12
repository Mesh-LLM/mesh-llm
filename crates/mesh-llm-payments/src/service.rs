use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

pub mod metered;

use crate::invoice::Invoice;
use crate::ledger::{ApprovalMode, Charge, Ledger, RequestTerms};
use crate::wallet::{Balance, PayError, PaymentStatus, Transaction, WalletProvider};

/// Owns wallet I/O and durable authorization for a single data directory.
/// Bound on the one-shot lookup after an incoming-payment wait times out.
const FINAL_LOOKUP_TIMEOUT: Duration = Duration::from_secs(2);

pub struct PaymentService {
    pub ledger: Ledger,
    directory: PathBuf,
    wallet: OnceCell<Arc<dyn WalletProvider>>,
    factory: Arc<dyn crate::provisioning::WalletFactory>,
    payment_lock: Mutex<()>,
    receivable_lock: Mutex<()>,
    metered_standing: Mutex<std::collections::HashSet<String>>,
    _process_lock: std::fs::File,
}

impl PaymentService {
    /// Ledger-only service. Wallet operations fail until a factory is injected
    /// with [`Self::with_factory`] or a provider with [`Self::with_provider`].
    pub fn open(directory: &Path) -> Result<Self> {
        Self::with_factory(directory, Arc::new(crate::provisioning::NoWalletFactory))
    }

    pub fn with_factory(
        directory: &Path,
        factory: Arc<dyn crate::provisioning::WalletFactory>,
    ) -> Result<Self> {
        let ledger = Ledger::open(directory)?;
        let process_lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("service.lock"))?;
        fs2::FileExt::try_lock_exclusive(&process_lock)
            .context("payment service is already running; use its local API")?;
        ledger.close_interrupted_metered()?;
        ledger.close_abandoned_approvals()?;
        ledger.fail_unsubmitted_input_charges()?;
        ledger.finalize_terminal_requests()?;
        Ok(Self {
            ledger,
            directory: directory.to_owned(),
            wallet: OnceCell::new(),
            factory,
            payment_lock: Mutex::new(()),
            receivable_lock: Mutex::new(()),
            metered_standing: Mutex::new(Default::default()),
            _process_lock: process_lock,
        })
    }

    pub fn with_provider(directory: &Path, wallet: Arc<dyn WalletProvider>) -> Result<Self> {
        let mut service = Self::open(directory)?;
        service.wallet = OnceCell::new_with(Some(wallet));
        Ok(service)
    }

    pub fn has_wallet(&self) -> bool {
        self.wallet.get().is_some() || self.factory.is_provisioned(&self.directory)
    }

    pub async fn wallet(&self) -> Result<&Arc<dyn WalletProvider>> {
        self.wallet
            .get_or_try_init(|| self.factory.open(&self.directory))
            .await
    }

    pub async fn approve(&self, id: &str) -> Result<()> {
        let balance = self.wallet().await?.balance().await?;
        self.ledger
            .approve(id, balance.spendable_msat, crate::now_ms())
    }

    /// Read the balance in the background, so its latency, including waking
    /// a sleeping wallet, overlaps other work.
    pub fn prefetch_balance(self: &Arc<Self>) -> JoinHandle<Result<Balance>> {
        let service = self.clone();
        tokio::spawn(async move { service.wallet().await?.balance().await })
    }

    pub async fn await_authorization(&self, terms: &RequestTerms) -> Result<()> {
        self.await_authorization_with(terms, async { self.wallet().await?.balance().await })
            .await
    }

    /// [`Self::await_authorization`] against a balance the caller may have
    /// started reading before the terms arrived.
    pub async fn await_authorization_with(
        &self,
        terms: &RequestTerms,
        balance: impl Future<Output = Result<Balance>>,
    ) -> Result<()> {
        ensure!(
            self.ledger.policy()?.mode == ApprovalMode::Automatic,
            "paid inference is disabled by free-only policy"
        );
        self.ledger.propose(terms)?;
        let balance = balance.await?;
        self.ledger
            .approve(&terms.id, balance.spendable_msat, crate::now_ms())
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

    /// A charge left by the retired per-request payment protocol. It is never
    /// submitted or resubmitted; only wallet facts settle it. A charge the
    /// wallet never recorded is failed once its invoice expires (nothing can
    /// pay it after that) or immediately if it was never submitted.
    async fn observe_retired_charge(&self, charge: &Charge) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        let hash = &charge.invoice.payment_hash;
        let state = self
            .ledger
            .charge_state(hash)?
            .context("payment intent missing")?;
        if state == "prepared" {
            return self.ledger.fail_unsubmitted(hash);
        }
        let wallet = self.wallet().await?;
        if let Some(payment) = wallet.lookup(hash).await? {
            validate_payment_update(&payment, hash, false)?;
            return self.ledger.reconcile(&payment, crate::now_ms());
        }
        // No wallet record is not proof of failure, even after invoice expiry:
        // keep the charge and its reservation pending for a later lookup.
        Ok(())
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
        // A prepared charge was never submitted, so skip the lookup. If the
        // invoice was paid outside this service, `pay` recovers that payment.
        if state != "prepared" {
            if let Some(payment) = wallet.lookup(&charge.invoice.payment_hash).await? {
                validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
                return Ok(payment);
            }
            // Nothing can settle an expired invoice, so an attempt the wallet
            // never recorded paid nothing: release the reservation.
            if charge.invoice.expires_at_ms <= crate::now_ms() {
                self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
                anyhow::bail!("Lightning payment failed; invoice expired unpaid");
            }
            // Providers treat a repeat `pay` for the same invoice as a lookup
            // of the existing payment (Lexe SDK >= 0.1.24), so resubmitting
            // through the normal preflight/fee-checked path cannot double-pay.
            return match self.submit(wallet, charge).await {
                // An earlier submission may still land; only expiry proves
                // nothing was paid, so a rejection here stays pending.
                Err(PayError::NotSubmitted(error)) => Err(error.context(
                    "payment resubmission rejected; awaiting authoritative wallet status",
                )),
                result => result.map_err(Into::into),
            };
        }
        if let Err(error) = charge
            .invoice
            .validate_payment(charge.amount_msat, crate::now_ms())
        {
            self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
            return Err(error);
        }
        self.ledger.begin_submission(&charge.invoice.payment_hash)?;
        match self.submit(wallet, charge).await {
            Err(PayError::NotSubmitted(error)) => {
                self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
                Err(error.context("payment was not submitted"))
            }
            result => result.map_err(Into::into),
        }
    }

    async fn submit(
        &self,
        wallet: &Arc<dyn WalletProvider>,
        charge: &Charge,
    ) -> Result<Transaction, PayError> {
        let payment = wallet
            .pay(&charge.invoice, charge.amount_msat, charge.max_total_msat)
            .await?;
        validate_payment_update(&payment, &charge.invoice.payment_hash, false)
            .map_err(PayError::Uncertain)?;
        Ok(payment)
    }

    /// Recover each charge independently. A pending charge the wallet has no
    /// record of is resubmitted (idempotent per payment hash) until its
    /// invoice expires, then failed and its reservation released.
    pub async fn reconcile_pending(&self) -> Result<()> {
        self.ledger.finalize_terminal_requests()?;
        let mut first_error = None;
        for charge in self.ledger.pending_charges()? {
            let result = if self.metered_authorization(&charge.request_id)?.is_some() {
                self.metered_reconcile_charge(&charge).await
            } else if self.ledger.is_wallet_send(&charge.request_id)? {
                match self.start_or_observe_charge(&charge).await {
                    Ok(payment) => self.ledger.reconcile(&payment, crate::now_ms()),
                    Err(error) => Err(error),
                }
            } else {
                self.observe_retired_charge(&charge).await
            };
            if let Err(error) = result {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub async fn wait_received(&self, invoice: &Invoice) -> Result<()> {
        let payment = self.await_incoming(invoice).await?;
        ensure!(
            payment.status == PaymentStatus::Succeeded,
            "incoming payment did not succeed"
        );
        Ok(())
    }

    async fn await_incoming(&self, invoice: &Invoice) -> Result<Transaction> {
        let wallet = self.wallet().await?;
        let until_expiry = invoice.expires_at_ms.saturating_sub(crate::now_ms());
        let remaining = until_expiry;
        let timed_out = "payment invoice expired";
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
            let waiting = wallet.wait_for_payment(&invoice.payment_hash);
            match tokio::time::timeout(Duration::from_millis(remaining), waiting).await {
                Ok(payment) => payment?,
                // The watcher may lag the wallet: look once more so a payment
                // that completed before the deadline is not reported missing.
                // Bounded so it resolves well inside the gate's pause slack.
                Err(_) => {
                    tokio::time::timeout(FINAL_LOOKUP_TIMEOUT, wallet.lookup(&invoice.payment_hash))
                        .await
                        .context(timed_out)??
                        .filter(|payment| payment.status == PaymentStatus::Succeeded)
                        .context(timed_out)?
                }
            }
        };
        validate_payment_update(&payment, &invoice.payment_hash, true)?;
        ensure!(
            payment.status != PaymentStatus::Failed,
            "incoming payment did not succeed"
        );
        Ok(payment)
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
