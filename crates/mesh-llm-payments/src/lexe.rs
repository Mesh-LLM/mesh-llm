//! Only this module may depend on Lexe SDK types.

use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use fs2::FileExt;
use lexe::config::WalletEnvConfig;
use lexe::types::auth::{CredentialsRef, RootSeed};
use lexe::types::bitcoin::Amount;
use lexe::types::command::CreateInvoiceRequest;
use lexe::types::payment::{Payment, PaymentDirection, PaymentFilter};
use lexe::wallet::LexeWallet;
use lexe_api_core::def::UserNodeRunApi;
use lexe_api_core::models::command::{
    PayInvoicePreflightRequest, PayInvoiceRequest, PaymentIdStruct,
};
use lexe_api_core::types::payments::{PaymentId, PaymentKind};

use crate::invoice::Invoice;
use crate::wallet::{Balance, PayError, PaymentStatus, Transaction, WalletProvider};

pub(crate) fn is_provisioned(directory: &Path) -> bool {
    directory.join("lexe/seedphrase.txt").exists()
}

pub(crate) struct LexeProvider {
    wallet: LexeWallet,
    // Lexe's local cache is not a multiprocess ledger. CLI clients should use
    // the running node's management API while it owns this lock.
    _lock: File,
}

impl LexeProvider {
    pub(crate) async fn open(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(directory.join("wallet.lock"))?;
        lock.try_lock_exclusive()
            .context("wallet is already open; use the running node's wallet API")?;
        let seed_path = directory.join("seedphrase.txt");
        let seed = match RootSeed::read_from_path(&seed_path)
            .map_err(|_| anyhow::anyhow!("could not read wallet seed"))?
        {
            Some(seed) => seed,
            None => {
                let seed = RootSeed::generate();
                seed.write_to_path(&seed_path)
                    .map_err(|_| anyhow::anyhow!("could not persist wallet seed"))?;
                // Persist recovery material before any provisioning side effect.
                File::open(&seed_path)?.sync_all()?;
                #[cfg(unix)]
                File::open(directory)?.sync_all()?;
                seed
            }
        };
        let wallet = load_wallet(&seed, directory)?;
        // Signup is idempotent, including recovery after a crash between seed
        // persistence and provisioning.
        wallet
            .signup(&seed, None)
            .await
            .map_err(|_| anyhow::anyhow!("Lexe wallet provisioning failed"))?;
        Ok(Self {
            wallet,
            _lock: lock,
        })
    }

    fn transaction(payment: Payment) -> Transaction {
        Transaction {
            id: payment.index.to_string(),
            payment_hash: payment.hash.map(|hash| hash.to_string()),
            inbound: payment.direction == PaymentDirection::Inbound,
            amount_msat: payment.amount.map_or(0, |amount| amount.msat()),
            fee_msat: payment.fees.msat(),
            status: match payment.status {
                lexe::types::payment::PaymentStatus::Pending => PaymentStatus::Pending,
                lexe::types::payment::PaymentStatus::Completed => PaymentStatus::Succeeded,
                lexe::types::payment::PaymentStatus::Failed => PaymentStatus::Failed,
            },
            status_msg: Some(payment.status_msg),
            created_at_ms: payment.created_at.to_millis(),
            settled_at_ms: payment.finalized_at.map(|time| time.to_millis()),
        }
    }
}

fn load_wallet(seed: &RootSeed, directory: &Path) -> Result<LexeWallet> {
    // Lexe and the mesh enable different Rustls backends. Standalone wallet
    // commands do not run Nostr's TLS initialization before constructing Lexe's
    // BIP353 client. Preserve an embedding application's provider if present.
    let _ = rustls::crypto::ring::default_provider().install_default();
    LexeWallet::load_or_fresh(
        WalletEnvConfig::mainnet(),
        CredentialsRef::from(seed),
        Some(directory.to_path_buf()),
    )
    .map_err(|_| anyhow::anyhow!("could not load Lexe wallet"))
}

#[async_trait]
impl WalletProvider for LexeProvider {
    async fn balance(&self) -> Result<Balance> {
        let info = self
            .wallet
            .node_info()
            .await
            .map_err(|_| anyhow::anyhow!("Lexe balance query failed"))?;
        Ok(Balance {
            spendable_msat: info.lightning_sendable_balance.msat(),
        })
    }

    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>> {
        ensure!(
            (1..=1000).contains(&limit),
            "transaction limit must be between 1 and 1000"
        );
        self.wallet
            .sync_payments()
            .await
            .map_err(|_| anyhow::anyhow!("Lexe payment synchronization failed"))?;
        let payments = self
            .wallet
            .list_payments(&PaymentFilter::All, None, Some(limit), None)
            .map_err(|_| anyhow::anyhow!("Lexe transaction query failed"))?;
        Ok(payments
            .payments
            .into_iter()
            .map(Self::transaction)
            .collect())
    }

    async fn create_invoice(&self, amount_msat: Option<u64>) -> Result<Invoice> {
        ensure!(
            amount_msat != Some(0),
            "zero amount invoice is not supported"
        );
        let result = self
            .wallet
            .create_invoice(CreateInvoiceRequest {
                amount: amount_msat.map(Amount::from_msat),
                description: Some("mesh-llm".into()),
                ..Default::default()
            })
            .await
            .map_err(|_| anyhow::anyhow!("Lexe invoice creation failed"))?;
        Invoice::parse(&result.invoice.to_string())
    }

    async fn pay(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_total_msat: u64,
    ) -> Result<Transaction, PayError> {
        // Phase timing for the payer-side critical path. Static field names
        // and durations only; invoices and hashes are operator data.
        let started = std::time::Instant::now();
        let mut mark = started;
        let lap = |phase: &'static str, mark: &mut std::time::Instant| {
            let now = std::time::Instant::now();
            tracing::debug!(
                target: "mesh_llm::payments::timing",
                phase,
                ms = now.duration_since(*mark).as_millis() as u64,
                total_ms = now.duration_since(started).as_millis() as u64,
                "payer phase"
            );
            *mark = now;
        };
        invoice
            .validate_payment(amount_msat, crate::now_ms())
            .map_err(PayError::NotSubmitted)?;
        if let Some(existing) = self
            .lookup(&invoice.payment_hash)
            .await
            .map_err(PayError::NotSubmitted)?
        {
            if existing.inbound {
                return Err(PayError::NotSubmitted(anyhow::anyhow!(
                    "cannot pay this wallet's own invoice"
                )));
            }
            return Ok(existing);
        }
        lap("duplicate_lookup", &mut mark);
        let parsed: lexe::types::bitcoin::Invoice = invoice
            .bolt11
            .parse()
            .map_err(|_| PayError::NotSubmitted(anyhow::anyhow!("invalid invoice")))?;
        let fallback_amount = invoice
            .amount_msat
            .is_none()
            .then(|| Amount::from_msat(amount_msat));
        let route = self
            .wallet
            .node_client()
            .pay_invoice_preflight(PayInvoicePreflightRequest {
                invoice: parsed.clone(),
                fallback_amount,
                kind: PaymentKind::Invoice,
            })
            .await
            .map_err(|_| {
                PayError::NotSubmitted(anyhow::anyhow!("Lexe payment preflight failed"))
            })?;
        lap("preflight", &mut mark);
        let debit = route
            .amount
            .msat()
            .checked_add(route.fees.msat())
            .context("wallet fee overflow")
            .map_err(PayError::NotSubmitted)?;
        if debit > max_total_msat {
            return Err(PayError::NotSubmitted(anyhow::anyhow!(
                "payment including routing fees exceeds authorized amount"
            )));
        }
        // Reuse the preflighted route: the high-level SDK currently discards it.
        self.wallet
            .node_client()
            .pay_invoice(PayInvoiceRequest {
                invoice: parsed,
                fallback_amount,
                message: None,
                personal_note: None,
                kind: PaymentKind::Invoice,
                ldk_route: Some(route.ldk_route),
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!("Lexe payment outcome uncertain; reconcile by payment hash")
            })?;
        lap("submit", &mut mark);
        let submitted = self
            .lookup(&invoice.payment_hash)
            .await?
            .context("payment submitted; status not yet available")?;
        lap("submitted_lookup", &mut mark);
        Ok(submitted)
    }

    async fn lookup(&self, payment_hash: &str) -> Result<Option<Transaction>> {
        let id: PaymentId = format!("ln_{payment_hash}")
            .parse()
            .context("invalid payment hash")?;
        let result = self
            .wallet
            .node_client()
            .get_payment_by_id(PaymentIdStruct { id })
            .await
            .map_err(|_| anyhow::anyhow!("Lexe payment status query failed"))?;
        Ok(result
            .maybe_payment
            .map(Payment::from)
            .map(Self::transaction))
    }

    async fn wait_for_payment(&self, payment_hash: &str) -> Result<Transaction> {
        self.poll_until(payment_hash, |payment| {
            payment.status != PaymentStatus::Pending
        })
        .await
    }

    async fn wait_for_arrival(&self, payment_hash: &str) -> Result<Transaction> {
        self.poll_until(payment_hash, |payment| {
            payment.status != PaymentStatus::Pending || payment.is_claiming()
        })
        .await
    }
}

impl LexeProvider {
    /// Lexe exposes payment lookup rather than push notifications. Keep that
    /// transport detail behind the provider interface.
    ///
    /// This is a lookup-start cadence, not an inter-poll sleep. A warm lookup
    /// round trip is itself 171-255 ms on the reference pair, so sleeping a
    /// further fixed interval would nearly double the effective period and can
    /// miss the receiver's few-hundred-millisecond claiming state. Ticks that
    /// a slow lookup has already consumed are skipped rather than queued.
    async fn poll_until(
        &self,
        payment_hash: &str,
        settled: impl Fn(&Transaction) -> bool,
    ) -> Result<Transaction> {
        let mut cadence = tokio::time::interval(POLL_INTERVAL);
        cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            cadence.tick().await;
            let started = std::time::Instant::now();
            let payment = self.lookup(payment_hash).await?;
            tracing::debug!(
                target: "mesh_llm::payments::timing",
                phase = "lookup",
                ms = started.elapsed().as_millis() as u64,
                "wallet lookup"
            );
            if let Some(payment) = payment
                && settled(&payment)
            {
                return Ok(payment);
            }
        }
    }
}

const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn wallet_constructs_without_mesh_tls_initialization() {
        let directory = tempfile::tempdir().unwrap();
        let seed = lexe::types::auth::RootSeed::generate();
        // Construction exercises the real SDK TLS setup without signup,
        // network access, or moving funds.
        let _wallet = super::load_wallet(&seed, directory.path()).unwrap();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}
