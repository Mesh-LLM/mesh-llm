//! Lexe-backed `WalletProvider`. This is the only module in the workspace
//! that depends on Lexe SDK types. The shipped binary links this module for
//! its built-in wallet plugin, which executes in a separate child process.

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

use mesh_llm_wallet::contract::WalletIdentity;
use mesh_llm_wallet::invoice::Invoice;
use mesh_llm_wallet::provider::{Balance, PayError, PaymentStatus, Transaction, WalletProvider};

pub struct LexeProvider {
    wallet: LexeWallet,
    // Lexe's local cache is not a multiprocess ledger. CLI clients should use
    // the running node's management API while it owns this lock.
    _lock: File,
}

/// Result of opening: the provider plus the identity the host pins.
pub struct Opened {
    pub provider: LexeProvider,
    pub identity: WalletIdentity,
    pub created: bool,
}

impl LexeProvider {
    pub async fn open(directory: &Path) -> Result<Opened> {
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
        let mut created = false;
        let seed = match RootSeed::read_from_path(&seed_path)
            .map_err(|_| anyhow::anyhow!("could not read wallet seed"))?
        {
            Some(seed) => seed,
            None => {
                created = true;
                let seed = RootSeed::generate();
                seed.write_to_path(&seed_path)
                    .map_err(|_| anyhow::anyhow!("could not persist wallet seed"))?;
                // Persist recovery material before any provisioning side effect.
                OpenOptions::new()
                    .write(true)
                    .open(&seed_path)?
                    .sync_all()?;
                #[cfg(unix)]
                File::open(directory)?.sync_all()?;
                seed
            }
        };
        // The seed is plaintext recovery material. The SDK creates it 0600;
        // re-assert that on every open so a seed copied or restored with looser
        // permissions is tightened rather than trusted as-is.
        restrict_seed_permissions(&seed_path)?;
        let wallet = load_wallet(&seed, directory)?;
        // Signup is idempotent, including recovery after a crash between seed
        // persistence and provisioning.
        wallet
            .signup(&seed, None)
            .await
            .map_err(|_| anyhow::anyhow!("Lexe wallet provisioning failed"))?;
        // The Lexe user public key is derived from the seed and is stable for
        // the life of the wallet, which is exactly what the host pin needs.
        let identity = WalletIdentity {
            wallet_id: wallet.user_config().user_pk.to_string(),
            provider: "lexe".into(),
            network: "mainnet".into(),
        };
        Ok(Opened {
            provider: Self {
                wallet,
                _lock: lock,
            },
            identity,
            created,
        })
    }

    fn transaction(payment: Payment) -> Transaction {
        let inbound = payment.direction == PaymentDirection::Inbound;
        let status = match payment.status {
            lexe::types::payment::PaymentStatus::Pending => PaymentStatus::Pending,
            lexe::types::payment::PaymentStatus::Completed => PaymentStatus::Succeeded,
            lexe::types::payment::PaymentStatus::Failed => PaymentStatus::Failed,
        };
        Transaction {
            id: payment.index.to_string(),
            payment_hash: payment.hash.map(|hash| hash.to_string()),
            inbound,
            amount_msat: payment.amount.map_or(0, |amount| amount.msat()),
            fee_msat: payment.fees.msat(),
            status,
            claiming: inbound
                && status == PaymentStatus::Pending
                && is_claiming_status(&payment.status_msg),
            status_msg: Some(payment.status_msg),
            created_at_ms: payment.created_at.to_millis(),
            settled_at_ms: payment.finalized_at.map(|time| time.to_millis()),
        }
    }
}

#[cfg(unix)]
fn restrict_seed_permissions(seed_path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(seed_path).context("wallet seed is unreadable")?;
    if metadata.permissions().mode() & 0o077 != 0 {
        std::fs::set_permissions(seed_path, std::fs::Permissions::from_mode(0o600))
            .context("could not restrict wallet seed permissions")?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn restrict_seed_permissions(_seed_path: &Path) -> Result<()> {
    // The wallet directory inherits the user profile ACL on Windows; there is
    // no portable mode bit to tighten.
    Ok(())
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

    async fn create_invoice(&self, amount_msat: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        ensure!(
            amount_msat != Some(0),
            "zero amount invoice is not supported"
        );
        ensure!(expiry_secs > 0, "invoice expiry must be positive");
        ensure!(
            expiry_secs <= MAX_INVOICE_EXPIRY_SECS,
            "invoice expiry exceeds the provider maximum of {MAX_INVOICE_EXPIRY_SECS}s"
        );
        let result = self
            .wallet
            .create_invoice(CreateInvoiceRequest {
                // Always explicit: `None` would silently become the SDK's
                // one-day default, and the host relies on short inference
                // invoices expiring at the payee.
                expiration_secs: Some(expiry_secs),
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
                target: "mesh_wallet_lexe::timing",
                phase,
                ms = now.duration_since(*mark).as_millis() as u64,
                total_ms = now.duration_since(started).as_millis() as u64,
                "payer phase"
            );
            *mark = now;
        };
        invoice
            .validate_payment(amount_msat, mesh_llm_wallet::now_ms())
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
                target: "mesh_wallet_lexe::timing",
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

/// Lexe rejects invoice expiries above one week
/// (`lexe_api_core::models::command::CreateInvoiceRequest::MAX_EXPIRATION_SECS`).
const MAX_INVOICE_EXPIRY_SECS: u32 =
    lexe_api_core::models::command::CreateInvoiceRequest::MAX_EXPIRATION_SECS;

/// Map Lexe's per-type display status onto the normalized `claiming` flag.
///
/// The SDK documents `status_msg` as a human-readable string produced by the
/// node, so this is the one place the wire string is interpreted. The
/// comparison is deliberately loose (case, surrounding whitespace) and a miss
/// only degrades the receiver to waiting for `completed`, which the contract
/// permits. LDK only reports claiming once the HTLC is irrevocably committed.
fn is_claiming_status(status_msg: &str) -> bool {
    status_msg.trim().eq_ignore_ascii_case("claiming")
}

#[cfg(test)]
mod tests {
    #[test]
    fn claiming_is_normalized_from_the_display_status_only_for_pending_inbound() {
        assert!(super::is_claiming_status("claiming"));
        assert!(super::is_claiming_status(" Claiming "));
        assert!(!super::is_claiming_status("invoice generated"));
        assert!(!super::is_claiming_status("completed"));
    }

    #[cfg(unix)]
    #[test]
    fn seed_permissions_are_tightened_on_open() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let seed_path = directory.path().join("seedphrase.txt");
        std::fs::write(&seed_path, "legacy seed\n").unwrap();
        std::fs::set_permissions(&seed_path, std::fs::Permissions::from_mode(0o644)).unwrap();
        super::restrict_seed_permissions(&seed_path).unwrap();
        let mode = std::fs::metadata(&seed_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

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
