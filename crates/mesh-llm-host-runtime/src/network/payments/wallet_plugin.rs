//! Plugin-backed wallet: adapts the `wallet.v1` capability onto the
//! `WalletFactory` / `WalletProvider` seams the payment ledger drives.
//!
//! Boundary rules, in order of importance:
//!
//! 1. **Money-moving calls are never retried blindly.** A transport failure
//!    during `pay` is `PayError::Uncertain`; the ledger reconciles by hash.
//!    Only a structured `not_open` *before* submission triggers a single
//!    re-open-and-retry.
//! 2. **The wallet identity is pinned.** After the first successful open the
//!    host writes `payments/wallet-provider.json`. Later opens must return the
//!    same identity or fail; outstanding ledger state is only meaningful
//!    against the wallet that created it.
//! 3. **Settlement waits carry no IPC deadline.** `wait_for_*` block on the
//!    plugin for as long as the caller is willing to wait; the caller owns
//!    cancellation by dropping the future.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use mesh_llm_wallet::contract::{
    self, CAPABILITY, Empty, LookupResponse, OpenRequest, OpenResponse, PayRequest,
    PaymentHashRequest, TransactionsRequest, WalletError, WalletErrorKind, ops,
};
use mesh_llm_wallet::invoice::Invoice;
use mesh_llm_wallet::provider::{Balance, PayError, Transaction, WalletProvider};
use mesh_llm_wallet::provisioning::{WalletFactory, WalletPin};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

use crate::plugin::PluginManager;

/// Sub-directory under the payment directory handed to the wallet plugin.
/// Kept as `lexe/` for on-disk compatibility with wallets provisioned by the
/// in-process implementation this replaces.
const WALLET_SUBDIR: &str = "lexe";

/// Where the factory finds the plugin manager. It is resolved at `open()`
/// time, not construction time: the payment service can be created during
/// startup (gossip advertises prices) before the plugin manager exists, and
/// capturing `None` then would leave the node wallet-less for its lifetime.
pub type PluginManagerSlot = Arc<Mutex<Option<PluginManager>>>;

/// Resolves the `wallet.v1` provider through the plugin manager.
pub struct PluginWalletFactory {
    plugin_manager: PluginManagerSlot,
}

impl PluginWalletFactory {
    pub fn new(plugin_manager: PluginManagerSlot) -> Self {
        Self { plugin_manager }
    }

    fn wallet_directory(payment_directory: &Path) -> PathBuf {
        payment_directory.join(WALLET_SUBDIR)
    }

    /// A wallet provisioned by the in-process implementation this replaces
    /// has a seed but no pin yet. It must still count as a wallet: paid
    /// routing and the legacy-bridge ingress guard both key off this, and
    /// reading "no wallet" after an upgrade would silently disable both.
    fn legacy_wallet_present(payment_directory: &Path) -> bool {
        Self::wallet_directory(payment_directory)
            .join("seedphrase.txt")
            .is_file()
    }

    async fn plugin_manager(&self) -> Result<PluginManager> {
        self.plugin_manager.lock().await.clone().ok_or_else(|| {
            anyhow!("plugin manager is not running yet; retry once startup completes")
        })
    }
}

#[async_trait]
impl WalletFactory for PluginWalletFactory {
    fn is_provisioned(&self, payment_directory: &Path) -> bool {
        // Side-effect free. A corrupt pin still reads as "a wallet exists":
        // `open` refuses to proceed and reports why, which is safer than
        // pretending there is nothing to protect.
        !matches!(WalletPin::load(payment_directory), Ok(None))
            || Self::legacy_wallet_present(payment_directory)
    }

    async fn open(&self, payment_directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        let plugin_manager = self.plugin_manager().await?;
        let provider = plugin_manager
            .available_provider_for_capability(CAPABILITY)
            .await?
            .ok_or_else(|| {
                anyhow!("no wallet plugin is running (capability '{CAPABILITY}' unavailable)")
            })?;
        let wallet = PluginWalletProvider {
            plugin_manager,
            plugin_name: provider.plugin_name,
            payment_directory: payment_directory.to_path_buf(),
            wallet_directory: Self::wallet_directory(payment_directory),
            open_lock: Mutex::new(()),
        };
        wallet.open_and_pin().await?;
        Ok(Arc::new(wallet))
    }
}

/// One opened wallet, bound to a specific plugin and payment directory.
pub struct PluginWalletProvider {
    plugin_manager: PluginManager,
    plugin_name: String,
    payment_directory: PathBuf,
    wallet_directory: PathBuf,
    /// Serializes re-opens. After a plugin restart every in-flight request
    /// observes `not_open` at once; only one of them should drive the open
    /// and the pin check.
    open_lock: Mutex<()>,
}

impl PluginWalletProvider {
    /// Ask the plugin to open the wallet, then verify or write the host pin.
    async fn open_and_pin(&self) -> Result<()> {
        let _guard = self.open_lock.lock().await;
        // Read the pin before contacting the plugin so a corrupt pin is
        // reported without provisioning anything.
        let pin = WalletPin::load(&self.payment_directory)?;
        let response: OpenResponse = self
            .call(
                ops::OPEN,
                &OpenRequest {
                    directory: self.wallet_directory.display().to_string(),
                },
                Some(OPEN_TIMEOUT),
            )
            .await
            .map_err(|error| anyhow!("{error}"))?;
        let identity = response.identity;
        if identity.network != "mainnet" {
            bail!(
                "wallet plugin '{}' opened a {} wallet; paid inference requires mainnet",
                self.plugin_name,
                identity.network
            );
        }
        match pin {
            Some(pin) => {
                if pin.plugin != self.plugin_name || pin.wallet_id != identity.wallet_id {
                    bail!(
                        "wallet identity mismatch: ledger is pinned to plugin '{}' wallet '{}', \
                         but plugin '{}' opened wallet '{}'. Refusing to settle against a \
                         different wallet.",
                        pin.plugin,
                        pin.wallet_id,
                        self.plugin_name,
                        identity.wallet_id
                    );
                }
            }
            None => {
                WalletPin {
                    plugin: self.plugin_name.clone(),
                    wallet_id: identity.wallet_id.clone(),
                    provider: identity.provider.clone(),
                    network: identity.network.clone(),
                }
                .store(&self.payment_directory)
                .context("persist wallet pin")?;
            }
        }
        Ok(())
    }

    /// Invoke one operation on the bound plugin and decode the result.
    ///
    /// Transport errors (plugin gone, IPC broken, timeout) surface as
    /// `WalletErrorKind::Uncertain`; structured plugin errors keep their kind.
    async fn call<Req: Serialize, Res: DeserializeOwned>(
        &self,
        operation: &str,
        request: &Req,
        timeout: Option<std::time::Duration>,
    ) -> Result<Res, WalletError> {
        let input = serde_json::to_string(request)
            .map_err(|err| WalletError::invalid(format!("encode {operation}: {err}")))?;
        let result = self
            .plugin_manager
            .invoke_operation_with_timeout(&self.plugin_name, operation, &input, timeout)
            .await
            .map_err(|err| {
                WalletError::new(
                    WalletErrorKind::Uncertain,
                    format!("wallet plugin '{}' {operation}: {err}", self.plugin_name),
                )
            })?;
        if result.is_error {
            return Err(WalletError::decode(&result.content_json));
        }
        serde_json::from_str(&result.content_json).map_err(|err| {
            WalletError::new(
                WalletErrorKind::Uncertain,
                format!("decode {operation} response: {err}"),
            )
        })
    }

    /// Run a non-payment operation, transparently re-opening the wallet once
    /// if the plugin restarted and lost its open state.
    async fn call_reopening<Req: Serialize, Res: DeserializeOwned>(
        &self,
        operation: &str,
        request: &Req,
        timeout: Option<std::time::Duration>,
    ) -> Result<Res> {
        match self.call(operation, request, timeout).await {
            Err(error) if error.kind == WalletErrorKind::NotOpen => {
                self.open_and_pin().await?;
                self.call(operation, request, timeout)
                    .await
                    .map_err(|error| anyhow!("{error}"))
            }
            other => other.map_err(|error| anyhow!("{error}")),
        }
    }
}

/// Opening may provision and contact the provider network; give it room but
/// do not wait forever on a wedged plugin.
const OPEN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Ordinary queries: a wallet lookup that takes longer than this is broken.
const QUERY_TIMEOUT: Option<std::time::Duration> = Some(std::time::Duration::from_secs(30));

#[async_trait]
impl WalletProvider for PluginWalletProvider {
    async fn balance(&self) -> Result<Balance> {
        self.call_reopening(ops::BALANCE, &Empty {}, QUERY_TIMEOUT)
            .await
    }

    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>> {
        self.call_reopening(
            ops::TRANSACTIONS,
            &TransactionsRequest { limit },
            QUERY_TIMEOUT,
        )
        .await
    }

    async fn create_invoice(&self, amount_msat: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        self.call_reopening(
            ops::CREATE_INVOICE,
            &contract::CreateInvoiceRequest {
                amount_msat,
                expiry_secs,
            },
            QUERY_TIMEOUT,
        )
        .await
    }

    async fn pay(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_total_msat: u64,
    ) -> Result<Transaction, PayError> {
        let request = PayRequest {
            invoice: invoice.clone(),
            amount_msat,
            max_total_msat,
        };
        // No IPC deadline: a slow route-find is not a failure, and a timeout
        // here could not be classified as not-submitted anyway.
        match self.call::<_, Transaction>(ops::PAY, &request, None).await {
            Ok(transaction) => Ok(transaction),
            Err(error) if error.kind == WalletErrorKind::NotOpen => {
                // `not_open` is emitted before the plugin touches the wallet, so
                // nothing was submitted. Re-open once, then retry exactly once;
                // a second `not_open` is treated as uncertain rather than looped.
                self.open_and_pin().await.map_err(PayError::NotSubmitted)?;
                self.call::<_, Transaction>(ops::PAY, &request, None)
                    .await
                    .map_err(|error| match error.kind {
                        WalletErrorKind::NotOpen => PayError::Uncertain(anyhow!("{error}")),
                        _ => error.into_pay_error(),
                    })
            }
            Err(error) => Err(error.into_pay_error()),
        }
    }

    async fn lookup(&self, payment_hash: &str) -> Result<Option<Transaction>> {
        let response: LookupResponse = self
            .call_reopening(
                ops::LOOKUP,
                &PaymentHashRequest {
                    payment_hash: payment_hash.to_owned(),
                },
                QUERY_TIMEOUT,
            )
            .await?;
        Ok(response.transaction)
    }

    async fn wait_for_payment(&self, payment_hash: &str) -> Result<Transaction> {
        self.call_reopening(
            ops::WAIT_FOR_PAYMENT,
            &PaymentHashRequest {
                payment_hash: payment_hash.to_owned(),
            },
            None,
        )
        .await
    }

    async fn wait_for_arrival(&self, payment_hash: &str) -> Result<Transaction> {
        self.call_reopening(
            ops::WAIT_FOR_ARRIVAL,
            &PaymentHashRequest {
                payment_hash: payment_hash.to_owned(),
            },
            None,
        )
        .await
    }
}

#[cfg(test)]
mod tests;
