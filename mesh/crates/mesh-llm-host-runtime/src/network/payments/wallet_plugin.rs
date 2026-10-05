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
//!    against the wallet that created it. Which plugin is opened is decided
//!    in [`selection`].
//! 3. **Settlement waits carry no IPC deadline.** `wait_for_*` block on the
//!    plugin for as long as the caller is willing to wait; the caller owns
//!    cancellation by dropping the future.
//! 4. **Fees count against the daily budget.** Each `pay` carries the fee
//!    headroom the host authorized, and whatever fee the wallet reports is
//!    recorded as spend. A wallet that cannot bound fees can exceed that
//!    headroom, and so the daily budget and the request's cap; the overrun is
//!    recorded after the fact, not prevented.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use mesh_llm_wallet::contract::{
    self, CAPABILITY, Empty, LookupResponse, OpenRequest, OpenResponse, PayRequest,
    PaymentHashRequest, TransactionsRequest, WalletError, WalletErrorKind, WalletFeatures, ops,
};
use mesh_llm_wallet::invoice::Invoice;
use mesh_llm_wallet::provider::{Balance, PayError, Transaction, WalletProvider};
use mesh_llm_wallet::provisioning::{WalletFactory, WalletPin};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

use crate::plugin::PluginManager;
use crate::plugin::operations::TransportReplay;

pub mod selection;

use selection::{possible_wallets_not_running, select_wallet_plugin};

/// Parent of every wallet plugin's data directory.
const WALLETS_SUBDIR: &str = "wallets";

/// The directory handed to `plugin` as its wallet state directory.
///
/// Each plugin gets its own, so switching wallets never hands one backend
/// another's seed or credentials. Plugin names come from operator config, so
/// anything that could escape the payments directory is refused.
fn wallet_directory(payment_directory: &Path, plugin: &str) -> Result<PathBuf> {
    let safe = !plugin.is_empty()
        && plugin != "."
        && plugin != ".."
        && plugin
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !safe {
        bail!("wallet plugin name '{plugin}' cannot be used as a directory name");
    }
    Ok(payment_directory.join(WALLETS_SUBDIR).join(plugin))
}

/// Where the factory finds the plugin manager. It is resolved at `open()`
/// time, not construction time: the payment service can be created during
/// startup (gossip advertises prices) before the plugin manager exists, and
/// capturing `None` then would leave the node wallet-less for its lifetime.
pub type PluginManagerSlot = Arc<Mutex<Option<PluginManager>>>;

/// Resolves the `wallet.v1` provider through the plugin manager.
pub struct PluginWalletFactory {
    plugin_manager: PluginManagerSlot,
    /// The wallet plugin chosen in `[payments] wallet`, if any.
    configured_wallet: Option<String>,
}

impl PluginWalletFactory {
    pub fn new(plugin_manager: PluginManagerSlot, configured_wallet: Option<String>) -> Self {
        Self {
            plugin_manager,
            configured_wallet,
        }
    }

    /// Names of the running plugins that serve `wallet.v1`.
    async fn running_wallet_plugins(plugin_manager: &PluginManager) -> Result<Vec<String>> {
        let mut names: Vec<String> = plugin_manager
            .capability_providers()
            .await?
            .into_iter()
            .filter(|provider| provider.capability == CAPABILITY && provider.available)
            .map(|provider| provider.plugin_name)
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
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
    }

    async fn open(&self, payment_directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        let plugin_manager = self.plugin_manager().await?;
        // Read the pin before contacting any plugin so a corrupt pin is
        // reported without provisioning anything.
        let pin = WalletPin::load(payment_directory)?;
        let running = Self::running_wallet_plugins(&plugin_manager).await?;
        let down = possible_wallets_not_running(&plugin_manager.list().await);
        let plugin_name = select_wallet_plugin(
            &running,
            &down,
            pin.as_ref(),
            self.configured_wallet.as_deref(),
        )?;
        let wallet = PluginWalletProvider {
            plugin_manager,
            wallet_directory: wallet_directory(payment_directory, &plugin_name)?,
            plugin_name,
            payment_directory: payment_directory.to_path_buf(),
            features: std::sync::Mutex::new(WalletFeatures::default()),
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
    /// What the plugin reported it supports at the most recent open.
    features: std::sync::Mutex<WalletFeatures>,
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
                         different wallet; `mesh-llm wallet unpin` switches wallets once no \
                         payment is outstanding.",
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
        *self
            .features
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = response.features;
        Ok(())
    }

    fn features(&self) -> WalletFeatures {
        self.features
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
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
            .invoke_operation_with_replay(
                &self.plugin_name,
                operation,
                &input,
                timeout,
                if operation == ops::PAY {
                    TransportReplay::Never
                } else {
                    TransportReplay::Allow
                },
            )
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
        if amount_msat.is_none() && !self.features().amountless_invoices {
            bail!(
                "wallet plugin '{}' cannot create amount-less invoices; specify an amount",
                self.plugin_name
            );
        }
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
                // A structured `not_open` from this unreplayed attempt means no
                // submission occurred. This does not hold across lost/replayed
                // attempts. Re-open once, then retry exactly once;
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
