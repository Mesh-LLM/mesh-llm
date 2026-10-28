//! Wallet semantics over an operator-fixed host binding; never replay payments.
use anyhow::{Result, ensure};
use async_trait::async_trait;
use mesh_llm_plugin::{
    PluginContext,
    operation_binding::{self, BoundOperationRequest, BoundOperationResponse},
};
use mesh_llm_wallet::{
    contract::*,
    invoice::Invoice,
    provider::*,
    provisioning::{WalletFactory, WalletPin},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc};

pub struct BoundWalletFactory {
    pub context: PluginContext<'static>,
    pub binding: String,
    pub allow_provisioning: bool,
}

#[async_trait]
impl WalletFactory for BoundWalletFactory {
    fn is_provisioned(&self, directory: &Path) -> bool {
        matches!(WalletPin::load(directory), Ok(Some(_)))
    }
    async fn open(&self, directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        let pinned = WalletPin::load(directory)?;
        ensure!(
            pinned.is_some() || self.allow_provisioning,
            "first wallet adoption requires explicit approval"
        );
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(90);
        let response = loop {
            let response: BoundOperationResponse = self
                .context
                .request_host(
                    operation_binding::METHOD,
                    BoundOperationRequest {
                        binding: self.binding.clone(),
                        handle: None,
                        operation: String::new(),
                        input_json: String::new(),
                    },
                )
                .await?;
            if matches!(response, BoundOperationResponse::NotDispatched { .. })
                && tokio::time::Instant::now() < deadline
            {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
            break response;
        };
        let (handle, output) = decode(response)?;
        let opened: OpenResponse = serde_json::from_str(&output)?;
        ensure!(
            opened.identity.network == "mainnet",
            "wallet is not mainnet"
        );
        ensure!(
            pinned
                .as_ref()
                .is_none_or(|old| old.wallet_id == opened.identity.wallet_id
                    && old.provider == opened.identity.provider
                    && old.network == opened.identity.network),
            "wallet identity changed"
        );
        WalletPin {
            plugin: self.binding.clone(),
            wallet_id: opened.identity.wallet_id,
            provider: opened.identity.provider,
            network: opened.identity.network,
        }
        .store(directory)?;
        Ok(Arc::new(BoundWallet {
            context: self.context.clone(),
            binding: self.binding.clone(),
            handle,
        }))
    }
}
struct BoundWallet {
    context: PluginContext<'static>,
    binding: String,
    handle: String,
}
#[derive(Debug)]
struct NotDispatched(String);
impl std::fmt::Display for NotDispatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for NotDispatched {}
fn decode(response: BoundOperationResponse) -> Result<(String, String)> {
    match response {
        BoundOperationResponse::Responded {
            handle,
            output_json,
            is_error,
        } => {
            if is_error {
                return Err(anyhow::Error::new(WalletCallError(WalletError::decode(
                    &output_json,
                ))));
            }
            Ok((handle, output_json))
        }
        BoundOperationResponse::NotDispatched { reason } => Err(NotDispatched(reason).into()),
        BoundOperationResponse::Uncertain { reason } => {
            anyhow::bail!("uncertain wallet operation: {reason}")
        }
    }
}
impl BoundWallet {
    async fn call<T: Serialize, R: DeserializeOwned>(
        &self,
        operation: &str,
        input: &T,
    ) -> Result<R> {
        loop {
            let response: BoundOperationResponse = self
                .context
                .request_host(
                    operation_binding::METHOD,
                    BoundOperationRequest {
                        binding: self.binding.clone(),
                        handle: Some(self.handle.clone()),
                        operation: operation.into(),
                        input_json: serde_json::to_string(input)?,
                    },
                )
                .await?;
            // Only long-poll observations are reissued. Never retry money-moving calls.
            if matches!(response, BoundOperationResponse::Uncertain { .. })
                && matches!(operation, ops::WAIT_FOR_PAYMENT | ops::WAIT_FOR_ARRIVAL)
            {
                continue;
            }
            let (_, output) = decode(response)?;
            return Ok(serde_json::from_str(&output)?);
        }
    }
}
#[derive(Debug)]
struct WalletCallError(WalletError);
impl std::fmt::Display for WalletCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.message)
    }
}
impl std::error::Error for WalletCallError {}

#[async_trait]
impl WalletProvider for BoundWallet {
    async fn balance(&self) -> Result<Balance> {
        self.call(ops::BALANCE, &Empty {}).await
    }
    async fn transactions(&self, limit: usize) -> Result<Vec<Transaction>> {
        self.call(ops::TRANSACTIONS, &TransactionsRequest { limit })
            .await
    }
    async fn create_invoice(&self, amount_msat: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        self.call(
            ops::CREATE_INVOICE,
            &CreateInvoiceRequest {
                amount_msat,
                expiry_secs,
            },
        )
        .await
    }
    async fn pay(
        &self,
        invoice: &Invoice,
        amount_msat: u64,
        max_total_msat: u64,
    ) -> Result<Transaction, PayError> {
        self.call(
            ops::PAY,
            &PayRequest {
                invoice: invoice.clone(),
                amount_msat,
                max_total_msat,
            },
        )
        .await
        .map_err(|error| {
            if let Some(structured) = error.downcast_ref::<WalletCallError>() {
                structured.0.clone().into_pay_error()
            } else if error.is::<NotDispatched>() {
                PayError::NotSubmitted(error)
            } else {
                PayError::Uncertain(error)
            }
        })
    }
    async fn lookup(&self, payment_hash: &str) -> Result<Option<Transaction>> {
        let reply: LookupResponse = self
            .call(
                ops::LOOKUP,
                &PaymentHashRequest {
                    payment_hash: payment_hash.into(),
                },
            )
            .await?;
        Ok(reply.transaction)
    }
    async fn wait_for_payment(&self, payment_hash: &str) -> Result<Transaction> {
        self.call(
            ops::WAIT_FOR_PAYMENT,
            &PaymentHashRequest {
                payment_hash: payment_hash.into(),
            },
        )
        .await
    }
    async fn wait_for_arrival(&self, payment_hash: &str) -> Result<Transaction> {
        self.call(
            ops::WAIT_FOR_ARRIVAL,
            &PaymentHashRequest {
                payment_hash: payment_hash.into(),
            },
        )
        .await
    }
}

/// Defers access to the host connection until initialization has completed.
pub(crate) struct InitializedWalletFactory {
    pub context: Arc<std::sync::OnceLock<PluginContext<'static>>>,
    pub binding: String,
    pub allow_provisioning: bool,
}

#[async_trait]
impl WalletFactory for InitializedWalletFactory {
    fn is_provisioned(&self, directory: &Path) -> bool {
        matches!(WalletPin::load(directory), Ok(Some(_)))
    }

    async fn open(&self, directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        BoundWalletFactory {
            context: self
                .context
                .get()
                .ok_or_else(|| anyhow::anyhow!("host connection not initialized"))?
                .owned(),
            binding: self.binding.clone(),
            allow_provisioning: self.allow_provisioning,
        }
        .open(directory)
        .await
    }
}
