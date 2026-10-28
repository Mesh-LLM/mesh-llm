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
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

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
        let transport: Arc<dyn BindingTransport> = Arc::new(self.context.clone());
        BoundWallet::open(transport, &self.binding, directory, self.allow_provisioning)
            .await
            .map(|wallet| Arc::new(wallet) as Arc<dyn WalletProvider>)
    }
}

/// One round trip to the host's operation-binding method.
#[async_trait]
trait BindingTransport: Send + Sync {
    async fn bound_operation(
        &self,
        request: BoundOperationRequest,
    ) -> Result<BoundOperationResponse>;
}

#[async_trait]
impl BindingTransport for PluginContext<'static> {
    async fn bound_operation(
        &self,
        request: BoundOperationRequest,
    ) -> Result<BoundOperationResponse> {
        self.request_host(operation_binding::METHOD, request).await
    }
}

/// Run the binding's setup operation, then verify or write the wallet pin.
/// Returns the handle the host issued for the provider's current process.
async fn open_handle(
    transport: &dyn BindingTransport,
    binding: &str,
    directory: &Path,
    allow_provisioning: bool,
) -> Result<String> {
    let pinned = WalletPin::load(directory)?;
    ensure!(
        pinned.is_some() || allow_provisioning,
        "first wallet adoption requires explicit approval"
    );
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(90);
    let response = loop {
        let response = transport
            .bound_operation(BoundOperationRequest {
                binding: binding.to_owned(),
                handle: None,
                operation: String::new(),
                input_json: String::new(),
            })
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
        plugin: binding.to_owned(),
        wallet_id: opened.identity.wallet_id,
        provider: opened.identity.provider,
        network: opened.identity.network,
    }
    .store(directory)?;
    Ok(handle)
}

/// Host refusals that mean the provider must be set up again: it restarted
/// (its generation changed) or has not finished starting. The host refuses
/// these before dispatching anything, so setting up again and retrying the
/// operation cannot repeat work on the provider.
fn setup_required(reason: &str) -> bool {
    matches!(
        reason,
        "handle requires setup and identity revalidation" | "provider not initialized"
    )
}

struct BoundWallet {
    transport: Arc<dyn BindingTransport>,
    binding: String,
    directory: PathBuf,
    /// Handle for the provider process the last setup bound to.
    handle: std::sync::Mutex<String>,
    /// Serializes setups. After a provider restart every in-flight call is
    /// refused at once; only one of them should run the setup and pin check.
    open_lock: tokio::sync::Mutex<()>,
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
    async fn open(
        transport: Arc<dyn BindingTransport>,
        binding: &str,
        directory: &Path,
        allow_provisioning: bool,
    ) -> Result<Self> {
        let handle = open_handle(&*transport, binding, directory, allow_provisioning).await?;
        Ok(Self {
            transport,
            binding: binding.to_owned(),
            directory: directory.to_owned(),
            handle: std::sync::Mutex::new(handle),
            open_lock: tokio::sync::Mutex::new(()),
        })
    }
    fn current_handle(&self) -> String {
        self.handle.lock().unwrap().clone()
    }
    /// Set up the restarted provider again and re-verify the pinned identity,
    /// unless another call already replaced `stale` while this one waited.
    async fn reopen(&self, stale: &str) -> Result<()> {
        let _guard = self.open_lock.lock().await;
        if self.current_handle() != stale {
            return Ok(());
        }
        let handle = open_handle(&*self.transport, &self.binding, &self.directory, false).await?;
        *self.handle.lock().unwrap() = handle;
        Ok(())
    }
    async fn call<T: Serialize, R: DeserializeOwned>(
        &self,
        operation: &str,
        input: &T,
    ) -> Result<R> {
        let input_json = serde_json::to_string(input)?;
        let mut reopened = false;
        loop {
            let handle = self.current_handle();
            let response = self
                .transport
                .bound_operation(BoundOperationRequest {
                    binding: self.binding.clone(),
                    handle: Some(handle.clone()),
                    operation: operation.into(),
                    input_json: input_json.clone(),
                })
                .await?;
            match response {
                // Only long-poll observations are reissued. Never retry money-moving calls.
                BoundOperationResponse::Uncertain { .. }
                    if matches!(operation, ops::WAIT_FOR_PAYMENT | ops::WAIT_FOR_ARRIVAL) =>
                {
                    continue;
                }
                // The provider restarted and the host refused before dispatching
                // anything, so nothing reached it: set up once more and retry the
                // same operation, `pay` included. A second refusal is reported.
                BoundOperationResponse::NotDispatched { reason }
                    if !reopened && setup_required(&reason) =>
                {
                    reopened = true;
                    if let Err(error) = self.reopen(&handle).await {
                        return Err(NotDispatched(format!("wallet setup failed: {error:#}")).into());
                    }
                }
                response => {
                    let (_, output) = decode(response)?;
                    return Ok(serde_json::from_str(&output)?);
                }
            }
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

#[cfg(test)]
mod tests {
    //! The fake host mirrors `operation_bindings.rs`: a handle is valid only
    //! for the provider generation its setup bound to, and a refused call is
    //! never dispatched to the provider.
    use super::*;
    use mesh_llm_wallet::provider::{PaymentStatus, Transaction};
    use serde_json::json;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };

    #[derive(Default)]
    struct FakeHost {
        provider_generation: AtomicU64,
        bound: Mutex<Option<(String, u64)>>,
        setups: AtomicU64,
        dispatched: Mutex<Vec<String>>,
        /// Setup responses to answer with this identity instead of the usual.
        identity_override: Mutex<Option<String>>,
        /// Refuse every handle, including freshly issued ones.
        refuse_handles: std::sync::atomic::AtomicBool,
    }

    impl FakeHost {
        fn restart_provider(&self) {
            self.provider_generation.fetch_add(1, Ordering::SeqCst);
        }
        fn dispatched(&self) -> Vec<String> {
            self.dispatched.lock().unwrap().clone()
        }
        fn transaction() -> Transaction {
            Transaction {
                id: "t1".into(),
                payment_hash: None,
                inbound: false,
                amount_msat: 1000,
                fee_msat: 0,
                status: PaymentStatus::Succeeded,
                claiming: false,
                status_msg: None,
                created_at_ms: 1,
                settled_at_ms: Some(2),
            }
        }
    }

    #[async_trait]
    impl BindingTransport for FakeHost {
        async fn bound_operation(
            &self,
            request: BoundOperationRequest,
        ) -> Result<BoundOperationResponse> {
            let generation = self.provider_generation.load(Ordering::SeqCst);
            let Some(handle) = request.handle else {
                let handle = format!("handle-{}", self.setups.fetch_add(1, Ordering::SeqCst));
                *self.bound.lock().unwrap() = Some((handle.clone(), generation));
                let wallet_id = self
                    .identity_override
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "wallet-1".into());
                return Ok(BoundOperationResponse::Responded {
                    handle,
                    output_json: json!({
                        "identity": {"wallet_id": wallet_id, "provider": "fake", "network": "mainnet"},
                        "created": false,
                    })
                    .to_string(),
                    is_error: false,
                });
            };
            let valid = self
                .bound
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|(bound, target)| *bound == handle && *target == generation);
            if !valid || self.refuse_handles.load(Ordering::SeqCst) {
                return Ok(BoundOperationResponse::NotDispatched {
                    reason: "handle requires setup and identity revalidation".into(),
                });
            }
            self.dispatched
                .lock()
                .unwrap()
                .push(request.operation.clone());
            let output = match request.operation.as_str() {
                ops::BALANCE => json!({"spendable_msat": 5000}),
                ops::PAY => serde_json::to_value(Self::transaction()).unwrap(),
                other => panic!("unexpected operation {other}"),
            };
            Ok(BoundOperationResponse::Responded {
                handle,
                output_json: output.to_string(),
                is_error: false,
            })
        }
    }

    async fn opened(host: &Arc<FakeHost>) -> (BoundWallet, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let transport: Arc<dyn BindingTransport> = host.clone();
        let wallet = BoundWallet::open(transport, "wallet", directory.path(), true)
            .await
            .unwrap();
        (wallet, directory)
    }

    #[tokio::test]
    async fn a_restarted_provider_is_set_up_again_and_the_call_retried() {
        let host = Arc::new(FakeHost::default());
        let (wallet, _directory) = opened(&host).await;
        assert_eq!(wallet.current_handle(), "handle-0");

        host.restart_provider();
        let balance = wallet.balance().await.unwrap();

        assert_eq!(balance.spendable_msat, 5000);
        assert_eq!(wallet.current_handle(), "handle-1");
        assert_eq!(host.setups.load(Ordering::SeqCst), 2);
        assert_eq!(host.dispatched(), vec![ops::BALANCE]);
    }

    #[tokio::test]
    async fn a_pay_refused_before_dispatch_is_submitted_once_after_setup() {
        let host = Arc::new(FakeHost::default());
        let (wallet, _directory) = opened(&host).await;
        let invoice = crate::tests::invoice(1, 1000);

        host.restart_provider();
        let payment = wallet.pay(&invoice, 1000, 1100).await.unwrap();

        assert_eq!(payment.status, PaymentStatus::Succeeded);
        assert_eq!(host.dispatched(), vec![ops::PAY]);
    }

    #[tokio::test]
    async fn concurrent_refusals_share_one_setup() {
        let host = Arc::new(FakeHost::default());
        let (wallet, _directory) = opened(&host).await;
        let wallet = Arc::new(wallet);

        host.restart_provider();
        let calls: Vec<_> = (0..8)
            .map(|_| {
                let wallet = wallet.clone();
                tokio::spawn(async move { wallet.balance().await.map(|b| b.spendable_msat) })
            })
            .collect();
        for call in calls {
            assert_eq!(call.await.unwrap().unwrap(), 5000);
        }

        assert_eq!(host.setups.load(Ordering::SeqCst), 2);
        assert_eq!(host.dispatched().len(), 8);
    }

    #[tokio::test]
    async fn a_second_refusal_is_reported_as_not_submitted() {
        let host = Arc::new(FakeHost::default());
        let (wallet, _directory) = opened(&host).await;
        let invoice = crate::tests::invoice(2, 1000);

        host.refuse_handles.store(true, Ordering::SeqCst);
        let error = wallet.pay(&invoice, 1000, 1100).await.unwrap_err();

        assert!(matches!(error, PayError::NotSubmitted(_)), "{error:?}");
        assert_eq!(host.setups.load(Ordering::SeqCst), 2);
        assert!(host.dispatched().is_empty());
    }

    #[tokio::test]
    async fn a_provider_that_comes_back_as_another_wallet_is_refused() {
        let host = Arc::new(FakeHost::default());
        let (wallet, _directory) = opened(&host).await;

        host.restart_provider();
        *host.identity_override.lock().unwrap() = Some("wallet-2".into());
        let error = wallet.balance().await.unwrap_err();

        assert!(
            error.to_string().contains("wallet identity changed"),
            "{error:#}"
        );
        assert!(host.dispatched().is_empty());
        // The stale handle is kept, so later calls fail the same way instead
        // of using the rejected wallet.
        assert_eq!(wallet.current_handle(), "handle-0");
    }
}
