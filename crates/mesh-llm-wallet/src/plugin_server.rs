//! Project a [`WalletBackend`] onto the `wallet.v1` plugin contract.
//!
//! Wallet plugin executables call [`wallet_plugin`] with their backend and
//! hand the result to `mesh_llm_plugin::PluginRuntime::run`. The adapter owns
//! open-state tracking, request decoding, and structured error mapping so a
//! backend only needs to implement [`WalletBackend`] and [`WalletProvider`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mesh_llm_plugin::{
    InternalRpcPlugin, InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginResult,
    capability, json_schema_operation, plugin_server_info,
};
use rmcp::model::CallToolResult;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

use crate::backend::{OpenedWallet, WalletBackend};
use crate::contract::{
    CAPABILITY, CreateInvoiceRequest, Empty, LookupResponse, OpenRequest, OpenResponse, PayRequest,
    PaymentHashRequest, StatusRequest, StatusResponse, TransactionsRequest, WalletError,
    WalletIdentity, ops,
};
use crate::provider::WalletProvider;

struct OpenState {
    directory: PathBuf,
    identity: WalletIdentity,
    provider: Arc<dyn WalletProvider>,
}

/// Shared adapter state: one backend, at most one open wallet per process.
pub struct WalletServer<B: WalletBackend> {
    backend: B,
    open: Mutex<Option<OpenState>>,
    // Serializes concurrent `wallet_open` calls so two hosts racing on the
    // same directory cannot double-provision.
    open_lock: Mutex<()>,
}

impl<B: WalletBackend> WalletServer<B> {
    pub fn new(backend: B) -> Arc<Self> {
        Arc::new(Self {
            backend,
            open: Mutex::new(None),
            open_lock: Mutex::new(()),
        })
    }

    async fn provider(&self) -> Result<Arc<dyn WalletProvider>, WalletError> {
        self.open
            .lock()
            .await
            .as_ref()
            .map(|state| Arc::clone(&state.provider))
            .ok_or_else(WalletError::not_open)
    }

    pub async fn open(&self, request: OpenRequest) -> Result<OpenResponse, WalletError> {
        let directory = validate_directory(&request.directory)?;
        let _guard = self.open_lock.lock().await;
        if let Some(state) = self.open.lock().await.as_ref() {
            if state.directory != directory {
                return Err(WalletError::invalid(format!(
                    "wallet already open on a different directory ({})",
                    state.directory.display()
                )));
            }
            return Ok(OpenResponse {
                identity: state.identity.clone(),
                created: false,
            });
        }
        let OpenedWallet {
            identity,
            provider,
            created,
        } = self
            .backend
            .open(&directory)
            .await
            .map_err(|error| WalletError::failed(error.to_string()))?;
        *self.open.lock().await = Some(OpenState {
            directory,
            identity: identity.clone(),
            provider,
        });
        Ok(OpenResponse { identity, created })
    }

    pub async fn status(&self, request: StatusRequest) -> Result<StatusResponse, WalletError> {
        let directory = validate_directory(&request.directory)?;
        let open = self.open.lock().await;
        let open_here = open.as_ref().filter(|state| state.directory == directory);
        Ok(StatusResponse {
            provisioned: open_here.is_some() || self.backend.is_provisioned(&directory),
            open: open_here.is_some(),
            identity: open_here.map(|state| state.identity.clone()),
        })
    }
}

fn validate_directory(raw: &str) -> Result<PathBuf, WalletError> {
    let path = Path::new(raw.trim());
    if raw.trim().is_empty() || !path.is_absolute() {
        return Err(WalletError::invalid(
            "wallet directory must be an absolute path",
        ));
    }
    Ok(path.to_path_buf())
}

fn provider_failed(error: anyhow::Error) -> WalletError {
    WalletError::failed(error.to_string())
}

fn tool_result<T: Serialize>(result: Result<T, WalletError>) -> PluginResult<CallToolResult> {
    match result {
        Ok(value) => {
            let value = serde_json::to_value(value)
                .map_err(|err| mesh_llm_plugin::PluginError::internal(err.to_string()))?;
            Ok(CallToolResult::structured(value))
        }
        Err(error) => {
            let value = serde_json::to_value(error)
                .map_err(|err| mesh_llm_plugin::PluginError::internal(err.to_string()))?;
            Ok(CallToolResult::structured_error(value))
        }
    }
}

fn add_op<B, Req, Res, F, Fut>(
    router: &mut OperationRouter,
    server: &Arc<WalletServer<B>>,
    name: &'static str,
    description: &'static str,
    handler: F,
) where
    B: WalletBackend,
    Req: DeserializeOwned + schemars::JsonSchema + Send + 'static,
    Res: Serialize + Send + 'static,
    F: Fn(Arc<WalletServer<B>>, Req) -> Fut + Send + Sync + Copy + 'static,
    Fut: std::future::Future<Output = Result<Res, WalletError>> + Send + 'static,
{
    let server = Arc::clone(server);
    router.add_raw(
        json_schema_operation::<Req>(name, description),
        move |request, _context| {
            let server = Arc::clone(&server);
            Box::pin(async move {
                let args: Req = match request.arguments() {
                    Ok(args) => args,
                    Err(err) => {
                        return tool_result::<Res>(Err(WalletError::invalid(err.to_string())));
                    }
                };
                tool_result(handler(server, args).await)
            })
        },
    );
}

/// Build the `wallet.v1` operation router for `server`.
pub fn wallet_operation_router<B: WalletBackend>(server: &Arc<WalletServer<B>>) -> OperationRouter {
    let mut router = OperationRouter::new();

    add_op(
        &mut router,
        server,
        ops::OPEN,
        "Open or provision the wallet.",
        |s, req: OpenRequest| async move { s.open(req).await },
    );
    add_op(
        &mut router,
        server,
        ops::STATUS,
        "Inspect persisted wallet state.",
        |s, req: StatusRequest| async move { s.status(req).await },
    );
    add_op(
        &mut router,
        server,
        ops::BALANCE,
        "Spendable balance.",
        |s, _req: Empty| async move { s.provider().await?.balance().await.map_err(provider_failed) },
    );
    add_op(
        &mut router,
        server,
        ops::TRANSACTIONS,
        "Recent transactions.",
        |s, req: TransactionsRequest| async move {
            s.provider()
                .await?
                .transactions(req.limit)
                .await
                .map_err(provider_failed)
        },
    );
    add_op(
        &mut router,
        server,
        ops::CREATE_INVOICE,
        "Create a BOLT11 invoice.",
        |s, req: CreateInvoiceRequest| async move {
            s.provider()
                .await?
                .create_invoice(req.amount_msat, req.expiry_secs)
                .await
                .map_err(provider_failed)
        },
    );
    add_op(
        &mut router,
        server,
        ops::PAY,
        "Pay an invoice within a fee cap.",
        |s, req: PayRequest| async move {
            s.provider()
                .await?
                .pay(&req.invoice, req.amount_msat, req.max_total_msat)
                .await
                .map_err(WalletError::from)
        },
    );
    add_op(
        &mut router,
        server,
        ops::LOOKUP,
        "Look up a payment by hash.",
        |s, req: PaymentHashRequest| async move {
            s.provider()
                .await?
                .lookup(&req.payment_hash)
                .await
                .map(|transaction| LookupResponse { transaction })
                .map_err(provider_failed)
        },
    );
    add_op(
        &mut router,
        server,
        ops::WAIT_FOR_PAYMENT,
        "Wait for a terminal payment state.",
        |s, req: PaymentHashRequest| async move {
            s.provider()
                .await?
                .wait_for_payment(&req.payment_hash)
                .await
                .map_err(provider_failed)
        },
    );
    add_op(
        &mut router,
        server,
        ops::WAIT_FOR_ARRIVAL,
        "Wait for receiver-side arrival evidence.",
        |s, req: PaymentHashRequest| async move {
            s.provider()
                .await?
                .wait_for_arrival(&req.payment_hash)
                .await
                .map_err(provider_failed)
        },
    );

    router
}

/// Assemble a complete wallet plugin.
///
/// `plugin_name` must match the name the host launched the process under
/// (`MESH_LLM_PLUGIN_NAME`); `version` is reported to the host.
pub fn wallet_plugin<B: WalletBackend>(
    plugin_name: impl Into<String>,
    version: impl Into<String>,
    backend: B,
) -> InternalRpcPlugin {
    let plugin_name = plugin_name.into();
    let version = version.into();
    let server = WalletServer::new(backend);
    let provider_name = server.backend.provider_name();
    let health_server = Arc::clone(&server);

    InternalRpcPluginBuilder::new(PluginMetadata::new(
        plugin_name.clone(),
        version.clone(),
        plugin_server_info(
            format!("mesh-wallet-{provider_name}"),
            version,
            format!("Mesh {provider_name} wallet"),
            format!("Lightning wallet backend ({provider_name}) for mesh-llm paid inference."),
            Some("Internal wallet capability for the mesh-llm host. Not intended for direct agent use."),
        ),
    ))
    .with_capabilities(vec![CAPABILITY.into()])
    .with_manifest(mesh_llm_plugin::plugin_manifest![capability(CAPABILITY)])
    .with_operation_router(wallet_operation_router(&server))
    .with_health(move |_context| {
        let server = Arc::clone(&health_server);
        Box::pin(async move {
            let open = server.open.lock().await;
            Ok(match open.as_ref() {
                Some(state) => format!(
                    "provider={} open=true wallet_id={}",
                    state.identity.provider, state.identity.wallet_id
                ),
                None => format!("provider={provider_name} open=false"),
            })
        })
    })
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::WalletErrorKind;
    use crate::invoice::Invoice;
    use crate::provider::{Balance, PayError, Transaction};
    use anyhow::Result;
    use async_trait::async_trait;
    use mesh_llm_plugin::Plugin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeProvider;

    #[async_trait]
    impl WalletProvider for FakeProvider {
        async fn balance(&self) -> Result<Balance> {
            Ok(Balance { spendable_msat: 42 })
        }
        async fn transactions(&self, _limit: usize) -> Result<Vec<Transaction>> {
            Ok(Vec::new())
        }
        async fn create_invoice(
            &self,
            _amount_msat: Option<u64>,
            _expiry_secs: u32,
        ) -> Result<Invoice> {
            anyhow::bail!("no invoices in tests")
        }
        async fn pay(&self, _: &Invoice, _: u64, _: u64) -> Result<Transaction, PayError> {
            Err(PayError::NotSubmitted(anyhow::anyhow!("declined")))
        }
        async fn lookup(&self, _payment_hash: &str) -> Result<Option<Transaction>> {
            Ok(None)
        }
        async fn wait_for_payment(&self, _payment_hash: &str) -> Result<Transaction> {
            anyhow::bail!("never settles")
        }
    }

    struct FakeBackend {
        opens: AtomicUsize,
    }

    #[async_trait]
    impl WalletBackend for FakeBackend {
        fn provider_name(&self) -> &'static str {
            "fake"
        }
        fn is_provisioned(&self, directory: &Path) -> bool {
            directory.join("seed").exists()
        }
        async fn open(&self, directory: &Path) -> Result<OpenedWallet> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            std::fs::create_dir_all(directory)?;
            let created = !directory.join("seed").exists();
            std::fs::write(directory.join("seed"), b"s")?;
            Ok(OpenedWallet {
                identity: WalletIdentity {
                    wallet_id: "fake-1".into(),
                    provider: "fake".into(),
                    network: "regtest".into(),
                },
                provider: Arc::new(FakeProvider),
                created,
            })
        }
    }

    fn server() -> Arc<WalletServer<FakeBackend>> {
        WalletServer::new(FakeBackend {
            opens: AtomicUsize::new(0),
        })
    }

    #[tokio::test]
    async fn operations_before_open_report_not_open() {
        let server = server();
        let error = server.provider().await.err().unwrap();
        assert_eq!(error.kind, WalletErrorKind::NotOpen);
    }

    #[tokio::test]
    async fn open_is_idempotent_and_pins_directory() {
        let server = server();
        let dir = std::env::temp_dir().join(format!("mesh-wallet-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = server
            .open(OpenRequest {
                directory: dir.display().to_string(),
            })
            .await
            .unwrap();
        assert!(first.created);
        let second = server
            .open(OpenRequest {
                directory: dir.display().to_string(),
            })
            .await
            .unwrap();
        assert!(!second.created);
        assert_eq!(first.identity, second.identity);
        assert_eq!(server.backend.opens.load(Ordering::SeqCst), 1);

        let other = server
            .open(OpenRequest {
                directory: dir.join("other").display().to_string(),
            })
            .await
            .unwrap_err();
        assert_eq!(other.kind, WalletErrorKind::InvalidRequest);

        let status = server
            .status(StatusRequest {
                directory: dir.display().to_string(),
            })
            .await
            .unwrap();
        assert!(status.provisioned && status.open);
        assert_eq!(status.identity, Some(first.identity));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn relative_directory_is_rejected() {
        let error = server()
            .open(OpenRequest {
                directory: "relative/path".into(),
            })
            .await
            .unwrap_err();
        assert_eq!(error.kind, WalletErrorKind::InvalidRequest);
    }

    #[tokio::test]
    async fn pay_rejection_is_structured_not_submitted() {
        let server = server();
        let dir = std::env::temp_dir().join(format!("mesh-wallet-pay-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        server
            .open(OpenRequest {
                directory: dir.display().to_string(),
            })
            .await
            .unwrap();
        let result = server
            .provider()
            .await
            .unwrap()
            .pay(&crate::invoice::tests::sample_invoice(1, 1000), 1000, 2000)
            .await
            .map_err(WalletError::from)
            .unwrap_err();
        assert_eq!(result.kind, WalletErrorKind::NotSubmitted);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plugin_advertises_capability_and_every_operation() {
        let plugin = wallet_plugin(
            "wallet-fake",
            "0.0.0",
            FakeBackend {
                opens: AtomicUsize::new(0),
            },
        );
        assert_eq!(plugin.plugin_id(), "wallet-fake");
        assert_eq!(plugin.capabilities(), vec![CAPABILITY.to_string()]);
        let manifest = plugin.manifest().unwrap();
        assert_eq!(manifest.capabilities, vec![CAPABILITY.to_string()]);
    }
}
