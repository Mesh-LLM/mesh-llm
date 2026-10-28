//! Dedicated wallet.v1 child; transport failures never cause a send retry.
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto};
use mesh_llm_wallet::{contract::*, invoice::Invoice, provider::*, provisioning::WalletPin};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc};
use tokio::sync::Mutex;

pub struct ChildWallet {
    transport: crate::wallet_transport::Transport,
    _child: Mutex<tokio::process::Child>,
    _ownership: std::fs::File,
}

impl ChildWallet {
    #[cfg(unix)]
    pub async fn open(executable: &Path, directory: &Path, plugin: &str) -> Result<Arc<Self>> {
        Self::open_command(tokio::process::Command::new(executable), directory, plugin).await
    }

    #[cfg(unix)]
    async fn open_command(
        mut command: tokio::process::Command,
        directory: &Path,
        plugin: &str,
    ) -> Result<Arc<Self>> {
        let _timing = crate::diagnostics::Timing::start("wallet", "startup");
        // Fail on a corrupt pin before invoking a potentially provisioning process.
        let pinned = WalletPin::load(directory)?;
        ensure!(!plugin.is_empty(), "wallet plugin identity missing");
        ensure!(
            pinned.as_ref().is_none_or(|pin| pin.plugin == plugin),
            "selected wallet differs from pinned provider"
        );
        tokio::fs::create_dir_all(directory).await?;
        let ownership = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(directory.join("wallet-child.lock"))?;
        ownership.try_lock().map_err(|error| {
            anyhow::anyhow!("wallet profile already owned or cannot be locked: {error}")
        })?;

        let socket_directory = tempfile::tempdir()?;
        let socket = socket_directory.path().join("wallet.sock");
        ensure!(!socket.exists(), "wallet socket already exists");
        let listener = tokio::net::UnixListener::bind(&socket)?;

        command
            .env("MESH_LLM_PLUGIN_ENDPOINT", &socket)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .env("MESH_LLM_PLUGIN_NAME", plugin)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        let child_result = command.spawn();
        if child_result.is_err() {
            let _ = tokio::fs::remove_file(&socket).await;
        }
        let child = child_result?;

        let connected =
            tokio::time::timeout(std::time::Duration::from_secs(15), listener.accept()).await;
        let _ = tokio::fs::remove_file(&socket).await;
        let (stream, _) = connected??;
        let wallet = Arc::new(Self {
            transport: crate::wallet_transport::Transport::start(
                LocalStream::Unix(stream),
                plugin.into(),
            ),
            _child: Mutex::new(child),
            _ownership: ownership,
        });
        let initialized = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            wallet.exchange(proto::envelope::Payload::InitializeRequest(
                proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    host_version: env!("CARGO_PKG_VERSION").into(),
                    ..Default::default()
                },
            )),
        )
        .await??;
        let proto::envelope::Payload::InitializeResponse(initialized) = initialized else {
            anyhow::bail!("wallet rejected initialization")
        };
        ensure!(
            initialized.plugin_id == plugin
                && initialized.plugin_protocol_version == PROTOCOL_VERSION
                && initialized.capabilities.iter().any(|cap| cap == CAPABILITY),
            "incompatible wallet child"
        );

        let opened: OpenResponse = wallet
            .call(
                ops::OPEN,
                &OpenRequest {
                    directory: directory.to_string_lossy().into(),
                },
            )
            .await?;
        let pin = WalletPin {
            plugin: plugin.into(),
            wallet_id: opened.identity.wallet_id,
            provider: opened.identity.provider,
            network: opened.identity.network,
        };
        ensure!(pin.network == "mainnet", "wallet is not mainnet");
        ensure!(
            pinned.as_ref().is_none_or(|old| old == &pin),
            "wallet identity changed"
        );
        pin.store(directory)?;
        Ok(wallet)
    }

    async fn exchange(
        &self,
        payload: proto::envelope::Payload,
    ) -> Result<proto::envelope::Payload> {
        self.transport.exchange(payload).await
    }

    async fn call<T: Serialize, R: DeserializeOwned>(
        &self,
        operation: &str,
        input: &T,
    ) -> Result<R> {
        let _timing = crate::diagnostics::Timing::start("wallet", operation);
        let reply = self
            .exchange(proto::envelope::Payload::InvokeServiceRequest(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Operation as i32,
                    service_name: operation.into(),
                    input_json: serde_json::to_string(input)?,
                },
            ))
            .await?;
        let proto::envelope::Payload::InvokeServiceResponse(reply) = reply else {
            anyhow::bail!("unexpected wallet reply")
        };
        if reply.is_error {
            return Err(anyhow::Error::new(WalletCallError(WalletError::decode(
                &reply.output_json,
            ))));
        }
        Ok(serde_json::from_str(&reply.output_json)?)
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
impl WalletProvider for ChildWallet {
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
            } else if error.is::<crate::wallet_transport::NotDispatched>() {
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

/// Cached wallet construction. Plugin startup explicitly opens the provider;
/// manifest inspection and factory discovery never do.
pub struct ChildWalletFactory {
    pub executable: std::path::PathBuf,
    pub plugin: String,
    pub directory: std::path::PathBuf,
}
#[async_trait]
impl mesh_llm_wallet::provisioning::WalletFactory for ChildWalletFactory {
    fn is_provisioned(&self, _directory: &Path) -> bool {
        matches!(WalletPin::load(&self.directory), Ok(Some(_)))
    }
    async fn open(&self, directory: &Path) -> Result<Arc<dyn WalletProvider>> {
        // Payment obligations pin the wallet too: changing the wallet state path
        // must never let an existing ledger adopt a new funded identity.
        let ledger_pin = WalletPin::load(directory)?;
        let wallet_pin = WalletPin::load(&self.directory)?;
        ensure!(
            ledger_pin
                .as_ref()
                .is_none_or(|pin| Some(pin) == wallet_pin.as_ref()),
            "payment ledger and selected wallet identity differ"
        );
        let wallet = ChildWallet::open(&self.executable, &self.directory, &self.plugin).await?;
        let opened = WalletPin::load(&self.directory)?.context("wallet identity pin missing")?;
        opened.store(directory)?;
        Ok(wallet)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use mesh_llm_plugin::{connect_from_env, read_envelope, write_envelope};
    use serde_json::json;

    // This ignored helper is only launched by the tests below, never a real wallet.
    #[tokio::test]
    #[ignore = "offline mock wallet subprocess entrypoint"]
    async fn mock_wallet_child() {
        let mut stream = connect_from_env().await.unwrap();
        while let Ok(request) = read_envelope(&mut stream).await {
            let payload = match request.payload.unwrap() {
                proto::envelope::Payload::InitializeRequest(_) => {
                    proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                        plugin_id: std::env::var("MESH_LLM_PLUGIN_NAME").unwrap(),
                        plugin_protocol_version: PROTOCOL_VERSION,
                        capabilities: vec![CAPABILITY.into()],
                        ..Default::default()
                    })
                }
                proto::envelope::Payload::InvokeServiceRequest(op) => {
                    if op.service_name == ops::PAY {
                        return;
                    } // Disconnect after submission is ambiguous.
                    let output = if op.service_name == ops::OPEN {
                        json!({"identity":{"wallet_id":"offline-fixture","provider":"mock","network":"mainnet"},"created":false})
                    } else {
                        json!({"spendable_msat": 1234})
                    };
                    proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse {
                        output_json: output.to_string(),
                        is_error: false,
                    })
                }
                _ => panic!("unexpected fixture request"),
            };
            write_envelope(
                &mut stream,
                &proto::Envelope {
                    payload: Some(payload),
                    ..request
                },
            )
            .await
            .unwrap();
        }
    }
    #[tokio::test]
    async fn real_child_transport_opens_pins_and_keeps_disconnect_uncertain() {
        let directory = tempfile::tempdir().unwrap();
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "wallet_child::tests::mock_wallet_child",
            "--ignored",
        ]);
        let wallet = ChildWallet::open_command(command, directory.path(), "lexe-wallet")
            .await
            .unwrap();
        assert_eq!(wallet.balance().await.unwrap().spendable_msat, 1234);
        assert_eq!(
            WalletPin::load(directory.path())
                .unwrap()
                .unwrap()
                .wallet_id,
            "offline-fixture"
        );
        let invoice = crate::tests::invoice(98, 1000);
        assert!(matches!(
            wallet.pay(&invoice, 1000, 4000).await,
            Err(PayError::Uncertain(_))
        ));
    }
    #[tokio::test]
    async fn distinct_wallet_names_use_the_same_adapter_and_exclude_duplicate_owners() {
        for plugin in ["first-wallet", "second-wallet"] {
            let directory = tempfile::tempdir().unwrap();
            let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "wallet_child::tests::mock_wallet_child",
                "--ignored",
            ]);
            let wallet = ChildWallet::open_command(command, directory.path(), plugin)
                .await
                .unwrap();
            assert_eq!(wallet.balance().await.unwrap().spendable_msat, 1234);
            assert_eq!(
                WalletPin::load(directory.path()).unwrap().unwrap().plugin,
                plugin
            );
            // A second utility must fail before spawning its wallet process.
            let error =
                ChildWallet::open(Path::new("/not/a/wallet"), directory.path(), plugin).await;
            assert!(error.err().unwrap().to_string().contains("already owned"));
        }
    }

    #[tokio::test]
    async fn ledger_pin_prevents_selecting_a_fresh_wallet_directory() {
        use mesh_llm_wallet::provisioning::WalletFactory;
        let ledger = tempfile::tempdir().unwrap();
        let wallet = tempfile::tempdir().unwrap();
        WalletPin {
            plugin: "first-wallet".into(),
            wallet_id: "funded-wallet".into(),
            provider: "mock".into(),
            network: "mainnet".into(),
        }
        .store(ledger.path())
        .unwrap();
        let factory = ChildWalletFactory {
            executable: "/not/a/wallet".into(),
            plugin: "second-wallet".into(),
            directory: wallet.path().into(),
        };
        let error = factory.open(ledger.path()).await.err().unwrap();
        assert!(
            error
                .to_string()
                .contains("ledger and selected wallet identity differ")
        );
        assert!(WalletPin::load(wallet.path()).unwrap().is_none());
    }

    #[tokio::test]
    async fn corrupt_pin_fails_before_spawning_a_child() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(WalletPin::path(directory.path()), "corrupt").unwrap();
        assert!(
            ChildWallet::open(Path::new("/not/a/wallet"), directory.path(), "lexe-wallet")
                .await
                .is_err()
        );
    }
}

#[cfg(all(test, unix))]
mod identity_tests {
    use super::*;
    #[tokio::test]
    async fn different_pinned_wallet_is_rejected_before_sending() {
        let directory = tempfile::tempdir().unwrap();
        WalletPin {
            plugin: "lexe-wallet".into(),
            wallet_id: "another-wallet".into(),
            provider: "mock".into(),
            network: "mainnet".into(),
        }
        .store(directory.path())
        .unwrap();
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "wallet_child::tests::mock_wallet_child",
            "--ignored",
        ]);
        assert!(
            ChildWallet::open_command(command, directory.path(), "lexe-wallet")
                .await
                .is_err()
        );
        assert_eq!(
            WalletPin::load(directory.path())
                .unwrap()
                .unwrap()
                .wallet_id,
            "another-wallet"
        );
    }
}
