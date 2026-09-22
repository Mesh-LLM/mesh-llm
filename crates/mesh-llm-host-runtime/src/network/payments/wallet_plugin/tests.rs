//! The adapter is exercised against a scripted in-process wallet plugin over
//! the plugin manager's test bridge, so every IPC-boundary rule is checked
//! without a real child process.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mesh_llm_payments::provisioning::{WalletFactory, WalletPin};
use mesh_llm_payments::wallet::{PayError, PaymentStatus, Transaction};
use mesh_llm_wallet::contract::{
    CAPABILITY, LookupResponse, OpenRequest, OpenResponse, WalletError, WalletErrorKind,
    WalletIdentity, ops,
};
use rmcp::model::{CallToolResult, ErrorCode};
use serde_json::json;

use super::PluginWalletFactory;
use crate::plugin::{self, PluginManager, PluginRpcBridge};

const PLUGIN: &str = "wallet-fake";

/// What the fake plugin should do on the next call of a given operation.
#[derive(Clone)]
enum Script {
    Ok(serde_json::Value),
    /// Structured `wallet.v1` error.
    WalletError(WalletErrorKind, &'static str),
    /// Transport-level failure (plugin crashed / IPC broken).
    Transport,
}

struct FakeWalletPlugin {
    scripts: std::sync::Mutex<BTreeMap<String, Vec<Script>>>,
    calls: std::sync::Mutex<Vec<String>>,
    opens: AtomicUsize,
    identity: std::sync::Mutex<WalletIdentity>,
}

impl FakeWalletPlugin {
    fn new(wallet_id: &str) -> Arc<Self> {
        Arc::new(Self {
            scripts: Default::default(),
            calls: Default::default(),
            opens: AtomicUsize::new(0),
            identity: std::sync::Mutex::new(identity(wallet_id)),
        })
    }

    fn script(&self, op: &str, steps: Vec<Script>) {
        self.scripts.lock().unwrap().insert(op.to_owned(), steps);
    }

    fn next(&self, op: &str) -> Option<Script> {
        let mut scripts = self.scripts.lock().unwrap();
        let steps = scripts.get_mut(op)?;
        if steps.is_empty() {
            return None;
        }
        Some(steps.remove(0))
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

fn identity(wallet_id: &str) -> WalletIdentity {
    WalletIdentity {
        wallet_id: wallet_id.into(),
        provider: "fake".into(),
        network: "mainnet".into(),
    }
}

fn transport_error(message: &str) -> plugin::proto::ErrorResponse {
    plugin::proto::ErrorResponse {
        code: ErrorCode::INTERNAL_ERROR.0,
        message: message.into(),
        data_json: String::new(),
    }
}

fn tool_json(result: CallToolResult) -> Result<plugin::RpcResult, plugin::proto::ErrorResponse> {
    Ok(plugin::RpcResult {
        result_json: serde_json::to_string(&result).map_err(|e| transport_error(&e.to_string()))?,
    })
}

impl PluginRpcBridge for Arc<FakeWalletPlugin> {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        let this = Arc::clone(self);
        Box::pin(async move {
            assert_eq!(plugin_name, PLUGIN);
            assert_eq!(method, "tools/call");
            let request: mesh_llm_plugin::OperationRequest =
                serde_json::from_str(&params_json).map_err(|e| transport_error(&e.to_string()))?;
            this.calls.lock().unwrap().push(request.name.clone());

            if let Some(step) = this.next(&request.name) {
                return match step {
                    Script::Ok(value) => tool_json(CallToolResult::structured(value)),
                    Script::WalletError(kind, message) => {
                        tool_json(CallToolResult::structured_error(
                            serde_json::to_value(WalletError::new(kind, message)).unwrap(),
                        ))
                    }
                    Script::Transport => Err(transport_error("plugin connection closed")),
                };
            }

            // Default behaviors when nothing is scripted.
            match request.name.as_str() {
                ops::OPEN => {
                    let open: OpenRequest = serde_json::from_value(request.arguments).unwrap();
                    assert!(open.directory.ends_with("/lexe"), "{}", open.directory);
                    this.opens.fetch_add(1, Ordering::SeqCst);
                    tool_json(CallToolResult::structured(
                        serde_json::to_value(OpenResponse {
                            identity: this.identity.lock().unwrap().clone(),
                            created: false,
                        })
                        .unwrap(),
                    ))
                }
                ops::BALANCE => tool_json(CallToolResult::structured(json!({"spendable_msat": 7}))),
                ops::LOOKUP => tool_json(CallToolResult::structured(
                    serde_json::to_value(LookupResponse { transaction: None }).unwrap(),
                )),
                other => Err(transport_error(&format!("unscripted op {other}"))),
            }
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

async fn manager_for(plugin: &Arc<FakeWalletPlugin>) -> PluginManager {
    let manager = PluginManager::for_test_bridge(&[PLUGIN], Arc::new(Arc::clone(plugin)));
    let mut manifests = BTreeMap::new();
    manifests.insert(
        PLUGIN.to_string(),
        mesh_llm_plugin::proto::PluginManifest {
            capabilities: vec![CAPABILITY.into()],
            ..Default::default()
        },
    );
    manager.set_test_manifests(manifests).await;
    manager
}

fn transaction(hash: &str, status: PaymentStatus) -> Transaction {
    Transaction {
        id: format!("tx-{hash}"),
        payment_hash: Some(hash.into()),
        inbound: false,
        amount_msat: 1000,
        fee_msat: 1,
        status,
        status_msg: None,
        created_at_ms: 1,
        settled_at_ms: None,
    }
}

fn sample_invoice() -> mesh_llm_wallet::invoice::Invoice {
    // Reuse the payments crate's signed test invoice helper shape: build one
    // directly so this module does not depend on private test helpers.
    use bitcoin::secp256k1::{Secp256k1, SecretKey};
    use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
    let secret = SecretKey::from_slice(&[7; 32]).unwrap();
    let bolt11 = InvoiceBuilder::new(Currency::Bitcoin)
        .description("test".into())
        .payment_hash(PaymentHash([9; 32]))
        .payment_secret(PaymentSecret([42; 32]))
        .current_timestamp()
        .expiry_time(std::time::Duration::from_secs(3600))
        .min_final_cltv_expiry_delta(144)
        .amount_milli_satoshis(1000)
        .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
        .unwrap()
        .to_string();
    mesh_llm_wallet::invoice::Invoice::parse(&bolt11).unwrap()
}

#[tokio::test]
async fn open_writes_pin_and_is_provisioned_afterwards() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let factory = PluginWalletFactory::new(manager);
    let dir = tempfile::tempdir().unwrap();

    assert!(!factory.is_provisioned(dir.path()));
    let wallet = factory.open(dir.path()).await.unwrap();
    assert!(factory.is_provisioned(dir.path()));
    let pin = WalletPin::load(dir.path()).unwrap();
    assert_eq!(pin.plugin, PLUGIN);
    assert_eq!(pin.wallet_id, "w1");
    assert_eq!(pin.network, "mainnet");
    assert_eq!(wallet.balance().await.unwrap().spendable_msat, 7);
    assert_eq!(plugin.opens.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn open_refuses_a_different_wallet_identity() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let factory = PluginWalletFactory::new(manager);
    let dir = tempfile::tempdir().unwrap();
    factory.open(dir.path()).await.unwrap();

    *plugin.identity.lock().unwrap() = identity("w2");
    let error = factory
        .open(dir.path())
        .await
        .err()
        .expect("open must fail")
        .to_string();
    assert!(error.contains("wallet identity mismatch"), "{error}");
    // The pin is never overwritten by a mismatching open.
    assert_eq!(WalletPin::load(dir.path()).unwrap().wallet_id, "w1");
}

#[tokio::test]
async fn open_refuses_non_mainnet_wallets() {
    let plugin = FakeWalletPlugin::new("w1");
    plugin.identity.lock().unwrap().network = "signet".into();
    let manager = manager_for(&plugin).await;
    let factory = PluginWalletFactory::new(manager);
    let dir = tempfile::tempdir().unwrap();
    let error = factory
        .open(dir.path())
        .await
        .err()
        .expect("open must fail")
        .to_string();
    assert!(error.contains("requires mainnet"), "{error}");
    assert!(WalletPin::load(dir.path()).is_none());
}

#[tokio::test]
async fn open_without_a_wallet_plugin_fails_cleanly() {
    let manager = PluginManager::for_test_bridge(&[], Arc::new(FakeWalletPlugin::new("x")));
    let factory = PluginWalletFactory::new(manager);
    let dir = tempfile::tempdir().unwrap();
    let error = factory
        .open(dir.path())
        .await
        .err()
        .expect("open must fail")
        .to_string();
    assert!(error.contains("no wallet plugin is running"), "{error}");
    assert!(!factory.is_provisioned(dir.path()));
}

#[tokio::test]
async fn queries_reopen_once_after_plugin_restart() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();
    plugin.script(
        ops::BALANCE,
        vec![Script::WalletError(WalletErrorKind::NotOpen, "restarted")],
    );
    assert_eq!(wallet.balance().await.unwrap().spendable_msat, 7);
    assert_eq!(plugin.opens.load(Ordering::SeqCst), 2);
    assert_eq!(
        plugin.calls(),
        vec![ops::OPEN, ops::BALANCE, ops::OPEN, ops::BALANCE]
    );
}

#[tokio::test]
async fn transport_loss_during_pay_is_uncertain_and_not_retried() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();
    plugin.script(ops::PAY, vec![Script::Transport]);
    let error = wallet.pay(&sample_invoice(), 1000, 2000).await.unwrap_err();
    assert!(matches!(error, PayError::Uncertain(_)), "{error}");
    let pays = plugin.calls().iter().filter(|c| *c == ops::PAY).count();
    assert_eq!(pays, 1, "pay must never be re-sent after transport loss");
}

#[tokio::test]
async fn structured_not_submitted_is_preserved_across_ipc() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();
    plugin.script(
        ops::PAY,
        vec![Script::WalletError(
            WalletErrorKind::NotSubmitted,
            "fee cap exceeded",
        )],
    );
    let error = wallet.pay(&sample_invoice(), 1000, 2000).await.unwrap_err();
    assert!(matches!(error, PayError::NotSubmitted(_)), "{error}");
    assert!(error.to_string().contains("fee cap exceeded"));
}

#[tokio::test]
async fn pay_reopens_once_on_not_open_then_gives_up() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();

    // First: not_open, reopen, success.
    plugin.script(
        ops::PAY,
        vec![
            Script::WalletError(WalletErrorKind::NotOpen, "restarted"),
            Script::Ok(serde_json::to_value(transaction("h", PaymentStatus::Pending)).unwrap()),
        ],
    );
    let tx = wallet.pay(&sample_invoice(), 1000, 2000).await.unwrap();
    assert_eq!(tx.status, PaymentStatus::Pending);
    assert_eq!(plugin.opens.load(Ordering::SeqCst), 2);

    // Second: not_open twice in a row is not looped; it becomes uncertain.
    plugin.script(
        ops::PAY,
        vec![
            Script::WalletError(WalletErrorKind::NotOpen, "restarted"),
            Script::WalletError(WalletErrorKind::NotOpen, "restarted again"),
        ],
    );
    let error = wallet.pay(&sample_invoice(), 1000, 2000).await.unwrap_err();
    assert!(matches!(error, PayError::Uncertain(_)), "{error}");
    assert_eq!(plugin.opens.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn unstructured_plugin_error_is_uncertain_for_pay() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();
    plugin.script(
        ops::PAY,
        vec![Script::WalletError(
            WalletErrorKind::Failed,
            "unknown provider error",
        )],
    );
    let error = wallet.pay(&sample_invoice(), 1000, 2000).await.unwrap_err();
    assert!(matches!(error, PayError::Uncertain(_)), "{error}");
}

#[tokio::test]
async fn lookup_decodes_optional_transaction() {
    let plugin = FakeWalletPlugin::new("w1");
    let manager = manager_for(&plugin).await;
    let wallet = PluginWalletFactory::new(manager)
        .open(tempfile::tempdir().unwrap().path())
        .await
        .unwrap();
    assert!(wallet.lookup("h").await.unwrap().is_none());
    plugin.script(
        ops::LOOKUP,
        vec![Script::Ok(
            serde_json::to_value(LookupResponse {
                transaction: Some(transaction("h", PaymentStatus::Succeeded)),
            })
            .unwrap(),
        )],
    );
    let found = wallet.lookup("h").await.unwrap().unwrap();
    assert_eq!(found.status, PaymentStatus::Succeeded);
}
