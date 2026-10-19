//! Real plugin IPC regressions: the scripted test bridge does not apply deadlines.
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use mesh_llm_payments::ledger::metered_invoices::MeteredInvoiceKind;
use mesh_llm_payments::service::PaymentService;
use mesh_llm_payments_types::contract::{ServeAdmitRequest, ServeAdmitResponse};
use mesh_llm_payments_types::pricing::Pricing;
use mesh_llm_plugin::{
    InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginRuntime,
    operation_with_schema, plugin_server_info,
};
use mesh_llm_wallet::provider::WalletProvider;

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, PluginHostMode, ResolvedPlugins};

const NAME: &str = "slow-wallet";

fn slow_wallet(calls: Arc<AtomicUsize>) -> mesh_llm_plugin::InternalRpcPlugin {
    let mut router = OperationRouter::new();
    router.add_json(
        operation_with_schema(
            ops::CREATE_INVOICE,
            "Create a delayed invoice",
            Default::default(),
        ),
        move |_args: serde_json::Value, _context| {
            let calls = calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(31)).await;
                Ok(tests::sample_invoice())
            })
        },
    );
    router.add_json(
        operation_with_schema(ops::BALANCE, "Read a delayed balance", Default::default()),
        |_args: serde_json::Value, _context| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(31)).await;
                Ok(Balance { spendable_msat: 0 })
            })
        },
    );
    InternalRpcPluginBuilder::new(PluginMetadata::new(
        NAME,
        "0.0.0",
        plugin_server_info(NAME, "0.0.0", NAME, NAME, None::<String>),
    ))
    .with_capabilities(vec![CAPABILITY.into()])
    .with_manifest(mesh_llm_plugin::plugin_manifest![
        mesh_llm_plugin::capability(CAPABILITY)
    ])
    .with_operation_router(router)
    .build()
}

async fn wallet(
    directory: &Path,
) -> (
    Arc<PluginWalletProvider>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let starts = Arc::new(AtomicUsize::new(0));
    let runner: InProcessPluginRunner = {
        let calls = calls.clone();
        let starts = starts.clone();
        Arc::new(move |stream| {
            starts.fetch_add(1, Ordering::SeqCst);
            Box::pin(PluginRuntime::run_with_stream(
                slow_wallet(calls.clone()),
                stream,
            ))
        })
    };
    let wallet = wallet_with_runner(directory, runner).await;
    (wallet, calls, starts)
}

async fn wallet_with_runner(
    directory: &Path,
    runner: InProcessPluginRunner,
) -> Arc<PluginWalletProvider> {
    let (tx, _rx) = tokio::sync::mpsc::channel(32);
    let manager = PluginManager::start_with_in_process(
        &ResolvedPlugins {
            externals: vec![crate::plugin::in_process_builtin_spec(NAME)],
            inactive: vec![],
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
        InProcessPlugins::default().with(NAME, runner),
    )
    .await
    .unwrap();
    Arc::new(PluginWalletProvider {
        plugin_manager: manager,
        plugin_name: NAME.into(),
        payment_directory: directory.into(),
        wallet_directory: directory.join("wallet"),
        features: Default::default(),
        open_lock: Mutex::new(()),
        // This fixture exercises an already-open wallet over real IPC.
        identity_verified: AtomicBool::new(true),
    })
}

#[tokio::test(start_paused = true)]
async fn cold_invoice_outlives_query_deadline_without_replay() {
    let directory = tempfile::tempdir().unwrap();
    let (wallet, calls, starts) = wallet(directory.path()).await;
    let service = PaymentService::with_provider(directory.path(), wallet.clone()).unwrap();
    let pricing = Pricing {
        input_msat_per_million: 1000,
        output_msat_per_million: 1000,
        cache_discount_percent: 0,
    };
    service.ledger.set_pricing("model", Some(&pricing)).unwrap();
    let request = ServeAdmitRequest {
        billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
        request_id: "cold-request".into(),
        buyer: "buyer".into(),
        model: "model".into(),
        pricing,
    };
    let result = service.serve_admit(request.clone()).await;
    let repeated = service.serve_admit(request).await;
    wallet.plugin_manager.shutdown().await;
    let ServeAdmitResponse::Deposit { invoice, .. } = result.expect("cold invoice must complete")
    else {
        panic!("expected deposit");
    };
    let ServeAdmitResponse::Deposit { invoice: again, .. } = repeated.unwrap() else {
        panic!("expected reused deposit");
    };
    let receipt = service
        .ledger
        .metered_invoice("cold-request", MeteredInvoiceKind::Deposit)
        .unwrap()
        .unwrap();
    assert_eq!(invoice.payment_hash, receipt.invoice.payment_hash);
    assert_eq!(invoice.payment_hash, again.payment_hash);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(starts.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn slow_balance_keeps_query_deadline() {
    let directory = tempfile::tempdir().unwrap();
    let (wallet, _, _) = wallet(directory.path()).await;
    let result = wallet.balance().await;
    wallet.plugin_manager.shutdown().await;
    assert!(result.unwrap_err().to_string().contains("timed out"));
}

async fn lose_invoice_reply(
    mut stream: mesh_llm_plugin::LocalStream,
    calls: Arc<AtomicUsize>,
) -> Result<()> {
    use crate::plugin::proto;
    use mesh_llm_plugin::{read_envelope, write_envelope};
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: NAME.into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    manifest: Some(proto::PluginManifest::default()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(_)) => {
                calls.fetch_add(1, Ordering::SeqCst);
                // Simulate a committed mutation whose reply never reaches the host.
                return Ok(());
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: crate::plugin::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

#[tokio::test]
async fn lost_invoice_reply_is_not_replayed_by_wallet_adapter() {
    let directory = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let starts = Arc::new(AtomicUsize::new(0));
    let runner: InProcessPluginRunner = {
        let calls = calls.clone();
        let starts = starts.clone();
        Arc::new(move |stream| {
            starts.fetch_add(1, Ordering::SeqCst);
            Box::pin(lose_invoice_reply(stream, calls.clone()))
        })
    };
    let wallet = wallet_with_runner(directory.path(), runner).await;
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        wallet.create_invoice(Some(1000), 300),
    )
    .await;
    wallet.plugin_manager.shutdown().await;
    assert!(result.expect("lost reply must return an error").is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(starts.load(Ordering::SeqCst), 1);
}
