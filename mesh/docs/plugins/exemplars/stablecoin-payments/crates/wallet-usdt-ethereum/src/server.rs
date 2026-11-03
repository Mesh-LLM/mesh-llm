use crate::wallet::{Settings, Wallet};
use anyhow::{Result, ensure};
use mesh_llm_plugin::{
    OperationRouter, PluginError, PluginMetadata, PluginRuntime, SimplePlugin, capability,
    config_path, config_schema, config_setting, config_string, config_url, operation_with_schema,
    plugin_server_info, proto,
};
use serde_json::Value;
use std::sync::{Arc, OnceLock};
use wallet_contract::{LookupRequest, ops};

pub fn manifest() -> proto::PluginManifest {
    // Capability-only operations are callable by explicitly granted bindings,
    // not by the host's public HTTP/MCP tool endpoints.
    mesh_llm_plugin::plugin_manifest![
        capability(wallet_contract::CAPABILITY),
        config_schema("wallet-usdt-ethereum")
          .setting(config_setting("data_dir",config_path()).required(true).description("Absolute initialized owner-only Ethereum wallet state directory."))
          .setting(config_setting("rpc_url",config_url()).default_value(&"https://ethereum-rpc.publicnode.com").description("Ethereum mainnet RPC; HTTPS required except local fixtures."))
          .setting(config_setting("password_file",config_path()).description("Owner-only password file; omit for receive-only operation."))
          .setting(config_setting("node_binary",config_string()).default_value(&"node").description("Node.js 22+ executable for the bundled WDK worker."))
          .setting(config_setting("worker_path",config_path()).description("Optional absolute worker override for development; packaged default is wdk/worker.mjs."))
          .setting(config_setting("max_payment",config_string()).default_value(&"1000000").description("Local per-payment micro-USDT limit."))
          .setting(config_setting("max_fee",config_string()).default_value(&"0").description("Optional estimated ETH fee cap in wei; zero disables the application cap for this demo."))
    ]
}

pub async fn run() -> Result<()> {
    let state: Arc<OnceLock<Arc<Wallet>>> = Default::default();
    let mut router = OperationRouter::new();
    for (name, description) in [
        (
            ops::OPEN,
            "Open the configured noncustodial USDT wallet; never initializes keys.",
        ),
        (
            ops::BALANCE,
            "Read finalized USDT balance and local reservations.",
        ),
        (
            ops::ESTIMATE,
            "Estimate a USDT transfer; fees are native ETH in wei.",
        ),
        (
            ops::PAY,
            "Sign, journal, then submit one direct USDT transfer using WDK.",
        ),
        (
            ops::LOOKUP,
            "Reconcile the original journaled transaction without replacing it.",
        ),
        (
            ops::VERIFY,
            "Verify exact USDT delivery and signed calldata memo in finalized chain state.",
        ),
        (
            ops::ATTEMPTS,
            "List safe payment-attempt metadata; no keys or signed payloads.",
        ),
    ] {
        let state = state.clone();
        router.add_raw(operation_with_schema(name,description,serde_json::Map::new()), move |request,_| {
            let state = state.clone();
            Box::pin(async move {
                let wallet = state.get().cloned().ok_or_else(|| PluginError::internal("Ethereum wallet initialization incomplete"))?;
                let result: Result<Value> = async {
                    let input = request.arguments;
                    match name {
                        ops::OPEN => {
                            ensure!(input.as_object().is_some_and(|value| value.is_empty()),"wallet_open accepts only an empty object; wallet selection belongs to operator settings");
                            Ok(serde_json::to_value(wallet.info().await?)?)
                        }
                        ops::BALANCE => Ok(serde_json::to_value(wallet.balance().await?)?),
                        ops::ESTIMATE => Ok(serde_json::to_value(wallet.estimate(serde_json::from_value(input)?).await?)?),
                        ops::PAY => Ok(serde_json::to_value(wallet.pay(serde_json::from_value(input)?).await?)?),
                        ops::LOOKUP => {
                            let lookup: LookupRequest = serde_json::from_value(input)?;
                            Ok(serde_json::to_value(wallet.lookup(&lookup.payment_id).await?)?)
                        }
                        ops::VERIFY => Ok(serde_json::to_value(wallet.verify(serde_json::from_value(input)?).await?)?),
                        ops::ATTEMPTS => Ok(serde_json::to_value(wallet.journal.list()?)?),
                        _ => anyhow::bail!("unknown wallet operation"),
                    }
                }.await;
                match result {
                    Ok(value) => mesh_llm_plugin::structured_tool_result(value),
                    Err(error) => Ok(mesh_llm_plugin::operation_error(error.to_string())),
                }
            })
        });
    }
    let initialized = state.clone();
    let name =
        std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| "wallet-usdt-ethereum".into());
    let plugin = SimplePlugin::new(PluginMetadata::new(
        name,
        env!("CARGO_PKG_VERSION"),
        plugin_server_info(
            "wallet-usdt-ethereum",
            env!("CARGO_PKG_VERSION"),
            "USDT Ethereum wallet",
            "Noncustodial mainnet USDT wallet using Tether WDK",
            None::<String>,
        ),
    ))
    .with_capabilities(vec![wallet_contract::CAPABILITY.into()])
    .with_manifest(manifest())
    .with_operation_router(router)
    .on_initialize(move |request, _| {
        let state = initialized.clone();
        Box::pin(async move {
            let settings: Settings = serde_json::from_str(&request.settings_json)
                .map_err(|error| PluginError::invalid_params(error.to_string()))?;
            let wallet = Wallet::load(settings)
                .await
                .map_err(|error| PluginError::internal(error.to_string()))?;
            state
                .set(wallet)
                .map_err(|_| PluginError::internal("wallet already initialized"))?;
            Ok(())
        })
    });
    PluginRuntime::run(plugin).await
}
