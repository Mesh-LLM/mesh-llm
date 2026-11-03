//! Explicit local fake chain. This binary has no key generation, signing, or RPC client.
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::{
    OperationRouter, PluginMetadata, PluginRuntime, SimplePlugin, capability, config_path,
    config_schema, config_setting, config_string, operation_with_schema, plugin_manifest,
    plugin_server_info, structured_tool_result,
};
use rusqlite::{Connection, params};
use serde::Deserialize;
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};
use wallet_contract::{
    Asset, Balance, FeeEstimate, LookupRequest, PayRequest, PaymentAttempt, PaymentRequest,
    PaymentState, Receipt, VerifyRequest, VerifyResult, WalletInfo, ops,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    mode: String,
    chain_path: PathBuf,
    address: String,
}
struct Fixture {
    connection: Mutex<Connection>,
    address: String,
}

const OPERATIONS: [&str; 7] = [
    ops::OPEN,
    ops::BALANCE,
    ops::ESTIMATE,
    ops::PAY,
    ops::LOOKUP,
    ops::VERIFY,
    ops::ATTEMPTS,
];

fn manifest() -> mesh_llm_plugin::proto::PluginManifest {
    plugin_manifest()
        .item(capability(wallet_contract::CAPABILITY))
        .item(
            config_schema("demo-wallet-fixture")
                .setting(config_setting("mode", config_string()).required(true))
                .setting(config_setting("chain_path", config_path()).required(true))
                .setting(config_setting("address", config_string()).required(true)),
        )
        .build()
}

impl Fixture {
    fn open(settings: Settings) -> Result<Self> {
        ensure!(
            settings.mode == "offline-fixture",
            "explicit offline fixture mode required"
        );
        ensure!(
            settings.chain_path.is_absolute(),
            "fixture database path must be absolute"
        );
        wallet_contract::validate_address(&settings.address)?;
        let connection = Connection::open(&settings.chain_path)?;
        connection.busy_timeout(std::time::Duration::from_secs(10))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS fixture_payments (id TEXT PRIMARY KEY, attempt TEXT NOT NULL);")?;
        Ok(Self {
            connection: Mutex::new(connection),
            address: settings.address,
        })
    }
    fn execute(&self, name: &str, input: Value) -> Result<Value> {
        match name {
            ops::OPEN => {
                ensure!(
                    input.as_object().is_some_and(|o| o.is_empty()),
                    "setup accepts only empty object"
                );
                Ok(serde_json::to_value(WalletInfo {
                    wallet_id: format!("OFFLINE-{}", self.address),
                    address: self.address.clone(),
                    asset: Asset::default(),
                    method: wallet_contract::METHOD.into(),
                    can_send: true,
                })?)
            }
            ops::BALANCE => Ok(serde_json::to_value(Balance {
                balance: 1_000_000_000,
                reserved: 0,
                spendable: 1_000_000_000,
                block_number: 10,
            })?),
            ops::ESTIMATE => {
                let request: PaymentRequest = serde_json::from_value(input)?;
                request.validate()?;
                Ok(serde_json::to_value(FeeEstimate {
                    gas_limit: 1,
                    fee: 0,
                    fee_asset: Asset::default(),
                    block_number: 10,
                })?)
            }
            ops::PAY => self.pay(serde_json::from_value(input)?),
            ops::LOOKUP => {
                let lookup: LookupRequest = serde_json::from_value(input)?;
                Ok(serde_json::to_value(self.lookup(&lookup.payment_id)?)?)
            }
            ops::VERIFY => {
                let verify: VerifyRequest = serde_json::from_value(input)?;
                verify.request.validate()?;
                let attempt = self.lookup(&verify.request.obligation.id)?;
                ensure!(
                    attempt.request == verify.request
                        && attempt.tx_hashes.contains(&verify.tx_hash),
                    "fixture payment does not match"
                );
                Ok(serde_json::to_value(VerifyResult {
                    state: PaymentState::Finalized,
                    receipt: attempt.receipt,
                    reason: None,
                })?)
            }
            ops::ATTEMPTS => {
                let connection = self.connection.lock().unwrap();
                let mut statement =
                    connection.prepare("SELECT attempt FROM fixture_payments ORDER BY id")?;
                let entries = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let attempts = entries
                    .into_iter()
                    .map(|s| serde_json::from_str::<PaymentAttempt>(&s))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(serde_json::to_value(attempts)?)
            }
            _ => anyhow::bail!("unsupported fixture operation"),
        }
    }
    fn lookup(&self, id: &str) -> Result<PaymentAttempt> {
        let encoded: String = self
            .connection
            .lock()
            .unwrap()
            .query_row(
                "SELECT attempt FROM fixture_payments WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .context("unknown fixture payment")?;
        Ok(serde_json::from_str(&encoded)?)
    }
    fn pay(&self, pay: PayRequest) -> Result<Value> {
        pay.request.validate()?;
        ensure!(
            pay.request.obligation.payer == self.address,
            "payer is not this fixture wallet"
        );
        ensure!(
            pay.max_total >= pay.request.obligation.amount,
            "amount exceeds total limit"
        );
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        let existing = transaction.query_row(
            "SELECT attempt FROM fixture_payments WHERE id=?1",
            [&pay.payment_id],
            |row| row.get::<_, String>(0),
        );
        if let Ok(encoded) = existing {
            let old: PaymentAttempt = serde_json::from_str(&encoded)?;
            ensure!(old.request == pay.request, "fixture intent changed");
            return Ok(serde_json::to_value(old)?);
        }
        let receipt = Receipt {
            tx_hash: pay.request.memo.clone(),
            block_hash: format!("0x{}", "f0".repeat(32)),
            block_number: 10,
            transfer_log_index: 0,
            payer: self.address.clone(),
            recipient: pay.request.obligation.recipient.clone(),
            amount: pay.request.obligation.amount,
            memo: pay.request.memo.clone(),
            finalized: true,
        };
        let attempt = PaymentAttempt {
            payment_id: pay.payment_id.clone(),
            request: pay.request,
            state: PaymentState::Finalized,
            tx_hashes: vec![receipt.tx_hash.clone()],
            receipt: Some(receipt),
            error: None,
        };
        transaction.execute(
            "INSERT INTO fixture_payments(id,attempt) VALUES (?1,?2)",
            params![pay.payment_id, serde_json::to_string(&attempt)?],
        )?;
        transaction.commit()?;
        Ok(serde_json::to_value(attempt)?)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--manifest") {
        println!("{}", mesh_llm_plugin::package_manifest_json(&manifest())?);
        return Ok(());
    }
    let state: Arc<OnceLock<Fixture>> = Arc::new(OnceLock::new());
    let mut router = OperationRouter::new();
    for name in OPERATIONS {
        let state = state.clone();
        router.add_raw(
            operation_with_schema(name, "OFFLINE wallet fixture", Default::default()),
            move |request, _| {
                let state = state.clone();
                Box::pin(async move {
                    let result = state
                        .get()
                        .context("fixture is not initialized")?
                        .execute(name, request.arguments);
                    match result {
                        Ok(value) => structured_tool_result(value),
                        Err(error) => Ok(mesh_llm_plugin::operation_error(error.to_string())),
                    }
                })
            },
        );
    }
    let plugin = SimplePlugin::new(
        PluginMetadata::new(
            "demo-wallet-fixture",
            env!("CARGO_PKG_VERSION"),
            plugin_server_info(
                "demo-wallet-fixture",
                env!("CARGO_PKG_VERSION"),
                "Offline wallet fixture",
                "Fabricated payments in a local SQLite fixture; no real funds",
                None::<String>,
            ),
        )
        .with_manifest(manifest()),
    )
    .with_capabilities(vec![wallet_contract::CAPABILITY.into()])
    .with_operation_router(router)
    .on_initialize(move |request, _| {
        let state = state.clone();
        Box::pin(async move {
            let settings: Settings = serde_json::from_str(&request.settings_json)
                .map_err(|e| mesh_llm_plugin::PluginError::invalid_params(e.to_string()))?;
            let fixture = Fixture::open(settings)
                .map_err(|e| mesh_llm_plugin::PluginError::internal(e.to_string()))?;
            state
                .set(fixture)
                .map_err(|_| mesh_llm_plugin::PluginError::internal("fixture initialized twice"))?;
            Ok(())
        })
    });
    PluginRuntime::run(plugin).await
}
