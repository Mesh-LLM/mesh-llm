//! `mesh-llm wallet …`: a thin forward to the installed payments plugin.
//!
//! The host links no payment code. Each subcommand becomes the plugin's
//! `ControlCommand` JSON and is posted to the running node's local plugin API
//! (`POST /api/plugins/<plugin>/tools/control`); the plugin's answer is printed
//! as is. The plugin name defaults to `pay-lightning`.
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mesh_llm_cli::wallet::{PaymentMode, WalletCommand};
use serde_json::{Value, json};

/// The payments plugin whose `control` operation serves these commands.
pub const DEFAULT_PAYMENTS_PLUGIN: &str = "pay-lightning";

pub async fn run(command: &WalletCommand, port: u16, _config: Option<&Path>) -> Result<()> {
    let request = control_command(command)?;
    let plugin = std::env::var("MESH_LLM_PAYMENTS_PLUGIN")
        .ok()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PAYMENTS_PLUGIN.to_owned());
    let client = reqwest::Client::builder().no_proxy().build()?;
    let response = client
        .post(format!(
            "http://127.0.0.1:{port}/api/plugins/{plugin}/tools/control"
        ))
        .json(&request)
        .send()
        .await;
    let value: Value = match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                if status.as_u16() == 404 {
                    bail!(
                        "the node on port {port} has no running payments plugin named \
                         '{plugin}' publishing `control`: install it (`mesh-llm plugins \
                         install …`) and restart the node ({})",
                        body.trim()
                    );
                }
                bail!("payments plugin rejected the request: {}", body.trim());
            }
            serde_json::from_str(&body).unwrap_or(Value::String(body))
        }
        // Never repeat a request after it may have reached the server.
        Err(error) if error.is_connect() => bail!(
            "mesh-llm is not running on port {port}; wallet commands need the running node \
             because only it hosts the payments and wallet plugins. Start `mesh-llm` and retry."
        ),
        Err(error) => return Err(error.into()),
    };
    writeln!(
        mesh_llm_events::machine_out(),
        "{}",
        serde_json::to_string_pretty(&value)?
    )?;
    Ok(())
}

fn sats_to_msat(sats: u64, what: &str) -> Result<u64> {
    sats.checked_mul(1000)
        .with_context(|| format!("{what} is too large to express in millisatoshis"))
}

/// The plugin's `ControlCommand` wire shape: `{"command": <snake_case>, …fields}`.
fn control_command(command: &WalletCommand) -> Result<Value> {
    Ok(match command {
        WalletCommand::GetBalance => json!({ "command": "balance" }),
        WalletCommand::GetTransactions { limit } => {
            json!({ "command": "transactions", "limit": limit })
        }
        WalletCommand::FundWallet { amount_sats } => json!({
            "command": "fund",
            "amount_msat": amount_sats
                .map(|sats| sats_to_msat(sats, "amount-sats"))
                .transpose()?,
        }),
        WalletCommand::Send {
            invoice,
            amount_msat,
            max_fee_msat,
        } => json!({
            "command": "send",
            "invoice": invoice,
            "amount_msat": amount_msat,
            "max_fee_msat": max_fee_msat,
        }),
        WalletCommand::Pending => json!({ "command": "pending" }),
        WalletCommand::Blocked => json!({ "command": "blocked" }),
        WalletCommand::Unblock { peer } => json!({ "command": "unblock", "peer": peer }),
        WalletCommand::Unpin => json!({ "command": "unpin" }),
        WalletCommand::Policy {
            mode,
            daily_budget_sats,
        } => json!({
            "command": "policy",
            "value": mode
                .map(|mode| -> Result<Value> {
                    Ok(json!({
                        "mode": match mode {
                            PaymentMode::FreeOnly => "free_only",
                            PaymentMode::Automatic => "automatic",
                        },
                        "daily_budget_msat": daily_budget_sats
                            .map(|sats| sats_to_msat(sats, "budget"))
                            .transpose()?,
                    }))
                })
                .transpose()?,
        }),
        WalletCommand::MinPayment { sats } => json!({
            "command": "min_payment",
            "value_msat": sats.map(|sats| sats_to_msat(sats, "minimum payment")).transpose()?,
        }),
        WalletCommand::AdmissionDeposit { sats, default } => json!({
            "command": "admission_deposit",
            "value_msat": sats.map(|sats| sats_to_msat(sats, "deposit")).transpose()?,
            "reset": default,
        }),
        WalletCommand::Pricing {
            model,
            input_msat_per_million,
            output_msat_per_million,
            cache_discount_percent,
            free,
            deposit_sats,
            default_deposit,
        } => match model {
            Some(model) if deposit_sats.is_some() || *default_deposit => json!({
                "command": "set_model_admission_deposit",
                "model": model,
                "value_msat": deposit_sats.map(|sats| sats_to_msat(sats, "deposit")).transpose()?,
            }),
            Some(model) if *free => {
                json!({ "command": "set_pricing", "model": model, "value": null })
            }
            Some(model)
                if input_msat_per_million.is_none()
                    && output_msat_per_million.is_none()
                    && cache_discount_percent.is_some() =>
            {
                json!({
                    "command": "set_cache_discount",
                    "model": model,
                    "percent": cache_discount_percent,
                })
            }
            Some(model) => json!({
                "command": "set_pricing",
                "model": model,
                "value": {
                    "input_msat_per_million": input_msat_per_million.unwrap_or(500),
                    "output_msat_per_million": output_msat_per_million.unwrap_or(1500),
                    "cache_discount_percent": cache_discount_percent.unwrap_or(0),
                },
            }),
            None => json!({ "command": "pricing" }),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_take_the_plugin_control_wire_shape() -> Result<()> {
        assert_eq!(
            control_command(&WalletCommand::Policy {
                mode: Some(PaymentMode::Automatic),
                daily_budget_sats: Some(100),
            })?,
            json!({"command": "policy", "value": {"mode": "automatic", "daily_budget_msat": 100_000}})
        );
        assert_eq!(
            control_command(&WalletCommand::GetBalance)?,
            json!({"command": "balance"})
        );
        assert_eq!(
            control_command(&WalletCommand::Pricing {
                model: Some("m".into()),
                input_msat_per_million: None,
                output_msat_per_million: None,
                cache_discount_percent: None,
                free: false,
                deposit_sats: None,
                default_deposit: false,
            })?,
            json!({"command": "set_pricing", "model": "m", "value": {"input_msat_per_million": 500, "output_msat_per_million": 1500, "cache_discount_percent": 0}})
        );
        Ok(())
    }
}
