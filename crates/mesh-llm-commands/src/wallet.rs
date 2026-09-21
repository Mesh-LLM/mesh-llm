use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use mesh_llm_cli::wallet::{PaymentMode, WalletCommand};
use mesh_llm_payments::control::ControlCommand;
use mesh_llm_payments::ledger::{ApprovalMode, Policy};
use mesh_llm_payments::pricing::Pricing;
use mesh_llm_payments::service::PaymentService;

pub async fn run(command: &WalletCommand, port: u16, config: Option<&Path>) -> Result<()> {
    let command = control_command(command)?;
    let mut request = serde_json::to_value(&command)?;
    if let Some(config) = config {
        let directory = config
            .canonicalize()?
            .parent()
            .context("config directory unavailable")?
            .join("payments");
        request["expected_directory"] = serde_json::to_value(directory)?;
    }
    let client = reqwest::Client::builder().no_proxy().build()?;
    let result = client
        .post(format!("http://127.0.0.1:{port}/api/wallet"))
        .json(&request)
        .send()
        .await;
    let value: serde_json::Value = match result {
        Ok(response) => {
            let success = response.status().is_success();
            let body: serde_json::Value = response.json().await?;
            if !success {
                bail!("wallet API rejected request: {}", body);
            }
            body
        }
        // Never repeat a request after it may have reached the server.
        Err(error) if error.is_connect() => {
            let directory = if let Some(config) = config {
                config.parent().unwrap_or(Path::new(".")).join("payments")
            } else {
                dirs::home_dir()
                    .context("home directory unavailable")?
                    .join(".mesh-llm/payments")
            };
            PaymentService::open(&directory)?.control(command).await?
        }
        Err(error) => return Err(error.into()),
    };
    writeln!(
        mesh_llm_events::machine_out(),
        "{}",
        serde_json::to_string_pretty(&value)?
    )?;
    Ok(())
}

fn control_command(command: &WalletCommand) -> Result<ControlCommand> {
    Ok(match command {
        WalletCommand::GetBalance => ControlCommand::Balance,
        WalletCommand::GetTransactions { limit } => ControlCommand::Transactions { limit: *limit },
        WalletCommand::FundWallet { amount_sats } => ControlCommand::Fund {
            amount_msat: match amount_sats {
                Some(sats) => Some(
                    sats.checked_mul(1000)
                        .context("amount-sats is too large to express in millisatoshis")?,
                ),
                None => None,
            },
        },
        WalletCommand::Send {
            invoice,
            amount_msat,
            max_fee_msat,
        } => ControlCommand::Send {
            invoice: invoice.clone(),
            amount_msat: *amount_msat,
            max_fee_msat: *max_fee_msat,
        },
        WalletCommand::Pending => ControlCommand::Pending,
        WalletCommand::Policy {
            mode,
            daily_budget_sats,
        } => ControlCommand::Policy {
            value: mode
                .map(|mode| -> Result<Policy> {
                    Ok(Policy {
                        mode: match mode {
                            PaymentMode::FreeOnly => ApprovalMode::FreeOnly,
                            PaymentMode::Automatic => ApprovalMode::Automatic,
                        },
                        daily_budget_msat: daily_budget_sats
                            .map(|sats| sats.checked_mul(1000).context("budget overflow"))
                            .transpose()?,
                    })
                })
                .transpose()?,
        },
        WalletCommand::Pricing {
            model,
            input_msat_per_million,
            output_msat_per_million,
            minimum_invoice_msat,
            free,
        } => {
            if let Some(model) = model {
                ControlCommand::SetPricing {
                    model: model.clone(),
                    value: if *free {
                        None
                    } else {
                        Some(Pricing {
                            input_msat_per_million: input_msat_per_million.unwrap_or(500),
                            output_msat_per_million: output_msat_per_million.unwrap_or(1500),
                            minimum_invoice_msat: *minimum_invoice_msat,
                        })
                    },
                }
            } else {
                ControlCommand::Pricing
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_policy_is_one_command_and_inspection_is_read_only() -> Result<()> {
        let command = control_command(&WalletCommand::Policy {
            mode: Some(PaymentMode::Automatic),
            daily_budget_sats: Some(100),
        })?;
        assert_eq!(
            serde_json::to_value(command)?,
            serde_json::json!({
                "command": "policy", "value": {"mode":"automatic", "daily_budget_msat":100000}
            })
        );
        assert_eq!(
            serde_json::to_value(control_command(&WalletCommand::Policy {
                mode: None,
                daily_budget_sats: None,
            })?)?,
            serde_json::json!({"command":"policy","value":null})
        );
        assert!(
            control_command(&WalletCommand::Policy {
                mode: Some(PaymentMode::Automatic),
                daily_budget_sats: Some(u64::MAX),
            })
            .is_err()
        );
        Ok(())
    }
}
