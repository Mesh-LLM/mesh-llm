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
            if requires_wallet(&command) {
                bail!(
                    "mesh-llm is not running on port {port}; wallet operations need the running \
                     node because only it owns the wallet plugin. Start `mesh-llm` and retry. \
                     (ledger-only commands such as policy, pricing, min-payment, admission-deposit, \
                     pending, blocked, unblock and unpin work offline)"
                );
            }
            let directory = if let Some(config) = config {
                config.parent().unwrap_or(Path::new(".")).join("payments")
            } else {
                dirs::home_dir()
                    .context("home directory unavailable")?
                    .join(".mesh-llm/payments")
            };
            // Same owner-only directory the running node prepares.
            std::fs::create_dir_all(&directory)?;
            mesh_llm_config::restrict_private_directory(&directory)
                .with_context(|| format!("restricting {}", directory.display()))?;
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

/// Commands that need a live wallet, which only the running node can reach.
fn requires_wallet(command: &ControlCommand) -> bool {
    matches!(
        command,
        ControlCommand::Balance
            | ControlCommand::Transactions { .. }
            | ControlCommand::Fund { .. }
            | ControlCommand::Send { .. }
    )
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
        WalletCommand::Blocked => ControlCommand::Blocked,
        WalletCommand::Unblock { peer } => ControlCommand::Unblock { peer: peer.clone() },
        WalletCommand::Unpin => ControlCommand::Unpin,
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
        WalletCommand::MinPayment { sats } => ControlCommand::MinPayment {
            value_msat: sats
                .map(|sats| sats.checked_mul(1000).context("minimum payment overflow"))
                .transpose()?,
        },
        WalletCommand::AdmissionDeposit { sats, default } => ControlCommand::AdmissionDeposit {
            value_msat: sats
                .map(|sats| sats.checked_mul(1000).context("deposit overflow"))
                .transpose()?,
            reset: *default,
        },
        WalletCommand::Pricing {
            model,
            input_msat_per_million,
            output_msat_per_million,
            cache_discount_percent,
            free,
            deposit_sats,
            default_deposit,
        } => {
            if let Some(model) = model
                .as_ref()
                .filter(|_| deposit_sats.is_some() || *default_deposit)
            {
                ControlCommand::SetModelAdmissionDeposit {
                    model: model.clone(),
                    value_msat: deposit_sats
                        .map(|sats| sats.checked_mul(1000).context("deposit overflow"))
                        .transpose()?,
                }
            } else if let (Some(model), Some(percent), None, None) = (
                model,
                cache_discount_percent,
                input_msat_per_million,
                output_msat_per_million,
            ) {
                ControlCommand::SetCacheDiscount {
                    model: model.clone(),
                    percent: *percent,
                }
            } else if let Some(model) = model {
                ControlCommand::SetPricing {
                    model: model.clone(),
                    value: if *free {
                        None
                    } else {
                        Some(Pricing {
                            input_msat_per_million: input_msat_per_million.unwrap_or(500),
                            output_msat_per_million: output_msat_per_million.unwrap_or(1500),
                            cache_discount_percent: cache_discount_percent.unwrap_or(0),
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

    #[test]
    fn unpin_is_a_ledger_only_command() -> Result<()> {
        let command = control_command(&WalletCommand::Unpin)?;
        assert!(!requires_wallet(&command));
        assert_eq!(
            serde_json::to_value(command)?,
            serde_json::json!({"command": "unpin"})
        );
        Ok(())
    }

    #[test]
    fn a_model_deposit_sets_or_clears_only_that_deposit() -> Result<()> {
        let pricing = |deposit_sats, default_deposit| WalletCommand::Pricing {
            model: Some("m".into()),
            input_msat_per_million: None,
            output_msat_per_million: None,
            cache_discount_percent: None,
            free: false,
            deposit_sats,
            default_deposit,
        };
        assert_eq!(
            serde_json::to_value(control_command(&pricing(Some(5), false))?)?,
            serde_json::json!({"command": "set_model_admission_deposit", "model": "m", "value_msat": 5000})
        );
        assert_eq!(
            serde_json::to_value(control_command(&pricing(None, true))?)?,
            serde_json::json!({"command": "set_model_admission_deposit", "model": "m", "value_msat": null})
        );
        assert!(!requires_wallet(&control_command(&pricing(None, true))?));
        Ok(())
    }

    #[test]
    fn admission_deposit_default_resets_the_set_deposit() -> Result<()> {
        assert_eq!(
            serde_json::to_value(control_command(&WalletCommand::AdmissionDeposit {
                sats: None,
                default: true,
            })?)?,
            serde_json::json!({"command": "admission_deposit", "value_msat": null, "reset": true})
        );
        Ok(())
    }
}
