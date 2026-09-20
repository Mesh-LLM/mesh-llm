//! Local operator API. Never dispatch commands received from a mesh peer.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::invoice::Invoice;
use crate::ledger::{Charge, Policy, RequestTerms};
use crate::pricing::Pricing;
use crate::service::PaymentService;

#[derive(Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlCommand {
    Balance,
    InspectInvoice {
        invoice: String,
    },
    Transactions {
        limit: usize,
    },
    Fund {
        #[serde(default)]
        amount_msat: Option<u64>,
    },
    Send {
        invoice: String,
        amount_msat: Option<u64>,
        max_fee_msat: u64,
    },
    Pending,
    Approve {
        id: String,
    },
    Reject {
        id: String,
    },
    Policy {
        value: Option<Policy>,
    },
    PaymentIntent {
        value: Option<crate::intent::PaymentIntent>,
    },
    VettingPolicy {
        value: Option<crate::vetting::VettingPolicy>,
    },
    Pricing,
    SetPricing {
        model: String,
        value: Option<Pricing>,
    },
}

impl PaymentService {
    pub async fn control(&self, command: ControlCommand) -> Result<Value> {
        match command {
            ControlCommand::Balance => {
                let balance = self.wallet().await?.balance().await?;
                let available = self
                    .ledger
                    .available_budget(balance.spendable_msat, crate::now_ms())?;
                Ok(
                    json!({"spendable_msat": balance.spendable_msat, "available_for_inference_msat": available}),
                )
            }
            ControlCommand::InspectInvoice { invoice } => {
                Ok(serde_json::to_value(Invoice::parse(&invoice)?)?)
            }
            ControlCommand::Transactions { limit } => {
                ensure!(
                    (1..=1000).contains(&limit),
                    "limit must be between 1 and 1000"
                );
                Ok(serde_json::to_value(
                    self.wallet().await?.transactions(limit).await?,
                )?)
            }
            ControlCommand::Fund { amount_msat } => {
                if let Some(amount_msat) = amount_msat {
                    ensure!(amount_msat > 0, "amount must be greater than zero");
                }
                Ok(serde_json::to_value(
                    self.wallet().await?.create_invoice(amount_msat).await?,
                )?)
            }
            ControlCommand::Send {
                invoice,
                amount_msat,
                max_fee_msat,
            } => self.send_invoice(&invoice, amount_msat, max_fee_msat).await,
            ControlCommand::Pending => Ok(serde_json::to_value(self.ledger.requests()?)?),
            ControlCommand::Approve { id } => {
                self.approve(&id).await?;
                Ok(json!({"approved": id}))
            }
            ControlCommand::Reject { id } => {
                self.ledger.reject(&id)?;
                Ok(json!({"rejected": id}))
            }
            ControlCommand::Policy { value } => {
                if let Some(value) = value {
                    self.ledger.set_policy(&value)?;
                }
                Ok(serde_json::to_value(self.ledger.policy()?)?)
            }
            ControlCommand::PaymentIntent { value } => {
                if let Some(value) = value {
                    self.ledger.set_payment_intent(&value)?;
                }
                Ok(serde_json::to_value(self.ledger.payment_intent()?)?)
            }
            ControlCommand::VettingPolicy { value } => {
                if let Some(value) = value {
                    self.ledger.set_vetting_policy(&value)?;
                }
                Ok(serde_json::to_value(self.ledger.vetting_policy()?)?)
            }
            ControlCommand::Pricing => Ok(serde_json::to_value(self.ledger.pricing()?)?),
            ControlCommand::SetPricing { model, value } => {
                self.ledger.set_pricing(&model, value.as_ref())?;
                Ok(serde_json::to_value(self.ledger.pricing()?)?)
            }
        }
    }

    async fn send_invoice(&self, bolt11: &str, amount: Option<u64>, fee: u64) -> Result<Value> {
        let invoice = Invoice::parse(bolt11)?;
        let amount_msat = amount
            .or(invoice.amount_msat)
            .context("amount-less invoice requires amount_msat")?;
        let cap = amount_msat
            .checked_add(fee)
            .context("payment cap overflow")?;
        let id = format!("send-{}", invoice.payment_hash);
        if let Some(payment) = self.wallet().await?.lookup(&invoice.payment_hash).await? {
            ensure!(!payment.inbound, "cannot pay own invoice");
            ensure!(
                payment.payment_hash.as_deref() == Some(invoice.payment_hash.as_str()),
                "wallet returned a different payment hash"
            );
            let tracked = self.ledger.charge_state(&invoice.payment_hash)?.is_some();
            if tracked {
                self.ledger.reconcile(&payment, crate::now_ms())?;
            }
            match payment.status {
                crate::wallet::PaymentStatus::Succeeded => {
                    // Also repair phantom approvals made by earlier versions.
                    match self.ledger.request_state(&id)?.as_deref() {
                        Some("pending") => self.ledger.reject(&id)?,
                        Some("approved") => self.ledger.finish(&id)?,
                        _ => {}
                    }
                    return Ok(serde_json::to_value(payment)?);
                }
                crate::wallet::PaymentStatus::Failed => {
                    anyhow::bail!("invoice payment previously failed; no new payment submitted")
                }
                crate::wallet::PaymentStatus::Pending => {
                    ensure!(
                        tracked && self.ledger.request_state(&id)?.as_deref() == Some("approved"),
                        "payment is already pending outside this send request"
                    );
                }
            }
        }
        ensure!(
            self.ledger.charge_state(&invoice.payment_hash)?.is_none()
                || self.ledger.request_state(&id)?.is_some(),
            "invoice already belongs to an inference request; no new payment submitted"
        );
        if self.ledger.request_state(&id)?.is_none() {
            invoice.validate_payment(amount_msat, crate::now_ms())?;
        }
        // The payment hash is the idempotency key, including retries after a
        // CLI disconnect or node restart. Changing its authorization is refused.
        self.ledger.propose(&RequestTerms {
            id: id.clone(),
            peer: "wallet-send".into(),
            payee: Some(invoice.payee.clone()),
            model: "wallet-send".into(),
            pricing: Pricing {
                input_msat_per_million: 1,
                output_msat_per_million: 1,
                minimum_invoice_msat: 1,
            },
            input_tokens: 0,
            max_output_tokens: 1,
            max_total_msat: cap,
            expires_at_ms: invoice.expires_at_ms,
        })?;
        self.approve(&id).await?;
        let payment = self
            .pay_charge(&Charge {
                request_id: id.clone(),
                segment: 0,
                invoice,
                amount_msat,
                max_total_msat: cap,
            })
            .await?;
        self.ledger.finish(&id)?;
        Ok(serde_json::to_value(payment)?)
    }
}
