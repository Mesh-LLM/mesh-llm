//! Profile-derived payment intent. The type lives in `mesh-llm-payments-types`.
use crate::ledger::Ledger;
use anyhow::Result;
pub use mesh_llm_payments_types::intent::PaymentIntent;

impl Ledger {
    /// Derive request eligibility from the single spending policy. No separate opt-in.
    pub fn payment_intent(&self) -> Result<PaymentIntent> {
        let policy = self.policy()?;
        policy.validate()?;
        Ok(match policy.mode {
            crate::ledger::ApprovalMode::FreeOnly => PaymentIntent::FreeOnly,
            crate::ledger::ApprovalMode::Automatic => PaymentIntent::AllowPaid {
                max_input_msat_per_million: u64::MAX,
                max_output_msat_per_million: u64::MAX,
                max_total_msat: policy.daily_budget_msat.unwrap_or(0),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::Pricing;
    #[test]
    fn requests_cannot_expand_profile_authority() {
        let profile = PaymentIntent::AllowPaid {
            max_input_msat_per_million: 10,
            max_output_msat_per_million: 20,
            max_total_msat: 4000,
        };
        let broad = PaymentIntent::AllowPaid {
            max_input_msat_per_million: 100,
            max_output_msat_per_million: 200,
            max_total_msat: 8000,
        };
        let price = Pricing {
            input_msat_per_million: 10,
            output_msat_per_million: 20,
            minimum_invoice_msat: 1,
        };
        assert!(profile.restrict(&broad).permits(&price, 4000));
        assert!(!profile.restrict(&broad).permits(&price, 4001));
        assert!(!PaymentIntent::FreeOnly.restrict(&broad).permits(&price, 1));
        assert!(
            !profile
                .restrict(&PaymentIntent::FreeOnly)
                .permits(&price, 1)
        );
    }

    #[test]
    fn intent_defaults_free_and_caps_rates_and_fees_across_restart() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let price = Pricing {
            input_msat_per_million: 10,
            output_msat_per_million: 20,
            minimum_invoice_msat: 1000,
        };
        {
            let ledger = Ledger::open(directory.path())?;
            assert!(!ledger.payment_intent()?.permits(&price, 4000));
            ledger.set_policy(&crate::ledger::Policy {
                mode: crate::ledger::ApprovalMode::Automatic,
                daily_budget_msat: Some(4000),
            })?;
        }
        let ledger = Ledger::open(directory.path())?;
        let intent = ledger.payment_intent()?;
        assert!(intent.permits(&price, 4000));
        assert!(!intent.permits(&price, 4001));
        let expensive = Pricing {
            input_msat_per_million: 11,
            ..price
        };
        assert!(intent.permits(&expensive, 2000));
        ledger.set_policy(&crate::ledger::Policy::default())?;
        assert!(!ledger.payment_intent()?.permits(&expensive, 1));
        assert!(
            serde_json::from_str::<PaymentIntent>(r#"{"mode":"allow_paid","max_total_msat":1000}"#)
                .is_err()
        );
        Ok(())
    }
}
