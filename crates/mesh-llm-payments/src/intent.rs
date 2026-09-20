//! Client willingness to pay is distinct from wallet approval and balance.
use crate::{ledger::Ledger, pricing::Pricing};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum PaymentIntent {
    #[default]
    FreeOnly,
    AllowPaid {
        max_input_msat_per_million: u64,
        max_output_msat_per_million: u64,
        max_total_msat: u64,
    },
}

impl PaymentIntent {
    pub fn validate(&self) -> Result<()> {
        if let Self::AllowPaid {
            max_input_msat_per_million,
            max_output_msat_per_million,
            max_total_msat,
        } = self
        {
            ensure!(
                *max_input_msat_per_million > 0 && *max_output_msat_per_million > 0,
                "paid intent requires positive rate caps"
            );
            ensure!(
                *max_total_msat > 0 && *max_total_msat <= i64::MAX as u64,
                "invalid total spending cap"
            );
        }
        Ok(())
    }

    /// A request may tighten a paid profile, never expand its authority.
    pub fn restrict(&self, request: &Self) -> Self {
        match (self, request) {
            (
                Self::AllowPaid {
                    max_input_msat_per_million: a,
                    max_output_msat_per_million: b,
                    max_total_msat: c,
                },
                Self::AllowPaid {
                    max_input_msat_per_million: x,
                    max_output_msat_per_million: y,
                    max_total_msat: z,
                },
            ) => Self::AllowPaid {
                max_input_msat_per_million: (*a).min(*x),
                max_output_msat_per_million: (*b).min(*y),
                max_total_msat: (*c).min(*z),
            },
            _ => Self::FreeOnly,
        }
    }

    pub fn permits(&self, price: &Pricing, total_including_fees: u64) -> bool {
        if self.validate().is_err() || price.validate().is_err() {
            return false;
        }
        match self {
            Self::FreeOnly => false,
            Self::AllowPaid {
                max_input_msat_per_million,
                max_output_msat_per_million,
                max_total_msat,
            } => {
                price.input_msat_per_million <= *max_input_msat_per_million
                    && price.output_msat_per_million <= *max_output_msat_per_million
                    && total_including_fees <= *max_total_msat
            }
        }
    }
}

impl Ledger {
    pub fn payment_intent(&self) -> Result<PaymentIntent> {
        self.get_setting("payment_intent")?.map_or_else(
            || Ok(PaymentIntent::default()),
            |value| Ok(serde_json::from_str(&value)?),
        )
    }

    pub fn set_payment_intent(&self, intent: &PaymentIntent) -> Result<()> {
        intent.validate()?;
        self.set_setting("payment_intent", &serde_json::to_string(intent)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            ledger.set_payment_intent(&PaymentIntent::AllowPaid {
                max_input_msat_per_million: 10,
                max_output_msat_per_million: 20,
                max_total_msat: 4000,
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
        assert!(!intent.permits(&expensive, 2000));
        ledger.set_payment_intent(&PaymentIntent::FreeOnly)?;
        assert!(!ledger.payment_intent()?.permits(&expensive, 1));
        assert!(
            serde_json::from_str::<PaymentIntent>(r#"{"mode":"allow_paid","max_total_msat":1000}"#)
                .is_err()
        );
        Ok(())
    }
}
