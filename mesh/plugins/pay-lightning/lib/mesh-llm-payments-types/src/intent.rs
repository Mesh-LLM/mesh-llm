//! Request-local restrictions over the profile spending policy; never a saved opt-in.
use crate::pricing::Pricing;
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

    /// Rate caps only. `max_total_msat` is a legacy terms field, not a
    /// per-request spending cap: the daily budget alone gates new admissions.
    pub fn permits(&self, price: &Pricing) -> bool {
        if self.validate().is_err() || price.validate().is_err() {
            return false;
        }
        match self {
            Self::FreeOnly => false,
            Self::AllowPaid {
                max_input_msat_per_million,
                max_output_msat_per_million,
                ..
            } => {
                price.input_msat_per_million <= *max_input_msat_per_million
                    && price.output_msat_per_million <= *max_output_msat_per_million
            }
        }
    }
}
