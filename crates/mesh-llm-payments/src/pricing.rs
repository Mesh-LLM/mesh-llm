use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Each of the two PoC invoices reserves this maximum outgoing routing fee.
pub const FEE_ALLOWANCE_MSAT: u64 = 1000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Pricing {
    pub input_msat_per_million: u64,
    pub output_msat_per_million: u64,
    pub minimum_invoice_msat: u64,
}

impl Pricing {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.minimum_invoice_msat > 0,
            "minimum invoice must be positive"
        );
        ensure!(
            self.input_msat_per_million > 0 && self.output_msat_per_million > 0,
            "paid serving requires positive input and output rates"
        );
        Ok(())
    }

    pub fn input_charge(&self, tokens: u64) -> Result<u64> {
        self.charge(self.input_msat_per_million, tokens)
    }

    pub fn output_charge(&self, tokens: u64) -> Result<u64> {
        self.charge(self.output_msat_per_million, tokens)
    }

    fn charge(&self, rate: u64, tokens: u64) -> Result<u64> {
        self.validate()?;
        if tokens == 0 {
            return Ok(0);
        }
        let charge = (u128::from(rate) * u128::from(tokens)).div_ceil(1_000_000);
        let minimum = u128::from(self.minimum_invoice_msat);
        (charge.div_ceil(minimum) * minimum)
            .try_into()
            .context("inference charge overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_rates_round_once_at_the_invoice_boundary() {
        let rates = Pricing {
            input_msat_per_million: 500,
            output_msat_per_million: 1500,
            minimum_invoice_msat: 1,
        };
        assert_eq!(rates.input_charge(1000).unwrap(), 1);
        assert_eq!(rates.output_charge(1000).unwrap(), 2);
        assert_eq!(rates.output_charge(0).unwrap(), 0);
        assert_eq!(rates.input_charge(1_000_000).unwrap(), 500);
    }

    #[test]
    fn provider_granularity_and_overflow_are_enforced() {
        let mut rates = Pricing {
            input_msat_per_million: 1_000_001,
            output_msat_per_million: 1,
            minimum_invoice_msat: 1000,
        };
        assert_eq!(rates.input_charge(1000).unwrap(), 2000);
        rates.input_msat_per_million = u64::MAX;
        assert!(rates.input_charge(u64::MAX).is_err());
    }
}
