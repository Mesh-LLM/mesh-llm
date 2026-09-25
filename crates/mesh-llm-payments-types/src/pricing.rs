use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Smallest outgoing routing-fee reservation per payment, in msat. Lightning
/// hops commonly charge a base fee of about one sat each and inference invoices
/// are often only a few msat, so a purely proportional allowance would starve
/// small payments of any route; three sats covers a typical short route.
pub const FEE_ALLOWANCE_FLOOR_MSAT: u64 = 3_000;
/// Proportional part of the fee reservation, in parts per million of the
/// payment amount (1%). Routing fees on well-connected paths are usually far
/// below this; it is a cap the payer reserves, not an amount it spends.
pub const FEE_ALLOWANCE_PPM: u64 = 10_000;

/// Maximum routing fee the payer authorizes on top of `amount_msat`.
///
/// This is the one place the fee policy lives. Both peers compute it: the
/// seller when it proposes request terms and the payer when it validates
/// them, so the result must be a pure function of the amount. The wallet is
/// told the resulting cap (`WalletProvider::pay(.., max_total_msat)`) and must
/// not submit a payment whose amount plus fees exceeds it; the ledger records
/// what was actually debited.
pub fn fee_allowance_msat(amount_msat: u64) -> Result<u64> {
    let proportional =
        (u128::from(amount_msat) * u128::from(FEE_ALLOWANCE_PPM)).div_ceil(1_000_000);
    u64::try_from(proportional.max(u128::from(FEE_ALLOWANCE_FLOOR_MSAT)))
        .context("fee allowance overflow")
}

/// `amount_msat` plus its fee allowance: the cap handed to the wallet for one
/// payment.
pub fn payment_cap_msat(amount_msat: u64) -> Result<u64> {
    amount_msat
        .checked_add(fee_allowance_msat(amount_msat)?)
        .context("payment cap overflow")
}

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

    /// The payer's total reservation for one request: both inference charges
    /// plus the routing-fee allowance for each. Seller and payer both compute
    /// this from the same inputs and the payer rejects terms that disagree, so
    /// it must stay a pure function of pricing, input charge and output
    /// allowance.
    pub fn request_cap_msat(&self, input_amount: u64, max_output: u64) -> Result<u64> {
        let output_amount = self.output_charge(max_output)?;
        payment_cap_msat(input_amount)?
            .checked_add(payment_cap_msat(output_amount)?)
            .context("request price overflow")
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
    fn fee_allowance_has_a_floor_and_grows_with_the_amount() {
        assert_eq!(fee_allowance_msat(0).unwrap(), FEE_ALLOWANCE_FLOOR_MSAT);
        assert_eq!(fee_allowance_msat(1).unwrap(), FEE_ALLOWANCE_FLOOR_MSAT);
        // 1% of 200,000 msat is 2,000 msat, still under the floor.
        assert_eq!(
            fee_allowance_msat(200_000).unwrap(),
            FEE_ALLOWANCE_FLOOR_MSAT
        );
        assert_eq!(
            fee_allowance_msat(300_000).unwrap(),
            FEE_ALLOWANCE_FLOOR_MSAT
        );
        // 1% of 1,000,000 msat is 10,000 msat, above the floor.
        assert_eq!(fee_allowance_msat(1_000_000).unwrap(), 10_000);
        // Rounds up, never down.
        assert_eq!(fee_allowance_msat(1_000_001).unwrap(), 10_001);
        assert_eq!(payment_cap_msat(1_000_000).unwrap(), 1_010_000);
        assert!(payment_cap_msat(u64::MAX).is_err());
        let rates = Pricing {
            input_msat_per_million: 1000,
            output_msat_per_million: 1000,
            minimum_invoice_msat: 1,
        };
        // 100 msat input + 100 msat output, each with the floor allowance.
        assert_eq!(
            rates.request_cap_msat(100, 100_000).unwrap(),
            2 * (100 + FEE_ALLOWANCE_FLOOR_MSAT)
        );
    }

    #[test]
    fn fee_allowance_is_monotone_so_charge_caps_fit_the_request_cap() {
        let mut previous = 0;
        for amount in [0, 1, 999, 1_000, 299_999, 300_000, 300_001, 10_000_000] {
            let allowance = fee_allowance_msat(amount).unwrap();
            assert!(allowance >= previous, "{amount}");
            previous = allowance;
        }
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
