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
    /// Legacy v0.78.x invoice quantum, kept only for mixed-version wire
    /// compatibility. Sellers before #2310 require this field in serde payment
    /// frames, compare it for exact pricing equality, and round each nonzero
    /// charge up to a multiple of it. Current sellers never configure it: they
    /// always store and send 1, which leaves charges exact. A buyer carries the
    /// value a legacy seller advertised (gossip field 4) so its request,
    /// terms, and charge checks match that seller. Not user configurable.
    #[serde(default = "legacy_minimum_default")]
    pub minimum_invoice_msat: u64,
}

/// Value meaning "no legacy rounding"; also what pre-#2310 readers require.
pub const LEGACY_MINIMUM_NONE: u64 = 1;

fn legacy_minimum_default() -> u64 {
    LEGACY_MINIMUM_NONE
}

impl Pricing {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.input_msat_per_million > 0 && self.output_msat_per_million > 0,
            "paid serving requires positive input and output rates"
        );
        ensure!(
            self.minimum_invoice_msat > 0,
            "legacy minimum invoice must be positive"
        );
        Ok(())
    }

    /// Exact rates with no legacy rounding.
    pub fn exact(input_msat_per_million: u64, output_msat_per_million: u64) -> Self {
        Self {
            input_msat_per_million,
            output_msat_per_million,
            minimum_invoice_msat: LEGACY_MINIMUM_NONE,
        }
    }

    /// The same rates without any legacy quantum, for prices this node sets
    /// and serves itself.
    pub fn without_legacy_minimum(mut self) -> Self {
        self.minimum_invoice_msat = LEGACY_MINIMUM_NONE;
        self
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
        let rates = Pricing::exact(500, 1500);
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
        let rates = Pricing::exact(1000, 1000);
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
    fn charges_round_up_to_the_msat_and_overflow_is_enforced() {
        let mut rates = Pricing::exact(1_000_001, 1);
        assert_eq!(rates.input_charge(1000).unwrap(), 1001);
        rates.input_msat_per_million = u64::MAX;
        assert!(rates.input_charge(u64::MAX).is_err());
    }

    #[test]
    fn legacy_minimum_rounds_like_v0781_and_defaults_to_exact() {
        let mut rates = Pricing::exact(500, 1500);
        assert_eq!(rates.output_charge(1000).unwrap(), 2);
        rates.minimum_invoice_msat = 1000;
        assert_eq!(rates.input_charge(1000).unwrap(), 1000);
        assert_eq!(rates.output_charge(1_000_000).unwrap(), 2000);
        assert_eq!(rates.output_charge(0).unwrap(), 0);
        rates.minimum_invoice_msat = 0;
        assert!(rates.validate().is_err());
        assert_eq!(
            rates.clone().without_legacy_minimum(),
            Pricing::exact(500, 1500)
        );
    }

    /// Exactly the v0.78.1 `Pricing` shape: every field required.
    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(deny_unknown_fields)]
    struct PricingV0781 {
        input_msat_per_million: u64,
        output_msat_per_million: u64,
        minimum_invoice_msat: u64,
    }

    #[test]
    fn pricing_json_interoperates_with_v0781() {
        for minimum in [1, 1000] {
            let mut current = Pricing::exact(500, 1500);
            current.minimum_invoice_msat = minimum;
            let json = serde_json::to_string(&current).unwrap();
            let old: PricingV0781 = serde_json::from_str(&json).unwrap();
            assert_eq!(old.minimum_invoice_msat, minimum);
            let back: Pricing =
                serde_json::from_str(&serde_json::to_string(&old).unwrap()).unwrap();
            assert_eq!(back, current);
        }
        // #2310-era JSON without the field reads as exact pricing.
        let bare: Pricing =
            serde_json::from_str(r#"{"input_msat_per_million":5,"output_msat_per_million":6}"#)
                .unwrap();
        assert_eq!(bare, Pricing::exact(5, 6));
    }
}
