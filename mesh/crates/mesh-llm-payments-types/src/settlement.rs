//! Final billing from complete backend usage, with a paid deposit credited once.
//! Neither estimates nor delivered byte/chunk counts are final usage.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{metering::MeteredUsage, pricing::Pricing};

/// Invoice arithmetic only: no wallet reservation or authority to submit payment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalBill {
    pub usage: MeteredUsage,
    pub total_msat: u64,
    pub deposit_msat: u64,
    pub remaining_msat: u64,
}

impl FinalBill {
    /// Price inclusive backend counts and deduct the confirmed deposit.
    /// A deposit exceeding the total leaves no further invoice; it does not
    /// fabricate a negative bill or automatically initiate a refund.
    pub fn reconcile(pricing: &Pricing, usage: MeteredUsage, deposit_msat: u64) -> Result<Self> {
        let total_msat = pricing
            .input_charge(usage.prompt_tokens)?
            .checked_add(pricing.output_charge(usage.completion_tokens)?)
            .context("final bill overflow")?;
        Ok(Self {
            usage,
            total_msat,
            deposit_msat,
            remaining_msat: total_msat.saturating_sub(deposit_msat),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pricing() -> Pricing {
        Pricing {
            input_msat_per_million: 1_000_000,
            output_msat_per_million: 2_000_000,
        }
    }

    #[test]
    fn final_bill_uses_actual_input_and_output_and_credits_deposit_once() {
        let bill = FinalBill::reconcile(
            &pricing(),
            MeteredUsage {
                prompt_tokens: 33,
                completion_tokens: 10,
            },
            7,
        )
        .unwrap();
        assert_eq!(bill.total_msat, 53);
        assert_eq!(bill.remaining_msat, 46);
        assert_eq!(bill.deposit_msat + bill.remaining_msat, bill.total_msat);
    }

    #[test]
    fn repeat_buyer_without_deposit_pays_the_actual_total() {
        let bill = FinalBill::reconcile(
            &pricing(),
            MeteredUsage {
                prompt_tokens: 33,
                completion_tokens: 10,
            },
            0,
        )
        .unwrap();
        assert_eq!(bill.remaining_msat, 53);
    }

    #[test]
    fn deposit_covering_the_actual_bill_requires_no_second_payment() {
        for deposit in [53, 100] {
            assert_eq!(
                FinalBill::reconcile(
                    &pricing(),
                    MeteredUsage {
                        prompt_tokens: 33,
                        completion_tokens: 10
                    },
                    deposit
                )
                .unwrap()
                .remaining_msat,
                0
            );
        }
    }

    #[test]
    fn zero_usage_is_valid_and_overflow_fails_closed() {
        assert_eq!(
            FinalBill::reconcile(
                &pricing(),
                MeteredUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0
                },
                0
            )
            .unwrap()
            .total_msat,
            0
        );
        assert!(
            FinalBill::reconcile(
                &pricing(),
                MeteredUsage {
                    prompt_tokens: u64::MAX,
                    completion_tokens: 1
                },
                0
            )
            .is_err()
        );
    }
}
