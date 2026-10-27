//! Backend-neutral evidence for token billing: only backend-reported usage
//! is billed.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The two inclusive backend counters used for final billing. Reasoning and
/// cached-token breakdowns must not be added to these counters again.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default, skip_serializing_if = "crate::pricing::is_zero")]
    pub cached_tokens: u64,
}

impl MeteredUsage {
    /// Accept only a complete, nonnegative integer usage report. Missing,
    /// malformed, or partial reports are unknown, not zero-token bills.
    pub fn from_response(response: &Value) -> Option<Self> {
        let usage = response.get("usage")?;
        let prompt_tokens = usage.get("prompt_tokens")?.as_u64()?;
        let completion_tokens = usage.get("completion_tokens")?.as_u64()?;
        prompt_tokens.checked_add(completion_tokens)?;
        let cached_tokens = match usage.get("prompt_tokens_details") {
            None | Some(Value::Null) => 0,
            Some(details) => match details.as_object()?.get("cached_tokens") {
                None => 0,
                Some(value) => value.as_u64()?,
            },
        };
        if cached_tokens > prompt_tokens {
            return None;
        }
        Some(Self {
            prompt_tokens,
            completion_tokens,
            cached_tokens,
        })
    }
}

/// Default smallest Lightning payment the metered flow sends: 1 sat. Sub-sat
/// payments are not reliably routable (live: 60/90 msat found no route).
/// Operators may override it (`mesh-llm wallet min-payment`); the seller's
/// value travels in [`crate::metered_wire::AdmissionTerms`].
pub const PAYMENT_FLOOR_MSAT: u64 = 1_000;

/// Largest minimum payment or deposit an operator may configure (1000 sats),
/// so a typo cannot turn every tiny request into a large charge. A deposit is
/// kept even when it exceeds the bill, so this also bounds what one costs a buyer.
pub const MAX_PAYMENT_FLOOR_MSAT: u64 = 1_000_000;

/// Reject a minimum payment outside `1..=MAX_PAYMENT_FLOOR_MSAT` msat.
pub fn validate_payment_floor(floor_msat: u64) -> Result<()> {
    anyhow::ensure!(
        (1..=MAX_PAYMENT_FLOOR_MSAT).contains(&floor_msat),
        "minimum payment must be 1..={MAX_PAYMENT_FLOOR_MSAT} msat"
    );
    Ok(())
}

/// Reject a deposit outside `1..=MAX_PAYMENT_FLOOR_MSAT` msat. A seller that asks
/// no deposit states 0 in the admission terms instead.
pub fn validate_deposit(deposit_msat: u64) -> Result<()> {
    anyhow::ensure!(
        (1..=MAX_PAYMENT_FLOOR_MSAT).contains(&deposit_msat),
        "deposit must be 1..={MAX_PAYMENT_FLOOR_MSAT} msat"
    );
    Ok(())
}

/// What the seller invoices after the deposit: `bill - deposit`, exact to the
/// msat. Only a remainder below the minimum payment is rounded: every positive
/// remainder under `floor_msat` is raised to `floor_msat`.
/// Buyer and seller both derive the invoice from this one function.
pub fn final_invoice_msat(bill_msat: u64, deposit_applied_msat: u64, floor_msat: u64) -> u64 {
    let remaining = bill_msat.saturating_sub(deposit_applied_msat);
    match remaining {
        r if r >= floor_msat => r,
        r if r > 0 => floor_msat,
        _ => 0,
    }
}

impl MeteredUsage {
    /// The bill for this usage at `pricing`: input plus output charges.
    pub fn bill_msat(&self, pricing: &crate::pricing::Pricing) -> Result<u64> {
        pricing
            .cached_input_charge(self.prompt_tokens, self.cached_tokens)?
            .checked_add(pricing.output_charge(self.completion_tokens)?)
            .ok_or_else(|| anyhow::anyhow!("bill overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn final_invoice_is_exact_and_only_rounds_a_sub_sat_remainder() {
        assert_eq!(final_invoice_msat(0, 0, PAYMENT_FLOOR_MSAT), 0);
        assert_eq!(final_invoice_msat(1, 0, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_001, 1_000, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_000, 1_000, PAYMENT_FLOOR_MSAT), 0);
        assert_eq!(final_invoice_msat(499, 0, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(500, 0, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(999, 0, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_000, 0, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_499, 0, PAYMENT_FLOOR_MSAT), 1_499);
        assert_eq!(final_invoice_msat(4_045, 0, PAYMENT_FLOOR_MSAT), 4_045);
        assert_eq!(final_invoice_msat(1_499, 1_000, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_500, 1_000, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(1_640, 1_000, PAYMENT_FLOOR_MSAT), 1_000);
        assert_eq!(final_invoice_msat(2_640, 1_000, PAYMENT_FLOOR_MSAT), 1_640);
        assert_eq!(final_invoice_msat(500, 1_000, PAYMENT_FLOOR_MSAT), 0);
    }

    #[test]
    fn an_overridden_floor_rounds_the_same_way_and_is_bounded() {
        assert_eq!(final_invoice_msat(1, 0, 2_000), 2_000);
        assert_eq!(final_invoice_msat(2_500, 1_000, 2_000), 2_000);
        assert_eq!(final_invoice_msat(4_000, 2_000, 2_000), 2_000);
        assert_eq!(final_invoice_msat(1_500, 0, 1), 1_500);
        assert!(validate_payment_floor(0).is_err());
        assert!(validate_payment_floor(MAX_PAYMENT_FLOOR_MSAT + 1).is_err());
        assert!(validate_payment_floor(1).is_ok());
        assert_eq!(
            final_invoice_msat(u64::MAX, 0, PAYMENT_FLOOR_MSAT),
            u64::MAX
        );
    }

    #[test]
    fn usage_is_backend_reported_inclusive_and_may_be_zero() {
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":10,"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":15},"prompt_tokens_details":{"cached_tokens":5}}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                cached_tokens: 5,
            })
        );
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":0,"completion_tokens":0}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                cached_tokens: 0,
            })
        );
        for value in [
            json!({}),
            json!({"usage":null}),
            json!({"usage":{"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":-1,"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":1.5,"completion_tokens":10}}),
            json!({"usage":{"prompt_tokens":u64::MAX,"completion_tokens":1}}),
        ] {
            assert_eq!(MeteredUsage::from_response(&value), None);
        }
    }
}
