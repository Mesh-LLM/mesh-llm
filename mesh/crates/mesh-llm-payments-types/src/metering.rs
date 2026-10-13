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
}

impl MeteredUsage {
    /// Accept only a complete, nonnegative integer usage report. Missing,
    /// malformed, or partial reports are unknown, not zero-token bills.
    pub fn from_response(response: &Value) -> Option<Self> {
        let usage = response.get("usage")?;
        let prompt_tokens = usage.get("prompt_tokens")?.as_u64()?;
        let completion_tokens = usage.get("completion_tokens")?.as_u64()?;
        prompt_tokens.checked_add(completion_tokens)?;
        Some(Self {
            prompt_tokens,
            completion_tokens,
        })
    }
}

/// Smallest Lightning payment the metered flow sends: 1 sat. Sub-sat
/// payments are not reliably routable (live: 60/90 msat found no route).
pub const PAYMENT_FLOOR_MSAT: u64 = 1_000;

/// Flat deposit asked of a peer outside the provider's trust cache. It only
/// proves the buyer can pay before backend work starts; it is not sized to
/// the request and is applied against the final bill.
pub const DEPOSIT_MSAT: u64 = PAYMENT_FLOOR_MSAT;

/// What the seller invoices after the deposit: `bill - deposit`, waived when
/// it is below [`PAYMENT_FLOOR_MSAT`] (the seller absorbs it).
pub fn final_invoice_msat(bill_msat: u64, deposit_applied_msat: u64) -> u64 {
    let remaining = bill_msat.saturating_sub(deposit_applied_msat);
    if remaining < PAYMENT_FLOOR_MSAT {
        0
    } else {
        remaining
    }
}

impl MeteredUsage {
    /// The bill for this usage at `pricing`: input plus output charges.
    pub fn bill_msat(&self, pricing: &crate::pricing::Pricing) -> Result<u64> {
        pricing
            .input_charge(self.prompt_tokens)?
            .checked_add(pricing.output_charge(self.completion_tokens)?)
            .ok_or_else(|| anyhow::anyhow!("bill overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_is_backend_reported_inclusive_and_may_be_zero() {
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":10,"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":15},"prompt_tokens_details":{"cached_tokens":5}}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 10,
                completion_tokens: 20
            })
        );
        assert_eq!(
            MeteredUsage::from_response(
                &json!({"usage":{"prompt_tokens":0,"completion_tokens":0}})
            ),
            Some(MeteredUsage {
                prompt_tokens: 0,
                completion_tokens: 0
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
