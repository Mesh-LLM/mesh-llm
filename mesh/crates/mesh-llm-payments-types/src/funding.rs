//! Rolling funding policy for one paid request: deposit → input checkpoint →
//! output top-ups → final reconciliation.
//!
//! This is the *payment policy* of the metering design, kept pure so the
//! seller and the payer can both evaluate it from the same facts and agree.
//! It knows nothing about wallets, transports or backends: the caller feeds
//! it token counts (from whichever meter the backend supports) and funding
//! events, and asks what to do next.
//!
//! Vocabulary:
//! - **funded**: msat the payer has paid or has an accepted invoice for.
//! - **exposure**: msat of work already performed/delivered but not funded.
//!   Bounded by [`FundingPolicy::max_exposure_msat`]; beyond it the caller
//!   must pause (decode or delivery, whichever it controls).
//! - **segments**: invoice numbering. `0` is the admission/input invoice,
//!   `1..` are rolling top-ups; the final reconciliation reuses the next
//!   free segment. The ledger already keys receivables and charges by
//!   `(request_id, segment)`, so no schema change is needed.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::pricing::Pricing;

/// Operator-tunable knobs. Defaults are deliberately small so a payer's
/// unfunded exposure per request stays in the "tiny deposit" regime.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FundingPolicy {
    /// Output tokens each top-up invoice covers.
    pub top_up_tokens: u64,
    /// Ask for the next top-up when funded headroom drops to this many
    /// output tokens, so payment latency overlaps generation.
    pub top_up_lead_tokens: u64,
    /// Work the seller will perform ahead of funding before pausing.
    pub max_exposure_msat: u64,
}

impl Default for FundingPolicy {
    fn default() -> Self {
        Self {
            top_up_tokens: 512,
            top_up_lead_tokens: 128,
            max_exposure_msat: 0,
        }
    }
}

/// What the caller should do now. At most one action is pending at a time;
/// the caller reports the outcome back via [`Funding::invoice_issued`] /
/// [`Funding::funded`] before asking again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Nothing to do: keep serving.
    Continue,
    /// Issue invoice `segment` for `amount_msat` covering `tokens` more output
    /// tokens (or the input, for segment 0).
    Invoice {
        segment: u32,
        tokens: u64,
        amount_msat: u64,
    },
    /// Exposure limit reached and an invoice is outstanding: hold work until
    /// it is funded (or give up per the caller's timeout policy).
    Pause { outstanding_segment: u32 },
}

/// Final reconciliation once usage is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settlement {
    /// Everything already funded covers the bill exactly.
    Settled,
    /// The payer still owes this much: issue a final invoice.
    Owed { segment: u32, amount_msat: u64 },
    /// The payer funded more than the bill: credit/refund per contract.
    Overpaid { amount_msat: u64 },
}

/// Per-request funding state. One instance per paid request on each side.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Funding {
    pricing: Pricing,
    policy: FundingPolicy,
    /// Exact or estimated prompt tokens once known.
    input_tokens: Option<u64>,
    /// Output tokens covered by funded invoices (segments >= 1).
    funded_output_tokens: u64,
    /// msat of all funded invoices (input included).
    funded_msat: u64,
    /// An invoice the payer has not yet funded.
    outstanding: Option<Outstanding>,
    next_segment: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Outstanding {
    segment: u32,
    tokens: u64,
    amount_msat: u64,
}

impl Funding {
    pub fn new(pricing: Pricing, policy: FundingPolicy) -> Result<Self> {
        pricing.validate()?;
        ensure!(policy.top_up_tokens > 0, "top-up size must be positive");
        ensure!(
            policy.top_up_lead_tokens < policy.top_up_tokens,
            "top-up lead must be smaller than the top-up size"
        );
        Ok(Self {
            pricing,
            policy,
            input_tokens: None,
            funded_output_tokens: 0,
            funded_msat: 0,
            outstanding: None,
            next_segment: 0,
        })
    }

    pub fn funded_msat(&self) -> u64 {
        self.funded_msat
    }

    /// Output tokens the payer has funded so far. A decode gate pauses at
    /// this watermark (plus whatever run-ahead the exposure limit allows).
    pub fn funded_output_tokens(&self) -> u64 {
        self.funded_output_tokens
    }

    /// The meter reported the prompt size (exact after prefill, or an
    /// estimate before it). Idempotent for the same value.
    pub fn input_counted(&mut self, tokens: u64) -> Result<()> {
        ensure!(tokens > 0, "empty prompt");
        if let Some(previous) = self.input_tokens {
            ensure!(previous == tokens, "input count changed after invoicing");
        }
        self.input_tokens = Some(tokens);
        Ok(())
    }

    /// Decide the next step given how much output the backend has committed
    /// (decode-side) or the transport has delivered (delivery-side). Callers
    /// pass whichever watermark they gate on.
    pub fn next(&mut self, output_tokens: u64) -> Result<Action> {
        if let Some(outstanding) = &self.outstanding {
            // One invoice at a time. Keep working until exposure is spent.
            let exposure = self.exposure_msat(output_tokens)?;
            return Ok(if exposure > self.policy.max_exposure_msat {
                Action::Pause {
                    outstanding_segment: outstanding.segment,
                }
            } else {
                Action::Continue
            });
        }
        if self.next_segment == 0 {
            // Segment 0: the input checkpoint. Nothing to invoice until the
            // meter has a count; a caller wanting a flat admission deposit
            // passes its estimate here and reconciles at the end.
            let Some(input) = self.input_tokens else {
                return Ok(Action::Continue);
            };
            let amount_msat = self.pricing.input_charge(input)?;
            return Ok(self.issue(input, amount_msat));
        }
        let headroom = self.funded_output_tokens.saturating_sub(output_tokens);
        if headroom <= self.policy.top_up_lead_tokens {
            let tokens = self.policy.top_up_tokens;
            let amount_msat = self.pricing.output_charge(tokens)?;
            return Ok(self.issue(tokens, amount_msat));
        }
        Ok(Action::Continue)
    }

    fn issue(&mut self, tokens: u64, amount_msat: u64) -> Action {
        let segment = self.next_segment;
        self.outstanding = Some(Outstanding {
            segment,
            tokens,
            amount_msat,
        });
        Action::Invoice {
            segment,
            tokens,
            amount_msat,
        }
    }

    /// The outstanding invoice was funded (arrival seen by the receiver, or
    /// paid by the payer). Returns the funded segment.
    pub fn funded(&mut self, segment: u32) -> Result<u32> {
        let outstanding = self.outstanding.take().context("no outstanding invoice")?;
        ensure!(outstanding.segment == segment, "funded the wrong segment");
        self.funded_msat = self
            .funded_msat
            .checked_add(outstanding.amount_msat)
            .context("funded total overflow")?;
        if segment > 0 {
            self.funded_output_tokens = self
                .funded_output_tokens
                .checked_add(outstanding.tokens)
                .context("funded tokens overflow")?;
        }
        self.next_segment = segment.checked_add(1).context("segment overflow")?;
        Ok(segment)
    }

    /// msat of work done beyond what is funded. Output already covered by
    /// an outstanding (unpaid) invoice still counts as exposure.
    pub fn exposure_msat(&self, output_tokens: u64) -> Result<u64> {
        let billed = self.bill_msat(output_tokens)?;
        Ok(billed.saturating_sub(self.funded_msat))
    }

    /// The token bill at agreed rates for the usage so far.
    pub fn bill_msat(&self, output_tokens: u64) -> Result<u64> {
        let input = self.pricing.input_charge(self.input_tokens.unwrap_or(0))?;
        let output = self.pricing.output_charge(output_tokens)?;
        input.checked_add(output).context("bill overflow")
    }

    /// Final reconciliation from the backend's final usage. Any outstanding
    /// invoice is abandoned (the final invoice supersedes it) so a payer is
    /// never asked to pay for the same tokens twice.
    pub fn settle(
        &mut self,
        final_input_tokens: u64,
        final_output_tokens: u64,
    ) -> Result<Settlement> {
        self.outstanding = None;
        self.input_tokens = Some(final_input_tokens);
        let bill = self.bill_msat(final_output_tokens)?;
        Ok(match bill.cmp(&self.funded_msat) {
            std::cmp::Ordering::Equal => Settlement::Settled,
            std::cmp::Ordering::Greater => Settlement::Owed {
                segment: self.next_segment,
                amount_msat: bill - self.funded_msat,
            },
            std::cmp::Ordering::Less => Settlement::Overpaid {
                amount_msat: self.funded_msat - bill,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pricing() -> Pricing {
        Pricing {
            input_msat_per_million: 1_000_000,  // 1 msat/token
            output_msat_per_million: 2_000_000, // 2 msat/token
            minimum_invoice_msat: 1,
        }
    }

    fn funding(policy: FundingPolicy) -> Funding {
        Funding::new(pricing(), policy).unwrap()
    }

    #[test]
    fn input_checkpoint_then_rolling_top_ups_then_exact_settlement() {
        let mut f = funding(FundingPolicy {
            top_up_tokens: 100,
            top_up_lead_tokens: 20,
            max_exposure_msat: 0,
        });
        // Before any count there is nothing to invoice.
        assert_eq!(f.next(0).unwrap(), Action::Continue);
        f.input_counted(40).unwrap();
        assert_eq!(
            f.next(0).unwrap(),
            Action::Invoice {
                segment: 0,
                tokens: 40,
                amount_msat: 40
            }
        );
        // Outstanding and zero exposure allowed: any output pauses.
        assert_eq!(
            f.next(0).unwrap(),
            Action::Pause {
                outstanding_segment: 0
            }
        );
        f.funded(0).unwrap();
        // Headroom is zero, so the first top-up is issued immediately.
        assert_eq!(
            f.next(0).unwrap(),
            Action::Invoice {
                segment: 1,
                tokens: 100,
                amount_msat: 200
            }
        );
        f.funded(1).unwrap();
        assert_eq!(f.funded_output_tokens(), 100);
        // Plenty of headroom: continue. At the lead mark: next invoice.
        assert_eq!(f.next(50).unwrap(), Action::Continue);
        assert_eq!(
            f.next(80).unwrap(),
            Action::Invoice {
                segment: 2,
                tokens: 100,
                amount_msat: 200
            }
        );
        // Still inside funded tokens while invoice 2 is outstanding: no pause.
        assert_eq!(f.next(99).unwrap(), Action::Continue);
        // Past funded output with zero exposure allowed: pause.
        assert_eq!(
            f.next(101).unwrap(),
            Action::Pause {
                outstanding_segment: 2
            }
        );
        f.funded(2).unwrap();
        assert_eq!(f.funded_output_tokens(), 200);
        assert_eq!(f.funded_msat(), 40 + 200 + 200);
        // Backend finished at 150 output tokens: bill 40 + 300 = 340, funded 440.
        assert_eq!(
            f.settle(40, 150).unwrap(),
            Settlement::Overpaid { amount_msat: 100 }
        );
    }

    #[test]
    fn exposure_allowance_lets_work_run_ahead_of_an_outstanding_invoice() {
        let mut f = funding(FundingPolicy {
            top_up_tokens: 100,
            top_up_lead_tokens: 10,
            max_exposure_msat: 50, // 25 output tokens at 2 msat each
        });
        f.input_counted(10).unwrap();
        assert!(matches!(
            f.next(0).unwrap(),
            Action::Invoice { segment: 0, .. }
        ));
        // Input (10 msat) unfunded + 20 output tokens (40 msat) = 50: at limit.
        assert_eq!(f.next(20).unwrap(), Action::Continue);
        assert_eq!(
            f.next(21).unwrap(),
            Action::Pause {
                outstanding_segment: 0
            }
        );
        f.funded(0).unwrap();
        assert!(matches!(
            f.next(21).unwrap(),
            Action::Invoice { segment: 1, .. }
        ));
        // 21 output tokens = 42 msat exposure, under 50.
        assert_eq!(f.next(25).unwrap(), Action::Continue);
        assert_eq!(
            f.next(26).unwrap(),
            Action::Pause {
                outstanding_segment: 1
            }
        );
    }

    #[test]
    fn settlement_bills_the_unfunded_remainder_on_a_fresh_segment() {
        let mut f = funding(FundingPolicy {
            top_up_tokens: 100,
            top_up_lead_tokens: 10,
            max_exposure_msat: 1_000,
        });
        f.input_counted(40).unwrap();
        f.next(0).unwrap();
        f.funded(0).unwrap();
        f.next(0).unwrap(); // segment 1 issued, never funded (payer slow)
        // Generation ended at 130 output tokens inside the exposure allowance.
        assert_eq!(
            f.settle(40, 130).unwrap(),
            Settlement::Owed {
                segment: 1,
                amount_msat: 260
            }
        );
        // An outstanding invoice is superseded, never double-billed.
        assert!(f.funded(1).is_err());
    }

    #[test]
    fn settlement_is_exact_when_funding_matches_the_bill() {
        let mut f = funding(FundingPolicy {
            top_up_tokens: 50,
            top_up_lead_tokens: 5,
            max_exposure_msat: 0,
        });
        f.input_counted(10).unwrap();
        f.next(0).unwrap();
        f.funded(0).unwrap();
        f.next(0).unwrap();
        f.funded(1).unwrap();
        assert_eq!(f.settle(10, 50).unwrap(), Settlement::Settled);
    }

    #[test]
    fn invalid_policies_and_counts_are_refused() {
        assert!(
            Funding::new(
                pricing(),
                FundingPolicy {
                    top_up_tokens: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            Funding::new(
                pricing(),
                FundingPolicy {
                    top_up_tokens: 10,
                    top_up_lead_tokens: 10,
                    max_exposure_msat: 0
                }
            )
            .is_err()
        );
        let mut f = funding(FundingPolicy::default());
        assert!(f.input_counted(0).is_err());
        f.input_counted(5).unwrap();
        assert!(f.input_counted(6).is_err());
        assert!(f.funded(0).is_err(), "nothing outstanding");
        f.next(0).unwrap();
        assert!(f.funded(1).is_err(), "wrong segment");
    }
}
