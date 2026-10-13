//! Provider-side ledger for metered (`/mesh/payment/v2`) requests.
//!
//! One lifecycle per request: `admitted` (deposit stated, backend untouched)
//! → `billed` (backend-reported usage priced at the admitted rates, deposit
//! applied, remainder on one final invoice) or `abandoned` (no usage, no bill,
//! no debt). There are no reservations, exposure sums or credits: the deposit
//! is a share of the estimated input cost, a covering deposit means no second
//! invoice, and nothing is refunded or carried forward.
//!
//! The deposit and final invoices reuse `receivables` as segments 0 and 1.
//! A buyer is blocked only by [`STRIKE_LIMIT`] distinct issued and unpaid
//! final invoices; deposits, retries and abandoned requests are not strikes.

use anyhow::{Result, bail, ensure};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use super::{Ledger, read_amount, sql_amount};
use crate::pricing::Pricing;
use mesh_llm_payments_types::metering::MeteredUsage;

/// Deposit share of the estimated input cost, in tenths.
const DEPOSIT_TENTHS: u128 = 3;

/// Distinct issued, unpaid final invoices at which a buyer is refused.
pub const STRIKE_LIMIT: u64 = 3;

const UNPAID_FINALS: &str = "SELECT COUNT(*) FROM metered_requests m \
    WHERE m.peer=?1 AND m.state='billed' AND EXISTS(\
    SELECT 1 FROM receivables WHERE request_id=m.id AND segment=1 AND state!='paid')";

/// Admission inputs. The deposit is 0.3 x estimated input cost when
/// `require_deposit`; the provider skips it for peers in its trust cache.
#[derive(Clone, Copy, Debug)]
pub struct MeteredRequest<'a> {
    pub id: &'a str,
    pub peer: &'a str,
    pub pricing: &'a Pricing,
    pub estimated_input_tokens: u64,
    pub require_deposit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MeteredOutcome {
    /// Complete backend usage. Bills at the rates frozen at admission.
    Usage(MeteredUsage),
    /// No authoritative usage: backend failure, buyer disconnect, cancelled
    /// before dispatch, or a response without a `usage` object. No bill, no
    /// invoice, no debt; a received deposit stays received.
    Abandoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeteredStatus {
    Admitted,
    /// Bill recorded; `remaining_msat` is still owed on the final invoice.
    Billed,
    /// Nothing owed: the deposit covered the bill, or the final invoice is paid.
    Settled,
    Abandoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredSettlement {
    pub status: MeteredStatus,
    pub usage: Option<MeteredUsage>,
    /// `None` unless usage was reported. Never a zero bill for unknown usage.
    pub bill_msat: Option<u64>,
    pub deposit_paid_msat: u64,
    /// Amount the final invoice (segment 1) must carry; 0 means none.
    pub remaining_msat: u64,
}

impl Ledger {
    /// Record one metered admission in a single IMMEDIATE txn and return the
    /// deposit to request. Rates are frozen here; `metered_finalize` accepts
    /// usage only. Replaying the same arguments is idempotent; reusing the ID
    /// with different terms fails. Refuses peers at the strike limit.
    pub fn metered_admit(&self, request: &MeteredRequest<'_>) -> Result<u64> {
        let MeteredRequest {
            id,
            peer,
            pricing,
            estimated_input_tokens,
            require_deposit,
        } = *request;
        let input = pricing.input_charge(estimated_input_tokens)?;
        let deposit = if require_deposit {
            u64::try_from((u128::from(input) * DEPOSIT_TENTHS).div_ceil(10))?
        } else {
            0
        };
        let pricing_json = serde_json::to_string(pricing)?;
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Identity is the exact admitted terms, not the rounded msat amounts:
        // different token terms can round to the same charge at low rates.
        let existing: Option<(String, String, u64, bool)> = tx
            .query_row(
                "SELECT peer,pricing,estimated_input_tokens,require_deposit FROM metered_requests WHERE id=?",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, read_amount(r, 2)?, r.get(3)?)),
            )
            .optional()?;
        if let Some(row) = existing {
            ensure!(
                row == (
                    peer.to_owned(),
                    pricing_json,
                    estimated_input_tokens,
                    require_deposit
                ),
                "request identity reused with different metered terms"
            );
            return Ok(deposit);
        }
        let unpaid: u64 = tx.query_row(UNPAID_FINALS, [peer], |r| read_amount(r, 0))?;
        ensure!(
            unpaid < STRIKE_LIMIT,
            "buyer has {unpaid} unpaid final invoices"
        );
        tx.execute(
            "INSERT INTO serving_requests(id,peer) VALUES (?1,?2)",
            params![id, peer],
        )?;
        tx.execute(
            "INSERT INTO metered_requests(id,peer,pricing,estimated_input_tokens,require_deposit,deposit_msat,state) VALUES (?1,?2,?3,?4,?5,?6,'admitted')",
            params![
                id,
                peer,
                pricing_json,
                sql_amount(estimated_input_tokens)?,
                require_deposit,
                sql_amount(deposit)?
            ],
        )?;
        tx.commit()?;
        Ok(deposit)
    }

    /// Finalize once, atomically: price the reported usage, apply the received
    /// deposit and record what the final invoice must carry. Replaying the same
    /// outcome returns the stored settlement; a different outcome fails.
    pub fn metered_finalize(&self, id: &str, outcome: MeteredOutcome) -> Result<MeteredSettlement> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (pricing, deposit, state, stored_outcome): (String, u64, String, Option<String>) = tx
            .query_row(
                "SELECT pricing,deposit_msat,state,outcome FROM metered_requests WHERE id=?",
                [id],
                |r| Ok((r.get(0)?, read_amount(r, 1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("unknown metered request"))?;
        let outcome_json = serde_json::to_string(&outcome)?;
        if state != "admitted" {
            ensure!(
                stored_outcome.as_deref() == Some(outcome_json.as_str()),
                "metered request already finalized differently"
            );
            drop(tx);
            drop(connection);
            return self.metered_settlement(id);
        }
        // An unpaid deposit contributes nothing: a bill is never reduced by
        // money not received.
        let deposit_paid = if deposit > 0 && deposit_paid(&tx, id)? {
            deposit
        } else {
            0
        };
        let pricing: Pricing = serde_json::from_str(&pricing)?;
        let (state, bill, remaining) = match outcome {
            MeteredOutcome::Usage(usage) => {
                let bill = pricing
                    .input_charge(usage.prompt_tokens)?
                    .checked_add(pricing.output_charge(usage.completion_tokens)?)
                    .ok_or_else(|| anyhow::anyhow!("bill overflow"))?;
                ("billed", Some(bill), bill.saturating_sub(deposit_paid))
            }
            MeteredOutcome::Abandoned => ("abandoned", None, 0),
        };
        let changed = tx.execute(
            "UPDATE metered_requests SET state=?2,outcome=?3,bill_msat=?4,deposit_paid_msat=?5,remaining_msat=?6 WHERE id=?1 AND state='admitted'",
            params![
                id,
                state,
                outcome_json,
                bill.map(sql_amount).transpose()?,
                sql_amount(deposit_paid)?,
                sql_amount(remaining)?
            ],
        )?;
        ensure!(changed == 1, "metered finalize raced");
        tx.commit()?;
        drop(connection);
        self.metered_settlement(id)
    }

    /// Current settlement. `Billed` becomes `Settled` once the segment-1
    /// receivable is marked paid; there is no separate transition to race.
    pub fn metered_settlement(&self, id: &str) -> Result<MeteredSettlement> {
        let connection = self.lock()?;
        let row = connection.query_row(
            "SELECT m.state,m.outcome,m.bill_msat,m.deposit_paid_msat,m.remaining_msat, EXISTS(SELECT 1 FROM receivables WHERE request_id=m.id AND segment=1 AND state='paid') FROM metered_requests m WHERE m.id=?",
            [id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    read_amount(r, 3)?,
                    read_amount(r, 4)?,
                    r.get::<_, bool>(5)?,
                ))
            },
        )?;
        let (state, outcome, bill, deposit_paid_msat, remaining, final_paid) = row;
        let usage = match outcome.as_deref().map(serde_json::from_str).transpose()? {
            Some(MeteredOutcome::Usage(usage)) => Some(usage),
            _ => None,
        };
        let status = match state.as_str() {
            "admitted" => MeteredStatus::Admitted,
            "billed" if remaining == 0 || final_paid => MeteredStatus::Settled,
            "billed" => MeteredStatus::Billed,
            "abandoned" => MeteredStatus::Abandoned,
            other => bail!("unknown metered state {other}"),
        };
        Ok(MeteredSettlement {
            status,
            usage,
            bill_msat: bill.map(u64::try_from).transpose()?,
            deposit_paid_msat,
            remaining_msat: remaining,
        })
    }

    /// Durable once-only dispatch authorization. Admission is replayable, so
    /// it alone must never authorize (re)generating a request. Returns `true`
    /// to exactly one caller; later calls (including after restart) get
    /// `false`. Fails for an unknown ID, another peer, or a finalized row.
    pub fn metered_claim_dispatch(&self, id: &str, peer: &str) -> Result<bool> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (owner, state, dispatched): (String, String, bool) = tx
            .query_row(
                "SELECT peer,state,dispatched FROM metered_requests WHERE id=?",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("unknown metered request"))?;
        ensure!(owner == peer, "metered request owned by another peer");
        if dispatched {
            return Ok(false);
        }
        ensure!(state == "admitted", "metered request already finalized");
        tx.execute("UPDATE metered_requests SET dispatched=1 WHERE id=?", [id])?;
        tx.commit()?;
        Ok(true)
    }

    /// Owning peer, for authenticated recovery.
    pub fn metered_owner(&self, id: &str) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .query_row("SELECT peer FROM metered_requests WHERE id=?", [id], |r| {
                r.get(0)
            })
            .optional()?)
    }

    /// Startup-only: run after the exclusive service lock, before serving.
    /// Rows left `admitted` by a crash are abandoned: whatever the backend
    /// did, no usage was recorded, so there is no bill.
    pub fn close_interrupted_metered(&self) -> Result<()> {
        let rows: Vec<String> = {
            let connection = self.lock()?;
            let mut statement =
                connection.prepare("SELECT id FROM metered_requests WHERE state='admitted'")?;
            statement
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        for id in rows {
            self.metered_finalize(&id, MeteredOutcome::Abandoned)?;
        }
        Ok(())
    }

    /// Distinct issued and unpaid final invoices for `peer`.
    pub fn metered_unpaid_finals(&self, peer: &str) -> Result<u64> {
        Ok(self
            .lock()?
            .query_row(UNPAID_FINALS, [peer], |r| read_amount(r, 0))?)
    }
}

/// Refuse a metered receivable that does not match current state: segment 0
/// must be the admitted deposit on a still-admitted row; segment 1 must equal
/// the `remaining_msat` of a billed row. Non-metered rows pass through.
pub(super) fn check_metered_receivable(
    tx: &Transaction<'_>,
    receipt: &super::receivables::Receivable,
) -> Result<()> {
    let row: Option<(String, u64, u64)> = tx
        .query_row(
            "SELECT state,deposit_msat,remaining_msat FROM metered_requests WHERE id=?",
            [&receipt.request_id],
            |r| Ok((r.get(0)?, read_amount(r, 1)?, read_amount(r, 2)?)),
        )
        .optional()?;
    let Some((state, deposit, remaining)) = row else {
        return Ok(());
    };
    let amount = receipt.invoice.amount_msat;
    match receipt.segment {
        0 => ensure!(
            state == "admitted" && deposit > 0 && amount == Some(deposit),
            "metered deposit invoice does not match admitted deposit"
        ),
        1 => ensure!(
            state == "billed" && remaining > 0 && amount == Some(remaining),
            "metered final invoice must equal the billed remainder"
        ),
        _ => bail!("metered requests have only segments 0 and 1"),
    }
    Ok(())
}

fn deposit_paid(tx: &Transaction<'_>, id: &str) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM receivables WHERE request_id=? AND segment=0 AND state='paid')",
        [id],
        |r| r.get(0),
    )?)
}
