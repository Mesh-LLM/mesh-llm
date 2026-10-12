//! Provider-side ledger for metered (`/mesh/payment/v2`) requests.
//!
//! Admission reserves exposure (estimated input + bounded output at frozen
//! rates) and states an optional deposit of 0.3 x estimated input cost. The
//! deposit and final invoices reuse `receivables` as segments 0 and 1, but
//! metered requests are excluded from every legacy recovery and blocking query:
//! their exposure is accounted here instead.
//!
//! Exposure per peer is derived, never stored, so restart needs no rebuild:
//! - `reserved` and `usage_missing` rows hold their reservation;
//! - `billed` rows hold the unpaid remainder until the final invoice is paid;
//! - `backend_error` rows and paid `billed` rows hold nothing.

use anyhow::{Result, bail, ensure};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use super::{Ledger, read_amount, sql_amount};
use crate::pricing::Pricing;
use mesh_llm_payments_types::metering::MeteredUsage;

/// Deposit share of the estimated input cost, in tenths.
const DEPOSIT_TENTHS: u128 = 3;

/// Exposure SQL for one peer, shared by admission and status.
const PEER_EXPOSURE: &str = "SELECT COALESCE(SUM(CASE \
    WHEN m.state IN ('reserved','usage_missing') THEN m.reserved_msat \
    WHEN m.state='billed' AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=m.id AND segment=1 AND state='paid') THEN m.remaining_msat \
    ELSE 0 END),0) FROM metered_requests m WHERE m.peer=?1";

/// Admission inputs. The deposit is 0.3 x estimated input cost when
/// `require_deposit`; the provider skips it for peers in its trust cache.
#[derive(Clone, Copy, Debug)]
pub struct MeteredRequest<'a> {
    pub id: &'a str,
    pub peer: &'a str,
    pub pricing: &'a Pricing,
    pub estimated_input_tokens: u64,
    pub max_output_tokens: u64,
    pub peer_cap_msat: u64,
    pub require_deposit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredAdmission {
    /// Exposure counted against the peer cap. Never charged.
    pub reserved_msat: u64,
    /// Upfront payment required before dispatch; 0 for trusted peers.
    pub deposit_msat: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MeteredOutcome {
    /// Complete backend usage. Bills at the rates frozen at admission.
    Usage(MeteredUsage),
    /// Backend failed, or the request was admitted but never dispatched:
    /// no bill, any paid deposit becomes peer credit, exposure released.
    BackendError,
    /// Response delivered but usage unknown. Left unresolved for
    /// reconciliation: no bill, no forfeiture, no credit, reservation held.
    /// A later `Usage` or `BackendError` finalize resolves it exactly once.
    UsageMissing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeteredStatus {
    Reserved,
    /// Bill recorded; `remaining_msat` is still owed on the final invoice.
    Billed,
    /// Nothing owed: remaining was zero, or the final invoice is paid.
    Settled,
    BackendError,
    UsageMissing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredSettlement {
    pub status: MeteredStatus,
    pub usage: Option<MeteredUsage>,
    /// `None` unless usage was reported. Never a zero bill for unknown usage.
    pub bill_msat: Option<u64>,
    pub deposit_paid_msat: u64,
    /// Existing peer credit consumed by this request.
    pub credit_applied_msat: u64,
    /// Peer credit created by this request (overpaid or refunded deposit).
    pub credit_created_msat: u64,
    /// Amount the final invoice (segment 1) must carry; 0 means none.
    pub remaining_msat: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeteredOwner {
    pub peer: String,
    pub dispatched: bool,
}

impl Ledger {
    /// Reserve exposure for one metered request in a single IMMEDIATE txn.
    /// Rates are frozen here; `finalize` accepts usage only. Replaying the
    /// same arguments is idempotent; reusing the ID differently fails.
    pub fn metered_admit(&self, request: &MeteredRequest<'_>) -> Result<MeteredAdmission> {
        let MeteredRequest {
            id,
            peer,
            pricing,
            estimated_input_tokens,
            max_output_tokens,
            peer_cap_msat,
            require_deposit,
        } = *request;
        ensure!(
            max_output_tokens > 0,
            "finite positive output allowance required"
        );
        let input = pricing.input_charge(estimated_input_tokens)?;
        let reserved = input
            .checked_add(pricing.output_charge(max_output_tokens)?)
            .ok_or_else(|| anyhow::anyhow!("reservation overflow"))?;
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
        let existing: Option<(String, String, u64, u64, bool)> = tx
            .query_row(
                "SELECT peer,pricing,estimated_input_tokens,max_output_tokens,require_deposit FROM metered_requests WHERE id=?",
                [id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        read_amount(r, 2)?,
                        read_amount(r, 3)?,
                        r.get(4)?,
                    ))
                },
            )
            .optional()?;
        if let Some(row) = existing {
            ensure!(
                row == (
                    peer.to_owned(),
                    pricing_json,
                    estimated_input_tokens,
                    max_output_tokens,
                    require_deposit
                ),
                "request identity reused with different metered terms"
            );
            return Ok(MeteredAdmission {
                reserved_msat: reserved,
                deposit_msat: deposit,
            });
        }
        let exposure: u64 = tx.query_row(PEER_EXPOSURE, [peer], |r| read_amount(r, 0))?;
        ensure!(
            exposure
                .checked_add(reserved)
                .is_some_and(|total| total <= peer_cap_msat),
            "client exposure cap exceeded"
        );
        // serving_requests owns peer identity for receivables; no
        // serving_accounting row (that table only holds pre-cutover rows).
        tx.execute(
            "INSERT INTO serving_requests(id,peer) VALUES (?1,?2)",
            params![id, peer],
        )?;
        tx.execute(
            "INSERT INTO metered_requests(id,peer,pricing,estimated_input_tokens,max_output_tokens,require_deposit,reserved_msat,deposit_msat,state) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,'reserved')",
            params![
                id,
                peer,
                pricing_json,
                sql_amount(estimated_input_tokens)?,
                sql_amount(max_output_tokens)?,
                require_deposit,
                sql_amount(reserved)?,
                sql_amount(deposit)?
            ],
        )?;
        tx.commit()?;
        Ok(MeteredAdmission {
            reserved_msat: reserved,
            deposit_msat: deposit,
        })
    }

    /// Finalize once, atomically: write the bill, apply/create peer credit and
    /// replace the reservation, all keyed on `id`. Replaying the same outcome
    /// returns the stored settlement without moving credit again.
    pub fn metered_finalize(&self, id: &str, outcome: MeteredOutcome) -> Result<MeteredSettlement> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (peer, pricing, deposit, state, stored_outcome): (
            String,
            String,
            u64,
            String,
            Option<String>,
        ) = tx
            .query_row(
                "SELECT peer,pricing,deposit_msat,state,outcome FROM metered_requests WHERE id=?",
                [id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        read_amount(r, 2)?,
                        r.get(3)?,
                        r.get(4)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("unknown metered request"))?;
        let outcome_json = serde_json::to_string(&outcome)?;
        // `usage_missing` is the only resolvable final state: later
        // authoritative usage (or a backend-error finding) replaces it once.
        let resolvable = state == "reserved"
            || (state == "usage_missing" && outcome != MeteredOutcome::UsageMissing);
        if !resolvable {
            ensure!(
                stored_outcome.as_deref() == Some(outcome_json.as_str()),
                "metered request already finalized differently"
            );
            drop(tx);
            drop(connection);
            return self.metered_settlement(id);
        }
        // An unpaid deposit contributes nothing: the request may be released
        // before payment, and a bill is never reduced by money not received.
        let deposit_paid = if deposit > 0 && deposit_paid(&tx, id)? {
            deposit
        } else {
            0
        };
        let pricing: Pricing = serde_json::from_str(&pricing)?;
        let (state, bill, applied, created, remaining) = match outcome {
            MeteredOutcome::Usage(usage) => {
                let bill = pricing
                    .input_charge(usage.prompt_tokens)?
                    .checked_add(pricing.output_charge(usage.completion_tokens)?)
                    .ok_or_else(|| anyhow::anyhow!("bill overflow"))?;
                let created = deposit_paid.saturating_sub(bill);
                let owed = bill.saturating_sub(deposit_paid);
                let applied = owed.min(peer_credit(&tx, &peer)?);
                ("billed", Some(bill), applied, created, owed - applied)
            }
            MeteredOutcome::BackendError => ("backend_error", None, 0, deposit_paid, 0),
            MeteredOutcome::UsageMissing => ("usage_missing", None, 0, 0, 0),
        };
        // Credit only flows one way per request; consuming it requires the
        // row the balance was read from, so no negative insert is possible.
        if created > 0 {
            tx.execute(
                "INSERT INTO peer_credit(peer,msat) VALUES (?1,?2) ON CONFLICT(peer) DO UPDATE SET msat=msat+excluded.msat",
                params![peer, sql_amount(created)?],
            )?;
        }
        if applied > 0 {
            tx.execute(
                "UPDATE peer_credit SET msat=msat-?2 WHERE peer=?1",
                params![peer, sql_amount(applied)?],
            )?;
        }
        let changed = tx.execute(
            "UPDATE metered_requests SET state=?2,outcome=?3,bill_msat=?4,deposit_paid_msat=?5,credit_applied_msat=?6,credit_created_msat=?7,remaining_msat=?8 WHERE id=?1 AND state IN ('reserved','usage_missing')",
            params![
                id,
                state,
                outcome_json,
                bill.map(sql_amount).transpose()?,
                sql_amount(deposit_paid)?,
                sql_amount(applied)?,
                sql_amount(created)?,
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
            "SELECT m.state,m.outcome,m.bill_msat,m.deposit_paid_msat,m.credit_applied_msat,m.credit_created_msat,m.remaining_msat, EXISTS(SELECT 1 FROM receivables WHERE request_id=m.id AND segment=1 AND state='paid') FROM metered_requests m WHERE m.id=?",
            [id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    read_amount(r, 3)?,
                    read_amount(r, 4)?,
                    read_amount(r, 5)?,
                    read_amount(r, 6)?,
                    r.get::<_, bool>(7)?,
                ))
            },
        )?;
        let (state, outcome, bill, deposit_paid_msat, applied, created, remaining, final_paid) =
            row;
        let usage = match outcome.as_deref().map(serde_json::from_str).transpose()? {
            Some(MeteredOutcome::Usage(usage)) => Some(usage),
            _ => None,
        };
        let status = match state.as_str() {
            "reserved" => MeteredStatus::Reserved,
            "billed" if remaining == 0 || final_paid => MeteredStatus::Settled,
            "billed" => MeteredStatus::Billed,
            "backend_error" => MeteredStatus::BackendError,
            "usage_missing" => MeteredStatus::UsageMissing,
            other => bail!("unknown metered state {other}"),
        };
        Ok(MeteredSettlement {
            status,
            usage,
            bill_msat: bill.map(u64::try_from).transpose()?,
            deposit_paid_msat,
            credit_applied_msat: applied,
            credit_created_msat: created,
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
        ensure!(state == "reserved", "metered request already finalized");
        tx.execute("UPDATE metered_requests SET dispatched=1 WHERE id=?", [id])?;
        tx.commit()?;
        Ok(true)
    }

    /// Owner and dispatch marker, for authenticated recovery. A dispatched
    /// request whose usage is unknown must be finalized `UsageMissing`, never
    /// `BackendError` as if it had not been dispatched.
    pub fn metered_owner(&self, id: &str) -> Result<Option<MeteredOwner>> {
        Ok(self
            .lock()?
            .query_row(
                "SELECT peer,dispatched FROM metered_requests WHERE id=?",
                [id],
                |r| {
                    Ok(MeteredOwner {
                        peer: r.get(0)?,
                        dispatched: r.get(1)?,
                    })
                },
            )
            .optional()?)
    }

    /// Startup-only: run after the exclusive service lock, before serving and
    /// never from live Recover (it would race active inference). Rows left
    /// `reserved` by a crash close as `UsageMissing` if dispatch was claimed
    /// (the backend may have run) or `BackendError` if not (deposit refunded
    /// as credit now, or on late arrival via `mark_received`).
    pub fn close_interrupted_metered(&self) -> Result<()> {
        let rows: Vec<(String, bool)> = {
            let connection = self.lock()?;
            let mut statement = connection
                .prepare("SELECT id,dispatched FROM metered_requests WHERE state='reserved'")?;
            statement
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        for (id, dispatched) in rows {
            let outcome = if dispatched {
                MeteredOutcome::UsageMissing
            } else {
                MeteredOutcome::BackendError
            };
            self.metered_finalize(&id, outcome)?;
        }
        Ok(())
    }

    /// Exposure currently counted against `peer`'s cap.
    pub fn metered_exposure(&self, peer: &str) -> Result<u64> {
        Ok(self
            .lock()?
            .query_row(PEER_EXPOSURE, [peer], |r| read_amount(r, 0))?)
    }

    pub fn peer_credit(&self, peer: &str) -> Result<u64> {
        let connection = self.lock()?;
        let tx = connection.unchecked_transaction()?;
        peer_credit(&tx, peer)
    }
}

/// Refuse a metered receivable that does not match current state: segment 0
/// must be the admitted deposit on a still-reserved row; segment 1 must equal
/// the current `remaining_msat` of a billed row. A stale amount errors so the
/// caller re-reads the settlement and retries. Non-metered rows pass through.
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
            state == "reserved" && deposit > 0 && amount == Some(deposit),
            "metered deposit invoice does not match admitted deposit"
        ),
        1 => ensure!(
            state == "billed" && remaining > 0 && amount == Some(remaining),
            "stale metered final invoice: amount must equal current remaining"
        ),
        _ => bail!("metered requests have only segments 0 and 1"),
    }
    Ok(())
}

/// A metered deposit (segment 0) paid after finalization, applied inside the
/// `mark_received` transaction. Called only on an unpaid->paid transition, so
/// replays cannot mint credit twice. Before finalization nothing happens:
/// finalize reads the paid deposit itself. `usage_missing` stays untouched
/// until resolved, which then counts the deposit.
pub(super) fn apply_late_deposit(tx: &Transaction<'_>, id: &str) -> Result<()> {
    let row: Option<(String, String, u64, u64, u64)> = tx
        .query_row(
            "SELECT peer,state,deposit_msat,deposit_paid_msat,remaining_msat FROM metered_requests WHERE id=?",
            [id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    read_amount(r, 2)?,
                    read_amount(r, 3)?,
                    read_amount(r, 4)?,
                ))
            },
        )
        .optional()?;
    let Some((peer, state, deposit, already, remaining)) = row else {
        return Ok(());
    };
    if deposit == 0 || already > 0 || !matches!(state.as_str(), "billed" | "backend_error") {
        return Ok(());
    }
    // Once a final invoice exists its amount is fixed, so the deposit can only
    // become credit; otherwise it first reduces what is still owed.
    let final_recorded: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM receivables WHERE request_id=? AND segment=1)",
        [id],
        |r| r.get(0),
    )?;
    let reduce = if state == "billed" && !final_recorded {
        deposit.min(remaining)
    } else {
        0
    };
    let created = deposit - reduce;
    if created > 0 {
        tx.execute(
            "INSERT INTO peer_credit(peer,msat) VALUES (?1,?2) ON CONFLICT(peer) DO UPDATE SET msat=msat+excluded.msat",
            params![peer, sql_amount(created)?],
        )?;
    }
    tx.execute(
        "UPDATE metered_requests SET deposit_paid_msat=?2,remaining_msat=remaining_msat-?3,credit_created_msat=credit_created_msat+?4 WHERE id=?1",
        params![id, sql_amount(deposit)?, sql_amount(reduce)?, sql_amount(created)?],
    )?;
    Ok(())
}

fn deposit_paid(tx: &Transaction<'_>, id: &str) -> Result<bool> {
    Ok(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM receivables WHERE request_id=? AND segment=0 AND state='paid')",
        [id],
        |r| r.get(0),
    )?)
}

fn peer_credit(tx: &Transaction<'_>, peer: &str) -> Result<u64> {
    Ok(tx
        .query_row("SELECT msat FROM peer_credit WHERE peer=?", [peer], |r| {
            read_amount(r, 0)
        })
        .optional()?
        .unwrap_or(0))
}
