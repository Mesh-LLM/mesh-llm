//! SQLite owns approvals and reservations across processes and restarts.

pub mod receivables;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

use crate::invoice::Invoice;
use crate::pricing::Pricing;
use crate::wallet::{PaymentStatus, Transaction};

const DAY_MS: u64 = 86_400_000;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    #[default]
    Manual,
    Automatic,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Policy {
    pub mode: ApprovalMode,
    pub daily_budget_msat: Option<u64>,
}

impl Policy {
    pub fn validate(&self) -> Result<()> {
        if self.mode == ApprovalMode::Automatic {
            ensure!(
                self.daily_budget_msat
                    .is_some_and(|n| n > 0 && n <= i64::MAX as u64),
                "automatic payments require a positive daily budget"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTerms {
    pub id: String,
    pub peer: String,
    #[serde(default)]
    pub payee: Option<String>,
    pub model: String,
    pub pricing: Pricing,
    pub input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_total_msat: u64,
    pub expires_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestRecord {
    pub terms: RequestTerms,
    pub state: String,
    pub spent_msat: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Charge {
    pub request_id: String,
    pub segment: u32,
    pub invoice: Invoice,
    pub amount_msat: u64,
    pub max_total_msat: u64,
}

pub struct Ledger {
    connection: Mutex<Connection>,
}

impl Ledger {
    pub fn open(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        }
        let connection = Connection::open(directory.join("payments.sqlite3"))?;
        connection.busy_timeout(std::time::Duration::from_secs(10))?;
        connection.execute_batch(include_str!("ledger/schema.sql"))?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| anyhow::anyhow!("payment ledger lock poisoned"))
    }

    pub fn policy(&self) -> Result<Policy> {
        self.get_setting("policy")?
            .map_or_else(|| Ok(Policy::default()), |v| Ok(serde_json::from_str(&v)?))
    }

    pub fn set_policy(&self, policy: &Policy) -> Result<()> {
        policy.validate()?;
        self.set_setting("policy", &serde_json::to_string(policy)?)
    }

    fn get_setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .query_row("SELECT value FROM settings WHERE key=?", [key], |r| {
                r.get(0)
            })
            .optional()?)
    }

    fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.lock()?.execute("INSERT INTO settings VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value])?;
        Ok(())
    }

    pub fn set_pricing(&self, model: &str, pricing: Option<&Pricing>) -> Result<()> {
        ensure!(
            !model.is_empty() && model.len() <= 1024,
            "model must be 1..1024 bytes"
        );
        if let Some(pricing) = pricing {
            pricing.validate()?;
            let existing = self.pricing()?;
            ensure!(
                existing.contains_key(model) || existing.len() < 128,
                "maximum 128 paid model offers"
            );
            self.lock()?.execute("INSERT INTO pricing VALUES (?1,?2) ON CONFLICT(model) DO UPDATE SET value=excluded.value", params![model, serde_json::to_string(pricing)?])?;
        } else {
            self.lock()?
                .execute("DELETE FROM pricing WHERE model=?", [model])?;
        }
        Ok(())
    }

    pub fn pricing(&self) -> Result<std::collections::BTreeMap<String, Pricing>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare("SELECT model,value FROM pricing ORDER BY model")?;
        let rows =
            statement.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        rows.map(|row| {
            let (model, value) = row?;
            Ok((model, serde_json::from_str(&value)?))
        })
        .collect()
    }

    /// Only the local-ingress owner may call this. Remote requests must never be
    /// translated into payer requests through a caller-supplied identity field.
    pub fn propose(&self, terms: &RequestTerms) -> Result<()> {
        terms.pricing.validate()?;
        ensure!(
            terms.max_total_msat > 0 && terms.max_total_msat <= i64::MAX as u64,
            "invalid spending limit"
        );
        ensure!(
            terms.max_output_tokens > 0,
            "finite positive output allowance required"
        );
        let value = serde_json::to_string(terms)?;
        let connection = self.lock()?;
        connection.execute("INSERT OR IGNORE INTO requests(id,terms,state,cap,spent) VALUES (?1,?2,'pending',?3,0)", params![terms.id, value, sql_amount(terms.max_total_msat)?])?;
        let existing: String =
            connection.query_row("SELECT terms FROM requests WHERE id=?", [&terms.id], |r| {
                r.get(0)
            })?;
        ensure!(
            existing == value,
            "request identity reused with different payment terms"
        );
        Ok(())
    }

    /// Reserve funds in an immediate transaction; independent processes cannot
    /// authorize against the same unreserved balance or daily allowance.
    pub fn available_budget(&self, balance_msat: u64, now_ms: u64) -> Result<u64> {
        let policy = self.policy()?;
        let connection = self.lock()?;
        let reserved: u64 = connection.query_row(
            "SELECT COALESCE(SUM(MAX(cap-spent,0)),0) FROM requests WHERE state='approved'",
            [],
            |r| read_amount(r, 0),
        )?;
        let available = balance_msat.saturating_sub(reserved);
        if policy.mode == ApprovalMode::Manual {
            return Ok(available);
        }
        let spent: u64 = connection.query_row(
            "SELECT COALESCE(SUM(total),0) FROM charges WHERE state='succeeded' AND settled_day=?",
            [sql_amount(now_ms / DAY_MS)?],
            |r| read_amount(r, 0),
        )?;
        Ok(available.min(
            policy
                .daily_budget_msat
                .unwrap_or(0)
                .saturating_sub(spent)
                .saturating_sub(reserved),
        ))
    }

    pub fn approve(&self, id: &str, balance_msat: u64, now_ms: u64) -> Result<()> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, json): (String, String) =
            transaction.query_row("SELECT state,terms FROM requests WHERE id=?", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        if state == "approved" {
            return Ok(());
        }
        ensure!(
            state == "pending",
            "request cannot be approved in state {state}"
        );
        let terms: RequestTerms = serde_json::from_str(&json)?;
        ensure!(terms.expires_at_ms > now_ms, "input invoice has expired");
        let reserved: u64 = transaction.query_row(
            "SELECT COALESCE(SUM(MAX(cap-spent,0)),0) FROM requests WHERE state='approved'",
            [],
            |r| read_amount(r, 0),
        )?;
        ensure!(
            terms.max_total_msat <= balance_msat.saturating_sub(reserved),
            "insufficient unreserved wallet balance"
        );
        let policy_json: Option<String> = transaction
            .query_row("SELECT value FROM settings WHERE key='policy'", [], |r| {
                r.get(0)
            })
            .optional()?;
        let policy: Policy = policy_json
            .map(|j| serde_json::from_str(&j))
            .transpose()?
            .unwrap_or_default();
        if policy.mode == ApprovalMode::Automatic {
            policy.validate()?;
            let spent: u64 = transaction.query_row("SELECT COALESCE(SUM(total),0) FROM charges WHERE state='succeeded' AND settled_day=?", [sql_amount(now_ms / DAY_MS)?], |r| read_amount(r, 0))?;
            let available = policy
                .daily_budget_msat
                .unwrap_or(0)
                .saturating_sub(spent)
                .saturating_sub(reserved);
            ensure!(
                terms.max_total_msat <= available,
                "daily payment budget exhausted"
            );
        }
        transaction.execute("UPDATE requests SET state='approved' WHERE id=?", [id])?;
        transaction.commit()?;
        Ok(())
    }

    pub fn reject(&self, id: &str) -> Result<()> {
        let changed = self.lock()?.execute(
            "UPDATE requests SET state='rejected' WHERE id=? AND state='pending'",
            [id],
        )?;
        ensure!(changed == 1, "only pending requests may be rejected");
        Ok(())
    }

    pub fn requests(&self) -> Result<Vec<RequestRecord>> {
        let connection = self.lock()?;
        let mut statement =
            connection.prepare("SELECT terms,state,spent FROM requests ORDER BY rowid DESC")?;
        let rows = statement.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get(1)?, read_amount(r, 2)?))
        })?;
        rows.map(|row| {
            let (terms, state, spent_msat) = row?;
            Ok(RequestRecord {
                terms: serde_json::from_str(&terms)?,
                state,
                spent_msat,
            })
        })
        .collect()
    }

    pub fn request_state(&self, id: &str) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .query_row("SELECT state FROM requests WHERE id=?", [id], |r| r.get(0))
            .optional()?)
    }

    pub fn charge_state(&self, hash: &str) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .query_row("SELECT state FROM charges WHERE hash=?", [hash], |r| {
                r.get(0)
            })
            .optional()?)
    }

    /// Persist the uncertainty boundary before making any submission call.
    /// The payment service serializes this with wallet I/O.
    pub fn begin_submission(&self, hash: &str) -> Result<()> {
        let changed = self.lock()?.execute(
            "UPDATE charges SET state='pending' WHERE hash=? AND state='prepared'",
            [hash],
        )?;
        ensure!(changed == 1, "payment submission already started");
        Ok(())
    }

    pub fn fail_authorization_if_idle(&self, id: &str) -> Result<()> {
        self.lock()?.execute("UPDATE requests SET state='failed' WHERE id=?1 AND state='approved' AND NOT EXISTS(SELECT 1 FROM charges WHERE request_id=?1 AND state IN ('prepared','pending'))", [id])?;
        Ok(())
    }

    /// Only call after the provider guarantees that this attempt did not submit.
    pub fn fail_unsubmitted(&self, hash: &str) -> Result<()> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "UPDATE charges SET state='failed' WHERE hash=? AND state IN ('prepared','pending')",
            [hash],
        )?;
        finalize_terminal_requests(&tx)?;
        tx.commit()?;
        Ok(())
    }

    /// Repair terminal requests left behind by an older process or a crash.
    pub fn finalize_terminal_requests(&self) -> Result<()> {
        let connection = self.lock()?;
        finalize_terminal_requests(&connection)
    }

    /// Commit intent before contacting the wallet. A segment/hash can never be
    /// repurposed; errors leave the reservation held until reconciliation.
    pub fn prepare_charge(&self, charge: &Charge) -> Result<bool> {
        ensure!(
            charge.max_total_msat >= charge.amount_msat,
            "charge cap below amount"
        );
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String, u64, u64)> = transaction
            .query_row(
                "SELECT hash,invoice,amount,max_total FROM charges WHERE request_id=?1 AND segment=?2",
                params![charge.request_id, charge.segment],
                |r| Ok((r.get(0)?, r.get(1)?, read_amount(r, 2)?, read_amount(r, 3)?)),
            )
            .optional()?;
        if let Some((hash, invoice, amount, cap)) = existing {
            ensure!(
                hash == charge.invoice.payment_hash
                    && invoice == charge.invoice.bolt11
                    && amount == charge.amount_msat
                    && cap == charge.max_total_msat,
                "segment already has different payment terms"
            );
            return Ok(false);
        }
        charge
            .invoice
            .validate_payment(charge.amount_msat, crate::now_ms())?;
        let (state, cap, spent): (String, u64, u64) = transaction.query_row(
            "SELECT state,cap,spent FROM requests WHERE id=?",
            [&charge.request_id],
            |r| Ok((r.get(0)?, read_amount(r, 1)?, read_amount(r, 2)?)),
        )?;
        ensure!(state == "approved", "request is not approved");
        let failed: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM charges WHERE request_id=? AND state='failed')",
            [&charge.request_id],
            |r| r.get(0),
        )?;
        ensure!(!failed, "request has a failed payment");
        let pending: u64 = transaction.query_row(
            "SELECT COALESCE(SUM(max_total),0) FROM charges WHERE request_id=? AND state IN ('prepared','pending')",
            [&charge.request_id],
            |r| read_amount(r, 0),
        )?;
        ensure!(
            charge.max_total_msat <= cap.saturating_sub(spent).saturating_sub(pending),
            "request payment limit exceeded"
        );
        transaction.execute("INSERT INTO charges(hash,request_id,segment,invoice,amount,max_total,state,total) VALUES (?1,?2,?3,?4,?5,?6,'prepared',0)", params![charge.invoice.payment_hash,charge.request_id,charge.segment,charge.invoice.bolt11,sql_amount(charge.amount_msat)?,sql_amount(charge.max_total_msat)?])?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn reconcile(&self, payment: &Transaction, now_ms: u64) -> Result<()> {
        ensure!(
            !payment.inbound,
            "cannot reconcile outgoing charge with incoming payment"
        );
        let hash = payment
            .payment_hash
            .as_ref()
            .context("missing payment hash")?;
        if payment.status == PaymentStatus::Pending {
            return Ok(());
        }
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, request_id, amount): (String, String, u64) = transaction.query_row(
            "SELECT state,request_id,amount FROM charges WHERE hash=?",
            [hash],
            |r| Ok((r.get(0)?, r.get(1)?, read_amount(r, 2)?)),
        )?;
        if !matches!(state.as_str(), "prepared" | "pending") {
            return Ok(());
        }
        let (state, total) = if payment.status == PaymentStatus::Succeeded {
            ensure!(
                payment.amount_msat >= amount,
                "wallet settled less than the requested amount"
            );
            (
                "succeeded",
                payment
                    .amount_msat
                    .checked_add(payment.fee_msat)
                    .context("payment total overflow")?,
            )
        } else {
            ("failed", 0)
        };
        // Record the actual debit even if a provider violates its fee guarantee.
        // Keeping real spend prevents an erroneous provider from resetting budget.
        transaction.execute(
            "UPDATE charges SET state=?1,total=?2,settled_day=?3 WHERE hash=?4",
            params![
                state,
                sql_amount(total)?,
                sql_amount(payment.settled_at_ms.unwrap_or(now_ms) / DAY_MS)?,
                hash
            ],
        )?;
        transaction.execute(
            "UPDATE requests SET spent=spent+?1 WHERE id=?2",
            params![sql_amount(total)?, request_id],
        )?;
        finalize_terminal_requests(&transaction)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn pending_charges(&self) -> Result<Vec<Charge>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare("SELECT request_id,segment,invoice,amount,max_total FROM charges WHERE state IN ('prepared','pending')")?;
        let rows = statement.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get(1)?,
                r.get::<_, String>(2)?,
                read_amount(r, 3)?,
                read_amount(r, 4)?,
            ))
        })?;
        rows.map(|row| {
            let (request_id, segment, invoice, amount_msat, max_total_msat) = row?;
            Ok(Charge {
                request_id,
                segment,
                invoice: Invoice::parse(&invoice)?,
                amount_msat,
                max_total_msat,
            })
        })
        .collect()
    }

    pub fn finish(&self, id: &str) -> Result<()> {
        let connection = self.lock()?;
        let changed = connection.execute("UPDATE requests SET state='completed' WHERE id=?1 AND state='approved' AND NOT EXISTS(SELECT 1 FROM charges WHERE request_id=?1 AND state IN ('prepared','pending'))", [id])?;
        let completed: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM requests WHERE id=? AND state='completed')",
            [id],
            |r| r.get(0),
        )?;
        ensure!(
            changed == 1 || completed,
            "request has uncertain payments or is not approved"
        );
        Ok(())
    }
}

fn finalize_terminal_requests(connection: &Connection) -> Result<()> {
    connection.execute("UPDATE charges SET state='failed' WHERE state='prepared' AND EXISTS(SELECT 1 FROM charges sibling WHERE sibling.request_id=charges.request_id AND sibling.state='failed')", [])?;
    // Failure revokes unused authorization, but cannot release any uncertain
    // sibling charge. A send has only one charge and can close on success.
    connection.execute("UPDATE requests SET state=CASE WHEN EXISTS(SELECT 1 FROM charges WHERE request_id=requests.id AND state='failed') THEN 'failed' ELSE 'completed' END WHERE state='approved' AND NOT EXISTS(SELECT 1 FROM charges WHERE request_id=requests.id AND state IN ('prepared','pending')) AND (EXISTS(SELECT 1 FROM charges WHERE request_id=requests.id AND state='failed') OR (json_extract(terms,'$.peer')='wallet-send' AND EXISTS(SELECT 1 FROM charges WHERE request_id=requests.id AND state='succeeded')))", [])?;
    Ok(())
}

fn sql_amount(value: u64) -> Result<i64> {
    value
        .try_into()
        .context("payment amount exceeds ledger range")
}

fn read_amount(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<u64> {
    let value: i64 = row.get(index)?;
    value
        .try_into()
        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(index, value))
}
