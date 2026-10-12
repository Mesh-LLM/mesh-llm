use anyhow::{Result, ensure};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use super::{Ledger, read_amount, sql_amount};
use crate::invoice::Invoice;

#[derive(Clone, Serialize, Deserialize)]
pub struct Receivable {
    pub request_id: String,
    pub peer: String,
    pub segment: u32,
    pub invoice: Invoice,
    pub tokens: u64,
    pub paid: bool,
}

impl Ledger {
    pub fn record_receivable(&self, receipt: &Receivable) -> Result<()> {
        ensure!(
            receipt.invoice == Invoice::parse(&receipt.invoice.bolt11)?,
            "invalid invoice metadata"
        );
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let peer: Option<String> = tx
            .query_row(
                "SELECT peer FROM serving_requests WHERE id=?",
                [&receipt.request_id],
                |r| r.get(0),
            )
            .optional()?;
        ensure!(
            peer.as_deref() == Some(&receipt.peer),
            "unknown serving request"
        );
        // Metered amounts are checked against current state in this txn, so a
        // snapshot made stale by a late deposit is refused, not persisted.
        super::metered::check_metered_receivable(&tx, receipt)?;
        tx.execute("INSERT INTO receivables(hash,request_id,peer,segment,invoice,tokens,state) VALUES (?1,?2,?3,?4,?5,?6,'unpaid')", params![receipt.invoice.payment_hash,receipt.request_id,receipt.peer,receipt.segment,receipt.invoice.bolt11,sql_amount(receipt.tokens)?])?;
        tx.commit()?;
        Ok(())
    }

    pub fn receivables(&self, request_id: Option<&str>) -> Result<Vec<Receivable>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare("SELECT request_id,peer,segment,invoice,tokens,state FROM receivables WHERE (?1 IS NULL OR request_id=?1) ORDER BY segment")?;
        let rows = statement.query_map([request_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, u32>(2)?,
                r.get::<_, String>(3)?,
                read_amount(r, 4)?,
                r.get::<_, String>(5)?,
            ))
        })?;
        rows.map(|row| {
            let (request_id, peer, segment, invoice, tokens, state) = row?;
            Ok(Receivable {
                request_id,
                peer,
                segment,
                invoice: Invoice::parse(&invoice)?,
                tokens,
                paid: state == "paid",
            })
        })
        .collect()
    }

    pub fn mark_received(&self, hash: &str) -> Result<()> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        // Only an actual transition moves money, so replays are no-ops.
        let paid: Option<(String, u32)> = tx
            .query_row(
                "UPDATE receivables SET state='paid' WHERE hash=? AND state!='paid' RETURNING request_id,segment",
                [hash],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((request_id, 0)) = paid {
            super::metered::apply_late_deposit(&tx, &request_id)?;
        }
        tx.commit()?;
        Ok(())
    }
}
