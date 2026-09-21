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
    /// Claim an unguessable request ID before running inference. A replay can
    /// recover invoices, but cannot start a second generation under that ID.
    pub fn begin_serving(
        &self,
        id: &str,
        peer: &str,
        pricing: &crate::pricing::Pricing,
        max_output: u64,
    ) -> Result<()> {
        let mut connection = self.lock()?;
        let connection =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let blocked: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM receivables WHERE peer=?1 AND state='unpaid') OR EXISTS(SELECT 1 FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE r.peer=?1 AND a.finished=1 AND a.tokens>0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=r.id AND segment=1))",
            [peer],
            |r| r.get(0),
        )?;
        ensure!(!blocked, "client has an outstanding payment");
        connection.execute(
            "INSERT INTO serving_requests(id,peer) VALUES (?1,?2)",
            params![id, peer],
        )?;
        connection.execute(
            "INSERT INTO serving_accounting(id,pricing,max_output) VALUES (?1,?2,?3)",
            params![id, serde_json::to_string(pricing)?, sql_amount(max_output)?],
        )?;
        connection.commit()?;
        Ok(())
    }

    /// Whether admission must wait for recorded debt. This is advisory; the
    /// transactional check in `begin_serving` remains the authority.
    pub fn has_outstanding_payment(&self, peer: &str) -> Result<bool> {
        Ok(self.lock()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM receivables WHERE peer=?1 AND state='unpaid') OR EXISTS(SELECT 1 FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE r.peer=?1 AND a.finished=1 AND a.tokens>0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=r.id AND segment=1))",
            [peer],
            |row| row.get(0),
        )?)
    }

    /// Freeze the backend's resolved context allowance before creating the
    /// input invoice. It may be smaller than the caller's explicit ceiling.
    pub fn resolve_serving_output_allowance(&self, id: &str, tokens: u64) -> Result<()> {
        ensure!(tokens > 0, "empty backend output allowance");
        let changed = self.lock()?.execute(
            "UPDATE serving_accounting SET max_output=?2 WHERE id=?1 AND finished=0 AND tokens=0 AND max_output>=?2 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=?1)",
            params![id, sql_amount(tokens)?],
        )?;
        ensure!(changed == 1, "invalid backend output allowance");
        Ok(())
    }

    pub fn record_delivered_tokens(&self, id: &str, tokens: u64) -> Result<()> {
        let changed = self.lock()?.execute("UPDATE serving_accounting SET tokens=?2 WHERE id=?1 AND finished=0 AND tokens<=?2 AND max_output>=?2", params![id,sql_amount(tokens)?])?;
        ensure!(changed == 1, "invalid delivered token watermark");
        Ok(())
    }

    pub fn finish_serving(&self, id: &str) -> Result<()> {
        self.lock()?
            .execute("UPDATE serving_accounting SET finished=1 WHERE id=?", [id])?;
        self.observe_payment(id);
        Ok(())
    }

    /// Only run once when opening the process's service. Native KV state is not
    /// resumable after a process restart, while already committed debt is.
    pub fn close_interrupted_serving(&self) -> Result<()> {
        self.lock()?.execute(
            "UPDATE serving_accounting SET finished=1 WHERE finished=0",
            [],
        )?;
        Ok(())
    }

    pub fn serving_account(
        &self,
        id: &str,
    ) -> Result<(String, crate::pricing::Pricing, u64, bool)> {
        let (peer,price,tokens,finished): (String,String,u64,bool) = self.lock()?.query_row("SELECT r.peer,a.pricing,a.tokens,a.finished FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE r.id=?", [id], |r| Ok((r.get(0)?,r.get(1)?,read_amount(r,2)?,r.get(3)?)))?;
        Ok((peer, serde_json::from_str(&price)?, tokens, finished))
    }

    pub fn record_receivable(&self, receipt: &Receivable) -> Result<()> {
        ensure!(
            receipt.invoice == Invoice::parse(&receipt.invoice.bolt11)?,
            "invalid invoice metadata"
        );
        let connection = self.lock()?;
        let peer: Option<String> = connection
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
        connection.execute("INSERT INTO receivables(hash,request_id,peer,segment,invoice,tokens,state) VALUES (?1,?2,?3,?4,?5,?6,'unpaid')", params![receipt.invoice.payment_hash,receipt.request_id,receipt.peer,receipt.segment,receipt.invoice.bolt11,sql_amount(receipt.tokens)?])?;
        drop(connection);
        self.observe_payment(&receipt.request_id);
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
        let connection = self.lock()?;
        connection.execute("UPDATE receivables SET state='paid' WHERE hash=?", [hash])?;
        let id: Option<String> = connection
            .query_row(
                "SELECT request_id FROM receivables WHERE hash=?",
                [hash],
                |r| r.get(0),
            )
            .optional()?;
        drop(connection);
        if let Some(id) = id {
            self.observe_payment(&id);
        }
        Ok(())
    }

    pub fn unpaid_invoices(&self, peer: &str) -> Result<Vec<Invoice>> {
        let connection = self.lock()?;
        let mut statement = connection
            .prepare("SELECT invoice FROM receivables WHERE peer=? AND state='unpaid'")?;
        statement
            .query_map([peer], |r| r.get::<_, String>(0))?
            .map(|invoice| Invoice::parse(&invoice?))
            .collect()
    }

    /// Bounded batches of durable output debt needing an invoice after a crash
    /// or temporary wallet failure. Admission blocks on this debt immediately.
    pub fn uninvoiced_output(&self) -> Result<Vec<String>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare("SELECT a.id FROM serving_accounting a WHERE a.finished=1 AND a.tokens>0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=a.id AND segment=1) LIMIT 32")?;
        Ok(statement
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }
}
