use anyhow::{Result, bail, ensure};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use super::{Ledger, read_amount, sql_amount};
use crate::invoice::Invoice;

/// One recorded debt that currently refuses a peer paid inference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerDebt {
    pub request_id: String,
    pub segment: u32,
    /// `unpaid_invoice` for a recorded invoice the peer has not paid, or
    /// `uninvoiced_output` for delivered output that has no invoice yet.
    pub kind: String,
    pub tokens: u64,
    pub amount_msat: Option<u64>,
    pub expires_at_ms: Option<u64>,
}

/// A peer refused paid inference, with the identifier `unblock` takes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockedPeer {
    /// Full authenticated mesh endpoint ID, as recorded at admission.
    pub peer: String,
    /// The abbreviated form the console and `/api/status` display.
    pub peer_short: String,
    pub debts: Vec<PeerDebt>,
}

/// What [`Ledger::unblock_peer`] forgave.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Forgiven {
    pub peer: String,
    pub invoices: usize,
    pub output_requests: usize,
}

/// Minimum length of a peer prefix accepted by [`Ledger::resolve_blocked_peer`].
pub const MIN_PEER_PREFIX: usize = 8;

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
            "SELECT EXISTS(SELECT 1 FROM receivables WHERE peer=?1 AND state='unpaid') OR EXISTS(SELECT 1 FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE r.peer=?1 AND a.finished=1 AND a.tokens>0 AND a.forgiven=0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=r.id AND segment=1))",
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
    /// transactional check in `begin_serving` remains the authority. Debt an
    /// operator has forgiven with [`Self::unblock_peer`] no longer counts.
    pub fn has_outstanding_payment(&self, peer: &str) -> Result<bool> {
        Ok(self.lock()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM receivables WHERE peer=?1 AND state='unpaid') OR EXISTS(SELECT 1 FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE r.peer=?1 AND a.finished=1 AND a.tokens>0 AND a.forgiven=0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=r.id AND segment=1))",
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
        Ok(())
    }

    /// Raise the delivered-token watermark to `tokens` and close serving
    /// accounting atomically. Idempotent: closing an already-finished request
    /// succeeds only if its recorded watermark already covers `tokens`, so a
    /// retry can never report success over a lower frozen watermark. A
    /// watermark above the output allowance is refused and leaves the request
    /// open (fail closed; startup recovery closes it).
    pub fn finish_serving_at(&self, id: &str, tokens: u64) -> Result<()> {
        let connection = self.lock()?;
        let changed = connection.execute(
            "UPDATE serving_accounting SET tokens=MAX(tokens,?2), finished=1 WHERE id=?1 AND finished=0 AND max_output>=?2",
            params![id, sql_amount(tokens)?],
        )?;
        if changed == 1 {
            return Ok(());
        }
        let covered: bool = connection.query_row(
            "SELECT finished=1 AND tokens>=?2 FROM serving_accounting WHERE id=?1",
            params![id, sql_amount(tokens)?],
            |row| row.get(0),
        )?;
        ensure!(covered, "invalid final delivered token watermark");
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
        Ok(())
    }

    /// Bounded rotating scan; rowid is local bookkeeping, never payment identity.
    pub fn unpaid_input_batch(&self, after: i64) -> Result<Vec<(i64, String)>> {
        let connection = self.lock()?;
        let query = |cursor| -> Result<Vec<(i64, String)>> {
            let mut statement = connection.prepare(
                "SELECT rowid,request_id FROM receivables WHERE segment=0 AND state='unpaid' AND rowid>? ORDER BY rowid LIMIT 32",
            )?;
            Ok(statement
                .query_map([cursor], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        };
        let rows = query(after)?;
        if rows.is_empty() && after != 0 {
            query(0)
        } else {
            Ok(rows)
        }
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

    /// Stop an abandoned input invoice counting as debt. Applies only when the
    /// invoice expired at least [`INPUT_LAPSE_GRACE`] ago, the request has
    /// finished, and no output was delivered: the buyer paid nothing and got
    /// nothing, so a slow or interrupted payment is not punished. The caller
    /// must first have looked the payment up and found no success or pending
    /// arrival. Delivered-but-unpaid output still blocks. A later receipt is
    /// still recorded as paid by [`Self::mark_received`].
    ///
    /// [`INPUT_LAPSE_GRACE`]: crate::lifetimes::INPUT_LAPSE_GRACE
    pub fn lapse_abandoned_input(
        &self,
        request_id: &str,
        invoice: &Invoice,
        now_ms: u64,
    ) -> Result<bool> {
        let grace = crate::lifetimes::INPUT_LAPSE_GRACE.as_millis() as u64;
        if invoice.expires_at_ms.saturating_add(grace) > now_ms {
            return Ok(false);
        }
        let changed = self.lock()?.execute(
            "UPDATE receivables SET state='lapsed' WHERE hash=?1 AND request_id=?2 AND segment=0 AND state='unpaid' AND EXISTS(SELECT 1 FROM serving_accounting WHERE id=?2 AND finished=1 AND tokens=0)",
            params![invoice.payment_hash, request_id],
        )?;
        Ok(changed == 1)
    }

    /// Requests whose input invoice this peer has not paid.
    pub fn unpaid_input_requests(&self, peer: &str) -> Result<Vec<String>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT request_id FROM receivables WHERE peer=? AND segment=0 AND state='unpaid'",
        )?;
        Ok(statement
            .query_map([peer], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn mark_received(&self, hash: &str) -> Result<()> {
        self.lock()?
            .execute("UPDATE receivables SET state='paid' WHERE hash=?", [hash])?;
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
        let mut statement = connection.prepare("SELECT a.id FROM serving_accounting a WHERE a.finished=1 AND a.tokens>0 AND a.forgiven=0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=a.id AND segment=1) LIMIT 32")?;
        Ok(statement
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Every peer currently refused paid inference, with each recorded debt.
    /// Read-only and wallet-free, so it works while the node is stopped.
    pub fn blocked_peers(&self) -> Result<Vec<BlockedPeer>> {
        let connection = self.lock()?;
        let mut by_peer: std::collections::BTreeMap<String, Vec<PeerDebt>> = Default::default();
        let mut unpaid = connection.prepare(
            "SELECT peer,request_id,segment,invoice,tokens FROM receivables WHERE state='unpaid' ORDER BY rowid",
        )?;
        for row in unpaid.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, u32>(2)?,
                r.get::<_, String>(3)?,
                read_amount(r, 4)?,
            ))
        })? {
            let (peer, request_id, segment, invoice, tokens) = row?;
            let invoice = Invoice::parse(&invoice)?;
            by_peer.entry(peer).or_default().push(PeerDebt {
                request_id,
                segment,
                kind: "unpaid_invoice".into(),
                tokens,
                amount_msat: invoice.amount_msat,
                expires_at_ms: Some(invoice.expires_at_ms),
            });
        }
        let mut uninvoiced = connection.prepare(
            "SELECT r.peer,r.id,a.tokens FROM serving_requests r JOIN serving_accounting a ON a.id=r.id WHERE a.finished=1 AND a.tokens>0 AND a.forgiven=0 AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=r.id AND segment=1) ORDER BY r.rowid",
        )?;
        for row in uninvoiced.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                read_amount(r, 2)?,
            ))
        })? {
            let (peer, request_id, tokens) = row?;
            by_peer.entry(peer).or_default().push(PeerDebt {
                request_id,
                segment: 1,
                kind: "uninvoiced_output".into(),
                tokens,
                amount_msat: None,
                expires_at_ms: None,
            });
        }
        Ok(by_peer
            .into_iter()
            .map(|(peer, debts)| BlockedPeer {
                peer_short: peer.chars().take(10).collect(),
                peer,
                debts,
            })
            .collect())
    }

    /// Accept the full endpoint ID or a unique prefix of at least
    /// [`MIN_PEER_PREFIX`] characters among currently blocked peers, so the
    /// value shown by `blocked` can be pasted whole or abbreviated.
    pub fn resolve_blocked_peer(&self, needle: &str) -> Result<String> {
        let needle = needle.trim();
        ensure!(
            needle.len() >= MIN_PEER_PREFIX,
            "peer identifier must be at least {MIN_PEER_PREFIX} characters"
        );
        let blocked = self.blocked_peers()?;
        if let Some(exact) = blocked.iter().find(|entry| entry.peer == needle) {
            return Ok(exact.peer.clone());
        }
        let matches: Vec<&BlockedPeer> = blocked
            .iter()
            .filter(|entry| entry.peer.starts_with(needle))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one.peer.clone()),
            [] => bail!("no blocked peer matches {needle}"),
            _ => bail!(
                "peer prefix {needle} is ambiguous; use the full identifier from `wallet blocked`"
            ),
        }
    }

    /// Operator override: forgive every recorded debt for `peer` so it may
    /// request paid inference again. Unpaid invoices are marked `forgiven`
    /// (a later payment is still recorded as received) and delivered output
    /// that has no invoice yet is never invoiced. Nothing is refunded.
    pub fn unblock_peer(&self, peer: &str) -> Result<Forgiven> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let invoices = tx.execute(
            "UPDATE receivables SET state='forgiven' WHERE peer=?1 AND state='unpaid'",
            [peer],
        )?;
        let output_requests = tx.execute(
            "UPDATE serving_accounting SET forgiven=1 WHERE finished=1 AND tokens>0 AND forgiven=0 AND id IN (SELECT id FROM serving_requests WHERE peer=?1) AND NOT EXISTS(SELECT 1 FROM receivables WHERE request_id=serving_accounting.id AND segment=1)",
            [peer],
        )?;
        ensure!(
            invoices + output_requests > 0,
            "peer has no recorded debt; nothing to unblock"
        );
        tx.commit()?;
        Ok(Forgiven {
            peer: peer.to_owned(),
            invoices,
            output_requests,
        })
    }
}
