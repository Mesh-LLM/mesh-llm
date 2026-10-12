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
    /// `unpaid_invoice` for a recorded invoice the peer has not paid,
    /// `metered_invoice` for an unpaid metered (v2) invoice, which `unblock`
    /// does not forgive, or `uninvoiced_output` for delivered output that
    /// has no invoice yet.
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

    /// Every peer currently refused paid inference, with each recorded debt.
    /// Read-only and wallet-free, so it works while the node is stopped.
    pub fn blocked_peers(&self) -> Result<Vec<BlockedPeer>> {
        let connection = self.lock()?;
        let mut by_peer: std::collections::BTreeMap<String, Vec<PeerDebt>> = Default::default();
        let mut unpaid = connection.prepare(
            "SELECT peer,request_id,segment,invoice,tokens,EXISTS(SELECT 1 FROM metered_requests WHERE id=receivables.request_id) FROM receivables WHERE state='unpaid' ORDER BY rowid",
        )?;
        for row in unpaid.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, u32>(2)?,
                r.get::<_, String>(3)?,
                read_amount(r, 4)?,
                r.get::<_, bool>(5)?,
            ))
        })? {
            let (peer, request_id, segment, invoice, tokens, metered) = row?;
            let invoice = Invoice::parse(&invoice)?;
            by_peer.entry(peer).or_default().push(PeerDebt {
                request_id,
                segment,
                kind: if metered {
                    "metered_invoice"
                } else {
                    "unpaid_invoice"
                }
                .into(),
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
    /// Metered invoices are settled through the metered ledger and are never
    /// forgiven here; they keep blocking and counting as exposure.
    pub fn unblock_peer(&self, peer: &str) -> Result<Forgiven> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let invoices = tx.execute(
            "UPDATE receivables SET state='forgiven' WHERE peer=?1 AND state='unpaid' AND NOT EXISTS(SELECT 1 FROM metered_requests WHERE id=receivables.request_id)",
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
