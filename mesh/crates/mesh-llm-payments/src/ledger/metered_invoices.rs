//! Invoices issued for metered requests: one deposit and one final invoice at
//! most per request. Issued, unpaid final invoices are the only debt a buyer
//! can carry; the operator surface for inspecting and forgiving it lives here.

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use super::metered::STRIKE_LIMIT;
use super::{Ledger, read_amount};
use crate::invoice::Invoice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeteredInvoiceKind {
    Deposit,
    Final,
}

impl MeteredInvoiceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deposit => "deposit",
            Self::Final => "final",
        }
    }

    fn parse(kind: &str) -> Result<Self> {
        match kind {
            "deposit" => Ok(Self::Deposit),
            "final" => Ok(Self::Final),
            other => bail!("unknown metered invoice kind {other}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredInvoice {
    pub request_id: String,
    pub peer: String,
    pub kind: MeteredInvoiceKind,
    pub invoice: Invoice,
    pub paid: bool,
}

/// One issued and unpaid final invoice.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerDebt {
    pub request_id: String,
    pub amount_msat: Option<u64>,
    pub expires_at_ms: u64,
}

/// A peer with unpaid final invoices, with the identifier `unblock` takes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockedPeer {
    /// Full authenticated mesh endpoint ID, as recorded at admission.
    pub peer: String,
    /// The abbreviated form the console and `/api/status` display.
    pub peer_short: String,
    /// Whether admission is currently refused ([`STRIKE_LIMIT`] reached).
    pub blocked: bool,
    pub debts: Vec<PeerDebt>,
}

/// What [`Ledger::unblock_peer`] forgave.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Forgiven {
    pub peer: String,
    pub invoices: usize,
}

/// Minimum length of a peer prefix accepted by [`Ledger::resolve_blocked_peer`].
pub const MIN_PEER_PREFIX: usize = 8;

impl Ledger {
    /// Record an issued invoice. A deposit must equal the admitted deposit of
    /// an `admitted` request; a final invoice must equal the billed remainder.
    pub fn record_metered_invoice(&self, receipt: &MeteredInvoice) -> Result<()> {
        ensure!(
            receipt.invoice == Invoice::parse(&receipt.invoice.bolt11)?,
            "invalid invoice metadata"
        );
        let amount = receipt
            .invoice
            .amount_msat
            .context("amountless metered invoice")?;
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let row: Option<(String, String, u64, u64)> = tx
            .query_row(
                "SELECT peer,state,deposit_msat,remaining_msat FROM metered_requests WHERE id=?",
                [&receipt.request_id],
                |r| Ok((r.get(0)?, r.get(1)?, read_amount(r, 2)?, read_amount(r, 3)?)),
            )
            .optional()?;
        let Some((peer, state, deposit, remaining)) = row else {
            bail!("unknown metered request")
        };
        ensure!(peer == receipt.peer, "metered invoice peer mismatch");
        match receipt.kind {
            MeteredInvoiceKind::Deposit => ensure!(
                state == "admitted" && deposit > 0 && amount == deposit,
                "metered deposit invoice does not match admitted deposit"
            ),
            MeteredInvoiceKind::Final => ensure!(
                state == "billed" && remaining > 0 && amount == remaining,
                "metered final invoice must equal the billed remainder"
            ),
        }
        tx.execute(
            "INSERT INTO metered_invoices(hash,request_id,peer,kind,invoice,state) VALUES (?1,?2,?3,?4,?5,'unpaid')",
            params![
                receipt.invoice.payment_hash,
                receipt.request_id,
                receipt.peer,
                receipt.kind.as_str(),
                receipt.invoice.bolt11
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Invoices of one request, or every invoice when `request_id` is `None`.
    pub fn metered_invoices(&self, request_id: Option<&str>) -> Result<Vec<MeteredInvoice>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT request_id,peer,kind,invoice,state FROM metered_invoices WHERE (?1 IS NULL OR request_id=?1) ORDER BY kind",
        )?;
        let rows = statement.query_map([request_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })?;
        rows.map(|row| {
            let (request_id, peer, kind, invoice, state) = row?;
            Ok(MeteredInvoice {
                request_id,
                peer,
                kind: MeteredInvoiceKind::parse(&kind)?,
                invoice: Invoice::parse(&invoice)?,
                paid: state == "paid",
            })
        })
        .collect()
    }

    /// The one invoice of `kind` issued for `request_id`, if any.
    pub fn metered_invoice(
        &self,
        request_id: &str,
        kind: MeteredInvoiceKind,
    ) -> Result<Option<MeteredInvoice>> {
        Ok(self
            .metered_invoices(Some(request_id))?
            .into_iter()
            .find(|receipt| receipt.kind == kind))
    }

    /// Issued invoices not yet recorded paid or forgiven.
    pub fn unpaid_metered_invoices(&self) -> Result<Vec<MeteredInvoice>> {
        Ok(self
            .metered_invoices(None)?
            .into_iter()
            .filter(|receipt| !receipt.paid)
            .collect())
    }

    /// Record an invoice paid. Applies to unpaid and forgiven invoices alike:
    /// a receipt that still turns up is a fact, not debt. Returns the
    /// receipt when the hash names a metered invoice.
    pub fn mark_metered_received(&self, hash: &str) -> Result<Option<MeteredInvoice>> {
        let connection = self.lock()?;
        connection.execute(
            "UPDATE metered_invoices SET state='paid' WHERE hash=? AND state IN ('unpaid','forgiven')",
            [hash],
        )?;
        let row: Option<(String, String, String, String)> = connection
            .query_row(
                "SELECT request_id,peer,kind,invoice FROM metered_invoices WHERE hash=?",
                [hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        row.map(|(request_id, peer, kind, invoice)| {
            Ok(MeteredInvoice {
                request_id,
                peer,
                kind: MeteredInvoiceKind::parse(&kind)?,
                invoice: Invoice::parse(&invoice)?,
                paid: true,
            })
        })
        .transpose()
    }

    /// Every peer with an issued, unpaid final invoice. Read-only and
    /// wallet-free, so it works while the node is stopped.
    pub fn blocked_peers(&self) -> Result<Vec<BlockedPeer>> {
        let connection = self.lock()?;
        let mut by_peer: std::collections::BTreeMap<String, Vec<PeerDebt>> = Default::default();
        let mut unpaid = connection.prepare(
            "SELECT peer,request_id,invoice FROM metered_invoices WHERE kind='final' AND state='unpaid' ORDER BY rowid",
        )?;
        for row in unpaid.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (peer, request_id, invoice) = row?;
            let invoice = Invoice::parse(&invoice)?;
            by_peer.entry(peer).or_default().push(PeerDebt {
                request_id,
                amount_msat: invoice.amount_msat,
                expires_at_ms: invoice.expires_at_ms,
            });
        }
        Ok(by_peer
            .into_iter()
            .map(|(peer, debts)| BlockedPeer {
                peer_short: peer.chars().take(10).collect(),
                peer,
                blocked: debts.len() as u64 >= STRIKE_LIMIT,
                debts,
            })
            .collect())
    }

    /// Accept the full endpoint ID or a unique prefix of at least
    /// [`MIN_PEER_PREFIX`] characters among peers with recorded debt, so the
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

    /// Operator override: forgive every unpaid invoice of `peer` so it is no
    /// longer a strike. A later payment is still recorded as received.
    /// Nothing is refunded.
    pub fn unblock_peer(&self, peer: &str) -> Result<Forgiven> {
        let invoices = self.lock()?.execute(
            "UPDATE metered_invoices SET state='forgiven' WHERE peer=?1 AND kind='final' AND state='unpaid'",
            [peer],
        )?;
        ensure!(
            invoices > 0,
            "peer has no recorded debt; nothing to unblock"
        );
        Ok(Forgiven {
            peer: peer.to_owned(),
            invoices,
        })
    }
}
