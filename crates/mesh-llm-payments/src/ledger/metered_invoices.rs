use anyhow::{Result, bail, ensure};
use rusqlite::{OptionalExtension, params};

use super::Ledger;
use crate::invoice::Invoice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
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
}

#[derive(Clone, serde::Serialize)]
pub struct MeteredInvoice {
    pub request_id: String,
    pub peer: String,
    pub kind: MeteredInvoiceKind,
    pub invoice: Invoice,
    pub paid: bool,
}

impl Ledger {
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
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        super::read_amount(r, 2)?,
                        super::read_amount(r, 3)?,
                    ))
                },
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
        tx.execute("INSERT INTO metered_invoices(hash,request_id,peer,kind,invoice,state) VALUES (?1,?2,?3,?4,?5,'unpaid')", params![receipt.invoice.payment_hash, receipt.request_id, receipt.peer, receipt.kind.as_str(), receipt.invoice.bolt11])?;
        tx.commit()?;
        Ok(())
    }

    pub fn metered_invoices(&self, request_id: Option<&str>) -> Result<Vec<MeteredInvoice>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare("SELECT request_id,peer,kind,invoice,state FROM metered_invoices WHERE (?1 IS NULL OR request_id=?1) ORDER BY kind")?;
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
            let kind = match kind.as_str() {
                "deposit" => MeteredInvoiceKind::Deposit,
                "final" => MeteredInvoiceKind::Final,
                other => bail!("unknown metered invoice kind {other}"),
            };
            Ok(MeteredInvoice {
                request_id,
                peer,
                kind,
                invoice: Invoice::parse(&invoice)?,
                paid: state == "paid",
            })
        })
        .collect()
    }

    pub fn mark_metered_received(&self, hash: &str) -> Result<()> {
        self.lock()?.execute(
            "UPDATE metered_invoices SET state='paid' WHERE hash=? AND state='unpaid'",
            [hash],
        )?;
        Ok(())
    }
}

use anyhow::Context;
