//! Whether the ledger may move to a different wallet.
//!
//! Outstanding state is only meaningful against the wallet that created it.
//! An unresolved outgoing payment can still settle or fail in that wallet, and
//! an unpaid invoice may already have been paid into it without the ledger
//! having recorded it yet, even if the invoice has since expired. Once the
//! ledger points at another wallet it would never learn either outcome.

use anyhow::Result;
use serde::Serialize;

use super::Ledger;

/// Payments that keep the ledger bound to its current wallet.
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct WalletSwitchBlockers {
    /// Payment hashes of outgoing payments that are prepared or pending.
    pub unresolved_payments: Vec<String>,
    /// Payment hashes of issued invoices the ledger still records as unpaid.
    pub unpaid_invoices: Vec<String>,
}

impl WalletSwitchBlockers {
    pub fn is_empty(&self) -> bool {
        self.unresolved_payments.is_empty() && self.unpaid_invoices.is_empty()
    }
}

impl Ledger {
    pub fn wallet_switch_blockers(&self) -> Result<WalletSwitchBlockers> {
        let connection = self.lock()?;
        let unresolved_payments = connection
            .prepare(
                "SELECT hash FROM charges WHERE state IN ('prepared','pending') ORDER BY rowid",
            )?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        let unpaid_invoices = connection
            .prepare("SELECT hash FROM receivables WHERE state='unpaid' ORDER BY rowid")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        Ok(WalletSwitchBlockers {
            unresolved_payments,
            unpaid_invoices,
        })
    }
}
