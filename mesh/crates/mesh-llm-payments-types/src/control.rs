//! Local operator commands and approval policy: plain data shared by the
//! host (which validates the request shape) and the engine (which runs it).

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::pricing::Pricing;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalMode {
    #[default]
    FreeOnly,
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

#[derive(Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlCommand {
    Balance,
    InspectInvoice {
        invoice: String,
    },
    Transactions {
        limit: usize,
    },
    Fund {
        #[serde(default)]
        amount_msat: Option<u64>,
    },
    Send {
        invoice: String,
        amount_msat: Option<u64>,
        max_fee_msat: u64,
    },
    Pending,
    /// Peers refused paid inference for recorded debt. Ledger-only.
    Blocked,
    /// Forgive a blocked peer's recorded debt. `peer` is the full endpoint ID
    /// or a unique prefix of at least eight characters. Ledger-only.
    Unblock {
        peer: String,
    },
    /// Forget which wallet backs this ledger so a different wallet plugin can
    /// be used. Refused while any payment could still settle in the pinned
    /// wallet, or while this process has the wallet open. Ledger-only.
    Unpin,
    Policy {
        value: Option<Policy>,
    },
    Pricing,
    /// Show, or set, the minimum Lightning payment in msat (default 1 sat):
    /// the amount a smaller positive final is raised to, and the largest
    /// seller minimum this node pays. Ledger-only.
    MinPayment {
        #[serde(default)]
        value_msat: Option<u64>,
    },
    /// Show, or set, the node-wide admission deposit in msat (default: the
    /// minimum payment, and never below it): what this seller asks peers
    /// outside its trust cache for a model without its own deposit, and the
    /// largest seller deposit this node pays. Shows per-model deposits too.
    /// Ledger-only.
    AdmissionDeposit {
        #[serde(default)]
        value_msat: Option<u64>,
        /// Forget the set deposit so it follows the minimum payment again.
        #[serde(default)]
        reset: bool,
    },
    /// Set the admission deposit of one priced model, or with `None` return
    /// it to the node-wide deposit. Ledger-only.
    SetModelAdmissionDeposit {
        model: String,
        value_msat: Option<u64>,
    },
    /// Patch only the cached-input discount of an already priced model.
    SetCacheDiscount {
        model: String,
        percent: u8,
    },
    SetPricing {
        model: String,
        value: Option<Pricing>,
    },
}
