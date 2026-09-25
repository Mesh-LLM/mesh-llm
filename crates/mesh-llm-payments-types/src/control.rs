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
    Policy {
        value: Option<Policy>,
    },
    Pricing,
    SetPricing {
        model: String,
        value: Option<Pricing>,
    },
}
