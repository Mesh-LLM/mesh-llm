use super::*;
use crate::control::ControlCommand;
use crate::ledger::receivables::{BlockedPeer, Forgiven, Receivable};

// Two peers sharing a 12-character prefix, so prefix resolution is exercised.
const PEER: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER: &str = "0123456789abffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn pricing() -> Pricing {
    Pricing {
        input_msat_per_million: 1000,
        output_msat_per_million: 1000,
        minimum_invoice_msat: 1,
    }
}

async fn unblock(service: &PaymentService, peer: &str) -> Result<Forgiven> {
    Ok(serde_json::from_value(
        service
            .control(ControlCommand::Unblock { peer: peer.into() })
            .await?,
    )?)
}

// A slow or interrupted payment is not debt: an input invoice that expired
// unpaid, past the observation grace, for a finished request that delivered
// nothing, must not block the buyer. Delivered output still does.
