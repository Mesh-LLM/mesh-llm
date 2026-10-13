//! Buyer-side recovery of metered requests after a restart or disconnect:
//! financial state only. Never regenerates or replays application output, and
//! never infers failure from invoice expiry.

use anyhow::{Result, bail};
use mesh_llm_payments_types::{
    RequestTerms,
    contract::{Empty, FinishRequest, ReconcileResponse, ops},
    metered_wire::{self as wire, Frame},
};

use crate::mesh::Node;
use crate::network::payments::client::Payments;

pub(crate) async fn recover(node: &Node) -> Result<()> {
    let Some(plugins) = node.plugin_manager().await else {
        return Ok(());
    };
    // Pin one provider for the whole pass, reconcile included: a request's
    // trailing settlement must stay on the ledger that reported it, so a
    // provider replaced mid-pass cannot split one request across two ledgers.
    let payments = Payments::for_plugins(plugins).await;
    // The provider ignores individual uncertain charges so one cannot block
    // other debts; it returns the admissions still open with their sellers.
    let pending: ReconcileResponse = payments.call(ops::RECONCILE, &Empty {}).await?;
    for terms in pending.approved {
        // Bound each peer independently so one unavailable seller cannot stop
        // reconciliation of other requests.
        let Ok(original_peer) = terms.peer.parse::<iroh::EndpointId>() else {
            continue;
        };
        // Recovery stays bound to the original authenticated endpoint. Replacing
        // that identity while retaining the wallet/database is not supported.
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            recover_request(node, &payments, &terms, original_peer),
        )
        .await;
    }
    Ok(())
}

async fn recover_request(
    node: &Node,
    payments: &Payments,
    terms: &RequestTerms,
    peer: iroh::EndpointId,
) -> Result<()> {
    let (mut send, mut recv) = node.open_http_tunnel(peer).await?;
    send.write_all(wire::HTTP_UPGRADE).await?;
    wire::write(
        &mut send,
        &Frame::Recover {
            id: terms.id.clone(),
        },
    )
    .await?;
    let admission = mesh_llm_payments_types::metered_wire::AdmissionTerms {
        id: terms.id.clone(),
        model: terms.model.clone(),
        pricing: terms.pricing.clone(),
        prompt_estimate: terms.input_tokens,
        deposit_msat: 0,
    };
    let mut settled = false;
    loop {
        match wire::read(&mut recv).await? {
            settlement @ Frame::Settlement { .. } => {
                // The provider checks the settlement against its stored
                // admission; the terms here only name the request.
                super::paid::settle(payments, &admission, settlement, &Default::default()).await?;
                settled = true;
            }
            Frame::Complete {} => {
                if !settled {
                    bail!("completion before settlement");
                }
                let _: Empty = payments
                    .call(
                        ops::FINISH,
                        &FinishRequest {
                            id: terms.id.clone(),
                        },
                    )
                    .await?;
                return Ok(());
            }
            Frame::Pending {} => return Ok(()),
            _ => bail!("invalid recovery response"),
        }
    }
}
