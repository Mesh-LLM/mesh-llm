use anyhow::{Result, bail, ensure};
use mesh_llm_payments_types::{
    RequestTerms,
    contract::{Empty, FinishRequest, ReconcileResponse, SettleOutputRequest, ops},
    wire::{self, Frame},
};

use crate::mesh::Node;
use crate::network::payments::client::Payments;

/// Recover financial state only. Never regenerate or replay application output
/// after a restart, and never infer failure just from invoice expiry.
pub(crate) async fn recover(node: &Node) -> Result<()> {
    let Some(plugins) = node.plugin_manager().await else {
        return Ok(());
    };
    // Pin one provider for the whole pass, reconcile included: a request's
    // trailing settlement must stay on the ledger that reported it, so a
    // provider replaced mid-pass cannot split one request across two ledgers.
    let payments = Payments::for_plugins(plugins).await;
    // The provider ignores individual uncertain charges so one cannot block
    // other debts; it returns the requests still owed by their sellers.
    let pending: ReconcileResponse = payments.call(ops::RECONCILE, &Empty {}).await?;
    for terms in pending.approved {
        // Bound each peer independently so one unavailable provider cannot stop
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
    loop {
        match wire::read(&mut recv).await? {
            Frame::OutputInvoice {
                request_id,
                tokens,
                invoice,
            } => {
                ensure!(request_id == terms.id, "recovery request mismatch");
                let request = SettleOutputRequest {
                    terms: terms.clone(),
                    tokens,
                    invoice,
                };
                let _: serde_json::Value = payments.call(ops::SETTLE_OUTPUT, &request).await?;
            }
            Frame::Complete => {
                let finish = FinishRequest {
                    id: terms.id.clone(),
                };
                let _: serde_json::Value = payments.call(ops::FINISH, &finish).await?;
                return Ok(());
            }
            Frame::Pending => return Ok(()),
            _ => bail!("invalid recovery response"),
        }
    }
}
