use anyhow::Result;
use mesh_llm_payments::{ledger::RequestTerms, service::PaymentService};

use crate::mesh::Node;

/// Recover financial state only. Never regenerate or replay application output
/// after a restart, and never infer failure just from invoice expiry.
pub(crate) async fn recover(node: &Node, service: &PaymentService) -> Result<()> {
    // One uncertain charge or unavailable invoice must not block other debts.
    let _ = service.reconcile_pending().await;
    for request in service.ledger.requests()? {
        if request.state != "approved" || request.terms.peer == "wallet-send" {
            continue;
        }
        // Bound each peer independently so one unavailable provider cannot stop
        // reconciliation of other requests.
        let Ok(original_peer) = request.terms.peer.parse::<iroh::EndpointId>() else {
            continue;
        };
        // Recovery stays bound to the original authenticated endpoint. Replacing
        // that identity while retaining the wallet/database is not supported.
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            recover_request(node, service, &request.terms, original_peer),
        )
        .await;
    }
    Ok(())
}

async fn recover_request(
    node: &Node,
    service: &PaymentService,
    terms: &RequestTerms,
    peer: iroh::EndpointId,
) -> Result<()> {
    if let Some(admission) = service.metered_authorization(&terms.id)? {
        super::metered_payer::recover(node, service, terms, &admission, peer).await?;
    }
    Ok(())
}
