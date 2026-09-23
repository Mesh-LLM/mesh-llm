use anyhow::{Result, bail, ensure};
use mesh_llm_payments::{
    ledger::RequestTerms,
    service::PaymentService,
    wire::{self, Frame},
};

use crate::mesh::Node;

/// Recover financial state only. Never regenerate or replay application output
/// after a restart, and never infer failure just from invoice expiry.
pub(crate) async fn recover(node: &Node, service: &PaymentService) -> Result<()> {
    // One uncertain charge or unavailable invoice must not block other debts.
    let _ = service.reconcile_pending().await;
    let _ = service.recover_output_debt().await;
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
                super::paid::settle_output(service, terms, tokens, invoice).await?;
            }
            Frame::Complete => {
                service.ledger.finish(&terms.id)?;
                return Ok(());
            }
            Frame::Pending => return Ok(()),
            _ => bail!("invalid recovery response"),
        }
    }
}
