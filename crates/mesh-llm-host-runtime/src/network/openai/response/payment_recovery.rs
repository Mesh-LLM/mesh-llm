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
        let mut candidates = vec![original_peer];
        for peer in node.peers().await {
            if peer.lightning_offers.contains_key(&request.terms.model)
                && !candidates.contains(&peer.addr.id)
            {
                candidates.push(peer.addr.id);
            }
        }
        for peer in candidates.into_iter().take(128) {
            if matches!(
                tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    recover_request(node, service, &request.terms, peer)
                )
                .await,
                Ok(Ok(()))
            ) {
                break;
            }
        }
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
    let mut authenticated_invoice = false;
    loop {
        match wire::read(&mut recv).await? {
            Frame::OutputInvoice {
                request_id,
                tokens,
                invoice,
            } => {
                ensure!(request_id == terms.id, "recovery request mismatch");
                super::paid::settle_output(service, terms, tokens, invoice).await?;
                authenticated_invoice = true;
            }
            Frame::Complete => {
                ensure!(
                    peer.to_string() == terms.peer || authenticated_invoice,
                    "recovery completion from an unverified provider"
                );
                service.ledger.finish(&terms.id)?;
                return Ok(());
            }
            Frame::Pending => return Ok(()),
            _ => bail!("invalid recovery response"),
        }
    }
}
