use super::*;
use crate::control::ControlCommand;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, STRIKE_LIMIT};
use crate::ledger::metered_invoices::{BlockedPeer, Forgiven, MeteredInvoice, MeteredInvoiceKind};
use mesh_llm_payments_types::metering::MeteredUsage;

// Two peers sharing a 12-character prefix, so prefix resolution is exercised.
const PEER: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER: &str = "0123456789abffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn pricing() -> Pricing {
    Pricing {
        input_msat_per_million: 100_000_000,
        output_msat_per_million: 100_000_000,
    }
}

/// Bill `peer` for one request and leave its final invoice unpaid.
pub(super) fn unpaid_final(
    service: &PaymentService,
    id: &str,
    peer: &str,
    n: u8,
) -> Result<Invoice> {
    service.ledger.metered_admit(&MeteredRequest {
        id,
        peer,
        pricing: &pricing(),
        estimated_input_tokens: 10,
        require_deposit: false,
    })?;
    service.ledger.metered_finalize(
        id,
        MeteredOutcome::Usage(MeteredUsage {
            prompt_tokens: 50,
            completion_tokens: 50,
        }),
    )?;
    let invoice = invoice(n, 10_000);
    service.ledger.record_metered_invoice(&MeteredInvoice {
        request_id: id.into(),
        peer: peer.into(),
        kind: MeteredInvoiceKind::Final,
        invoice: invoice.clone(),
        paid: false,
    })?;
    Ok(invoice)
}

async fn unblock(service: &PaymentService, peer: &str) -> Result<Forgiven> {
    Ok(serde_json::from_value(
        service
            .control(ControlCommand::Unblock { peer: peer.into() })
            .await?,
    )?)
}

#[tokio::test]
async fn blocked_peers_are_listed_with_their_identifier_and_unblocked_by_prefix() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    assert!(service.ledger.blocked_peers()?.is_empty());

    // PEER is at the strike limit; OTHER owes one final invoice and is still
    // admitted.
    for n in 0..STRIKE_LIMIT {
        unpaid_final(&service, &format!("peer-{n}"), PEER, n as u8 + 1)?;
    }
    unpaid_final(&service, "other-0", OTHER, 10)?;
    assert!(
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: "refused",
                peer: PEER,
                pricing: &pricing(),
                estimated_input_tokens: 10,
                require_deposit: true,
            })
            .is_err()
    );
    service.ledger.metered_admit(&MeteredRequest {
        id: "admitted",
        peer: OTHER,
        pricing: &pricing(),
        estimated_input_tokens: 10,
        require_deposit: true,
    })?;

    let blocked: Vec<BlockedPeer> =
        serde_json::from_value(service.control(ControlCommand::Blocked).await?)?;
    assert_eq!(blocked.len(), 2);
    let peer = blocked.iter().find(|entry| entry.peer == PEER).unwrap();
    assert_eq!(peer.peer_short, &PEER[..10]);
    assert!(peer.blocked);
    assert_eq!(peer.debts.len(), STRIKE_LIMIT as usize);
    assert_eq!(peer.debts[0].request_id, "peer-0");
    assert_eq!(peer.debts[0].amount_msat, Some(10_000));
    let other = blocked.iter().find(|entry| entry.peer == OTHER).unwrap();
    assert!(!other.blocked);
    assert_eq!(other.debts.len(), 1);
    assert_eq!(other.debts[0].request_id, "other-0");

    // Too short, unknown and ambiguous identifiers change nothing.
    assert!(unblock(&service, &PEER[..7]).await.is_err());
    assert!(unblock(&service, "ffffffffffff").await.is_err());
    assert!(unblock(&service, &PEER[..12]).await.is_err());
    assert_eq!(service.ledger.blocked_peers()?.len(), 2);

    // A unique prefix and the full identifier both work.
    let forgiven = unblock(&service, &PEER[..13]).await?;
    assert_eq!(
        (forgiven.peer.as_str(), forgiven.invoices),
        (PEER, STRIKE_LIMIT as usize)
    );
    let forgiven = unblock(&service, OTHER).await?;
    assert_eq!((forgiven.peer.as_str(), forgiven.invoices), (OTHER, 1));
    assert!(service.ledger.blocked_peers()?.is_empty());
    assert_eq!(service.ledger.metered_unpaid_finals(PEER)?, 0);
    service.ledger.metered_admit(&MeteredRequest {
        id: "readmitted",
        peer: PEER,
        pricing: &pricing(),
        estimated_input_tokens: 10,
        require_deposit: true,
    })?;
    assert!(unblock(&service, PEER).await.is_err());

    // Forgiveness is durable and the bill itself is preserved.
    drop(service);
    let service = PaymentService::open(dir.path())?;
    assert!(service.ledger.blocked_peers()?.is_empty());
    assert_eq!(
        service.ledger.metered_settlement("peer-0")?.remaining_msat,
        10_000
    );
    Ok(())
}

// Deposits are not strikes: an unpaid or expired deposit never blocks the
// buyer, whatever happened to the request.
#[tokio::test]
async fn unpaid_deposits_and_abandoned_requests_do_not_block() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    let expired = invoice_with_expiry(1, 1000, 0);
    for (id, abandon) in [("slow-a", true), ("slow-b", true), ("slow-c", false)] {
        service.ledger.metered_admit(&MeteredRequest {
            id,
            peer: PEER,
            pricing: &pricing(),
            estimated_input_tokens: 100,
            require_deposit: true,
        })?;
        if id == "slow-a" {
            service.ledger.record_metered_invoice(&MeteredInvoice {
                request_id: id.into(),
                peer: PEER.into(),
                kind: MeteredInvoiceKind::Deposit,
                invoice: expired.clone(),
                paid: false,
            })?;
        }
        if abandon {
            service
                .ledger
                .metered_finalize(id, MeteredOutcome::Abandoned)?;
        }
    }
    // A restart abandons the still-admitted row too.
    drop(service);
    let service = PaymentService::open(dir.path())?;
    assert_eq!(service.ledger.metered_unpaid_finals(PEER)?, 0);
    assert!(service.ledger.blocked_peers()?.is_empty());
    service.ledger.metered_admit(&MeteredRequest {
        id: "next",
        peer: PEER,
        pricing: &pricing(),
        estimated_input_tokens: 10,
        require_deposit: true,
    })?;
    // The expired deposit still counts against wallet switching until it is
    // observed or forgiven, and a receipt that turns up is recorded as paid.
    assert_eq!(
        service
            .ledger
            .wallet_switch_blockers()?
            .unpaid_invoices
            .len(),
        1
    );
    service
        .ledger
        .mark_metered_received(&expired.payment_hash)?;
    assert!(service.ledger.metered_invoices(Some("slow-a"))?[0].paid);
    assert!(
        service
            .ledger
            .wallet_switch_blockers()?
            .unpaid_invoices
            .is_empty()
    );
    Ok(())
}
