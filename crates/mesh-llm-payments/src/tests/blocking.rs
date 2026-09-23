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

#[tokio::test]
async fn blocked_peers_are_listed_with_their_identifier_and_unblocked_by_prefix() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    assert!(service.ledger.blocked_peers()?.is_empty());

    // PEER never paid its input invoice; OTHER received output that has not
    // been invoiced yet (for example the wallet was down when it finished).
    service
        .ledger
        .begin_serving("req-a", PEER, &pricing(), 10)?;
    service.ledger.record_receivable(&Receivable {
        request_id: "req-a".into(),
        peer: PEER.into(),
        segment: 0,
        invoice: invoice(1, 100),
        tokens: 10,
        paid: false,
    })?;
    service
        .ledger
        .begin_serving("req-b", OTHER, &pricing(), 10)?;
    service.ledger.record_delivered_tokens("req-b", 7)?;
    service.ledger.finish_serving("req-b")?;
    assert!(
        service
            .ledger
            .begin_serving("req-c", PEER, &pricing(), 10)
            .is_err()
    );
    assert!(
        service
            .ledger
            .begin_serving("req-d", OTHER, &pricing(), 10)
            .is_err()
    );
    assert!(service.ledger.has_outstanding_payment(PEER)?);
    assert!(service.ledger.has_outstanding_payment(OTHER)?);
    assert_eq!(
        service.ledger.uninvoiced_output()?,
        vec!["req-b".to_string()]
    );

    let blocked: Vec<BlockedPeer> =
        serde_json::from_value(service.control(ControlCommand::Blocked).await?)?;
    assert_eq!(blocked.len(), 2);
    let peer = blocked.iter().find(|entry| entry.peer == PEER).unwrap();
    assert_eq!(peer.peer_short, &PEER[..10]);
    assert_eq!(peer.debts.len(), 1);
    assert_eq!(peer.debts[0].kind, "unpaid_invoice");
    assert_eq!(peer.debts[0].request_id, "req-a");
    assert_eq!(peer.debts[0].amount_msat, Some(100));
    assert!(peer.debts[0].expires_at_ms.is_some());
    let other = blocked.iter().find(|entry| entry.peer == OTHER).unwrap();
    assert_eq!(other.debts.len(), 1);
    assert_eq!(other.debts[0].kind, "uninvoiced_output");
    assert_eq!(other.debts[0].request_id, "req-b");
    assert_eq!(other.debts[0].tokens, 7);
    assert_eq!(other.debts[0].amount_msat, None);

    // Too short, unknown and ambiguous identifiers change nothing.
    assert!(unblock(&service, &PEER[..7]).await.is_err());
    assert!(unblock(&service, "ffffffffffff").await.is_err());
    assert!(unblock(&service, &PEER[..12]).await.is_err());
    assert_eq!(service.ledger.blocked_peers()?.len(), 2);

    // A unique prefix and the full identifier both work.
    let forgiven = unblock(&service, &PEER[..13]).await?;
    assert_eq!(
        (
            forgiven.peer.as_str(),
            forgiven.invoices,
            forgiven.output_requests
        ),
        (PEER, 1, 0)
    );
    let forgiven = unblock(&service, OTHER).await?;
    assert_eq!(
        (
            forgiven.peer.as_str(),
            forgiven.invoices,
            forgiven.output_requests
        ),
        (OTHER, 0, 1)
    );
    assert!(service.ledger.blocked_peers()?.is_empty());
    assert!(!service.ledger.has_outstanding_payment(PEER)?);
    assert!(!service.ledger.has_outstanding_payment(OTHER)?);
    assert!(service.ledger.uninvoiced_output()?.is_empty());
    service
        .ledger
        .begin_serving("req-c", PEER, &pricing(), 10)?;
    service
        .ledger
        .begin_serving("req-d", OTHER, &pricing(), 10)?;
    assert!(unblock(&service, PEER).await.is_err());

    // Forgiveness is durable and the delivered token count is preserved.
    drop(service);
    let service = PaymentService::open(dir.path())?;
    assert!(service.ledger.blocked_peers()?.is_empty());
    let (_, _, tokens, finished) = service.ledger.serving_account("req-b")?;
    assert_eq!((tokens, finished), (7, true));
    Ok(())
}

// A slow payment is not debt: an input invoice that expired unpaid, with no
// output delivered, must not block the buyer. Delivered output still does.
#[tokio::test]
async fn expired_input_with_nothing_delivered_does_not_block() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    let expired = invoice_with_expiry(1, 100, 1);
    service.ledger.begin_serving("slow", PEER, &pricing(), 10)?;
    service.ledger.record_receivable(&Receivable {
        request_id: "slow".into(),
        peer: PEER.into(),
        segment: 0,
        invoice: expired.clone(),
        tokens: 10,
        paid: false,
    })?;
    // Not yet expired: still outstanding, and lapsing is refused.
    assert!(
        !service
            .ledger
            .lapse_expired_input("slow", &expired, expired.expires_at_ms - 1)?
    );
    assert!(service.ledger.has_outstanding_payment(PEER)?);
    // Expired with nothing delivered: lapses and the buyer is not blocked.
    assert!(
        service
            .ledger
            .lapse_expired_input("slow", &expired, expired.expires_at_ms)?
    );
    service.ledger.finish_serving("slow")?;
    assert!(!service.ledger.has_outstanding_payment(PEER)?);
    assert!(service.ledger.blocked_peers()?.is_empty());
    service.ledger.begin_serving("next", PEER, &pricing(), 10)?;

    // Output was delivered: the expired input invoice stays debt.
    let delivered = invoice_with_expiry(2, 100, 1);
    service
        .ledger
        .begin_serving("owed", OTHER, &pricing(), 10)?;
    service.ledger.record_receivable(&Receivable {
        request_id: "owed".into(),
        peer: OTHER.into(),
        segment: 0,
        invoice: delivered.clone(),
        tokens: 10,
        paid: false,
    })?;
    service.ledger.record_delivered_tokens("owed", 3)?;
    assert!(
        !service
            .ledger
            .lapse_expired_input("owed", &delivered, delivered.expires_at_ms)?
    );
    assert!(service.ledger.has_outstanding_payment(OTHER)?);
    Ok(())
}
