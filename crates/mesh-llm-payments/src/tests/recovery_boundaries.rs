use super::*;
use crate::ledger::receivables::Receivable;

pub(super) fn record_input(service: &PaymentService, id: &str, peer: &str) -> Result<Invoice> {
    let invoice = invoice(199, 1);
    service.ledger.record_receivable(&Receivable {
        request_id: id.into(),
        peer: peer.into(),
        segment: 0,
        invoice: invoice.clone(),
        tokens: 1,
        paid: false,
    })?;
    Ok(invoice)
}

#[tokio::test]
async fn startup_releases_approvals_that_never_reached_the_wallet() -> Result<()> {
    // Released: no charge at all, or an input charge that was prepared but
    // never submitted (the wallet was never called and the seller's prefill did
    // not survive our restart). Retained: anything that may have reached the
    // wallet, and a prepared output charge, which is owed for delivered work.
    for (segment, state, released) in [
        (0, "absent", true),
        (0, "prepared", true),
        (0, "pending", false),
        (1, "prepared", false),
        (1, "pending", false),
    ] {
        let dir = tempfile::tempdir()?;
        let wallet = Arc::new(MockWallet::default());
        {
            let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
            service.ledger.set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(100_000),
            })?;
            service.ledger.propose(&terms("request", 700))?;
            service.approve("request").await?;
            if state != "absent" {
                let charge = charge("request", segment, 3, 600, 700);
                service.ledger.prepare_charge(&charge)?;
                if state == "pending" {
                    service
                        .ledger
                        .begin_submission(&charge.invoice.payment_hash)?;
                }
            }
        }
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        assert_eq!(
            service.ledger.available_budget(1000, crate::now_ms())?,
            if released { 1000 } else { 300 },
            "segment {segment} {state}"
        );
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 0, "startup never pays");
    }
    Ok(())
}

#[tokio::test]
async fn cancellation_and_preparation_have_only_one_winner() -> Result<()> {
    for prepare_first in [false, true] {
        let dir = tempfile::tempdir()?;
        let service = PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default()))?;
        service.ledger.set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(100_000),
        })?;
        service.ledger.propose(&terms("request", 700))?;
        service.approve("request").await?;
        let charge = charge("request", 0, 3, 600, 700);
        if prepare_first {
            service.ledger.prepare_charge(&charge)?;
        }
        service.ledger.cancel_unstarted("request")?;
        if !prepare_first {
            assert!(service.ledger.prepare_charge(&charge).is_err());
        }
        assert_eq!(
            service.ledger.available_budget(1000, crate::now_ms())?,
            if prepare_first { 300 } else { 1000 }
        );
    }
    Ok(())
}

#[tokio::test]
async fn periodic_reconciliation_preserves_live_zero_charge_approval() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default()))?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })?;
    service.ledger.propose(&terms("live", 700))?;
    service.approve("live").await?;
    service.reconcile_pending().await?;
    assert_eq!(
        service.ledger.request_state("live")?.as_deref(),
        Some("approved")
    );
    service.ledger.cancel_unstarted("live")?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })?;
    service.ledger.propose(&terms("pending", 700))?;
    service.ledger.cancel_unstarted("pending")?;
    assert!(service.approve("pending").await.is_err());
    Ok(())
}

/// An input invoice issued `age` ago with a 1 s expiry.
fn aged_invoice(number: u8, age: std::time::Duration) -> Invoice {
    let secret = SecretKey::from_slice(&[7; 32]).unwrap();
    let bolt11 = InvoiceBuilder::new(Currency::Bitcoin)
        .description("test".into())
        .payment_hash(PaymentHash([number; 32]))
        .payment_secret(PaymentSecret([42; 32]))
        .timestamp(std::time::SystemTime::now() - age)
        .expiry_time(std::time::Duration::from_secs(1))
        .min_final_cltv_expiry_delta(144)
        .amount_milli_satoshis(100)
        .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
        .unwrap()
        .to_string();
    Invoice::parse(&bolt11).unwrap()
}

// The wallet reports an issued-but-unpaid invoice as inbound `Pending`
// without claiming. Through the service's own recovery path, that must lapse
// after expiry plus the grace for a zero-delivery request, and not before.
