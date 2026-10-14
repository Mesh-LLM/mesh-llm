use super::*;
#[tokio::test]
async fn startup_releases_approvals_that_never_reached_the_wallet() -> Result<()> {
    // Empty and unsubmitted legacy input approvals are released. Submitted
    // input and delivered output remain reserved for reconciliation.
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
            service.ledger.available_budget(crate::now_ms())?,
            if released { 100_000 } else { 99_300 },
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
            service.ledger.available_budget(crate::now_ms())?,
            if prepare_first { 99_300 } else { 100_000 }
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
