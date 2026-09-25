//! Recovery of `pending` charges whose outcome was uncertain. Providers treat
//! a repeat `pay` for the same invoice as a lookup of the existing payment, so
//! recovery may resubmit until the invoice expires.

use std::time::Duration;

use super::*;

const BUDGET: u64 = 100_000;

async fn uncertain_charge(
    wallet: &Arc<MockWallet>,
    directory: &std::path::Path,
    charge: &Charge,
) -> Result<PaymentService> {
    let service = PaymentService::with_provider(directory, wallet.clone())?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(BUDGET),
    })?;
    service.await_authorization(&terms("one", 1000)).await?;
    assert!(service.pay_charge(charge).await.is_err());
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    Ok(service)
}

fn payments_recorded(wallet: &MockWallet) -> usize {
    wallet.payments.lock().unwrap().len()
}

#[tokio::test]
async fn lost_submission_is_resubmitted_and_settles_once() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_submission.store(true, Ordering::SeqCst);
    let charge = charge("one", 0, 1, 100, 200);
    let service = uncertain_charge(&wallet, directory.path(), &charge).await?;
    assert_eq!(payments_recorded(&wallet), 0);

    service.reconcile_pending().await?;

    assert_eq!(wallet.calls.load(Ordering::SeqCst), 2);
    assert_eq!(payments_recorded(&wallet), 1);
    assert!(service.ledger.pending_charges()?.is_empty());
    assert_eq!(service.ledger.requests()?[0].spent_msat, 110);
    service.ledger.finish("one")?;
    assert_eq!(
        service.ledger.available_budget(BUDGET, crate::now_ms())?,
        BUDGET - 110
    );
    Ok(())
}

#[tokio::test]
async fn recorded_payment_reconciles_without_a_second_pay() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_response.store(true, Ordering::SeqCst);
    let charge = charge("one", 0, 1, 100, 200);
    let service = uncertain_charge(&wallet, directory.path(), &charge).await?;

    service.reconcile_pending().await?;

    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(payments_recorded(&wallet), 1);
    assert!(service.ledger.pending_charges()?.is_empty());
    assert_eq!(service.ledger.requests()?[0].spent_msat, 110);
    Ok(())
}

#[tokio::test]
async fn expired_invoice_with_no_payment_fails_and_releases() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_submission.store(true, Ordering::SeqCst);
    let service = PaymentService::with_provider(directory.path(), wallet.clone())?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(BUDGET),
    })?;
    // BOLT11 timestamps have one-second resolution, so a 1 s invoice is already
    // 0-1000 ms from expiry. Build it only after the slow part of the setup:
    // creating it first left the service open, the policy write and the
    // authorization inside that window, and under CPU starvation the invoice
    // expired before `pay_charge` validated it, so the charge was failed
    // instead of becoming pending.
    service.await_authorization(&terms("one", 1000)).await?;
    let mut charge = charge("one", 0, 1, 100, 200);
    charge.invoice = invoice_with_expiry(1, 100, 1);
    assert!(service.pay_charge(&charge).await.is_err());
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    let wait = charge.invoice.expires_at_ms.saturating_sub(crate::now_ms()) + 25;
    tokio::time::sleep(Duration::from_millis(wait)).await;

    assert!(service.reconcile_pending().await.is_err());

    assert_eq!(
        wallet.calls.load(Ordering::SeqCst),
        1,
        "no pay after expiry"
    );
    assert_eq!(payments_recorded(&wallet), 0);
    assert!(service.ledger.pending_charges()?.is_empty());
    assert_eq!(
        service.ledger.request_state("one")?.as_deref(),
        Some("failed")
    );
    assert_eq!(
        service.ledger.available_budget(BUDGET, crate::now_ms())?,
        BUDGET
    );
    // Later scans leave the failed charge alone.
    service.reconcile_pending().await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn resubmission_racing_the_first_landing_returns_the_existing_payment() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    // The first submission reached the node but the response was lost, and
    // the wallet has not indexed it yet when recovery looks it up.
    wallet.lose_response.store(true, Ordering::SeqCst);
    let charge = charge("one", 0, 1, 100, 200);
    let service = uncertain_charge(&wallet, directory.path(), &charge).await?;
    wallet.hide_payments.store(true, Ordering::SeqCst);

    service.reconcile_pending().await?;

    assert_eq!(wallet.calls.load(Ordering::SeqCst), 2);
    assert_eq!(payments_recorded(&wallet), 1);
    assert!(service.ledger.pending_charges()?.is_empty());
    assert_eq!(service.ledger.requests()?[0].spent_msat, 110);
    Ok(())
}

#[tokio::test]
async fn rejected_resubmission_stays_pending_until_expiry() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_submission.store(true, Ordering::SeqCst);
    let charge = charge("one", 0, 1, 100, 200);
    let service = uncertain_charge(&wallet, directory.path(), &charge).await?;
    wallet.reject_submission.store(true, Ordering::SeqCst);

    assert!(service.reconcile_pending().await.is_err());

    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    assert_eq!(
        service.ledger.available_budget(BUDGET, crate::now_ms())?,
        BUDGET - 1000,
        "the request reservation is held"
    );
    Ok(())
}
