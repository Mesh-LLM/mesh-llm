use super::*;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn simultaneous_requests_only_pay_one_with_one_reservation_available() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = Arc::new(PaymentService::with_provider(
        directory.path(),
        wallet.clone(),
    )?);
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(1000),
    })?;
    let barrier = Arc::new(tokio::sync::Barrier::new(16));
    let mut tasks = tokio::task::JoinSet::new();
    for number in 1..=16 {
        let service = service.clone();
        let barrier = barrier.clone();
        tasks.spawn(async move {
            let id = format!("request-{number}");
            barrier.wait().await;
            if service.await_authorization(&terms(&id, 700)).await.is_err() {
                return Ok::<_, anyhow::Error>(false);
            }
            service
                .pay_charge(&charge(&id, 0, number, 600, 700))
                .await?;
            Ok(true)
        });
    }
    let mut approved = 0;
    while let Some(result) = tasks.join_next().await {
        approved += usize::from(result??);
    }
    assert_eq!(approved, 1);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        300
    );
    let winner = service
        .ledger
        .requests()?
        .into_iter()
        .find(|r| r.state == "approved")
        .unwrap();
    assert_eq!(winner.spent_msat, 610);
    service.ledger.finish(&winner.terms.id)?;
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        390
    );
    Ok(())
}

#[tokio::test]
async fn pending_htlc_survives_restart_outage_and_expiry_until_success() -> Result<()> {
    pending_htlc_recovers(PaymentStatus::Succeeded).await
}

#[tokio::test]
async fn pending_htlc_survives_restart_outage_and_expiry_until_failure() -> Result<()> {
    pending_htlc_recovers(PaymentStatus::Failed).await
}

async fn pending_htlc_recovers(outcome: PaymentStatus) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.pending.store(true, Ordering::SeqCst);
    wallet.lose_response.store(true, Ordering::SeqCst);
    let mut charge = charge("uncertain", 0, 90, 600, 700);
    charge.invoice = invoice_with_expiry(90, 600, 2);
    {
        let service = PaymentService::with_provider(directory.path(), wallet.clone())?;
        service.ledger.set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(1000),
        })?;
        service
            .await_authorization(&terms("uncertain", 700))
            .await?;
        assert!(service.pay_charge(&charge).await.is_err());
    }
    let service = PaymentService::with_provider(directory.path(), wallet.clone())?;
    assert_pending_survives_outage_and_expiry(&service, &wallet, &charge).await?;
    {
        let mut payments = wallet.payments.lock().unwrap();
        let payment = payments.get_mut(&charge.invoice.payment_hash).unwrap();
        payment.status = outcome;
        payment.settled_at_ms = Some(crate::now_ms());
    }
    service.reconcile_pending().await?;
    let replay = service.pay_charge(&charge).await;
    assert_eq!(replay.is_ok(), outcome == PaymentStatus::Succeeded);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert!(service.ledger.pending_charges()?.is_empty());
    if outcome == PaymentStatus::Succeeded {
        service.ledger.finish("uncertain")?;
    } else {
        assert_eq!(
            service.ledger.request_state("uncertain")?.as_deref(),
            Some("failed")
        );
    }
    let remaining = if outcome == PaymentStatus::Succeeded {
        390
    } else {
        1000
    };
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        remaining
    );
    Ok(())
}

#[tokio::test]
async fn expired_automatic_authorization_never_calls_the_wallet() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(directory.path(), wallet.clone())?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })?;
    let mut request = terms("expired", 700);
    request.expires_at_ms = crate::now_ms().saturating_sub(1);
    assert!(service.await_authorization(&request).await.is_err());
    assert!(service.approve(&request.id).await.is_err());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        100_000
    );
    assert!(service.ledger.pending_charges()?.is_empty());
    Ok(())
}

async fn assert_pending_survives_outage_and_expiry(
    service: &PaymentService,
    wallet: &MockWallet,
    charge: &Charge,
) -> Result<()> {
    wallet.lookup_unavailable.store(true, Ordering::SeqCst);
    assert!(service.reconcile_pending().await.is_err());
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        300
    );
    wallet.lookup_unavailable.store(false, Ordering::SeqCst);
    service.reconcile_pending().await?;
    let wait = charge.invoice.expires_at_ms.saturating_sub(crate::now_ms()) + 25;
    tokio::time::sleep(Duration::from_millis(wait)).await;
    service.reconcile_pending().await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    assert!(service.ledger.finish("uncertain").is_err());
    assert!(
        service
            .await_authorization(&terms("second", 700))
            .await
            .is_err()
    );
    Ok(())
}
