use super::*;
use std::time::Duration;

pub(super) fn publish(
    wallet: &MockWallet,
    invoice: &Invoice,
    inbound: bool,
    status: PaymentStatus,
) {
    wallet.payments.lock().unwrap().insert(
        invoice.payment_hash.clone(),
        Transaction {
            id: invoice.payment_hash.clone(),
            payment_hash: Some(invoice.payment_hash.clone()),
            inbound,
            amount_msat: invoice.amount_msat.unwrap(),
            fee_msat: if inbound { 0 } else { 10 },
            status,
            claiming: false,
            status_msg: None,
            created_at_ms: crate::now_ms(),
            settled_at_ms: (status != PaymentStatus::Pending).then(crate::now_ms),
        },
    );
    wallet.updates.notify_waiters();
}

async fn wait_until_subscribed(wallet: &MockWallet, count: usize) {
    for _ in 0..100 {
        if wallet.waits.load(Ordering::SeqCst) == count {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("wallet watcher did not subscribe");
}

#[tokio::test(start_paused = true)]
async fn incoming_settlement_before_subscription_is_not_missed() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let invoice = invoice(1, 100);
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    service.wait_received(&invoice).await?;
    assert_eq!(wallet.waits.load(Ordering::SeqCst), 1);
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn expired_incoming_invoice_still_recognizes_an_existing_receipt() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let invoice = invoice_with_expiry(7, 100, 0);
    assert!(service.wait_received(&invoice).await.is_err());
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    service.wait_received(&invoice).await?;
    assert_eq!(wallet.waits.load(Ordering::SeqCst), 0);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn settlement_during_initial_lookup_is_not_missed() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let invoice = invoice(1, 100);
    publish(&wallet, &invoice, true, PaymentStatus::Pending);
    wallet.settle_during_lookup.store(true, Ordering::SeqCst);
    service.wait_received(&invoice).await?;
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn incoming_notifications_wake_multiple_waiters_without_polling() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    let invoice = invoice(1, 100);
    let mut waiters = tokio::task::JoinSet::new();
    for _ in 0..2 {
        let service = service.clone();
        let invoice = invoice.clone();
        waiters.spawn(async move { service.wait_received(&invoice).await });
    }
    wait_until_subscribed(&wallet, 2).await;
    let lookups = wallet.lookups.load(Ordering::SeqCst);
    tokio::time::advance(Duration::from_secs(20)).await;
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), lookups);
    publish(
        &wallet,
        &super::invoice(2, 100),
        true,
        PaymentStatus::Succeeded,
    );
    tokio::task::yield_now().await;
    assert!(waiters.try_join_next().is_none());
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    while let Some(result) = waiters.join_next().await {
        result??;
    }
    assert!(service.ledger.requests()?.is_empty());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn outgoing_success_and_failure_are_driven_by_notifications() -> Result<()> {
    outgoing_notification(PaymentStatus::Succeeded).await?;
    outgoing_notification(PaymentStatus::Failed).await
}

async fn outgoing_notification(status: PaymentStatus) -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.pending.store(true, Ordering::SeqCst);
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    service
        .await_authorization(&automatic_terms(&service)?)
        .await?;
    let charge = charge("notified", 0, 3, 100, 200);
    let paying_service = service.clone();
    let paying_charge = charge.clone();
    let paying = tokio::spawn(async move { paying_service.pay_charge(&paying_charge).await });
    wait_until_subscribed(&wallet, 1).await;
    let lookups = wallet.lookups.load(Ordering::SeqCst);
    // An outgoing wait must not impose an invoice-expiry deadline.
    tokio::time::advance(Duration::from_secs(7200)).await;
    assert!(!paying.is_finished());
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), lookups);
    publish(&wallet, &charge.invoice, false, status);
    assert_eq!(paying.await?.is_ok(), status == PaymentStatus::Succeeded);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert!(service.ledger.pending_charges()?.is_empty());
    assert_eq!(
        service.ledger.requests()?[0].spent_msat,
        if status == PaymentStatus::Succeeded {
            110
        } else {
            0
        }
    );
    Ok(())
}

fn automatic_terms(service: &PaymentService) -> Result<RequestTerms> {
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(1000),
    })?;
    Ok(terms("notified", 1000))
}

#[tokio::test(start_paused = true)]
async fn cancelled_observation_preserves_pending_payment_for_restart() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.pending.store(true, Ordering::SeqCst);
    let charge = charge("notified", 0, 4, 100, 200);
    {
        let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
        service
            .await_authorization(&automatic_terms(&service)?)
            .await?;
        let paying_service = service.clone();
        let paying_charge = charge.clone();
        let paying = tokio::spawn(async move { paying_service.pay_charge(&paying_charge).await });
        wait_until_subscribed(&wallet, 1).await;
        paying.abort();
        assert!(paying.await.unwrap_err().is_cancelled());
        assert_eq!(service.ledger.pending_charges()?.len(), 1);
        assert_eq!(
            service.ledger.available_budget(100_000, crate::now_ms())?,
            0
        );
    }
    publish(&wallet, &charge.invoice, false, PaymentStatus::Succeeded);
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    service.reconcile_pending().await?;
    service.pay_charge(&charge).await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.requests()?[0].spent_msat, 110);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn incoming_notification_wait_expires_without_polling() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let error = service
        .wait_received(&invoice_with_expiry(5, 100, 2))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("expired"));
    assert_eq!(wallet.waits.load(Ordering::SeqCst), 1);
    assert_eq!(wallet.lookups.load(Ordering::SeqCst), 2); // plus one final lookup at the deadline
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn notification_error_keeps_reservation_until_authoritative_recovery() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.pending.store(true, Ordering::SeqCst);
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    service
        .await_authorization(&automatic_terms(&service)?)
        .await?;
    let charge = charge("notified", 0, 6, 100, 200);
    let paying_service = service.clone();
    let paying_charge = charge.clone();
    let paying = tokio::spawn(async move { paying_service.pay_charge(&paying_charge).await });
    wait_until_subscribed(&wallet, 1).await;
    wallet.lookup_unavailable.store(true, Ordering::SeqCst);
    wallet.updates.notify_waiters();
    assert!(paying.await?.is_err());
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        0
    );
    wallet.lookup_unavailable.store(false, Ordering::SeqCst);
    publish(&wallet, &charge.invoice, false, PaymentStatus::Succeeded);
    service.reconcile_pending().await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.requests()?[0].spent_msat, 110);
    Ok(())
}

fn publish_claiming(wallet: &MockWallet, invoice: &Invoice) {
    wallet.payments.lock().unwrap().insert(
        invoice.payment_hash.clone(),
        Transaction {
            id: invoice.payment_hash.clone(),
            payment_hash: Some(invoice.payment_hash.clone()),
            inbound: true,
            amount_msat: invoice.amount_msat.unwrap(),
            fee_msat: 0,
            status: PaymentStatus::Pending,
            claiming: true,
            // Display text is provider-specific and must not drive the gate.
            status_msg: Some("Claiming inbound HTLC".into()),
            created_at_ms: crate::now_ms(),
            settled_at_ms: None,
        },
    );
    wallet.updates.notify_waiters();
}

#[tokio::test(start_paused = true)]
async fn arrival_returns_on_claiming_while_receipt_still_requires_completion() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    let invoice = invoice(1, 100);
    publish_claiming(&wallet, &invoice);

    // The gate opens on the transient claiming state, and says so.
    assert_eq!(
        service
            .wait_arrival(&invoice, Duration::from_secs(90))
            .await?,
        Arrival::Claiming
    );

    // Settlement does not: it stays blocked until the payment completes.
    let settling = tokio::spawn({
        let (service, invoice) = (service.clone(), invoice.clone());
        async move { service.wait_received(&invoice).await }
    });
    wait_until_subscribed(&wallet, 1).await;
    tokio::time::advance(Duration::from_secs(20)).await;
    assert!(!settling.is_finished());

    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    settling.await??;
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn arrival_without_a_claiming_state_falls_back_to_completion() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    let invoice = invoice(1, 100);
    publish(&wallet, &invoice, true, PaymentStatus::Pending);
    let waiting = tokio::spawn({
        let (service, invoice) = (service.clone(), invoice.clone());
        async move {
            service
                .wait_arrival(&invoice, Duration::from_secs(90))
                .await
        }
    });
    for _ in 0..100 {
        if wallet.arrival_waits.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(wallet.arrival_waits.load(Ordering::SeqCst), 1);
    tokio::time::advance(Duration::from_secs(20)).await;
    assert!(!waiting.is_finished());
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    assert_eq!(waiting.await??, Arrival::Terminal);
    Ok(())
}

#[tokio::test]
async fn a_failed_incoming_payment_never_opens_the_gate() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let invoice = invoice(1, 100);
    publish(&wallet, &invoice, true, PaymentStatus::Failed);
    assert!(
        service
            .wait_arrival(&invoice, Duration::from_secs(90))
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn arrival_wait_gives_up_at_its_own_deadline_before_the_invoice_expires() -> Result<()> {
    // A caller-supplied deadline shorter than the invoice lifetime ends the
    // wait on its own, with an error distinct from expiry. (The inference
    // seller passes a deadline equal to the invoice lifetime; see `lifetimes`.)
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = Arc::new(PaymentService::with_provider(dir.path(), wallet.clone())?);
    let invoice = invoice_with_expiry(1, 100, 300);
    publish(&wallet, &invoice, true, PaymentStatus::Pending);
    let waiting = tokio::spawn({
        let (service, invoice) = (service.clone(), invoice.clone());
        async move {
            service
                .wait_arrival(&invoice, Duration::from_secs(90))
                .await
        }
    });
    for _ in 0..100 {
        if wallet.arrival_waits.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    tokio::time::advance(Duration::from_secs(89)).await;
    assert!(!waiting.is_finished());
    tokio::time::advance(Duration::from_secs(2)).await;
    let error = waiting.await?.unwrap_err().to_string();
    assert!(error.contains("did not arrive in time"), "{error}");
    // Settlement observation, by contrast, is bounded only by the invoice.
    let settling = tokio::spawn({
        let (service, invoice) = (service.clone(), invoice.clone());
        async move { service.wait_received(&invoice).await }
    });
    wait_until_subscribed(&wallet, 1).await;
    tokio::time::advance(Duration::from_secs(150)).await;
    assert!(!settling.is_finished());
    publish(&wallet, &invoice, true, PaymentStatus::Succeeded);
    settling.await??;
    Ok(())
}

#[tokio::test]
async fn output_invoice_expiry_is_chosen_by_the_host_not_the_wallet() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let pricing = Pricing {
        input_msat_per_million: 1000,
        output_msat_per_million: 1000,
        minimum_invoice_msat: 1,
    };
    service
        .ledger
        .begin_serving("served", "peer", &pricing, 1000)?;
    service.ledger.record_delivered_tokens("served", 10)?;
    service.ledger.finish_serving("served")?;
    // Input receipt is a precondition for invoicing output.
    let input = invoice(9, 1);
    service
        .ledger
        .record_receivable(&crate::ledger::receivables::Receivable {
            request_id: "served".into(),
            peer: "peer".into(),
            segment: 0,
            invoice: input.clone(),
            tokens: 1,
            paid: false,
        })?;
    publish(&wallet, &input, true, PaymentStatus::Succeeded);
    let receipt = service
        .output_receivable("served")
        .await?
        .expect("output invoice");
    assert_eq!(receipt.segment, 1);
    assert_eq!(
        wallet.invoice_expiries.lock().unwrap().as_slice(),
        [crate::lifetimes::OUTPUT_INVOICE_EXPIRY_SECS]
    );
    // Funding invoices get the long human-facing lifetime.
    service
        .control(crate::control::ControlCommand::Fund {
            amount_msat: Some(1000),
        })
        .await?;
    assert_eq!(
        wallet.invoice_expiries.lock().unwrap().last().copied(),
        Some(crate::lifetimes::FUNDING_INVOICE_EXPIRY_SECS)
    );
    Ok(())
}
