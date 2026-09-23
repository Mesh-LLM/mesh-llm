use super::*;
use crate::control::ControlCommand;

fn automatic(service: &PaymentService) -> Result<()> {
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(1000),
    })
}

fn send(invoice: &Invoice) -> ControlCommand {
    ControlCommand::Send {
        invoice: invoice.bolt11.clone(),
        amount_msat: None,
        max_fee_msat: 100,
    }
}

#[tokio::test]
async fn declined_and_failed_sends_release_budget_without_manual_finish() -> Result<()> {
    for not_submitted in [true, false] {
        let dir = tempfile::tempdir()?;
        let wallet = Arc::new(MockWallet::default());
        wallet
            .reject_submission
            .store(not_submitted, Ordering::SeqCst);
        wallet
            .terminal_failure
            .store(!not_submitted, Ordering::SeqCst);
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        let invoice = invoice(1, 600);
        assert!(service.control(send(&invoice)).await.is_err());
        assert_eq!(
            service
                .ledger
                .charge_state(&invoice.payment_hash)?
                .as_deref(),
            Some("failed")
        );
        assert_eq!(service.ledger.requests()?[0].state, "failed");
        assert_eq!(
            service.ledger.available_budget(100_000, crate::now_ms())?,
            1000
        );
        service.reconcile_pending().await?;
        assert!(service.control(send(&invoice)).await.is_err());
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[tokio::test]
async fn send_recovery_closes_success_and_failure_after_restart() -> Result<()> {
    for failed in [false, true] {
        let dir = tempfile::tempdir()?;
        let wallet = Arc::new(MockWallet::default());
        wallet.lose_response.store(true, Ordering::SeqCst);
        wallet.terminal_failure.store(failed, Ordering::SeqCst);
        let invoice = invoice(2, 600);
        {
            let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
            automatic(&service)?;
            assert!(service.control(send(&invoice)).await.is_err());
            assert_eq!(
                service.ledger.available_budget(100_000, crate::now_ms())?,
                300
            );
        }
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        service.reconcile_pending().await?;
        assert_eq!(
            service.ledger.requests()?[0].state,
            if failed { "failed" } else { "completed" }
        );
        assert_eq!(
            service.ledger.available_budget(100_000, crate::now_ms())?,
            if failed { 1000 } else { 390 }
        );
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[tokio::test]
async fn unindexed_uncertain_payment_is_never_resubmitted_or_released() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_response.store(true, Ordering::SeqCst);
    let invoice = invoice(3, 600);
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        assert!(service.control(send(&invoice)).await.is_err());
    }
    wallet.hide_payments.store(true, Ordering::SeqCst);
    wallet.reject_submission.store(true, Ordering::SeqCst);
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    assert!(service.reconcile_pending().await.is_err());
    assert!(service.control(send(&invoice)).await.is_err());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        300
    );
    wallet.hide_payments.store(false, Ordering::SeqCst);
    service.reconcile_pending().await?;
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        390
    );
    Ok(())
}

#[tokio::test]
async fn prepared_output_payment_can_resume_without_resubmitting_started_payment() -> Result<()> {
    // Segment 1: the seller has delivered and is owed, so a never-submitted
    // output charge survives a restart and is paid exactly once.
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let charge = charge("prepared", 1, 4, 600, 700);
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        service.await_authorization(&terms("prepared", 700)).await?;
        service.ledger.prepare_charge(&charge)?;
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    service.reconcile_pending().await?;
    service.pay_charge(&charge).await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.requests()?[0].spent_msat, 610);
    Ok(())
}

#[tokio::test]
async fn prepared_input_payment_is_failed_on_reopen_without_a_wallet_call() -> Result<()> {
    // Segment 0 and still `prepared`: `begin_submission` never ran, so the
    // wallet was never called, and the seller's prefill state did not survive
    // our restart. Paying it would buy nothing; fail it and release the
    // reservation instead.
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let charge = charge("unsent", 0, 5, 600, 700);
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        service.await_authorization(&terms("unsent", 700)).await?;
        service.ledger.prepare_charge(&charge)?;
        assert_eq!(
            service.ledger.available_budget(100_000, crate::now_ms())?,
            300
        );
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    service.reconcile_pending().await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        service
            .ledger
            .charge_state(&charge.invoice.payment_hash)?
            .as_deref(),
        Some("failed")
    );
    assert_eq!(service.ledger.requests()?[0].state, "failed");
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        1000,
        "the request's reservation is released"
    );
    // The closed authorization cannot be reused to pay after the fact.
    assert!(service.pay_charge(&charge).await.is_err());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn pending_input_payment_survives_reopen_and_is_reconciled_not_failed() -> Result<()> {
    // `pending` means submission may have reached the wallet: the startup
    // sweep must leave it alone and reconciliation must find the payment.
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.lose_response.store(true, Ordering::SeqCst);
    let charge = charge("inflight", 0, 6, 600, 700);
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        service.await_authorization(&terms("inflight", 700)).await?;
        assert!(service.pay_charge(&charge).await.is_err());
        assert_eq!(
            service
                .ledger
                .charge_state(&charge.invoice.payment_hash)?
                .as_deref(),
            Some("pending")
        );
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    assert_eq!(
        service
            .ledger
            .charge_state(&charge.invoice.payment_hash)?
            .as_deref(),
        Some("pending")
    );
    service.reconcile_pending().await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.ledger.requests()?[0].spent_msat, 610);
    Ok(())
}

#[tokio::test]
async fn prepared_wallet_send_survives_reopen() -> Result<()> {
    // An explicit `wallet send` is user intent, not inference; a crash between
    // preparing and submitting must not fail it behind the user's back.
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let invoice = invoice(7, 100);
    let id = format!("send-{}", invoice.payment_hash);
    let charge = Charge {
        request_id: id.clone(),
        segment: 0,
        invoice: invoice.clone(),
        amount_msat: 100,
        max_total_msat: 200,
    };
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        automatic(&service)?;
        service.ledger.propose(&RequestTerms {
            exchange_id: None,
            id: id.clone(),
            peer: "wallet-send".into(),
            payee: Some(invoice.payee.clone()),
            model: "wallet-send".into(),
            pricing: Pricing {
                input_msat_per_million: 1,
                output_msat_per_million: 1,
                minimum_invoice_msat: 1,
            },
            input_tokens: 0,
            max_output_tokens: 1,
            max_total_msat: 200,
            expires_at_ms: invoice.expires_at_ms,
        })?;
        service.approve(&id).await?;
        service.ledger.prepare_charge(&charge)?;
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    assert_eq!(
        service
            .ledger
            .charge_state(&invoice.payment_hash)?
            .as_deref(),
        Some("prepared")
    );
    let result = service.control(send(&invoice)).await?;
    assert_eq!(result["status"], "succeeded");
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn already_paid_invoice_creates_no_phantom_approval() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let invoice = invoice(5, 100);
    wallet.pay(&invoice, 100, 200).await?;
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    let result = service.control(send(&invoice)).await?;
    assert_eq!(result["status"], "succeeded");
    assert!(service.ledger.requests()?.is_empty());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    // The same applies to an invoice already paid for inference in this ledger.
    let charge = charge("inference", 0, 6, 100, 200);
    automatic(&service)?;
    service
        .await_authorization(&terms("inference", 300))
        .await?;
    service.pay_charge(&charge).await?;
    service.control(send(&charge.invoice)).await?;
    assert_eq!(service.ledger.requests()?.len(), 1);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn failed_inference_invoice_and_invalid_amount_create_no_send_reservations() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    wallet.reject_submission.store(true, Ordering::SeqCst);
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    automatic(&service)?;
    service
        .await_authorization(&terms("inference", 300))
        .await?;
    let charge = charge("inference", 0, 9, 100, 200);
    assert!(service.pay_charge(&charge).await.is_err());
    assert!(service.control(send(&charge.invoice)).await.is_err());
    assert_eq!(service.ledger.requests()?.len(), 1);
    let result = service
        .control(ControlCommand::Send {
            invoice: invoice(10, 100).bolt11,
            amount_msat: Some(101),
            max_fee_msat: 100,
        })
        .await;
    assert!(result.is_err());
    assert_eq!(service.ledger.requests()?.len(), 1);
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        1000
    );
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn already_paid_send_repairs_legacy_phantom_authorizations() -> Result<()> {
    for approved in [false, true] {
        let dir = tempfile::tempdir()?;
        let wallet = Arc::new(MockWallet::default());
        let invoice = invoice(11, 100);
        wallet.pay(&invoice, 100, 200).await?;
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        let id = format!("send-{}", invoice.payment_hash);
        let mut request = terms(&id, 200);
        request.peer = "wallet-send".into();
        service.ledger.set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(100_000),
        })?;
        service.ledger.propose(&request)?;
        if approved {
            service.approve(&id).await?;
        }
        service.control(send(&invoice)).await?;
        assert_eq!(
            service.ledger.request_state(&id)?.as_deref(),
            Some(if approved { "completed" } else { "rejected" })
        );
        assert_eq!(
            service.ledger.available_budget(100_000, crate::now_ms())?,
            100_000
        );
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[tokio::test]
async fn uninvoiced_debt_blocks_admission_after_crash_and_invoice_failure() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let pricing = terms("debt", 1000).pricing;
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        service
            .ledger
            .begin_serving("debt", "debtor", &pricing, 8)?;
        let input = super::recovery_boundaries::record_input(&service, "debt", "debtor")?;
        service.ledger.mark_received(&input.payment_hash)?;
        service.ledger.record_delivered_tokens("debt", 3)?;
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    assert!(
        service
            .ledger
            .begin_serving("second", "debtor", &pricing, 8)
            .is_err()
    );
    service
        .ledger
        .begin_serving("unrelated", "other", &pricing, 8)?;
    wallet.invoice_unavailable.store(true, Ordering::SeqCst);
    assert!(service.recover_output_debt().await.is_err());
    assert!(
        service
            .ledger
            .begin_serving("third", "debtor", &pricing, 8)
            .is_err()
    );
    wallet.invoice_unavailable.store(false, Ordering::SeqCst);
    service.recover_output_debt().await?;
    let invoices = service.ledger.unpaid_invoices("debtor")?;
    assert_eq!(invoices.len(), 1);
    assert!(service.ledger.unpaid_invoices("other")?.is_empty());
    assert!(
        service
            .ledger
            .begin_serving("fourth", "debtor", &pricing, 8)
            .is_err()
    );
    service.ledger.mark_received(&invoices[0].payment_hash)?;
    service
        .ledger
        .begin_serving("settled", "debtor", &pricing, 8)?;
    Ok(())
}

#[tokio::test]
async fn failed_charge_does_not_release_an_uncertain_sibling() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    automatic(&service)?;
    service
        .await_authorization(&terms("siblings", 1000))
        .await?;
    let pending = charge("siblings", 0, 7, 100, 200);
    service.ledger.prepare_charge(&pending)?;
    service
        .ledger
        .begin_submission(&pending.invoice.payment_hash)?;
    wallet.reject_submission.store(true, Ordering::SeqCst);
    assert!(
        service
            .pay_charge(&charge("siblings", 1, 8, 100, 200))
            .await
            .is_err()
    );
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        0
    );
    assert_eq!(service.ledger.pending_charges()?.len(), 1);
    wallet.reject_submission.store(false, Ordering::SeqCst);
    wallet.pay(&pending.invoice, 100, 200).await?;
    service.reconcile_pending().await?;
    assert_eq!(
        service.ledger.available_budget(100_000, crate::now_ms())?,
        890
    );
    assert_eq!(service.ledger.requests()?[0].state, "failed");
    Ok(())
}

#[tokio::test]
async fn fund_honors_fixed_amount_and_rejects_zero() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;

    let amountless = service
        .control(ControlCommand::Fund { amount_msat: None })
        .await?;
    assert_eq!(amountless["amount_msat"], 1000);

    let fixed = service
        .control(ControlCommand::Fund {
            amount_msat: Some(10_000_000),
        })
        .await?;
    assert_eq!(fixed["amount_msat"], 10_000_000);

    assert!(
        service
            .control(ControlCommand::Fund {
                amount_msat: Some(0)
            })
            .await
            .is_err()
    );
    Ok(())
}
