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
async fn output_recovery_requires_terminal_input_and_reuses_invoice() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let input;
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        service
            .ledger
            .begin_serving("debt", "peer", &terms("debt", 1000).pricing, 8)?;
        input = record_input(&service, "debt", "peer")?;
        service.ledger.record_delivered_tokens("debt", 3)?;
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    assert!(service.output_receivable("debt").await.is_err());
    for status in [
        PaymentStatus::Pending,
        PaymentStatus::Failed,
        PaymentStatus::Succeeded,
    ] {
        super::payment_notifications::publish(&wallet, &input, true, status);
        if status != PaymentStatus::Succeeded {
            assert!(service.output_receivable("debt").await.is_err());
            assert_eq!(service.ledger.receivables(Some("debt"))?.len(), 1);
            assert!(service.ledger.has_outstanding_payment("peer")?);
        }
    }
    wallet.lookup_unavailable.store(true, Ordering::SeqCst);
    assert!(service.output_receivable("debt").await.is_err());
    wallet.lookup_unavailable.store(false, Ordering::SeqCst);
    let first = service.output_receivable("debt").await?.unwrap();
    let again = service.output_receivable("debt").await?.unwrap();
    assert_eq!(first.invoice, again.invoice);
    assert!(service.ledger.receivables(Some("debt"))?[0].paid);
    assert_eq!(service.ledger.receivables(Some("debt"))?.len(), 2);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn startup_releases_only_approvals_without_charges() -> Result<()> {
    for state in ["absent", "prepared", "pending"] {
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
                let charge = charge("request", 0, 3, 600, 700);
                service.ledger.prepare_charge(&charge)?;
                if state == "pending" {
                    service
                        .ledger
                        .begin_submission(&charge.invoice.payment_hash)?;
                }
            }
        }
        let service = PaymentService::with_provider(dir.path(), wallet)?;
        assert_eq!(
            service.ledger.available_budget(1000, crate::now_ms())?,
            if state == "absent" { 1000 } else { 300 }
        );
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

#[tokio::test]
async fn recovery_refreshes_zero_output_input_and_rejects_wrong_direction() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    service
        .ledger
        .begin_serving("empty", "peer", &terms("empty", 1000).pricing, 8)?;
    let input = record_input(&service, "empty", "peer")?;
    service.ledger.finish_serving("empty")?;
    super::payment_notifications::publish(&wallet, &input, false, PaymentStatus::Succeeded);
    assert!(service.recover_output_debt().await.is_err());
    assert!(!service.ledger.receivables(Some("empty"))?[0].paid);
    super::payment_notifications::publish(&wallet, &input, true, PaymentStatus::Succeeded);
    service.recover_output_debt().await?;
    assert!(service.ledger.receivables(Some("empty"))?[0].paid);
    assert!(service.output_receivable("empty").await?.is_none());
    Ok(())
}

#[tokio::test]
async fn unpaid_input_batches_progress_and_wrap_without_deleting_debt() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
    for n in 1..=35 {
        let id = format!("batch-{n}");
        let peer = format!("peer-{n}");
        service
            .ledger
            .begin_serving(&id, &peer, &terms(&id, 1000).pricing, 8)?;
        service.ledger.record_receivable(&Receivable {
            request_id: id,
            peer,
            segment: 0,
            invoice: invoice(n, 1),
            tokens: 1,
            paid: false,
        })?;
    }
    let first = service.ledger.unpaid_input_batch(0)?;
    assert_eq!(first.len(), 32);
    let second = service.ledger.unpaid_input_batch(first.last().unwrap().0)?;
    assert_eq!(second.len(), 3);
    assert_eq!(
        service
            .ledger
            .unpaid_input_batch(second.last().unwrap().0)?,
        first
    );
    // Lookup errors still advance the service cursor, rather than starving later rows.
    wallet.lookup_unavailable.store(true, Ordering::SeqCst);
    assert!(service.recover_output_debt().await.is_err());
    wallet.lookup_unavailable.store(false, Ordering::SeqCst);
    let last = invoice(35, 1);
    super::payment_notifications::publish(&wallet, &last, true, PaymentStatus::Succeeded);
    service.recover_output_debt().await?;
    assert!(service.ledger.receivables(Some("batch-35"))?[0].paid);
    assert_eq!(service.ledger.receivables(None)?.len(), 35);
    Ok(())
}
