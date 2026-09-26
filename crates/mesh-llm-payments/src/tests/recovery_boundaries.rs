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
#[tokio::test]
async fn recovery_lapses_an_unpaid_pending_invoice_after_the_grace() -> Result<()> {
    let grace = crate::lifetimes::INPUT_LAPSE_GRACE;
    for (id, number, age, lapses) in [
        ("fresh", 11, std::time::Duration::from_secs(5), false),
        ("stale", 12, grace + std::time::Duration::from_secs(5), true),
    ] {
        let dir = tempfile::tempdir()?;
        let wallet = Arc::new(MockWallet::default());
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        let invoice = aged_invoice(number, age);
        wallet.payments.lock().unwrap().insert(
            invoice.payment_hash.clone(),
            Transaction {
                id: invoice.payment_hash.clone(),
                payment_hash: Some(invoice.payment_hash.clone()),
                inbound: true,
                amount_msat: 100,
                fee_msat: 0,
                status: PaymentStatus::Pending,
                claiming: false,
                status_msg: None,
                created_at_ms: crate::now_ms(),
                settled_at_ms: None,
            },
        );
        service.ledger.begin_serving(
            id,
            "peer",
            &crate::pricing::Pricing {
                input_msat_per_million: 1000,
                output_msat_per_million: 1000,
                minimum_invoice_msat: 1,
            },
            8,
        )?;
        service.ledger.record_receivable(&Receivable {
            request_id: id.into(),
            peer: "peer".into(),
            segment: 0,
            invoice,
            tokens: 1,
            paid: false,
        })?;
        service.ledger.finish_serving(id)?;
        service.recover_output_debt().await?;
        assert_eq!(
            !service.ledger.has_outstanding_payment("peer")?,
            lapses,
            "{id}"
        );
    }
    Ok(())
}

/// Regression: a failed final watermark write followed by a successful close
/// used to freeze the older, lower count. Close now carries the final
/// watermark atomically.
#[tokio::test]
async fn serve_finish_records_final_watermark_atomically() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet)?;
    let pricing = terms("close", 1000).pricing;

    // Last batched flush reached 5; the final flush of 20 "failed" (never
    // written). The atomic close still lands 20.
    service
        .ledger
        .begin_serving("close", "peer", &pricing, 32)?;
    service.ledger.record_delivered_tokens("close", 5)?;
    service.ledger.finish_serving_at("close", 20)?;
    assert_eq!(service.ledger.serving_account("close")?.2, 20);
    assert!(service.ledger.serving_account("close")?.3);
    // Idempotent retry (Drop backstop) with the same or lower count succeeds.
    service.ledger.finish_serving_at("close", 20)?;
    service.ledger.finish_serving_at("close", 7)?;
    assert_eq!(service.ledger.serving_account("close")?.2, 20);

    // A close that already froze a lower count cannot report success for a
    // higher one: the caller sees the loss instead of silently dropping it.
    service
        .ledger
        .begin_serving("frozen", "peer-frozen", &pricing, 32)?;
    service.ledger.record_delivered_tokens("frozen", 5)?;
    service.ledger.finish_serving("frozen")?;
    assert!(service.ledger.finish_serving_at("frozen", 20).is_err());

    // A watermark beyond the output allowance is refused and does not close.
    service
        .ledger
        .begin_serving("over", "peer-over", &pricing, 8)?;
    assert!(service.ledger.finish_serving_at("over", 9).is_err());
    assert!(!service.ledger.serving_account("over")?.3);
    Ok(())
}
