//! Seller-side metered operations end to end: admission, deposit, settlement,
//! abandonment, recovery and the good-standing cache across restarts.
use super::*;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus, STRIKE_LIMIT};
use crate::ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind};
use mesh_llm_payments_types::contract::{
    FinalBill, ServeAdmitRequest, ServeAdmitResponse, ServeRecoverResponse, ServeSettleRequest,
};
use mesh_llm_payments_types::metering::MeteredUsage;

const MODEL: &str = "metered-model";

// 1 msat per token both ways keeps arithmetic readable.
fn rates() -> Pricing {
    Pricing {
        input_msat_per_million: 10_000_000,
        output_msat_per_million: 10_000_000,
        cache_discount_percent: 0,
    }
}

fn open(dir: &std::path::Path, wallet: Arc<MockWallet>) -> Arc<PaymentService> {
    let service = Arc::new(PaymentService::with_provider(dir, wallet).unwrap());
    service.ledger.set_pricing(MODEL, Some(&rates())).unwrap();
    service
}

fn admit(id: &str, buyer: &str) -> ServeAdmitRequest {
    ServeAdmitRequest {
        billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
        request_id: id.into(),
        buyer: buyer.into(),
        model: MODEL.into(),
        pricing: rates(),
    }
}

fn usage(prompt_tokens: u64, completion_tokens: u64) -> MeteredUsage {
    MeteredUsage {
        prompt_tokens,
        completion_tokens,
        cached_tokens: 0,
    }
}

fn settle(id: &str, prompt: u64, completion: u64) -> ServeSettleRequest {
    ServeSettleRequest {
        request_id: id.into(),
        usage: usage(prompt, completion),
    }
}

fn received(wallet: &MockWallet, inv: &Invoice) {
    wallet.payments.lock().unwrap().insert(
        inv.payment_hash.clone(),
        Transaction {
            id: inv.payment_hash.clone(),
            payment_hash: Some(inv.payment_hash.clone()),
            inbound: true,
            amount_msat: inv.amount_msat.unwrap(),
            fee_msat: 0,
            status: PaymentStatus::Succeeded,
            claiming: false,
            status_msg: None,
            created_at_ms: crate::now_ms(),
            settled_at_ms: Some(crate::now_ms()),
        },
    );
    wallet.updates.notify_waiters();
}

fn deposit_invoice(response: &ServeAdmitResponse) -> Invoice {
    match response {
        ServeAdmitResponse::Deposit { invoice, .. } => invoice.clone(),
        other => panic!("expected a deposit, got {other:?}"),
    }
}

fn billed(response: ServeRecoverResponse) -> FinalBill {
    match response {
        ServeRecoverResponse::Billed { bill } => bill,
        other => panic!("expected a bill, got {other:?}"),
    }
}

#[tokio::test]
async fn deposit_then_final_invoice_for_the_remainder_settles_trust() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    // Prices are the seller's own; a stale or forged quote is refused.
    let mut stale = admit("r", "peer");
    stale.pricing.input_msat_per_million += 1;
    assert!(service.serve_admit(stale).await.is_err());

    let first = service.serve_admit(admit("r", "peer")).await.unwrap();
    let deposit = deposit_invoice(&first);
    assert_eq!(deposit.amount_msat, Some(1000), "flat 1 sat deposit");
    // Replaying the admission returns the same persisted invoice; another
    // buyer reusing the ID is refused.
    assert_eq!(
        deposit_invoice(&service.serve_admit(admit("r", "peer")).await.unwrap()),
        deposit
    );
    assert!(service.serve_admit(admit("r", "other")).await.is_err());
    assert!(matches!(
        service.serve_recover("r").await.unwrap(),
        ServeRecoverResponse::Pending
    ));
    // Nothing may be billed before the deposit is confirmed and dispatched,
    // but the ledger still refuses a final invoice for an unbilled request.
    received(&wallet, &deposit);
    let settled = service.serve_deposit_confirmed("r").await.unwrap();
    assert_eq!(settled.credited_msat, Some(1000));
    assert!(
        service.serve_deposit_confirmed("r").await.is_err(),
        "dispatch is claimed exactly once"
    );

    let bill = service.serve_settle(settle("r", 900, 200)).await.unwrap();
    assert_eq!(
        (
            bill.bill_msat,
            bill.deposit_msat,
            bill.remaining_msat,
            bill.settled
        ),
        (11000, 1000, 10000, false)
    );
    let final_invoice = bill.invoice.clone().expect("final invoice");
    assert_eq!(final_invoice.amount_msat, Some(10000));
    // Idempotent: the same usage returns the same bill and invoice; a
    // different usage or an abandonment after billing is refused.
    assert_eq!(
        service.serve_settle(settle("r", 900, 200)).await.unwrap(),
        bill
    );
    assert!(service.serve_settle(settle("r", 900, 201)).await.is_err());
    assert!(service.serve_abandon("r").is_err());
    assert_eq!(billed(service.serve_recover("r").await.unwrap()), bill);
    assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 1);
    assert!(!service.metered_trusted("peer").await.unwrap());

    received(&wallet, &final_invoice);
    let recovered = billed(service.serve_recover("r").await.unwrap());
    assert!(recovered.settled);
    assert_eq!(recovered.invoice, Some(final_invoice));
    assert_eq!(
        service.ledger.metered_settlement("r").unwrap().status,
        MeteredStatus::Settled
    );
    assert!(service.metered_trusted("peer").await.unwrap());
    // Good standing: the next request is admitted without a deposit and
    // dispatch is claimed by the admission itself.
    assert!(matches!(
        service.serve_admit(admit("next", "peer")).await.unwrap(),
        ServeAdmitResponse::Admitted { .. }
    ));
    // A replay of a dispatched request is never a second dispatch.
    assert!(matches!(
        service.serve_admit(admit("next", "peer")).await.unwrap(),
        ServeAdmitResponse::Replayed { .. }
    ));
    assert!(service.serve_deposit_confirmed("next").await.is_err());
    // Strangers still owe a deposit.
    assert!(matches!(
        service.serve_admit(admit("s", "stranger")).await.unwrap(),
        ServeAdmitResponse::Deposit { .. }
    ));
}

#[tokio::test]
async fn covering_deposit_means_no_final_invoice_and_abandonment_keeps_the_deposit() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let deposit = deposit_invoice(&service.serve_admit(admit("c", "peer")).await.unwrap());
    received(&wallet, &deposit);
    service.serve_deposit_confirmed("c").await.unwrap();
    let bill = service.serve_settle(settle("c", 60, 40)).await.unwrap();
    assert_eq!(
        (
            bill.bill_msat,
            bill.deposit_msat,
            bill.remaining_msat,
            bill.settled
        ),
        (1000, 1000, 0, true)
    );
    assert!(bill.invoice.is_none(), "nothing owed, nothing refunded");
    assert_eq!(wallet.invoice_expiries.lock().unwrap().len(), 1);
    assert!(service.metered_trusted("peer").await.unwrap());

    // An abandoned request: deposit kept, no bill, no invoice, no strike, and
    // settlement afterwards is refused.
    let deposit = deposit_invoice(&service.serve_admit(admit("a", "other")).await.unwrap());
    received(&wallet, &deposit);
    service.serve_deposit_confirmed("a").await.unwrap();
    service.serve_abandon("a").unwrap();
    service.serve_abandon("a").unwrap();
    assert!(service.serve_settle(settle("a", 900, 200)).await.is_err());
    assert!(matches!(
        service.serve_recover("a").await.unwrap(),
        ServeRecoverResponse::Abandoned
    ));
    assert_eq!(
        service
            .ledger
            .metered_settlement("a")
            .unwrap()
            .deposit_paid_msat,
        1000
    );
    assert_eq!(service.ledger.metered_unpaid_finals("other").unwrap(), 0);
    assert!(!service.metered_trusted("other").await.unwrap());
    assert!(service.serve_recover("unknown").await.is_err());
}

#[tokio::test]
async fn unpaid_deposit_never_reduces_the_bill() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let deposit = deposit_invoice(&service.serve_admit(admit("u", "peer")).await.unwrap());
    // The host settles only after confirmation; a direct settle with the
    // deposit unpaid bills the full amount and the deposit arriving later
    // changes nothing already fixed.
    let bill = service.serve_settle(settle("u", 900, 200)).await.unwrap();
    assert_eq!((bill.deposit_msat, bill.remaining_msat), (0, 11000));
    received(&wallet, &deposit);
    let bill = billed(service.serve_recover("u").await.unwrap());
    assert_eq!((bill.deposit_msat, bill.remaining_msat), (0, 11000));
    assert!(
        service
            .ledger
            .metered_invoice("u", MeteredInvoiceKind::Deposit)
            .unwrap()
            .unwrap()
            .paid
    );
}

#[tokio::test]
async fn warm_peer_skips_deposits_with_finals_in_flight_until_strike_limit() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let deposit = deposit_invoice(&service.serve_admit(admit("seed", "peer")).await.unwrap());
    received(&wallet, &deposit);
    service.serve_deposit_confirmed("seed").await.unwrap();
    assert!(
        service
            .serve_settle(settle("seed", 50, 50))
            .await
            .unwrap()
            .settled
    );

    for n in 0..STRIKE_LIMIT {
        let id = format!("warm-{n}");
        assert!(service.metered_trusted("peer").await.unwrap());
        assert!(matches!(
            service.serve_admit(admit(&id, "peer")).await.unwrap(),
            ServeAdmitResponse::Admitted { .. }
        ));
        assert!(
            service
                .serve_settle(settle(&id, 100, 100))
                .await
                .unwrap()
                .invoice
                .is_some()
        );
        assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), n + 1);
    }
    assert!(service.metered_trusted("peer").await.unwrap());
    assert!(matches!(
        service
            .serve_admit(admit("blocked-warm", "peer"))
            .await
            .unwrap(),
        ServeAdmitResponse::Blocked { .. }
    ));
    assert_eq!(
        wallet.invoice_expiries.lock().unwrap().len(),
        4,
        "one cold deposit and three finals; no repeat deposits"
    );
}

#[tokio::test]
async fn three_unpaid_final_invoices_block_admission_until_one_is_paid() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let mut finals = Vec::new();
    for n in 0..STRIKE_LIMIT {
        let id = format!("s{n}");
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: &id,
                peer: "peer",
                pricing: &rates(),
                require_deposit: false,
            })
            .unwrap();
        // A billed remainder without an issued invoice is not a strike.
        service
            .ledger
            .metered_finalize(&id, MeteredOutcome::Usage(usage(100, 100)))
            .unwrap();
        assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), n);
        finals.push(
            service
                .serve_settle(settle(&id, 100, 100))
                .await
                .unwrap()
                .invoice
                .unwrap(),
        );
    }
    assert!(matches!(
        service.serve_admit(admit("blocked", "peer")).await.unwrap(),
        ServeAdmitResponse::Blocked { .. }
    ));
    assert!(service.ledger.metered_owner("blocked").unwrap().is_none());
    assert!(matches!(
        service.serve_admit(admit("fine", "other")).await.unwrap(),
        ServeAdmitResponse::Deposit { .. }
    ));
    // Paying one final (observed through reconciliation) unblocks.
    received(&wallet, &finals[0]);
    service.reconcile().await.unwrap();
    assert_eq!(
        service.ledger.metered_unpaid_finals("peer").unwrap(),
        STRIKE_LIMIT - 1
    );
    assert!(matches!(
        service.serve_admit(admit("again", "peer")).await.unwrap(),
        ServeAdmitResponse::Admitted { .. }
    ));
}

#[tokio::test]
async fn restart_abandons_in_flight_requests_and_never_redispatches() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let deposit;
    {
        let service = open(dir.path(), wallet.clone());
        deposit = deposit_invoice(&service.serve_admit(admit("never", "peer")).await.unwrap());
        let paid = deposit_invoice(
            &service
                .serve_admit(admit("dispatched", "peer"))
                .await
                .unwrap(),
        );
        received(&wallet, &paid);
        service.serve_deposit_confirmed("dispatched").await.unwrap();
    }
    let service = open(dir.path(), wallet.clone());
    for id in ["dispatched", "never"] {
        assert!(matches!(
            service.serve_recover(id).await.unwrap(),
            ServeRecoverResponse::Abandoned
        ));
        assert!(service.serve_settle(settle(id, 900, 200)).await.is_err());
    }
    // A deposit paid after abandonment is observed but creates nothing, and
    // can never start the backend.
    received(&wallet, &deposit);
    assert!(service.serve_deposit_confirmed("never").await.is_err());
    assert!(
        service
            .ledger
            .metered_invoice("never", MeteredInvoiceKind::Deposit)
            .unwrap()
            .unwrap()
            .paid
    );
    assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 0);
    assert!(!service.metered_trusted("peer").await.unwrap());
}

#[tokio::test]
async fn expired_nonpayment_survives_restart_without_revoking_reearned_trust() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let expired = invoice_with_expiry(91, 2000, 0);
    assert!(expired.expires_at_ms <= crate::now_ms());
    {
        let service = open(dir.path(), wallet.clone());
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: "unpaid",
                peer: "peer",
                pricing: &rates(),
                require_deposit: false,
            })
            .unwrap();
        service
            .ledger
            .metered_finalize("unpaid", MeteredOutcome::Usage(usage(100, 100)))
            .unwrap();
        service
            .ledger
            .record_metered_invoice(&MeteredInvoice {
                request_id: "unpaid".into(),
                peer: "peer".into(),
                kind: MeteredInvoiceKind::Final,
                invoice: expired.clone(),
                paid: false,
            })
            .unwrap();
        assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 1);
    }
    let service = open(dir.path(), wallet.clone());
    let debts = service.ledger.metered_invoices(Some("unpaid")).unwrap();
    assert_eq!(debts.len(), 1);
    assert_eq!(debts[0].invoice, expired);
    assert!(!debts[0].paid);
    // Recovery returns the stored, expired invoice rather than a fresh one.
    let bill = billed(service.serve_recover("unpaid").await.unwrap());
    assert_eq!(bill.invoice, Some(expired.clone()));
    assert!(!bill.settled);

    // Re-earn cache standing after restart. Persisted debt is independent of
    // trust: both peers skip deposits while below the separate strike limit.
    for peer in ["peer", "other"] {
        let seed = format!("settled-{peer}");
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: &seed,
                peer,
                pricing: &rates(),
                require_deposit: false,
            })
            .unwrap();
        assert!(
            service
                .serve_settle(settle(&seed, 0, 0))
                .await
                .unwrap()
                .settled
        );
        let trusted = service.metered_trusted(peer).await.unwrap();
        assert!(trusted, "{peer} remains warm with a final outstanding");
        let later = service
            .serve_admit(admit(&format!("later-{peer}"), peer))
            .await
            .unwrap();
        assert_eq!(
            matches!(later, ServeAdmitResponse::Admitted { .. }),
            trusted,
            "{peer}"
        );
    }
    // Clearing the debt changes strikes, not the already re-earned standing.
    service
        .ledger
        .mark_metered_received(&expired.payment_hash)
        .unwrap();
    assert!(service.metered_trusted("peer").await.unwrap());
    assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 0);
}

#[tokio::test]
async fn buyer_that_dropped_discount_is_refused_before_deposit() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let mut price = rates();
    price.cache_discount_percent = 50;
    service.ledger.set_pricing(MODEL, Some(&price))?;
    assert!(
        service
            .serve_admit(admit("old-buyer", "peer"))
            .await
            .is_err()
    );
    assert!(service.ledger.metered_owner("old-buyer")?.is_none());
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn restart_recovery_accepts_a_receipt_net_of_the_receivers_fee() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let final_invoice = invoice_with_expiry(92, 1000, 3600);
    let short = invoice_with_expiry(93, 1000, 3600);
    {
        let service = open(dir.path(), wallet.clone());
        for (id, invoice) in [("net", &final_invoice), ("short", &short)] {
            service
                .ledger
                .metered_admit(&MeteredRequest {
                    id,
                    peer: "peer",
                    pricing: &rates(),
                    require_deposit: false,
                })
                .unwrap();
            service
                .ledger
                .metered_finalize(id, MeteredOutcome::Usage(usage(50, 50)))
                .unwrap();
            service
                .ledger
                .record_metered_invoice(&MeteredInvoice {
                    request_id: id.into(),
                    peer: "peer".into(),
                    kind: MeteredInvoiceKind::Final,
                    invoice: invoice.clone(),
                    paid: false,
                })
                .unwrap();
        }
        assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 2);
    }
    // The receiving wallet credits 995 and reports its own 5-msat fee, as
    // Lexe does; a genuinely short payment (990 + 5) must stay unpaid.
    for (invoice, credited) in [(&final_invoice, 995), (&short, 990)] {
        received(&wallet, invoice);
        let mut payments = wallet.payments.lock().unwrap();
        let payment = payments.get_mut(&invoice.payment_hash).unwrap();
        payment.amount_msat = credited;
        payment.fee_msat = 5;
    }
    let service = open(dir.path(), wallet.clone());
    service.metered_refresh("net").await.unwrap();
    assert!(service.metered_refresh("short").await.is_err());
    let paid: Vec<_> = service
        .ledger
        .metered_invoices(None)
        .unwrap()
        .into_iter()
        .map(|receipt| (receipt.request_id, receipt.paid))
        .collect();
    assert!(paid.contains(&("net".into(), true)), "{paid:?}");
    assert!(paid.contains(&("short".into(), false)), "{paid:?}");
    assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), 1);
}

#[tokio::test]
async fn an_overridden_minimum_payment_is_the_deposit_and_the_final_round_up() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    assert!(service.ledger.set_min_payment_msat(0).is_err());
    service.ledger.set_min_payment_msat(2000).unwrap();
    let first = service.serve_admit(admit("r", "peer")).await.unwrap();
    let ServeAdmitResponse::Deposit {
        invoice,
        min_payment_msat,
        ..
    } = first
    else {
        panic!("expected a deposit, got {first:?}");
    };
    assert_eq!((invoice.amount_msat, min_payment_msat), (Some(2000), 2000));
    received(&wallet, &invoice);
    service.serve_deposit_confirmed("r").await.unwrap();
    // 10 msat per token: a 2,500-msat bill leaves 500, raised to the 2,000 floor.
    let bill = service.serve_settle(settle("r", 150, 100)).await.unwrap();
    assert_eq!(
        (bill.bill_msat, bill.deposit_msat, bill.remaining_msat),
        (2500, 2000, 2000)
    );
    assert_eq!(bill.invoice.unwrap().amount_msat, Some(2000));
}

#[tokio::test]
async fn a_never_paid_final_gives_up_promptly_and_still_blocks_the_buyer() {
    // The seller's bounded final-settlement wait must change no accounting:
    // giving up leaves each final an unpaid strike, three of them block the
    // buyer, and a payment that lands later is still recorded.
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = open(dir.path(), wallet.clone());
    let mut finals = Vec::new();
    for n in 0..STRIKE_LIMIT {
        let id = format!("never-{n}");
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: &id,
                peer: "peer",
                pricing: &rates(),
                require_deposit: false,
            })
            .unwrap();
        let invoice = service
            .serve_settle(settle(&id, 100, 100))
            .await
            .unwrap()
            .invoice
            .unwrap();
        let started = std::time::Instant::now();
        let waited = service
            .settle_received_within(&invoice, std::time::Duration::from_millis(50))
            .await;
        assert!(waited.is_err(), "an unpaid final must not settle");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "the wait is bounded by the deadline, not the hour-long invoice expiry"
        );
        assert_eq!(service.ledger.metered_unpaid_finals("peer").unwrap(), n + 1);
        finals.push(invoice);
    }
    assert!(matches!(
        service.serve_admit(admit("after", "peer")).await.unwrap(),
        ServeAdmitResponse::Blocked { .. }
    ));

    // A late payment, after the seller stopped waiting, is still recorded by
    // reconciliation and lifts the block.
    received(&wallet, &finals[0]);
    service.metered_refresh_all().await.unwrap();
    assert_eq!(
        service.ledger.metered_unpaid_finals("peer").unwrap(),
        STRIKE_LIMIT - 1
    );
    assert!(!matches!(
        service
            .serve_admit(admit("unblocked", "peer"))
            .await
            .unwrap(),
        ServeAdmitResponse::Blocked { .. }
    ));
}
