//! Buyer-side metered operations: an admission is accepted against the local
//! cap and pinned, the deposit and final are paid only at the admitted
//! amounts, settlements are checked against the admission, and recovery only
//! observes charges.
use super::*;
use mesh_llm_payments_types::contract::{
    AcceptAdmissionRequest, PayDepositRequest, SettleFinalRequest,
};
use mesh_llm_payments_types::metered_wire::AdmissionTerms;
use mesh_llm_payments_types::metering::MeteredUsage;

// 10 msat per token both ways; the deposit is the flat 1000 msat.
fn rates() -> Pricing {
    Pricing {
        input_msat_per_million: 10_000_000,
        output_msat_per_million: 10_000_000,
        cache_discount_percent: 0,
    }
}

fn local(id: &str, cap: u64) -> RequestTerms {
    RequestTerms {
        pricing: rates(),
        max_output_tokens: 100,
        ..terms(id, cap)
    }
}

fn admission(id: &str, deposit_msat: u64) -> AdmissionTerms {
    AdmissionTerms {
        id: id.into(),
        model: "test".into(),
        pricing: rates(),
        deposit_msat,
    }
}

fn open(dir: &std::path::Path) -> (Arc<MockWallet>, PaymentService) {
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir, wallet.clone()).unwrap();
    service
        .ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(100_000),
        })
        .unwrap();
    (wallet, service)
}

fn settlement(
    id: &str,
    usage: Option<(u64, u64)>,
    bill: Option<u64>,
    applied: u64,
    invoice: Option<Invoice>,
) -> SettleFinalRequest {
    SettleFinalRequest {
        request_id: id.into(),
        usage: usage.map(|(prompt_tokens, completion_tokens)| MeteredUsage {
            prompt_tokens,
            completion_tokens,
            cached_tokens: 0,
        }),
        bill_msat: bill,
        deposit_applied_msat: applied,
        invoice,
    }
}

#[tokio::test]
async fn an_admission_must_restate_the_terms_and_the_defined_deposit() {
    let dir = tempfile::tempdir().unwrap();
    let (_, service) = open(dir.path());
    for (admission, reason) in [
        (admission("other", 1000), "id"),
        (admission("r", 999), "deposit share"),
    ] {
        assert!(
            service
                .accept_admission(AcceptAdmissionRequest {
                    billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                    terms: local("r", 50_000),
                    admission,
                })
                .await
                .is_err(),
            "{reason} mismatch must be refused"
        );
    }
    assert!(service.ledger.requests().unwrap().is_empty());
    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms: local("r", 50_000),
            admission: admission("r", 1000),
        })
        .await
        .unwrap();
    assert_eq!(service.ledger.requests().unwrap()[0].state, "approved");
    // A changed admission under the same id is refused; the same one replays.
    assert!(
        service
            .accept_admission(AcceptAdmissionRequest {
                billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                terms: local("r", 50_000),
                admission: admission("r", 0),
            })
            .await
            .is_err()
    );
}

#[tokio::test]
async fn deposit_and_final_are_paid_only_at_the_admitted_amounts_to_one_payee() {
    let dir = tempfile::tempdir().unwrap();
    let (wallet, service) = open(dir.path());
    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms: local("r", 50_000),
            admission: admission("r", 1000),
        })
        .await
        .unwrap();
    assert!(
        service
            .pay_deposit(PayDepositRequest {
                request_id: "r".into(),
                invoice: invoice(1, 1001),
            })
            .await
            .is_err(),
        "a deposit invoice for another amount is refused"
    );
    assert!(
        service
            .pay_deposit(PayDepositRequest {
                request_id: "unknown".into(),
                invoice: invoice(1, 1000),
            })
            .await
            .is_err()
    );
    let deposit = service
        .pay_deposit(PayDepositRequest {
            request_id: "r".into(),
            invoice: invoice(1, 1000),
        })
        .await
        .unwrap();
    assert_eq!(deposit.amount_msat, 1000);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    // Bill 11000 at the admitted rates, 1000 applied: final invoice for 10000.
    for (request, reason) in [
        (
            settlement(
                "r",
                Some((900, 200)),
                Some(10_000),
                1000,
                Some(invoice(2, 9_000)),
            ),
            "bill not at the admitted rates",
        ),
        (
            settlement(
                "r",
                Some((900, 200)),
                Some(11_000),
                0,
                Some(invoice(2, 11_000)),
            ),
            "paid deposit omitted",
        ),
        (
            settlement(
                "r",
                Some((900, 200)),
                Some(11_000),
                500,
                Some(invoice(2, 10_500)),
            ),
            "paid deposit only partly applied",
        ),
        (
            settlement(
                "r",
                Some((900, 200)),
                Some(11_000),
                2000,
                Some(invoice(2, 9_000)),
            ),
            "more deposit applied than paid",
        ),
        (
            settlement(
                "r",
                Some((900, 200)),
                Some(11_000),
                1000,
                Some(invoice(2, 10_001)),
            ),
            "final invoice not for the remainder",
        ),
        (
            settlement("r", Some((900, 200)), Some(11_000), 300, None),
            "final invoice missing",
        ),
        (
            settlement(
                "r",
                Some((9_000, 200)),
                Some(92_000),
                1000,
                Some(invoice(2, 91_000)),
            ),
            "bill above the local cap",
        ),
    ] {
        assert!(service.settle_final(request).await.is_err(), "{reason}");
    }
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1, "nothing was paid");
    let settled = service
        .settle_final(settlement(
            "r",
            Some((900, 200)),
            Some(11_000),
            1000,
            Some(invoice(2, 10_000)),
        ))
        .await
        .unwrap();
    assert_eq!(settled.bill_msat, 11_000);
    assert_eq!(settled.payment.unwrap().amount_msat, 10_000);
    service.finish_metered("r").await.unwrap();
    let request = &service.ledger.requests().unwrap()[0];
    assert_eq!(request.state, "completed");
    assert_eq!(request.spent_msat, 1000 + 10_000 + 20);
}

#[tokio::test]
async fn an_abandoned_settlement_creates_no_charge_and_a_covered_bill_needs_no_invoice() {
    let dir = tempfile::tempdir().unwrap();
    let (wallet, service) = open(dir.path());
    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms: local("a", 50_000),
            admission: admission("a", 1000),
        })
        .await
        .unwrap();
    assert!(
        service
            .settle_final(settlement("a", None, Some(1), 0, None))
            .await
            .is_err(),
        "an abandoned request cannot carry a bill"
    );
    let settled = service
        .settle_final(settlement("a", None, None, 0, None))
        .await
        .unwrap();
    assert_eq!((settled.bill_msat, settled.payment), (0, None));
    // No deposit was ever submitted: finishing retires it, no payment moves.
    service.finish_metered("a").await.unwrap();
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    assert_eq!(service.ledger.requests().unwrap()[0].state, "completed");

    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms: local("c", 50_000),
            admission: admission("c", 1000),
        })
        .await
        .unwrap();
    service
        .pay_deposit(PayDepositRequest {
            request_id: "c".into(),
            invoice: invoice(3, 1000),
        })
        .await
        .unwrap();
    let settled = service
        .settle_final(settlement("c", Some((50, 50)), Some(1000), 1000, None))
        .await
        .unwrap();
    assert_eq!((settled.bill_msat, settled.payment), (1000, None));
    service.finish_metered("c").await.unwrap();
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_restart_keeps_an_accepted_admission_open_for_recovery() {
    let dir = tempfile::tempdir().unwrap();
    {
        let (_, service) = open(dir.path());
        service
            .accept_admission(AcceptAdmissionRequest {
                billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                terms: local("r", 50_000),
                admission: admission("r", 0),
            })
            .await
            .unwrap();
    }
    // A trusted (no-deposit) admission has no charge; a restart must not
    // close it as abandoned, since its final invoice may still be owed.
    let (_, service) = open(dir.path());
    let reconciled = service.reconcile().await.unwrap();
    assert_eq!(reconciled.approved.len(), 1);
    assert_eq!(reconciled.approved[0].id, "r");
    assert_eq!(service.ledger.requests().unwrap()[0].state, "approved");
}

// Seed historical payment intent with an already-expired signed invoice rather
// than sleeping for expiry. The mock wallet represents authoritative history.
async fn expired_final_history(
    dir: &std::path::Path,
    wallet: &Arc<MockWallet>,
    service: &PaymentService,
    recorded_success: bool,
) -> Invoice {
    let invoice = invoice_with_expiry(7, 11_000, 0);
    let cap = crate::pricing::payment_cap_msat(11_000).unwrap();
    rusqlite::Connection::open(dir.join("payments.sqlite3"))
        .unwrap()
        .execute(
            "INSERT INTO charges(hash,request_id,segment,invoice,amount,max_total,state,total) \
             VALUES (?1,'r',1,?2,11000,?3,'pending',0)",
            rusqlite::params![
                invoice.payment_hash,
                invoice.bolt11,
                i64::try_from(cap).unwrap()
            ],
        )
        .unwrap();
    let paid = wallet.pay(&invoice, 11_000, cap).await.unwrap();
    if recorded_success {
        service.ledger.reconcile(&paid, crate::now_ms()).unwrap();
    }
    invoice
}

#[tokio::test]
async fn expired_paid_final_recovers_after_restart_without_another_submission() {
    for recorded_success in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (wallet, service) = open(dir.path());
        service
            .accept_admission(AcceptAdmissionRequest {
                billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                terms: local("r", 50_000),
                admission: admission("r", 0),
            })
            .await
            .unwrap();
        let invoice = expired_final_history(dir.path(), &wallet, &service, recorded_success).await;
        assert!(invoice.expires_at_ms <= crate::now_ms());
        drop(service);
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        let settled = service
            .settle_final(settlement(
                "r",
                Some((900, 200)),
                Some(11_000),
                0,
                Some(invoice),
            ))
            .await
            .unwrap();
        assert_eq!(settled.payment.unwrap().status, PaymentStatus::Succeeded);
        service.finish_metered("r").await.unwrap();
        assert_eq!(service.ledger.requests().unwrap()[0].state, "completed");
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
        assert_eq!(service.ledger.requests().unwrap()[0].spent_msat, 11_010);
    }
}

#[tokio::test]
async fn final_invoice_still_requires_authentic_metadata_and_unexpired_new_intent() {
    for tampered in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (wallet, service) = open(dir.path());
        service
            .accept_admission(AcceptAdmissionRequest {
                billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
                terms: local("r", 50_000),
                admission: admission("r", 0),
            })
            .await
            .unwrap();
        let mut invoice = invoice_with_expiry(8, 11_000, 0);
        if tampered {
            invoice.expires_at_ms = crate::now_ms() + 3_600_000;
        }
        assert!(
            service
                .settle_final(settlement(
                    "r",
                    Some((900, 200)),
                    Some(11_000),
                    0,
                    Some(invoice),
                ))
                .await
                .is_err()
        );
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn full_credit_does_not_require_the_buyers_deposit_observation_to_finish() {
    let dir = tempfile::tempdir().unwrap();
    let (wallet, service) = open(dir.path());
    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms: local("r", 50_000),
            admission: admission("r", 1000),
        })
        .await
        .unwrap();
    wallet.lose_response.store(true, Ordering::SeqCst);
    assert!(
        service
            .pay_deposit(PayDepositRequest {
                request_id: "r".into(),
                invoice: invoice(9, 1000),
            })
            .await
            .is_err()
    );
    assert_eq!(
        service
            .ledger
            .charge_state(&invoice(9, 1000).payment_hash)
            .unwrap()
            .as_deref(),
        Some("pending")
    );
    // Seller observed the deposit first. A bill smaller than the deposit
    // credits only the bill, with no refund or additional wallet submission.
    service
        .settle_final(settlement("r", Some((20, 30)), Some(500), 500, None))
        .await
        .unwrap();
    assert!(
        service
            .settle_final(settlement(
                "r",
                Some((20, 30)),
                Some(500),
                0,
                Some(invoice(10, 1000))
            ))
            .await
            .is_err()
    );
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cached_discount_uses_frozen_price_and_same_seller_buyer_bill() -> Result<()> {
    use crate::ledger::metered::{MeteredOutcome, MeteredRequest};
    let dir = tempfile::tempdir()?;
    let (wallet, service) = open(dir.path());
    let mut price = rates();
    price.cache_discount_percent = 75;
    let mut terms = local("cached", 50_000);
    terms.pricing = price.clone();
    let mut accepted = admission("cached", 0);
    accepted.pricing = price.clone();
    service
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms,
            admission: accepted,
        })
        .await?;
    service.ledger.metered_admit(&MeteredRequest {
        id: "cached",
        peer: "peer",
        pricing: &price,
        require_deposit: false,
    })?;
    service.ledger.set_pricing("test", Some(&rates()))?;
    let usage = MeteredUsage {
        prompt_tokens: 1000,
        completion_tokens: 100,
        cached_tokens: 800,
    };
    let seller = service
        .ledger
        .metered_finalize("cached", MeteredOutcome::Usage(usage))?;
    assert_eq!(seller.bill_msat, Some(5000));
    service
        .settle_final(SettleFinalRequest {
            request_id: "cached".into(),
            usage: Some(usage),
            bill_msat: seller.bill_msat,
            deposit_applied_msat: 0,
            invoice: Some(invoice(51, 5000)),
        })
        .await?;
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn discounted_subsat_final_survives_both_restarts_and_pays_once() -> Result<()> {
    use crate::ledger::metered::MeteredRequest;
    use mesh_llm_payments_types::contract::{ServeRecoverResponse, ServeSettleRequest};
    let buyer_dir = tempfile::tempdir()?;
    let seller_dir = tempfile::tempdir()?;
    let (wallet, buyer) = open(buyer_dir.path());
    let seller_wallet = Arc::new(MockWallet::default());
    let seller = PaymentService::with_provider(seller_dir.path(), seller_wallet.clone())?;
    let mut price = rates();
    price.cache_discount_percent = 75;
    let mut terms = local("cached-floor", 5000);
    terms.pricing = price.clone();
    let mut accepted = admission("cached-floor", 0);
    accepted.pricing = price.clone();
    buyer
        .accept_admission(AcceptAdmissionRequest {
            billing_revision: mesh_llm_payments_types::contract::BillingRevision::V3,
            terms,
            admission: accepted,
        })
        .await?;
    seller.ledger.metered_admit(&MeteredRequest {
        id: "cached-floor",
        peer: "peer",
        pricing: &price,
        require_deposit: false,
    })?;
    let usage = MeteredUsage {
        prompt_tokens: 100,
        completion_tokens: 10,
        cached_tokens: 100,
    };
    // 250 discounted input + 100 output = 350 msat; positive final becomes 1000.
    let bill = seller
        .serve_settle(ServeSettleRequest {
            request_id: "cached-floor".into(),
            usage,
        })
        .await?;
    assert_eq!(
        (bill.bill_msat, bill.deposit_msat, bill.remaining_msat),
        (350, 0, 1000)
    );
    let final_invoice = bill.invoice.clone().unwrap();
    assert_eq!(final_invoice.amount_msat, Some(1000));
    drop(buyer);
    drop(seller);
    let seller = PaymentService::with_provider(seller_dir.path(), seller_wallet.clone())?;
    let buyer = PaymentService::with_provider(buyer_dir.path(), wallet.clone())?;
    let ServeRecoverResponse::Billed { bill: recovered } =
        seller.serve_recover("cached-floor").await?
    else {
        panic!("persisted final must be recoverable");
    };
    assert_eq!(recovered, bill);
    assert!(
        seller
            .serve_deposit_confirmed("cached-floor")
            .await
            .is_err(),
        "no redispatch"
    );
    let request = SettleFinalRequest {
        request_id: "cached-floor".into(),
        usage: Some(usage),
        bill_msat: Some(bill.bill_msat),
        deposit_applied_msat: bill.deposit_msat,
        invoice: recovered.invoice,
    };
    buyer.settle_final(request.clone()).await?;
    assert_eq!(buyer.ledger.requests()?[0].spent_msat, 1010);
    drop(buyer);
    drop(seller);
    let buyer = PaymentService::with_provider(buyer_dir.path(), wallet.clone())?;
    let seller = PaymentService::with_provider(seller_dir.path(), seller_wallet.clone())?;
    let ServeRecoverResponse::Billed { bill: recovered } =
        seller.serve_recover("cached-floor").await?
    else {
        panic!("persisted final must remain recoverable");
    };
    assert_eq!(recovered.invoice, Some(final_invoice));
    buyer.settle_final(request).await?;
    assert_eq!(buyer.ledger.requests()?[0].spent_msat, 1010);
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 1);
    assert_eq!(seller_wallet.invoice_expiries.lock().unwrap().len(), 1);
    Ok(())
}
