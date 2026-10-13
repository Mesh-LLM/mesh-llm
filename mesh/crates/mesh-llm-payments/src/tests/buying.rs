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
