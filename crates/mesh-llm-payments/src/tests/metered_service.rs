//! Real service startup and wallet-observation boundaries for metered requests.
use super::*;
use crate::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus};
use mesh_llm_payments_types::{metered_wire::AdmissionTerms, metering::MeteredUsage};

fn rates() -> Pricing {
    Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    }
}
fn admit(service: &PaymentService, id: &str) {
    service
        .ledger
        .metered_admit(&MeteredRequest {
            id,
            peer: "peer",
            pricing: &rates(),
            estimated_input_tokens: 1000,
            max_output_tokens: 500,
            peer_cap_msat: 10_000,
            require_deposit: true,
        })
        .unwrap();
}
fn report() -> MeteredOutcome {
    MeteredOutcome::Usage(MeteredUsage {
        prompt_tokens: 900,
        completion_tokens: 200,
    })
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
}

#[tokio::test]
async fn late_deposit_before_and_after_fixed_invoice_is_conserved() {
    for fixed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let wallet = Arc::new(MockWallet::default());
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        admit(&service, "r");
        let deposit = service
            .metered_invoice("r", "peer", 0, 300)
            .await
            .unwrap()
            .unwrap();
        service.ledger.metered_claim_dispatch("r", "peer").unwrap();
        service.ledger.metered_finalize("r", report()).unwrap();
        let final_before = if fixed {
            service.metered_final_invoice("r", "peer").await.unwrap()
        } else {
            None
        };
        received(&wallet, &deposit);
        service.metered_refresh("r").await.unwrap();
        service.metered_refresh("r").await.unwrap();
        let final_invoice = service
            .metered_final_invoice("r", "peer")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            final_invoice.amount_msat,
            Some(if fixed { 1100 } else { 800 })
        );
        if let Some(before) = final_before {
            assert_eq!(before, final_invoice);
        }
        assert_eq!(
            service.ledger.peer_credit("peer").unwrap(),
            if fixed { 300 } else { 0 }
        );
        assert!(!service.metered_trusted("peer").await.unwrap());
        received(&wallet, &final_invoice);
        service.metered_refresh("r").await.unwrap();
        service.metered_note_settled("r", "peer").await.unwrap();
        assert!(service.metered_trusted("peer").await.unwrap());
        assert_eq!(service.ledger.metered_exposure("peer").unwrap(), 0);
    }
}

#[tokio::test]
async fn restart_after_dispatch_claim_retains_unknown_usage_and_never_redispatches() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let deposit;
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        admit(&service, "dispatched");
        admit(&service, "never");
        deposit = service
            .metered_invoice("never", "peer", 0, 300)
            .await
            .unwrap()
            .unwrap();
        assert!(
            service
                .ledger
                .metered_claim_dispatch("dispatched", "peer")
                .unwrap()
        );
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    assert_eq!(
        service
            .ledger
            .metered_settlement("dispatched")
            .unwrap()
            .status,
        MeteredStatus::UsageMissing
    );
    assert_eq!(
        service.ledger.metered_settlement("never").unwrap().status,
        MeteredStatus::BackendError
    );
    assert!(
        service
            .ledger
            .metered_claim_dispatch("dispatched", "peer")
            .is_ok_and(|claimed| !claimed)
    );
    assert_eq!(service.ledger.metered_exposure("peer").unwrap(), 1500);
    received(&wallet, &deposit);
    service.metered_refresh("never").await.unwrap();
    assert_eq!(service.ledger.peer_credit("peer").unwrap(), 300);
    assert!(!service.metered_trusted("peer").await.unwrap());
}

#[tokio::test]
async fn payer_recovery_only_observes_submitted_v2_deposits() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    service
        .ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(10_000),
        })
        .unwrap();
    for (id, submitted) in [("prepared", false), ("submitted", true)] {
        let payer = terms(id, 1000);
        service
            .save_metered_authorization(&AdmissionTerms {
                id: id.into(),
                model: payer.model.clone(),
                pricing: payer.pricing.clone(),
                prompt_estimate: 100,
                max_output_tokens: 100,
                max_total_msat: 1000,
                deposit_msat: 100,
                credit_msat: 0,
            })
            .unwrap();
        service.ledger.propose(&payer).unwrap();
        service.approve(id).await.unwrap();
        let c = charge(id, 0, if submitted { 2 } else { 1 }, 100, 200);
        service.ledger.prepare_charge(&c).unwrap();
        if submitted {
            service
                .ledger
                .begin_submission(&c.invoice.payment_hash)
                .unwrap();
        }
    }
    service.reconcile_pending().await.unwrap();
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    assert_eq!(service.ledger.pending_charges().unwrap().len(), 2);
}

#[tokio::test]
async fn complete_retires_prepared_and_payee_pin_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        service
            .ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(10_000),
            })
            .unwrap();
        let payer = terms("r", 6000);
        service
            .save_metered_authorization(&AdmissionTerms {
                id: "r".into(),
                model: payer.model.clone(),
                pricing: payer.pricing.clone(),
                prompt_estimate: 100,
                max_output_tokens: 100,
                max_total_msat: 6000,
                deposit_msat: 100,
                credit_msat: 0,
            })
            .unwrap();
        service.await_authorization(&payer).await.unwrap();
        service
            .ledger
            .prepare_charge(&charge("r", 0, 1, 100, 200))
            .unwrap();
        service.metered_pin_payee("r", "wallet-one").await.unwrap();
    }
    let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
    service.metered_pin_payee("r", "wallet-one").await.unwrap();
    assert!(service.metered_pin_payee("r", "wallet-two").await.is_err());
    service.metered_complete("r").await.unwrap();
    service.reconcile_pending().await.unwrap();
    assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        service.ledger.request_state("r").unwrap().as_deref(),
        Some("completed")
    );
}

#[tokio::test]
async fn expired_nonpayment_survives_restart_and_requires_deposit_in_later_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let wallet = Arc::new(MockWallet::default());
    let expired = invoice_with_expiry(91, 1100, 0);
    assert!(expired.expires_at_ms <= crate::now_ms());
    {
        let service = PaymentService::with_provider(dir.path(), wallet.clone()).unwrap();
        admit(&service, "unpaid");
        service
            .ledger
            .metered_claim_dispatch("unpaid", "peer")
            .unwrap();
        service.ledger.metered_finalize("unpaid", report()).unwrap();
        service
            .ledger
            .record_receivable(&crate::ledger::receivables::Receivable {
                request_id: "unpaid".into(),
                peer: "peer".into(),
                segment: 1,
                invoice: expired.clone(),
                tokens: 0,
                paid: false,
            })
            .unwrap();
        assert_eq!(service.ledger.metered_exposure("peer").unwrap(), 1100);
    }
    let service = PaymentService::with_provider(dir.path(), wallet).unwrap();
    let debts = service.ledger.receivables(Some("unpaid")).unwrap();
    assert_eq!(debts.len(), 1);
    assert_eq!(debts[0].invoice, expired);
    assert!(!debts[0].paid);
    assert_eq!(service.ledger.metered_exposure("peer").unwrap(), 1100);

    // Re-earn cache standing after restart. A denial with an empty cache alone
    // would not prove that the persisted unpaid invoice affects future trust.
    for peer in ["peer", "other"] {
        let seed = format!("settled-{peer}");
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: &seed,
                peer,
                pricing: &rates(),
                estimated_input_tokens: 1,
                max_output_tokens: 1,
                peer_cap_msat: 10_000,
                require_deposit: true,
            })
            .unwrap();
        service
            .ledger
            .metered_finalize(
                &seed,
                MeteredOutcome::Usage(MeteredUsage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                }),
            )
            .unwrap();
        service.metered_note_settled(&seed, peer).await.unwrap();
        let trusted = service.metered_trusted(peer).await.unwrap();
        assert_eq!(trusted, peer == "other");
        let later = service
            .ledger
            .metered_admit(&MeteredRequest {
                id: &format!("later-{peer}"),
                peer,
                pricing: &rates(),
                estimated_input_tokens: 1000,
                max_output_tokens: 500,
                peer_cap_msat: 10_000,
                require_deposit: !trusted,
            })
            .unwrap();
        assert_eq!(later.deposit_msat, if peer == "peer" { 300 } else { 0 });
    }
    // Persisted exposure also independently refuses further work at the cap.
    assert!(
        service
            .ledger
            .metered_admit(&MeteredRequest {
                id: "over-cap",
                peer: "peer",
                pricing: &rates(),
                estimated_input_tokens: 1000,
                max_output_tokens: 500,
                peer_cap_msat: 2600,
                require_deposit: true,
            })
            .is_err()
    );
    assert!(service.ledger.metered_owner("over-cap").unwrap().is_none());
    // Clearing only the invoice debt now restores the already re-earned
    // standing, proving the earlier denial was not merely an empty cache.
    service.ledger.mark_received(&expired.payment_hash).unwrap();
    assert!(service.metered_trusted("peer").await.unwrap());
    assert_eq!(service.ledger.metered_exposure("peer").unwrap(), 1500);
}
