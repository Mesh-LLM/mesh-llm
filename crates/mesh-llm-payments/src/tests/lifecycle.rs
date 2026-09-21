use super::*;
use crate::lifecycle::PaymentPhase;

#[tokio::test]
async fn lifecycle_survives_restart_and_never_exposes_recovery_capability() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut terms = terms("secret-recovery-capability", 300);
    terms.exchange_id = Some("public-evidence-id".into());
    let invoice = invoice(211, 100);
    let charge = Charge {
        request_id: terms.id.clone(),
        segment: 0,
        invoice: invoice.clone(),
        amount_msat: 100,
        max_total_msat: 120,
    };
    let ids;
    {
        let service = PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default()))?;
        let mut events = service.ledger.subscribe_events();
        service.ledger.propose(&terms)?;
        service.approve(&terms.id).await?;
        assert!(matches!(
            events.try_recv()?.phase,
            PaymentPhase::TermsAccepted
        ));
        service.pay_charge(&charge).await?;
        service.ledger.finish(&terms.id)?;
        let observed = service.ledger.payment_observations(&terms.id)?;
        assert!(
            observed
                .iter()
                .any(|e| matches!(e.phase, PaymentPhase::FinalAmount))
        );
        let json = serde_json::to_string(&observed)?;
        assert!(!json.contains(&terms.id));
        assert!(!json.contains(&invoice.bolt11));
        assert!(!json.contains("preimage"));
        assert!(
            observed
                .iter()
                .all(|e| e.exchange_id == "public-evidence-id")
        );
        ids = observed
            .iter()
            .map(|e| e.event_id.clone())
            .collect::<Vec<_>>();
    }
    let ledger = Ledger::open(dir.path())?;
    assert_eq!(
        ids,
        ledger
            .payment_observations(&terms.id)?
            .iter()
            .map(|e| e.event_id.clone())
            .collect::<Vec<_>>()
    );
    let mut legacy = serde_json::to_value(&terms)?;
    legacy.as_object_mut().unwrap().remove("exchange_id");
    assert!(
        serde_json::from_value::<RequestTerms>(legacy)?
            .exchange_id
            .is_none()
    );
    Ok(())
}

#[tokio::test]
async fn pending_and_failed_charges_do_not_claim_settlement() -> Result<()> {
    for status in [PaymentStatus::Pending, PaymentStatus::Failed] {
        let dir = tempfile::tempdir()?;
        let service = PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default()))?;
        let mut terms = terms("request", 300);
        terms.exchange_id = Some("evidence".into());
        service.ledger.propose(&terms)?;
        service.approve(&terms.id).await?;
        let invoice = invoice(212, 100);
        service.ledger.prepare_charge(&Charge {
            request_id: terms.id.clone(),
            segment: 0,
            invoice: invoice.clone(),
            amount_msat: 100,
            max_total_msat: 120,
        })?;
        service.ledger.reconcile(
            &Transaction {
                id: "wallet-reference".into(),
                payment_hash: Some(invoice.payment_hash),
                inbound: false,
                amount_msat: 100,
                fee_msat: 0,
                status,
                status_msg: Some("claiming".into()),
                created_at_ms: 0,
                settled_at_ms: None,
            },
            crate::now_ms(),
        )?;
        assert!(
            !service
                .ledger
                .payment_observations(&terms.id)?
                .iter()
                .any(|e| matches!(
                    e.phase,
                    PaymentPhase::InvoiceSettled | PaymentPhase::FinalAmount
                ))
        );
    }
    Ok(())
}

#[test]
fn provider_terms_digest_matches_payer_and_receipt_is_not_payment_authority() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let ledger = Ledger::open(dir.path())?;
    let mut terms = terms("bearer", 300);
    terms.exchange_id = Some("correlation".into());
    ledger.begin_serving(&terms.id, &terms.peer, &terms.pricing, 8)?;
    ledger.record_serving_terms(&terms)?;
    let invoice = invoice(213, 1);
    ledger.record_receivable(&crate::ledger::receivables::Receivable {
        request_id: terms.id.clone(),
        peer: terms.peer.clone(),
        segment: 0,
        invoice: invoice.clone(),
        tokens: 1,
        paid: false,
    })?;
    ledger.finish_serving(&terms.id)?;
    let before = ledger.payment_observations(&terms.id)?;
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].evidence, "provider_claim");
    terms.peer = "other-authenticated-peer".into();
    let payer = crate::lifecycle::PaymentEvent::new(
        &terms,
        crate::lifecycle::PaymentRole::Payer,
        PaymentPhase::TermsAccepted,
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(before[0].terms_digest, payer.terms_digest);
    ledger.mark_received(&invoice.payment_hash)?;
    let after = ledger.payment_observations(&terms.id)?;
    assert!(after.iter().any(
        |e| matches!(e.phase, PaymentPhase::InvoiceSettled) && e.evidence == "wallet_reported"
    ));
    assert!(
        after
            .iter()
            .any(|e| matches!(e.phase, PaymentPhase::FinalAmount))
    );
    Ok(())
}
