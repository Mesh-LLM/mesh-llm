use super::*;
use mesh_llm_payments::ledger::receivables::Receivable;

#[tokio::test]
async fn sequential_admission_waits_for_terminal_output_payment() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let service = PaymentService::with_provider(
        directory.path(),
        Arc::new(TestWallet {
            owner: 1,
            network: network.clone(),
        }),
    )?;
    let price = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        minimum_invoice_msat: 1,
    };
    service.ledger.begin_serving("first", "peer", &price, 8)?;
    service.ledger.record_delivered_tokens("first", 3)?;
    service.ledger.finish_serving("first")?;
    // Reproduce the old immediate-next-request rejection before the output
    // invoice even exists. Other peers remain independent.
    assert!(
        service
            .ledger
            .begin_serving("second", "peer", &price, 8)
            .is_err()
    );
    assert!(!service.ledger.has_outstanding_payment("other")?);
    let wait =
        super::super::server::await_prior_settlement(&service, "peer", Duration::from_secs(2));
    tokio::pin!(wait);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut wait)
            .await
            .is_err()
    );
    let receipt = service.output_receivable("first").await?.unwrap();
    {
        let mut entries = network.entries.lock().unwrap();
        let payment = &mut entries.get_mut(&receipt.invoice.payment_hash).unwrap().1;
        payment.status_msg = Some("claiming".into());
    }
    // Arrival may open this request's output gate, but cannot clear previous debt.
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut wait)
            .await
            .is_err()
    );
    {
        let mut entries = network.entries.lock().unwrap();
        let payment = &mut entries.get_mut(&receipt.invoice.payment_hash).unwrap().1;
        payment.status = PaymentStatus::Succeeded;
    }
    wait.await?;
    service.ledger.begin_serving("second", "peer", &price, 8)?;
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn admission_deadline_preserves_unpaid_debt() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let service = PaymentService::with_provider(
        directory.path(),
        Arc::new(TestWallet { owner: 1, network }),
    )?;
    let price = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        minimum_invoice_msat: 1,
    };
    service.ledger.begin_serving("first", "peer", &price, 8)?;
    let invoice = service.wallet().await?.create_invoice(Some(1)).await?;
    service.ledger.record_receivable(&Receivable {
        request_id: "first".into(),
        peer: "peer".into(),
        segment: 0,
        invoice,
        tokens: 1,
        paid: false,
    })?;
    assert!(
        super::super::server::await_prior_settlement(&service, "peer", Duration::from_millis(20))
            .await
            .is_err()
    );
    assert!(service.ledger.has_outstanding_payment("peer")?);
    assert!(
        service
            .ledger
            .begin_serving("second", "peer", &price, 8)
            .is_err()
    );
    Ok(())
}
