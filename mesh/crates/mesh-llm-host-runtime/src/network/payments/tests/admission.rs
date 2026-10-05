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
    let invoice = service
        .wallet()
        .await?
        .create_invoice(Some(1), 3600)
        .await?;
    service.ledger.record_receivable(&Receivable {
        request_id: "first".into(),
        peer: "peer".into(),
        segment: 0,
        invoice: invoice.clone(),
        tokens: 1,
        paid: false,
    })?;
    service.ledger.mark_received(&invoice.payment_hash)?;
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
    let wait = service.await_prior_settlement("peer", Duration::from_secs(2));
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
        payment.claiming = true;
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
    let invoice = service
        .wallet()
        .await?
        .create_invoice(Some(1), 3600)
        .await?;
    service.ledger.record_receivable(&Receivable {
        request_id: "first".into(),
        peer: "peer".into(),
        segment: 0,
        invoice,
        tokens: 1,
        paid: false,
    })?;
    assert!(
        service
            .await_prior_settlement("peer", Duration::from_millis(20))
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

#[tokio::test]
async fn recovery_reports_pending_until_input_settles() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    let service = Arc::new(PaymentService::with_provider(
        directory.path(),
        Arc::new(TestWallet {
            owner: 1,
            network: network.clone(),
        }),
    )?);
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
    let payments = super::super::client::Payments::attach_for_tests(&node, service.clone()).await?;
    let id = uuid::Uuid::new_v4().to_string();
    let price = Pricing {
        input_msat_per_million: 1,
        output_msat_per_million: 1,
        minimum_invoice_msat: 1,
    };
    service.ledger.begin_serving(&id, "peer", &price, 8)?;
    let invoice = service
        .wallet()
        .await?
        .create_invoice(Some(1), 3600)
        .await?;
    service.ledger.record_receivable(&Receivable {
        request_id: id.clone(),
        peer: "peer".into(),
        segment: 0,
        invoice: invoice.clone(),
        tokens: 1,
        paid: false,
    })?;
    service.ledger.record_delivered_tokens(&id, 3)?;
    service.ledger.finish_serving(&id)?;
    for status in [PaymentStatus::Pending, PaymentStatus::Failed] {
        {
            let mut entries = network.entries.lock().unwrap();
            let payment = &mut entries.get_mut(&invoice.payment_hash).unwrap().1;
            payment.status = status;
            payment.claiming = true;
        }
        let (mut writer, mut reader) = tokio::io::duplex(8192);
        super::super::server::recover(&payments, &id, &mut writer).await?;
        assert!(matches!(wire::read(&mut reader).await?, Frame::Pending));
        assert_eq!(service.ledger.receivables(Some(&id))?.len(), 1);
    }
    network
        .entries
        .lock()
        .unwrap()
        .get_mut(&invoice.payment_hash)
        .unwrap()
        .1
        .status = PaymentStatus::Succeeded;
    for _ in 0..2 {
        let (mut writer, mut reader) = tokio::io::duplex(8192);
        super::super::server::recover(&payments, &id, &mut writer).await?;
        assert!(matches!(
            wire::read(&mut reader).await?,
            Frame::OutputInvoice { tokens: 3, .. }
        ));
        assert!(matches!(wire::read(&mut reader).await?, Frame::Complete));
    }
    assert_eq!(service.ledger.receivables(Some(&id))?.len(), 2);
    Ok(())
}
