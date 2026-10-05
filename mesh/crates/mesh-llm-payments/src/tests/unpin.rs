use super::*;
use crate::control::ControlCommand;
use crate::ledger::receivables::Receivable;
use crate::provisioning::WalletPin;

const PEER: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn pin() -> WalletPin {
    WalletPin {
        plugin: "wallet-lexe".into(),
        wallet_id: "w1".into(),
        provider: "lexe".into(),
        network: "mainnet".into(),
    }
}

fn unpaid(request_id: &str, invoice: Invoice) -> Receivable {
    Receivable {
        request_id: request_id.into(),
        peer: PEER.into(),
        segment: 0,
        invoice,
        tokens: 10,
        paid: false,
    }
}

fn pricing() -> Pricing {
    Pricing {
        input_msat_per_million: 1000,
        output_msat_per_million: 1000,
        minimum_invoice_msat: 1,
    }
}

#[tokio::test]
async fn unpin_removes_the_pin_when_nothing_is_outstanding() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    let result = service.control(ControlCommand::Unpin).await?;
    assert_eq!(result, serde_json::json!({"unpinned": null}));

    pin().store(dir.path())?;
    let result = service.control(ControlCommand::Unpin).await?;
    assert_eq!(result["unpinned"]["plugin"], "wallet-lexe");
    assert!(WalletPin::load(dir.path())?.is_none());
    Ok(())
}

#[tokio::test]
async fn unpin_is_refused_while_an_issued_invoice_is_unpaid() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    pin().store(dir.path())?;
    service.ledger.begin_serving("req", PEER, &pricing(), 10)?;
    service
        .ledger
        .record_receivable(&unpaid("req", invoice(1, 100)))?;

    let error = service
        .control(ControlCommand::Unpin)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("1 unpaid invoices"), "{error}");
    assert!(error.contains("wallet unblock"), "{error}");
    assert_eq!(WalletPin::load(dir.path())?, Some(pin()));
    Ok(())
}

#[tokio::test]
async fn an_unpaid_invoice_blocks_until_it_is_forgiven() -> Result<()> {
    // Expiry does not release it: it may have been paid before expiring
    // without the ledger recording it, and only the old wallet can say.
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    pin().store(dir.path())?;
    service.ledger.begin_serving("req", PEER, &pricing(), 10)?;
    service
        .ledger
        .record_receivable(&unpaid("req", invoice(1, 100)))?;
    assert_eq!(
        service
            .ledger
            .wallet_switch_blockers()?
            .unpaid_invoices
            .len(),
        1
    );
    assert!(service.control(ControlCommand::Unpin).await.is_err());

    service.ledger.unblock_peer(PEER)?;
    service.control(ControlCommand::Unpin).await?;
    assert!(WalletPin::load(dir.path())?.is_none());
    Ok(())
}

#[tokio::test]
async fn unpin_is_refused_while_an_outgoing_payment_is_unresolved() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let service = PaymentService::open(dir.path())?;
    pin().store(dir.path())?;
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })?;
    service.ledger.propose(&terms("one", 1000))?;
    service.ledger.approve("one", 100_000, crate::now_ms())?;
    let pending = charge("one", 0, 1, 100, 200);
    assert!(service.ledger.prepare_charge(&pending)?);

    let error = service
        .control(ControlCommand::Unpin)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("1 unresolved outgoing"), "{error}");
    assert_eq!(
        service.ledger.wallet_switch_blockers()?.unresolved_payments,
        vec![pending.invoice.payment_hash]
    );
    assert_eq!(WalletPin::load(dir.path())?, Some(pin()));
    Ok(())
}

#[tokio::test]
async fn unpin_is_refused_by_a_service_that_can_reach_a_wallet() -> Result<()> {
    // A running node's service may be opening its wallet against the pin,
    // opened or not, so only a ledger-only service may unpin.
    let dir = tempfile::tempdir()?;
    pin().store(dir.path())?;
    let with_wallet = PaymentService::with_provider(dir.path(), Arc::new(MockWallet::default()))?;
    let error = with_wallet
        .control(ControlCommand::Unpin)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("stop mesh-llm"), "{error}");
    drop(with_wallet);

    let unopened =
        PaymentService::with_factory(dir.path(), Arc::new(crate::provisioning::NoWalletFactory))?;
    assert!(unopened.control(ControlCommand::Unpin).await.is_err());
    assert_eq!(WalletPin::load(dir.path())?, Some(pin()));
    Ok(())
}
