//! Open actual main-v2 SQL, not a current ledger relabelled with an old version.
use super::*;
use rusqlite::{Connection, params};

fn main_ledger(path: &std::path::Path, state: &str) -> Result<Invoice> {
    let connection = Connection::open(path.join("payments.sqlite3"))?;
    connection.execute_batch(include_str!("../ledger/migrations/fixtures/main_v2.sql"))?;
    connection.pragma_update(None, "user_version", 2)?;
    let policy = Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(20_000),
    };
    connection.execute(
        "INSERT INTO settings VALUES ('policy',?1)",
        [serde_json::to_string(&policy)?],
    )?;
    let terms = terms("old", 4000);
    connection.execute(
        "INSERT INTO requests VALUES ('old',?1,'approved',4000,0)",
        [serde_json::to_string(&terms)?],
    )?;
    let invoice = invoice(33, 1000);
    if state != "absent" {
        connection.execute(
            "INSERT INTO charges VALUES (?1,'old',0,?2,1000,4000,?3,0,NULL)",
            params![invoice.payment_hash, invoice.bolt11, state],
        )?;
    }
    Ok(invoice)
}

#[tokio::test]
async fn main_v2_submitted_success_failure_and_outage_preserve_exact_budget() -> Result<()> {
    for outcome in ["success", "failure", "unknown"] {
        let dir = tempfile::tempdir()?;
        let invoice = main_ledger(dir.path(), "pending")?;
        let wallet = Arc::new(MockWallet::default());
        if outcome == "unknown" {
            wallet.lookup_unavailable.store(true, Ordering::SeqCst);
        } else {
            wallet
                .terminal_failure
                .store(outcome == "failure", Ordering::SeqCst);
            wallet.pay(&invoice, 1000, 4000).await?;
        }
        let calls = wallet.calls.load(Ordering::SeqCst);
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        assert_eq!(service.ledger.available_budget(crate::now_ms())?, 20_000);
        assert_eq!(
            wallet.calls.load(Ordering::SeqCst),
            calls,
            "startup must not pay"
        );
        let result = service.reconcile_pending().await;
        assert_eq!(result.is_err(), outcome == "unknown");
        let (charge_state, remaining) = match outcome {
            "success" => ("succeeded", 18_990),
            "failure" => ("failed", 20_000),
            _ => ("pending", 20_000),
        };
        assert_eq!(
            service
                .ledger
                .charge_state(&invoice.payment_hash)?
                .as_deref(),
            Some(charge_state)
        );
        assert_eq!(service.ledger.available_budget(crate::now_ms())?, remaining);
        assert_eq!(
            wallet.calls.load(Ordering::SeqCst),
            calls,
            "observation adds no submission"
        );
        if outcome == "success" {
            // Legacy inference approvals do not auto-complete on success; their
            // actual charge, not the old request cap, consumes today's budget.
            assert_eq!(
                service.ledger.request_state("old")?.as_deref(),
                Some("approved")
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn main_v2_empty_and_unsubmitted_input_approvals_close_without_payment() -> Result<()> {
    for state in ["absent", "prepared"] {
        let dir = tempfile::tempdir()?;
        main_ledger(dir.path(), state)?;
        let wallet = Arc::new(MockWallet::default());
        let service = PaymentService::with_provider(dir.path(), wallet.clone())?;
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
        assert_eq!(service.ledger.available_budget(crate::now_ms())?, 20_000);
        service.reconcile_pending().await?;
        assert_eq!(
            service.ledger.request_state("old")?.as_deref(),
            Some("failed")
        );
        assert_eq!(wallet.calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}
