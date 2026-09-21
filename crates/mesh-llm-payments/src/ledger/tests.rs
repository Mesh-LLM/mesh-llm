#[cfg(test)]
use super::*;

#[cfg(test)]
fn terms(id: &str, cap: u64) -> RequestTerms {
    RequestTerms {
        exchange_id: None,
        id: id.into(),
        peer: "peer".into(),
        payee: None,
        model: "model".into(),
        pricing: Pricing {
            input_msat_per_million: 1000,
            output_msat_per_million: 1000,
            minimum_invoice_msat: 1,
        },
        input_tokens: 10,
        max_output_tokens: 10,
        max_total_msat: cap,
        expires_at_ms: u64::MAX,
    }
}

#[test]
fn backend_context_allowance_is_durable_and_bounds_delivery() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger
        .begin_serving(
            "context",
            "peer",
            &terms("context", 1).pricing,
            u64::from(u32::MAX),
        )
        .unwrap();
    ledger
        .resolve_serving_output_allowance("context", 6000)
        .unwrap();
    assert!(
        ledger
            .resolve_serving_output_allowance("context", 6001)
            .is_err()
    );
    assert!(
        ledger
            .resolve_serving_output_allowance("context", 0)
            .is_err()
    );
    ledger.record_delivered_tokens("context", 5000).unwrap();
    assert!(
        ledger
            .resolve_serving_output_allowance("context", 6000)
            .is_err()
    );
    drop(ledger);
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger.record_delivered_tokens("context", 6000).unwrap();
    assert!(ledger.record_delivered_tokens("context", 6001).is_err());
}

#[test]
fn restart_and_day_rollover_preserve_reservations() {
    let directory = tempfile::tempdir().unwrap();
    {
        let ledger = Ledger::open(directory.path()).unwrap();
        ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: Some(1000),
            })
            .unwrap();
        ledger.propose(&terms("one", 700)).unwrap();
        ledger.approve("one", 1000, 1).unwrap();
    }
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger.propose(&terms("two", 400)).unwrap();
    assert!(ledger.approve("two", 1000, DAY_MS + 1).is_err());
    ledger.finish("one").unwrap();
    ledger.approve("two", 1000, DAY_MS + 1).unwrap();
}

#[test]
fn independent_connections_cannot_double_reserve() {
    let directory = tempfile::tempdir().unwrap();
    let first = Ledger::open(directory.path()).unwrap();
    let second = Ledger::open(directory.path()).unwrap();
    first.propose(&terms("one", 700)).unwrap();
    second.propose(&terms("two", 700)).unwrap();
    let barrier = std::sync::Barrier::new(2);
    let results = std::thread::scope(|scope| {
        let one = scope.spawn(|| {
            barrier.wait();
            first.approve("one", 1000, 1).is_ok()
        });
        let two = scope.spawn(|| {
            barrier.wait();
            second.approve("two", 1000, 1).is_ok()
        });
        (one.join().unwrap(), two.join().unwrap())
    });
    assert_ne!(results.0, results.1);
}

#[test]
fn approvals_cannot_change_terms_or_revive_rejected_requests() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger.propose(&terms("one", 700)).unwrap();
    assert!(ledger.propose(&terms("one", 800)).is_err());
    ledger.reject("one").unwrap();
    assert!(ledger.approve("one", 1000, 1).is_err());
    assert!(
        ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: None
            })
            .is_err()
    );
}
