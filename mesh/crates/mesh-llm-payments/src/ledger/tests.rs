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
            cache_discount_percent: 0,
        },
        max_output_tokens: 10,
        max_total_msat: cap,
        expires_at_ms: u64::MAX,
    }
}

#[test]
fn approvals_do_not_reserve_the_daily_budget() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger
        .set_policy(&Policy {
            mode: ApprovalMode::Automatic,
            daily_budget_msat: Some(1000),
        })
        .unwrap();
    ledger.propose(&terms("one", 700)).unwrap();
    ledger.propose(&terms("two", 700)).unwrap();
    ledger.approve("one", 1).unwrap();
    ledger.approve("two", 1).unwrap();
    assert_eq!(ledger.available_budget(1).unwrap(), 1000);
}

#[test]
fn approvals_cannot_change_terms_or_revive_rejected_requests() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(directory.path()).unwrap();
    ledger.propose(&terms("one", 700)).unwrap();
    assert!(ledger.propose(&terms("one", 800)).is_err());
    ledger.reject("one").unwrap();
    assert!(ledger.approve("one", 1).is_err());
    assert!(
        ledger
            .set_policy(&Policy {
                mode: ApprovalMode::Automatic,
                daily_budget_msat: None
            })
            .is_err()
    );
}

#[test]
fn evidence_correlation_survives_reopen_without_schema_change() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let mut request = terms("private-recovery", 100);
    request.exchange_id = Some("public-exchange".into());
    let ledger = Ledger::open(directory.path())?;
    ledger.propose(&request)?;
    drop(ledger);
    let reopened = Ledger::open(directory.path())?;
    assert_eq!(
        reopened.requests()?[0].terms.exchange_id,
        request.exchange_id
    );
    let mut legacy = serde_json::to_value(&request)?;
    legacy.as_object_mut().unwrap().remove("exchange_id");
    assert!(
        serde_json::from_value::<RequestTerms>(legacy)?
            .exchange_id
            .is_none()
    );
    Ok(())
}

#[test]
fn single_policy_controls_eligibility() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let ledger = Ledger::open(directory.path())?;
    assert_eq!(ledger.policy()?.mode, ApprovalMode::FreeOnly);
    assert!(matches!(
        ledger.payment_intent()?,
        crate::intent::PaymentIntent::FreeOnly
    ));
    assert_eq!(ledger.available_budget(1)?, 0);
    ledger.propose(&terms("disabled", 500))?;
    assert!(ledger.approve("disabled", 1).is_err());
    ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(1000),
    })?;
    ledger.approve("disabled", 1)?;
    let status = ledger.policy_status(1)?;
    assert_eq!(status["in_flight_msat"], 0);
    assert_eq!(status["remaining_daily_budget_msat"], 1000);
    ledger.set_policy(&Policy::default())?;
    assert_eq!(ledger.policy_status(1)?["remaining_daily_budget_msat"], 0);
    assert_eq!(
        ledger.request_state("disabled")?.as_deref(),
        Some("approved")
    );
    Ok(())
}
