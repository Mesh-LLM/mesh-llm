use super::*;
use crate::ci_governance::operations::preparation_fixtures::{NOW, fixture};

#[test]
fn static_intent_needs_no_self_referencing_commit_or_unknown_future_run() {
    let (policy, expected, intent, _) = fixture();
    intent.validate(&policy, &expected, NOW).unwrap();
    let value = serde_json::to_value(&intent).unwrap();
    assert!(value.get("policy_revision").is_none());
    assert!(value.get("execution").is_none());
    assert_eq!(intent.canonical_digest().unwrap().len(), 64);
}

#[test]
fn stale_policy_state_expiry_and_unbounded_operation_are_refused() {
    let (policy, expected, intent, _) = fixture();
    let mut candidate = intent.clone();
    candidate.policy_sha256 = "b".repeat(64);
    assert!(candidate.validate(&policy, &expected, NOW).is_err());
    candidate = intent.clone();
    candidate.before.native_rule.required_approvals = 2;
    assert!(candidate.validate(&policy, &expected, NOW).is_err());
    candidate = intent.clone();
    candidate.expires_at_unix = NOW;
    assert!(candidate.validate(&policy, &expected, NOW).is_err());
    let mut value = serde_json::to_value(intent).unwrap();
    value["command"] = serde_json::json!("never an operation payload");
    assert!(serde_json::from_value::<ReviewedIntent>(value).is_err());
}

#[test]
fn independent_review_still_required_and_intent_has_no_embedded_approval_claim() {
    let (policy, expected, mut intent, _) = fixture();
    intent.sponsor.login = "ndizazzo".into();
    assert!(intent.validate(&policy, &expected, NOW).is_err());
    let (policy, expected, mut intent, _) = fixture();
    intent.controls.push(Control::ReviewerTeam);
    assert!(intent.validate(&policy, &expected, NOW).is_err());
}
