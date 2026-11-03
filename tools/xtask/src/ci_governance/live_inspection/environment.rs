//! Partial environment configuration signals, never deployment authorization.
use super::{boolean, known, membership::valid_login, unknown};
use crate::ci_governance::policy::REVIEWERS;
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn reviewer_set(rule: &Value) -> Value {
    let Some(reviewers) = rule.get("reviewers").and_then(Value::as_array) else {
        return unknown("missing_or_invalid_field");
    };
    if reviewers.len() > 6 {
        return unknown("missing_or_invalid_field");
    }
    let mut ids = BTreeSet::new();
    let mut logins = BTreeSet::new();
    let mut only_humans = true;
    for entry in reviewers {
        let Some(reviewer) = entry.get("reviewer") else {
            return unknown("missing_or_invalid_field");
        };
        let Some(id) = reviewer
            .get("id")
            .and_then(Value::as_u64)
            .filter(|id| *id > 0)
        else {
            return unknown("missing_or_invalid_field");
        };
        if !ids.insert(id) {
            return unknown("duplicate_reviewer_identity");
        }
        match entry.get("type").and_then(Value::as_str) {
            Some("User") if reviewer.get("type").and_then(Value::as_str) == Some("User") => {
                let Some(login) = reviewer
                    .get("login")
                    .and_then(Value::as_str)
                    .filter(|s| valid_login(s))
                else {
                    return unknown("missing_or_invalid_field");
                };
                if !logins.insert(login.to_ascii_lowercase()) {
                    return unknown("duplicate_reviewer_identity");
                }
            }
            Some("Team") if reviewer.as_object().is_some() => only_humans = false,
            _ => return unknown("missing_or_invalid_field"),
        }
    }
    let expected: BTreeSet<_> = REVIEWERS.into_iter().map(str::to_owned).collect();
    known(json!(only_humans && logins == expected))
}

fn rules(value: &Value) -> Option<Vec<&Value>> {
    let rules = value.get("protection_rules")?.as_array()?;
    if rules.len() > 100 {
        return None;
    }
    let mut ids = BTreeSet::new();
    for rule in rules {
        let kind = rule.get("type")?.as_str()?;
        let id = rule.get("id")?.as_u64()?;
        if kind.is_empty() || kind.len() > 64 || id == 0 || !ids.insert(id) {
            return None;
        }
    }
    Some(rules.iter().collect())
}

fn custom_branch_mode(value: &Value) -> Value {
    let selection = value.get("deployment_branch_policy");
    match (
        selection
            .and_then(|v| v.get("protected_branches"))
            .and_then(Value::as_bool),
        selection
            .and_then(|v| v.get("custom_branch_policies"))
            .and_then(Value::as_bool),
    ) {
        (Some(protected), Some(custom)) => known(json!(!protected && custom)),
        _ => unknown("missing_or_invalid_field"),
    }
}

pub(super) fn inspect(value: &Value) -> Value {
    let Some(rules) = rules(value) else {
        return json!({
            "required_reviewers_rule":unknown("invalid_or_incomplete_rules"),
            "prevent_self_review":unknown("invalid_or_incomplete_rules"),
            "expected_human_login_set":unknown("invalid_or_incomplete_rules"),
            "custom_branch_mode":custom_branch_mode(value),
            "administrator_bypass":unknown("not_verified_by_documented_get")
        });
    };
    let reviewers = rules
        .iter()
        .copied()
        .filter(|rule| rule.get("type").and_then(Value::as_str) == Some("required_reviewers"))
        .collect::<Vec<_>>();
    let (self_review, reviewer_logins) = match reviewers.as_slice() {
        [rule] => (boolean(rule.get("prevent_self_review")), reviewer_set(rule)),
        _ => (
            unknown("unique_reviewer_rule_unavailable"),
            unknown("unique_reviewer_rule_unavailable"),
        ),
    };
    json!({
        "required_reviewers_rule":known(json!(reviewers.len() == 1)),
        "prevent_self_review":self_review,
        "expected_human_login_set":reviewer_logins,
        "custom_branch_mode":custom_branch_mode(value),
        "administrator_bypass":unknown("not_verified_by_documented_get")
    })
}
