//! Bounded read-only observations before activation. This is not an authority,
//! effective-rules evaluator, authenticated drift snapshot, or approval.
use super::{
    collector::{GithubClient, GithubRead, Request, SecretToken, request::REPOSITORY},
    policy::valid_sha,
};
use crate::{command::DynResult, command_interrupt::Interrupt};
use serde_json::{Map, Value, json};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn run(root: &Path, args: &[String]) -> DynResult<Value> {
    if !args.is_empty() {
        return Err("inspect-live-controls accepts no arguments".into());
    }
    // Never use gh's implicit credential fallback or mint a credential here.
    // The operator supplies an existing, read-only token explicitly.
    let credential = std::env::var("MESH_CI_GOVERNANCE_AUDIT_TOKEN")
        .map_err(|_| "MESH_CI_GOVERNANCE_AUDIT_TOKEN is required for read-only inspection")?;
    let credential = SecretToken::checked(credential)?;
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut client = GithubClient::installation(root, &cancellation, credential)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let report = inspect(&mut client, now);
    interrupt.finish()?;
    Ok(report)
}

fn unknown(reason: &str) -> Value {
    json!({"state":"unknown", "reason":reason})
}

fn known(value: Value) -> Value {
    json!({"state":"observed", "value":value})
}

fn boolean(value: Option<&Value>) -> Value {
    value.and_then(Value::as_bool).map_or_else(
        || unknown("missing_or_invalid_field"),
        |value| known(json!(value)),
    )
}

fn positive(value: Option<&Value>) -> Value {
    value
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .map_or_else(
            || unknown("missing_or_invalid_field"),
            |value| known(json!(value)),
        )
}

fn count(value: Option<&Value>) -> Value {
    value
        .and_then(Value::as_u64)
        .filter(|value| *value <= 1000)
        .map_or_else(
            || unknown("missing_or_invalid_field"),
            |value| known(json!(value)),
        )
}

fn exact(value: Option<&Value>, expected: &str) -> Value {
    value.and_then(Value::as_str).map_or_else(
        || unknown("missing_or_invalid_field"),
        |value| known(json!(value == expected)),
    )
}

fn sha(value: Option<&Value>) -> Value {
    value
        .and_then(Value::as_str)
        .filter(|value| valid_sha(value))
        .map_or_else(
            || unknown("missing_or_invalid_field"),
            |value| known(json!(value)),
        )
}

fn repository(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"numeric_id":positive(value.get("id")),
        "expected_name":exact(value.get("full_name"), REPOSITORY),
        "default_branch_is_main":exact(value.get("default_branch"), "main")}))
}

fn branch(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"branch_is_main":exact(value.get("name"), "main"),
        "protected_flag":boolean(value.get("protected")),
        "head_sha":sha(value.pointer("/commit/sha"))}))
}

fn team(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"numeric_id":positive(value.get("id")),
    "expected_slug":exact(value.get("slug"), "ci-policy-reviewers"),
    "visible":match value.get("privacy").and_then(Value::as_str) {
        Some("closed") => known(json!(true)),
        Some("secret") => known(json!(false)),
        _ => unknown("missing_or_invalid_field")
    }}))
}

fn team_permission(value: &Value) -> Option<Value> {
    value.as_object()?;
    let permission = match value.get("role_name").and_then(Value::as_str) {
        Some(role @ ("read" | "triage" | "write" | "maintain" | "admin")) => known(json!(role)),
        _ => unknown("missing_or_invalid_field"),
    };
    Some(json!({"repository_id":positive(value.get("id")),
        "expected_repository":exact(value.get("full_name"), REPOSITORY),
        "role":permission}))
}

fn environment(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"numeric_id":positive(value.get("id")),
        "expected_name":exact(value.get("name"), "ci-governance-administration"),
        "protection_rule_count":value.get("protection_rules").and_then(Value::as_array)
            .filter(|rules| rules.len() <= 100).map_or_else(
                || unknown("missing_or_invalid_field"), |rules| known(json!(rules.len())))}))
}

fn legacy_protection(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(
        json!({"required_approvals":count(value.pointer("/required_pull_request_reviews/required_approving_review_count")),
        "dismiss_stale_reviews":boolean(value.pointer("/required_pull_request_reviews/dismiss_stale_reviews")),
        "require_code_owner_review":boolean(value.pointer("/required_pull_request_reviews/require_code_owner_reviews")),
        "enforce_admins":boolean(value.pointer("/enforce_admins/enabled"))}),
    )
}

fn rulesets_first_page(value: &Value) -> Option<Value> {
    let rules = value.as_array()?;
    (rules.len() <= 100).then(|| {
        json!({
            "first_page_count":known(json!(rules.len())),
            "complete_effective_census":unknown("pagination_and_inherited_rules_not_assessed")
        })
    })
}

type Extract = fn(&Value) -> Option<Value>;

/// A transport failure stops the transaction; a denied individual request is
/// recorded once as unknown and does not turn 404 into evidence of absence.
fn inspect(client: &mut impl GithubRead, now: u64) -> Value {
    let sources: [(&str, Request, Extract); 7] = [
        ("repository", Request::Repository, repository),
        ("main_branch", Request::CurrentMain, branch),
        ("reviewer_team", Request::Team, team),
        (
            "team_repository_access",
            Request::TeamPermission,
            team_permission,
        ),
        (
            "administration_environment",
            Request::AdministrationEnvironment,
            environment,
        ),
        (
            "legacy_branch_protection",
            Request::MainBranchProtection,
            legacy_protection,
        ),
        (
            "repository_rulesets_first_page",
            Request::RepositoryRulesetsFirstPage,
            rulesets_first_page,
        ),
    ];
    let mut observations = Map::new();
    let mut stopped = false;
    for (name, request, extract) in sources {
        let observation = if stopped {
            unknown("not_queried_after_transport_failure")
        } else {
            match client.send(&request) {
                Err(_) => {
                    stopped = true;
                    unknown("transport_failure")
                }
                Ok(response) if response.status == 200 => {
                    match serde_json::from_slice::<Value>(&response.body)
                        .ok()
                        .and_then(|value| extract(&value))
                    {
                        Some(fields) => json!({"state":"observed", "fields":fields}),
                        None => unknown("invalid_or_incomplete_response"),
                    }
                }
                Ok(response) => {
                    let reason = match response.status {
                        401 | 403 => "access_denied",
                        404 => "not_found_or_inaccessible",
                        _ => "http_error",
                    };
                    json!({"state":"unknown", "reason":reason, "http_status":response.status})
                }
            }
        };
        observations.insert(name.to_owned(), observation);
    }
    json!({
        "schema_version":1, "status":"PARTIAL_UNVERIFIED", "observed_at_unix":now,
        "observations":observations,
        "unassessed":{
            "reviewer_membership":unknown("not_collected"),
            "effective_branch_rules_and_path_scope":unknown("not_collected"),
            "required_check_source_and_bypasses":unknown("not_collected"),
            "environment_reviewers_and_branch_policy":unknown("not_collected"),
            "credential_permissions_and_boundaries":unknown("not_collected")
        },
        "drift_status":"UNKNOWN", "evidence_authenticated":false,
        "execution_authorized":false, "activation_authorized":false,
        "limitations":"Independent read-only responses are not an atomic or complete effective-policy census; token identity, permissions and numeric anchors are not verified."
    })
}

#[cfg(test)]
#[path = "live_inspection_tests.rs"]
mod tests;
