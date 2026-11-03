//! Bounded read-only observations before activation. This is not an authority,
//! effective-rules evaluator, authenticated drift snapshot, or approval.
mod branches;
mod effective_rules;
mod environment;
mod membership;

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
    let report = inspect_all(&mut client, now);
    interrupt.finish()?;
    Ok(report)
}

fn inspect_all(client: &mut impl GithubRead, now: u64) -> Value {
    let mut report = inspect(client, now);
    let observations = &report["observations"];
    let stopped = observations.as_object().is_some_and(|items| {
        items.values().any(|value| {
            matches!(
                value.get("reason").and_then(Value::as_str),
                Some("transport_failure" | "not_queried_after_transport_failure")
            )
        })
    });
    let protection_unavailable = observations
        .pointer("/legacy_branch_protection/state")
        .and_then(Value::as_str)
        != Some("observed")
        || [
            "required_approvals",
            "dismiss_stale_reviews",
            "require_code_owner_review",
            "enforce_admins",
        ]
        .iter()
        .any(|field| {
            observations
                .pointer(&format!("/legacy_branch_protection/fields/{field}/state"))
                .and_then(Value::as_str)
                != Some("observed")
        });
    let rules = if stopped {
        unknown("not_queried_after_transport_failure")
    } else if protection_unavailable {
        unknown("not_queried_after_protection_unknown")
    } else {
        effective_rules::inspect(
            client,
            observations.get("repository"),
            observations.get("main_branch"),
            observations.get("reviewer_team"),
        )
    };
    report["observations"]["main_active_rulesets"] = rules;
    report
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

fn parent_state(parent: Option<&Value>) -> Option<bool> {
    match parent {
        Some(Value::Null) => Some(true),
        Some(Value::Object(parent))
            if parent
                .get("id")
                .and_then(Value::as_u64)
                .is_some_and(|id| id > 0)
                && parent
                    .get("slug")
                    .and_then(Value::as_str)
                    .is_some_and(|slug| {
                        !slug.is_empty()
                            && slug.len() <= 100
                            && !slug.chars().any(|ch| ch.is_control() || ch.is_whitespace())
                    }) =>
        {
            Some(false)
        }
        _ => None,
    }
}

fn team(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"numeric_id":positive(value.get("id")),
    "expected_slug":exact(value.get("slug"), "ci-policy-reviewers"),
    "expected_organization":exact(value.pointer("/organization/login"), "Mesh-LLM"),
    "no_parent":parent_state(value.get("parent")).map_or_else(
        || unknown("missing_or_invalid_field"), |no_parent| known(json!(no_parent))),
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
    let permissions = value.get("permissions");
    let matches_write = match (
        permissions
            .and_then(|p| p.get("pull"))
            .and_then(Value::as_bool),
        permissions
            .and_then(|p| p.get("push"))
            .and_then(Value::as_bool),
        permissions
            .and_then(|p| p.get("admin"))
            .and_then(Value::as_bool),
        permissions
            .and_then(|p| p.get("maintain"))
            .and_then(Value::as_bool),
    ) {
        (Some(pull), Some(push), Some(admin), Some(maintain)) => {
            known(json!(pull && push && !admin && !maintain))
        }
        _ => unknown("missing_or_invalid_field"),
    };
    Some(json!({"repository_id":positive(value.get("id")),
        "expected_repository":exact(value.get("full_name"), REPOSITORY),
        "role":permission,"write_permission_flags":matches_write}))
}

fn environment(value: &Value) -> Option<Value> {
    value.as_object()?;
    Some(json!({"numeric_id":positive(value.get("id")),
        "expected_name":exact(value.get("name"), "ci-governance-administration"),
        "protection_rule_count":value.get("protection_rules").and_then(Value::as_array)
            .filter(|rules| rules.len() <= 100).map_or_else(
                || unknown("missing_or_invalid_field"), |rules| known(json!(rules.len()))),
        "review_policy":environment::inspect(value)}))
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

fn environment_indicators(observations: &Value, findings: &mut Vec<&'static str>) {
    if observations
        .pointer("/administration_environment/fields/expected_name/value")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return;
    }
    for (field, control) in [
        (
            "required_reviewers_rule",
            "administration_environment.required_reviewers_rule",
        ),
        (
            "prevent_self_review",
            "administration_environment.prevent_self_review",
        ),
        (
            "expected_human_login_set",
            "administration_environment.reviewers",
        ),
        (
            "custom_branch_mode",
            "administration_environment.custom_branch_mode",
        ),
    ] {
        let pointer = format!("/administration_environment/fields/review_policy/{field}/value");
        if observations.pointer(&pointer).and_then(Value::as_bool) == Some(false) {
            findings.push(control);
        }
    }
}

fn partial_drift_indicators(observations: &Map<String, Value>) -> Vec<&'static str> {
    let observations = Value::Object(observations.clone());
    let mut findings = Vec::new();
    for (pointer, control) in [
        (
            "/repository/fields/expected_name/value",
            "repository.identity",
        ),
        (
            "/repository/fields/default_branch_is_main/value",
            "repository.default_branch",
        ),
        (
            "/main_branch/fields/branch_is_main/value",
            "main_branch.identity",
        ),
        (
            "/reviewer_team/fields/expected_slug/value",
            "reviewer_team.slug",
        ),
        (
            "/reviewer_team/fields/expected_organization/value",
            "reviewer_team.organization",
        ),
        (
            "/reviewer_team/fields/visible/value",
            "reviewer_team.visibility",
        ),
        (
            "/reviewer_team/fields/no_parent/value",
            "reviewer_team.parent",
        ),
        (
            "/team_repository_access/fields/expected_repository/value",
            "reviewer_team.repository",
        ),
        (
            "/administration_environment/fields/expected_name/value",
            "administration_environment.name",
        ),
        (
            "/reviewer_membership_names/fields/expected_human_login_set/value",
            "reviewer_team.membership",
        ),
        (
            "/administration_branch_policies/fields/expected_main_branch_only/value",
            "administration_environment.allowed_refs",
        ),
    ] {
        if observations.pointer(pointer).and_then(Value::as_bool) == Some(false) {
            findings.push(control);
        }
    }
    if observations
        .pointer("/team_repository_access/fields/expected_repository/value")
        .and_then(Value::as_bool)
        == Some(true)
        && observations
            .pointer("/team_repository_access/fields/role/value")
            .and_then(Value::as_str)
            .is_some_and(|role| role != "write")
    {
        findings.push("reviewer_team.repository_role");
    }
    if observations
        .pointer("/team_repository_access/fields/expected_repository/value")
        .and_then(Value::as_bool)
        == Some(true)
        && observations
            .pointer("/team_repository_access/fields/write_permission_flags/value")
            .and_then(Value::as_bool)
            == Some(false)
    {
        findings.push("reviewer_team.repository_permission_flags");
    }
    environment_indicators(&observations, &mut findings);
    if let (Some(repository_id), Some(access_id)) = (
        observations
            .pointer("/repository/fields/numeric_id/value")
            .and_then(Value::as_u64),
        observations
            .pointer("/team_repository_access/fields/repository_id/value")
            .and_then(Value::as_u64),
    ) && repository_id != access_id
    {
        findings.push("reviewer_team.repository_numeric_identity");
    }
    findings
}

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
    let membership = if stopped {
        unknown("not_queried_after_transport_failure")
    } else {
        membership::inspect(client, observations.get("reviewer_team"))
    };
    if membership.get("reason").and_then(Value::as_str) == Some("transport_failure") {
        stopped = true;
    }
    observations.insert("reviewer_membership_names".into(), membership);
    let branches = if stopped {
        unknown("not_queried_after_transport_failure")
    } else {
        branches::inspect(client, observations.get("administration_environment"))
    };
    observations.insert("administration_branch_policies".into(), branches);
    let indicators = partial_drift_indicators(&observations);
    json!({
        "schema_version":1, "status":"PARTIAL_UNVERIFIED", "observed_at_unix":now,
        "observations":observations,
        "unassessed":{
            "reviewer_membership":unknown("numeric_anchors_and_effective_membership_not_verified"),
            "effective_branch_rules_and_path_scope":unknown("active_ruleset_summary_does_not_establish_complete_policy_or_path_scope"),
            "required_check_source_and_bypasses":unknown("not_collected"),
            "environment_reviewers_and_branch_policy":unknown("numeric_anchors_bypass_and_complete_branch_policy_not_verified"),
            "credential_permissions_and_boundaries":unknown("not_collected")
        },
        "drift_status":"UNKNOWN", "partial_drift_indicators":indicators,
        "partial_indicators_authenticated":false, "evidence_authenticated":false,
        "execution_authorized":false, "activation_authorized":false,
        "limitations":"Independent read-only responses are not an atomic or complete effective-policy census; token identity, permissions and numeric anchors are not verified."
    })
}

#[cfg(test)]
#[path = "live_inspection_tests.rs"]
mod tests;
