//! Read a bounded environment branch-policy list without authorizing a deployment.
use super::{GithubRead, Request, environment, known, unknown};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;

const MAX_PAGES: u32 = 10;
const MAX_BODY: usize = 2 * 1024 * 1024;

#[derive(Deserialize)]
struct Page {
    total_count: usize,
    branch_policies: Vec<BranchPolicy>,
}

#[derive(Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct BranchPolicy {
    id: u64,
    name: String,
    #[serde(rename = "type")]
    kind: Option<String>,
}

fn get(client: &mut impl GithubRead, request: &Request) -> Result<Value, &'static str> {
    let response = client.send(request).map_err(|_| "transport_failure")?;
    if response.status != 200 {
        return Err(match response.status {
            401 | 403 => "access_denied",
            404 => "not_found_or_inaccessible",
            _ => "http_error",
        });
    }
    if response.body.len() > MAX_BODY {
        return Err("invalid_or_incomplete_response");
    }
    serde_json::from_slice(&response.body).map_err(|_| "invalid_or_incomplete_response")
}

fn read_policies(client: &mut impl GithubRead) -> Result<Vec<BranchPolicy>, &'static str> {
    let mut policies = Vec::new();
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    let mut total = None;
    for page in 1..=MAX_PAGES {
        let raw = get(client, &Request::AdministrationEnvironmentBranches { page })?;
        let read: Page =
            serde_json::from_value(raw).map_err(|_| "invalid_or_incomplete_response")?;
        if read.total_count > 1000
            || read.branch_policies.len() > 100
            || total.is_some_and(|count| count != read.total_count)
            || (read.branch_policies.is_empty() && policies.len() < read.total_count)
        {
            return Err("invalid_or_incomplete_response");
        }
        total = Some(read.total_count);
        for policy in read.branch_policies {
            if policy.id == 0
                || policy.name.is_empty()
                || policy.name.len() > 255
                || policy
                    .name
                    .chars()
                    .any(|ch| ch.is_control() || ch.is_whitespace())
                || policy
                    .kind
                    .as_deref()
                    .is_some_and(|kind| !matches!(kind, "branch" | "tag"))
            {
                return Err("invalid_or_incomplete_response");
            }
            if !ids.insert(policy.id) || !names.insert((policy.name.clone(), policy.kind.clone())) {
                return Err("duplicate_branch_policy");
            }
            policies.push(policy);
        }
        if policies.len() > read.total_count {
            return Err("invalid_or_incomplete_response");
        }
        if policies.len() == read.total_count {
            policies.sort();
            return Ok(policies);
        }
    }
    Err("pagination_limit_reached")
}

fn same_environment(client: &mut impl GithubRead, id: u64) -> Result<(), &'static str> {
    let value = get(client, &Request::AdministrationEnvironment)?;
    if value.get("id").and_then(Value::as_u64) != Some(id)
        || value.get("name").and_then(Value::as_str) != Some("ci-governance-administration")
        || environment::inspect(&value)
            .pointer("/custom_branch_mode/value")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("environment_identity_or_mode_changed");
    }
    Ok(())
}

pub(super) fn inspect(client: &mut impl GithubRead, observation: Option<&Value>) -> Value {
    let Some(observation) = observation.filter(|item| {
        item.get("state").and_then(Value::as_str) == Some("observed")
            && item
                .pointer("/fields/expected_name/value")
                .and_then(Value::as_bool)
                == Some(true)
    }) else {
        return unknown("environment_metadata_unavailable");
    };
    let Some(id) = observation
        .pointer("/fields/numeric_id/value")
        .and_then(Value::as_u64)
    else {
        return unknown("environment_identity_unavailable");
    };
    if observation
        .pointer("/fields/review_policy/custom_branch_mode/value")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return unknown("custom_branch_mode_unavailable");
    }
    let result = (|| {
        let first = read_policies(client)?;
        let second = read_policies(client)?;
        if first != second {
            return Err("branch_policies_changed_during_collection");
        }
        same_environment(client, id)?;
        if first.iter().any(|policy| policy.kind.is_none()) {
            return Err("branch_policy_type_unavailable");
        }
        let matches = first.len() == 1
            && first[0].name == "main"
            && first[0].kind.as_deref() == Some("branch");
        Ok(json!({"state":"observed", "fields":{
            "count":known(json!(first.len())),
            "expected_main_branch_only":known(json!(matches)),
            "repeat_sample_equal":known(json!(true)),
            "numeric_identity_pinned":unknown("activation_anchors_not_verified")
        }}))
    })();
    result.unwrap_or_else(unknown)
}
