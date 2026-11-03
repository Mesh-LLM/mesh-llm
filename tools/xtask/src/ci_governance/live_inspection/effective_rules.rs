//! Bounded summary of active rulesets for main, not an effective-policy verdict.
use super::{GithubRead, Request, known, unknown};
use crate::ci_governance::collector::request::{CHECK_NAME, REPOSITORY};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const MAX_PAGES: u32 = 10;
const PAGE_SIZE: usize = 100;
const MAX_BODY: usize = 2 * 1024 * 1024;

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

fn bounded_text(value: Option<&Value>, max: usize) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty() && text.len() <= max && !text.chars().any(char::is_control))
}

fn read_rules(client: &mut impl GithubRead) -> Result<Vec<Value>, &'static str> {
    let mut rules = Vec::new();
    let mut seen = BTreeSet::new();
    for page in 1..=MAX_PAGES {
        let response = get(client, &Request::EffectiveMainRulesPage { page })?;
        let page = response
            .as_array()
            .ok_or("invalid_or_incomplete_response")?;
        if page.len() > PAGE_SIZE {
            return Err("invalid_or_incomplete_response");
        }
        if page.is_empty() {
            rules.sort_by_key(Value::to_string);
            return Ok(rules);
        }
        for rule in page {
            let kind =
                bounded_text(rule.get("type"), 100).ok_or("invalid_or_incomplete_response")?;
            if !kind
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
                || rule
                    .get("ruleset_id")
                    .and_then(Value::as_u64)
                    .filter(|id| *id > 0)
                    .is_none()
                || !matches!(
                    rule.get("ruleset_source_type").and_then(Value::as_str),
                    Some("Repository" | "Organization" | "Enterprise")
                )
                || bounded_text(rule.get("ruleset_source"), 255).is_none()
            {
                return Err("invalid_or_incomplete_response");
            }
            if !seen.insert(rule.to_string()) {
                return Err("duplicate_rule");
            }
            rules.push(rule.clone());
        }
    }
    Err("pagination_limit_reached")
}

fn same_identity(
    client: &mut impl GithubRead,
    repository_id: u64,
    head: &str,
) -> Result<(), &'static str> {
    let repository = get(client, &Request::Repository)?;
    let branch = get(client, &Request::CurrentMain)?;
    if repository.get("id").and_then(Value::as_u64) != Some(repository_id)
        || repository.get("full_name").and_then(Value::as_str) != Some(REPOSITORY)
        || repository.get("default_branch").and_then(Value::as_str) != Some("main")
        || branch.get("name").and_then(Value::as_str) != Some("main")
        || branch.pointer("/commit/sha").and_then(Value::as_str) != Some(head)
    {
        return Err("repository_or_main_changed_during_collection");
    }
    Ok(())
}

fn observed_flag(value: Option<bool>) -> Value {
    value.map_or_else(
        || unknown("missing_or_invalid_rule_parameters"),
        |v| known(json!(v)),
    )
}

fn pull_review_summary(rules: &[Value], team_id: Option<u64>) -> (Option<bool>, Option<bool>) {
    let Some(team_id) = team_id else {
        return (None, None);
    };
    let mut team_requirement = false;
    let mut stale_dismissal = false;
    for rule in rules
        .iter()
        .filter(|rule| rule.get("type").and_then(Value::as_str) == Some("pull_request"))
    {
        let Some(parameters) = rule.get("parameters") else {
            return (None, None);
        };
        let Some(stale) = parameters
            .get("dismiss_stale_reviews_on_push")
            .and_then(Value::as_bool)
        else {
            return (None, None);
        };
        let Some(reviewers) = parameters
            .get("required_reviewers")
            .and_then(Value::as_array)
            .filter(|items| items.len() <= 100)
        else {
            return (None, None);
        };
        stale_dismissal |= stale;
        for reviewer in reviewers {
            let Some(id) = reviewer
                .pointer("/reviewer/id")
                .and_then(Value::as_u64)
                .filter(|id| *id > 0)
            else {
                return (None, None);
            };
            if reviewer.pointer("/reviewer/type").and_then(Value::as_str) != Some("Team") {
                return (None, None);
            }
            let Some(minimum) = reviewer.get("minimum_approvals").and_then(Value::as_u64) else {
                return (None, None);
            };
            let Some(patterns) = reviewer
                .get("file_patterns")
                .and_then(Value::as_array)
                .filter(|items| !items.is_empty() && items.len() <= 1000)
            else {
                return (None, None);
            };
            if patterns
                .iter()
                .any(|pattern| bounded_text(Some(pattern), 1024).is_none())
            {
                return (None, None);
            }
            team_requirement |= id == team_id && minimum == 1;
        }
    }
    (Some(team_requirement), Some(stale_dismissal))
}

fn check_summary(rules: &[Value]) -> (Option<bool>, Option<bool>) {
    let mut found = false;
    let mut every_match_pinned = true;
    for rule in rules
        .iter()
        .filter(|rule| rule.get("type").and_then(Value::as_str) == Some("required_status_checks"))
    {
        let Some(checks) = rule
            .pointer("/parameters/required_status_checks")
            .and_then(Value::as_array)
            .filter(|items| items.len() <= 1000)
        else {
            return (None, None);
        };
        for check in checks {
            let Some(context) = bounded_text(check.get("context"), 255) else {
                return (None, None);
            };
            let pinned = match check.get("integration_id") {
                None | Some(Value::Null) => false,
                Some(id) if id.as_u64().is_some_and(|id| id > 0) => true,
                _ => return (None, None),
            };
            if context == CHECK_NAME {
                found = true;
                every_match_pinned &= pinned;
            }
        }
    }
    (Some(found), found.then_some(every_match_pinned))
}

pub(super) fn inspect(
    client: &mut impl GithubRead,
    repository: Option<&Value>,
    branch: Option<&Value>,
    team: Option<&Value>,
) -> Value {
    let Some(repository) = repository.filter(|item| {
        item.pointer("/fields/expected_name/value")
            .and_then(Value::as_bool)
            == Some(true)
            && item
                .pointer("/fields/default_branch_is_main/value")
                .and_then(Value::as_bool)
                == Some(true)
    }) else {
        return unknown("repository_identity_unavailable");
    };
    let Some(repository_id) = repository
        .pointer("/fields/numeric_id/value")
        .and_then(Value::as_u64)
        .filter(|id| *id > 0)
    else {
        return unknown("repository_identity_unavailable");
    };
    let Some(branch) = branch.filter(|item| {
        item.pointer("/fields/branch_is_main/value")
            .and_then(Value::as_bool)
            == Some(true)
    }) else {
        return unknown("main_identity_unavailable");
    };
    let Some(head) = branch
        .pointer("/fields/head_sha/value")
        .and_then(Value::as_str)
    else {
        return unknown("main_identity_unavailable");
    };
    let team_id = team
        .filter(|item| {
            item.pointer("/fields/expected_slug/value")
                .and_then(Value::as_bool)
                == Some(true)
                && item
                    .pointer("/fields/expected_organization/value")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
        .and_then(|item| item.pointer("/fields/numeric_id/value"))
        .and_then(Value::as_u64)
        .filter(|id| *id > 0);
    let result = (|| {
        let first = read_rules(client)?;
        let second = read_rules(client)?;
        if first != second {
            return Err("rules_changed_during_collection");
        }
        same_identity(client, repository_id, head)?;
        let (team_requirement, stale) = pull_review_summary(&first, team_id);
        let (check, pinned) = check_summary(&first);
        Ok(json!({"state":"observed", "fields":{
            "active_rule_count":known(json!(first.len())),
            "repeat_sample_equal":known(json!(true)),
            "team_with_one_minimum_observed":observed_flag(team_requirement),
            "stale_review_dismissal_observed":observed_flag(stale),
            "governance_check_context_observed":observed_flag(check),
            "all_matching_governance_checks_have_integration_id":pinned.map_or_else(|| unknown("check_absent_or_parameters_unavailable"), |value| known(json!(value))),
            "file_scope_and_exceptions":unknown("not_evaluated"),
            "check_producer_identity":unknown("activation_anchors_not_verified"),
            "bypass_actors":unknown("not_exposed_by_branch_rules_endpoint")
        }}))
    })();
    result.unwrap_or_else(unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ci_governance::collector::ApiResponse, command::DynResult};
    use std::collections::VecDeque;

    struct Mock {
        replies: VecDeque<(Request, Option<ApiResponse>)>,
        requests: Vec<Request>,
    }

    impl GithubRead for Mock {
        fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
            self.requests.push(request.clone());
            let (expected, reply) = self.replies.pop_front().expect("unexpected request");
            assert_eq!(&expected, request);
            reply.ok_or_else(|| "private transport error".into())
        }
    }

    fn reply(status: u16, value: Value) -> Option<ApiResponse> {
        Some(ApiResponse {
            status,
            body: serde_json::to_vec(&value).unwrap(),
        })
    }

    fn repository() -> Value {
        json!({"fields":{"numeric_id":{"value":1},"expected_name":{"value":true},"default_branch_is_main":{"value":true}}})
    }

    fn branch() -> Value {
        json!({"fields":{"branch_is_main":{"value":true},"head_sha":{"value":"1111111111111111111111111111111111111111"}}})
    }

    fn team() -> Value {
        json!({"fields":{"numeric_id":{"value":2},"expected_slug":{"value":true},"expected_organization":{"value":true}}})
    }

    fn rules() -> Value {
        json!([
            {"type":"pull_request","ruleset_id":7,"ruleset_source_type":"Repository","ruleset_source":"Mesh-LLM/mesh-llm","parameters":{
                "dismiss_stale_reviews_on_push":true,
                "required_reviewers":[{"reviewer":{"type":"Team","id":2},"minimum_approvals":1,"file_patterns":[".github/**/*"]}]
            }},
            {"type":"required_status_checks","ruleset_id":8,"ruleset_source_type":"Organization","ruleset_source":"private-provider-value","parameters":{
                "required_status_checks":[{"context":CHECK_NAME,"integration_id":42}]
            }}
        ])
    }

    fn stable(first: Value, second: Value) -> Mock {
        Mock {
            requests: Vec::new(),
            replies: VecDeque::from([
                (
                    Request::EffectiveMainRulesPage { page: 1 },
                    reply(200, first),
                ),
                (
                    Request::EffectiveMainRulesPage { page: 2 },
                    reply(200, json!([])),
                ),
                (
                    Request::EffectiveMainRulesPage { page: 1 },
                    reply(200, second),
                ),
                (
                    Request::EffectiveMainRulesPage { page: 2 },
                    reply(200, json!([])),
                ),
                (
                    Request::Repository,
                    reply(
                        200,
                        json!({"id":1,"full_name":REPOSITORY,"default_branch":"main"}),
                    ),
                ),
                (
                    Request::CurrentMain,
                    reply(
                        200,
                        json!({"name":"main","commit":{"sha":"1111111111111111111111111111111111111111"}}),
                    ),
                ),
            ]),
        }
    }

    fn run(client: &mut Mock) -> Value {
        inspect(client, Some(&repository()), Some(&branch()), Some(&team()))
    }

    #[test]
    fn stable_rules_report_only_bounded_non_authoritative_facts() {
        let mut client = stable(rules(), rules());
        let result = run(&mut client);
        assert_eq!(result["state"], "observed");
        let fields = &result["fields"];
        assert_eq!(fields["active_rule_count"]["value"], 2);
        assert_eq!(fields["team_with_one_minimum_observed"]["value"], true);
        assert_eq!(fields["stale_review_dismissal_observed"]["value"], true);
        assert_eq!(fields["governance_check_context_observed"]["value"], true);
        assert_eq!(
            fields["all_matching_governance_checks_have_integration_id"]["value"],
            true
        );
        assert_eq!(fields["file_scope_and_exceptions"]["state"], "unknown");
        assert_eq!(fields["check_producer_identity"]["state"], "unknown");
        assert_eq!(fields["bypass_actors"]["state"], "unknown");
        assert!(!result.to_string().contains("private-provider-value"));
    }

    #[test]
    fn missing_or_malformed_parameters_never_become_absence() {
        let mut missing = rules();
        missing[0]["parameters"]
            .as_object_mut()
            .unwrap()
            .remove("required_reviewers");
        missing[1]["parameters"]["required_status_checks"] = json!("not-an-array");
        let result = run(&mut stable(missing.clone(), missing));
        assert_eq!(
            result["fields"]["team_with_one_minimum_observed"]["state"],
            "unknown"
        );
        assert_eq!(
            result["fields"]["governance_check_context_observed"]["state"],
            "unknown"
        );
    }

    #[test]
    fn missing_team_or_producer_anchors_never_gain_authority() {
        let mut unpinned = rules();
        unpinned[1]["parameters"]["required_status_checks"][0]
            .as_object_mut()
            .unwrap()
            .remove("integration_id");
        let result = run(&mut stable(unpinned.clone(), unpinned));
        assert_eq!(
            result["fields"]["all_matching_governance_checks_have_integration_id"]["value"],
            false
        );
        assert_eq!(
            result["fields"]["check_producer_identity"]["state"],
            "unknown"
        );
        let mut client = stable(rules(), rules());
        let result = inspect(&mut client, Some(&repository()), Some(&branch()), None);
        assert_eq!(
            result["fields"]["team_with_one_minimum_observed"]["state"],
            "unknown"
        );
    }

    #[test]
    fn wrong_team_minimum_and_missing_fixed_context_are_reported_as_sample_facts() {
        let mut changed = rules();
        changed[0]["parameters"]["required_reviewers"][0]["minimum_approvals"] = json!(2);
        changed[1]["parameters"]["required_status_checks"][0]["context"] = json!("different check");
        let result = run(&mut stable(changed.clone(), changed));
        assert_eq!(
            result["fields"]["team_with_one_minimum_observed"]["value"],
            false
        );
        assert_eq!(
            result["fields"]["governance_check_context_observed"]["value"],
            false
        );
        assert_eq!(
            result["fields"]["all_matching_governance_checks_have_integration_id"]["state"],
            "unknown"
        );
        assert_eq!(
            result["fields"]["file_scope_and_exceptions"]["state"],
            "unknown"
        );
    }

    #[test]
    fn denial_is_unknown_and_never_retried() {
        let mut client = Mock {
            requests: Vec::new(),
            replies: VecDeque::from([(
                Request::EffectiveMainRulesPage { page: 1 },
                reply(403, json!({"message":"private-provider-error"})),
            )]),
        };
        let result = run(&mut client);
        assert_eq!(result["reason"], "access_denied");
        assert_eq!(client.requests.len(), 1);
        assert!(!result.to_string().contains("private-provider-error"));
    }

    #[test]
    fn changing_rules_or_repository_or_head_are_unknown() {
        let mut changed = rules();
        changed[0]["parameters"]["dismiss_stale_reviews_on_push"] = json!(false);
        assert_eq!(
            run(&mut stable(rules(), changed))["reason"],
            "rules_changed_during_collection"
        );
        let mut client = stable(rules(), rules());
        client.replies[5].1 = reply(
            200,
            json!({"name":"main","commit":{"sha":"2222222222222222222222222222222222222222"}}),
        );
        assert_eq!(
            run(&mut client)["reason"],
            "repository_or_main_changed_during_collection"
        );
    }

    #[test]
    fn pagination_requires_a_terminal_empty_page_and_rejects_duplicates() {
        let mut duplicate = rules();
        let first = duplicate[0].clone();
        duplicate.as_array_mut().unwrap().push(first);
        let result = run(&mut stable(duplicate.clone(), duplicate));
        assert_eq!(result["reason"], "duplicate_rule");
        let mut client = Mock {
            requests: Vec::new(),
            replies: VecDeque::new(),
        };
        for page in 1..=MAX_PAGES {
            client.replies.push_back((Request::EffectiveMainRulesPage { page }, reply(200, json!([{
                "type":"pull_request", "ruleset_id":page, "ruleset_source_type":"Repository", "ruleset_source":REPOSITORY
            }]))));
        }
        assert_eq!(run(&mut client)["reason"], "pagination_limit_reached");
        assert_eq!(client.requests.len(), MAX_PAGES as usize);
    }

    #[test]
    fn paginated_samples_include_the_later_page() {
        let entries = rules();
        let first = json!([entries[0].clone()]);
        let second = json!([entries[1].clone()]);
        let mut client = Mock {
            requests: Vec::new(),
            replies: VecDeque::new(),
        };
        for _ in 0..2 {
            client.replies.extend([
                (
                    Request::EffectiveMainRulesPage { page: 1 },
                    reply(200, first.clone()),
                ),
                (
                    Request::EffectiveMainRulesPage { page: 2 },
                    reply(200, second.clone()),
                ),
                (
                    Request::EffectiveMainRulesPage { page: 3 },
                    reply(200, json!([])),
                ),
            ]);
        }
        client.replies.extend([
            (Request::Repository, reply(200, json!({"id":1,"full_name":REPOSITORY,"default_branch":"main"}))),
            (Request::CurrentMain, reply(200, json!({"name":"main","commit":{"sha":"1111111111111111111111111111111111111111"}}))),
        ]);
        let result = run(&mut client);
        assert_eq!(result["fields"]["active_rule_count"]["value"], 2);
        assert_eq!(
            result["fields"]["governance_check_context_observed"]["value"],
            true
        );
    }
}
