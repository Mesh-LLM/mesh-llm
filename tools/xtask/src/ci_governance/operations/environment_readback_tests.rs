use super::*;
use crate::ci_governance::collector::ApiResponse;
use crate::ci_governance::operations::preparation_fixtures::{self, fixture};
use serde_json::{Value, json};
use std::collections::VecDeque;

struct Replies(VecDeque<(Request, u16, Value)>);
impl GithubRead for Replies {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
        let (expected, status, body) =
            self.0.pop_front().ok_or("unexpected extra fixed request")?;
        assert_eq!(request, &expected);
        Ok(ApiResponse {
            status,
            body: serde_json::to_vec(&body)?,
        })
    }
}
fn replies(environment: Value, branches: Value) -> Replies {
    Replies(VecDeque::from([
        (Request::AdministrationEnvironment, 200, environment),
        (
            Request::AdministrationEnvironmentBranches { page: 1 },
            200,
            branches,
        ),
    ]))
}

#[test]
fn exact_environment_configuration_is_distinct_from_an_approval() {
    let (policy, _, _, anchors) = fixture();
    let mut source = replies(
        preparation_fixtures::environment(),
        preparation_fixtures::branches(),
    );
    let result = collect_environment(&mut source, &anchors, &policy).unwrap();
    assert!(!result.configuration_verified);
    assert_eq!(result.administrator_bypass, None);
    assert!(!result.approval_verified);
    assert_eq!(result.environment_id, 3);
    assert!(source.0.is_empty());
}

#[test]
fn bypass_self_review_missing_config_and_replaced_identity_fail_closed() {
    let (policy, _, _, anchors) = fixture();
    for (pointer, replacement) in [
        ("/can_admins_bypass", json!(true)),
        ("/protection_rules/0/prevent_self_review", json!(false)),
        ("/id", json!(900)),
        ("/deployment_branch_policy/protected_branches", json!(true)),
        (
            "/protection_rules/0/reviewers/0/reviewer/type",
            json!("Bot"),
        ),
    ] {
        let mut environment = preparation_fixtures::environment();
        *environment.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            collect_environment(
                &mut replies(environment, preparation_fixtures::branches()),
                &anchors,
                &policy
            )
            .is_err(),
            "{pointer}"
        );
    }
    let mut environment = preparation_fixtures::environment();
    environment
        .as_object_mut()
        .unwrap()
        .remove("can_admins_bypass");
    let partial = collect_environment(
        &mut replies(environment, preparation_fixtures::branches()),
        &anchors,
        &policy,
    )
    .unwrap();
    assert!(!partial.configuration_verified);
}

#[test]
fn documented_optional_bypass_and_branch_type_are_unverified_not_invented() {
    let (policy, _, _, anchors) = fixture();
    let mut environment = preparation_fixtures::environment();
    environment
        .as_object_mut()
        .unwrap()
        .remove("can_admins_bypass");
    let mut branches = preparation_fixtures::branches();
    branches["branch_policies"][0]
        .as_object_mut()
        .unwrap()
        .remove("type");
    let result =
        collect_environment(&mut replies(environment, branches), &anchors, &policy).unwrap();
    assert_eq!(result.administrator_bypass, None);
    assert_eq!(result.allowed_branch_kind, None);
    assert_eq!(
        result.unverified_fields,
        ["administrator_bypass", "allowed_branch_kind"]
    );
    assert!(!result.configuration_verified);
    assert!(!result.approval_verified);
}

#[test]
fn wildcard_tag_extra_branch_unknown_rule_and_incomplete_reads_are_refused() {
    let (policy, _, _, anchors) = fixture();
    for replacement in [
        json!({"id":33,"name":"*","type":"branch"}),
        json!({"id":33,"name":"main","type":"tag"}),
    ] {
        let mut branches = preparation_fixtures::branches();
        branches["branch_policies"][0] = replacement;
        assert!(
            collect_environment(
                &mut replies(preparation_fixtures::environment(), branches),
                &anchors,
                &policy
            )
            .is_err()
        );
    }
    let mut environment = preparation_fixtures::environment();
    environment["protection_rules"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"custom","id":34}));
    assert!(
        collect_environment(
            &mut replies(environment, preparation_fixtures::branches()),
            &anchors,
            &policy
        )
        .is_err()
    );
    let mut source = Replies(VecDeque::from([(
        Request::AdministrationEnvironment,
        403,
        json!({}),
    )]));
    assert!(collect_environment(&mut source, &anchors, &policy).is_err());
    let mut branches = preparation_fixtures::branches();
    branches["total_count"] = json!(2);
    assert!(
        collect_environment(
            &mut replies(preparation_fixtures::environment(), branches),
            &anchors,
            &policy
        )
        .is_err()
    );
}

#[test]
fn even_approved_history_never_fabricates_attempt_or_packet_binding() {
    let (_, _, _, anchors) = fixture();
    let history = json!([{"state":"approved","user":{"login":"ndizazzo","id":102,"type":"User"},
        "environments":[{"id":3,"name":"ci-governance-administration"}]}]);
    let mut source = Replies(VecDeque::from([(
        Request::WorkflowRunApprovals { run_id: 77 },
        200,
        history,
    )]));
    let result = collect_approval_history(&mut source, &anchors, 77).unwrap();
    assert_eq!(result.reviews_observed, 1);
    assert!(!result.authorization_verified);
    assert!(!result.exact_attempt_binding_verified);
    assert!(!result.prepared_packet_digest_binding_verified);
}
