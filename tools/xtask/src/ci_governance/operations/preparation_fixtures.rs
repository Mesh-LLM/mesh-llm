//! Finite synthetic identities; these are never live GitHub observations.
use super::{
    digest,
    intent::ReviewedIntent,
    record::{Control, ExecutionMethod},
    state::ExpectedState,
};
use crate::ci_governance::{
    activation::Anchors,
    approval::{Account, AccountKind},
    policy::Policy,
};

pub(super) const NOW: u64 = 1_800_000_000;
pub(super) const ID: &str = "ba2e26e4-76a4-462f-894b-1a33d80d01aa";

pub(super) fn fixture() -> (Policy, ExpectedState, ReviewedIntent, Anchors) {
    let policy = crate::ci_governance::policy::fixture();
    let expected: ExpectedState = serde_json::from_str(include_str!(
        "../../../../../ci/governance/expected-state.json"
    ))
    .unwrap();
    let mut before = expected.state.clone();
    before.native_rule.required_approvals = 0;
    let owner = Account {
        login: "micspiral".into(),
        id: 101,
        kind: AccountKind::Human,
    };
    let intent = ReviewedIntent {
        schema_version: 1,
        operation_id: ID.into(),
        repository: policy.repository.clone(),
        policy_sha256: digest(&policy).unwrap(),
        desired_state_sha256: expected.state.canonical_digest().unwrap(),
        expected_before_sha256: before.canonical_digest().unwrap(),
        before,
        desired_after: expected.state.clone(),
        controls: vec![Control::CiRuleset],
        execution_method: ExecutionMethod::ManualProviderOperation,
        owner: owner.clone(),
        author: owner.clone(),
        sponsor: owner,
        rationale: "Synthetic local bounded governance preparation fixture".into(),
        affected_jobs: vec!["CI policy / independent approval".into()],
        tests: vec!["finite local fixtures".into()],
        rollout: "Human authorization and actual before/after readback remain required".into(),
        rollback: "Separately reviewed recovery of captured state".into(),
        created_at_unix: NOW - 100,
        expires_at_unix: NOW + 7200,
    };
    let anchors = Anchors {
        repository_id: 1,
        reviewer_team_id: 2,
        administration_environment_id: 3,
        observer_app_id: 4,
        observer_installation_id: 5,
        publisher_app_id: 4,
        publisher_installation_id: 5,
        native_ruleset_ids: vec![8],
        human_ids: std::collections::BTreeMap::from([
            ("micspiral".into(), 101),
            ("ndizazzo".into(), 102),
            ("i386".into(), 103),
        ]),
    };
    (policy, expected, intent, anchors)
}

pub(super) fn environment() -> serde_json::Value {
    serde_json::json!({"id":3,"name":"ci-governance-administration","can_admins_bypass":false,
        "protection_rules":[{"type":"required_reviewers","id":31,"prevent_self_review":true,"reviewers":[
            {"type":"User","reviewer":{"login":"micspiral","id":101,"type":"User"}},
            {"type":"User","reviewer":{"login":"ndizazzo","id":102,"type":"User"}},
            {"type":"User","reviewer":{"login":"i386","id":103,"type":"User"}}]},
            {"type":"branch_policy","id":32}],
        "deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}})
}

pub(super) fn branches() -> serde_json::Value {
    serde_json::json!({"total_count":1,"branch_policies":[{"id":33,"name":"main","type":"branch"}]})
}
