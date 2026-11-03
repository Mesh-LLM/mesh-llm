use super::*;
use crate::ci_governance::collector::ApiResponse;
use std::collections::VecDeque;

struct Mock {
    replies: VecDeque<(Request, Option<ApiResponse>)>,
    requests: Vec<Request>,
}

impl GithubRead for Mock {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
        self.requests.push(request.clone());
        let index = self
            .replies
            .iter()
            .position(|(expected, _)| expected == request)
            .expect("unexpected request");
        let (_, reply) = self.replies.remove(index).unwrap();
        reply.ok_or_else(|| "transport failure, do not echo this error".into())
    }
}

fn response(status: u16, body: Value) -> Option<ApiResponse> {
    Some(ApiResponse {
        status,
        body: serde_json::to_vec(&body).unwrap(),
    })
}

fn expected_members() -> Value {
    json!([
        {"id":11,"login":"micspiral","type":"User"},
        {"id":12,"login":"ndizazzo","type":"User"},
        {"id":13,"login":"i386","type":"User"}
    ])
}

fn team_metadata() -> Value {
    json!({"id":2,"slug":"ci-policy-reviewers","privacy":"closed","organization":{"login":"Mesh-LLM"},"parent":null})
}

fn environment_metadata() -> Value {
    json!({"id":3,"name":"ci-governance-administration", "protection_rules":[
        {"id":31,"type":"required_reviewers","prevent_self_review":true,"reviewers":[
            {"type":"User","reviewer":{"id":11,"login":"micspiral","type":"User"}},
            {"type":"User","reviewer":{"id":12,"login":"ndizazzo","type":"User"}},
            {"type":"User","reviewer":{"id":13,"login":"i386","type":"User"}}
        ]},
        {"id":32,"type":"branch_policy"}
    ],"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}})
}

fn branch_page() -> Value {
    json!({"total_count":1,"branch_policies":[{"id":33,"name":"main","type":"branch"}]})
}

fn add_branch_replies(client: &mut Mock) {
    client.replies.extend([
        (
            Request::AdministrationEnvironmentBranches { page: 1 },
            response(200, branch_page()),
        ),
        (
            Request::AdministrationEnvironmentBranches { page: 1 },
            response(200, branch_page()),
        ),
        (
            Request::AdministrationEnvironment,
            response(200, environment_metadata()),
        ),
    ]);
}

fn mock() -> Mock {
    let mut client = Mock {
        requests: Vec::new(),
        replies: VecDeque::from([
            (
                Request::Repository,
                response(
                    200,
                    json!({"id":1,"full_name":REPOSITORY,"default_branch":"main"}),
                ),
            ),
            (
                Request::CurrentMain,
                response(
                    200,
                    json!({"name":"main","protected":true,"commit":{"sha":"1111111111111111111111111111111111111111"}}),
                ),
            ),
            (Request::Team, response(200, team_metadata())),
            (
                Request::TeamPermission,
                response(
                    200,
                    json!({"id":1,"full_name":REPOSITORY,"role_name":"write"}),
                ),
            ),
            (
                Request::AdministrationEnvironment,
                response(200, environment_metadata()),
            ),
            (
                Request::MainBranchProtection,
                response(
                    200,
                    json!({"required_pull_request_reviews":{"required_approving_review_count":0,"dismiss_stale_reviews":false,"require_code_owner_reviews":false},"enforce_admins":{"enabled":false}}),
                ),
            ),
            (
                Request::RepositoryRulesetsFirstPage,
                response(200, json!([{"id":7,"secret":"do-not-print"}])),
            ),
            (
                Request::TeamMembersPage { page: 1 },
                response(200, expected_members()),
            ),
            (
                Request::TeamMembersPage { page: 2 },
                response(200, json!([])),
            ),
            (
                Request::TeamMembersPage { page: 1 },
                response(200, expected_members()),
            ),
            (
                Request::TeamMembersPage { page: 2 },
                response(200, json!([])),
            ),
            (Request::Team, response(200, team_metadata())),
        ]),
    };
    add_branch_replies(&mut client);
    client
}

#[test]
fn successful_reads_are_partial_observations_not_authorization() {
    let mut client = mock();
    let result = inspect(&mut client, 123);
    assert_eq!(client.requests.len(), 15);
    assert_eq!(result["status"], "PARTIAL_UNVERIFIED");
    assert_eq!(result["drift_status"], "UNKNOWN");
    assert_eq!(result["activation_authorized"], false);
    assert_eq!(result["evidence_authenticated"], false);
    assert_eq!(
        result["observations"]["legacy_branch_protection"]["fields"]["required_approvals"]["value"],
        0
    );
    assert_eq!(
        result["observations"]["repository_rulesets_first_page"]["fields"]["complete_effective_census"]
            ["state"],
        "unknown"
    );
    assert_eq!(
        result["unassessed"]["reviewer_membership"]["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["fields"]["expected_human_login_set"]["value"],
        true
    );
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["fields"]["numeric_identity_pinned"]["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["administration_environment"]["fields"]["review_policy"]["administrator_bypass"]
            ["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["administration_branch_policies"]["fields"]["expected_main_branch_only"]
            ["value"],
        true
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert!(!result.to_string().contains("do-not-print"));
}

#[test]
fn denied_and_ambiguous_not_found_remain_unknown_without_retry() {
    let mut client = mock();
    client.replies[2].1 = response(404, json!({"message":"secret-provider-message"}));
    client.replies[5].1 = response(403, json!({"message":"secret-provider-message"}));
    let result = inspect(&mut client, 123);
    assert_eq!(
        client
            .requests
            .iter()
            .filter(|request| **request == Request::Team)
            .count(),
        1
    );
    assert_eq!(
        client
            .requests
            .iter()
            .filter(|request| **request == Request::MainBranchProtection)
            .count(),
        1
    );
    assert_eq!(
        result["observations"]["reviewer_team"]["reason"],
        "not_found_or_inaccessible"
    );
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "team_metadata_unavailable"
    );
    assert!(
        !client
            .requests
            .iter()
            .any(|request| matches!(request, Request::TeamMembersPage { .. }))
    );
    assert_eq!(
        result["observations"]["legacy_branch_protection"]["reason"],
        "access_denied"
    );
    assert_eq!(
        result["observations"]["legacy_branch_protection"]["http_status"],
        403
    );
    assert_eq!(result["drift_status"], "UNKNOWN");
    assert!(!result.to_string().contains("secret-provider-message"));
}

#[test]
fn missing_invalid_fields_are_unknown_and_no_untrusted_strings_are_echoed() {
    let mut client = mock();
    client.replies[0].1 = response(
        200,
        json!({"id":0,"full_name":"secret-provider-message","default_branch":null}),
    );
    client.replies[1].1 = response(
        200,
        json!({"name":"main","protected":"true","commit":{"sha":"not-a-sha"}}),
    );
    client.replies[5].1 = response(
        200,
        json!({"required_pull_request_reviews":{"required_approving_review_count":-1}}),
    );
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["observations"]["repository"]["fields"]["numeric_id"]["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["repository"]["fields"]["expected_name"]["value"],
        false
    );
    assert_eq!(
        result["observations"]["main_branch"]["fields"]["head_sha"]["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["legacy_branch_protection"]["fields"]["required_approvals"]["state"],
        "unknown"
    );
    assert!(!result.to_string().contains("secret-provider-message"));
}

#[test]
fn transport_failure_stops_further_requests_and_marks_them_unknown() {
    let mut client = mock();
    client.replies[2].1 = None;
    let result = inspect(&mut client, 123);
    assert_eq!(client.requests.len(), 3);
    assert_eq!(
        result["observations"]["reviewer_team"]["reason"],
        "transport_failure"
    );
    assert_eq!(
        result["observations"]["legacy_branch_protection"]["reason"],
        "not_queried_after_transport_failure"
    );
    assert!(!result.to_string().contains("do not echo"));
}

#[test]
fn stable_member_additions_and_removals_are_reported_without_account_names() {
    for (members, missing, unexpected) in [
        (
            json!([
                {"id":11,"login":"micspiral","type":"User"},
                {"id":12,"login":"ndizazzo","type":"User"}
            ]),
            json!(["i386"]),
            0,
        ),
        (
            json!([
                {"id":11,"login":"micspiral","type":"User"},
                {"id":12,"login":"ndizazzo","type":"User"},
                {"id":13,"login":"i386","type":"User"},
                {"id":14,"login":"unlisted-user","type":"User"}
            ]),
            json!([]),
            1,
        ),
    ] {
        let mut client = mock();
        client.replies[7].1 = response(200, members.clone());
        client.replies[9].1 = response(200, members);
        let result = inspect(&mut client, 123);
        assert_eq!(result["drift_status"], "UNKNOWN");
        assert_eq!(result["partial_indicators_authenticated"], false);
        assert_eq!(
            result["partial_drift_indicators"],
            json!(["reviewer_team.membership"])
        );
        let fields = &result["observations"]["reviewer_membership_names"]["fields"];
        assert_eq!(fields["missing_expected_human_logins"]["value"], missing);
        assert_eq!(fields["unexpected_member_count"]["value"], unexpected);
        assert!(!result.to_string().contains("unlisted-user"));
    }
}

#[test]
fn changed_denied_or_duplicate_member_reads_do_not_create_a_finding() {
    let mut changed = mock();
    changed.replies[9].1 = response(200, json!([]));
    changed.replies.remove(10);
    let result = inspect(&mut changed, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "membership_changed_during_collection"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert_eq!(changed.requests.len(), 13);

    let mut denied = mock();
    denied.replies[7].1 = response(403, json!({"message":"private-provider-error"}));
    let result = inspect(&mut denied, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "access_denied"
    );
    assert_eq!(denied.requests.len(), 11);
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert!(!result.to_string().contains("private-provider-error"));

    let mut duplicate = mock();
    duplicate.replies[7].1 = response(
        200,
        json!([
            {"id":11,"login":"micspiral","type":"User"},
            {"id":11,"login":"unlisted-user","type":"User"}
        ]),
    );
    let result = inspect(&mut duplicate, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "duplicate_member_identity"
    );
    assert_eq!(duplicate.requests.len(), 11);
}

#[test]
fn membership_pagination_reads_until_empty_and_rejects_a_changed_team() {
    let mut client = mock();
    client.replies.truncate(7);
    let first = json!([{"id":11,"login":"micspiral","type":"User"}]);
    let second = json!([
        {"id":12,"login":"ndizazzo","type":"User"},
        {"id":13,"login":"i386","type":"User"}
    ]);
    for _ in 0..2 {
        client.replies.extend([
            (
                Request::TeamMembersPage { page: 1 },
                response(200, first.clone()),
            ),
            (
                Request::TeamMembersPage { page: 2 },
                response(200, second.clone()),
            ),
            (
                Request::TeamMembersPage { page: 3 },
                response(200, json!([])),
            ),
        ]);
    }
    client.replies.push_back((
        Request::Team,
        response(
            200,
            json!({"id":99,"slug":"ci-policy-reviewers","organization":{"login":"Mesh-LLM"}}),
        ),
    ));
    add_branch_replies(&mut client);
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "team_identity_changed_or_unverified"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert_eq!(client.requests.len(), 17);
}

#[test]
fn membership_page_denial_and_pagination_limit_stay_unknown_without_retry() {
    let mut denied = mock();
    denied.replies[8].1 = response(403, json!({"message":"private-provider-error"}));
    let result = inspect(&mut denied, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "access_denied"
    );
    assert_eq!(denied.requests.len(), 12);
    assert_eq!(result["partial_drift_indicators"], json!([]));

    let mut bounded = mock();
    bounded.replies.truncate(7);
    for page in 1..=10 {
        bounded.replies.push_back((
            Request::TeamMembersPage { page },
            response(
                200,
                json!([{"id":page,"login":format!("member-{page}"),"type":"User"}]),
            ),
        ));
    }
    add_branch_replies(&mut bounded);
    let result = inspect(&mut bounded, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "pagination_limit_reached"
    );
    assert_eq!(bounded.requests.len(), 20);
    assert_eq!(result["drift_status"], "UNKNOWN");
}

#[test]
fn partial_indicators_never_treat_legacy_review_counts_as_effective_rules() {
    let mut client = mock();
    client.replies[2].1 = response(
        200,
        json!({"id":2,"slug":"ci-policy-reviewers","privacy":"secret","organization":{"login":"Mesh-LLM"},"parent":null}),
    );
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["partial_drift_indicators"],
        json!(["reviewer_team.visibility"])
    );
    assert_eq!(result["drift_status"], "UNKNOWN");
    assert_eq!(result["execution_authorized"], false);
}

#[test]
fn observed_repository_team_role_mismatch_is_a_non_authoritative_indicator() {
    let mut client = mock();
    client.replies[3].1 = response(
        200,
        json!({"id":1,"full_name":REPOSITORY,"role_name":"admin"}),
    );
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["partial_drift_indicators"],
        json!(["reviewer_team.repository_role"])
    );
    assert_eq!(result["drift_status"], "UNKNOWN");
    assert_eq!(result["activation_authorized"], false);
}

#[test]
fn parent_team_and_repository_permission_conflicts_are_partial_indicators() {
    let mut client = mock();
    let team = json!({"id":2,"slug":"ci-policy-reviewers","privacy":"closed","organization":{"login":"Mesh-LLM"},"parent":{"id":99,"slug":"parent-team"}});
    client.replies[2].1 = response(200, team.clone());
    client.replies[11].1 = response(200, team);
    client.replies[3].1 = response(
        200,
        json!({"id":4,"full_name":REPOSITORY,"role_name":"write", "permissions":{"pull":true,"push":false,"admin":false,"maintain":false}}),
    );
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["partial_drift_indicators"],
        json!([
            "reviewer_team.parent",
            "reviewer_team.repository_permission_flags",
            "reviewer_team.repository_numeric_identity"
        ])
    );
    assert_eq!(result["drift_status"], "UNKNOWN");
}

#[test]
fn environment_review_and_branch_mismatches_are_only_partial_indicators() {
    for (pointer, replacement, indicator) in [
        (
            "/protection_rules/0/prevent_self_review",
            json!(false),
            "administration_environment.prevent_self_review",
        ),
        (
            "/protection_rules/0/reviewers/1/reviewer/login",
            json!("unlisted-reviewer"),
            "administration_environment.reviewers",
        ),
        (
            "/deployment_branch_policy/custom_branch_policies",
            json!(false),
            "administration_environment.custom_branch_mode",
        ),
        (
            "/protection_rules",
            json!([{"id":32,"type":"branch_policy"}]),
            "administration_environment.required_reviewers_rule",
        ),
    ] {
        let mut client = mock();
        let mut environment = environment_metadata();
        *environment.pointer_mut(pointer).unwrap() = replacement;
        client.replies[4].1 = response(200, environment);
        let result = inspect(&mut client, 123);
        assert_eq!(result["partial_drift_indicators"], json!([indicator]));
        assert_eq!(result["drift_status"], "UNKNOWN");
        assert_eq!(result["evidence_authenticated"], false);
        assert!(!result.to_string().contains("unlisted-reviewer"));
    }
}

#[test]
fn malformed_environment_reviewers_and_undocumented_bypass_stay_unknown() {
    let mut client = mock();
    let mut environment = environment_metadata();
    environment["protection_rules"][0]["reviewers"][1]["reviewer"]["id"] = json!(11);
    environment["can_admins_bypass"] = json!(false);
    client.replies[4].1 = response(200, environment);
    let result = inspect(&mut client, 123);
    let policy = &result["observations"]["administration_environment"]["fields"]["review_policy"];
    assert_eq!(
        policy["expected_human_login_set"]["reason"],
        "duplicate_reviewer_identity"
    );
    assert_eq!(policy["administrator_bypass"]["state"], "unknown");
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert_eq!(result["activation_authorized"], false);
}

#[test]
fn malformed_parent_team_metadata_does_not_imply_a_parent_or_membership_result() {
    let mut client = mock();
    let mut team = team_metadata();
    team["parent"] = json!({});
    client.replies[2].1 = response(200, team);
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["observations"]["reviewer_team"]["fields"]["no_parent"]["state"],
        "unknown"
    );
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "team_identity_unavailable"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
    assert!(
        !client
            .requests
            .iter()
            .any(|request| matches!(request, Request::TeamMembersPage { .. }))
    );
}

#[test]
fn stable_unexpected_environment_refs_are_partial_drift_indicators() {
    for policies in [
        json!([]),
        json!([{"id":33,"name":"*","type":"branch"}]),
        json!([{"id":33,"name":"main","type":"tag"}]),
        json!([{"id":33,"name":"main","type":"branch"},{"id":34,"name":"private-other-branch","type":"branch"}]),
    ] {
        let page =
            json!({"total_count":policies.as_array().unwrap().len(),"branch_policies":policies});
        let mut client = mock();
        client.replies[12].1 = response(200, page.clone());
        client.replies[13].1 = response(200, page);
        let result = inspect(&mut client, 123);
        assert_eq!(
            result["partial_drift_indicators"],
            json!(["administration_environment.allowed_refs"])
        );
        assert_eq!(result["drift_status"], "UNKNOWN");
        assert!(!result.to_string().contains("private-other-branch"));
    }
}

#[test]
fn denied_incomplete_or_changing_environment_refs_stay_unknown() {
    let mut denied = mock();
    denied.replies[12].1 = response(403, json!({"message":"private-provider-error"}));
    let result = inspect(&mut denied, 123);
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "access_denied"
    );
    assert_eq!(
        denied
            .requests
            .iter()
            .filter(|r| matches!(r, Request::AdministrationEnvironmentBranches { .. }))
            .count(),
        1
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));

    let mut changing = mock();
    changing.replies[13].1 = response(
        200,
        json!({"total_count":1,"branch_policies":[{"id":33,"name":"other","type":"branch"}]}),
    );
    let result = inspect(&mut changing, 123);
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "branch_policies_changed_during_collection"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));

    let mut incomplete = mock();
    incomplete.replies[12].1 = response(
        200,
        json!({"total_count":2,"branch_policies":[{"id":33,"name":"main","type":"branch"}]}),
    );
    incomplete.replies.push_back((
        Request::AdministrationEnvironmentBranches { page: 2 },
        response(200, json!({"total_count":2,"branch_policies":[]})),
    ));
    let result = inspect(&mut incomplete, 123);
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "invalid_or_incomplete_response"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));

    let mut missing_type = mock();
    let page = json!({"total_count":1,"branch_policies":[{"id":33,"name":"main"}]});
    missing_type.replies[12].1 = response(200, page.clone());
    missing_type.replies[13].1 = response(200, page);
    let result = inspect(&mut missing_type, 123);
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "branch_policy_type_unavailable"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
}

#[test]
fn multi_page_environment_refs_are_compared_and_environment_replacement_is_unknown() {
    let mut multiple = mock();
    let first =
        json!({"total_count":2,"branch_policies":[{"id":33,"name":"main","type":"branch"}]});
    let second =
        json!({"total_count":2,"branch_policies":[{"id":34,"name":"other","type":"branch"}]});
    multiple.replies[12].1 = response(200, first.clone());
    multiple.replies[13].1 = response(200, first);
    for _ in 0..2 {
        multiple.replies.push_back((
            Request::AdministrationEnvironmentBranches { page: 2 },
            response(200, second.clone()),
        ));
    }
    let result = inspect(&mut multiple, 123);
    assert_eq!(
        result["partial_drift_indicators"],
        json!(["administration_environment.allowed_refs"])
    );
    assert_eq!(
        result["observations"]["administration_branch_policies"]["fields"]["count"]["value"],
        2
    );

    let mut changed = mock();
    let mut replacement = environment_metadata();
    replacement["id"] = json!(99);
    changed.replies[14].1 = response(200, replacement);
    let result = inspect(&mut changed, 123);
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "environment_identity_or_mode_changed"
    );
    assert_eq!(result["partial_drift_indicators"], json!([]));
}

#[test]
fn membership_transport_failure_stops_independent_branch_reads() {
    let mut client = mock();
    client.replies[7].1 = None;
    let result = inspect(&mut client, 123);
    assert_eq!(
        result["observations"]["reviewer_membership_names"]["reason"],
        "transport_failure"
    );
    assert_eq!(
        result["observations"]["administration_branch_policies"]["reason"],
        "not_queried_after_transport_failure"
    );
    assert!(
        !client
            .requests
            .iter()
            .any(|r| matches!(r, Request::AdministrationEnvironmentBranches { .. }))
    );
}

#[test]
fn all_inspection_adds_active_rules_without_claiming_authority() {
    let mut client = mock();
    for _ in 0..2 {
        client.replies.push_back((
            Request::EffectiveMainRulesPage { page: 1 },
            response(200, json!([])),
        ));
    }
    client.replies.push_back((
        Request::Repository,
        response(
            200,
            json!({"id":1,"full_name":REPOSITORY,"default_branch":"main"}),
        ),
    ));
    client.replies.push_back((
        Request::CurrentMain,
        response(
            200,
            json!({"name":"main","commit":{"sha":"1111111111111111111111111111111111111111"}}),
        ),
    ));
    let result = inspect_all(&mut client, 123);
    assert_eq!(
        result["observations"]["main_active_rulesets"]["fields"]["active_rule_count"]["value"],
        0
    );
    assert_eq!(result["drift_status"], "UNKNOWN");
    assert_eq!(result["evidence_authenticated"], false);
    assert_eq!(result["activation_authorized"], false);
}

#[test]
fn all_inspection_does_not_continue_after_transport_failure() {
    let mut client = mock();
    client.replies[2].1 = None;
    let result = inspect_all(&mut client, 123);
    assert_eq!(
        result["observations"]["main_active_rulesets"]["reason"],
        "not_queried_after_transport_failure"
    );
    assert!(
        !client
            .requests
            .iter()
            .any(|request| matches!(request, Request::EffectiveMainRulesPage { .. }))
    );
}

#[test]
fn all_inspection_does_not_use_rules_endpoints_after_protection_unknown() {
    for (status, body) in [
        (401, json!({"message":"private-provider-error"})),
        (403, json!({"message":"private-provider-error"})),
        (404, json!({"message":"private-provider-error"})),
        (500, json!({"message":"private-provider-error"})),
        (200, json!({})),
        (200, json!({"enforce_admins":{"enabled":true}})),
    ] {
        let mut client = mock();
        client.replies[5].1 = response(status, body);
        let result = inspect_all(&mut client, 123);
        assert_eq!(
            result["observations"]["repository_rulesets_first_page"]["reason"],
            "not_queried_after_protection_unknown"
        );
        assert_eq!(
            result["observations"]["main_active_rulesets"]["reason"],
            "not_queried_after_protection_unknown"
        );
        assert!(!client.requests.iter().any(|request| matches!(
            request,
            Request::RepositoryRulesetsFirstPage | Request::EffectiveMainRulesPage { .. }
        )));
        assert!(!result.to_string().contains("private-provider-error"));
    }
}

#[test]
fn all_inspection_does_not_use_active_rules_after_ruleset_list_unknown() {
    for (status, body) in [
        (401, json!({"message":"private-provider-error"})),
        (403, json!({"message":"private-provider-error"})),
        (404, json!({"message":"private-provider-error"})),
        (500, json!({"message":"private-provider-error"})),
        (200, json!({})),
    ] {
        let mut client = mock();
        client.replies[6].1 = response(status, body);
        let result = inspect_all(&mut client, 123);
        assert_eq!(
            result["observations"]["main_active_rulesets"]["reason"],
            "not_queried_after_rulesets_unknown"
        );
        assert_eq!(
            client
                .requests
                .iter()
                .filter(|request| **request == Request::RepositoryRulesetsFirstPage)
                .count(),
            1
        );
        assert!(
            !client
                .requests
                .iter()
                .any(|request| matches!(request, Request::EffectiveMainRulesPage { .. }))
        );
        assert!(!result.to_string().contains("private-provider-error"));
    }
}
