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
        let (expected, reply) = self.replies.pop_front().expect("unexpected request");
        assert_eq!(*request, expected);
        reply.ok_or_else(|| "transport failure, do not echo this error".into())
    }
}

fn response(status: u16, body: Value) -> Option<ApiResponse> {
    Some(ApiResponse {
        status,
        body: serde_json::to_vec(&body).unwrap(),
    })
}

fn mock() -> Mock {
    Mock {
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
            (
                Request::Team,
                response(
                    200,
                    json!({"id":2,"slug":"ci-policy-reviewers","privacy":"closed"}),
                ),
            ),
            (
                Request::TeamPermission,
                response(
                    200,
                    json!({"id":1,"full_name":REPOSITORY,"role_name":"write"}),
                ),
            ),
            (
                Request::AdministrationEnvironment,
                response(
                    200,
                    json!({"id":3,"name":"ci-governance-administration","protection_rules":[]}),
                ),
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
        ]),
    }
}

#[test]
fn successful_reads_are_partial_observations_not_authorization() {
    let mut client = mock();
    let result = inspect(&mut client, 123);
    assert_eq!(client.requests.len(), 7);
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
