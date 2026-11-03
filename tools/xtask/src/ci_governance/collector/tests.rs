use super::*;
use crate::ci_governance::{activation::Anchors, approval::evaluate_at};
use serde_json::{Value, json};
use std::collections::VecDeque;

const MAIN: &str = "1111111111111111111111111111111111111111";
const HEAD: &str = "2222222222222222222222222222222222222222";
const NOW: u64 = 1_700_000_000;
struct FixedClock;
impl Clock for FixedClock {
    fn now(&self) -> crate::command::DynResult<u64> {
        Ok(NOW)
    }
}
struct Mock {
    queue: VecDeque<(Request, ApiResponse)>,
}
impl GithubRead for Mock {
    fn send(&mut self, request: &Request) -> crate::command::DynResult<ApiResponse> {
        let (expected, response) = self.queue.pop_front().expect("unexpected API call");
        assert_eq!(*request, expected);
        Ok(response)
    }
}
impl Mock {
    fn change(
        &mut self,
        predicate: impl Fn(&Request) -> bool,
        occurrence: usize,
        change: impl FnOnce(&mut Value),
    ) {
        let (_, response) = self
            .queue
            .iter_mut()
            .filter(|(request, _)| predicate(request))
            .nth(occurrence)
            .unwrap();
        let mut value: Value = serde_json::from_slice(&response.body).unwrap();
        change(&mut value);
        response.body = serde_json::to_vec(&value).unwrap();
    }
}
fn anchors() -> Anchors {
    Anchors {
        repository_id: 1,
        reviewer_team_id: 2,
        administration_environment_id: 3,
        observer_app_id: 4,
        observer_installation_id: 5,
        publisher_app_id: 6,
        publisher_installation_id: 7,
        native_ruleset_ids: vec![8],
        human_ids: std::collections::BTreeMap::from([
            ("micspiral".into(), 11),
            ("ndizazzo".into(), 12),
            ("i386".into(), 13),
        ]),
    }
}
fn reply(value: Value) -> ApiResponse {
    ApiResponse {
        status: 200,
        body: serde_json::to_vec(&value).unwrap(),
    }
}
fn actor(login: &str, id: u64) -> Value {
    json!({"__typename":"User","login":login,"databaseId":id})
}
fn reviews() -> Value {
    json!({"data":{"repository":{"pullRequest":{"reviews":{"totalCount":1,"nodes":[{
    "fullDatabaseId":"4294967296","state":"APPROVED","submittedAt":"2023-11-14T22:13:20Z","commit":{"oid":HEAD},"author":actor("micspiral",11)}],
    "pageInfo":{"hasNextPage":false,"endCursor":null}}}}}})
}
fn members() -> Value {
    json!({"data":{"organization":{"team":{"databaseId":2,"members":{"totalCount":3,
    "nodes":[actor("micspiral",11),actor("ndizazzo",12),actor("i386",13)],"pageInfo":{"hasNextPage":false,"endCursor":null}}}}}})
}
fn identity(queue: &mut VecDeque<(Request, ApiResponse)>, count: usize) {
    queue.push_back((
        Request::Repository,
        reply(json!({"id":1,"full_name":"Mesh-LLM/mesh-llm","default_branch":"main"})),
    ));
    queue.push_back((
        Request::CurrentMain,
        reply(json!({"name":"main","protected":true,"commit":{"sha":MAIN}})),
    ));
    queue.push_back((Request::Pull{number:2781},reply(json!({"number":2781,"state":"open",
        "user":{"id":99,"login":"contributor","type":"User"},"base":{"ref":"main","sha":MAIN,"repo":{"id":1,"full_name":"Mesh-LLM/mesh-llm"}},
        "head":{"sha":HEAD},"changed_files":count}))));
}
fn team(queue: &mut VecDeque<(Request, ApiResponse)>) {
    queue.push_back((Request::Team,reply(json!({"id":2,"slug":"ci-policy-reviewers","privacy":"closed","organization":{"login":"Mesh-LLM"},"parent":null}))));
    queue.push_back((Request::TeamMembers { cursor: None }, reply(members())));
    queue.push_back((Request::TeamPermission,reply(json!({"id":1,"full_name":"Mesh-LLM/mesh-llm","permissions":{"push":true,"pull":true,"admin":false,"maintain":false},"role_name":"write"}))));
}
fn flow(files: Vec<Value>, controlled: bool) -> Mock {
    let mut queue = VecDeque::new();
    identity(&mut queue, files.len());
    queue.push_back((
        Request::Policy {
            revision: MAIN.into(),
        },
        ApiResponse {
            status: 200,
            body: include_bytes!("../../../../../ci/governance/policy.json").to_vec(),
        },
    ));
    if files.is_empty() {
        queue.push_back((
            Request::Files {
                number: 2781,
                page: 1,
            },
            reply(json!([])),
        ));
    }
    for (page, chunk) in files.chunks(100).enumerate() {
        queue.push_back((
            Request::Files {
                number: 2781,
                page: page as u32 + 1,
            },
            reply(json!(chunk)),
        ));
    }
    if controlled {
        team(&mut queue);
        queue.push_back((
            Request::Reviews {
                number: 2781,
                cursor: None,
            },
            reply(reviews()),
        ));
        queue.push_back((
            Request::Reviews {
                number: 2781,
                cursor: None,
            },
            reply(reviews()),
        ));
        team(&mut queue);
    }
    identity(&mut queue, files.len());
    Mock { queue }
}
fn policy_file() -> Vec<Value> {
    vec![
        json!({"filename":"ci/governance/policy.json","status":"modified","previous_filename":null}),
    ]
}

#[test]
fn collected_protected_policy_and_each_independent_human_passes() {
    for (login, id) in [("micspiral", 11), ("ndizazzo", 12), ("i386", 13)] {
        let mut mock = flow(policy_file(), true);
        for occurrence in 0..2 {
            mock.change(
                |request| matches!(request, Request::Reviews { .. }),
                occurrence,
                |value| {
                    value["data"]["repository"]["pullRequest"]["reviews"]["nodes"][0]["author"] =
                        actor(login, id);
                },
            );
        }
        let collected = collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).unwrap();
        let decision = evaluate_at(
            &collected.policy,
            &collected.snapshot,
            &collected.policy_sha256,
            NOW,
        )
        .unwrap();
        assert_eq!(decision.independent_approvers, [login]);
        assert_eq!(collected.snapshot.reviews[0].id, 4_294_967_296);
        assert_eq!(
            collected.policy_bytes,
            include_bytes!("../../../../../ci/governance/policy.json")
        );
        assert!(mock.queue.is_empty());
    }
}

#[test]
fn review_connections_read_every_page_and_verify_totals() {
    let mut mock = flow(policy_file(), true);
    let mut queue = VecDeque::new();
    for (request, response) in mock.queue {
        if matches!(request, Request::Reviews { .. }) {
            let template =
                reviews()["data"]["repository"]["pullRequest"]["reviews"]["nodes"][0].clone();
            for (page, start, end) in [(0, 1, 101), (1, 101, 102)] {
                let mut value = reviews();
                let connection = &mut value["data"]["repository"]["pullRequest"]["reviews"];
                connection["totalCount"] = json!(101);
                connection["nodes"] = json!(
                    (start..end)
                        .map(|id| {
                            let mut node = template.clone();
                            node["fullDatabaseId"] = json!(id.to_string());
                            node
                        })
                        .collect::<Vec<_>>()
                );
                connection["pageInfo"] = if page == 0 {
                    json!({"hasNextPage":true,"endCursor":"next"})
                } else {
                    json!({"hasNextPage":false,"endCursor":null})
                };
                queue.push_back((
                    Request::Reviews {
                        number: 2781,
                        cursor: if page == 0 { None } else { Some("next".into()) },
                    },
                    reply(value),
                ));
            }
        } else {
            queue.push_back((request, response));
        }
    }
    mock.queue = queue;
    assert_eq!(
        collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock)
            .unwrap()
            .snapshot
            .reviews
            .len(),
        101
    );
    assert!(mock.queue.is_empty());
}

#[test]
fn member_pages_require_stable_totals_unique_cursors_and_complete_termination() {
    fn page(total: usize, ids: std::ops::Range<u64>, next: bool, cursor: Option<&str>) -> Value {
        let mut value = members();
        let connection = &mut value["data"]["organization"]["team"]["members"];
        connection["totalCount"] = json!(total);
        connection["nodes"] = json!(
            ids.map(|id| actor(&format!("user-{id}"), id))
                .collect::<Vec<_>>()
        );
        connection["pageInfo"] = json!({"hasNextPage":next,"endCursor":cursor});
        value
    }
    let mut mock = Mock {
        queue: VecDeque::from([
            (
                Request::TeamMembers { cursor: None },
                reply(page(101, 1..101, true, Some("next"))),
            ),
            (
                Request::TeamMembers {
                    cursor: Some("next".into()),
                },
                reply(page(101, 101..102, false, None)),
            ),
        ]),
    };
    assert_eq!(pagination::members(&mut mock, 2).unwrap().len(), 101);
    for last in [
        page(102, 101..102, false, None),
        page(101, 100..101, false, None),
        page(101, 101..102, true, Some("next")),
        page(101, 101..101, false, None),
    ] {
        let mut mock = Mock {
            queue: VecDeque::from([
                (
                    Request::TeamMembers { cursor: None },
                    reply(page(101, 1..101, true, Some("next"))),
                ),
                (
                    Request::TeamMembers {
                        cursor: Some("next".into()),
                    },
                    reply(last),
                ),
            ]),
        };
        assert!(pagination::members(&mut mock, 2).is_err());
    }
    let mut empty = Mock {
        queue: VecDeque::new(),
    };
    assert!(pagination::files(&mut empty, 2781, 3001).is_err());
}

#[test]
fn invalid_bigint_and_reassigned_eligible_account_ids_are_rejected() {
    for id in [
        json!(10),
        json!("0"),
        json!("-1"),
        json!("18446744073709551616"),
    ] {
        let mut mock = flow(policy_file(), true);
        mock.change(|request|matches!(request,Request::Reviews{..}),0,|value|{
            value["data"]["repository"]["pullRequest"]["reviews"]["nodes"][0]["fullDatabaseId"]=id;
        });
        assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
    }
    let mut mock = flow(policy_file(), true);
    for occurrence in 0..2 {
        mock.change(|request|matches!(request,Request::Reviews{..}),occurrence,|value|{
        value["data"]["repository"]["pullRequest"]["reviews"]["nodes"][0]["author"]["databaseId"]=json!(999);
    });
    }
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
}

#[test]
fn denied_approval_still_returns_collected_evidence_for_publisher() {
    let mut mock = flow(policy_file(), true);
    for occurrence in 0..2 {
        mock.change(
            |request| matches!(request, Request::Reviews { .. }),
            occurrence,
            |value| {
                value["data"]["repository"]["pullRequest"]["reviews"]["nodes"] = json!([]);
                value["data"]["repository"]["pullRequest"]["reviews"]["totalCount"] = json!(0);
            },
        );
    }
    let collected = collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).unwrap();
    assert!(
        evaluate_at(
            &collected.policy,
            &collected.snapshot,
            &collected.policy_sha256,
            NOW
        )
        .is_err()
    );
}

#[test]
fn product_only_collection_skips_extra_approval_and_membership_reads() {
    let files = vec![
        json!({"filename":"mesh/crates/mesh-llm-ui/src/example.test.tsx","status":"modified"}),
    ];
    let mut mock = flow(files, false);
    let collected = collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).unwrap();
    assert!(
        !evaluate_at(
            &collected.policy,
            &collected.snapshot,
            &collected.policy_sha256,
            NOW
        )
        .unwrap()
        .policy_change
    );
    assert!(mock.queue.is_empty());
}

#[test]
fn file_pages_are_complete_at_and_beyond_exact_hundred_boundary() {
    for count in [100, 101] {
        let files = (0..count)
            .map(
                |index| json!({"filename":format!("mesh/src/file-{index}.rs"),"status":"modified"}),
            )
            .collect();
        let mut mock = flow(files, false);
        assert_eq!(
            collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock)
                .unwrap()
                .snapshot
                .changes
                .len(),
            count
        );
        assert!(mock.queue.is_empty());
    }
}

#[test]
fn renamed_deleted_and_newline_control_paths_are_preserved() {
    for file in [
        json!({"filename":"mesh/new.rs","status":"renamed","previous_filename":"ci/old\npolicy.json"}),
        json!({"filename":"ci/old\npolicy.json","status":"removed"}),
    ] {
        let mut mock = flow(vec![file], true);
        assert!(
            collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock)
                .unwrap()
                .snapshot
                .team
                .is_some()
        );
    }
}

#[test]
fn historical_controller_and_wrong_numeric_anchor_are_rejected() {
    let mut mock = flow(policy_file(), true);
    assert!(collect(&mut mock, HEAD, 2781, &anchors(), &FixedClock).is_err());
    let mut wrong = anchors();
    wrong.repository_id = 99;
    let mut mock = flow(policy_file(), true);
    assert!(collect(&mut mock, MAIN, 2781, &wrong, &FixedClock).is_err());
}

#[test]
fn inherited_parent_team_access_or_missing_parent_metadata_fails() {
    for missing in [true, false] {
        let mut mock = flow(policy_file(), true);
        mock.change(
            |request| matches!(request, Request::Team),
            0,
            |value| {
                if missing {
                    value.as_object_mut().unwrap().remove("parent");
                } else {
                    value["parent"] = json!({"id":999,"slug":"parent-team"});
                }
            },
        );
        assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
    }
}

#[test]
fn exact_write_role_requires_positive_evidence_and_no_privilege_contradiction() {
    for variant in 0..4 {
        let mut mock = flow(policy_file(), true);
        mock.change(
            |request| matches!(request, Request::TeamPermission),
            0,
            |value| match variant {
                0 => {
                    value.as_object_mut().unwrap().remove("role_name");
                }
                1 => {
                    value.as_object_mut().unwrap().remove("role_name");
                    value["permissions"]
                        .as_object_mut()
                        .unwrap()
                        .remove("maintain");
                }
                2 => value["permissions"]["maintain"] = json!(true),
                _ => value["permissions"]["admin"] = json!(true),
            },
        );
        assert!(
            collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err(),
            "variant {variant}"
        );
    }
    let mut mock = flow(policy_file(), true);
    for occurrence in 0..2 {
        mock.change(
            |request| matches!(request, Request::TeamPermission),
            occurrence,
            |value| {
                value["permissions"]
                    .as_object_mut()
                    .unwrap()
                    .remove("maintain");
            },
        );
    }
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_ok());
}

#[test]
fn main_head_author_file_count_and_membership_changes_during_collection_fail() {
    for pointer in ["/head/sha", "/base/sha", "/user/id", "/changed_files"] {
        let mut mock = flow(policy_file(), true);
        mock.change(
            |request| matches!(request, Request::Pull { .. }),
            1,
            |value| {
                *value.pointer_mut(pointer).unwrap() = if pointer.ends_with("sha") {
                    json!("3".repeat(40))
                } else {
                    json!(100)
                };
            },
        );
        assert!(
            collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err(),
            "{pointer}"
        );
    }
    let mut mock = flow(policy_file(), true);
    mock.change(
        |request| matches!(request, Request::TeamMembers { .. }),
        1,
        |value| {
            value["data"]["organization"]["team"]["members"]["nodes"][0]["login"] =
                json!("changed");
        },
    );
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
}

#[test]
fn review_dismissal_or_permission_change_between_passes_fails() {
    for reviews_changed in [true, false] {
        let mut mock = flow(policy_file(), true);
        if reviews_changed {
            mock.change(
                |request| matches!(request, Request::Reviews { .. }),
                1,
                |value| {
                    value["data"]["repository"]["pullRequest"]["reviews"]["nodes"][0]["state"] =
                        json!("DISMISSED");
                },
            );
        } else {
            mock.change(
                |request| matches!(request, Request::TeamPermission),
                1,
                |value| {
                    value["permissions"]["push"] = json!(false);
                    value["role_name"] = json!("read");
                },
            );
        }
        assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
    }
}

#[test]
fn partial_graphql_duplicate_files_and_missing_file_pages_fail_closed() {
    let mut mock = flow(policy_file(), true);
    mock.change(
        |request| matches!(request, Request::Reviews { .. }),
        0,
        |value| {
            value["errors"] = json!([{"message":"fixture"}]);
        },
    );
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
    let mut mock = flow(
        vec![policy_file()[0].clone(), policy_file()[0].clone()],
        true,
    );
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
    let mut mock = flow(policy_file(), true);
    mock.change(
        |request| matches!(request, Request::Files { .. }),
        0,
        |value| *value = json!([]),
    );
    assert!(collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock).is_err());
}

#[test]
fn forbidden_status_never_reflects_provider_body() {
    let mut mock = flow(policy_file(), true);
    let (_, response) = mock.queue.front_mut().unwrap();
    response.status = 403;
    response.body = b"credential-fixture-secret".to_vec();
    let error = collect(&mut mock, MAIN, 2781, &anchors(), &FixedClock)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("403"));
    assert!(!error.contains("credential-fixture-secret"));
}

#[test]
fn github_timestamp_calendar_and_epoch_are_strict() {
    assert_eq!(wire::timestamp("2023-11-14T22:13:20Z").unwrap(), NOW);
    assert_eq!(wire::timestamp("1970-01-01T00:00:00Z").unwrap(), 0);
    for value in [
        "2023-02-29T00:00:00Z",
        "2024-02-30T00:00:00Z",
        "2024-01-01T24:00:00Z",
        "2024-01-01T00:00:00+00:00",
    ] {
        assert!(wire::timestamp(value).is_err());
    }
}
