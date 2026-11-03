use super::*;
use crate::ci_governance::{
    collector::ApiResponse,
    operations::preparation_fixtures::{self, ID, NOW, fixture},
};
use std::collections::{BTreeMap, VecDeque};

struct Replies(VecDeque<(Request, Value)>);
impl GithubRead for Replies {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
        let (expected, value) = self.0.pop_front().ok_or("unexpected preparation request")?;
        assert_eq!(&expected, request);
        Ok(ApiResponse {
            status: 200,
            body: serde_json::to_vec(&value)?,
        })
    }
}

fn locator() -> RuntimeLocator {
    RuntimeLocator {
        run_id: 77,
        attempt: 2,
        sha: "a".repeat(40),
        actor_id: 101,
        triggering_actor: "micspiral".into(),
    }
}

fn runtime_values() -> [(Request, Value); 3] {
    let repository = json!({"id":1,"full_name":REPOSITORY,"default_branch":"main"});
    [
        (Request::Repository, repository.clone()),
        (
            Request::CurrentMain,
            json!({"name":"main","protected":true,"commit":{"sha":"a".repeat(40)}}),
        ),
        (
            Request::WorkflowRun { run_id: 77 },
            json!({"id":77,"run_attempt":2,"head_branch":"main","head_sha":"a".repeat(40),
            "event":"workflow_dispatch","path":WORKFLOW,"status":"in_progress","repository":repository,
            "actor":{"login":"micspiral","id":101,"type":"User"},
            "triggering_actor":{"login":"micspiral","id":101,"type":"User"}}),
        ),
    ]
}

fn prepared(mutate_repeat: bool) -> DynResult<Value> {
    let (policy, expected, intent, anchors) = fixture();
    let initial = runtime_values();
    let mut client = Replies(VecDeque::from(initial));
    let runtime = collect_runtime(&mut client, &anchors, &locator())?;
    client.0.extend([
        (
            Request::AdministrationEnvironment,
            preparation_fixtures::environment(),
        ),
        (
            Request::AdministrationEnvironmentBranches { page: 1 },
            preparation_fixtures::branches(),
        ),
        (
            Request::WorkflowRunApprovals { run_id: 77 },
            json!([{"state":"approved",
            "user":{"login":"ndizazzo","id":102,"type":"User"},
            "environments":[{"id":3,"name":"ci-governance-administration"}]}]),
        ),
    ]);
    let mut repeated = runtime_values();
    if mutate_repeat {
        repeated[1].1["commit"]["sha"] = json!("b".repeat(40));
    }
    client.0.extend(repeated);
    let resolved = ResolvedIntent {
        protected_main_revision: "a".repeat(40),
        record_sha256: "c".repeat(64),
        intent,
        policy,
        expected,
    };
    let packet = prepare(&mut client, &anchors, &locator(), runtime, resolved, NOW)?;
    assert!(client.0.is_empty());
    Ok(packet)
}

#[test]
fn concrete_packet_is_bounded_and_never_becomes_execution_authority() {
    let result = prepared(false).unwrap();
    assert_eq!(result["status"], "PREPARED_PENDING_AUTHORIZATION");
    assert_eq!(result["packet"]["operation_id"], ID);
    assert_eq!(result["packet"]["runtime"]["execution"]["run_attempt"], 2);
    assert_eq!(result["packet"]["expires_at_unix"], NOW + 3600);
    assert_eq!(
        result["packet"]["environment_configuration"]["administrator_bypass"],
        Value::Null
    );
    assert_eq!(
        result["packet"]["environment_configuration"]["configuration_verified"],
        false
    );
    assert_eq!(
        result["packet"]["manual_instructions"][0]["control"],
        "ci-ruleset"
    );
    assert_eq!(result["packet_sha256"], digest(&result["packet"]).unwrap());
    for pointer in [
        "/execution_authorized",
        "/credential_release_authorized",
        "/authorization/exact_attempt_and_packet_approval_verified",
        "/authorization/live_state_precondition_verified",
        "/authorization/replay_consumption_verified",
        "/authorization/runtime_attested",
    ] {
        assert_eq!(
            result.pointer(pointer).unwrap().as_bool(),
            Some(false),
            "{pointer}"
        );
    }
    assert!(!result.to_string().contains("access_token"));
}

#[test]
fn current_main_change_during_preparation_refuses_the_packet() {
    assert!(prepared(true).is_err());
}

#[test]
fn only_fixed_record_identity_flags_and_protected_runtime_are_supported() {
    for args in [
        vec!["--record-id", ID, "--record-path", "/tmp/record"],
        vec!["--record-id", ID, "--record-id", ID],
        vec!["--record-id", ID],
    ] {
        assert!(selector(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
    }
    let mut values = BTreeMap::from([
        ("GITHUB_REPOSITORY", REPOSITORY.to_owned()),
        ("GITHUB_REF", "refs/heads/main".into()),
        ("GITHUB_EVENT_NAME", "workflow_dispatch".into()),
        ("GITHUB_SHA", "a".repeat(40)),
        ("GITHUB_WORKFLOW_SHA", "a".repeat(40)),
        (
            "GITHUB_WORKFLOW_REF",
            format!("{REPOSITORY}/{WORKFLOW}@refs/heads/main"),
        ),
        ("GITHUB_RUN_ID", "77".into()),
        ("GITHUB_RUN_ATTEMPT", "2".into()),
        ("GITHUB_ACTOR_ID", "101".into()),
        ("GITHUB_TRIGGERING_ACTOR", "micspiral".into()),
    ]);
    assert!(RuntimeLocator::from_lookup(|key| values.get(key).cloned()).is_ok());
    values.insert("GITHUB_REF", "refs/heads/feature".into());
    assert!(RuntimeLocator::from_lookup(|key| values.get(key).cloned()).is_err());
}

#[test]
fn authenticated_run_attempt_source_and_human_actor_substitutions_are_refused() {
    let (_, _, _, anchors) = fixture();
    for (pointer, value) in [
        ("/run_attempt", json!(1)),
        ("/head_sha", json!("b".repeat(40))),
        ("/actor/type", json!("Bot")),
        ("/triggering_actor/id", json!(0)),
        ("/path", json!(".github/workflows/other.yml")),
    ] {
        let mut replies = runtime_values();
        *replies[2].1.pointer_mut(pointer).unwrap() = value;
        let mut client = Replies(VecDeque::from(replies));
        assert!(
            collect_runtime(&mut client, &anchors, &locator()).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn documented_workflow_path_forms_remain_fixed_to_this_repository_and_main() {
    let (_, _, _, anchors) = fixture();
    for path in [
        format!("{WORKFLOW}@main"),
        format!("{REPOSITORY}/{WORKFLOW}@main"),
        format!("{WORKFLOW}@refs/heads/main"),
        format!("{REPOSITORY}/{WORKFLOW}@refs/heads/main"),
    ] {
        let mut replies = runtime_values();
        replies[2].1["path"] = json!(path);
        assert!(
            collect_runtime(&mut Replies(VecDeque::from(replies)), &anchors, &locator()).is_ok()
        );
    }
    for path in [
        format!("{WORKFLOW}@feature"),
        format!("foreign/repository/{WORKFLOW}@main"),
        format!("{WORKFLOW}@refs/tags/main"),
        format!("{WORKFLOW}.extra@main"),
    ] {
        let mut replies = runtime_values();
        replies[2].1["path"] = json!(path);
        assert!(
            collect_runtime(&mut Replies(VecDeque::from(replies)), &anchors, &locator()).is_err()
        );
    }
}
