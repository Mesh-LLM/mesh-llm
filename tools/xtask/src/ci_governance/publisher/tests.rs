use super::*;
use serde_json::json;

#[test]
fn check_readback_cannot_substitute_app_name_head_operation_or_conclusion() {
    let binding = CheckBinding {
        id: 7,
        head_sha: "a".repeat(40),
        external_id: "fixed-binding".into(),
        app_id: 4,
    };
    let baseline = json!({"id":7,"head_sha":"a".repeat(40),"name":CHECK_NAME,
        "external_id":"fixed-binding","app":{"id":4},"status":"in_progress","conclusion":null});
    assert!(
        binding
            .verify(&serde_json::from_value(baseline.clone()).unwrap(), None)
            .is_ok()
    );
    for (pointer, replacement) in [
        ("/id", json!(8)),
        ("/head_sha", json!("b".repeat(40))),
        ("/name", json!("forged")),
        ("/external_id", json!("other")),
        ("/app/id", json!(15368)),
        ("/status", json!("completed")),
        ("/conclusion", json!("success")),
    ] {
        let mut value = baseline.clone();
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            binding
                .verify(&serde_json::from_value(value).unwrap(), None)
                .is_err(),
            "{pointer}"
        );
    }
}

struct FailedRead;
impl GithubRead for FailedRead {
    fn send(&mut self, _: &Request) -> DynResult<collector::ApiResponse> {
        Err("fixture initial collection unavailable".into())
    }
}
struct NoWrites;
impl GithubRead for NoWrites {
    fn send(&mut self, _: &Request) -> DynResult<collector::ApiResponse> {
        panic!("an initial read failure must not claim check replacement")
    }
}
struct Fixed;
impl Clock for Fixed {
    fn now(&self) -> DynResult<u64> {
        Ok(1_800_000_000)
    }
}

#[test]
fn initial_api_outage_does_not_claim_that_a_previous_green_check_was_replaced() {
    let anchors = activation::Anchors {
        repository_id: 1,
        reviewer_team_id: 2,
        administration_environment_id: 3,
        observer_app_id: 4,
        observer_installation_id: 5,
        publisher_app_id: 4,
        publisher_installation_id: 5,
        native_ruleset_ids: vec![8],
        human_ids: [
            ("micspiral".into(), 11),
            ("ndizazzo".into(), 12),
            ("i386".into(), 13),
        ]
        .into(),
    };
    assert!(
        publish(
            &mut FailedRead,
            &mut NoWrites,
            &anchors,
            &"a".repeat(40),
            1,
            &Fixed
        )
        .is_err()
    );
}
