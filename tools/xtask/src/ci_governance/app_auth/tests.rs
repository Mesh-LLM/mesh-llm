use super::*;
use serde_json::{Value, json};
use std::collections::VecDeque;

fn anchors() -> Anchors {
    Anchors {
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
    }
}
struct Fixed;
impl Clock for Fixed {
    fn now(&self) -> DynResult<u64> {
        Ok(1_800_000_000)
    }
}
struct Fake(VecDeque<(Request, u16, Value)>);
impl GithubRead for Fake {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
        let (expected, status, value) = self
            .0
            .pop_front()
            .expect("unexpected authenticated API call");
        assert_eq!(request, &expected);
        Ok(ApiResponse {
            status,
            body: serde_json::to_vec(&value)?,
        })
    }
}
fn installation() -> Value {
    json!({"id":5,"app_id":4,"target_type":"Organization","account":{"login":"Mesh-LLM"},
        "permissions":app_permissions(),"repository_selection":"selected","suspended_at":null})
}
fn token(publisher: bool) -> Value {
    let permissions: BTreeMap<String, String> = if publisher {
        PUBLISHER_PERMISSIONS.as_slice()
    } else {
        OBSERVER_PERMISSIONS.as_slice()
    }
    .iter()
    .map(|(key, value)| ((*key).into(), (*value).into()))
    .collect();
    json!({"token":"fixture-token-never-in-diagnostics","expires_at":"2027-01-15T08:10:00Z",
        "permissions":permissions,"repositories":[{"id":1,"full_name":"Mesh-LLM/mesh-llm"}],"repository_selection":"selected"})
}

#[test]
fn app_and_installation_both_need_exact_numeric_anchors_and_no_administration_write() {
    for variant in 0..5 {
        let mut installed = installation();
        match variant {
            0 => installed["app_id"] = json!(99),
            1 => installed["id"] = json!(99),
            2 => installed["permissions"]["administration"] = json!("write"),
            3 => installed["repository_selection"] = json!("all"),
            _ => installed["suspended_at"] = json!("2027-01-15T00:00:00Z"),
        }
        let mut client = Fake(
            [
                (
                    Request::App,
                    200,
                    json!({"id":4,"permissions":app_permissions()}),
                ),
                (Request::AppInstallation { id: 5 }, 200, installed),
            ]
            .into(),
        );
        assert!(verify_app(&mut client, &anchors()).is_err());
    }
}

#[test]
fn minted_tokens_reject_broader_permissions_wrong_repository_and_expired_grants() {
    for publisher in [false, true] {
        for variant in 0..5 {
            let mut reply = token(publisher);
            match variant {
                0 => reply["permissions"]["administration"] = json!("write"),
                1 => reply["repositories"][0]["id"] = json!(99),
                2 => reply["repository_selection"] = json!("all"),
                3 => reply["expires_at"] = json!("2000-01-01T00:00:00Z"),
                _ => reply["repositories"] = json!([]),
            }
            let request = if publisher {
                Request::MintPublisherToken {
                    installation_id: 5,
                    repository_id: 1,
                }
            } else {
                Request::MintObserverToken {
                    installation_id: 5,
                    repository_id: 1,
                }
            };
            let mut client = Fake([(request, 201, reply)].into());
            let error = mint(&mut client, &anchors(), publisher, &Fixed)
                .err()
                .unwrap()
                .to_string();
            assert!(!error.contains("fixture-token-never-in-diagnostics"));
        }
    }
}

#[test]
fn protocol_refusals_never_echo_error_body_credentials() {
    let result = response::<AppReply>(
        ApiResponse {
            status: 403,
            body: b"credential-reflection-sentinel".to_vec(),
        },
        200,
    );
    assert!(
        !result
            .err()
            .unwrap()
            .to_string()
            .contains("credential-reflection-sentinel")
    );
    let result = response::<AppReply>(
        ApiResponse {
            status: 200,
            body: br#"{"id":"credential-reflection-sentinel"}"#.to_vec(),
        },
        200,
    );
    assert!(
        !result
            .err()
            .unwrap()
            .to_string()
            .contains("credential-reflection-sentinel")
    );
}
