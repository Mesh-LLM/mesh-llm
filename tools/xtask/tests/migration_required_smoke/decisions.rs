use super::super::evidence::{Evidence, RESPONSE_LIMIT, classify};
use super::{support::*, *};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    check: FixtureCheck,
    body: String,
    expected: Expected,
    model: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum FixtureCheck {
    Runtime,
    Models,
    Chat,
    Auto,
    Stream,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Expected {
    Advance,
    Pending,
    Reject,
}

#[test]
fn source_derived_cases_enforce_product_evidence_types() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("fixtures/standalone.json")).unwrap();
    assert_eq!(fixtures.cases.len(), 25);
    for case in fixtures.cases {
        let check = match case.check {
            FixtureCheck::Runtime => Check::Runtime,
            FixtureCheck::Models => Check::Models,
            FixtureCheck::Chat => Check::Chat,
            FixtureCheck::Auto => Check::Auto,
            FixtureCheck::Stream => Check::Stream,
        };
        let actual = classify(
            check,
            response(case.body.as_bytes()),
            &Attestation::Disabled,
        );
        match case.expected {
            Expected::Advance => match (actual, case.model) {
                (Ok(Evidence::Model(actual)), Some(expected)) => {
                    assert_eq!(actual, expected, "{}", case.name)
                }
                (Ok(Evidence::Accepted), None) => (),
                _ => panic!("{} did not advance as expected", case.name),
            },
            Expected::Pending => assert!(matches!(actual, Ok(Evidence::Pending)), "{}", case.name),
            Expected::Reject => assert_eq!(
                actual.err(),
                Some(Rejection::Evidence(check)),
                "{}",
                case.name
            ),
        }
    }
}

#[test]
fn models_require_a_nonempty_string_identifier() {
    for body in [
        br#"{"data":[{"id":null}]}"#.as_slice(),
        br#"{"data":[{"id":7}]}"#,
        br#"{"data":[{"id":false}]}"#,
        br#"{"data":[{"id":[]}]}"#,
        br#"{"data":[{"id":{}}]}"#,
        br#"{"data":[{"id":" \n"}]}"#,
    ] {
        let result = classify(Check::Models, response(body), &Attestation::Disabled);
        assert!(matches!(result, Ok(Evidence::Pending)));
    }
    for id in ["org/repo:Q4_K_M", "fixture", "opaque\nidentifier"] {
        let body = serde_json::json!({"data": [{"id": id}]}).to_string();
        let result = classify(
            Check::Models,
            response(body.as_bytes()),
            &Attestation::Disabled,
        );
        assert!(matches!(result, Ok(Evidence::Model(model)) if model == id));
    }
}

#[test]
fn runtime_readiness_requires_a_boolean() {
    for body in [
        br#"{"llama_ready":"True"}"#.as_slice(),
        br#"{"llama_ready":1}"#,
        br#"{"llama_ready":null}"#,
    ] {
        assert!(matches!(
            classify(Check::Runtime, response(body), &Attestation::Disabled),
            Ok(Evidence::Pending)
        ));
    }
}

#[test]
fn attestation_status_requires_the_exact_string() {
    let requirement = Attestation::Required {
        expected: "missing".into(),
    };
    for status in [
        serde_json::json!(null),
        serde_json::json!(false),
        serde_json::json!(7),
        serde_json::json!(["missing"]),
        serde_json::json!({"status": "missing"}),
        serde_json::json!("missing\n"),
    ] {
        let body = serde_json::json!({"status": status}).to_string();
        assert_eq!(
            classify(
                Check::InspectAttestation,
                response(body.as_bytes()),
                &requirement
            )
            .err(),
            Some(Rejection::Evidence(Check::InspectAttestation))
        );
    }
}

#[test]
fn chat_evidence_requires_product_assistant_text() {
    for content in [
        serde_json::json!(7),
        serde_json::json!(true),
        serde_json::json!(["text"]),
        serde_json::json!({"text": "text"}),
    ] {
        let body = serde_json::json!({"object": "chat.completion",
            "choices": [{"message": {"content": content}}]})
        .to_string();
        for check in [Check::Chat, Check::Auto] {
            assert_eq!(
                classify(check, response(body.as_bytes()), &Attestation::Disabled).err(),
                Some(Rejection::Evidence(check))
            );
        }
    }
}

#[test]
fn headless_reachability_does_not_silently_become_a_json_gate() {
    for check in [Check::HeadlessModels, Check::HeadlessStatus] {
        let result = classify(check, response(b"not JSON"), &Attestation::Disabled);
        assert!(matches!(result, Ok(Evidence::Accepted)));
    }
}

#[test]
fn failed_readiness_transfer_is_pending_but_failed_chat_is_terminal() {
    for (check, pending) in [
        (Check::Runtime, true),
        (Check::Models, true),
        (Check::Chat, false),
    ] {
        let result = classify(check, Transfer::Failed, &Attestation::Disabled);
        assert_eq!(matches!(result, Ok(Evidence::Pending)), pending);
    }
}

#[test]
fn transfer_failure_cannot_be_overridden_by_valid_body() {
    let result = classify(
        Check::Chat,
        Transfer::Complete(Response {
            status: 500,
            body: br#"{"object":"chat.completion","choices":[{"message":{"content":"hi"}}]}"#,
        }),
        &Attestation::Disabled,
    );
    assert_eq!(result.err(), Some(Rejection::Transfer(Check::Chat)));
}

#[test]
fn bounded_adapter_failures_never_become_readiness() {
    for (transfer, expected) in [
        (Transfer::TimedOut, Rejection::Deadline(Check::Runtime)),
        (
            Transfer::Oversized,
            Rejection::ResponseLimit(Check::Runtime),
        ),
    ] {
        let result = classify(Check::Runtime, transfer, &Attestation::Disabled);
        assert_eq!(result.err(), Some(expected));
    }
}

#[test]
fn oversized_response_is_rejected_before_decoding() {
    let mut body = br#"{"llama_ready":true}"#.to_vec();
    body.resize(RESPONSE_LIMIT + 1, b' ');
    let result = classify(Check::Runtime, response(&body), &Attestation::Disabled);
    assert_eq!(result.err(), Some(Rejection::ResponseLimit(Check::Runtime)));
}

#[test]
fn invalid_utf8_does_not_supply_runtime_readiness() {
    let result = classify(
        Check::Runtime,
        response(b"{\"llama_ready\":true}\xff"),
        &Attestation::Disabled,
    );
    assert!(matches!(result, Ok(Evidence::Pending)));
}

#[test]
fn attestation_uses_the_expected_status_at_each_existing_location() {
    let requirement = Attestation::Required {
        expected: "missing".into(),
    };
    for (check, good, bad) in [
        (
            Check::InspectAttestation,
            br#"{"status":"missing"}"#.as_slice(),
            br#"{"status":"valid"}"#.as_slice(),
        ),
        (
            Check::RuntimeAttestation,
            br#"{"release_attestation":{"status":"missing"}}"#,
            br#"{"release_attestation":{"status":"valid"}}"#,
        ),
        (
            Check::HeadlessAttestation,
            br#"{"release_attestation":{"status":"missing"}}"#,
            br#"{"status":"missing"}"#,
        ),
    ] {
        assert!(matches!(
            classify(check, response(good), &requirement),
            Ok(Evidence::Accepted)
        ));
        assert_eq!(
            classify(check, response(bad), &requirement).err(),
            Some(Rejection::Evidence(check))
        );
    }
}
