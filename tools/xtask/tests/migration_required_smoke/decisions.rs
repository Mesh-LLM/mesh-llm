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
fn source_derived_cases_distinguish_legacy_decisions() {
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
fn models_preserve_python_stringification_instead_of_claiming_model_identity() {
    for (body, expected) in [
        (br#"{"data":[{"id":null}]}"#.as_slice(), "null"),
        (br#"{"data":[{"id":7}]}"#, "7"),
        (br#"{"data":[{"id":false}]}"#, "false"),
        (br#"{"data":[{"id":"first\n\n"}]}"#, "first"),
    ] {
        let result = classify(Check::Models, response(body), &Attestation::Disabled);
        assert!(matches!(result, Ok(Evidence::Model(model)) if model == expected));
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
