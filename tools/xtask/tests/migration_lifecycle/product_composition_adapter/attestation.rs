//! Actual composer verifies a prebuilt attestation observer before mutation.
//! This binds checksum, invocation and refusal behavior, not signature validity.
use super::{Fixture, digest, executable, snapshot};
use std::fs;

fn verifier(fixture: &Fixture) -> std::path::PathBuf {
    let path = fixture
        .root
        .join("inputs/host/release-attestation-verifier");
    executable(
        &path,
        r#"#!/bin/sh
[ "$#" -eq 7 ] && [ "$1" = release-attestation ] && [ "$2" = inspect ] &&
  [ "$3" = --binary ] && [ "$4" = "$PWD/inputs/host/mesh-llm" ] &&
  [ "$5" = --public-key-file ] && [ "$6" = "$PWD/public-key.json" ] &&
  [ "$7" = --json ] || exit 98
printf 'attestation-inspect\n' >> "$PRODUCT_EVENTS"
[ "$PRODUCT_ATTESTATION_FAILURE" = false ] || exit 73
printf '{"fixture_attestation":"accepted"}\n'
"#,
    );
    fs::write(
        path.with_file_name("release-attestation-verifier.sha256"),
        format!(
            "{}  release-attestation-verifier\n",
            digest(&fs::read(&path).unwrap())
        ),
    )
    .unwrap();
    fs::write(
        fixture.root.join("public-key.json"),
        b"finite observer key input",
    )
    .unwrap();
    path
}
fn inputs(fixture: &Fixture, fail: bool) -> [(&'static str, String); 2] {
    [
        (
            "INPUT_ATTESTATION_PUBLIC_KEY_FILE",
            fixture.root.join("public-key.json").display().to_string(),
        ),
        ("PRODUCT_ATTESTATION_FAILURE", fail.to_string()),
    ]
}
#[test]
fn product_composer_prebuilt_attestation_invocation_precedes_composition_and_refuses_failure() {
    for fail in [false, true] {
        let fixture = Fixture::new("1.2.3");
        verifier(&fixture);
        let before = snapshot(&fixture.root.join("inputs"));
        let report = fixture.run_inputs("product", "", &inputs(&fixture, fail));
        if fail {
            fixture.refusal(&report, &before);
            assert_eq!(
                fs::read(fixture.root.join("product/previous-staging")).unwrap(),
                b"staging keep"
            );
        } else {
            fixture.accepted(&report, &before);
            let events = fixture.events();
            let verified = events
                .iter()
                .position(|event| event == "attestation-inspect")
                .unwrap();
            let compose = events
                .iter()
                .position(|event| event == "product compose")
                .unwrap();
            assert!(verified < compose);
            assert!(
                events[..verified]
                    .iter()
                    .filter(|event| *event == "artifact verify-checksum")
                    .count()
                    >= 2
            );
        }
    }
}
#[test]
fn product_composer_verifier_sidecar_must_be_canonical_before_execution_or_staging() {
    for multiline in [false, true] {
        let fixture = Fixture::new("1.2.3");
        let path = verifier(&fixture);
        let checksum = digest(&fs::read(&path).unwrap());
        let contents = if multiline {
            format!(
                "{checksum}  release-attestation-verifier\n{checksum}  release-attestation-verifier\n"
            )
        } else {
            format!("{checksum}  unrelated-verifier\n")
        };
        fs::write(
            path.with_file_name("release-attestation-verifier.sha256"),
            contents,
        )
        .unwrap();
        let before = snapshot(&fixture.root.join("inputs"));
        let report = fixture.run_inputs("product", "", &inputs(&fixture, false));
        fixture.refusal(&report, &before);
        assert!(
            !fixture
                .events()
                .iter()
                .any(|event| event == "attestation-inspect")
        );
        assert_eq!(
            fs::read(fixture.root.join("product/previous-staging")).unwrap(),
            b"staging keep"
        );
    }
}
#[test]
fn product_composer_refuses_missing_key_verifier_or_sidecar_and_orphaned_verifier_configuration() {
    for missing in ["key", "verifier", "sidecar", "orphaned"] {
        let fixture = Fixture::new("1.2.3");
        let path = verifier(&fixture);
        let mut environment = inputs(&fixture, false).to_vec();
        match missing {
            "key" => fs::remove_file(fixture.root.join("public-key.json")).unwrap(),
            "verifier" => fs::remove_file(&path).unwrap(),
            "sidecar" => {
                fs::remove_file(path.with_file_name("release-attestation-verifier.sha256")).unwrap()
            }
            "orphaned" => {
                environment[0].1.clear();
                environment.push(("INPUT_ATTESTATION_VERIFIER", path.display().to_string()));
            }
            _ => unreachable!(),
        }
        let before = snapshot(&fixture.root.join("inputs"));
        let report = fixture.run_inputs("product", "", &environment);
        fixture.refusal(&report, &before);
        assert!(
            !fixture
                .events()
                .iter()
                .any(|event| event == "attestation-inspect")
        );
        assert_eq!(
            fs::read(fixture.root.join("product/previous-staging")).unwrap(),
            b"staging keep"
        );
    }
}
