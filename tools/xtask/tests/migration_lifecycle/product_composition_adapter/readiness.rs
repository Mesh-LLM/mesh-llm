//! Composer readiness admission, bounded caller observers, failure receipts.
use super::{Fixture, OLD_ARCHIVE, OLD_RECEIPT, digest, executable, snapshot};
use std::fs;

fn observers(fixture: &Fixture) {
    let host = r#"#!/bin/sh
if [ "$#" -eq 1 ] && [ "$1" = --version ]; then
  printf 'host-version\n' >> "$PRODUCT_EVENTS"
  printf 'mesh-llm 1.2.3\n'
  exit 0
fi
[ "$MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" = "$GITHUB_WORKSPACE/product/native-runtimes" ] || exit 95
[ "$1" = --log-format ] && [ "$2" = json ] || exit 95
if [ "$#" -eq 3 ] && [ "$3" = --version ]; then
  printf 'readiness host-version\n' >> "$PRODUCT_EVENTS"
  [ "$READINESS_FAILURE" != version ] || exit 73
  printf 'mesh-llm 1.2.3\n'
elif [ "$#" -eq 4 ] && [ "$3" = runtime ] && [ "$4" = list ]; then
  printf 'readiness runtime-list\n' >> "$PRODUCT_EVENTS"
  [ "$READINESS_FAILURE" != runtime-list ] || exit 73
  printf '{"fixture":"runtime-list observer"}\n'
else
  exit 95
fi
"#;
    executable(&fixture.root.join("inputs/host/mesh-llm"), host);
    fs::write(
        fixture.root.join("inputs/host/mesh-llm.sha256"),
        format!("{}  mesh-llm\n", digest(host.as_bytes())),
    )
    .unwrap();
    executable(
        &fixture.root.join("bin/uname"),
        r#"#!/bin/sh
[ "$#" -eq 1 ] && [ "$1" = -s ] || exit 95
printf '%s\n' "$READINESS_OS"
"#,
    );
    executable(
        &fixture.root.join("scripts/ci-prepare-native-runtime.sh"),
        r#"#!/bin/sh
[ "$#" -eq 4 ] && [ "$1" = "$GITHUB_WORKSPACE/product/sdk-runtime-fallback" ] || exit 95
[ "$2" = cpu ] && [ "$3" = --reuse-from-binary ] || exit 95
[ "$4" = "$GITHUB_WORKSPACE/product/mesh-llm" ] || exit 95
[ "$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK" = 0 ] || exit 95
printf 'readiness SDK reuse\n' >> "$PRODUCT_EVENTS"
[ "$READINESS_FAILURE" != sdk ] || exit 73
"#,
    );
    executable(
        &fixture.root.join("scripts/ci-client-readiness-smoke.sh"),
        r#"#!/bin/sh
[ "$#" -eq 2 ] && [ "$1" = "$GITHUB_WORKSPACE/product/mesh-llm" ] || exit 95
[ "$2" = "$GITHUB_WORKSPACE/product/native-runtimes" ] || exit 95
printf 'readiness client\n' >> "$PRODUCT_EVENTS"
[ "$READINESS_FAILURE" != client ] || exit 73
"#,
    );
}

#[test]
fn actual_composer_readiness_uses_composed_runtime_and_linux_sdk_reuse_before_archive() {
    for platform in ["Linux", "Darwin"] {
        let fixture = Fixture::new("1.2.3");
        observers(&fixture);
        let before = snapshot(&fixture.root.join("inputs"));
        let report = fixture.run_inputs(
            "product",
            "1.2.3",
            &[
                ("INPUT_READINESS_SMOKE", "true".into()),
                ("READINESS_OS", platform.into()),
                ("READINESS_FAILURE", String::new()),
            ],
        );
        fixture.accepted(&report, &before);
        let readiness: Vec<_> = fixture
            .events()
            .into_iter()
            .filter(|event| event.starts_with("readiness "))
            .collect();
        let mut expected = vec!["readiness host-version", "readiness runtime-list"];
        if platform == "Linux" {
            expected.push("readiness SDK reuse");
        }
        expected.push("readiness client");
        assert_eq!(readiness, expected);
        assert_ne!(
            fs::read(fixture.root.join("product.tar.gz")).unwrap(),
            OLD_ARCHIVE
        );
    }
}

#[test]
fn actual_composer_readiness_failures_and_invalid_mode_preserve_prior_publication() {
    for failure in ["version", "runtime-list", "sdk", "client", "invalid-mode"] {
        let fixture = Fixture::new("1.2.3");
        observers(&fixture);
        let before = snapshot(&fixture.root.join("inputs"));
        let report = fixture.run_inputs(
            "product",
            "1.2.3",
            &[
                (
                    "INPUT_READINESS_SMOKE",
                    if failure == "invalid-mode" {
                        "yes"
                    } else {
                        "true"
                    }
                    .into(),
                ),
                ("READINESS_OS", "Linux".into()),
                ("READINESS_FAILURE", failure.into()),
            ],
        );
        assert!(!report.process.success(), "{failure}: {report:?}");
        assert_eq!(snapshot(&fixture.root.join("inputs")), before);
        assert_eq!(
            fs::read(fixture.root.join("product.tar.gz")).unwrap(),
            OLD_ARCHIVE
        );
        assert_eq!(
            fs::read_to_string(fixture.root.join("github-output")).unwrap(),
            OLD_RECEIPT
        );
        assert!(
            !fixture
                .events()
                .iter()
                .any(|event| event.starts_with("forbidden"))
        );
    }
}
