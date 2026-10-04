//! Actual producer action handoff, finite build/attestation observers, real consumer.
use super::{Fixture, Value, digest, executable, invoke_arguments, snapshot};
use crate::workflow_yaml::{self, Node};
use std::{fs, path::Path};

fn body() -> String {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let document = workflow_yaml::parse(
        &fs::read_to_string(repository.join(".github/actions/prepare-host-input/action.yml"))
            .unwrap(),
    )
    .unwrap();
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("composite steps required");
    };
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].get("id").and_then(Node::text), Some("prepare"));
    assert_eq!(steps[0].get("shell").and_then(Node::text), Some("bash"));
    let body = steps[0].get("run").and_then(Node::text).unwrap().to_owned();
    assert!(!body.contains("${{"));
    body
}

fn observers(fixture: &Fixture, failure: &str) {
    fs::create_dir_all(fixture.root.join("target/debug")).unwrap();
    executable(
        &fixture.root.join("scripts/build-host.sh"),
        r#"#!/bin/sh
[ "$#" -eq 2 ] && [ "$1" = --profile ] && [ "$2" = release ] || exit 95
printf 'producer host-build\n' >> "$PRODUCT_EVENTS"
mkdir -p target/release || exit 95
cp inputs/host/mesh-llm target/release/mesh-llm || exit 95
"#,
    );
    executable(
        &fixture.root.join("target/debug/xtask"),
        r#"#!/bin/sh
[ "$1" = release-attestation ] || exit 95
case "$2" in
stamp)
  [ "$#" -eq 9 ] && [ "$3" = --binary ] && [ "$5" = --signing-key-file ] || exit 95
  [ "$6" = signing-key ] && [ "$7" = --require-source-commit ] || exit 95
  [ "$8" = --commit ] && [ "$9" = 1111111111111111111111111111111111111111 ] || exit 95
  [ "$4" = host-produced/mesh-llm ] || exit 95
  printf 'producer stamp\n' >> "$PRODUCT_EVENTS"
  [ "$PRODUCER_FAILURE" != stamp ] || exit 73
  ;;
inspect)
  [ "$#" -eq 7 ] && [ "$3" = --binary ] && [ "$5" = --public-key-file ] || exit 95
  [ "$6" = public-key ] && [ "$7" = --json ] || exit 95
  case "$4" in host-produced/mesh-llm|*/host-produced/mesh-llm) ;; *) exit 95 ;; esac
  printf 'producer inspect\n' >> "$PRODUCT_EVENTS"
  [ "$PRODUCER_FAILURE" != inspect ] || exit 73
  printf '{"fixture":"prebuilt verifier invocation only"}\n'
  ;;
*) exit 95 ;;
esac
"#,
    );
    executable(
        &fixture.root.join("bin/cargo"),
        r#"#!/bin/sh
case "$1" in
build)
  [ "$#" -eq 6 ] && [ "$2" = -q ] && [ "$3" = -p ] || exit 95
  [ "$4" = xtask ] && [ "$5" = --bin ] && [ "$6" = xtask ] || exit 95
  printf 'producer verifier-build\n' >> "$PRODUCT_EVENTS"
  [ "$PRODUCER_FAILURE" != build ] || exit 73
  ;;
metadata)
  [ "$#" -eq 4 ] && [ "$2" = --no-deps ] || exit 95
  [ "$3" = --format-version ] && [ "$4" = 1 ] || exit 95
  printf '{"target_directory":"%s/target"}\n' "$GITHUB_WORKSPACE"
  ;;
xtool)
  shift
  if [ "$1" = repository ]; then
    [ "$#" -eq 2 ] && [ "$2" = cargo-target-directory ] || exit 95
    exec "$PRODUCT_REAL_XTASK" "$@"
  fi
  [ "$#" -eq 7 ] && [ "$1" = native ] && [ "$2" = verify-host-dependencies ] || exit 95
  [ "$3" = host-produced/mesh-llm ] && [ "$4" = --report ] || exit 95
  [ "$5" = host-produced/host-imports.json ] && [ "$6" = --max-glibc ] && [ "$7" = declared ] || exit 95
  printf 'producer imports\n' >> "$PRODUCT_EVENTS"
  [ "$PRODUCER_FAILURE" != imports ] || exit 73
  cp inputs/host/host-imports.json "$5" || exit 95
  ;;
*) exit 95 ;;
esac
"#,
    );
    fs::write(fixture.root.join("signing-key"), b"inert observer key").unwrap();
    fs::write(fixture.root.join("public-key"), b"inert observer key").unwrap();
    fs::write(fixture.root.join("failure-case"), failure).unwrap();
}

fn produce(fixture: &Fixture, failure: &str, public_key: &str) -> crate::process::RawProcessReport {
    let environment = fixture.environment(
        "host-produced",
        "1.2.3",
        &[
            ("INPUT_PROFILE", "release".into()),
            ("INPUT_SKIP_UI", "false".into()),
            ("INPUT_BUILD_VERSION", String::new()),
            ("INPUT_ATTESTATION_SIGNING_KEY_FILE", "signing-key".into()),
            ("INPUT_ATTESTATION_PUBLIC_KEY_FILE", public_key.into()),
            (
                "INPUT_COMMIT",
                "1111111111111111111111111111111111111111".into(),
            ),
            ("PRODUCER_FAILURE", failure.into()),
        ],
    );
    invoke_arguments(
        &fixture.root,
        vec![Value::Public("-c".into()), Value::Public(body().into())],
        environment,
    )
}

#[test]
fn actual_host_attestation_producer_exports_checksum_bound_verifier_consumed_without_building() {
    let fixture = Fixture::new("1.2.3");
    observers(&fixture, "");
    let before = snapshot(&fixture.root.join("inputs"));
    let report = produce(&fixture, "", "public-key");
    assert!(report.process.success(), "{report:?}");
    assert_eq!(snapshot(&fixture.root.join("inputs")), before);
    let verifier = fs::read(
        fixture
            .root
            .join("host-produced/release-attestation-verifier"),
    )
    .unwrap();
    assert_eq!(
        fs::read_to_string(
            fixture
                .root
                .join("host-produced/release-attestation-verifier.sha256")
        )
        .unwrap(),
        format!("{}  release-attestation-verifier\n", digest(&verifier))
    );
    assert!(
        fs::read_to_string(fixture.root.join("github-output"))
            .unwrap()
            .lines()
            .any(|line| line
                == format!(
                    "attestation_verifier_path={}/host-produced/release-attestation-verifier",
                    fixture.root.display()
                ))
    );
    assert_eq!(
        fixture.events(),
        [
            "producer host-build",
            "producer verifier-build",
            "producer stamp",
            "producer inspect",
            "producer imports"
        ]
    );
    fs::write(fixture.root.join("events"), []).unwrap();
    // Composer must consume the copied verifier, never rebuild it.
    executable(
        &fixture.root.join("bin/cargo"),
        "#!/bin/sh\nprintf 'forbidden cargo\\n' >> \"$PRODUCT_EVENTS\"\nexit 98\n",
    );
    let report = fixture.run_inputs(
        "product",
        "1.2.3",
        &[
            ("INPUT_HOST_INPUT_DIR", "host-produced".into()),
            ("INPUT_ATTESTATION_PUBLIC_KEY_FILE", "public-key".into()),
            ("PRODUCER_FAILURE", String::new()),
        ],
    );
    fixture.accepted(&report, &before);
    assert!(
        fixture
            .events()
            .iter()
            .any(|event| event == "producer inspect")
    );
}

#[test]
fn actual_host_attestation_producer_failures_never_publish_success_receipts() {
    for failure in ["build", "stamp", "inspect", "imports", "missing-key"] {
        let fixture = Fixture::new("1.2.3");
        observers(&fixture, failure);
        let before = snapshot(&fixture.root.join("inputs"));
        let receipt = fs::read(fixture.root.join("github-output")).unwrap();
        let report = produce(
            &fixture,
            failure,
            if failure == "missing-key" {
                ""
            } else {
                "public-key"
            },
        );
        assert!(!report.process.success(), "{failure}: {report:?}");
        assert_eq!(
            fs::read(fixture.root.join("github-output")).unwrap(),
            receipt
        );
        assert_eq!(snapshot(&fixture.root.join("inputs")), before);
        assert!(
            !fixture
                .events()
                .iter()
                .any(|event| event.starts_with("forbidden"))
        );
    }
}
