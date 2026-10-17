//! Execute the generator check's native-test phase with finite compiler boundaries.
use super::{Fixture, fs, root};

fn script() -> String {
    let source =
        fs::read_to_string(root().join("skippy/scripts/check-skippy-generated-family-patch.sh"))
            .unwrap();
    let scope = source
        .split_once("# Native validation scope begins.\n")
        .unwrap()
        .1
        .split_once("# Native validation scope ends.")
        .unwrap()
        .0;
    let phase = source.split_once("if diff -qr ").unwrap().1;
    format!("set -euo pipefail\n{scope}\nif diff -qr {phase}")
}

fn fixture(failure: &str) -> Fixture {
    let fixture = Fixture::new();
    for directory in ["checked", "generated", "artifacts", "source"] {
        fs::create_dir(fixture.path().join(directory)).unwrap();
    }
    if failure == "patch" {
        fs::write(fixture.path().join("generated/drift.patch"), b"drift").unwrap();
    }
    fixture.executable("bin/cargo", CARGO);
    fixture.executable("bin/cmake", CMAKE);
    fixture.executable(
        "bin/ctest",
        "printf 'ctest\\n' >> \"$FIXTURE_ROOT/order\"\nprintf '%s\\n' \"$@\" > \"$FIXTURE_ROOT/ctest.args\"\n[[ $FAILURE != test ]]",
    );
    fixture
}

fn run(fixture: &Fixture, full: bool, failure: &str) -> super::Receipt {
    let mut command = fixture.command();
    command.args(["-c", &script(), "checker"]);
    if full {
        command.arg("--full-native-suite");
    }
    for (name, relative) in [
        ("CHECKED_PATCH_DIR", "checked"),
        ("GENERATED_PATCH_DIR", "generated"),
        ("ARTIFACT_ROOT", "artifacts"),
        ("SOURCE_ROOT", "source"),
        ("FIRST_REPORT", "report.json"),
        ("SECOND_REPORT", "report-second.json"),
    ] {
        command.env(name, fixture.path().join(relative));
    }
    command
        .env("FAILURE", failure)
        .env("CMAKE_BUILD_TYPE", "Release");
    fixture.run(command)
}

#[test]
fn generated_tree_runs_one_native_suite_after_patch_and_idempotence_admission() {
    for full in [false, true] {
        let fixture = fixture("");
        let receipt = run(&fixture, full, "");
        assert!(receipt.status.success(), "{:?}", receipt.stderr);
        assert_eq!(
            fs::read_to_string(fixture.path().join("order")).unwrap(),
            "validate\nidempotence\nconfigure\nbuild\nctest\nvalidate\n"
        );
        let tests = fs::read_to_string(fixture.path().join("ctest.args")).unwrap();
        assert!(tests.contains("--timeout\n900\n"));
        assert!(tests.contains("--build-config\nRelease\n"));
        assert_eq!(tests.contains("-R\n^skippy_\n"), !full);
        let build = fs::read_to_string(fixture.path().join("build.args")).unwrap();
        assert_eq!(build.contains("--target\n"), !full);
        let configure = fs::read_to_string(fixture.path().join("configure.args")).unwrap();
        assert!(configure.contains("-DLLAMA_STAGE_BUILD_TESTS=ON\n"));
        assert_eq!(configure.contains("-DLLAMA_BUILD_TESTS=ON\n"), full);
        assert_eq!(configure.contains("-DLLAMA_BUILD_SERVER=ON\n"), full);
    }
}

#[test]
fn native_owner_rejects_each_failed_gate_without_duplicate_or_premature_tests() {
    for failure in [
        "patch",
        "idempotence",
        "configure",
        "build",
        "test",
        "report",
    ] {
        let fixture = fixture(failure);
        let receipt = run(&fixture, true, failure);
        assert!(!receipt.status.success(), "failure={failure}");
        let order = fs::read_to_string(fixture.path().join("order")).unwrap();
        let expected_tests = usize::from(matches!(failure, "test" | "report"));
        assert_eq!(
            order.lines().filter(|line| *line == "ctest").count(),
            expected_tests
        );
        if matches!(failure, "patch" | "idempotence") {
            assert!(!order.contains("configure"));
        }
        assert!(
            !String::from_utf8(receipt.stdout)
                .unwrap()
                .contains("is deterministic and current")
        );
    }
}

const CMAKE: &str = r#"
if [[ $1 == --build ]]; then phase=build; else phase=configure; fi
printf '%s\n' "$phase" >> "$FIXTURE_ROOT/order"
printf '%s\n' "$@" > "$FIXTURE_ROOT/$phase.args"
[[ $FAILURE != "$phase" ]]
"#;

const CARGO: &str = r#"
case "$*" in
  *'contracts select-shards'*) exit 0 ;;
  *'--mode idempotence'*)
    printf 'idempotence\n' >> "$FIXTURE_ROOT/order"
    [[ $FAILURE != idempotence ]]
    ;;
  *)
    printf 'validate\n' >> "$FIXTURE_ROOT/order"
    case "$*" in
      *'--patch-check fail'*|*'--compile-result fail'*|*'--graph-verify-result fail'*) exit 23 ;;
      *'--compile-result pass'*) [[ $FAILURE != report ]] ;;
    esac
    ;;
esac
"#;
