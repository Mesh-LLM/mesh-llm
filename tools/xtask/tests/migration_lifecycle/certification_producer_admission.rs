//! Admission of producer-owned environment by the actual certification caller.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn invoke(root: &Path) -> process::RawProcessReport {
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![Value::Public("caller.sh".into())],
            cwd: root.into(),
            environment: BTreeMap::from([(
                "PATH".into(),
                Value::Public(std::env::var_os("PATH").unwrap()),
            )]),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        result.process.cleanup.complete && result.process.failure.is_none(),
        "{:?}",
        result.process
    );
    result
}

fn scripts(root: &Path, output: &str, producer_status: i32, child_status: i32) {
    fs::create_dir(root.join("scripts")).unwrap();
    fs::write(root.join("producer-output"), output).unwrap();
    fs::write(
        root.join("scripts/skippy-workload-oracles-build.sh"),
        format!(
            r#"#!/bin/bash
set -euo pipefail
[[ "$1" == --print-env && "$2" == "$PWD/native-workloads" ]] || exit 97
cat producer-output
exit {producer_status}
"#
        ),
    )
    .unwrap();
    let child = root.join("scripts/skippy-family-battery.sh");
    fs::write(
        &child,
        format!(
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$PRODUCER" "$WORKLOAD_PATH" "$FAMILY_BATTERY_RUN_ID" > consumed-environment
exit {child_status}
"#
        ),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(child, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn copied_real_certification_refuses_failed_or_empty_producer_and_preserves_admitted_environment_and_status()
 {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let function = source
        .split("run_certification() {\n")
        .nth(1)
        .unwrap()
        .split("\nrun_candidate_gates() {\n")
        .next()
        .unwrap();
    for (output, producer_status, child_status, accepted) in [
        ("PARTIAL=1\n", 9, 0, false),
        ("", 0, 0, false),
        (
            "PRODUCER=fixture-ready\nWORKLOAD_PATH=owned path with spaces\n",
            0,
            0,
            true,
        ),
        (
            "PRODUCER=fixture-ready\nWORKLOAD_PATH=owned path with spaces\n",
            0,
            37,
            true,
        ),
    ] {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();
        scripts(root, output, producer_status, child_status);
        let caller = format!(
            r#"set -euo pipefail
HARNESS_MODE=verify
LLAMA_STAGE_BUILD_DIR="$PWD/native"
HF_CACHE="$PWD/unused-cache"
PLAN_PATH="$PWD/plan.json"
FAMILY_BATTERY_RUN_ID=finite-certification
CERTIFY_LOG="$PWD/certify.log"
# Recording seams for independently qualified parity and family-plan policy.
verification_source_inspection() {{ printf 'parity\n' >> gates; }}
repair_family_plan() {{ printf 'family-plan\n' >> gates; }}
run_verification_logged() {{
  local label="$1"
  shift 2
  printf 'certification\n' >> gates
  "$@"
}}
run_certification() {{
{function}
run_certification
"#
        );
        fs::write(root.join("caller.sh"), caller).unwrap();
        let result = invoke(root);
        assert_eq!(
            result.process.status.unwrap().code(),
            Some(if accepted { child_status } else { 1 })
        );
        if accepted {
            assert_eq!(
                fs::read_to_string(root.join("gates")).unwrap(),
                "parity\nfamily-plan\ncertification\n"
            );
            assert_eq!(
                fs::read_to_string(root.join("consumed-environment")).unwrap(),
                "fixture-ready\nowned path with spaces\nfinite-certification\n"
            );
        } else {
            assert!(!root.join("gates").exists());
            assert!(!root.join("consumed-environment").exists());
            assert!(!root.join("certify.log").exists());
        }
    }
}

fn early_producer_fixture(root: &Path, mode: &str) -> String {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::PermissionsExt;
    if mode == "native-refusal" {
        let closure = root.join("native-workloads");
        fs::create_dir_all(closure.join("cargo/debug")).unwrap();
        fs::create_dir(closure.join("native")).unwrap();
        fs::write(closure.join("cargo/debug/skippy"), b"finite candidate").unwrap();
        fs::write(closure.join("producer.json"), serde_json::json!({
            "schema_version":0, "source":{"head":"a".repeat(40), "worktree_sha256":"b".repeat(64)}, "files":{}
        }).to_string()).unwrap();
    }
    scripts(
        root,
        "PRODUCER=fixture-ready\nWORKLOAD_PATH=owned path\n",
        0,
        0,
    );
    let controller = root.join("controller");
    fs::write(&controller, format!(
        r#"#!/bin/bash
set -euo pipefail
printf '%s\0' "$@" > "$PWD/producer-arguments"
[[ "$#" == 8 && "$1" == automation && "$2" == canary-receipts && "$3" == workload-manifest && "$4" == verify ]] || exit 97
[[ "$5" == "$PWD" && "$6" == "$PWD/native-workloads/cargo/debug/skippy" && "$7" == "$PWD/native-workloads/native" && "$8" == "$PWD/native-workloads/producer.json" ]] || exit 98
case {mode} in
  refusal) exit 37 ;;
  changed-after) printf '\n# replaced after verification\n' >> "$0" ;;
  native-refusal) exec '{native}' "$@" ;;
esac
"#, native = env!("CARGO_BIN_EXE_xtask"))).unwrap();
    fs::set_permissions(&controller, fs::Permissions::from_mode(0o755)).unwrap();
    let test = root.join("mm-test");
    fs::write(&test, "#!/bin/bash\nexit 0\n").unwrap();
    fs::set_permissions(&test, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(
        root.join("mm-build.jsonl"),
        serde_json::json!({
            "reason":"compiler-artifact", "profile":{"test":true},
            "target":{"name":"skippy_serving"}, "executable":test
        })
        .to_string(),
    )
    .unwrap();
    hex::encode(Sha256::digest(fs::read(controller).unwrap()))
}

#[test]
fn early_metal_caller_binds_native_producer_and_refuses_before_downstream_consumption() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let early = source
        .split("run_early_metal_certification() {\n")
        .nth(1)
        .unwrap()
        .split("\nrun_candidate_gates() {\n")
        .next()
        .unwrap();
    let guard = source
        .split("  repair_workload_controller_unchanged() {")
        .nth(1)
        .unwrap()
        .split("\n# Legacy workload automation selection ends.")
        .next()
        .unwrap();
    for mode in [
        "success",
        "refusal",
        "changed-before",
        "changed-after",
        "native-refusal",
    ] {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();
        let digest = early_producer_fixture(root, mode);
        let before = if mode == "changed-before" {
            "printf '\n# changed before\n' >> \"$repair_workload_controller\""
        } else {
            ":"
        };
        fs::write(root.join("caller.sh"), format!(
            r#"set -euo pipefail
ROOT="$PWD"
LLAMA_STAGE_BUILD_DIR="$ROOT/native"
STATE_DIR="$ROOT"
CERTIFY_LOG="$ROOT/certify.log"
FAMILY_BATTERY_RUN_ID=finite-certification
repair_workload_controller="$ROOT/controller"
repair_workload_controller_sha='{digest}'
repair_workload_automation=("$repair_workload_controller")
repair_workload_controller_unchanged() {{{guard}
verification_candidate_unchanged() {{ printf 'custody\n' >> custody; }}
run_verification_logged() {{ local label="$1" log="$2"; shift 2; printf '%s\n' "$label" >> "$log"; "$@"; }}
run_early_metal_certification() {{
{early}
{before}
run_early_metal_certification
"#)).unwrap();
        let report = invoke(root);
        if mode == "native-refusal" {
            assert!(
                String::from_utf8_lossy(
                    report
                        .stderr
                        .as_ref()
                        .expect("captured refusal stderr")
                        .as_bytes()
                )
                .contains("unsupported workload producer schema"),
                "{report:?}"
            );
        }
        assert_eq!(
            report.process.status.unwrap().code(),
            Some(i32::from(mode != "success")),
            "mode={mode}: {report:?}"
        );
        if mode == "success" {
            assert_eq!(
                fs::read_to_string(root.join("custody")).unwrap(),
                "custody\ncustody\n"
            );
            assert_eq!(
                fs::read_to_string(root.join("consumed-environment")).unwrap(),
                "fixture-ready\nowned path\nfinite-certification-early\n"
            );
        } else {
            assert!(
                !root.join("consumed-environment").exists(),
                "mode={mode}: {report:?}"
            );
            if mode == "changed-before" {
                assert!(!root.join("producer-arguments").exists());
            } else {
                assert!(root.join("producer-arguments").exists());
            }
        }
    }
}
