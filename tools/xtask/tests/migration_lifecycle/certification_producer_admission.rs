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
