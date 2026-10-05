//! Execute named production repair functions with finite owned component observers.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::PathBuf, time::Duration};

fn source() -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap()
}
fn declaration(source: &str, name: &str) -> String {
    // Deliberately selects only the reviewed top-level production declarations.
    // No shell emulation: Bash parses and executes the selected original bytes.
    let marker = format!("\n{name}() {{\n");
    assert_eq!(source.matches(&marker).count(), 1);
    let tail = source.split_once(&marker).unwrap().1;
    let body = tail.split_once("\n}\n").unwrap().0;
    format!("{name}() {{\n{body}\n}}\n")
}
struct Observed {
    status: i32,
    stdout: String,
    stderr: String,
}
struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }
    fn run(&self, declarations: &str, setup: &str, invocation: &str) -> Observed {
        let script = format!("set -euo pipefail\n{declarations}\n{setup}\n{invocation}\n");
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.0.path().into(),
                arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
                environment: BTreeMap::from([
                    ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
                    ("HOME".into(), Value::Public(self.0.path().into())),
                    (
                        "GITHUB_OUTPUT".into(),
                        Value::Public(self.0.path().join("outputs").into()),
                    ),
                ]),
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 32768,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(32768),
                stderr: NonZeroUsize::new(32768),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited, "{report:?}");
        assert!(report.process.failure.is_none() && report.process.cleanup.failure.is_none());
        assert!(
            report.process.cleanup.complete
                && !report.process.cleanup.forced
                && !report.process.cleanup.graceful_signal_failed
        );
        for stream in [&report.process.stdout, &report.process.stderr] {
            assert!(!stream.truncated && stream.suppressed_lines == 0);
        }
        let stdout = report.stdout.unwrap();
        let stderr = report.stderr.unwrap();
        assert_eq!(
            u64::try_from(stdout.as_bytes().len()).unwrap(),
            report.process.stdout.bytes_seen
        );
        assert_eq!(
            u64::try_from(stderr.as_bytes().len()).unwrap(),
            report.process.stderr.bytes_seen
        );
        Observed {
            status: report.process.status.unwrap().code().unwrap(),
            stdout: String::from_utf8(stdout.as_bytes().to_vec()).unwrap(),
            stderr: String::from_utf8(stderr.as_bytes().to_vec()).unwrap(),
        }
    }
    fn finish(self) {
        self.0.close().expect("owned repair fixture cleanup failed");
    }
}
fn loop_declarations() -> String {
    let source = source();
    [
        "remaining_verification_seconds",
        "remaining_repair_seconds",
        "record_failure_class",
        "repair_candidate_until_green",
    ]
    .iter()
    .map(|name| declaration(&source, name))
    .collect()
}
const CLOCK_SETUP: &str = r#"
now=1000
AGENT_TIMEOUT_SECONDS=100
VERIFICATION_TIMEOUT_SECONDS=200
turns=0
date() { printf '%s\n' "$now"; }
agent_prompt() { printf 'initial'; }
agent_feedback_prompt() { printf 'feedback'; }
assert_agent_control_unchanged() { :; }
validate_agent_manifest_changes() { :; }
agent_session_step() { turns=$((turns+1)); printf 'agent %s prompt=%s\n' "$turns" "$1"; now=$((now+10)); }
"#;
#[test]
fn returned_second_candidate_keeps_full_verification_window_after_repair_admission_closes() {
    for second_status in [0, 1] {
        let fixture = Fixture::new();
        let setup = format!(
            r#"{CLOCK_SETUP}
run_candidate_gates() {{
  local budget
  budget="$(remaining_verification_seconds)" || return 124
  printf 'gate budget=%s mode=%s\n' "$budget" "$1"
  if (( turns == 1 )); then now=$((now+80)); return 1; fi
  now=$((now+150))
  return {second_status}
}}
"#
        );
        let observed = fixture.run(&loop_declarations(), &setup, "repair_candidate_until_green");
        assert_eq!(observed.status, if second_status == 0 { 0 } else { 124 });
        assert_eq!(
            observed
                .stdout
                .matches("gate budget=200 mode=refresh")
                .count(),
            2
        );
        assert!(
            observed.stdout.contains("agent 1 prompt=initial")
                && observed.stdout.contains("agent 2 prompt=feedback")
        );
        assert!(!observed.stdout.contains("agent 3"));
        assert_eq!(
            observed
                .stderr
                .contains("class=candidate stage=trusted-gates"),
            second_status != 0
        );
        fixture.finish();
    }
}
#[test]
fn agent_failure_preserves_exit_and_infrastructure_receipt_without_candidate_gates() {
    let fixture = Fixture::new();
    let setup = format!(
        r#"{CLOCK_SETUP}
agent_session_step() {{ printf 'agent rejected\n' >&2; return 23; }}
run_candidate_gates() {{ printf 'FORBIDDEN_GATE\n'; return 0; }}
"#
    );
    let observed = fixture.run(&loop_declarations(), &setup, "repair_candidate_until_green");
    assert_eq!(observed.status, 23);
    assert!(!observed.stdout.contains("FORBIDDEN_GATE"));
    assert!(
        observed
            .stderr
            .contains("class=infrastructure stage=agent-runtime")
    );
    assert_eq!(
        fs::read_to_string(fixture.0.path().join("outputs")).unwrap(),
        "failure_class=infrastructure\nfailure_stage=agent-runtime\n"
    );
    fixture.finish();
}
#[test]
fn exhausted_repair_window_does_not_admit_another_coding_turn() {
    let fixture = Fixture::new();
    let setup = format!(
        r#"{CLOCK_SETUP}
run_candidate_gates() {{ printf 'gate once\n'; now=$((now+100)); return 1; }}
"#
    );
    let observed = fixture.run(&loop_declarations(), &setup, "repair_candidate_until_green");
    assert_eq!(observed.status, 124);
    assert!(observed.stdout.contains("agent 1 prompt=initial"));
    assert!(!observed.stdout.contains("agent 2"));
    assert!(
        observed
            .stderr
            .contains("class=candidate stage=trusted-gates")
    );
    fixture.finish();
}
const GATE_SETUP: &str = r#"
HARNESS_MODE=repair
PREPARE_LOG=prepare.log
MANIFEST_POLICY_LOG=policy.log
BUILD_LOG=build.log
CERTIFY_LOG=certify.log
observe() { printf '%s\n' "$1"; [[ "$1" != "$FAIL_AT" ]]; }
run_prepare() { observe prepare; }
write_split_certification_roster() { observe refresh; }
validate_agent_manifest_changes() { observe policy; }
run_full_build() { observe build; }
run_certification() { observe certification; }
"#;
#[test]
fn candidate_gates_preserve_order_refresh_boundary_and_short_circuit_every_failure() {
    let source = declaration(&source(), "run_candidate_gates");
    for mode in ["verify", "refresh"] {
        let expected: Vec<_> = if mode == "refresh" {
            vec!["prepare", "refresh", "policy", "build", "certification"]
        } else {
            vec!["prepare", "policy", "build", "certification"]
        };
        for failure in std::iter::once("none").chain(expected.iter().copied()) {
            let fixture = Fixture::new();
            let setup = format!("{GATE_SETUP}\nFAIL_AT={failure}");
            let observed = fixture.run(&source, &setup, &format!("run_candidate_gates {mode}"));
            assert_eq!(observed.status, if failure == "none" { 0 } else { 1 });
            let length = expected
                .iter()
                .position(|stage| *stage == failure)
                .map_or(expected.len(), |index| index + 1);
            assert_eq!(
                observed.stdout.lines().collect::<Vec<_>>(),
                expected[..length]
            );
            fixture.finish();
        }
    }
}
#[test]
fn unknown_gate_mode_refuses_before_logs_or_gate_calls() {
    let fixture = Fixture::new();
    let observed = fixture.run(
        &declaration(&source(), "run_candidate_gates"),
        &format!("{GATE_SETUP}\nFAIL_AT=none"),
        "run_candidate_gates unexpected",
    );
    assert_eq!(observed.status, 2);
    assert!(observed.stdout.is_empty());
    assert!(!fixture.0.path().join("prepare.log").exists());
    assert!(
        observed
            .stderr
            .contains("invalid candidate-gate roster mode")
    );
    fixture.finish();
}
#[test]
fn pinned_build_checks_actual_pin_without_invoking_updater_or_rewriting_bytes() {
    for pin_matches in [true, false] {
        let fixture = Fixture::new();
        fs::create_dir(fixture.0.path().join("scripts")).unwrap();
        fs::write(
            fixture.0.path().join("scripts/update-llama-pin.sh"),
            "#!/bin/sh\nprintf FORBIDDEN_UPDATER > updater-called\nexit 99\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(
            fixture.0.path().join("scripts/update-llama-pin.sh"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        let pin = if pin_matches {
            "a".repeat(40)
        } else {
            "b".repeat(40)
        };
        let bytes = format!(" {pin}\n");
        fs::write(fixture.0.path().join("pin"), &bytes).unwrap();
        let declarations = declaration(&source(), "write_repair_pin")
            + &declaration(&source(), "verify_repair_pin");
        let observed = fixture.run(
            &declarations,
            &format!(
                "HARNESS_MODE=pinned-build\nUPSTREAM_SHA={}\nPIN_FILE=pin",
                "a".repeat(40)
            ),
            "write_repair_pin",
        );
        assert_eq!(observed.status, if pin_matches { 0 } else { 1 });
        assert!(!fixture.0.path().join("updater-called").exists());
        assert_eq!(
            fs::read_to_string(fixture.0.path().join("pin")).unwrap(),
            bytes
        );
        fixture.finish();
    }
}
