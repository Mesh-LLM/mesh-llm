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

#[test]
fn actual_pin_updater_writes_only_admitted_explicit_or_prepared_sha_and_preserves_invalid_bytes() {
    let fixture = Fixture::new();
    let scripts = fixture.0.path().join("scripts");
    fs::create_dir(&scripts).unwrap();
    fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/update-llama-pin.sh"),
        scripts.join("update-llama-pin.sh"),
    )
    .unwrap();
    fs::create_dir(fixture.0.path().join("prepared")).unwrap();
    let setup = "export LLAMA_PIN_FILE=\"$PWD/pin\" LLAMA_WORKDIR=\"$PWD/prepared\"";
    let target = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let accepted = fixture.run(
        "",
        setup,
        &format!("/bin/bash ./scripts/update-llama-pin.sh {target}"),
    );
    assert_eq!(accepted.status, 0, "{}", accepted.stderr);
    assert_eq!(
        fs::read(fixture.0.path().join("pin")).unwrap(),
        format!("{target}\n").as_bytes()
    );
    let rejected = fixture.run(
        "",
        setup,
        "/bin/bash ./scripts/update-llama-pin.sh not-a-sha",
    );
    assert_eq!(rejected.status, 1);
    assert!(rejected.stderr.contains("refusing to write a non-40-hex"));
    assert_eq!(
        fs::read(fixture.0.path().join("pin")).unwrap(),
        format!("{target}\n").as_bytes()
    );
    let prepared = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    fs::write(
        fixture.0.path().join("prepared/.mesh-llm-upstream-sha"),
        format!(" \n{prepared}\n "),
    )
    .unwrap();
    let marker = fixture.run("", setup, "/bin/bash ./scripts/update-llama-pin.sh");
    assert_eq!(marker.status, 0, "{}", marker.stderr);
    assert_eq!(
        fs::read(fixture.0.path().join("pin")).unwrap(),
        format!("{prepared}\n").as_bytes()
    );
    fixture.finish();
}
#[test]
fn actual_family_core_and_state_callers_allocate_os_ports_and_retry_only_address_conflicts() {
    let fixture = Fixture::new();
    std::os::unix::fs::symlink(
        env!("CARGO_BIN_EXE_xtask"),
        fixture.0.path().join("controller"),
    )
    .unwrap();
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/family-certify.sh"),
    )
    .unwrap();
    assert!(source.lines().any(|line| line == "PORT_START_ATTEMPTS=3"));
    let declarations = [
        "run_logged_core_parity",
        "address_in_use_log",
        "run_logged_state_handoff",
    ]
    .iter()
    .map(|n| declaration(&source, n))
    .collect::<String>();
    let setup = r#"
PORT_START_ATTEMPTS=3
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
family_automation=(port_allocator)
port_allocator() { printf '%s\n' "$*" >> "$PWD/allocator.argv"; "$PWD/controller" "$@"; }
quote_cmd() { printf '%q ' "$@"; }
record_event() { printf '%s|%s|%s\n' "$1" "$2" "$3" >> "$PWD/events"; }
core_attempt=0
core_probe() {
 core_attempt=$((core_attempt+1)); printf '%s\0' "$@" > "$PWD/core-$core_attempt.argv"
 if (( core_attempt == 1 )); then printf 'Address already in use\n'; return 23; fi
 : > "$PWD/single.json"; : > "$PWD/chain.json"
}
state_probe() { printf '%s\0' "$@" > "$PWD/state.argv"; : > "$PWD/state.json"; }
"#;
    let result = fixture.run(
        &declarations,
        setup,
        r#"
run_logged_core_parity "$PWD/single.json" "$PWD/chain.json" core_probe
run_logged_state_handoff "$PWD/state.json" state_probe
"#,
    );
    assert_eq!(result.status, 0, "{}", result.stderr);
    assert_eq!(
        fs::read_to_string(fixture.0.path().join("allocator.argv"))
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        [
            "automation local-ports 3",
            "automation local-ports 3",
            "automation local-ports 2"
        ]
    );
    for (file, flags) in [
        (
            "core-2.argv",
            vec![
                "--single-stage1-bind-addr",
                "--chain-stage1-bind-addr",
                "--chain-stage2-bind-addr",
            ],
        ),
        (
            "state.argv",
            vec!["--source-bind-addr", "--restore-bind-addr"],
        ),
    ] {
        let bytes = fs::read(fixture.0.path().join(file)).unwrap();
        let parts = bytes
            .split(|b| *b == 0)
            .filter(|b| !b.is_empty())
            .map(|b| std::str::from_utf8(b).unwrap())
            .collect::<Vec<_>>();
        let mut ports = std::collections::BTreeSet::new();
        for flag in flags {
            let index = parts.iter().position(|p| *p == flag).unwrap();
            let address: std::net::SocketAddr = parts[index + 1].parse().unwrap();
            assert!(address.ip().is_loopback() && address.port() != 0);
            assert!(ports.insert(address.port()));
        }
    }
    let events = fs::read_to_string(fixture.0.path().join("events")).unwrap();
    assert!(
        events.contains("single-step|pass|0")
            && events.contains("chain|pass|0")
            && events.contains("state-handoff|pass|0")
    );
    let other = fixture.run(
        &declarations,
        &format!("{setup}\ncore_probe() {{ printf 'other startup failure\\n'; return 41; }}"),
        r#"
: > "$PWD/allocator.argv"
run_logged_core_parity "$PWD/single.json" "$PWD/chain.json" core_probe
"#,
    );
    assert_eq!(other.status, 0, "{}", other.stderr);
    assert_eq!(
        fs::read_to_string(fixture.0.path().join("allocator.argv"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(
        fs::read_to_string(fixture.0.path().join("events"))
            .unwrap()
            .contains("single-step|fail|41")
    );
    let exhausted = fixture.run(
        &declarations,
        &format!("{setup}\ncore_probe() {{ printf 'EADDRINUSE\\n'; return 23; }}"),
        r#"
: > "$PWD/allocator.argv"
run_logged_core_parity "$PWD/single.json" "$PWD/chain.json" core_probe
"#,
    );
    assert_eq!(exhausted.status, 0, "{}", exhausted.stderr);
    assert_eq!(
        fs::read_to_string(fixture.0.path().join("allocator.argv"))
            .unwrap()
            .lines()
            .count(),
        3
    );
    assert!(
        fs::read_to_string(fixture.0.path().join("events"))
            .unwrap()
            .contains("single-step|fail|23")
    );
    fixture.finish();
}
