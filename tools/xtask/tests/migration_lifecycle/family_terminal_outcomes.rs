//! Actual sourced classifier and producer declarations with private terminal evidence.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::PathBuf, time::Duration};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn record_declaration() -> String {
    let source = fs::read_to_string(repository().join("skippy/scripts/family-certify.sh")).unwrap();
    let marker = "\nrecord_event() {\n";
    assert_eq!(source.matches(marker).count(), 1);
    format!(
        "record_event() {{\n{}\n}}\n",
        source
            .split_once(marker)
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0
    )
}
struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }
    fn run(
        &self,
        script: &str,
        values: &[(&str, String)],
        private_path: bool,
    ) -> (i32, String, String) {
        let mut environment = BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(if private_path {
                    self.0.path().join("bin").into_os_string()
                } else {
                    std::env::var_os("PATH").unwrap()
                }),
            ),
            ("HOME".into(), Value::Public(self.0.path().into())),
            (
                "CLASSIFIER".into(),
                Value::Public(repository().join("scripts/lib/family-outcome.sh").into()),
            ),
            (
                "GITHUB_PATH".into(),
                Value::Public(self.0.path().join("github-path").into()),
            ),
        ]);
        for (key, value) in values {
            environment.insert((*key).into(), Value::Public(value.as_str().into()));
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
                cwd: self.0.path().into(),
                environment,
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
        assert_eq!(result.process.outcome, Outcome::Exited, "{result:?}");
        assert!(result.process.failure.is_none() && result.process.cleanup.failure.is_none());
        assert!(
            result.process.cleanup.complete
                && !result.process.cleanup.forced
                && !result.process.cleanup.graceful_signal_failed
        );
        for stream in [&result.process.stdout, &result.process.stderr] {
            assert!(!stream.truncated && stream.suppressed_lines == 0);
        }
        let stdout = result.stdout.unwrap();
        let stderr = result.stderr.unwrap();
        assert_eq!(
            u64::try_from(stdout.as_bytes().len()).unwrap(),
            result.process.stdout.bytes_seen
        );
        assert_eq!(
            u64::try_from(stderr.as_bytes().len()).unwrap(),
            result.process.stderr.bytes_seen
        );
        (
            result.process.status.unwrap().code().unwrap(),
            String::from_utf8(stdout.as_bytes().to_vec()).unwrap(),
            String::from_utf8(stderr.as_bytes().to_vec()).unwrap(),
        )
    }
    fn finish(self) {
        self.0
            .close()
            .expect("owned classifier fixture cleanup failed");
    }
}
const TERMINALS: [(&str, &str); 7] = [
    (
        "runtime-error",
        "+ tool --startup-timeout-secs 900\nlistener disconnected\n",
    ),
    (
        "runtime-error",
        "+ tool --allow-mismatch\nlistener disconnected\n",
    ),
    ("timeout", "stage 1 binary server did not become ready\n"),
    (
        "unsupported",
        "Unsupported: stage graph did not expose a stable output activation boundary\n",
    ),
    ("model-invalid", "missing tensor blk.5.ssm_in.weight\n"),
    ("mismatch", "authoritative token mismatch\n"),
    ("harness", "corpus file does not exist\n"),
];
#[test]
fn actual_terminal_classifier_ignores_recorded_options_and_preserves_each_failure_class() {
    for (expected, evidence) in TERMINALS {
        let fixture = Fixture::new();
        fs::write(fixture.0.path().join("lane.log"), evidence).unwrap();
        let (status, stdout, stderr) = fixture.run(
            "set -euo pipefail\nsource \"$CLASSIFIER\"\nclassify_family_outcome fail lane.log ''",
            &[],
            false,
        );
        assert_eq!(status, 0, "{stderr}");
        assert_eq!(stdout, format!("{expected}\n"));
        fixture.finish();
    }
}
#[test]
fn actual_record_event_keeps_status_exit_code_and_typed_terminal_outcome_in_receipt() {
    for (expected, evidence) in TERMINALS {
        let fixture = Fixture::new();
        fs::write(fixture.0.path().join("lane.log"), evidence).unwrap();
        let script = format!(
            "set -euo pipefail\nsource \"$CLASSIFIER\"\n{}\nCOMMANDS_JSONL=commands.jsonl\nrecord_event fixture fail 23 lane.log '' ''",
            record_declaration()
        );
        let (status, _, stderr) = fixture.run(&script, &[], false);
        assert_eq!(status, 0, "{stderr}");
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.0.path().join("commands.jsonl")).unwrap())
                .unwrap();
        assert_eq!(receipt["name"], "fixture");
        assert_eq!(receipt["status"], "fail");
        assert_eq!(receipt["exit_code"], 23);
        assert_eq!(receipt["outcome"], expected);
        assert_eq!(receipt["log"], "lane.log");
        assert!(receipt.get("report").is_none() && receipt.get("note").is_none());
        fixture.finish();
    }
}
#[test]
fn classifier_pass_skip_and_note_precedence_are_not_reclassified_by_stale_logs() {
    for (status, note, expected) in [
        ("pass", "deadline exceeded", "pass"),
        ("skipped", "deadline exceeded", "skipped"),
        ("fail", "deadline exceeded", "timeout"),
    ] {
        let fixture = Fixture::new();
        fs::write(
            fixture.0.path().join("lane.log"),
            "authoritative token mismatch\n",
        )
        .unwrap();
        let (code, stdout, stderr) = fixture.run("set -euo pipefail\nsource \"$CLASSIFIER\"\nclassify_family_outcome \"$STATUS\" lane.log \"$NOTE\"", &[("STATUS", status.into()), ("NOTE", note.into())], false);
        assert_eq!(code, 0, "{stderr}");
        assert_eq!(stdout, format!("{expected}\n"));
        fixture.finish();
    }
}
#[test]
fn actual_changed_pin_agent_preflight_rejects_missing_failed_or_empty_version() {
    let action = crate::workflow_yaml::parse(
        &fs::read_to_string(repository().join(".github/actions/setup-canary-runner/action.yml"))
            .unwrap(),
    )
    .unwrap();
    let crate::workflow_yaml::Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap()
    else {
        panic!("steps");
    };
    let step = steps
        .iter()
        .find(|step| {
            step.get("name").and_then(crate::workflow_yaml::Node::text)
                == Some("Verify changed-pin agent executable")
        })
        .unwrap();
    assert_eq!(
        step.get("if").and_then(crate::workflow_yaml::Node::text),
        Some("${{ inputs.require-agent == 'true' }}")
    );
    let script = step
        .get("run")
        .and_then(crate::workflow_yaml::Node::text)
        .unwrap();
    for version in [
        None,
        Some("exit 23"),
        Some("exit 0"),
        Some("printf 'fixture-goose 1\\n'"),
    ] {
        let fixture = Fixture::new();
        fs::create_dir(fixture.0.path().join("bin")).unwrap();
        if let Some(body) = version {
            let executable = fixture.0.path().join(".local/bin/goose");
            fs::create_dir_all(executable.parent().unwrap()).unwrap();
            fs::write(
                &executable,
                format!("#!/bin/sh\n[ \"$#\" = 1 ] && [ \"$1\" = --version ] || exit 92\n{body}\n"),
            )
            .unwrap();
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(executable, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let (status, stdout, stderr) = fixture.run(script, &[], true);
        assert_eq!(
            status,
            i32::from(version != Some("printf 'fixture-goose 1\\n'")),
            "{stdout}{stderr}"
        );
        assert_eq!(
            stdout.contains("goose version: fixture-goose 1"),
            status == 0
        );
        assert_eq!(
            fs::read_to_string(fixture.0.path().join("github-path")).unwrap(),
            format!("{}/.local/bin\n", fixture.0.path().display())
        );
        fixture.finish();
    }
}
