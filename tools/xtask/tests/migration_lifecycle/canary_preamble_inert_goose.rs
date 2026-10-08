//! Execute production preamble and session functions using finite inert tools.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    time::Duration,
};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn wrapper() -> String {
    fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap()
}
fn declaration(source: &str, name: &str) -> String {
    let marker = format!("\n{name}() {{\n");
    assert_eq!(source.matches(&marker).count(), 1);
    let tail = source.split_once(&marker).unwrap().1;
    let body = tail.split_once("\n}\n").unwrap().0;
    format!("{name}() {{\n{body}\n}}\n")
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct Fixture {
    temporary: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        fs::create_dir(root.join("tools")).unwrap();
        fs::create_dir(root.join("tmp")).unwrap();
        Self { temporary, root }
    }
    fn install_timeout_tools(&self) {
        let path = std::env::var_os("PATH").expect("fixture qualification requires prepared jq");
        let jq = std::env::split_paths(&path)
            .map(|directory| directory.join("jq"))
            .find(|candidate| candidate.is_file())
            .expect("fixture qualification requires prepared jq")
            .canonicalize()
            .unwrap();
        // Execute the prepared jq in its original platform context. Relocating
        // an Apple arm64e system binary into a fixture can break its execution.
        let quoted = jq.to_str().unwrap().replace('\'', "'\\''");
        executable(
            &self.root.join("tools/jq"),
            &format!("#!/bin/bash\nexec '{quoted}' \"$@\"\n"),
        );
    }
    fn environment(&self) -> BTreeMap<std::ffi::OsString, Value> {
        BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(
                    format!("{}:/usr/bin:/bin", self.root.join("tools").display()).into(),
                ),
            ),
            (
                "HOME".into(),
                Value::Public(self.root.clone().into_os_string()),
            ),
            (
                "RUNNER_TEMP".into(),
                Value::Public(self.root.join("tmp").into_os_string()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            ("GIT_MASTER".into(), Value::Public("1".into())),
            ("GIT_OPTIONAL_LOCKS".into(), Value::Public("0".into())),
        ])
    }
    fn run(
        &self,
        script: &Path,
        arguments: &[&str],
        environment: BTreeMap<std::ffi::OsString, Value>,
    ) -> process::RawProcessReport {
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: std::iter::once(script.as_os_str().to_owned())
                    .chain(
                        arguments
                            .iter()
                            .map(|argument| std::ffi::OsString::from(*argument)),
                    )
                    .map(Value::Public)
                    .collect(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(30),
                // Nested native canary-timeout has 10s graceful + 10s forced.
                graceful_shutdown: Duration::from_secs(25),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none() && report.process.cleanup.failure.is_none());
        assert!(
            report.process.cleanup.complete
                && !report.process.cleanup.forced
                && !report.process.cleanup.graceful_signal_failed
        );
        for (stream, raw) in [
            (&report.process.stdout, report.stdout.as_ref().unwrap()),
            (&report.process.stderr, report.stderr.as_ref().unwrap()),
        ] {
            assert!(!stream.truncated && stream.line_capture_complete);
            assert_eq!(stream.oversized_lines, 0);
            assert_eq!(
                u64::try_from(raw.as_bytes().len()).unwrap(),
                stream.bytes_seen
            );
        }
        report
    }
    fn finish(self) {
        self.temporary
            .close()
            .expect("owned inert canary fixture cleanup");
    }
}

#[test]
fn actual_repair_preamble_reenters_arm64_before_any_initialization() {
    let source = wrapper();
    let preamble = source.split_once("\nROOT=\"").unwrap().0;
    for (os, architecture, arm64, reentered) in [
        ("Darwin", "x86_64", "1", true),
        ("Darwin", "arm64", "1", false),
        ("Darwin", "x86_64", "0", false),
        ("Linux", "x86_64", "1", false),
    ] {
        let fixture = Fixture::new();
        executable(
            &fixture.root.join("tools/uname"),
            "#!/bin/bash\ncase \"$1\" in -s) printf '%s\\n' \"$INERT_OS\";; -m) printf '%s\\n' \"$INERT_ARCH\";; *) exit 91;; esac\n",
        );
        executable(
            &fixture.root.join("tools/sysctl"),
            "#!/bin/bash\n[[ \"$*\" == '-n hw.optional.arm64' ]] || exit 92\nprintf '%s\\n' \"$INERT_ARM64\"\n",
        );
        executable(
            &fixture.root.join("tools/arch"),
            "#!/bin/bash\nprintf '%s\\0' \"$@\" > arch.args\nprintf 'REENTERED\\n'\n",
        );
        let script = fixture.root.join("preamble.sh");
        fs::write(
            &script,
            format!("{preamble}\nprintf INITIALIZED > initialized\n"),
        )
        .unwrap();
        let mut env = fixture.environment();
        for (key, value) in [
            ("INERT_OS", os),
            ("INERT_ARCH", architecture),
            ("INERT_ARM64", arm64),
        ] {
            env.insert(key.into(), Value::Public(value.into()));
        }
        let report = fixture.run(&script, &["space value", "*.gguf"], env);
        assert_eq!(report.process.status.unwrap().code(), Some(0));
        assert_eq!(fixture.root.join("initialized").exists(), !reentered);
        if reentered {
            let expected =
                ["-arm64", script.to_str().unwrap(), "space value", "*.gguf"].join("\0") + "\0";
            assert_eq!(
                fs::read(fixture.root.join("arch.args")).unwrap(),
                expected.as_bytes()
            );
            assert_eq!(report.stdout.unwrap().as_bytes(), b"REENTERED\n");
        } else {
            assert!(!fixture.root.join("arch.args").exists());
        }
        fixture.finish();
    }
}

#[test]
fn actual_complete_wrapper_refuses_malicious_sha_before_tools_or_state() {
    let fixture = Fixture::new();
    executable(
        &fixture.root.join("tools/uname"),
        "#!/bin/bash\ncase \"$1\" in -s) printf Linux;; -m) printf x86_64;; *) exit 91;; esac\n",
    );
    executable(
        &fixture.root.join("tools/git"),
        "#!/bin/bash\nprintf forbidden > git-called\nexit 99\n",
    );
    let mut environment = fixture.environment();
    environment.insert(
        "UPSTREAM_SHA_INPUT".into(),
        Value::Public("not-a-sha; echo pwned".into()),
    );
    let report = fixture.run(
        &repository().join("scripts/llama-canary-agent-repair.sh"),
        &[],
        environment,
    );
    assert_eq!(report.process.status.unwrap().code(), Some(1));
    let output = [
        report.stdout.unwrap().as_bytes(),
        report.stderr.unwrap().as_bytes(),
    ]
    .concat();
    let diagnostic = String::from_utf8(output).unwrap();
    assert!(diagnostic.contains("non-40-hex upstream SHA"));
    assert_eq!(diagnostic.matches("pwned").count(), 1);
    assert!(!fixture.root.join("git-called").exists());
    assert!(!fixture.root.join("pwned").exists());
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
    fixture.finish();
}

const GOOSE: &str = r#"#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
count=0
[[ ! -f "$root/goose.count" ]] || read -r count < "$root/goose.count"
count=$((count+1))
printf '%s\n' "$count" > "$root/goose.count"
printf '%s\0' "$@" > "$root/goose-$count.args"
for key in GH_TOKEN GITHUB_TOKEN CANARY_REPAIR_TOKEN; do
  if [[ "${!key+set}" == set ]]; then printf forbidden > "$root/credential-leak"; exit 95; fi
done
[[ "$GOOSE_MODE" == auto && "$GOOSE_DISABLE_SESSION_NAMING" == true ]] || exit 96
[[ "$CANARY_REPAIR_LOG_DIR" == "$root/command-logs" ]] || exit 96
for ((attempt=0; attempt<200; attempt++)); do
  if [[ -f "$root/heartbeat-sleep.pids" ]]; then
    lines=$(wc -l < "$root/heartbeat-sleep.pids")
    if (( lines >= count )); then break; fi
  fi
  /bin/sleep 0.01
done
[[ -f "$root/heartbeat-sleep.pids" ]] || exit 97
lines=$(wc -l < "$root/heartbeat-sleep.pids")
(( lines >= count )) || exit 97
printf 'inert goose call %s\n' "$count"
if (( count == 2 )); then exit 23; fi
"#;

#[test]
fn actual_two_agent_steps_preserve_named_resume_credentials_exit_log_and_heartbeat_cleanup() {
    let fixture = Fixture::new();
    fixture.install_timeout_tools();
    executable(&fixture.root.join("tools/goose"), GOOSE);
    executable(
        &fixture.root.join("tools/sleep"),
        r#"#!/bin/bash
set -euo pipefail
[[ "$*" == 600 ]] || exit 94
root="$(cd "$(dirname "$0")/.." && pwd)"
printf '%s\n' "$$" >> "$root/heartbeat-sleep.pids"
exec /bin/sleep 600
"#,
    );
    let source = wrapper();
    let declarations = ["run_for", "remaining_repair_seconds", "agent_session_step"]
        .iter()
        .map(|name| declaration(&source, name))
        .collect::<String>();
    let script = fixture.root.join("session.sh");
    fs::write(
        &script,
        format!(
            r#"set -euo pipefail
ROOT="$PWD"
HARNESS_MODE=repair-build
AGENT_PROVIDER=fixture_provider
AGENT_MODEL=fixture_model
AGENT_LOG="$ROOT/agent.log"
AGENT_COMMAND_LOG_DIR="$ROOT/command-logs"
mkdir -p "$AGENT_COMMAND_LOG_DIR"
AGENT_SESSION_NAME=llama-canary-repair-fixture-1-local
AGENT_SESSION_STARTED=false
REPAIR_DEADLINE_AT="$(( $(date +%s) + 20 ))"
{declarations}
agent_session_step 'first space prompt'
[[ "$AGENT_SESSION_STARTED" == true ]] || exit 98
if agent_session_step 'second space prompt'; then exit 99; else status=$?; fi
[[ "$AGENT_SESSION_STARTED" == true ]] || exit 98
exit "$status"
"#
        ),
    )
    .unwrap();
    let mut environment = fixture.environment();
    for key in ["GH_TOKEN", "GITHUB_TOKEN", "CANARY_REPAIR_TOKEN"] {
        environment.insert(key.into(), Value::Secret("inert credential".into()));
    }
    let report = fixture.run(&script, &[], environment);
    assert_eq!(
        report.process.status.unwrap().code(),
        Some(23),
        "stdout={} stderr={}",
        String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()),
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
    );
    assert!(!fixture.root.join("credential-leak").exists());
    let common = [
        "run",
        "--provider",
        "fixture_provider",
        "--model",
        "fixture_model",
        "--with-builtin",
        "developer",
        "--no-profile",
        "--max-turns",
        "1000",
        "--output-format",
        "text",
        "--name",
        "llama-canary-repair-fixture-1-local",
    ];
    for (index, prompt) in [(1, "first space prompt"), (2, "second space prompt")] {
        let mut expected = common.to_vec();
        if index == 2 {
            expected.push("--resume");
        }
        expected.extend(["--text", prompt]);
        let bytes = expected.join("\0") + "\0";
        assert_eq!(
            fs::read(fixture.root.join(format!("goose-{index}.args"))).unwrap(),
            bytes.as_bytes()
        );
    }
    let log = fs::read_to_string(fixture.root.join("agent.log")).unwrap();
    assert!(log.contains("inert goose call 1") && log.contains("inert goose call 2"));
    assert!(log.contains("agent developer task exited with status 23"));
    assert!(!log.contains("inert credential"));
    let sleepers = fs::read_to_string(fixture.root.join("heartbeat-sleep.pids")).unwrap();
    assert_eq!(sleepers.lines().count(), 2);
    for sleeper in sleepers.lines() {
        let pid: i32 = sleeper.parse().unwrap();
        assert!(pid > 0);
        // SAFETY: zero only queries the fixture-observed PID; it sends no signal.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
    fixture.finish();
}
