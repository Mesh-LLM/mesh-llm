//! CPU producer orchestration with finite tool boundaries; no native builds.
#![cfg(unix)]

#[allow(dead_code, unused_imports)]
#[path = "../src/process/mod.rs"]
mod process;

use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    time::Duration,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

struct Receipt {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

struct Fixture(tempfile::TempDir);

impl Fixture {
    fn new() -> Self {
        let fixture = Self(tempfile::tempdir().unwrap());
        for dir in [
            "bin",
            "scripts",
            "primary-metal/native",
            "primary-metal/cargo",
        ] {
            fs::create_dir_all(fixture.path().join(dir)).unwrap();
        }
        fs::copy(
            root().join("scripts/skippy-workload-oracles-build.sh"),
            fixture
                .path()
                .join("scripts/skippy-workload-oracles-build.sh"),
        )
        .unwrap();
        fs::write(
            fixture.path().join("primary-metal/native/sentinel"),
            b"Metal native bytes",
        )
        .unwrap();
        fs::write(
            fixture.path().join("primary-metal/cargo/sentinel"),
            b"Metal Cargo bytes",
        )
        .unwrap();
        fixture
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn executable(&self, relative: &str, body: &str) {
        let path = self.path().join(relative);
        fs::write(&path, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }

    fn command(&self) -> Command {
        let mut command = Command::new("bash");
        command
            .env("FIXTURE_ROOT", self.path())
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    self.path().join("bin").display(),
                    std::env::var("PATH").unwrap()
                ),
            )
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
            .env("LLAMA_STAGE_BACKEND", "metal")
            .env("LLAMA_STAGE_LINK_MODE", "dynamic")
            .env("LLAMA_BUILD_DIR", self.path().join("primary-metal/native"))
            .env(
                "LLAMA_STAGE_BUILD_DIR",
                self.path().join("primary-metal/native"),
            )
            .env("CARGO_TARGET_DIR", self.path().join("primary-metal/cargo"));
        command
    }

    fn run(&self, command: Command) -> Receipt {
        let mut environment = ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
            .into_iter()
            .filter_map(|key| {
                std::env::var_os(key).map(|value| (key.into(), process::Value::Public(value)))
            })
            .collect::<BTreeMap<_, _>>();
        for (key, value) in command.get_envs() {
            if let Some(value) = value {
                environment.insert(key.to_owned(), process::Value::Public(value.to_owned()));
            } else {
                environment.remove(key);
            }
        }
        let executable = if command.get_program() == "bash" {
            PathBuf::from("/bin/bash")
        } else {
            PathBuf::from(command.get_program())
        };
        let spec = process::ProcessSpec {
            executable,
            cwd: command
                .get_current_dir()
                .unwrap_or_else(|| self.path())
                .to_owned(),
            arguments: command
                .get_args()
                .map(|arg| process::Value::Public(arg.to_owned()))
                .collect(),
            environment,
        };
        let limits = process::Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        };
        let raw = process::supervise_raw(
            &spec,
            &limits,
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(1024 * 1024),
                stderr: std::num::NonZeroUsize::new(1024 * 1024),
            },
        )
        .unwrap();
        let report = raw.process;
        assert!(report.cleanup.complete, "{report:?}");
        assert!(
            !report.stdout.truncated && !report.stderr.truncated,
            "{report:?}"
        );
        assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
        Receipt {
            status: report.status.unwrap(),
            stdout: raw.stdout.unwrap().as_bytes().to_vec(),
            stderr: raw.stderr.unwrap().as_bytes().to_vec(),
        }
    }

    fn producer(&self, print: bool, build_root: &Path) -> Command {
        let mut command = self.command();
        command.arg(self.path().join("scripts/skippy-workload-oracles-build.sh"));
        if print {
            command.arg("--print-env");
        }
        command.arg(build_root);
        command
    }

    fn install_tools(&self) {
        self.executable("bin/automation", r#"
[[ "$1" == automation && "$2" == canary-receipts ]] || exit 97
shift 2
case "$1" in
prepared-source)
  [[ $# == 3 && "$2" == --root && "$3" == "$FIXTURE_ROOT" ]] || exit 97
  [[ -f "$FIXTURE_ROOT/cpu closure/source.json" ]] || exit 97
  printf 'prepared\n' >> "$FIXTURE_ROOT/events"
  if [[ "${FIXTURE_FAIL_STAGE:-}" == prepared ]]; then exit 41; fi
  ;;
workload-manifest)
  shift
  case "$1" in
  snapshot)
    [[ $# == 3 && "$2" == "$FIXTURE_ROOT" && "$3" == "$FIXTURE_ROOT/cpu closure/source.json" ]] || exit 97
    [[ ! -e "$FIXTURE_ROOT/cpu closure/native" && ! -e "$FIXTURE_ROOT/cpu closure/cargo" ]] || exit 97
    printf 'snapshot\n' >> "$FIXTURE_ROOT/events"
    if [[ "${FIXTURE_FAIL_STAGE:-}" == snapshot ]]; then exit 40; fi
    printf '{"fixture":"captured source"}\n' > "$3"
    ;;
  produce)
    [[ $# == 5 && "$2" == "$FIXTURE_ROOT" && "$3" == "$FIXTURE_ROOT/cpu closure" && "$4" == "$3/cargo/debug/skippy-test" && "$5" == "$3/source.json" ]] || exit 97
    [[ -f "$3/cargo/debug/skippy-server" && -d "$3/native" && -f "$4" && -f "$5" ]] || exit 97
    for tool in skippy-server skippy-model-package skippy-correctness skippy-topology; do [[ -f "$3/cargo/debug/$tool" ]] || exit 97; done
    for tool in llama-server llama-completion llama-tts; do [[ -f "$3/native/bin/$tool" ]] || exit 97; done
    printf 'manifest\n' >> "$FIXTURE_ROOT/events"
    if [[ "${FIXTURE_FAIL_STAGE:-}" == manifest ]]; then exit 43; fi
    printf '%s\n' "$3/cargo/debug/skippy-server" "$3/native" "$4" "$5" > "$3/producer.json"
    ;;
  *) exit 97 ;;
  esac
  ;;
*) exit 97 ;;
esac
"#);
        self.executable("bin/python3", "exit 98");
        let cpu_env = r#"
[[ "$LLAMA_STAGE_BACKEND" == cpu && "$LLAMA_STAGE_LINK_MODE" == static ]] || exit 97
[[ "$LLAMA_STAGE_WORKLOAD_ORACLE" == ON && "$LLAMA_STAGE_UPSTREAM_TESTS" == OFF ]] || exit 97
[[ "$LLAMA_STAGE_FULL_REPLAY" == OFF && "$LLAMA_STAGE_BUILD_TESTS" == OFF ]] || exit 97
[[ "$LLAMA_BUILD_DIR" == "$FIXTURE_ROOT/cpu closure/native" ]] || exit 97
[[ "$LLAMA_STAGE_BUILD_DIR" == "$LLAMA_BUILD_DIR" ]] || exit 97
[[ "$CARGO_TARGET_DIR" == "$FIXTURE_ROOT/cpu closure/cargo" ]] || exit 97
"#;
        self.executable("scripts/build-llama.sh", &format!(r#"{cpu_env}
if [[ "$(uname -s)" == Darwin ]]; then
  [[ $# == 1 && "$1" == -DCMAKE_OSX_ARCHITECTURES=arm64 ]] || exit 97
else
  [[ $# == 0 ]] || exit 97
fi
printf 'native\n' >> "$FIXTURE_ROOT/events"
if [[ "${{FIXTURE_FAIL_STAGE:-}}" == native ]]; then exit 42; fi
mkdir -p "$LLAMA_BUILD_DIR/bin"
for tool in llama-server llama-completion llama-tts; do printf 'CPU oracle\n' > "$LLAMA_BUILD_DIR/bin/$tool"; done
"#));
        self.executable("bin/just", &format!(r#"{cpu_env}
[[ "$1" == with-lld && "$2" == cargo ]] || exit 97
shift 2
case "$1" in
build)
  [[ "$*" == 'build --locked -p skippy-server -p skippy-model-package -p skippy-correctness -p skippy-topology --bins' ]] || exit 97
  printf 'build\n' >> "$FIXTURE_ROOT/events"
  if [[ "${{FIXTURE_FAIL_STAGE:-}}" == build ]]; then exit 44; fi
  mkdir -p "$CARGO_TARGET_DIR/debug"
  for tool in skippy-server skippy-model-package skippy-correctness skippy-topology; do printf 'CPU candidate\n' > "$CARGO_TARGET_DIR/debug/$tool"; done
  ;;
test)
  [[ "$*" == 'test --locked -p skippy-server --lib --no-run --message-format=json' ]] || exit 97
  printf 'test\n' >> "$FIXTURE_ROOT/events"
  if [[ "${{FIXTURE_FAIL_STAGE:-}}" == test ]]; then exit 45; fi
  printf 'CPU library tests\n' > "$CARGO_TARGET_DIR/debug/skippy-test"
  printf '{{"reason":"compiler-artifact","profile":{{"test":true}},"target":{{"name":"skippy_server"}},"executable":"%s"}}\n' "$CARGO_TARGET_DIR/debug/skippy-test"
  ;;
*) exit 97 ;;
esac
"#));
        self.executable("bin/cargo", "exit 98");
    }

    fn graph(&self, failure: Option<&str>) -> Receipt {
        self.install_tools();
        let mut command = self.producer(false, &self.path().join("cpu closure"));
        command.env(
            "MESH_LLM_AUTOMATION_BIN",
            self.path().join("bin/automation"),
        );
        if let Some(failure) = failure {
            command.env("FIXTURE_FAIL_STAGE", failure);
        }
        self.run(command)
    }

    fn metal_untouched(&self) {
        assert_eq!(
            fs::read(self.path().join("primary-metal/native/sentinel")).unwrap(),
            b"Metal native bytes"
        );
        assert_eq!(
            fs::read(self.path().join("primary-metal/cargo/sentinel")).unwrap(),
            b"Metal Cargo bytes"
        );
        assert_eq!(
            fs::read_dir(self.path().join("primary-metal/native"))
                .unwrap()
                .count(),
            1
        );
        assert_eq!(
            fs::read_dir(self.path().join("primary-metal/cargo"))
                .unwrap()
                .count(),
            1
        );
    }
}

#[test]
fn print_env_has_exact_export_bytes_and_preserves_spaced_paths() {
    let fixture = Fixture::new();
    let build_root = fixture.path().join("cpu closure");
    let receipt = fixture.run(fixture.producer(true, &build_root));
    assert!(
        receipt.status.success(),
        "{}",
        String::from_utf8_lossy(&receipt.stderr)
    );
    let prefix = build_root.display();
    let expected = format!(
        "SKIPPY_WORKLOAD_ORACLE_SERVER={prefix}/native/bin/llama-server\nSKIPPY_WORKLOAD_ORACLE_COMPLETION={prefix}/native/bin/llama-completion\nSKIPPY_WORKLOAD_ORACLE_TTS={prefix}/native/bin/llama-tts\nSKIPPY_WORKLOAD_NATIVE_BUILD_DIR={prefix}/native\nSKIPPY_WORKLOAD_CANDIDATE_BIN_DIR={prefix}/cargo/debug\nSKIPPY_WORKLOAD_PRODUCER_MANIFEST={prefix}/producer.json\n"
    );
    assert_eq!(receipt.stdout, expected.as_bytes());
    assert!(receipt.stderr.is_empty());
    assert!(!build_root.exists());
    fixture.metal_untouched();
}

#[test]
fn export_rejects_relative_and_line_injection_paths_before_writes() {
    let fixture = Fixture::new();
    for path in ["relative", "/tmp/line\nGH_TOKEN=bad", "/tmp/line\rnext"] {
        let receipt = fixture.run(fixture.producer(true, Path::new(path)));
        assert!(!receipt.status.success());
        assert!(receipt.stdout.is_empty());
    }
    assert!(!fixture.path().join("events").exists());
    fixture.metal_untouched();
}

#[test]
fn default_rust_family_plan_contains_all_six_non_chat_oracle_classes() {
    let fixture = Fixture::new();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(root())
        .args(["--repo-root"])
        .arg(root())
        .args(["ci", "family-plan"]);
    let receipt = fixture.run(command);
    assert!(
        receipt.status.success(),
        "{}",
        String::from_utf8_lossy(&receipt.stderr)
    );
    let plan: Value = serde_json::from_slice(&receipt.stdout).unwrap();
    let rows: Vec<_> = plan["selected_models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["class"] != "causal_generation")
        .collect();
    assert_eq!(rows.len(), 6);
    let classes: BTreeSet<_> = rows
        .iter()
        .map(|row| row["class"].as_str().unwrap())
        .collect();
    assert_eq!(
        classes,
        BTreeSet::from([
            "embedding",
            "rerank",
            "encoder_decoder",
            "ocr",
            "speech_synthesis",
            "speech_recognition"
        ])
    );
    assert!(rows.iter().all(|row| row["profile"] == "workload-oracle"));
}

#[test]
fn cpu_graph_is_isolated_and_exports_all_producer_artifact_paths() {
    let fixture = Fixture::new();
    let receipt = fixture.graph(None);
    assert!(
        receipt.status.success(),
        "{}",
        String::from_utf8_lossy(&receipt.stderr)
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("events")).unwrap(),
        "snapshot\nprepared\nnative\nbuild\ntest\nmanifest\n"
    );
    let build = fixture.path().join("cpu closure");
    let expected = format!(
        "{}/cargo/debug/skippy-server\n{}/native\n{}/cargo/debug/skippy-test\n{}/source.json\n",
        build.display(),
        build.display(),
        build.display(),
        build.display()
    );
    assert_eq!(
        fs::read(build.join("producer.json")).unwrap(),
        expected.as_bytes()
    );
    for oracle in ["llama-server", "llama-completion", "llama-tts"] {
        assert_eq!(
            fs::read(build.join("native/bin").join(oracle)).unwrap(),
            b"CPU oracle\n"
        );
    }
    fixture.metal_untouched();
}

#[test]
fn failed_preparation_build_or_stamp_cannot_emit_successful_manifest() {
    for (failure, events) in [
        ("snapshot", "snapshot\n"),
        ("prepared", "snapshot\nprepared\n"),
        ("native", "snapshot\nprepared\nnative\n"),
        ("build", "snapshot\nprepared\nnative\nbuild\n"),
        ("test", "snapshot\nprepared\nnative\nbuild\ntest\n"),
        (
            "manifest",
            "snapshot\nprepared\nnative\nbuild\ntest\nmanifest\n",
        ),
    ] {
        let fixture = Fixture::new();
        let receipt = fixture.graph(Some(failure));
        assert!(
            !receipt.status.success(),
            "{failure}: status={:?}; stderr={}",
            receipt.status,
            String::from_utf8_lossy(&receipt.stderr)
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            events
        );
        assert!(!fixture.path().join("cpu closure/producer.json").exists());
        fixture.metal_untouched();
    }
}

#[test]
fn shared_canary_full_build_runs_cpu_producer_before_required_smokes_and_stops_on_failure() {
    let source = fs::read_to_string(root().join("scripts/llama-canary-agent-repair.sh")).unwrap();
    let body = source
        .split("run_full_build() {\n")
        .nth(1)
        .unwrap()
        .split("# Local CLI compatibility path.")
        .next()
        .unwrap();
    for mode in ["repair-build", "verify-build", "pinned-build"] {
        for fail_cpu in [false, true] {
            let fixture = Fixture::new();
            fixture.executable(
                "bin/lipo",
                "[[ $1 == -archs ]] || exit 97; printf 'arm64\\n'",
            );
            let harness = format!(
                r#"
set -euo pipefail
run_full_build() {{
{body}
BUILD_LOG="$FIXTURE_ROOT/build.log"
LLAMA_STAGE_BUILD_DIR="$FIXTURE_ROOT/primary-metal/native"
STATE_DIR="$FIXTURE_ROOT"
SYSTEMONE_SMOKE_DIR="$FIXTURE_ROOT/system-one"
HARNESS_MODE="$CASE_MODE"
run_verification_logged() {{
  local label="$1"
  shift 2
  printf '%s\n' "$label" >> "$FIXTURE_ROOT/gates"
  if [[ "$label" == 'pinned CPU workload oracles and candidate' ]]; then
    [[ $# == 3 && "$1" == just && "$2" == skippy-workload-oracles-build && "$3" == "$LLAMA_STAGE_BUILD_DIR-workloads" ]] || exit 97
    [[ "$FAIL_CPU" == 0 ]] || return 41
  fi
}}
run_full_build
"#
            );
            let mut command = fixture.command();
            command
                .args(["-c", &harness])
                .env("CASE_MODE", mode)
                .env("FAIL_CPU", if fail_cpu { "1" } else { "0" });
            let receipt = fixture.run(command);
            let gates = fs::read_to_string(fixture.path().join("gates")).unwrap();
            let before_cpu = "complete patched llama.cpp build\ngenerated model-family patch check\nstage runtime crate build\nSkippy smoke tests\npinned CPU workload oracles and candidate\n";
            if fail_cpu {
                assert!(!receipt.status.success());
                assert_eq!(gates, before_cpu);
            } else {
                assert!(
                    receipt.status.success(),
                    "{}",
                    String::from_utf8_lossy(&receipt.stderr)
                );
                assert_eq!(
                    gates,
                    format!(
                        "{before_cpu}build transferable multimodal test executable\nSystem One smoke\nLaya smoke\n"
                    )
                );
            }
            fixture.metal_untouched();
        }
    }
}

#[path = "workload_oracle_producer_contract/full_replay.rs"]
mod full_replay;
#[path = "workload_oracle_producer_contract/review_regressions.rs"]
mod review_regressions;
