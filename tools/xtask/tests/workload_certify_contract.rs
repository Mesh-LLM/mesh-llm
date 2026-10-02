//! Workload wrapper admission and CPU producer consumption, without native execution.
#![cfg(unix)]

use std::{
    fs,
    os::unix::{fs::PermissionsExt as _, process::CommandExt as _},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}

struct Fixture {
    directory: tempfile::TempDir,
}

struct Receipt {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        for path in ["bin", "native", "candidate", "work"] {
            fs::create_dir(fixture.path().join(path)).unwrap();
        }
        fs::write(fixture.path().join("model.gguf"), b"inert model fixture").unwrap();
        fs::write(fixture.path().join("projector.gguf"), b"inert projector").unwrap();
        for tool in ["cargo", "jq", "just"] {
            fixture.executable(
                &format!("bin/{tool}"),
                "printf 'unexpected execution\n' >> \"$FIXTURE_ROOT/executed\"; exit 97",
            );
        }
        fixture
    }

    fn path(&self) -> &Path {
        self.directory.path()
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
            .env(
                "SKIPPY_WORKLOAD_NATIVE_BUILD_DIR",
                self.path().join("native"),
            )
            .env(
                "SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR",
                self.path().join("candidate"),
            )
            .env(
                "SKIPPY_WORKLOAD_SDK_PYTHON",
                self.path().join("missing-sdk"),
            )
            .env_remove("SKIPPY_WORKLOAD_PRODUCER_MANIFEST")
            .current_dir(root());
        command
    }

    fn run(&self, mut command: Command) -> Receipt {
        let stdout = self.path().join("stdout");
        let stderr = self.path().join("stderr");
        let mut child = command
            .process_group(0)
            .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
            .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = Command::new("/bin/kill")
                    .args(["-KILL", "--", &format!("-{}", child.id())])
                    .status();
                let _ = child.kill();
                let _ = child.wait();
                panic!("bounded workload fixture exceeded deadline");
            }
            thread::sleep(Duration::from_millis(10));
        };
        Receipt {
            status,
            stdout: fs::read_to_string(stdout).unwrap(),
            stderr: fs::read_to_string(stderr).unwrap(),
        }
    }

    fn wrapper(&self, class: &str, lane: &str, extra: &[&str]) -> Receipt {
        let mut command = self.command();
        command
            .arg(root().join("scripts/skippy-workload-certify.sh"))
            .args(["--class", class, "--lane", lane, "--model-path"])
            .arg(self.path().join("model.gguf"))
            .args(["--model-id", "fixture", "--work-dir"])
            .arg(self.path().join("work"))
            .arg("--skip-build")
            .args(extra);
        self.run(command)
    }

    fn rejected_without_execution(&self, receipt: &Receipt, diagnostic: &str) {
        assert!(!receipt.status.success(), "{}", receipt.stdout);
        assert!(receipt.stderr.contains(diagnostic), "{}", receipt.stderr);
        assert!(!self.path().join("executed").exists());
        assert!(
            !self
                .path()
                .join("work/workload-oracle-evidence.json")
                .exists()
        );
        assert!(!self.path().join("work/stage-openai.json").exists());
    }
}

#[test]
fn help_exposes_the_actual_certification_inputs() {
    for help in ["-h", "--help"] {
        let fixture = Fixture::new();
        let receipt = fixture.wrapper("embedding", "embedding-smoke", &[help]);
        assert!(receipt.status.success());
        assert!(receipt.stdout.is_empty());
        for option in [
            "--class",
            "--lane",
            "--projector-path",
            "--oracle-server",
            "--oracle-completion",
            "--oracle-tts",
            "--startup-timeout-secs",
            "--require-oracle",
            "--skip-build",
        ] {
            assert!(receipt.stderr.contains(option), "missing {option}");
        }
        assert!(!fixture.path().join("executed").exists());
    }
}

#[test]
fn invalid_startup_budgets_and_unknown_classes_stop_before_execution() {
    for budget in ["0", "-1", "1.5", "01", "86401", "abc"] {
        let fixture = Fixture::new();
        let receipt = fixture.wrapper(
            "embedding",
            "embedding-smoke",
            &["--startup-timeout-secs", budget],
        );
        fixture.rejected_without_execution(&receipt, "--startup-timeout-secs");
    }
    let fixture = Fixture::new();
    let receipt = fixture.wrapper("guessed", "guessed-equivalence", &[]);
    fixture.rejected_without_execution(&receipt, "unsupported model class");
}

#[test]
fn lane_projector_and_required_oracle_admission_are_mandatory() {
    let fixture = Fixture::new();
    let receipt = fixture.wrapper("embedding", "rerank-smoke", &[]);
    fixture.rejected_without_execution(&receipt, "does not match class");
    let receipt = fixture.wrapper("embedding", "embedding-smoke", &["--require-oracle"]);
    fixture.rejected_without_execution(&receipt, "requires a class-appropriate");
    for (class, lane) in [
        ("ocr", "ocr-smoke"),
        ("speech_synthesis", "speech-synthesis-smoke"),
        ("speech_recognition", "speech-recognition-smoke"),
    ] {
        let fixture = Fixture::new();
        let receipt = fixture.wrapper(class, lane, &[]);
        fixture.rejected_without_execution(&receipt, "requires a projector path");
    }
}

#[test]
fn required_embedding_sdk_cannot_be_skipped_before_model_execution() {
    let fixture = Fixture::new();
    let receipt = fixture.wrapper("embedding", "embedding-smoke", &[]);
    fixture.rejected_without_execution(&receipt, "official openai-python SDK smoke requires");
    assert!(receipt.stdout.is_empty());
    assert_eq!(
        fs::read_dir(fixture.path().join("work")).unwrap().count(),
        0
    );
}

#[test]
fn oracle_executable_and_class_compatibility_fail_closed() {
    let fixture = Fixture::new();
    let model = fixture.path().join("model.gguf");
    let model = model.to_str().unwrap();
    let projector = fixture.path().join("projector.gguf");
    let projector = projector.to_str().unwrap();
    for (class, lane, extra, diagnostic) in [
        (
            "embedding",
            "embedding-smoke",
            vec!["--oracle-server", model],
            "not executable",
        ),
        (
            "speech_synthesis",
            "speech-synthesis-smoke",
            vec!["--projector-path", projector, "--oracle-server", model],
            "different local-monolithic",
        ),
        (
            "embedding",
            "embedding-smoke",
            vec!["--oracle-tts", model],
            "only valid for speech synthesis",
        ),
        (
            "encoder_decoder",
            "encoder-decoder-smoke",
            vec!["--oracle-server", model],
            "different local-monolithic",
        ),
    ] {
        let receipt = fixture.wrapper(class, lane, &extra);
        fixture.rejected_without_execution(&receipt, diagnostic);
    }
}

fn candidate_branches() -> String {
    let source = fs::read_to_string(root().join("scripts/skippy-workload-certify.sh")).unwrap();
    let candidate = source
        .split_once("require_pinned_cpu_candidate() {")
        .unwrap()
        .1
        .split_once("if [[ -n \"$ORACLE_SERVER\" ]]")
        .unwrap()
        .0;
    let consumption = source
        .split_once("# The canary explicitly produces")
        .unwrap()
        .1
        .split_once("MEDIA_PATH=\"\"")
        .unwrap()
        .0;
    format!(
        "require_pinned_cpu_candidate() {{{candidate}\n# The canary explicitly produces{consumption}"
    )
}

const CPU_FIXTURE: &str = r#"
set -euo pipefail
ROOT=selected-source CANDIDATE_BIN_DIR=candidate ORACLE_SERVER=oracle ORACLE_COMPLETION= ORACLE_TTS=
CANDIDATE_BUILD_DIR="$FIXTURE_ROOT/native"
TEST_COMMAND=(test)
write_stamp() {
  printf 'patched-sha=%s\nbackend=%s\nlink-mode=static\ncmake-arg=-DGGML_METAL=OFF\n' "$1" "$2" > "$CANDIDATE_BUILD_DIR/.mesh-llm-build-stamp"
}
workload_owner() {
  [[ "$*" == 'automation canary-receipts prepared-source --root selected-source' ]] || exit 91
  printf 'provenance\n' >> "$FIXTURE_ROOT/events"
  [[ "$PROVENANCE_FAIL" == 0 ]] || return 1
  printf 'current\n'
}
workload_automation=(workload_owner)
python3() {
  [[ "$1" == selected-source/scripts/check-skippy-workload-candidate.py ]] || exit 92
  shift
  [[ "$#" == 4 || "$#" == 6 ]] || exit 94
  [[ "$1" == --candidate-binary && "$2" == candidate/skippy-server &&
     "$3" == --native-build-dir && "$4" == "$CANDIDATE_BUILD_DIR" ]] || exit 95
  shift 4
  if (( $# > 0 )); then
    [[ "$1" == --producer-manifest && "$2" == manifest ]] || exit 96
  fi
  printf 'checked\n' >> "$FIXTURE_ROOT/events"
  [[ "$CHECK_FAIL" == 0 ]]
}
jq() { printf 'prebuilt-test\n'; }
cargo() {
  [[ "$*" == 'build -p skippy-server' ]] || exit 93
  printf 'built\n' >> "$FIXTURE_ROOT/events"
  if [[ "$BUILT" == metal ]]; then write_stamp current metal; else write_stamp "$BUILT" cpu; fi
}
if [[ -n "$INITIAL" ]]; then write_stamp "$INITIAL" cpu; fi
"#;

#[test]
fn cpu_stamp_admission_follows_build_and_never_weakens_prebuilt_consumption() {
    for (initial, built, skip, producer, success) in [
        ("", "current", 0, "", true),
        ("stale", "current", 0, "", true),
        ("", "stale", 0, "", false),
        ("", "metal", 0, "", false),
        ("current", "current", 1, "", false),
        ("current", "current", 1, "manifest", true),
        ("stale", "current", 1, "manifest", false),
    ] {
        let fixture = Fixture::new();
        let mut command = fixture.command();
        command
            .args(["-c", &format!("{CPU_FIXTURE}\n{}", candidate_branches())])
            .env("INITIAL", initial)
            .env("BUILT", built)
            .env("SKIP_BUILD", skip.to_string())
            .env("PRODUCER_MANIFEST", producer)
            .env("PROVENANCE_FAIL", "0")
            .env("CHECK_FAIL", "0");
        let receipt = fixture.run(command);
        assert_eq!(
            receipt.status.success(),
            success,
            "{initial}/{built}/{skip}/{producer}: {}",
            receipt.stderr
        );
        let events = fs::read_to_string(fixture.path().join("events")).unwrap_or_default();
        assert_eq!(
            events.lines().filter(|line| *line == "built").count(),
            usize::from(skip == 0)
        );
        if success && skip == 0 {
            assert_eq!(
                events.lines().collect::<Vec<_>>(),
                ["built", "provenance", "checked"]
            );
        } else if success {
            assert_eq!(
                events.lines().collect::<Vec<_>>(),
                ["checked", "provenance", "checked"]
            );
        }
    }
}

#[test]
fn provenance_and_candidate_verifier_errors_cannot_admit_cpu_outputs() {
    for (provenance_fail, check_fail) in [("1", "0"), ("0", "1")] {
        let fixture = Fixture::new();
        let mut command = fixture.command();
        command
            .args(["-c", &format!("{CPU_FIXTURE}\n{}", candidate_branches())])
            .env("INITIAL", "current")
            .env("BUILT", "current")
            .env("SKIP_BUILD", "1")
            .env("PRODUCER_MANIFEST", "manifest")
            .env("PROVENANCE_FAIL", provenance_fail)
            .env("CHECK_FAIL", check_fail);
        let receipt = fixture.run(command);
        assert!(!receipt.status.success());
        assert!(
            !fs::read_to_string(fixture.path().join("events"))
                .unwrap()
                .contains("built")
        );
    }
}
