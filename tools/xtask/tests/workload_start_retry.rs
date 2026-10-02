//! Behavioral retry boundaries of the actual workload candidate launcher.
#![cfg(unix)]
use serde_json::Value;
use std::{
    fs::{self, File},
    os::unix::{fs::PermissionsExt as _, process::CommandExt as _},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Fixture {
    state: tempfile::TempDir,
    script: String,
}

fn function(source: &str, name: &str) -> String {
    let marker = format!("{name}() {{\n");
    let body = source
        .split_once(&marker)
        .unwrap()
        .1
        .split_once("\n}\n")
        .unwrap()
        .0;
    format!("{marker}{body}\n}}\n")
}

fn executable(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = fs::read_to_string(root.join("scripts/skippy-workload-certify.sh")).unwrap();
        let selector = source
            .split_once("# Frozen automation selection begins.\n")
            .unwrap()
            .1
            .split_once("# Frozen automation selection ends.")
            .unwrap()
            .0;
        let config = source
            .split_once("\"${workload_automation[@]}\" automation workload-smoke-config \\\n")
            .unwrap()
            .1
            .split_once("\nSERVER_LOG=")
            .unwrap()
            .0;
        let script = format!(
            "set -euo pipefail\n{selector}\n\"${{workload_automation[@]}}\" automation workload-smoke-config \\\n{config}\n{}\nwait_for_workload_server() {{ wait \"$1\"; }}\n{}\nstart_candidate_server\n",
            function(&source, "address_in_use_log"),
            function(&source, "start_candidate_server")
        );
        let state = tempfile::tempdir().unwrap();
        let fixture = Self { state, script };
        fs::create_dir(fixture.path().join("bin")).unwrap();
        executable(
            &fixture.path().join("bin/owner with spaces"),
            r#"#!/bin/bash
if [[ "$1" == automation && "$2" == local-ports && "$3" == 1 && $# == 3 ]]; then
  if [[ -f "$FIXTURE_ROOT/ports" ]]; then port=41002; else port=41001; fi
  printf '%s\n' "$port" >> "$FIXTURE_ROOT/ports"
  printf '%s\n' "$port"
elif [[ "$1" == automation && "$2" == workload-smoke-config ]]; then
  exec "$ACTUAL_AUTOMATION" "$@"
else
  echo unexpected-automation-boundary >&2
  exit 93
fi
"#,
        );
        executable(
            &fixture.path().join("bin/skippy-server"),
            r#"#!/bin/bash
printf '%s\0' "$@" >> "$FIXTURE_ROOT/candidate-argv"
while (( $# > 0 )); do
  if [[ "$1" == --bind-addr ]]; then printf '%s\n' "$2" >> "$FIXTURE_ROOT/calls"; fi
  shift
done
if [[ $(wc -l < "$FIXTURE_ROOT/calls") -eq 1 ]]; then
  printf '%s\n' "$FAILURE_MESSAGE" >&2
  exit 1
fi
printf 'fixture candidate started\n'
"#,
        );
        fixture
    }

    fn path(&self) -> &Path {
        self.state.path()
    }

    fn run(&self, failure: &str) -> ExitStatus {
        let child = Command::new("/bin/bash")
            .args(["-c", &self.script])
            .env("ROOT", self.path())
            .env("FIXTURE_ROOT", self.path())
            .env("ACTUAL_AUTOMATION", env!("CARGO_BIN_EXE_xtask"))
            .env(
                "MESH_LLM_AUTOMATION_BIN",
                self.path().join("bin/owner with spaces"),
            )
            .env("CANDIDATE_BIN_DIR", self.path().join("bin"))
            .env("CONFIG_PATH", self.path().join("stage config.json"))
            .env("SERVER_LOG", self.path().join("server.log"))
            .env("MODEL_ID", "captured model id")
            .env("MODEL_PATH", "/selected/model with spaces.gguf")
            .env("MODEL_SHA256", "a".repeat(64))
            .env("LAYER_END", "8")
            .env("N_GPU_LAYERS", "0")
            .env("PROJECTOR_PATH", "")
            .env("BACKEND", "cpu")
            .env("PORT", "")
            .env("PORT_START_ATTEMPTS", "3")
            .env("FAILURE_MESSAGE", failure)
            .env("PATH", "/usr/bin:/bin")
            .stdin(Stdio::null())
            .stdout(File::create(self.path().join("stdout")).unwrap())
            .stderr(File::create(self.path().join("stderr")).unwrap())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut owned = OwnedGroup {
            child,
            finished: false,
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = owned.child.try_wait().unwrap() {
                owned.finished = true;
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "workload retry fixture deadline: {}",
                fs::read_to_string(self.path().join("stderr")).unwrap()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn assert_launch_arguments(&self, ports: &[u16]) {
        let bytes = fs::read(self.path().join("candidate-argv")).unwrap();
        let actual: Vec<&str> = bytes
            .split(|byte| *byte == 0)
            .filter(|word| !word.is_empty())
            .map(|word| std::str::from_utf8(word).unwrap())
            .collect();
        let mut expected = Vec::new();
        for port in ports {
            expected.extend([
                "serve-openai".to_owned(),
                "--config".to_owned(),
                self.path()
                    .join("stage config.json")
                    .to_str()
                    .unwrap()
                    .to_owned(),
                "--bind-addr".to_owned(),
                format!("127.0.0.1:{port}"),
                "--default-max-tokens".to_owned(),
                "128".to_owned(),
                "--telemetry-level".to_owned(),
                "off".to_owned(),
            ]);
        }
        assert_eq!(actual, expected);
    }

    fn assert_config(&self) {
        let config: Value =
            serde_json::from_slice(&fs::read(self.path().join("stage config.json")).unwrap())
                .unwrap();
        assert_eq!(config["model_id"], "captured model id");
        assert_eq!(config["model_path"], "/selected/model with spaces.gguf");
        assert_eq!(config["source_model_sha256"], "a".repeat(64));
        assert_eq!(config["layer_end"], 8);
        assert_eq!(config["n_gpu_layers"], 0);
    }
}

struct OwnedGroup {
    child: Child,
    finished: bool,
}

impl Drop for OwnedGroup {
    fn drop(&mut self) {
        if !self.finished {
            let _kill = Command::new("/bin/kill")
                .args(["-KILL", "--", &format!("-{}", self.child.id())])
                .status();
            let _reap = self.child.wait();
        }
    }
}

#[test]
fn candidate_start_retries_only_address_in_use_with_fresh_port_and_retained_history() {
    let fixture = Fixture::new();
    let failure = "Address already in use (os error 48)";
    assert!(fixture.run(failure).success());
    fixture.assert_config();
    fixture.assert_launch_arguments(&[41001, 41002]);
    assert_eq!(
        fs::read_to_string(fixture.path().join("calls")).unwrap(),
        "127.0.0.1:41001\n127.0.0.1:41002\n"
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("ports")).unwrap(),
        "41001\n41002\n"
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("server.log.attempt-1")).unwrap(),
        format!("{failure}\n")
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("server.log")).unwrap(),
        "fixture candidate started\n"
    );
    assert!(!fixture.path().join("server.log.attempt-2").exists());
    assert!(
        fs::read_to_string(fixture.path().join("stderr"))
            .unwrap()
            .contains("retrying with a fresh port (1/3)")
    );
}

#[test]
fn candidate_start_does_not_retry_other_startup_failures() {
    let fixture = Fixture::new();
    assert_eq!(fixture.run("model initialization failed").code(), Some(1));
    fixture.assert_config();
    fixture.assert_launch_arguments(&[41001]);
    assert_eq!(
        fs::read_to_string(fixture.path().join("calls")).unwrap(),
        "127.0.0.1:41001\n"
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("ports")).unwrap(),
        "41001\n"
    );
    assert_eq!(
        fs::read_to_string(fixture.path().join("server.log")).unwrap(),
        "model initialization failed\n"
    );
    assert!(!fixture.path().join("server.log.attempt-1").exists());
    assert!(!fixture.path().join("server.log.attempt-2").exists());
    assert!(
        fs::read_to_string(fixture.path().join("stderr"))
            .unwrap()
            .is_empty()
    );
}
