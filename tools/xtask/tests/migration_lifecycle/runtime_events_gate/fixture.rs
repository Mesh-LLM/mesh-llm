use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
pub(super) fn invoke(
    executable: PathBuf,
    cwd: PathBuf,
    arguments: Vec<String>,
    environment: BTreeMap<std::ffi::OsString, Value>,
) -> process::RawProcessReport {
    let report = process::supervise_raw(
        &ProcessSpec {
            executable,
            cwd,
            environment,
            arguments: arguments
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(8),
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
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    report
}
pub(super) struct Fixture {
    _temporary: tempfile::TempDir,
    pub directory: PathBuf,
    pub caller: PathBuf,
    script: PathBuf,
}
impl Fixture {
    pub fn new(mode: &str) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().canonicalize().unwrap();
        for name in [
            "source/scripts",
            "caller/bundle directory",
            "bin",
            "capture",
        ] {
            fs::create_dir_all(directory.join(name)).unwrap();
        }
        let caller = directory.join("caller");
        fs::write(
            caller.join("model file.gguf"),
            b"finite fixture; never loaded",
        )
        .unwrap();
        fs::write(directory.join("capture/mode"), mode).unwrap();
        let original = repository().join("scripts/ci-runtime-events-native-gate.sh");
        assert_ne!(
            fs::metadata(&original).unwrap().permissions().mode() & 0o111,
            0
        );
        let script = directory.join("source/scripts/ci-runtime-events-native-gate.sh");
        fs::copy(original, &script).unwrap();
        let cargo = directory.join("bin/cargo");
        fs::write(&cargo, r#"#!/bin/bash
set -euo pipefail
printf '%s\0' "$@" > "$CAPTURE/argv"
printf '%s\0' "$PWD" "$MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST" "$MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" "$MESH_LLM_RUNTIME_EVENTS_MODEL" "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE" > "$CAPTURE/environment"
[[ ! -s "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE" ]] || exit 88
cd "$CAPTURE"
case "$(cat mode)" in
  executed) printf 'executed\n' >> "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE" ;;
  blocked) printf 'blocked: finite fixture\n' >> "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE" ;;
  absent) : ;;
  failure) exit 7 ;;
  *) exit 99 ;;
esac
"#).unwrap();
        fs::set_permissions(cargo, fs::Permissions::from_mode(0o755)).unwrap();
        Self {
            _temporary: temporary,
            directory,
            caller,
            script,
        }
    }
    pub fn args(&self, relative: bool) -> Vec<String> {
        let path = |value: &str| {
            if relative {
                value.into()
            } else {
                self.caller.join(value).to_str().unwrap().into()
            }
        };
        vec![
            "--bundle-dir".into(),
            path("bundle directory"),
            "--model".into(),
            path("model file.gguf"),
            "--evidence".into(),
            path("nested evidence/result.txt"),
        ]
    }
    pub fn run(&self, arguments: Vec<String>) -> process::RawProcessReport {
        invoke(
            self.script.clone(),
            self.caller.clone(),
            arguments,
            BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public(
                        format!(
                            "{}:{}",
                            self.directory.join("bin").display(),
                            std::env::var("PATH").unwrap()
                        )
                        .into(),
                    ),
                ),
                (
                    "CAPTURE".into(),
                    Value::Public(self.directory.join("capture").into_os_string()),
                ),
            ]),
        )
    }
    pub fn captured(&self, name: &str) -> Vec<String> {
        fs::read(self.directory.join("capture").join(name))
            .unwrap()
            .split(|byte| *byte == 0)
            .filter(|value| !value.is_empty())
            .map(|value| String::from_utf8(value.to_vec()).unwrap())
            .collect()
    }
    pub fn evidence(&self) -> PathBuf {
        self.caller.join("nested evidence/result.txt")
    }
}
