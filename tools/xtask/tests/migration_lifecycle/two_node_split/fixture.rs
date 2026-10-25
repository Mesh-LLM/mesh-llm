use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, RawProcessReport,
    Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
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
pub(super) fn executable(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
pub(super) struct Fixture {
    _directory: tempfile::TempDir,
    pub root: PathBuf,
    source: String,
}
impl Fixture {
    pub fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("workspace with spaces");
        for name in ["bin", "home", "tmp", "private-process"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }
        for name in ["cargo", "just", "cmake"] {
            executable(
                &root.join("bin").join(name),
                "printf '%s\\n' \"$0\" >> \"$FORBIDDEN\"; exit 98",
            );
        }
        let source =
            fs::read_to_string(repository().join("scripts/ci-two-node-split-smoke.sh")).unwrap();
        Self {
            _directory: directory,
            root,
            source,
        }
    }
    pub fn functions(&self, ranges: &[(&str, &str)]) -> String {
        let mut script = "set -euo pipefail\nautomation=(\"$NATIVE_AUTOMATION\")\n".to_owned();
        for (first, following) in ranges {
            let start = self.source.find(&format!("{first}() {{")).unwrap();
            let end = self.source[start..]
                .find(&format!("{following}() {{"))
                .unwrap()
                + start;
            script.push_str(&self.source[start..end]);
        }
        script
    }
    pub fn section(&self, first: &str, following: &str) -> String {
        assert_eq!(self.source.matches(first).count(), 1);
        let start = self.source.find(first).unwrap();
        let rest = &self.source[start..];
        assert_eq!(rest.matches(following).count(), 1);
        rest[..rest.find(following).unwrap()].to_owned()
    }
    fn environment(&self, extra: &[(&str, String)]) -> BTreeMap<std::ffi::OsString, Value> {
        let values = [
            (
                "PATH",
                format!(
                    "{}:/usr/bin:/bin:/opt/homebrew/bin",
                    self.root.join("bin").display()
                ),
            ),
            ("HOME", self.root.join("home").display().to_string()),
            ("TMPDIR", self.root.join("tmp").display().to_string()),
            ("WORK_DIR", self.root.join("work").display().to_string()),
            (
                "FORBIDDEN",
                self.root.join("forbidden").display().to_string(),
            ),
            ("NATIVE_AUTOMATION", env!("CARGO_BIN_EXE_xtask").into()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
            (
                "MESH_TWO_NODE_SPLIT_PROCESS_ROOT",
                self.root.join("private-process").display().to_string(),
            ),
        ];
        values
            .into_iter()
            .chain(extra.iter().map(|(key, value)| (*key, value.clone())))
            .map(|(key, value)| (key.into(), Value::Public(value.into())))
            .collect()
    }
    pub fn capture(&self, arguments: Vec<Value>, extra: &[(&str, String)]) -> RawProcessReport {
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment: self.environment(extra),
                arguments,
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
                stdout: NonZeroUsize::new(1048576),
                stderr: NonZeroUsize::new(1048576),
            },
        )
        .unwrap();
        assert!(result.process.failure.is_none(), "{:?}", result.process);
        assert!(result.process.cleanup.complete, "{:?}", result.process);
        assert!(
            !self.root.join("forbidden").exists(),
            "consumer attempted an implicit build"
        );
        result
    }
    pub fn run(&self, script: String, extra: &[(&str, String)]) -> RawProcessReport {
        self.capture(
            vec![Value::Public("-c".into()), Value::Public(script.into())],
            extra,
        )
    }
    pub fn whole_adapter(&self, model: &Path, extra: &[(&str, String)]) -> RawProcessReport {
        self.capture(
            [
                repository()
                    .join("scripts/ci-two-node-split-smoke.sh")
                    .into_os_string(),
                self.root.join("missing-mesh-llm").into_os_string(),
                "/bin".into(),
                model.as_os_str().to_owned(),
            ]
            .into_iter()
            .map(Value::Public)
            .collect(),
            extra,
        )
    }
}
pub(super) fn stdout(result: &RawProcessReport) -> String {
    String::from_utf8(result.stdout.as_ref().unwrap().as_bytes().to_vec()).unwrap()
}
pub(super) fn stderr(result: &RawProcessReport) -> String {
    String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes()).into_owned()
}
