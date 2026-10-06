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

pub(super) fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
pub(super) fn source() -> String {
    fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap()
}
pub(super) fn declaration(source: &str, name: &str) -> String {
    let marker = format!("\n{name}() {{\n");
    assert_eq!(source.matches(&marker).count(), 1);
    let body = source
        .split_once(&marker)
        .unwrap()
        .1
        .split_once("\n}\n")
        .unwrap()
        .0;
    format!("{name}() {{\n{body}\n}}\n")
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
pub(super) struct Fixture {
    temporary: tempfile::TempDir,
    pub(super) root: PathBuf,
}
impl Fixture {
    pub(super) fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        for directory in [
            "tools",
            "tmp",
            "scripts",
            "third_party/llama.cpp",
            ".deps/llama.cpp",
            "native/src",
        ] {
            fs::create_dir_all(root.join(directory)).unwrap();
        }
        fs::write(
            root.join("third_party/llama.cpp/upstream.txt"),
            format!("{}\n", "a".repeat(40)),
        )
        .unwrap();
        fs::write(root.join("native/src/libllama.a"), b"inert archive bytes").unwrap();
        let path = std::env::var_os("PATH").expect("prepared jq prerequisite");
        let jq = std::env::split_paths(&path)
            .map(|directory| directory.join("jq"))
            .find(|candidate| candidate.is_file())
            .expect("prepared jq prerequisite")
            .canonicalize()
            .unwrap();
        // Execute the prepared jq in its original platform context. Relocating
        // an Apple arm64e system binary into a fixture can break its execution.
        let quoted = jq.to_str().unwrap().replace('\'', "'\\''");
        executable(
            &root.join("tools/jq"),
            &format!("#!/bin/bash\nexec '{quoted}' \"$@\"\n"),
        );
        Self { temporary, root }
    }
    pub(super) fn tool(&self, path: &str, label: &str, tail: &str) {
        assert!(
            label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        );
        executable(
            &self.root.join(path),
            &format!(
                r#"#!/bin/bash
set -euo pipefail
printf '%s\n' '{label}' >> "$INERT_ROOT/order"
count_file="$INERT_ROOT/{label}.count"
count=0
if [[ -f "$count_file" ]]; then read -r count < "$count_file"; fi
count="$(( count + 1 ))"
printf '%s\n' "$count" > "$count_file"
printf '%s\0' "$@" > "$INERT_ROOT/{label}-$count.args"
printf '%s\0' "$@" > "$INERT_ROOT/{label}.args"
if [[ "$INERT_FAIL" == '{label}' ]]; then exit 17; fi
{tail}
"#
            ),
        );
    }
    pub(super) fn build_tools(&self) {
        for (path, label) in [
            (
                "scripts/check-skippy-generated-family-patch.sh",
                "generated",
            ),
            ("tools/cargo", "cargo"),
            ("scripts/skippy-ci-smoke.sh", "smoke"),
            ("tools/just", "oracles"),
        ] {
            self.tool(path, label, "");
        }
        self.tool(
            "tools/uv",
            "uv",
            r#"printf '%s' "$LLAMA_STAGE_UPSTREAM_TESTS" > "$INERT_ROOT/upstream-tests""#,
        );
        self.tool("tools/lipo", "archive", "printf '%s\\n' \"$INERT_ARCHIVE\"");
        self.tool("scripts/skippy-system-one-smoke.sh", "systemone", r#"printf '%s\0' "$WORK_DIR" "$SYSTEMONE_SMOKE_CADENCE" "$SYSTEMONE_SMOKE_BUILD_BACKEND" "$SYSTEMONE_SMOKE_CERTIFIED_BACKENDS" "$SYSTEMONE_SMOKE_REQUIRE_QUALIFIED" > "$INERT_ROOT/systemone.environment""#);
        self.tool("scripts/skippy-laya-smoke.sh", "laya", r#"printf '%s\0' "$WORK_DIR" "$LAYA_SMOKE_CADENCE" "$LAYA_SMOKE_DEVICE" > "$INERT_ROOT/laya.environment""#);
    }
    pub(super) fn run(
        &self,
        mode: &str,
        names: &[&str],
        setup: &str,
        invocation: &str,
        extra: &[(&str, String)],
    ) -> process::RawProcessReport {
        let source = source();
        let selection = source
            .split_once("# Legacy workload automation selection begins.\n")
            .unwrap()
            .1
            .split_once("# Legacy workload automation selection ends.")
            .unwrap()
            .0;
        let admission_marker = "if [[ ! \"$HARNESS_MODE\" =~ ^(repair|verify|repair-build|verify-build|pinned-build)$ ]]; then\n";
        let admission_body = source
            .split_once(admission_marker)
            .unwrap()
            .1
            .split_once("\nfi\n")
            .unwrap()
            .0;
        let admission = format!("{admission_marker}{admission_body}\nfi\n");
        let declarations = names
            .iter()
            .map(|name| declaration(&source, name))
            .collect::<String>();
        let script = self.root.join("gate.sh");
        fs::write(
            &script,
            format!(
                r#"set -euo pipefail
ROOT="$PWD"
TRUSTED_ROOT="$PWD"
PIN_FILE="$ROOT/third_party/llama.cpp/upstream.txt"
UPSTREAM_SHA={sha}
LLAMA_STAGE_BUILD_DIR="$ROOT/native"
STATE_DIR="$ROOT"
SYSTEMONE_SMOKE_DIR="$ROOT/target/skippy-system-one-smoke"
PREPARE_LOG="$ROOT/prepare.log"
BUILD_LOG="$ROOT/build.log"
CERTIFY_LOG="$ROOT/certify.log"
FAMILY_BATTERY_RUN_ID=fixture
PLAN_PATH="$ROOT/plan.json"
VERIFICATION_DEADLINE_AT="$(( $(date +%s) + 30 ))"
{admission}
{selection}
{declarations}
{setup}
{invocation}
"#,
                sha = "a".repeat(40)
            ),
        )
        .unwrap();
        let mut environment = BTreeMap::from([
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
            ("HARNESS_MODE".into(), Value::Public(mode.into())),
            (
                "INERT_ROOT".into(),
                Value::Public(self.root.clone().into_os_string()),
            ),
            ("INERT_FAIL".into(), Value::Public("none".into())),
            ("INERT_ARCHIVE".into(), Value::Public("arm64".into())),
            ("GIT_MASTER".into(), Value::Public("1".into())),
            ("GIT_OPTIONAL_LOCKS".into(), Value::Public("0".into())),
        ]);
        for (key, value) in extra {
            environment.insert((*key).into(), Value::Public(value.into()));
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![Value::Public(script.into_os_string())],
                cwd: self.root.clone(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(40),
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
        assert_eq!(fs::read_dir(self.root.join("tmp")).unwrap().count(), 0);
        report
    }
    pub(super) fn diagnostic(&self, report: &process::RawProcessReport) -> String {
        let mut output = format!(
            "stdout={}\nstderr={}",
            String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()),
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
        );
        for name in ["prepare.log", "build.log", "certify.log"] {
            if let Ok(bytes) = fs::read(self.root.join(name)) {
                output.push_str(&format!("\n{name}={}", String::from_utf8_lossy(&bytes)));
            }
        }
        output
    }
    pub(super) fn args(&self, label: &str) -> Vec<String> {
        let bytes = fs::read(self.root.join(format!("{label}.args"))).unwrap();
        assert_eq!(bytes.last(), Some(&0));
        bytes[..bytes.len() - 1]
            .split(|byte| *byte == 0)
            .map(|arg| String::from_utf8(arg.to_vec()).unwrap())
            .collect()
    }
    pub(super) fn values(&self, file: &str) -> Vec<String> {
        let bytes = fs::read(self.root.join(file)).unwrap();
        assert_eq!(bytes.last(), Some(&0));
        bytes[..bytes.len() - 1]
            .split(|byte| *byte == 0)
            .map(|arg| String::from_utf8(arg.to_vec()).unwrap())
            .collect()
    }
    pub(super) fn order(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("order"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
    pub(super) fn finish(self) {
        self.temporary.close().expect("owned gate fixture cleanup");
    }
}
