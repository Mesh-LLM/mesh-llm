//! Actual advisory download adapter, inert product process, and native output verifier.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    time::Duration,
};

struct Fixture {
    temporary: tempfile::TempDir,
    bin: PathBuf,
    scratch: PathBuf,
    source: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let bin = temporary.path().join("bin");
        let scratch = temporary.path().join("owned-temp");
        fs::create_dir(&bin).unwrap();
        fs::create_dir(&scratch).unwrap();
        for name in ["cat", "mkdir", "mktemp", "rm", "uname", "cp"] {
            let target = ["/bin", "/usr/bin"]
                .into_iter()
                .map(|root| std::path::Path::new(root).join(name))
                .find(|p| p.is_file())
                .unwrap();
            symlink(target, bin.join(name)).unwrap();
        }
        let source = temporary.path().join("source.gguf");
        fs::write(&source, vec![b'x'; 1024 * 1024]).unwrap();
        let fixture = Self {
            temporary,
            bin,
            scratch,
            source,
        };
        fixture.executable("sleep", "#!/bin/sh\nexit 0\n");
        fixture
    }
    fn executable(&self, name: &str, body: &str) -> PathBuf {
        let path = self.bin.join(name);
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }
    fn timeout(&self) {
        self.executable("timeout", "#!/bin/sh\n[ \"$1\" = --kill-after=10s ] && [ \"$2\" = 180s ] || exit 98\nshift 2\nexec \"$@\"\n");
    }
    fn run(&self, binary: PathBuf) -> (i32, String, String) {
        let root = self.temporary.path().canonicalize().unwrap();
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/ci-hf-xet-portability-smoke.sh")
            .canonicalize()
            .unwrap();
        let environment: BTreeMap<_, _> = [
            ("PATH", self.bin.clone().into_os_string()),
            ("TMPDIR", self.scratch.clone().into_os_string()),
            ("FIXTURE_SOURCE", self.source.clone().into_os_string()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
        ]
        .into_iter()
        .map(|(name, value)| (name.into(), Value::Public(value)))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: root,
                arguments: vec![
                    Value::Public(script.into_os_string()),
                    Value::Public(binary.into_os_string()),
                ],
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
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
        assert!(report.process.failure.is_none() && report.process.cleanup.complete);
        assert!(
            fs::read_dir(&self.scratch).unwrap().next().is_none(),
            "adapter temporary cache leaked"
        );
        assert_eq!(fs::read(&self.source).unwrap(), vec![b'x'; 1024 * 1024]);
        (
            report.process.status.unwrap().code().unwrap(),
            String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes()).into_owned(),
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
        )
    }
}

#[test]
fn missing_timeout_retains_advisory_skip_without_launching_product() {
    let fixture = Fixture::new();
    let binary = fixture.executable(
        "mesh-llm",
        "#!/bin/sh\n: > \"$FIXTURE_SOURCE.product-called\"\nexit 94\n",
    );
    let (code, _, stderr) = fixture.run(binary);
    assert_eq!(code, 0, "{stderr}");
    assert!(stderr.contains("advisory smoke skipped"));
    assert!(
        !fixture
            .source
            .with_extension("gguf.product-called")
            .exists()
    );
}

#[test]
fn transient_download_failures_retry_then_verify_native_isolated_fixture() {
    let fixture = Fixture::new();
    fixture.timeout();
    let binary = fixture.executable(
        "mesh-llm",
        r#"#!/bin/sh
count_file="$MESH_LLM_DATA_DIR/attempts"
count=0
if [ -f "$count_file" ]; then IFS= read -r count < "$count_file"; fi
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
if [ "$count" -lt 3 ]; then printf transient >&2; exit 1; fi
cp "$FIXTURE_SOURCE" "$HF_HOME/fixture.gguf"
printf '{"path":"%s"}\n' "$HF_HOME/fixture.gguf"
"#,
    );
    let (code, stdout, stderr) = fixture.run(binary);
    assert_eq!(code, 0, "{stderr}");
    for attempt in [1, 2, 3] {
        assert!(stderr.contains(&format!("attempt {attempt}/3")), "{stderr}");
    }
    assert!(stdout.contains("Xet portability smoke passed:"), "{stdout}");
    assert!(stdout.contains("(1048576 bytes)"), "{stdout}");
}

#[test]
fn sigill_retains_hard_failure_and_does_not_retry() {
    let fixture = Fixture::new();
    fixture.timeout();
    let binary = fixture.executable("mesh-llm", "#!/bin/sh\nulimit -c 0\nkill -ILL $$\n");
    let (code, stdout, stderr) = fixture.run(binary);
    assert_eq!(code, 132, "{stderr}");
    assert!(stderr.contains("SIGILL"));
    assert!(stderr.contains("attempt 1/3"));
    assert!(!stderr.contains("attempt 2/3"));
    assert!(!stdout.contains("Xet portability smoke passed:"));
}
