use crate::process::{
    Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    supervise_raw,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) struct Fixture {
    _temporary: tempfile::TempDir,
    pub(super) root: PathBuf,
}
pub(super) struct Output {
    pub(super) code: i32,
    pub(super) stdout: String,
    pub(super) stderr: String,
}
pub(super) fn function(script: &str, name: &str) -> String {
    let source = fs::read_to_string(repository().join("skippy/scripts").join(script)).unwrap();
    let marker = format!("{name}() {{\n");
    assert_eq!(
        source.matches(&marker).count(),
        1,
        "unique maintained function required"
    );
    let start = source.find(&marker).unwrap();
    let end = start + source[start..].find("\n}\n").unwrap() + 3;
    source[start..end].to_owned()
}
fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
impl Fixture {
    pub(super) fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("workload fixture with spaces");
        fs::create_dir(&root).unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }
    pub(super) fn text(&self, file: &str) -> String {
        fs::read_to_string(self.root.join(file)).unwrap()
    }
    pub(super) fn write(&self, file: &str, text: &str) {
        fs::write(self.root.join(file), text).unwrap();
    }
    pub(super) fn run(&self, script: &str, extra: &[(&str, String)]) -> Output {
        let mut environment: BTreeMap<_, _> = [
            ("PATH", std::env::var("PATH").unwrap()),
            ("HOME", self.root.display().to_string()),
            ("ROOT", repository().display().to_string()),
            ("CERT_DIR", self.root.display().to_string()),
            (
                "RESULTS_JSONL",
                self.root.join("results.jsonl").display().to_string(),
            ),
            (
                "POLICY_PLAN_COPY",
                self.root.join("plan.json").display().to_string(),
            ),
            (
                "SUMMARY_TSV",
                self.root.join("summary.tsv").display().to_string(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        for (key, value) in extra {
            environment.insert((*key).into(), Value::Public(value.as_str().into()));
        }
        let report = supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: vec![
                    Value::Public("-c".into()),
                    Value::Public(format!("set -euo pipefail\n{script}").into()),
                ],
            },
            &Limits {
                execution: Duration::from_secs(12),
                graceful_shutdown: Duration::from_secs(1),
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
        assert!(report.process.failure.is_none(), "{:?}", report.process);
        assert!(report.process.cleanup.complete);
        Output {
            code: report.process.status.unwrap().code().unwrap(),
            stdout: String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap(),
            stderr: String::from_utf8(report.stderr.unwrap().as_bytes().to_vec()).unwrap(),
        }
    }
    pub(super) fn battery(&self, dry_run: bool) -> Output {
        let script = format!(
            "TOTAL=0; CERT_FAILURE_COUNT=0; FAILURES=()\n{}\ncert_timeout_for_startup() {{ printf 1800; }}\n{}\nrun_workload_certify first embedding /unused/model.gguf fixture rev 600 1024 embedding-smoke,embedding-oracle \"\"\nrun_workload_certify second rerank /unused/model.gguf fixture rev 900 1024 rerank-smoke,rerank-oracle \"\"\nprintf 'counts=%s,%s\\n' \"$TOTAL\" \"$CERT_FAILURE_COUNT\"\n",
            function("skippy-family-battery.sh", "slugify"),
            function("skippy-family-battery.sh", "run_workload_certify")
        );
        self.run(&script, &[("DRY_RUN", u8::from(dry_run).to_string())])
    }
}
