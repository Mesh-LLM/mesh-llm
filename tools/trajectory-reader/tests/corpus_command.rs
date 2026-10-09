//! Offline actual native corpus command, using independently hashed local artifacts.
use super::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn fixture(root: &Path, quota: usize) -> Vec<String> {
    let input =
        b"{\"buggy\":\"bad\",\"fixed\":\"good\"}\n{\"buggy\":\"bad2\",\"fixed\":\"good2\"}\n";
    fs::write(root.join("input.jsonl"), input).unwrap();
    let sha: String = Sha256::digest(input)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    fs::write(root.join("config.json"),json!({"schema_version":1,"seed":2458,"tiers":{"smoke":{}},"sources":[{"name":"fixture","dataset":"owner/data","config":"default","split":"train","revision":"a".repeat(40),"family":"coding_edit","adapter":"code_refinement","quota":{"smoke":quota}}]}).to_string()).unwrap();
    fs::write(root.join("artifacts.json"),json!({"schema_version":1,"sources":[{"dataset":"owner/data","revision":"a".repeat(40),"config":"default","split":"train","conversion_provenance":{"fixture":"native","source_revision":"a".repeat(40)},"artifacts":[{"path":"input.jsonl","format":"jsonl","sha256":sha,"bytes":input.len()}]}]}).to_string()).unwrap();
    [
        "corpus",
        "smoke",
        "--config",
        "config.json",
        "--artifact-manifest",
        "artifacts.json",
        "--out-root",
        "output",
    ]
    .map(str::to_owned)
    .to_vec()
}
fn run(root: &Path, args: Vec<String>) -> process::ProcessReport {
    let mut environment = BTreeMap::new();
    for name in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Value::Public(value));
        }
    }
    process::supervise(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_trajectory-reader"))
                .canonicalize()
                .unwrap(),
            cwd: root.to_owned(),
            arguments: args.into_iter().map(|s| Value::Public(s.into())).collect(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 8192,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        process::OutputFiles::default(),
    )
    .unwrap()
}
fn success(report: &process::ProcessReport) -> bool {
    report.success()
}
#[test]
fn native_prompt_command_corpus_publishes_repeatable_digest_bound_documents() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path(), 2);
    let report = run(root.path(), args.clone());
    assert!(success(&report), "{report:?}");
    assert!(report.cleanup.complete);
    let corpus = fs::read(root.path().join("output/smoke/corpus.jsonl")).unwrap();
    let metadata = fs::read(root.path().join("output/smoke/manifest.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&metadata).unwrap();
    assert_eq!(parsed["row_count"], 2);
    assert_eq!(parsed["sources"][0]["resolved_revision"], "a".repeat(40));
    let sha: String = Sha256::digest(&corpus)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(parsed["corpus_sha256"], sha);
    let report = run(root.path(), args.clone());
    assert!(!success(&report), "existing output must refuse: {report:?}");
    assert!(report.cleanup.complete);
    assert_eq!(
        fs::read(root.path().join("output/smoke/corpus.jsonl")).unwrap(),
        corpus
    );
    assert_eq!(
        fs::read(root.path().join("output/smoke/manifest.json")).unwrap(),
        metadata
    );
    let mut fresh = args.clone();
    *fresh.last_mut().unwrap() = "output-two".into();
    let report = run(root.path(), fresh);
    assert!(success(&report), "fresh output: {report:?}");
    assert!(report.cleanup.complete);
    assert_eq!(
        fs::read(root.path().join("output-two/smoke/corpus.jsonl")).unwrap(),
        corpus
    );
    fs::write(root.path().join("input.jsonl"), "{}\n").unwrap();
    let mut stale = args;
    *stale.last_mut().unwrap() = "output-three".into();
    let report = run(root.path(), stale);
    assert!(!success(&report));
    assert!(report.cleanup.complete);
    assert!(!root.path().join("output-three").exists());
    assert_eq!(
        fs::read(root.path().join("output/smoke/manifest.json")).unwrap(),
        metadata
    );
}
#[test]
fn native_prompt_command_corpus_refuses_unfilled_quota_without_publication() {
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path(), 3);
    let report = run(root.path(), args);
    assert!(!success(&report), "{report:?}");
    assert!(report.cleanup.complete);
    assert!(!root.path().join("output").exists());
}
#[cfg(unix)]
#[test]
fn native_prompt_command_corpus_refuses_fifo_metadata_without_waiting_for_writer() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = tempfile::tempdir().unwrap();
    let args = fixture(root.path(), 2);
    fs::remove_file(root.path().join("artifacts.json")).unwrap();
    let path = CString::new(root.path().join("artifacts.json").as_os_str().as_bytes()).unwrap();
    // This fixture owns the path and deliberately supplies no FIFO writer.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let report = run(root.path(), args);
    assert!(!success(&report));
    assert!(!matches!(report.outcome, process::Outcome::Deadline));
    assert!(report.cleanup.complete);
    assert!(!root.path().join("output").exists());
}
