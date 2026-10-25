//! Execute the actual wrapper's main tail with bounded fixture gate functions.
//! The destructive/native preparatory prefix is deliberately not executed.
use crate::process::{
    Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value, supervise,
};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};

fn fixture(prefix: &str) -> crate::process::ProcessReport {
    let wrapper = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/llama-canary-agent-repair.sh");
    let source = fs::read_to_string(wrapper).unwrap();
    let marker = "\nif ! check_family_cache; then\n";
    let start = source
        .find(marker)
        .expect("real production wrapper must contain its main dispatch");
    let directory = tempfile::tempdir().unwrap();
    let script = directory.path().join("main-fixture.sh");
    fs::write(
        &script,
        format!("set -euo pipefail\n{prefix}\n{}", &source[start..]),
    )
    .unwrap();
    let spec = ProcessSpec {
        executable: "/bin/bash".into(),
        cwd: directory.path().to_path_buf(),
        arguments: vec![Value::Public(script.into())],
        environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
    };
    let limits = Limits {
        execution: Duration::from_secs(3),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap()
}

#[test]
fn actual_main_rejects_cache_failure_with_infrastructure_status() {
    let report = fixture(
        "check_family_cache() { return 1; }\nrecord_failure_class() { printf 'class=%s stage=%s\\n' \"$1\" \"$2\" >&2; }",
    );
    assert!(report.cleanup.complete);
    assert_eq!(report.status.unwrap().code(), Some(125));
    let stderr = String::from_utf8(report.stderr.bytes_retained).unwrap();
    assert!(stderr.contains("class=infrastructure stage=model-cache"));
    assert!(stderr.contains("candidate source was not evaluated"));
}

#[test]
fn actual_pinned_main_reaches_gate_and_preserves_its_failure() {
    let report = fixture(
        "check_family_cache() { return 0; }\nHARNESS_MODE=pinned-build\nBASE_HEAD=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nVERIFICATION_TIMEOUT_SECONDS=1\nrun_candidate_gates() { echo entered-real-main-gate; return 73; }",
    );
    assert!(report.cleanup.complete);
    assert_eq!(report.status.unwrap().code(), Some(73));
    assert!(
        String::from_utf8(report.stdout.bytes_retained)
            .unwrap()
            .contains("entered-real-main-gate")
    );
}
