//! Actual native frontend refusal before acquisition using pinned inert children.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
#[test]
fn actual_prefetch_cli_native_reader_refusal_precedes_any_acquisition_and_preserves_sources() {
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    let config = repo.join("skippy/evals/skippy-competitive-benchmark.json");
    let manifest = repo.join("ci/model-artifacts/manifests/competitive-benchmark.json");
    let original_config = std::fs::read(&config).unwrap();
    let original_manifest = std::fs::read(&manifest).unwrap();
    let helper = root.join("helper");
    let reader = root.join("reader");
    let helper_bytes = b"#!/bin/sh\nprintf invoked > acquisition-invoked\nexit 97\n";
    let reader_bytes = b"#!/bin/sh\nprintf 'wrong reader protocol\\n'\n";
    for (path, bytes) in [
        (&helper, helper_bytes.as_slice()),
        (&reader, reader_bytes.as_slice()),
    ] {
        std::fs::write(path, bytes).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let input = root.join("input.json");
    let output = root.join("materialized");
    let evidence = root.join("evidence");
    std::fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"config":config,"config_sha256":hash(&original_config),"model_manifest":manifest,"model_manifest_sha256":hash(&original_manifest),"model_keys":["llama32-dense"],"output_directory":output,"timeout_seconds":20,"maximum_bytes":1024,"credential_file":null,"export_sha256":{},"semantic_cases":{}})).unwrap()).unwrap();
    let args = vec![
        "automation".into(),
        "replay-matrix".into(),
        "competitive-inputs-prefetch".into(),
        "--request".into(),
        input.to_str().unwrap().into(),
        "--helper".into(),
        helper.to_str().unwrap().into(),
        "--helper-sha256".into(),
        hash(helper_bytes),
        "--reader".into(),
        reader.to_str().unwrap().into(),
        "--reader-sha256".into(),
        hash(reader_bytes),
        "--evidence-directory".into(),
        evidence.to_str().unwrap().into(),
        "--timeout-seconds".into(),
        "20".into(),
    ];
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: args
                .into_iter()
                .map(|v: String| Value::Public(v.into()))
                .collect(),
            cwd: root.clone(),
            environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
        },
        &Limits {
            execution: Duration::from_secs(24),
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
    let p = &report.process;
    assert_eq!(p.outcome, Outcome::Exited);
    assert!(!p.status.unwrap().success());
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    for (s, b) in [
        (&p.stdout, report.stdout.as_ref().unwrap()),
        (&p.stderr, report.stderr.as_ref().unwrap()),
    ] {
        assert!(s.line_capture_complete && !s.truncated);
        assert_eq!(s.bytes_seen, b.as_bytes().len() as u64);
    }
    assert!(!output.exists());
    assert!(!evidence.join("acquisition-invoked").exists());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(evidence.join("prefetch.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "FAILED");
    assert!(
        !receipt["model_authority"]["resolved"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(receipt["reader_preflight_clean"], false);
    assert!(
        receipt["error"]
            .as_str()
            .unwrap()
            .contains("reader preflight")
    );
    assert_eq!(std::fs::read(config).unwrap(), original_config);
    assert_eq!(std::fs::read(manifest).unwrap(), original_manifest);
    temp.close().unwrap();
}
