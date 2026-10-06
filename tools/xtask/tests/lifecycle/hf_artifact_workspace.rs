//! Actual native worker process; finite bytes establish staging, not model or hosted qualification.
#![cfg(target_os = "linux")]
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value as Arg,
};
use serde_json::{Value, json};
use std::{path::Path, time::Duration};
fn fixture() -> (tempfile::TempDir, Value) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let source = root.join("source");
    std::fs::create_dir(&source).unwrap();
    let mut files = Vec::new();
    for (name, bytes) in [
        ("README.md", b"card".to_vec()),
        (
            "skippy-convert-manifest.json",
            serde_json::to_vec(
                &json!({"expected_splits":2,"output_basename":"model","target_prefix":"BF16"}),
            )
            .unwrap(),
        ),
        ("model-00001-of-00002.gguf", b"GGUFone".to_vec()),
        ("model-00002-of-00002.gguf", b"GGUFtwo".to_vec()),
    ] {
        std::fs::write(source.join(name), &bytes).unwrap();
        files.push(json!({"name":name,"sha256":crate::automation::hf_certify::admission::digest(&bytes),"byte_size":bytes.len()}));
    }
    let request = json!({"schema_version":1,"repo":"fixture/converted","revision":"a".repeat(40),"source_directory":source,"work_directory":root.join("work"),"target_prefix":"BF16","output_basename":"model","requested_splits":1,"files":files,"timeout_seconds":10});
    (temp, request)
}
fn invoke_to(root: &Path, input: &Value, output: &Path) -> (process::RawProcessReport, Value) {
    let source = root.join("input.json");
    std::fs::write(&source, serde_json::to_vec(input).unwrap()).unwrap();
    let raw = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation",
                "hf-certify",
                "artifact-workspace-worker",
                "--input",
                source.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ]
            .map(|s| Arg::Public(s.into()))
            .into(),
            cwd: root.into(),
            environment: Default::default(),
        },
        &Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::new(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1048576),
            stderr: std::num::NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(
        raw.process.cleanup.complete
            && !raw.process.cleanup.forced
            && !raw.process.cleanup.graceful_signal_failed
            && raw.process.cleanup.failure.is_none()
    );
    for (stream, capture) in [
        (&raw.process.stdout, raw.stdout.as_ref()),
        (&raw.process.stderr, raw.stderr.as_ref()),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(capture.unwrap().as_bytes().len() as u64, stream.bytes_seen);
    }
    let receipt = std::fs::read(output)
        .map(|b| serde_json::from_slice(&b).unwrap())
        .unwrap_or(Value::Null);
    (raw, receipt)
}
#[test]
fn actual_upload_workspace_worker_effective_shards_and_literal_card_without_build() {
    let (temp, input) = fixture();
    let root = temp.path().canonicalize().unwrap();
    let (raw, receipt) = invoke(&root, &input);
    assert!(raw.process.status.unwrap().success());
    assert_eq!(receipt["completed"], true);
    assert_eq!(receipt["effective_splits"], 2);
    assert_eq!(receipt["built"], false);
    assert_eq!(receipt["converted"], false);
    for file in input["files"].as_array().unwrap() {
        let name = file["name"].as_str().unwrap();
        assert_eq!(
            std::fs::read(root.join("source").join(name)).unwrap(),
            std::fs::read(root.join("work/target/BF16").join(name)).unwrap()
        );
    }
    temp.close().unwrap();
}
#[test]
fn actual_upload_workspace_worker_pin_and_ancestry_refusals_preserve_mount() {
    for mode in ["pin", "overlap"] {
        let (temp, mut input) = fixture();
        let root = temp.path().canonicalize().unwrap();
        if mode == "pin" {
            input["files"][0]["sha256"] = json!("0".repeat(64));
        } else {
            input["work_directory"] = json!(root.join("source/work"));
        }
        let (raw, receipt) = invoke(&root, &input);
        assert!(!raw.process.status.unwrap().success());
        assert_eq!(receipt["completed"], false);
        assert!(!root.join("work").exists());
        assert!(!root.join("source/work").exists());
        assert_eq!(
            std::fs::read(root.join("source/README.md")).unwrap(),
            b"card"
        );
        temp.close().unwrap();
    }
}

fn invoke(root: &Path, input: &Value) -> (process::RawProcessReport, Value) {
    invoke_to(root, input, &root.join("receipt.json"))
}
#[test]
fn actual_upload_workspace_receipt_path_cannot_mutate_source_or_work() {
    for within in ["source", "work"] {
        let (temp, input) = fixture();
        let root = temp.path().canonicalize().unwrap();
        let output = if within == "source" {
            root.join("source/unpinned.json")
        } else {
            std::fs::create_dir(root.join("work")).unwrap();
            root.join("work/unpinned.json")
        };
        let before = std::fs::read_dir(root.join("source")).unwrap().count();
        let (raw, receipt) = invoke_to(&root, &input, &output);
        assert!(!raw.process.status.unwrap().success());
        assert!(receipt.is_null());
        assert!(!output.exists());
        assert_eq!(
            std::fs::read_dir(root.join("source")).unwrap().count(),
            before
        );
        assert_eq!(
            std::fs::read(root.join("source/README.md")).unwrap(),
            b"card"
        );
        if within == "work" {
            assert_eq!(std::fs::read_dir(root.join("work")).unwrap().count(), 0);
        } else {
            assert!(!root.join("work").exists());
        }
        temp.close().unwrap();
    }
}
