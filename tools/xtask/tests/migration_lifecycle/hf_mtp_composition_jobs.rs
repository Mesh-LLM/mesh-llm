//! Actual composition worker using existing inert build/product/publication/export helpers.
#![cfg(target_os = "linux")]
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value as Arg,
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn fixture(native: &str, publisher: &str) -> (tempfile::TempDir, tempfile::TempDir, Value) {
    let (temp, input, _, _) = super::hf_mtp_default_cli::fixture(native, publisher);
    let mut operator: Value = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    operator["credential_file"] = Value::Null;
    operator["bootstrap"]["timeout_seconds"] = json!(60);
    operator.as_object_mut().unwrap().remove("mtp_block");
    let (export, base) = super::hf_generic_jobs::fixture("ok", "ok");
    let request = json!({"schema_version":1,"workflow":"default-mtp-composition","timeout_secs":60,"runner":base["runner"],"operator":operator,"receipt_export":base["receipt_export"]});
    (temp, export, request)
}
fn invoke(
    root: &Path,
    input: &Value,
    label: &str,
    cancel: &Cancellation,
) -> process::RawProcessReport {
    let output = root.join(label);
    let request = root.join(format!("{label}-request.json"));
    std::fs::write(&request, serde_json::to_vec(input).unwrap()).unwrap();
    let raw = process::supervise_raw_with_files(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            environment: std::collections::BTreeMap::from([(
                "MESH_HF_PUBLICATION_TOKEN".into(),
                Arg::Secret("inert-explicit-token".into()),
            )]),
            arguments: [
                "automation",
                "hf-certify",
                "composition-job-worker",
                "--input",
                request.to_str().unwrap(),
                "--output-directory",
                output.to_str().unwrap(),
            ]
            .map(|a| Arg::Public(a.into()))
            .into(),
        },
        &Limits {
            execution: Duration::from_secs(50),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        Default::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1048576),
            stderr: std::num::NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(
        raw.process.failure.is_none()
            && raw.process.cleanup.complete
            && !raw.process.cleanup.forced
            && !raw.process.cleanup.graceful_signal_failed
            && raw.process.cleanup.failure.is_none()
    );
    for (bytes, stream) in [
        (&raw.stdout, &raw.process.stdout),
        (&raw.stderr, &raw.process.stderr),
    ] {
        assert!(!stream.truncated && stream.line_capture_complete);
        assert_eq!(stream.oversized_lines, 0);
        assert_eq!(
            bytes.as_ref().unwrap().as_bytes().len() as u64,
            stream.bytes_seen
        );
        assert!(
            !bytes
                .as_ref()
                .unwrap()
                .as_bytes()
                .windows(b"inert-explicit-token".len())
                .any(|s| s == b"inert-explicit-token")
        );
    }
    raw
}

fn receipt(root: &Path, label: &str, name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(root.join(label).join(name)).unwrap()).unwrap()
}
#[test]
fn actual_default_composition_jobs_worker_publishes_ordered_native_and_immutable_delivery() {
    let (temp, export, input) = fixture("ok", "ok");
    let root = temp.path().canonicalize().unwrap();
    let raw = invoke(&root, &input, "job", &Cancellation::default());
    assert_eq!(raw.process.outcome, Outcome::Exited);
    assert_eq!(raw.process.status.unwrap().code(), Some(0));
    let native = receipt(&root, "job", "native-job.json");
    let delivered = receipt(&root, "job", "native-job-delivery.json");
    assert_eq!(native["status"], "COMPOSITION_COMPLETED");
    assert_eq!(native["operator"]["status"], "COMPOSED_PUBLISHED");
    assert_eq!(
        native["operator"]["request_sha256"],
        native["operator_request_sha256"]
    );
    let publication = &native["operator"]["ordered_publication"]["final_receipt"]["publication"];
    assert_eq!(publication["completed"], true);
    assert_eq!(
        publication["ordered_paths"],
        json!([
            "Composite-00001-of-00003.gguf",
            "Composite-00002-of-00003.gguf",
            "Composite-00003-of-00003.gguf",
            "README.md"
        ])
    );
    assert_eq!(
        publication["remote_verified_paths"],
        publication["ordered_paths"]
    );
    assert_eq!(delivered["locator"]["delivery_complete"], true);
    assert_eq!(
        delivered["locator"]["artifact_sha256"],
        super::hf_mtp_default_cli::hash(&std::fs::read(root.join("job/native-job.json")).unwrap())
    );
    temp.close().unwrap();
    export.close().unwrap();
}
#[test]
fn actual_default_composition_jobs_worker_exports_failed_native_observations() {
    let (temp, export, input) = fixture("bad-verify", "ok");
    let root = temp.path().canonicalize().unwrap();
    let raw = invoke(&root, &input, "job", &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    let native = receipt(&root, "job", "native-job.json");
    let delivered = receipt(&root, "job", "native-job-delivery.json");
    assert_eq!(native["status"], "FAILED");
    assert_ne!(native["operator"]["bootstrap_phases"], json!([]));
    assert_eq!(delivered["status"], "FAILED");
    assert_eq!(delivered["locator"]["delivery_complete"], false);
    assert!(!native["operator"]["error"].is_null());
    temp.close().unwrap();
    export.close().unwrap();
}

#[test]
fn actual_default_composition_jobs_cancelled_publication_retains_partial_custody() {
    let (temp, export, input) = fixture("ok", "held");
    let root = temp.path().canonicalize().unwrap();
    let child_root = root.clone();
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let thread = std::thread::spawn(move || invoke(&child_root, &input, "job", &child_cancel));
    let marker = root.join("job/operator/publisher/publication/helper-output/publication-held");
    let until = Instant::now() + Duration::from_secs(30);
    while !marker.exists() && Instant::now() < until {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancel.cancel();
    let raw = thread.join().unwrap();
    assert!(observed);
    assert_eq!(raw.process.outcome, Outcome::Cancelled);
    let native = receipt(&root, "job", "native-job.json");
    let delivered = receipt(&root, "job", "native-job-delivery.json");
    assert_eq!(native["status"], "FAILED");
    assert_eq!(
        native["operator"]["native_composition"]["source_unchanged"],
        true
    );
    assert_eq!(
        native["operator"]["ordered_publication"]["partial_progress"]["publication"]["commit_attempted"],
        true
    );
    assert_eq!(delivered["status"], "FAILED");
    assert_ne!(delivered["locator"]["delivery_complete"], true);
    temp.close().unwrap();
    export.close().unwrap();
}
