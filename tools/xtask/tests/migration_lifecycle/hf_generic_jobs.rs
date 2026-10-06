//! Actual Linux generic worker with finite pinned tools and the existing inert regular exporter.
#![cfg(target_os = "linux")]
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value as Arg,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn pin(path: &Path) -> Value {
    json!({"path":path,"sha256":hex::encode(Sha256::digest(std::fs::read(path).unwrap()))})
}
pub(super) fn fixture(mode: &str, export_mode: &str) -> (tempfile::TempDir, Value) {
    let (temp, mut operator) = super::hf_generic_operator::fixture(mode);
    let root = temp.path().canonicalize().unwrap();
    operator["conversion"]["timeout_seconds"] = json!(40);
    let helper = root.join("generic-exporter");
    let test = std::env::current_exe().unwrap();
    let quote = |p: &Path| format!("'{}'", p.to_str().unwrap().replace('\'', "'\\''"));
    let script = format!(
        "#!/bin/sh\n[ \"$1\" = publish-regular-receipt ] || exit 41\nshift\n[ \"$1\" = --input ] || exit 42\nexport MESH_EXPORT_INPUT=\"$2\"\nshift 2\n[ \"$1\" = --output-directory ] || exit 43\nexport MESH_EXPORT_OUTPUT=\"$2\"\nshift 2\n[ \"$#\" = 0 ] || exit 44\nexport MESH_EXPORT_MODE={}\nexec {} --exact hf_job_worker_cli::regular_export_fixture_helper --ignored --nocapture\n",
        export_mode,
        quote(&test)
    );
    std::fs::write(&helper, script).unwrap();
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
    let runner = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .canonicalize()
        .unwrap();
    let request = json!({"schema_version":1,"workflow":"generic-conversion","timeout_secs":40,"runner":pin(&runner),"operator":operator,"receipt_export":{"helper":pin(&helper),"helper_source":pin(&helper),"repo":"fixture/evidence","parent_commit":"a".repeat(40),"credential_file":null,"credential_environment":true,"export_budget_secs":10,"path_in_repo":"runs/native-job.json"}});
    (temp, request)
}
fn invoke(
    root: &Path,
    input: &Value,
    label: &str,
    cancel: &Cancellation,
) -> process::RawProcessReport {
    let output = root.join(label);
    let raw = process::supervise_raw_with_files(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            environment: std::collections::BTreeMap::from([
                (
                    "MESH_HF_JOB_INPUT".into(),
                    Arg::Secret(serde_json::to_string(input).unwrap().into()),
                ),
                (
                    "MESH_HF_PUBLICATION_TOKEN".into(),
                    Arg::Secret("inert-explicit-token".into()),
                ),
            ]),
            arguments: [
                "automation",
                "hf-certify",
                "generic-job-worker",
                "--input-environment",
                "MESH_HF_JOB_INPUT",
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
fn generic_jobs_actual_worker_observes_g3_conversion_and_exact_immutable_delivery_hash() {
    for dry in [false, true] {
        let (temp, mut input) = fixture("ok", "ok");
        let root = temp.path().canonicalize().unwrap();
        input["operator"]["conversion"]["dry_run"] = json!(dry);
        let raw = invoke(&root, &input, "success", &Cancellation::default());
        assert_eq!(raw.process.outcome, Outcome::Exited);
        assert_eq!(raw.process.status.unwrap().code(), Some(0));
        let native = receipt(&root, "success", "native-job.json");
        let delivery = receipt(&root, "success", "native-job-delivery.json");
        assert_eq!(native["status"], "CONVERSION_COMPLETED");
        assert_eq!(native["operator"]["status"], "OPERATOR_COMPLETED");
        assert_eq!(
            native["operator"]["conversion_receipt"]["status"],
            if dry {
                "DRY_RUN_COMPLETED"
            } else {
                "LOCAL_ARTIFACT_READY"
            }
        );
        assert_eq!(
            native["operator"]["observed_bootstrap"]["mesh_commit"],
            "a".repeat(40)
        );
        assert_eq!(delivery["status"], "DELIVERED");
        assert_eq!(delivery["locator"]["delivery_complete"], true);
        assert_eq!(
            delivery["locator"]["artifact_sha256"],
            hex::encode(Sha256::digest(
                std::fs::read(root.join("success/native-job.json")).unwrap()
            ))
        );
        assert_eq!(
            delivery["locator"]["transport_input_sha256"],
            hex::encode(Sha256::digest(serde_json::to_vec(&input).unwrap()))
        );
        assert_eq!(native["image_observed"], false);
        assert_eq!(native["cost_observed"], false);
        temp.close().unwrap();
    }
}
#[test]
fn generic_jobs_actual_worker_exports_failed_native_observations_without_success() {
    for (mode, export) in [("wrong-head", "ok"), ("ok", "outer-error")] {
        let (temp, input) = fixture(mode, export);
        let root = temp.path().canonicalize().unwrap();
        let raw = invoke(&root, &input, "failed", &Cancellation::default());
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        let native = receipt(&root, "failed", "native-job.json");
        let delivery = receipt(&root, "failed", "native-job-delivery.json");
        assert_eq!(delivery["status"], "FAILED");
        if mode == "wrong-head" {
            assert_eq!(native["status"], "FAILED");
            assert_eq!(delivery["locator"]["delivery_complete"], false);
            assert!(
                !native["operator"]["bootstrap_phases"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            assert!(native["operator"]["conversion_receipt"].is_null());
        } else {
            assert_eq!(native["status"], "CONVERSION_COMPLETED");
            assert!(delivery["locator"].is_null());
        }
        temp.close().unwrap();
    }
}
#[test]
fn generic_jobs_actual_causal_build_cancellation_joins_owned_worker_and_retains_phases() {
    let (temp, input) = fixture("hold", "ok");
    let root = temp.path().canonicalize().unwrap();
    let cancellation = Cancellation::default();
    let child_cancel = cancellation.clone();
    let child_root = root.clone();
    let child = std::thread::spawn(move || invoke(&child_root, &input, "cancelled", &child_cancel));
    let marker = root.join("cancelled/operator/bootstrap/mesh-source/build-started");
    let until = Instant::now() + Duration::from_secs(20);
    while !marker.exists() && Instant::now() < until {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancellation.cancel();
    let raw = child.join().unwrap();
    assert!(observed && cancellation.is_cancelled());
    assert_eq!(raw.process.outcome, Outcome::Cancelled);
    let native = receipt(&root, "cancelled", "native-job.json");
    let delivery = receipt(&root, "cancelled", "native-job-delivery.json");
    assert_eq!(native["status"], "FAILED");
    assert_eq!(delivery["status"], "FAILED");
    assert!(delivery["locator"].is_null());
    assert!(
        !native["operator"]["bootstrap_phases"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    temp.close().unwrap();
}
#[test]
fn generic_jobs_actual_worker_refuses_source_output_overlap_and_unknown_workflow_before_tools() {
    for overlap in [false, true] {
        let (temp, mut input) = fixture("ok", "ok");
        let root = temp.path().canonicalize().unwrap();
        let source =
            Path::new(input["operator"]["conversion"]["source"].as_str().unwrap()).to_path_buf();
        let source_before = std::fs::read(source.join("config.json")).unwrap();
        let working = if overlap {
            source.clone()
        } else {
            root.clone()
        };
        if !overlap {
            input["workflow"] = json!("certification");
        }
        let raw = invoke(&working, &input, "refused", &Cancellation::default());
        assert_eq!(raw.process.outcome, Outcome::Exited);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        assert!(!working.join("refused").exists());
        assert_eq!(
            std::fs::read(source.join("config.json")).unwrap(),
            source_before
        );
        assert_eq!(std::fs::read_dir(&source).unwrap().count(), 1);
        temp.close().unwrap();
    }
}

#[test]
fn generic_jobs_actual_upload_only_mount_copies_effective_artifacts_without_bootstrap_or_convert() {
    let (temp, mut input) = fixture("wrong-head", "ok");
    let root = temp.path().canonicalize().unwrap();
    let source = root.join("mounted-converted");
    std::fs::create_dir(&source).unwrap();
    let mut files = Vec::new();
    for (name, bytes) in [
        ("README.md", b"original published card".to_vec()),
        (
            "skippy-convert-manifest.json",
            serde_json::to_vec(
                &json!({"expected_splits":2,"output_basename":"model","target_prefix":"BF16"}),
            )
            .unwrap(),
        ),
        ("model-00001-of-00002.gguf", b"GGUFfixture1".to_vec()),
        ("model-00002-of-00002.gguf", b"GGUFfixture2".to_vec()),
        (
            "skippy-convert-status.json",
            b"{\"phase\":\"complete\"}".to_vec(),
        ),
    ] {
        std::fs::write(source.join(name), &bytes).unwrap();
        files.push(json!({"name":name,"sha256":hex::encode(Sha256::digest(&bytes)),"byte_size":bytes.len()}));
    }
    let work = root.join("upload-work");
    let conversion = &mut input["operator"]["conversion"];
    conversion["source"] = json!(source);
    conversion["work_directory"] = json!(work);
    conversion["upload_only"] = json!(true);
    conversion["source_files"] = json!([]);
    conversion["target_prefix"] = json!("BF16");
    conversion["expected_splits"] = json!(1);
    input["upload_artifact"] = json!({"schema_version":1,"repo":"fixture/converted","revision":"a".repeat(40),"source_directory":source,"work_directory":work,"target_prefix":"BF16","output_basename":"model","requested_splits":1,"files":files,"timeout_seconds":40});
    let raw = invoke(&root, &input, "uploaded", &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(0));
    let native = receipt(&root, "uploaded", "native-job.json");
    let delivered = receipt(&root, "uploaded", "native-job-delivery.json");
    assert_eq!(native["status"], "CONVERSION_COMPLETED");
    assert_eq!(native["artifact_workspace"]["effective_splits"], 2);
    assert_eq!(native["artifact_workspace"]["built"], false);
    assert_eq!(native["artifact_workspace"]["converted"], false);
    assert!(native["operator"]["observed_bootstrap"].is_null());
    assert_eq!(
        native["operator"]["conversion_receipt"]["status"],
        "LOCAL_ARTIFACT_READY"
    );
    assert_eq!(delivered["locator"]["delivery_complete"], true);
    assert!(!root.join("uploaded/operator/bootstrap").exists());
    assert!(!work.join("argv").exists());
    for file in input["upload_artifact"]["files"].as_array().unwrap() {
        let name = file["name"].as_str().unwrap();
        assert_eq!(
            std::fs::read(source.join(name)).unwrap(),
            std::fs::read(work.join("target/BF16").join(name)).unwrap()
        );
    }
    temp.close().unwrap();
}
