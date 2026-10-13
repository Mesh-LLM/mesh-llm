#![cfg(target_os = "linux")]
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Duration};
fn run(root: &Path, manifest: &Value, label: &str, upload: bool) -> Value {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value as Arg,
    };
    let input = root.join(format!("{label}.json"));
    fs::write(&input, serde_json::to_vec(manifest).unwrap()).unwrap();
    let output = root.join(label);
    let mut args = vec![
        "automation".into(),
        "hf-certify".into(),
        "generic-job".into(),
        "--input".into(),
        input.to_str().unwrap().into(),
        "--output-directory".into(),
        output.to_str().unwrap().into(),
    ];
    if upload {
        args.push("--upload-only".into());
    }
    let result = process::supervise_raw_with_files(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            environment: Default::default(),
            arguments: args.into_iter().map(Arg::Public).collect(),
        },
        &Limits {
            execution: Duration::from_secs(50),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1048576),
            stderr: std::num::NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(result.process.failure.is_none());
    assert!(
        result.process.cleanup.complete
            && !result.process.cleanup.forced
            && result.process.cleanup.failure.is_none()
            && !result.process.cleanup.graceful_signal_failed
    );
    for (raw, stream) in [
        (&result.stdout, &result.process.stdout),
        (&result.stderr, &result.process.stderr),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
        assert_eq!(
            raw.as_ref().unwrap().as_bytes().len() as u64,
            stream.bytes_seen
        );
    }
    let value: Value =
        serde_json::from_slice(&fs::read(output.join("generic-job.json")).unwrap()).unwrap();
    assert_eq!(
        result.process.status.and_then(|s| s.code()) == Some(0),
        value["status"] == "OPERATOR_COMPLETED"
    );
    value
}
pub(super) fn fixture(mode: &str) -> (tempfile::TempDir, Value) {
    let (temp, input, _) = super::hf_bootstrap_cli::fixture(mode);
    let root = temp.path().canonicalize().unwrap();
    let mut bootstrap: Value = serde_json::from_slice(&fs::read(input).unwrap()).unwrap();
    let mut conversion = super::hf_generic_conversion::input(&root, false);
    conversion["mesh_revision"] = json!("a".repeat(40));
    conversion["timeout_seconds"] = json!(40);
    let source_binary = conversion["binary"]["path"].as_str().unwrap();
    let just = root.join("tools/just");
    let bytes = fs::read_to_string(&just).unwrap();
    let replaced=bytes.replace("printf 'inert built artifact\\n' > target/release/skippy-quantize",&format!("/bin/cp '{}' target/release/skippy-quantize; /bin/chmod 700 target/release/skippy-quantize",source_binary));
    assert_ne!(bytes, replaced);
    fs::write(&just, &replaced).unwrap();
    for tool in bootstrap["tools"].as_array_mut().unwrap() {
        if tool["name"] == "just" {
            tool["sha256"] = json!(hex::encode(Sha256::digest(replaced.as_bytes())));
        }
    }
    conversion["binary"] = Value::Null;
    (
        temp,
        json!({"schema_version":1,"bootstrap":bootstrap,"conversion":conversion}),
    )
}
#[test]
fn actual_generic_operator_uses_observed_g3_output_then_supports_upload_only_without_rebuild() {
    let (temp, mut input) = fixture("ok");
    let root = temp.path().canonicalize().unwrap();
    let receipt = run(&root, &input, "built", false);
    assert_eq!(receipt["status"], "OPERATOR_COMPLETED");
    assert_eq!(receipt["observed_bootstrap"]["mesh_commit"], "a".repeat(40));
    let binary = receipt["observed_bootstrap"]["binary"]["path"]
        .as_str()
        .unwrap();
    assert!(binary.contains("built/bootstrap/mesh-source/target/release/skippy-quantize"));
    assert_eq!(
        receipt["conversion_receipt"]["status"],
        "LOCAL_ARTIFACT_READY"
    );
    assert_eq!(
        receipt["conversion_receipt"]["artifact_roster"]
            .as_object()
            .unwrap()
            .len(),
        6
    );
    input["bootstrap"] = Value::Null;
    let uploaded = run(&root, &input, "upload-only", true);
    assert_eq!(uploaded["status"], "OPERATOR_COMPLETED");
    assert!(uploaded["observed_bootstrap"].is_null());
    assert!(!root.join("upload-only/bootstrap").exists());
    assert_eq!(
        uploaded["conversion_receipt"]["artifact_roster"]
            .as_object()
            .unwrap()
            .len(),
        5
    );
    temp.close().unwrap();
}
#[test]
fn actual_generic_operator_preserves_failed_bootstrap_phases_without_conversion_or_publication() {
    let (temp, input) = fixture("wrong-head");
    let root = temp.path().canonicalize().unwrap();
    let receipt = run(&root, &input, "refused", false);
    assert_eq!(receipt["status"], "FAILED");
    assert!(!receipt["bootstrap_phases"].as_array().unwrap().is_empty());
    assert!(receipt["conversion_receipt"].is_null());
    assert!(!root.join("refused/conversion").exists());
    temp.close().unwrap();
}
#[test]
fn actual_generic_job_shell_caller_forwards_literal_original_options_to_native_owner() {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value as Arg,
    };
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let wrapper = root.join("caller.sh");
    fs::write(
        &wrapper,
        include_str!("../../../../scripts/hf-skippy-convert-job.sh"),
    )
    .unwrap();
    let fake = root.join("finite-automation");
    let capture = root.join("argv");
    fs::write(
        &fake,
        format!(
            "#!/bin/bash\nset -eu\nprintf '%s\\n' \"$@\" > '{}'\n",
            capture.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();
    let args = [
        wrapper.to_str().unwrap(),
        "--input",
        "/prepared/request.json",
        "--output-directory",
        "/fresh/evidence",
        "--output-basename",
        "literal model;$(not-executed)",
    ];
    let result = process::supervise_raw_with_files(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.clone(),
            environment: std::collections::BTreeMap::from([(
                "MESH_LLM_AUTOMATION_BIN".into(),
                Arg::Public(fake.into_os_string()),
            )]),
            arguments: args.into_iter().map(|s| Arg::Public(s.into())).collect(),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert_eq!(result.process.status.and_then(|s| s.code()), Some(0));
    assert!(
        result.process.failure.is_none()
            && result.process.cleanup.complete
            && !result.process.cleanup.forced
            && !result.process.cleanup.graceful_signal_failed
            && result.process.cleanup.failure.is_none()
    );
    assert_eq!(result.process.outcome, process::Outcome::Exited);
    for (raw, stream) in [
        (&result.stdout, &result.process.stdout),
        (&result.stderr, &result.process.stderr),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated && stream.oversized_lines == 0);
        assert_eq!(
            raw.as_ref().map_or(0, |bytes| bytes.as_bytes().len()) as u64,
            stream.bytes_seen
        );
    }
    let observed = fs::read_to_string(capture).unwrap();
    assert_eq!(
        observed.lines().collect::<Vec<_>>(),
        vec![
            "automation",
            "hf-certify",
            "generic-job",
            "--input",
            "/prepared/request.json",
            "--output-directory",
            "/fresh/evidence",
            "--output-basename",
            "literal model;$(not-executed)"
        ]
    );
    assert!(!root.join("not-executed").exists());
    temp.close().unwrap();
}
