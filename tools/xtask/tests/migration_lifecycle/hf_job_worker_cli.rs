//! Actual inert Linux job-worker CLI, reusing the original owned bootstrap and product peers.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
fn pin(path: &Path) -> Json {
    json!({"path":path,"sha256":hex::encode(Sha256::digest(std::fs::read(path).unwrap()))})
}
pub(super) fn fixture(mode: &str) -> (tempfile::TempDir, PathBuf, PathBuf, String) {
    let (root, source, output) = super::hf_bootstrap_cli::fixture("ok");
    let native = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    assert!(native.is_file());
    let quote = |p: &Path| format!("'{}'", p.to_str().unwrap().replace('\'', "'\\''"));
    let wrapper = format!("#!/bin/sh\nexec {} \"$@\"\n", quote(&native));
    let built = root.path().join("native-output");
    std::fs::write(&built, &wrapper).unwrap();
    let mut bootstrap: Json = serde_json::from_slice(&std::fs::read(&source).unwrap()).unwrap();
    for tool in bootstrap["tools"].as_array_mut().unwrap() {
        let path = PathBuf::from(tool["path"].as_str().unwrap());
        let script = std::fs::read_to_string(&path).unwrap();
        let needle = "printf 'inert built artifact\\n' > target/release/skippy-quantize";
        assert_eq!(script.matches(needle).count(), 1);
        let script=script.replace(needle,&format!("/bin/cp {} target/release/skippy-quantize; /bin/chmod 700 target/release/skippy-quantize",quote(&built)));
        std::fs::write(&path, &script).unwrap();
        tool["sha256"] = json!(hex::encode(Sha256::digest(script.as_bytes())));
    }
    let projector = root.path().join("projector.gguf");
    std::fs::write(&projector, format!("GGUF{mode}")).unwrap();
    let runner = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .canonicalize()
        .unwrap();
    let request = json!({"schema_version":1,"workflow":"certification","timeout_secs":30,"runner":pin(&runner),"bootstrap":bootstrap,"certification":{"mode":"projector-only","projector":pin(&projector),"target_parts":[],"expected_parts":0,"mtp_draft":null,"layer_count":1,"mtp_layer_count":null,"ctx_size":64},"projector":{"kind":"supplied","artifact":pin(&projector)}});
    std::fs::write(&source, serde_json::to_vec(&request).unwrap()).unwrap();
    (
        root,
        source,
        output,
        hex::encode(Sha256::digest(wrapper.as_bytes())),
    )
}
fn invoke(input: &Path, output: &Path, cancel: &Cancellation) -> process::RawProcessReport {
    invoke_transport(input, output, cancel, false)
}
fn invoke_transport(
    input: &Path,
    output: &Path,
    cancel: &Cancellation,
    environment_input: bool,
) -> process::RawProcessReport {
    let mut spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        cwd: input.parent().unwrap().into(),
        environment: Default::default(),
        arguments: [
            "automation",
            "hf-certify",
            "job-worker",
            "--input",
            input.to_str().unwrap(),
            "--output-directory",
            output.to_str().unwrap(),
        ]
        .map(|s| Value::Public(s.into()))
        .into(),
    };
    if environment_input {
        spec.arguments[3] = Value::Public("--input-environment".into());
        spec.arguments[4] = Value::Public("MESH_HF_JOB_INPUT".into());
        spec.environment.insert(
            "MESH_HF_JOB_INPUT".into(),
            Value::Secret(std::fs::read_to_string(input).unwrap().into()),
        );
    }
    let request: Json = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    if request.get("worker").is_some() {
        spec.arguments.insert(3, Value::Public("operator".into()));
    }
    if request["receipt_export"]["credential_environment"] == true {
        spec.environment.insert(
            "MESH_HF_PUBLICATION_TOKEN".into(),
            Value::Secret("inert-explicit-token".into()),
        );
    }
    let report = process::supervise_raw(
        &spec,
        &Limits {
            execution: Duration::from_secs(40),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1048576),
            stderr: NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(report.process.failure.is_none() && report.process.cleanup.failure.is_none());
    assert!(
        report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
    );
    for (stream, raw) in [
        (&report.process.stdout, report.stdout.as_ref()),
        (&report.process.stderr, report.stderr.as_ref()),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
        assert_eq!(stream.bytes_seen, raw.unwrap().as_bytes().len() as u64);
    }
    report
}
fn receipt(output: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(output.join("native-job.json")).unwrap()).unwrap()
}
#[test]
fn actual_native_job_worker_consumes_observed_bootstrap_binary_and_native_certification() {
    let (root, input, output, hash) = fixture("ok");
    let process = invoke(&input, &output, &Cancellation::default());
    assert_eq!(process.process.outcome, Outcome::Exited);
    assert_eq!(process.process.status.unwrap().code(), Some(0));
    let report = receipt(&output);
    assert_eq!(report["status"], "CERTIFIED");
    assert_eq!(report["bootstrap"]["observed"]["binary"]["sha256"], hash);
    assert_eq!(
        report["bootstrap"]["observed"]["binary"]["path"],
        json!(output.join("bootstrap/mesh-source/target/release/skippy-quantize"))
    );
    assert_eq!(
        report["bootstrap"]["observed"]["mesh_commit"],
        "a".repeat(40)
    );
    assert_eq!(
        report["bootstrap"]["observed"]["prepared_llama_commit"],
        "d".repeat(40)
    );
    assert_eq!(
        report["acquisition"]["certification"]["source_unchanged"],
        true
    );
    assert_eq!(
        report["acquisition"]["certification"]["native_report"]["loaded"],
        true
    );
    for key in [
        "image_observed",
        "rate_or_cost_observed",
        "publication_completed",
    ] {
        assert_eq!(report[key], false);
    }
    root.close().unwrap();
}
#[test]
fn actual_native_job_worker_later_native_failure_retains_completed_bootstrap() {
    let (root, input, output, _) = fixture("malformed");
    let process = invoke(&input, &output, &Cancellation::default());
    assert_eq!(process.process.outcome, Outcome::Exited);
    assert_eq!(process.process.status.unwrap().code(), Some(1));
    let report = receipt(&output);
    assert_eq!(report["status"], "FAILED");
    assert_eq!(report["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    assert!(
        report["bootstrap"]["phases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["phase"] == "build")
    );
    assert!(report["acquisition"]["certification"]["native_report"].is_null());
    root.close().unwrap();
}
#[test]
fn actual_native_job_worker_causal_certification_cancel_retains_previous_phase() {
    let (root, input, output, _) = fixture("held");
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let child_output = output.clone();
    let worker = std::thread::spawn(move || invoke(&input, &child_output, &child_cancel));
    let marker = output.join("acquisition/certification/invoked.json");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !marker.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancel.cancel();
    let process = worker.join().unwrap();
    assert!(observed && cancel.is_cancelled());
    assert_eq!(process.process.outcome, Outcome::Cancelled);
    let report = receipt(&output);
    assert_eq!(report["status"], "FAILED");
    assert_eq!(report["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    assert_eq!(report["publication_completed"], false);
    root.close().unwrap();
}

#[test]
fn actual_native_job_worker_consumes_secret_environment_transport_and_same_typed_request() {
    let (root, input, output, hash) = fixture("ok");
    let process = invoke_transport(&input, &output, &Cancellation::default(), true);
    assert_eq!(process.process.outcome, Outcome::Exited);
    assert_eq!(process.process.status.unwrap().code(), Some(0));
    let report = receipt(&output);
    assert_eq!(report["status"], "CERTIFIED");
    assert_eq!(report["bootstrap"]["observed"]["binary"]["sha256"], hash);
    assert_eq!(
        report["acquisition"]["certification"]["native_report"]["loaded"],
        true
    );
    let bytes = std::fs::read(&input).unwrap();
    assert_eq!(
        report["transport_input_sha256"],
        hex::encode(Sha256::digest(&bytes))
    );
    assert_eq!(report["request_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(report["image_observed"], false);
    root.close().unwrap();
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct RegularFixtureArtifact {
    path: PathBuf,
    path_in_repo: String,
    sha256: String,
    byte_size: u64,
}
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct RegularFixtureInput {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    artifact: RegularFixtureArtifact,
    receipt_request_sha256: String,
    credential_file: PathBuf,
    execution_timeout_ms: u64,
}
#[test]
#[ignore = "owned inert regular publisher child, launched only by actual worker fixture"]
fn regular_export_fixture_helper() {
    let input = PathBuf::from(std::env::var_os("MESH_EXPORT_INPUT").unwrap());
    let output = PathBuf::from(std::env::var_os("MESH_EXPORT_OUTPUT").unwrap());
    let mode = std::env::var("MESH_EXPORT_MODE").unwrap();
    let typed: RegularFixtureInput =
        serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    use std::os::unix::fs::PermissionsExt as _;
    assert_eq!(
        std::fs::metadata(&typed.credential_file)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::read(&typed.credential_file).unwrap(),
        b"inert-explicit-token"
    );
    let hash = hex::encode(Sha256::digest(serde_json::to_vec(&typed).unwrap()));
    let bytes = std::fs::read(&typed.artifact.path).unwrap();
    assert_eq!(hex::encode(Sha256::digest(&bytes)), typed.artifact.sha256);
    assert_eq!(bytes.len() as u64, typed.artifact.byte_size);
    let native: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(native["request_sha256"], typed.receipt_request_sha256);
    std::fs::create_dir(&output).unwrap();
    let progress = json!({"schema_version":1,"request_sha256":hash,"status":"IN_PROGRESS","receipt_request_sha256":typed.receipt_request_sha256,"artifact_sha256":typed.artifact.sha256,"input_custody_verified":false,"error":null,"publication":{"schema_version":1,"repo":typed.repo,"parent_commit":typed.parent_commit,"commit_oid":"d".repeat(40),"mutation_attempted":true,"remote_verified_paths":[],"source_custody_verified":false,"completed":false,"error":null}});
    std::fs::write(
        output.join("progress.json"),
        serde_json::to_vec(&progress).unwrap(),
    )
    .unwrap();
    std::fs::write(output.join("publish-started"), b"owned inert phase").unwrap();
    if mode == "held" {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            std::thread::park_timeout(Duration::from_millis(10));
        }
        panic!("owned fixture expected cancellation");
    }
    let value = json!({"schema_version":1,"request_sha256":hash,"status":"PUBLISHED_REGULAR_RECEIPT","receipt_request_sha256":typed.receipt_request_sha256,"artifact_sha256":typed.artifact.sha256,"input_custody_verified":true,"error":if mode=="outer-error"{Some("inert contradiction")}else{None},"publication":{"schema_version":1,"repo":typed.repo,"parent_commit":typed.parent_commit,"commit_oid":"d".repeat(40),"mutation_attempted":true,"remote_verified_paths":[typed.artifact.path_in_repo],"source_custody_verified":true,"completed":true,"error":if mode=="nested-error"{Some("inert contradiction")}else{None}}});
    std::fs::write(
        output.join("publication.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
}
fn exporting_fixture(mode: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt as _;
    let (root, input, output, _) = fixture(if mode == "native-failed" {
        "malformed"
    } else {
        "ok"
    });
    let quote = |p: &Path| format!("'{}'", p.to_str().unwrap().replace('\'', "'\\''"));
    let helper = root.path().join("publisher-helper");
    let script = format!(
        "#!/bin/sh\n[ \"$1\" = publish-regular-receipt ] || exit 41\nshift\n[ \"$1\" = --input ] || exit 42\nexport MESH_EXPORT_INPUT=\"$2\"\nshift 2\n[ \"$1\" = --output-directory ] || exit 43\nexport MESH_EXPORT_OUTPUT=\"$2\"\nshift 2\n[ \"$#\" = 0 ] || exit 44\nexport MESH_EXPORT_MODE={}\nexec {} --exact hf_job_worker_cli::regular_export_fixture_helper --ignored --nocapture\n",
        mode,
        quote(&std::env::current_exe().unwrap())
    );
    std::fs::write(&helper, script).unwrap();
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
    let credential = root.path().join("private-token");
    std::fs::write(&credential, b"inert-explicit-token").unwrap();
    std::fs::set_permissions(&credential, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    value["receipt_export"] = json!({"helper":pin(&helper),"helper_source":pin(&helper),"repo":"fixture/evidence","parent_commit":"a".repeat(40),"credential_file":credential,"credential_environment":false,"export_budget_secs":10,"path_in_repo":"runs/native-job.json"});
    if mode == "ok" {
        value["receipt_export"]["credential_file"] = Json::Null;
        value["receipt_export"]["credential_environment"] = json!(true);
    }
    std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
    (root, input, output)
}
#[test]
fn actual_native_worker_regular_export_inert_protocol_binds_delivered_locator_and_refuses_errors() {
    for mode in ["ok", "outer-error", "nested-error", "native-failed"] {
        let (root, input, output) = exporting_fixture(mode);
        let process = invoke_transport(&input, &output, &Cancellation::default(), true);
        let delivery: Json = serde_json::from_slice(
            &std::fs::read(output.join("native-job-delivery.json")).unwrap(),
        )
        .unwrap();
        let native = receipt(&output);
        assert_eq!(
            native["status"],
            if mode == "native-failed" {
                "FAILED"
            } else {
                "CERTIFIED"
            }
        );
        if mode == "ok" {
            assert_eq!(process.process.status.unwrap().code(), Some(0));
            assert_eq!(delivery["status"], "DELIVERED");
            assert_eq!(delivery["locator"]["delivery_complete"], true);
            assert_eq!(
                delivery["locator"]["artifact_sha256"],
                hex::encode(Sha256::digest(
                    std::fs::read(output.join("native-job.json")).unwrap()
                ))
            );
            assert!(
                process
                    .stdout
                    .unwrap()
                    .as_bytes()
                    .windows(b"MESH_NATIVE_DELIVERY ".len())
                    .any(|w| w == b"MESH_NATIVE_DELIVERY ")
            );
        } else {
            assert_eq!(process.process.status.unwrap().code(), Some(1));
            assert_eq!(delivery["status"], "FAILED");
            if mode == "native-failed" {
                assert_eq!(delivery["locator"]["delivery_complete"], false);
                assert_eq!(delivery["export"]["completed"], true);
            } else {
                assert!(delivery["locator"].is_null());
            }
        }
        root.close().unwrap();
    }
}
#[test]
fn actual_native_worker_cancel_during_owned_regular_export_keeps_completed_native_phase() {
    let (root, input, output) = exporting_fixture("held");
    let cancellation = Cancellation::default();
    let child_cancel = cancellation.clone();
    let child_output = output.clone();
    let child =
        std::thread::spawn(move || invoke_transport(&input, &child_output, &child_cancel, true));
    let marker = output.join("receipt-publication/publish-started");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !marker.exists() && Instant::now() < deadline {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancellation.cancel();
    let process = child.join().unwrap();
    assert!(observed && cancellation.is_cancelled());
    assert_eq!(process.process.outcome, Outcome::Cancelled);
    assert_eq!(receipt(&output)["status"], "CERTIFIED");
    let delivery: Json =
        serde_json::from_slice(&std::fs::read(output.join("native-job-delivery.json")).unwrap())
            .unwrap();
    assert_eq!(delivery["status"], "FAILED");
    assert!(delivery["locator"].is_null());
    assert_eq!(delivery["export"]["mutation_possible"], true);
    assert_eq!(
        delivery["export"]["partial_progress"]["publication"]["commit_oid"],
        "d".repeat(40)
    );
    assert_eq!(
        delivery["export"]["partial_progress"]["publication"]["completed"],
        false
    );
    assert!(delivery["export"]["publication"].is_null());
    root.close().unwrap();
}

fn composition_fixture(mode: &str) -> (tempfile::TempDir, PathBuf, PathBuf, Vec<PathBuf>) {
    let (root, input, output, _) = fixture("ok");
    if mode == "terminal-refusal" {
        let built = root.path().join("native-output");
        let current = std::fs::read_to_string(&built).unwrap();
        std::fs::write(&built,current.replacen("#!/bin/sh\n","#!/bin/sh\nif [ \"$1\" = compose-mtp ]; then /bin/printf 'owned terminal collision' > ../native-job-delivery.json; fi\n",1)).unwrap();
    }
    let base = root.path().canonicalize().unwrap();
    let parts = (1..=3)
        .map(|i| {
            let p = base.join(format!("Target-{i:05}-of-00003.gguf"));
            std::fs::write(&p, format!("GGUFtarget-{i}")).unwrap();
            p
        })
        .collect::<Vec<_>>();
    let mtp = base.join("mtp.gguf");
    std::fs::write(&mtp, format!("GGUF{mode}")).unwrap();
    let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    value["workflow"] = json!("supplied-converted-compose");
    value.as_object_mut().unwrap().remove("certification");
    value.as_object_mut().unwrap().remove("projector");
    value["composition"] = json!({"target_parts":parts.iter().map(|p|pin(p)).collect::<Vec<_>>(),"mtp":{"kind":"supplied-converted","artifact":pin(&mtp)},"target_basename":"Target","composite_basename":"Composite","expected_parts":3,"mtp_block":88,"composite_repo":"fixture/composite"});
    std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
    (root, input, output, parts)
}
#[test]
fn actual_native_job_supplied_compose_uses_bootstrap_tool_and_preserves_middle_pin() {
    let (root, input, output, parts) = composition_fixture("ok");
    let middle = std::fs::read(&parts[1]).unwrap();
    let raw = invoke(&input, &output, &Cancellation::default());
    assert_eq!(raw.process.outcome, Outcome::Exited);
    assert_eq!(raw.process.status.unwrap().code(), Some(0));
    let native = receipt(&output);
    assert_eq!(native["status"], "COMPOSED");
    assert_eq!(native["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    assert_eq!(native["composition"]["source_unchanged"], true);
    let plan: Json =
        serde_json::from_slice(&std::fs::read(output.join("composition-plan.json")).unwrap())
            .unwrap();
    assert_eq!(
        plan["required_receipt"]["request_sha256"],
        native["request_sha256"]
    );
    assert_eq!(
        plan["required_receipt"]["sha256"],
        hex::encode(Sha256::digest(
            std::fs::read(output.join("native-job.json")).unwrap()
        ))
    );
    assert_eq!(plan["entries"].as_array().unwrap().len(), 3);
    assert_eq!(plan["entries"][1]["source_kind"], "untouched-pinned-middle");
    assert_eq!(
        plan["entries"][1]["sha256"],
        hex::encode(Sha256::digest(&middle))
    );
    assert_eq!(
        native["composition"]["admitted"]["binary"],
        native["bootstrap"]["observed"]["binary"]
    );
    assert_eq!(std::fs::read(&parts[1]).unwrap(), middle);
    let delivery: Json =
        serde_json::from_slice(&std::fs::read(output.join("native-job-delivery.json")).unwrap())
            .unwrap();
    assert_eq!(delivery["status"], "LOCAL_COMPOSED");
    assert!(!output.join("acquisition").exists());
    root.close().unwrap();
}
#[test]
fn actual_native_job_compose_native_refusal_and_unconverted_input_leave_no_eligible_plan() {
    for mode in ["malformed", "terminal-refusal"] {
        let (root, input, output, _) = composition_fixture(mode);
        let raw = invoke(&input, &output, &Cancellation::default());
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        assert_eq!(
            receipt(&output)["status"],
            if mode == "malformed" {
                "FAILED"
            } else {
                "COMPOSED"
            }
        );
        assert_eq!(
            receipt(&output)["bootstrap"]["status"],
            "BOOTSTRAP_COMPLETED"
        );
        assert!(!output.join("composition-plan.json").exists());
        if mode == "terminal-refusal" {
            assert_eq!(
                std::fs::read(output.join("native-job-delivery.json")).unwrap(),
                b"owned terminal collision"
            );
        }
        root.close().unwrap();
    }
    let (root, input, output, _) = composition_fixture("ok");
    let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    value["composition"]["mtp"] =
        json!({"kind":"nemotron-checkpoint","directory":root.path().join("checkpoint")});
    std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
    let raw = invoke(&input, &output, &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(!output.exists());
    root.close().unwrap();
}
#[test]
fn actual_native_job_compose_causal_cancel_keeps_bootstrap_without_terminal_plan() {
    let (root, input, output, _) = composition_fixture("held");
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let child_output = output.clone();
    let child = std::thread::spawn(move || invoke(&input, &child_output, &child_cancel));
    let marker = output.join("composition/held-marker");
    let until = Instant::now() + Duration::from_secs(20);
    while !marker.exists() && Instant::now() < until {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let seen = marker.exists();
    cancel.cancel();
    let raw = child.join().unwrap();
    assert!(seen && cancel.is_cancelled());
    assert_eq!(raw.process.outcome, Outcome::Cancelled);
    let report = receipt(&output);
    assert_eq!(report["status"], "FAILED");
    assert_eq!(report["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    assert!(!output.join("composition-plan.json").exists());
    root.close().unwrap();
}

fn native_conversion_fixture(mode: &str) -> (tempfile::TempDir, PathBuf, PathBuf, Vec<PathBuf>) {
    let (root, input, output, parts) = composition_fixture("ok");
    if mode == "terminal-refusal" {
        let built = root.path().join("native-output");
        let current = std::fs::read_to_string(&built).unwrap();
        std::fs::write(&built,current.replacen("#!/bin/sh\n","#!/bin/sh\nif [ \"$1\" = compose-mtp ]; then /bin/printf 'owned terminal collision' > ../native-job-delivery.json; fi\n",1)).unwrap();
    }
    let base = root.path().canonicalize().unwrap();
    let source = base.join("checkpoint");
    std::fs::create_dir(&source).unwrap();
    for (name, bytes) in [
        ("config.json", b"{\"fixture\":true}".as_slice()),
        ("tokenizer.json", b"{}".as_slice()),
        ("weights.safetensors", b"inert tensor bytes".as_slice()),
        ("fixture-mode", mode.as_bytes()),
    ] {
        std::fs::write(source.join(name), bytes).unwrap();
    }
    let profile = base.join("tokenizer-profile.json");
    std::fs::write(&profile, b"{}").unwrap();
    let mut request: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    let composition = request
        .as_object_mut()
        .unwrap()
        .remove("composition")
        .unwrap();
    request["workflow"] = json!("native-nemotron-compose");
    request["conversion"] = json!({"checkpoint_directory":source,"checkpoint_files":["config.json","tokenizer.json","weights.safetensors","fixture-mode"].iter().map(|n|pin(&source.join(n))).collect::<Vec<_>>(),"tokenizer_profile":pin(&profile),"target_parts":composition["target_parts"],"target_basename":"Target","composite_basename":"Composite","expected_parts":3,"mtp_block":88,"composite_repo":"fixture/composite"});
    std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    (root, input, output, parts)
}
#[test]
fn actual_native_job_raw_conversion_binds_bootstrap_full_roster_and_terminal_plan() {
    let (root, input, output, parts) = native_conversion_fixture("ok");
    let middle = std::fs::read(&parts[1]).unwrap();
    let process = invoke(&input, &output, &Cancellation::default());
    assert_eq!(process.process.outcome, Outcome::Exited);
    assert_eq!(process.process.status.unwrap().code(), Some(0));
    let native = receipt(&output);
    assert_eq!(native["status"], "COMPOSED");
    assert_eq!(native["composition"]["source_unchanged"], true);
    assert_eq!(
        native["composition"]["native_conversion_sources_before"]["binary"],
        native["bootstrap"]["observed"]["binary"]
    );
    assert_eq!(
        native["composition"]["native_conversion_sources_before"]["source_sha256"],
        native["composition"]["native_conversion_sources_final"]["source_sha256"]
    );
    assert_eq!(
        native["composition"]["native_conversion_verification"]["complete"],
        true
    );
    let calls = std::fs::read_to_string(
        output.join("composition/native-conversion/conversion-invocations.jsonl"),
    )
    .unwrap();
    let calls: Vec<Vec<String>> = calls
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0][0], "convert");
    assert_eq!(calls[1][0], "verify-job");
    assert!(
        calls[0]
            .windows(2)
            .any(|w| w == ["--backend", "native-rust"])
    );
    assert!(calls[0].iter().any(|a| a == "--mtp"));
    assert_eq!(std::fs::read(&parts[1]).unwrap(), middle);
    let plan: Json =
        serde_json::from_slice(&std::fs::read(output.join("composition-plan.json")).unwrap())
            .unwrap();
    assert_eq!(plan["entries"].as_array().unwrap().len(), 3);
    assert_eq!(
        plan["entries"][1]["sha256"],
        hex::encode(Sha256::digest(&middle))
    );
    assert_eq!(
        plan["required_receipt"]["request_sha256"],
        native["request_sha256"]
    );
    let delivery: Json =
        serde_json::from_slice(&std::fs::read(output.join("native-job-delivery.json")).unwrap())
            .unwrap();
    assert_eq!(delivery["status"], "LOCAL_COMPOSED");
    assert!(!output.join("acquisition").exists());
    root.close().unwrap();
}
#[test]
fn actual_native_job_raw_conversion_refuses_partial_missing_verification_and_input_drift() {
    for mode in [
        "nonzero",
        "missing",
        "bad-verify",
        "drift",
        "terminal-refusal",
    ] {
        let (root, input, output, _) = native_conversion_fixture(mode);
        let process = invoke(&input, &output, &Cancellation::default());
        assert_eq!(process.process.outcome, Outcome::Exited);
        assert_eq!(process.process.status.unwrap().code(), Some(1));
        let native = receipt(&output);
        assert_eq!(
            native["status"],
            if mode == "terminal-refusal" {
                "COMPOSED"
            } else {
                "FAILED"
            }
        );
        assert_eq!(native["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
        assert!(!native["composition"]["native_conversion_sources_before"].is_null());
        assert!(!output.join("composition-plan.json").exists());
        if mode == "nonzero" {
            assert!(
                output
                    .join("composition/native-conversion/partial-spool")
                    .is_file()
            );
        }
        if mode == "terminal-refusal" {
            assert_eq!(
                std::fs::read(output.join("native-job-delivery.json")).unwrap(),
                b"owned terminal collision"
            );
        }
        root.close().unwrap();
    }
}
#[test]
fn actual_native_job_raw_conversion_causal_cancel_keeps_observed_bootstrap_without_plan() {
    let (root, input, output, _) = native_conversion_fixture("held");
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let child_output = output.clone();
    let child = std::thread::spawn(move || invoke(&input, &child_output, &child_cancel));
    let marker = output.join("composition/native-conversion/conversion-held-marker");
    let until = Instant::now() + Duration::from_secs(20);
    while !marker.exists() && Instant::now() < until {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let seen = marker.exists();
    cancel.cancel();
    let process = child.join().unwrap();
    assert!(seen && cancel.is_cancelled());
    assert_eq!(process.process.outcome, Outcome::Cancelled);
    let native = receipt(&output);
    assert_eq!(native["status"], "FAILED");
    assert_eq!(native["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    assert!(!native["composition"]["native_conversion_sources_before"].is_null());
    assert!(!output.join("composition-plan.json").exists());
    root.close().unwrap();
}

fn operator_fixture(mode: &str, projector_only: bool) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let (root, source, output, _) = fixture(mode);
    let mut worker: Json = serde_json::from_slice(&std::fs::read(&source).unwrap()).unwrap();
    worker["timeout_secs"] = json!(60);
    worker["bootstrap"]["timeout_seconds"] = json!(60);
    let layout = root.path().join("mounted[fixture]");
    std::fs::create_dir_all(layout.join("nested")).unwrap();
    for name in ["part-02.gguf", "part-01.gguf"] {
        std::fs::write(layout.join("nested").join(name), format!("GGUF{name}")).unwrap();
    }
    let draft = root.path().join("draft.gguf");
    std::fs::write(&draft, b"GGUFinert draft").unwrap();
    worker["certification"]["mode"] = json!(if projector_only {
        "projector-only"
    } else {
        "mtp-attach"
    });
    worker["certification"]["mtp_layer_count"] = json!(1);
    let profile = worker["certification"].as_object_mut().unwrap();
    for name in ["target_parts", "expected_parts", "mtp_draft"] {
        profile.remove(name);
    }
    let request = json!({"schema_version":1,"model_root":layout,"model_pattern":"nested/part-*.gguf","expected_parts":2,"mtp_draft":draft,"worker":worker});
    std::fs::write(&source, serde_json::to_vec(&request).unwrap()).unwrap();
    (root, source, output)
}
fn operator_receipt(output: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(output.join("operator.json")).unwrap()).unwrap()
}
#[test]
fn actual_certification_operator_selects_complete_nested_glob_and_binds_existing_worker() {
    let (root, input, output) = operator_fixture("ok", false);
    let mut request: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    request["model_root"] = json!("mounted[fixture]");
    request["mtp_draft"] = json!("draft.gguf");
    std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    let process = invoke(&input, &output, &Cancellation::default());
    assert_eq!(process.process.status.unwrap().code(), Some(0));
    assert_eq!(operator_receipt(&output)["status"], "CERTIFIED");
    let actual_report: Json =
        serde_json::from_slice(&std::fs::read(output.join("certification-report.json")).unwrap())
            .unwrap();
    assert!(actual_report["session_created"] == true);
    assert_eq!(
        operator_receipt(&output)["request_sha256"],
        hex::encode(Sha256::digest(std::fs::read(&input).unwrap()))
    );
    let selected: Json =
        serde_json::from_slice(&std::fs::read(output.join("worker-input.json")).unwrap()).unwrap();
    let parts = selected["certification"]["target_parts"]
        .as_array()
        .unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts[0]["path"].as_str().unwrap().ends_with("part-01.gguf"));
    assert!(parts[1]["path"].as_str().unwrap().ends_with("part-02.gguf"));
    let job = receipt(&output.join("worker"));
    assert_eq!(job["status"], "CERTIFIED");
    assert_eq!(
        job["acquisition"]["certification"]["native_report"]["model_parts"][0],
        parts[0]["path"]
    );
    let bounded: Json =
        serde_json::from_slice(&std::fs::read(output.join("bounded-worker-input.json")).unwrap())
            .unwrap();
    assert!(bounded["timeout_secs"].as_u64().unwrap() < 60);
    assert_eq!(job["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
    root.close().unwrap();
}
#[test]
fn actual_certification_operator_refuses_count_magic_and_fifo_before_worker_bootstrap() {
    use std::os::unix::ffi::OsStrExt as _;
    for mode in ["count", "magic", "fifo"] {
        let (root, input, output) = operator_fixture("ok", false);
        let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        let part = PathBuf::from(value["model_root"].as_str().unwrap()).join("nested/part-01.gguf");
        match mode {
            "count" => value["expected_parts"] = json!(3),
            "magic" => std::fs::write(&part, b"not GGUF").unwrap(),
            _ => {
                std::fs::remove_file(&part).unwrap();
                let name = std::ffi::CString::new(part.as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            }
        }
        std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
        let process = invoke(&input, &output, &Cancellation::default());
        assert_eq!(process.process.status.unwrap().code(), Some(1));
        assert_eq!(operator_receipt(&output)["status"], "FAILED");
        assert!(!output.join("worker").exists());
        assert!(!output.join("worker-input.json").exists());
        root.close().unwrap();
    }
}
#[test]
fn actual_certification_operator_preserves_projector_only_and_regular_symlink_mounts() {
    for only in [false, true] {
        let (root, input, output) = operator_fixture("ok", only);
        let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        if only {
            value["model_root"] = json!(root.path().join("not-mounted"));
            value["model_pattern"] = json!("ignored");
            value["mtp_draft"] = json!(root.path().join("absent-draft.gguf"));
        } else {
            for (logical, canonical) in [
                ("part-01.gguf", "z-target.gguf"),
                ("part-02.gguf", "a-target.gguf"),
            ] {
                let part = PathBuf::from(value["model_root"].as_str().unwrap())
                    .join("nested")
                    .join(logical);
                let target = root.path().join(canonical);
                std::fs::rename(&part, &target).unwrap();
                std::os::unix::fs::symlink(&target, &part).unwrap();
            }
        }
        std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
        let process = invoke(&input, &output, &Cancellation::default());
        assert_eq!(process.process.status.unwrap().code(), Some(0));
        assert_eq!(operator_receipt(&output)["status"], "CERTIFIED");
        let selected: Json =
            serde_json::from_slice(&std::fs::read(output.join("worker-input.json")).unwrap())
                .unwrap();
        assert_eq!(
            selected["certification"]["expected_parts"],
            if only { 0 } else { 2 }
        );
        if !only {
            let expected = json!([
                root.path().canonicalize().unwrap().join("z-target.gguf"),
                root.path().canonicalize().unwrap().join("a-target.gguf")
            ]);
            let job = receipt(&output.join("worker"));
            assert_eq!(
                job["acquisition"]["certification"]["native_report"]["model_parts"],
                expected
            );
            assert_eq!(
                selected["certification"]["target_parts"][0]["path"],
                expected[0]
            );
            assert_eq!(
                selected["certification"]["target_parts"][1]["path"],
                expected[1]
            );
            let argv: Vec<String> = serde_json::from_slice(
                &std::fs::read(output.join("worker/acquisition/certification/invoked.json"))
                    .unwrap(),
            )
            .unwrap();
            let actual = argv
                .windows(2)
                .filter(|p| p[0] == "--model")
                .map(|p| p[1].clone())
                .collect::<Vec<_>>();
            assert_eq!(json!(actual), expected);
        }
        root.close().unwrap();
    }
}
#[test]
fn actual_certification_operator_native_failure_and_causal_cancel_retain_worker_observations() {
    for mode in ["malformed", "held"] {
        let (root, input, output) = operator_fixture(mode, true);
        let cancel = Cancellation::default();
        let child_cancel = cancel.clone();
        let child_output = output.clone();
        let worker = std::thread::spawn(move || invoke(&input, &child_output, &child_cancel));
        if mode == "held" {
            let marker = output.join("worker/acquisition/certification/invoked.json");
            let until = Instant::now() + Duration::from_secs(20);
            while !marker.exists() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
            let observed = marker.exists();
            cancel.cancel();
            let process = worker.join().unwrap();
            assert!(observed);
            assert_eq!(process.process.outcome, Outcome::Cancelled);
        } else {
            let process = worker.join().unwrap();
            assert_eq!(process.process.status.unwrap().code(), Some(1));
        }
        assert_eq!(operator_receipt(&output)["status"], "FAILED");
        let job = receipt(&output.join("worker"));
        assert_eq!(job["status"], "FAILED");
        assert_eq!(job["bootstrap"]["status"], "BOOTSTRAP_COMPLETED");
        assert_eq!(job["publication_completed"], false);
        root.close().unwrap();
    }
}
