//! Finite supplied-tool/child transport proofs, not quantizer/model qualification.
use super::*;

use std::{path::PathBuf, time::Duration};
#[cfg(unix)]
fn artifact(path: PathBuf, bytes: &[u8]) -> admission::Artifact {
    std::fs::write(&path, bytes).unwrap();
    admission::Artifact {
        path,
        sha256: admission::digest(bytes),
    }
}
#[cfg(unix)]
fn fixture(mode: &str) -> (tempfile::TempDir, Input, PathBuf) {
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let b = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(b.join("source/BF16")).unwrap();
    std::fs::create_dir(b.join("evidence")).unwrap();
    let bytes = b"GGUF\x03\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    let parts = (1..=2)
        .map(|i| {
            artifact(
                b.join(format!("source/BF16/model-{i:05}-of-00002.gguf")),
                bytes,
            )
        })
        .collect();
    let source = artifact(b.join("tool-source"), b"finite inert source");
    let runtime = artifact(b.join("runtime"), b"finite inert runtime");
    let recipe = artifact(b.join("recipe"), b"tensor fixture");
    let helper_source = artifact(b.join("helper-source"), b"finite helper transport fixture");
    let tool_script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > '{}/tool-argv'\nmkdir -p '{}/work/source-window/BF16' '{}/target/Q4'\ncp '{}/source/BF16/model-00001-of-00002.gguf' '{}/work/source-window/BF16/model-00001-of-00002.gguf'\nln -s '{}/source/BF16/model-00002-of-00002.gguf' '{}/work/source-window/BF16/model-00002-of-00002.gguf'\ncp '{}/source/BF16/model-00001-of-00002.gguf' '{}/target/Q4/model-00001-of-00002.gguf'\n",
        b.display(),
        b.display(),
        b.display(),
        b.display(),
        b.display(),
        b.display(),
        b.display(),
        b.display(),
        b.display()
    );
    let tool = artifact(b.join("tool"), tool_script.as_bytes());
    std::fs::set_permissions(&tool.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > '{}/helper-argv'\nexport QUANT_WINDOW_FIXTURE_ROOT='{}'\nexec '{}' --ignored --exact automation::hf_certify::quantization_window::tests::quant_window_helper_child --nocapture\n",
        b.display(),
        b.display(),
        std::env::current_exe().unwrap().display()
    );
    let helper = artifact(b.join("helper"), script.as_bytes());
    std::fs::set_permissions(&helper.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(b.join("mode"), mode).unwrap();
    let m = json!({"schema_version":1,"kind":"QUANTIZE_GGUF","source":b.join("source"),"source_prefix":"BF16","target":b.join("target"),"target_prefix":"Q4","output_basename":"model","expected_splits":2,"window_size":1,"quant":"Q4_0","tensor_type_file":recipe.path});
    let manifest = artifact(b.join("manifest"), &serde_json::to_vec(&m).unwrap());
    let credential = b.join("credential");
    std::fs::write(&credential, b"fixture-token").unwrap();
    std::fs::set_permissions(&credential, std::fs::Permissions::from_mode(0o600)).unwrap();
    let input = Input {
        schema_version: 1,
        tool_kind: "supplied-window-quantizer".into(),
        profile_version: "inert-transport-v1".into(),
        tool,
        tool_source: source,
        runtime,
        manifest,
        recipe,
        helper,
        helper_source,
        source_repo: "fixture/source".into(),
        source_revision: "a".repeat(40),
        source_root: b.join("source"),
        source_prefix: "BF16".into(),
        source_parts: parts,
        target_repo: "fixture/target".into(),
        target_root: b.join("target"),
        target_prefix: "Q4".into(),
        basename: "model".into(),
        quant: "Q4_0".into(),
        expected_splits: 2,
        ordinal: 1,
        work_root: b.join("work"),
        credential_file: credential,
        publication_confirmed: true,
        timeout_seconds: 30,
        resume: None,
    };
    (temp, input, b.join("evidence"))
}
#[cfg(unix)]
#[test]
fn quant_window_owned_range_upload_record_and_cleanup_follow_immutable_confirmation() {
    for mode in ["success", "refused"] {
        let (_temp, input, root) = fixture(mode);
        let mut evidence = json!({});
        let result = execute(
            &input,
            &root,
            Instant::now() + Duration::from_secs(30),
            &Cancellation::default(),
            &mut evidence,
        );
        assert_eq!(result.is_ok(), mode == "success");
        assert_eq!(input.output_path().exists(), mode != "success");
        assert_eq!(
            input.work_root.join("source-window").exists(),
            mode != "success"
        );
        let argv = std::fs::read_to_string(root.parent().unwrap().join("tool-argv")).unwrap();
        assert!(argv.contains("--first-split\n1\n--last-split\n1\n--keep-staged-source"));
        if mode == "success" {
            assert_eq!(evidence["window_uploaded"], true);
            assert_eq!(evidence["staged_input_cleaned"], true);
            assert!(bootstrap::contract::hex(
                evidence["record_commit"].as_str().unwrap(),
                40
            ));
        } else {
            assert!(evidence["upload-shard-receipt"].is_object());
        }
    }
}
#[cfg(unix)]
#[test]
fn quant_window_current_tool_authority_and_recipe_refuse_before_native_child() {
    for mode in ["current", "authority", "recipe"] {
        let (_temp, mut input, root) = fixture("success");
        if mode == "current" {
            input.tool_kind = "current-native-quantizer".into();
        }
        if mode == "authority" {
            input.publication_confirmed = false;
        }
        if mode == "recipe" {
            std::fs::write(&input.recipe.path, b"changed").unwrap();
        }
        let mut evidence = json!({});
        assert!(
            execute(
                &input,
                &root,
                Instant::now() + Duration::from_secs(20),
                &Cancellation::default(),
                &mut evidence
            )
            .is_err()
        );
        assert!(!root.parent().unwrap().join("tool-argv").exists());
        assert!(!input.target_root.exists());
    }
}
#[cfg(unix)]
#[test]
#[ignore = "owned helper subprocess finite publication transport only"]
fn quant_window_helper_child() {
    let b = PathBuf::from(std::env::var_os("QUANT_WINDOW_FIXTURE_ROOT").unwrap());
    let lines = std::fs::read_to_string(b.join("helper-argv")).unwrap();
    let args = lines.lines().collect::<Vec<_>>();
    let get = |flag: &str| {
        args.iter()
            .position(|s| *s == flag)
            .map(|i| args[i + 1])
            .unwrap()
    };
    let output = PathBuf::from(get("--output-directory"));
    std::fs::create_dir(&output).unwrap();
    let path = PathBuf::from(get("--artifact"));
    let data = std::fs::read(&path).unwrap();
    let hash = admission::digest(&data);
    let mode = std::fs::read_to_string(b.join("mode")).unwrap();
    let complete = mode != "refused";
    if args[0] == "upload" {
        assert!(
            b.join("work/source-window/BF16/model-00001-of-00002.gguf")
                .exists()
        );
        let unlink = args.contains(&"--unlink-after-success");
        let v = json!({"repo":get("--repo"),"revision":get("--revision"),"artifact":path,"relative_path":get("--relative-path"),"credential_file":get("--credential-file"),"output_directory":output,"maximum_attempts":8,"timeout_seconds":get("--timeout-seconds").parse::<u64>().unwrap(),"dataset":false,"create_pr":false,"unlink_after_success":unlink,"admit_only":false,"confirm":true});
        let options: helper::UploadOptions = serde_json::from_value(v).unwrap();
        let request = admission::digest(&serde_json::to_vec(&options).unwrap());
        let receipt = json!({"schema_version":1,"repo":get("--repo"),"revision":"main","path":get("--relative-path"),"identity":{"byte_size":data.len(),"sha256":hash},"attempts":[{"ordinal":1,"commit_oid":"b".repeat(40),"remote_verified":complete,"error":if complete {Value::Null} else {json!("refused")}}],"source_custody_verified":complete,"unlink_requested":unlink,"unlinked":complete&&unlink,"completed":complete,"error":if complete {Value::Null} else {json!("refused")}});
        std::fs::write(output.join("upload.json"),serde_json::to_vec(&json!({"schema_version":1,"request_sha256":request,"status":if complete {"PUBLISHED"} else {"FAILED"},"publication":receipt})).unwrap()).unwrap();
        if complete && unlink {
            std::fs::remove_file(path).unwrap();
        }
    } else {
        assert_eq!(args[0], "verify-upload");
        let v = json!({"repo":get("--repo"),"commit":get("--commit"),"artifact":path,"relative_path":get("--relative-path"),"credential_file":get("--credential-file"),"output_directory":output,"timeout_seconds":get("--timeout-seconds").parse::<u64>().unwrap()});
        let options: helper::VerifyOptions = serde_json::from_value(v).unwrap();
        let request = admission::digest(&serde_json::to_vec(&options).unwrap());
        let verified = json!({"repo":get("--repo"),"commit":get("--commit"),"path":get("--relative-path"),"identity":{"sha256":hash,"byte_size":data.len()},"local_custody_verified":true,"completed":true,"error":null});
        std::fs::write(output.join("verification.json"),serde_json::to_vec(&json!({"schema_version":1,"request_sha256":request,"status":"IMMUTABLE_VERIFIED","verification":verified})).unwrap()).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn quant_window_resume_checks_same_commit_and_context_without_reconversion_or_mount_unlink() {
    for mode in ["success", "foreign-context", "different-manifest"] {
        let (_temp, mut input, root) = fixture("success");
        let b = root.parent().unwrap();
        let shard = artifact(
            b.join("mounted-quant.gguf"),
            &std::fs::read(&input.source_parts[0].path).unwrap(),
        );
        let record = json!({"schema_version":1,"context_sha256":if mode!="foreign-context"{input.context_sha256().unwrap()}else{"f".repeat(64)},"ordinal":input.ordinal,"relative_path":input.remote_path(),"window_uploaded":true,"artifact":{"sha256":shard.sha256,"byte_size":24}});
        let record = artifact(
            b.join("mounted-window.json"),
            &serde_json::to_vec(&record).unwrap(),
        );
        if mode == "different-manifest" {
            let mut manifest: Value =
                serde_json::from_slice(&std::fs::read(&input.manifest.path).unwrap()).unwrap();
            manifest["supplied_profile_note"] = json!("changed manifest intent");
            let bytes = serde_json::to_vec(&manifest).unwrap();
            std::fs::write(&input.manifest.path, &bytes).unwrap();
            input.manifest.sha256 = admission::digest(&bytes);
        }
        input.resume = Some(contract::Resume {
            commit: "c".repeat(40),
            record,
            shard,
        });
        let mut evidence = json!({});
        let result = execute(
            &input,
            &root,
            Instant::now() + Duration::from_secs(30),
            &Cancellation::default(),
            &mut evidence,
        );
        assert_eq!(result.is_ok(), mode == "success");
        assert!(!b.join("tool-argv").exists());
        assert!(input.resume.as_ref().unwrap().shard.path.exists());
        assert!(!input.work_root.exists());
        assert!(!input.target_root.exists());
        if mode == "success" {
            assert_eq!(evidence["resumed_immutable_commit"], "c".repeat(40));
            assert_eq!(evidence["record_commit"], "c".repeat(40));
            assert_eq!(
                evidence["artifact"]["sha256"],
                input.resume.as_ref().unwrap().shard.sha256
            );
            assert_eq!(evidence["artifact"]["byte_size"], 24);
            assert_eq!(
                evidence["resume-record-receipt"]["verification"]["commit"],
                evidence["resume-shard-receipt"]["verification"]["commit"]
            );
        } else {
            assert!(!b.join("helper-argv").exists());
        }
    }
}

#[cfg(unix)]
#[test]
fn quant_window_actual_registered_frontdoor_retains_observations_without_job_qualification() {
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    let temp = tempfile::tempdir().unwrap();
    let report = process::supervise(
        &ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            arguments: [
                "--ignored",
                "--exact",
                "automation::hf_certify::quantization_window::tests::quant_window_frontdoor_child",
                "--nocapture",
            ]
            .map(|a| Arg::Public(a.into()))
            .to_vec(),
            cwd: temp.path().canonicalize().unwrap(),
            environment: Default::default(),
        },
        &Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.success()
            && report.failure.is_none()
            && report.cleanup.complete
            && !report.cleanup.forced
            && !report.cleanup.graceful_signal_failed
            && report.cleanup.failure.is_none()
    );
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert!(
        [&report.stdout, &report.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
}
#[cfg(unix)]
#[test]
#[ignore = "isolated actual registered quant window frontdoor"]
fn quant_window_frontdoor_child() {
    let (_temp, input, root) = fixture("success");
    let request = root.parent().unwrap().join("request.json");
    std::fs::write(&request, serde_json::to_vec(&input).unwrap()).unwrap();
    let output = root.parent().unwrap().join("dispatch-evidence");
    super::super::run(&[
        "quantization-window".into(),
        "--input".into(),
        request.to_string_lossy().into(),
        "--output-directory".into(),
        output.to_string_lossy().into(),
    ])
    .unwrap();
    let observed: Value =
        serde_json::from_slice(&std::fs::read(output.join("observations.json")).unwrap()).unwrap();
    assert_eq!(observed["window_uploaded"], true);
    assert_eq!(observed["phase_error"], false);
    assert_eq!(observed["completed_job"], false);
    assert_eq!(observed["tool_profile_qualified"], false);
    assert_eq!(observed["workflow_qualified"], false);
    assert!(!input.output_path().exists());
}
