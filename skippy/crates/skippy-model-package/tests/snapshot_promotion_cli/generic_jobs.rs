//! Actual facade executable: read-only preparation and pre-network refusal only.
#![cfg(unix)]
use skippy_model_package::jobs::{HardwareFlavor, plan_cpu_job_from_hardware};
use serde_json::{Value, json};
use std::{
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
fn fixture() -> Value {
    let plan = plan_cpu_job_from_hardware(
        &[HardwareFlavor {
            name: "cpu-basic".into(),
            pretty_name: None,
            cpu: Some("2 vCPU".into()),
            ram: Some("16 GB".into()),
            accelerator: None,
            unit_cost_usd: Some(0.01),
            unit_cost_micro_usd: None,
            unit_label: Some("minute".into()),
        }],
        "cpu-basic",
        259200,
        1024,
    )
    .unwrap();
    use sha2::{Digest, Sha256};
    let hash = hex_digest(&Sha256::digest(serde_json::to_vec(&plan).unwrap()));
    let pin = |path: &str| json!({"path":path,"sha256":"1".repeat(64)});
    let tools = [
        "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
    ]
    .map(|name| json!({"name":name,"path":format!("/opt/tools/{name}"),"sha256":"f".repeat(64)}));
    let bootstrap = json!({"schema_version":1,"mesh_commit":"a".repeat(40),"git_tree":"b".repeat(40),"llama_commit":"c".repeat(40),"upstream_file_sha256":"d".repeat(64),"image":format!("provided/native@sha256:{}","e".repeat(64)),"native_profile":"standalone-static-skippy-quantize-cpu","tools":tools,"path_directories":["/opt/tools"],"timeout_seconds":259200,"cpu_plan_receipt_sha256":hash,"declared_estimate_usd":plan.max_cost_usd,"max_cost_usd":plan.max_cost_usd});
    json!({"schema_version":1,"namespace":"fixture","mounts":[{"repo":"fixture/source","revision":"a".repeat(40),"mount_path":"/models/source"}],"cpu_plan":plan,"worker_input":{"schema_version":1,"workflow":"generic-conversion","timeout_secs":259200,"runner":pin("/opt/mesh/xtask"),"operator":{"schema_version":1,"bootstrap":bootstrap,"conversion":{"schema_version":1,"source_repo":"fixture/source","target_repo":"fixture/result","mesh_revision":"a".repeat(40),"output_basename":"model","source":"/models/source","source_files":[pin("/models/source/config.json")],"work_directory":"/work/conversion","target_prefix":"BF16","expected_splits":1,"upload_only":false,"dry_run":false,"publish_confirmed":false,"credential_file":null,"timeout_seconds":259200}},"receipt_export":{"helper":pin("/opt/mesh/model-package-publish"),"helper_source":pin("/opt/mesh/publisher-source"),"repo":"fixture/evidence","parent_commit":"a".repeat(40),"credential_file":null,"credential_environment":true,"path_in_repo":"runs/native-job.json"}}})
}
fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn invoke(root: &Path, args: &[&str]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_model-package-generic-jobs"))
        .args(args)
        .current_dir(root)
        .env_clear()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= until {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "finite facade timed out: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    child.wait_with_output().unwrap()
}
#[test]
fn actual_generic_jobs_facade_prepares_original_72h_without_network_or_token() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = root.join("input.json");
    std::fs::write(&input, serde_json::to_vec(&fixture()).unwrap()).unwrap();
    let output = root.join("prepared");
    let result = invoke(
        &root,
        &[
            "prepare",
            "--input",
            input.to_str().unwrap(),
            "--output-directory",
            output.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(output.join("result.json")).unwrap()).unwrap();
    assert_eq!(receipt["submitted"], false);
    assert_eq!(receipt["conversion_admitted"], false);
    let declaration: Value =
        serde_json::from_slice(&std::fs::read(output.join("declaration.json")).unwrap()).unwrap();
    assert_eq!(declaration["timeout_seconds"], 259200);
    assert_eq!(declaration["image_observed"], false);
    assert!(!output.join("submitted.json").exists());
    temp.close().unwrap();
}
#[test]
fn actual_generic_jobs_facade_refuses_unconfirmed_submit_and_foreign_collection_before_transport() {
    for verb in ["submit", "collect"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let input = root.join("input.json");
        std::fs::write(&input, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let output = root.join("refused");
        let result = invoke(
            &root,
            &[
                verb,
                "--input",
                input.to_str().unwrap(),
                "--output-directory",
                output.to_str().unwrap(),
                "--credential-file",
                root.join("absent-token").to_str().unwrap(),
            ],
        );
        assert!(!result.status.success());
        assert!(!output.exists());
        assert!(result.stdout.is_empty());
        temp.close().unwrap();
    }
}

#[test]
fn actual_composition_jobs_facade_prepares_distinct_complete_workflow_without_network() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let mut input = fixture();
    let b = input["worker_input"]["operator"]["bootstrap"].clone();
    let pin = |p: &str| json!({"path":p,"sha256":"1".repeat(64)});
    let tool = |p: &str| json!({"name":"helper","path":p,"sha256":"1".repeat(64)});
    input["worker_input"]["workflow"] = json!("default-mtp-composition");
    input["worker_input"]["operator"] = json!({"schema_version":1,"bootstrap":b,"staging_helper":tool("/opt/stitch"),"checkpoint":{"repo":"fixture/checkpoint","revision":"a".repeat(40),"files":{"config.json":"1".repeat(64),"model.safetensors":"2".repeat(64)}},"tokenizer_source":{"repo":"fixture/tokenizer","revision":"b".repeat(40),"files":{"tokenizer.json":"3".repeat(64)}},"tokenizer_profile":pin("/opt/profile.json"),"credential_file":null,"maximum_bytes":1048576,"target_parts":[pin("/models/source/part1.gguf"),pin("/models/source/part2.gguf")],"sidecars":[],"repository_helper":tool("/opt/repository"),"repository_helper_source":pin("/opt/repository-source"),"publisher_helper":pin("/opt/publisher"),"publisher_source":pin("/opt/publisher-source"),"overall_seconds":259200,"publication_reserve_seconds":60,"dry_run":false,"confirm_publication":true});
    let path = root.join("input.json");
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let out = root.join("prepared");
    let result = invoke(
        &root,
        &[
            "prepare",
            "--input",
            path.to_str().unwrap(),
            "--output-directory",
            out.to_str().unwrap(),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let declaration: Value =
        serde_json::from_slice(&std::fs::read(out.join("declaration.json")).unwrap()).unwrap();
    assert_eq!(declaration["timeout_seconds"], 259200);
    assert_eq!(declaration["native_certification_completed"], false);
    assert!(!root.join("submitted.json").exists());
    temp.close().unwrap();
}
