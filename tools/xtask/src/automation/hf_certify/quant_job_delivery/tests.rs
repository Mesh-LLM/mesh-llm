//! Finite outer worker and existing exporter coverage; no model, RSS or hosted proof.
use super::*;
use std::{os::unix::fs::PermissionsExt as _, path::PathBuf, process::Command};
#[path = "tests/child.rs"]
mod child;
const CHILD: &str =
    "automation::hf_certify::quant_job_delivery::tests::quant_delivery_owned_worker_child";
fn pin(path: PathBuf, bytes: &[u8]) -> admission::Artifact {
    std::fs::write(&path, bytes).unwrap();
    admission::Artifact {
        path,
        sha256: admission::digest(bytes),
    }
}
fn request(base: &Path, operator: &quant_job::contract::Input) -> Input {
    let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
    let runner = admission::Artifact {
        sha256: bootstrap::execution::observe_runner(
            &exe,
            Instant::now() + Duration::from_secs(60),
            &Cancellation::default(),
        )
        .unwrap(),
        path: exe.clone(),
    };
    let script = format!(
        "#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > '{}/export-argv'\nexport QUANT_DELIVERY_TEST_ROOT='{}'\nexport QUANT_DELIVERY_TEST_ROLE=export\n'{}' --ignored --exact '{}' --nocapture\n",
        base.display(),
        base.display(),
        exe.display(),
        CHILD
    );
    let helper = pin(base.join("receipt-helper"), script.as_bytes());
    std::fs::set_permissions(&helper.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut op = serde_json::to_value(operator).unwrap();
    op["timeout_seconds"] = json!(180);
    op["window_template"]["credential_file"] = Value::Null;
    Input {
        schema_version: 1,
        workflow: op["workflow"].as_str().unwrap().into(),
        timeout_secs: 180,
        runner,
        authority: json!({"schema_version":1,"image":format!("fixture/quant@sha256:{}", "a".repeat(64)),"mesh_commit":"b".repeat(40),"git_tree":"c".repeat(40),"cpu_plan_receipt_sha256":"d".repeat(64),"declared_estimate_usd":1.0,"max_cost_usd":2.0,"tool_kind":op["window_template"]["tool_kind"],"profile_version":op["window_template"]["profile_version"]}),
        operator: op,
        receipt_export: receipt_export::Config {
            helper,
            helper_source: pin(
                base.join("receipt-helper-source"),
                b"finite exporter source",
            ),
            repo: "fixture/evidence".into(),
            parent_commit: "e".repeat(40),
            credential_file: None,
            path_in_repo: "runs/native-job.json".into(),
            credential_environment: true,
            export_budget_secs: 20,
        },
    }
}
fn execute_case(
    mode: &str,
    combined: bool,
    corrupt: bool,
) -> (tempfile::TempDir, Value, Value, Value) {
    let (temp, op, _) = quant_job::chain_tests::fixture::new(mode, combined);
    let base = temp.path().canonicalize().unwrap();
    std::fs::write(base.join("export-mode"), if corrupt { "yes" } else { "no" }).unwrap();
    let input = request(&base, &op);
    input.validate().unwrap();
    let bytes = serde_json::to_vec(&input).unwrap();
    let path = base.join("worker-input.json");
    std::fs::write(&path, &bytes).unwrap();
    let output = base.join("delivery");
    let result = Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", CHILD, "--nocapture"])
        .env("QUANT_DELIVERY_TEST_ROLE", "worker")
        .env("QUANT_DELIVERY_TEST_ROOT", &base)
        .env("MESH_HF_PUBLICATION_TOKEN", "inert-fixture-token")
        .output()
        .unwrap();
    let success = mode == "success" && !corrupt;
    assert_eq!(
        result.status.success(),
        success,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let load = |name: &str| -> Value {
        serde_json::from_slice(&std::fs::read(output.join(name)).unwrap()).unwrap()
    };
    let native = load("native-job.json");
    let delivery = load("native-job-delivery.json");
    let exported: Value =
        serde_json::from_slice(&std::fs::read(base.join("exported-native.json")).unwrap()).unwrap();
    assert!(output.join("operator/common-commit/commit.json").exists());
    assert_eq!(native["transport_input_sha256"], admission::digest(&bytes));
    assert!(serde_json::to_vec(&native).unwrap().len() < 1048576);
    assert_eq!(native["image_observed"], false);
    assert_eq!(native["cost_observed"], false);
    assert_eq!(native["operator"]["tool_profile_qualified"], false);
    (temp, native, delivery, exported)
}
#[test]
fn quant_delivery_outer_worker_quant_and_combined_export_correlated_immutable_locator() {
    for combined in [false, true] {
        let (temp, native, delivery, exported) = execute_case("success", combined, false);
        assert_eq!(native, exported);
        assert_eq!(native["operator"]["completed_job"], true);
        assert_eq!(delivery["status"], "DELIVERED");
        assert_eq!(delivery["native_completed"], true);
        let locator = &delivery["locator"];
        assert_eq!(locator["delivery_complete"], true);
        assert_eq!(locator["commit_oid"], "f".repeat(40));
        assert_eq!(locator["repo"], "fixture/evidence");
        assert_eq!(locator["receipt_request_sha256"], native["request_sha256"]);
        assert_eq!(
            locator["transport_input_sha256"],
            native["transport_input_sha256"]
        );
        let bytes = std::fs::read(temp.path().join("exported-native.json")).unwrap();
        assert_eq!(locator["artifact_sha256"], admission::digest(&bytes));
        assert_eq!(locator["byte_size"], bytes.len() as u64);
        assert_eq!(native["operator"]["full_roster_verified"], true);
        if combined {
            assert_eq!(native["operator"]["package"]["completed"], true);
        }
        temp.close().unwrap();
    }
}
#[test]
fn quant_delivery_outer_worker_retains_native_failure_and_refuses_export_mismatch() {
    for (mode, corrupt) in [("native", false), ("success", true)] {
        let (temp, native, delivery, exported) = execute_case(mode, true, corrupt);
        assert_eq!(native["status"], "FAILED");
        assert_eq!(native["operator"]["completed_job"], false);
        assert_eq!(delivery["status"], "FAILED");
        assert_eq!(delivery["native_completed"], false);
        assert_eq!(native["operator"]["windows"].as_array().unwrap().len(), 2);
        if corrupt {
            assert!(delivery["locator"].is_null());
            assert_eq!(exported["operator"]["completed_job"], true);
        } else {
            assert_eq!(delivery["locator"]["delivery_complete"], false);
            assert_eq!(exported["operator"]["completed_job"], false);
            assert!(native["operator"]["package"].is_null());
        }
        temp.close().unwrap();
    }
}
#[test]
#[ignore = "owned finite worker/export subprocess; no model or hosted qualification"]
fn quant_delivery_owned_worker_child() {
    child::run();
}
