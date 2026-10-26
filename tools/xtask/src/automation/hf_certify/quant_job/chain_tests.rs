//! Actual coordinator child-chain with finite bytes, not a model/native/HF oracle.
use super::*;
use std::{path::PathBuf, time::Duration};
#[path = "chain_tests/child.rs"]
mod child;
#[path = "chain_tests/fixture.rs"]
pub(in crate::automation::hf_certify) mod fixture;
#[test]
fn quant_job_full_owned_chain_publishes_complete_quant_and_package_commits() {
    for combined in [false, true] {
        let (_temp, input, root) = fixture::new("success", combined);
        let mut receipt = Value::Null;
        execute(
            &input,
            &root,
            Instant::now() + Duration::from_secs(120),
            &Cancellation::default(),
            &mut receipt,
        )
        .unwrap_or_else(|e| panic!("{e}; receipt={receipt}"));
        assert_eq!(receipt["completed_job"], true);
        assert_eq!(receipt["full_roster_verified"], true);
        assert_eq!(receipt["verify_job"]["completed"], true);
        assert_eq!(
            receipt["status"],
            if combined {
                "QUANTIZATION_PACKAGED"
            } else {
                "QUANTIZATION_PUBLISHED"
            }
        );
        assert_eq!(receipt["tool_profile_qualified"], false);
        assert_eq!(receipt["workflow_qualified"], false);
        assert_eq!(receipt["windows"].as_array().unwrap().len(), 2);
        assert!(serde_json::to_vec(&receipt).unwrap().len() < 1048576);
        assert!(!input.window_template.target_root.exists());
        assert!(!input.window_template.work_root.exists());
        assert!(bootstrap::contract::hex(
            receipt["final_commit"].as_str().unwrap(),
            40
        ));
        let order = std::fs::read_to_string(root.parent().unwrap().join("events")).unwrap();
        assert!(order.find("quant-1").unwrap() < order.find("quant-2").unwrap());
        assert!(order.find("common-commit").unwrap() < order.find("native-verify").unwrap());
        if combined {
            assert_eq!(receipt["package"]["completed"], true);
            assert!(bootstrap::contract::hex(
                receipt["package"]["final_commit"].as_str().unwrap(),
                40
            ));
            assert!(order.find("native-verify").unwrap() < order.find("package-write").unwrap());
            assert!(order.find("package-verify").unwrap() < order.find("package-upload").unwrap());
        }
    }
}
#[test]
fn quant_job_full_chain_refuses_immutable_roster_native_and_package_failures() {
    for (mode, stage) in [
        ("roster", "common-commit"),
        ("native", "native-verify"),
        ("package", "package-verify"),
    ] {
        let (_temp, input, root) = fixture::new(mode, true);
        let mut receipt = Value::Null;
        assert!(
            execute(
                &input,
                &root,
                Instant::now() + Duration::from_secs(120),
                &Cancellation::default(),
                &mut receipt
            )
            .is_err()
        );
        assert_eq!(receipt["completed_job"], false);
        assert_eq!(receipt["status"], "FAILED");
        assert_eq!(receipt["windows"].as_array().unwrap().len(), 2);
        let events = std::fs::read_to_string(root.parent().unwrap().join("events")).unwrap();
        assert!(events.contains(stage));
        assert!(!events.contains("package-upload"));
        if mode == "roster" {
            assert!(!events.contains("native-verify"));
        }
        if mode == "native" {
            assert!(!events.contains("package-write"));
        }
        assert!(root.join("common-commit/commit.json").exists());
    }
}
#[test]
#[ignore = "owned finite subprocess transport; no native/model/remote qualification"]
fn quant_job_owned_chain_child() {
    child::run();
}
