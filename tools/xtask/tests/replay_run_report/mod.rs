use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) fn assert_failed_report(output: &Path, run: &serde_json::Value) {
    let report = std::fs::read_to_string(output.join("summary/REPORT.md")).unwrap();
    let failed_checks = run["gates"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|check| check["passed"] == false);
    for check in failed_checks {
        assert!(report.contains(check["name"].as_str().unwrap()));
        assert!(report.contains(check["detail"].as_str().unwrap()));
    }
    for artifact in [
        "summary/comparison.csv",
        "summary/charts/decode-throughput.svg",
        "summary/charts/workload-output-throughput.svg",
        "summary/charts/ttft-p50.svg",
    ] {
        assert!(output.join(artifact).is_file());
    }
    let inventory = std::fs::read_to_string(output.join("artifact-sha256.txt")).unwrap();
    for line in inventory.lines() {
        let (digest, relative) = line.split_once("  ").unwrap();
        assert_eq!(
            digest,
            hex::encode(Sha256::digest(
                std::fs::read(output.join(relative)).unwrap()
            ))
        );
    }
}
