use std::{
    net::{Ipv4Addr, TcpListener},
    path::Path,
};

fn run(marker: Option<&str>) -> (tempfile::TempDir, std::process::Output) {
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    if let Some(marker) = marker {
        std::fs::write(location.join(marker), b"fixture").unwrap();
    }
    let (base, listeners) = loop {
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if base > 65500 {
            continue;
        }
        let rest: Result<Vec<_>, _> = (1..20)
            .map(|index| TcpListener::bind((Ipv4Addr::LOCALHOST, base + index)))
            .collect();
        if let Ok(mut rest) = rest {
            rest.push(first);
            break (base, rest);
        }
    };
    drop(listeners);
    let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let output = std::process::Command::new("bash")
        .arg(repository.join("scripts/qa-runtime-daemon-lifecycle.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg("--current-binary")
        .arg(fixture)
        .arg("--evidence-dir")
        .arg(location.join("evidence"))
        .args([
            "--base-port",
            &base.to_string(),
            "--max-wait",
            "2",
            "--keep-logs",
        ])
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location)
        .output()
        .unwrap();
    (root, output)
}
fn evidence(root: &Path) -> std::path::PathBuf {
    std::fs::read_dir(root.join("evidence"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
}
#[test]
fn lifecycle_keeps_daemon_alive_across_expected_fail_fast_and_correlated_intents() {
    let (root, output) = run(None);
    let directory = evidence(root.path());
    assert!(
        output.status.success(),
        "{}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(directory.join("session.json")).unwrap_or_default()
    );
    let receipts: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(directory.join("processes.json")).unwrap()).unwrap();
    assert!(
        receipts
            .iter()
            .any(|receipt| receipt["name"] == "fail-fast"
                && receipt["expected_exit"]["status"] == 23)
    );
    assert!(
        receipts
            .iter()
            .all(|receipt| receipt["cleanup_complete"] == true)
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "runtime_drain_model")
    );
}
#[test]
fn unrelated_intent_cannot_certify_accepted_lifecycle_command() {
    let (root, output) = run(Some("wrong-intent"));
    assert!(!output.status.success());
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(evidence(root.path()).join("summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["overall"], "fail");
}
#[test]
fn activity_private_field_cannot_pass_coarse_privacy_contract() {
    let (_, output) = run(Some("private-activity"));
    assert!(!output.status.success());
}
#[test]
fn owner_identity_failure_is_prerequisite_and_keeps_other_checks_running() {
    let (root, output) = run(Some("owner-unavailable"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(evidence(root.path()).join("summary.json")).unwrap())
            .unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "prereq.owner-identity" && row["status"] == "PREREQ")
    );
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "privacy_no_raw_data" && row["status"] == "PASS")
    );
}
#[test]
fn on_demand_usage_exit_is_prerequisite_instead_of_aborting_session() {
    let (root, output) = run(Some("on-demand-usage"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(evidence(root.path()).join("summary.json")).unwrap())
            .unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "runtime_mode_on_demand" && row["status"] == "PREREQ")
    );
}
