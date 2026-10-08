use std::{
    net::{Ipv4Addr, TcpListener},
    path::Path,
};

fn execute(marker: Option<&str>, endpoint: bool) -> (tempfile::TempDir, std::process::Output) {
    // Each scenario releases a reserved port range before its child binds it.
    // Keep these scenarios sequential so another fixture cannot reuse that gap.
    static SCENARIO: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _scenario = SCENARIO.lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    if let Some(marker) = marker {
        std::fs::write(location.join(marker), b"fixture").unwrap();
    }
    let (base, reservations) = loop {
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if base > 65500 {
            continue;
        }
        let rest: Result<Vec<_>, _> = (1..34)
            .map(|index| TcpListener::bind((Ipv4Addr::LOCALHOST, base + index)))
            .collect();
        if let Ok(mut rest) = rest {
            rest.push(first);
            break (base, rest);
        }
    };
    drop(reservations);
    let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_logging_fixture");
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut command = std::process::Command::new("bash");
    command
        .arg(repository.join("scripts/qa-logging-recovery.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg("--current-binary")
        .arg(fixture)
        .arg("--evidence-dir")
        .arg(location.join("evidence"))
        .args([
            "--base-port",
            &base.to_string(),
            "--max-wait",
            "3",
            "--keep-logs",
        ])
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location);
    if endpoint {
        command.args(["--deterministic-openai-endpoint", "http://127.0.0.1:9"]);
    }
    let output = command.output().unwrap();
    (root, output)
}
fn directory(root: &Path) -> std::path::PathBuf {
    std::fs::read_dir(root.join("evidence"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
}
#[test]
fn recovery_correlates_delete_and_preserves_restart_generations() {
    let (root, output) = execute(None, true);
    let directory = directory(root.path());
    assert!(
        output.status.success(),
        "{}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(directory.join("session.json")).unwrap_or_default()
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "logging_fail_open_inference" && row["status"] == "PASS")
    );
    let receipts: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(directory.join("processes.json")).unwrap()).unwrap();
    assert_eq!(
        receipts.iter().filter(|row| row["name"] == "seed").count(),
        2
    );
    assert!(receipts.iter().all(|row| row["cleanup_complete"] == true));
}
#[test]
fn private_marker_in_request_summary_rejects_certification() {
    let (_, output) = execute(Some("private-leak"), false);
    assert!(!output.status.success());
}
#[test]
fn foreign_delete_operation_identity_rejects_cascade() {
    let (_, output) = execute(Some("wrong-delete"), false);
    assert!(!output.status.success());
}
#[test]
fn missing_endpoint_marks_inference_prerequisite_without_skipping_fail_open() {
    let (root, output) = execute(None, false);
    assert!(
        output.status.success(),
        "{}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(directory(root.path()).join("session.json")).unwrap_or_default()
    );
    let summary: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory(root.path()).join("summary.json")).unwrap(),
    )
    .unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "logging_fail_open_inference" && row["status"] == "PREREQ")
    );
}
