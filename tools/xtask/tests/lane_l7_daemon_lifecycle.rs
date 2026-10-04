use std::path::Path;

#[path = "lane_l7_daemon_lifecycle/port_reservation.rs"]
mod port_reservation;

// Each narrative releases a reserved port range before its child binds it.
// Keep this target's narratives exclusive across that handoff window.
static NARRATIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(marker: Option<&str>) -> (tempfile::TempDir, std::process::Output) {
    let _exclusive = NARRATIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    if let Some(marker) = marker {
        std::fs::write(location.join(marker), b"fixture").unwrap();
    }
    // Automatic outbound HTTP ports must not claim a later daemon's listener
    // between reservation and launch, or leave that service port in TIME_WAIT.
    let (base, listeners) = port_reservation::reserve();
    drop(listeners);
    let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples")
        .join(format!("l7_daemon_fixture{}", std::env::consts::EXE_SUFFIX));
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
fn assert_success(output: &std::process::Output, directory: &Path) {
    if output.status.success() {
        return;
    }
    let mut diagnostic = String::from_utf8_lossy(&output.stderr).into_owned();
    for name in ["session.json", "processes.json", "manifest.json"] {
        diagnostic.push_str(&format!(
            "\n{name}: {}",
            std::fs::read_to_string(directory.join(name)).unwrap_or_default()
        ));
    }
    for entry in std::fs::read_dir(directory.join("logs")).unwrap().flatten() {
        let path = entry.path();
        if path.to_string_lossy().ends_with(".stderr.log") {
            diagnostic.push_str(&format!(
                "\n{}: {}",
                entry.file_name().to_string_lossy(),
                std::fs::read_to_string(path).unwrap_or_default()
            ));
        }
    }
    panic!("{diagnostic}");
}
#[test]
fn lifecycle_keeps_daemon_alive_across_expected_fail_fast_and_correlated_intents() {
    let (root, output) = run(None);
    let directory = evidence(root.path());
    assert_success(&output, &directory);
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
    assert_success(&output, &evidence(root.path()));
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
    assert_success(&output, &evidence(root.path()));
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
