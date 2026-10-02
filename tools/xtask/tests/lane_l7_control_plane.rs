use std::{
    net::{Ipv4Addr, TcpListener},
    path::Path,
};

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
    let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_control_fixture");
    let released = location.join("released-fixture");
    std::fs::copy(&fixture, &released).unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&released)
        .unwrap()
        .write_all(b"inert released identity")
        .unwrap();
    let (base, reservations) = loop {
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if base > 65440 {
            continue;
        }
        let rest: Result<Vec<_>, _> = (1..90)
            .map(|offset| TcpListener::bind((Ipv4Addr::LOCALHOST, base + offset)))
            .collect();
        if let Ok(mut rest) = rest {
            rest.push(first);
            break (base, rest);
        }
    };
    drop(reservations);
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let output = std::process::Command::new("bash")
        .arg(repository.join("scripts/qa-control-plane-mixed-version.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg("--current-binary")
        .arg(fixture)
        .arg("--released-binary")
        .arg(released)
        .arg("--evidence-dir")
        .arg(location.join("evidence"))
        .args([
            "--config-only",
            "--skip-cargo-tests",
            "--keep-logs",
            "--base-port",
            &base.to_string(),
            "--max-wait",
            "4",
            "--stable-probes",
            "2",
        ])
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location)
        .output()
        .unwrap();
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
fn mixed_version_keeps_both_mesh_directions_alive_during_owner_control() {
    let (root, output) = run(None);
    let directory = directory(root.path());
    assert!(
        output.status.success(),
        "{}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(directory.join("session.json")).unwrap_or_default()
    );
    let receipts: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(directory.join("processes.json")).unwrap()).unwrap();
    for name in [
        "current-server",
        "released-client",
        "released-server",
        "current-client",
        "config-target",
        "config-controller",
        "config-wrong-owner",
    ] {
        assert!(
            receipts
                .iter()
                .any(|row| row["name"] == name && row["cleanup_complete"] == true)
        );
    }
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(
                |row| row["name"] == "config-current-scan-refresh-wrong-owner"
                    && row["status"] == "PASS"
            )
    );
}
#[test]
fn wrong_owner_execution_cannot_certify_control_rejection() {
    let (_, output) = run(Some("allow-wrong-owner"));
    assert!(!output.status.success());
}
#[test]
fn unsorted_scan_inventory_cannot_certify_current_owner() {
    let (_, output) = run(Some("unsorted-scan"));
    assert!(!output.status.success());
}
#[test]
fn recursive_control_status_leak_rejects_mesh_narrative() {
    let (_, output) = run(Some("control-leak"));
    assert!(!output.status.success());
}
