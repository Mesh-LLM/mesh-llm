use std::{
    net::{Ipv4Addr, TcpListener},
    path::Path,
};

fn range() -> (u16, Vec<TcpListener>) {
    loop {
        let seed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = seed.local_addr().unwrap().port();
        if base > 65531 {
            continue;
        }
        let others: Result<Vec<_>, _> = (1..4)
            .map(|index| TcpListener::bind((Ipv4Addr::LOCALHOST, base + index)))
            .collect();
        if let Ok(mut listeners) = others {
            listeners.push(seed);
            return (base, listeners);
        }
    }
}
fn run(stale: bool) -> (tempfile::TempDir, std::process::Output) {
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    if stale {
        std::fs::write(location.join("stale-run"), b"stale").unwrap();
    }
    let fixture = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples")
        .join(format!("l7_split_fixture{}", std::env::consts::EXE_SUFFIX));
    let (api, api_reservations) = range();
    let (console, console_reservations) = range();
    drop(api_reservations);
    drop(console_reservations);
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let output = std::process::Command::new("bash")
        .arg(repository.join("scripts/certify-split-startup-recovery.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg(fixture)
        .arg("fixture-model")
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location)
        .env("MESH_SPLIT_CERT_WORKERS", "3")
        .env("MESH_SPLIT_CERT_MAX_WAIT", "10")
        .env("MESH_SPLIT_CERT_RECOVERY_MAX_WAIT", "4")
        .env("MESH_SPLIT_CERT_STABLE_PROBES", "2")
        .env("MESH_SPLIT_CERT_BASE_API_PORT", api.to_string())
        .env("MESH_SPLIT_CERT_BASE_CONSOLE_PORT", console.to_string())
        .env("MESH_SPLIT_CERT_BASE_BIND_PORT", "54000")
        .env("MESH_SPLIT_CERT_WORK_DIR", location.join("evidence"))
        .env("MESH_SPLIT_CERT_PROCESS_ROOT", location.join("processes"))
        .env("MESH_SPLIT_CERT_EXPECT", "replacement")
        .env("MESH_SPLIT_CERT_RUN_INFERENCE", "0")
        .output()
        .unwrap();
    (root, output)
}
#[test]
fn three_worker_loss_requires_new_run_and_preserves_cleanup_receipts() {
    let (root, output) = run(false);
    if !output.status.success() {
        for entry in std::fs::read_dir(root.path().join("evidence")).unwrap() {
            let entry = entry.unwrap();
            if entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "log")
            {
                eprintln!(
                    "{}: {}",
                    entry.path().display(),
                    std::fs::read_to_string(entry.path()).unwrap_or_default()
                );
            }
        }
    }
    assert!(
        output.status.success(),
        "{}; {}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(root.path().join("evidence/session.json")).unwrap_or_default(),
        std::fs::read_to_string(root.path().join("evidence/seed.stderr.log")).unwrap_or_default()
    );
    let members: Vec<serde_json::Value> = serde_json::from_slice(
        &std::fs::read(root.path().join("evidence/processes.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(members.len(), 4);
    assert!(
        members
            .iter()
            .all(|member| member["cleanup_complete"] == true)
    );
    assert!(
        members
            .iter()
            .any(|member| member["name"] == "worker-1"
                && member["disposition"] == "intentional-stop")
    );
}
#[test]
fn stale_run_after_worker_loss_cannot_certify_replacement() {
    let (root, output) = run(true);
    assert!(!output.status.success());
    let results = std::fs::read_to_string(root.path().join("evidence/result.jsonl")).unwrap();
    assert!(
        results
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .any(|row| row["status"] == "FAIL")
    );
}
