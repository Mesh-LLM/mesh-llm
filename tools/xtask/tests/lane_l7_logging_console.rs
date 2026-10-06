use std::path::Path;

// Each narrative releases a reserved port range before its child binds it.
// Keep this target's narratives exclusive across that handoff window.
static NARRATIVE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn reserve_ports() -> (u16, Vec<std::net::TcpListener>, Vec<std::net::UdpSocket>) {
    for _ in 0..128 {
        let first = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let base = first.local_addr().unwrap().port();
        if !(1025..=65533).contains(&base) {
            continue;
        }
        let mut tcp = vec![first];
        let mut udp = Vec::new();
        for port in base..=base + 2 {
            if port != base {
                let Ok(listener) =
                    std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
                else {
                    break;
                };
                tcp.push(listener);
            }
            let Ok(socket) = std::net::UdpSocket::bind((std::net::Ipv4Addr::LOCALHOST, port))
            else {
                break;
            };
            udp.push(socket);
        }
        if tcp.len() == 3 && udp.len() == 3 {
            return (base, tcp, udp);
        }
    }
    panic!("no complete logging-console port range available");
}

fn run(browser_fail: bool) -> (tempfile::TempDir, std::process::Output) {
    let _exclusive = NARRATIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    let profile = Path::new(env!("CARGO_BIN_EXE_xtask")).parent().unwrap();
    let fixture = profile.join("examples").join(format!(
        "l7_logging_fixture{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::create_dir(location.join("bin")).unwrap();
    std::fs::copy(
        &fixture,
        location
            .join("bin")
            .join(format!("pnpm{}", std::env::consts::EXE_SUFFIX)),
    )
    .unwrap();
    if browser_fail {
        std::fs::write(location.join("browser-fail"), b"fail").unwrap();
    }
    let (port, tcp, udp) = reserve_ports();
    let paths = std::iter::once(location.join("bin"))
        .chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ))
        .collect::<Vec<_>>();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut command = std::process::Command::new("bash");
    command
        .arg(repository.join("scripts/qa-logging-console-e2e.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg("--current-binary")
        .arg(&fixture)
        .arg("--evidence-dir")
        .arg(location.join("evidence"))
        .args([
            "--base-port",
            &port.to_string(),
            "--max-wait",
            "5",
            "--keep-state",
        ])
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location)
        .env("PATH", std::env::join_paths(paths).unwrap());
    drop(tcp);
    drop(udp);
    let output = command.output().unwrap();
    (root, output)
}

#[test]
fn console_narrative_preserves_request_across_restart_and_executes_browser() {
    let (root, output) = run(false);
    let directory = std::fs::read_dir(root.path().join("evidence"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert!(
        output.status.success(),
        "{}; {}; {}",
        String::from_utf8_lossy(&output.stderr),
        std::fs::read_to_string(directory.join("summary.json")).unwrap(),
        std::fs::read_to_string(directory.join("logs/initial.stderr.log")).unwrap_or_default()
    );
    assert!(root.path().join("browser-ran").is_file());
    let directory = std::fs::read_dir(root.path().join("evidence"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert_eq!(
        std::fs::read_to_string(directory.join("state/launch-count")).unwrap(),
        "2"
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["overall"], "pass");
    assert!(
        summary["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == "restart_persistence")
    );
}

#[test]
fn browser_failure_rejects_console_narrative_and_preserves_summary() {
    let (root, output) = run(true);
    assert!(!output.status.success());
    let directory = std::fs::read_dir(root.path().join("evidence"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["overall"], "fail");
}
