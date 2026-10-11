use std::path::{Path, PathBuf};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}

fn port() -> u16 {
    std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn execute(fail: bool, install: bool) -> (tempfile::TempDir, std::process::Output) {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();
    std::fs::write(
        root_path.join("scenario"),
        if install {
            b"sdk-install".as_slice()
        } else {
            b"success".as_slice()
        },
    )
    .unwrap();
    std::fs::write(root_path.join("model.gguf"), b"inert fixture").unwrap();
    let binary = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_smoke_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
    let mut command = std::process::Command::new("bash");
    command
        .arg(repository().join("scripts/ci-sdk-fixture.sh"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .arg(&binary)
        .arg("unused-bin-dir")
        .arg(root_path.join("model.gguf"))
        .arg("--")
        .arg(&binary)
        .arg("--sdk-consumer")
        .arg(&root_path)
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &root_path)
        .env_remove("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")
        .env("MESH_SDK_API_PORT", port().to_string())
        .env("MESH_SDK_CONSOLE_PORT", port().to_string())
        .env("MESH_SDK_MAX_WAIT", "5")
        .env("MESH_SDK_COMMAND_MAX_WAIT", "5");
    if fail {
        command.arg("fail");
    }
    if install {
        command.env("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR", &root_path);
    }
    let output = command.output().unwrap();
    (root, output)
}

#[test]
fn fixture_hands_ready_identity_to_consumer_while_daemon_is_live() {
    let (root, output) = execute(false, false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(root.path().join("consumer-handoff")).unwrap(),
        b"ready"
    );
}

#[test]
fn consumer_failure_rejects_fixture_after_ready_handoff() {
    let (root, output) = execute(true, false);
    assert!(!output.status.success());
    assert_eq!(
        std::fs::read(root.path().join("consumer-handoff")).unwrap(),
        b"ready"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("sdk-fixture-invite"));
}

#[test]
fn artifact_installation_finishes_before_daemon_and_consumer_start() {
    let (root, output) = execute(false, true);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(root.path().join("installed")).unwrap(),
        b"installed"
    );
    assert_eq!(
        std::fs::read(root.path().join("consumer-handoff")).unwrap(),
        b"ready"
    );
}
