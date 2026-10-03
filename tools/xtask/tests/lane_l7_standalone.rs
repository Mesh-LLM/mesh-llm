use std::path::{Path, PathBuf};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}

#[test]
fn standalone_shell_runs_all_checks_with_explicit_sizing() {
    let root = tempfile::tempdir().unwrap();
    let location = root.path().canonicalize().unwrap();
    std::fs::write(location.join("scenario"), b"success").unwrap();
    std::fs::write(location.join("model.gguf"), b"inert fixture").unwrap();
    let listeners: Vec<_> = (0..4)
        .map(|_| std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let ports: Vec<_> = listeners
        .iter()
        .map(|listener| listener.local_addr().unwrap().port())
        .collect();
    drop(listeners);
    let binary = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_smoke_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
    let output = std::process::Command::new("bash")
        .arg(repository().join("scripts/ci-smoke-test.sh"))
        .arg(binary)
        .arg("unused-bin-dir")
        .arg(location.join("model.gguf"))
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &location)
        .env("MESH_CI_API_PORT", ports[0].to_string())
        .env("MESH_CI_CONSOLE_PORT", ports[1].to_string())
        .env("MESH_CI_HEADLESS_API_PORT", ports[2].to_string())
        .env("MESH_CI_HEADLESS_CONSOLE_PORT", ports[3].to_string())
        .env("MESH_CI_CTX_SIZE", "128")
        .env("MESH_CI_BATCH_SIZE", "64")
        .env("MESH_CI_UBATCH_SIZE", "32")
        .env("MESH_CI_MAX_WAIT", "5")
        .env_remove("MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE")
        .env_remove("MESH_TOKIO_STACK_SIZE")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["status"], "passed");
    for name in ["primary-audit.json", "headless-audit.json"] {
        let audit: serde_json::Value =
            serde_json::from_slice(&std::fs::read(location.join(name)).unwrap()).unwrap();
        let arguments = audit["arguments"].as_array().unwrap();
        assert!(
            arguments
                .windows(2)
                .any(|pair| pair[0] == "--ctx-size" && pair[1] == "128")
        );
        let config: toml::Value = toml::from_str(audit["config"].as_str().unwrap()).unwrap();
        assert_eq!(
            config["defaults"]["model_fit"]["batch"].as_integer(),
            Some(64)
        );
        assert_eq!(
            config["defaults"]["model_fit"]["ubatch"].as_integer(),
            Some(32)
        );
    }
}
