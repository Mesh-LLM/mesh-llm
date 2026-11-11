use std::path::Path;
use std::process::Command;

fn auth_policy_command(
    root: &Path,
    config: &Path,
    trust_store: &Path,
    audit_log: Option<&Path>,
) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mesh-llm"));
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("MESH_LLM_") {
            command.env_remove(key);
        }
    }
    command.arg("--config").arg(config);
    if let Some(path) = audit_log {
        command.arg("--audit-log-path").arg(path);
    }
    command
        .args(["auth", "trust", "policy", "allowlist", "--trust-store"])
        .arg(trust_store)
        .env("HOME", root)
        .env("USERPROFILE", root)
        .env("MESH_LLM_DATA_DIR", root.join("data"))
        .env("DO_NOT_TRACK", "1");
    command
}

// A one-shot `auth` process installs the audit file sink configured in
// `[logging.audit]`, the one `serve` writes to, before the command records
// its admin action.
#[test]
fn auth_trust_policy_writes_its_record_to_the_configured_audit_file() {
    let temp = tempfile::tempdir().unwrap();
    let audit_log = temp.path().join("audit").join("audit.jsonl");
    let config = temp.path().join("config.toml");
    std::fs::write(
        &config,
        format!("[logging.audit]\nlog_path = '{}'\n", audit_log.display()),
    )
    .unwrap();
    let trust_store = temp.path().join("trusted-owners.json");

    let output = auth_policy_command(temp.path(), &config, &trust_store, None)
        .output()
        .expect("mesh-llm command should run");
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "opening the audit file should not write to stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let records = std::fs::read_to_string(&audit_log).expect("audit file should exist");
    let record = records
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .find(|record| record["action"] == "auth_trust_policy")
        .unwrap_or_else(|| panic!("no auth_trust_policy record in {records:?}"));
    assert_eq!(record["category"], "admin_action");
    assert_eq!(record["resource"], "allowlist");
    assert_eq!(record["outcome"], "success");
}

#[test]
fn auth_trust_policy_does_not_change_the_store_when_audit_initialization_fails() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "").unwrap();
    let trust_store = temp.path().join("trusted-owners.json");
    let blocked_parent = temp.path().join("not-a-directory");
    std::fs::write(&blocked_parent, "occupied").unwrap();

    let output = auth_policy_command(
        temp.path(),
        &config,
        &trust_store,
        Some(&blocked_parent.join("audit.jsonl")),
    )
    .output()
    .expect("mesh-llm command should run");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Failed to initialize auth audit logging"),
        "{stderr}"
    );
    assert!(
        !trust_store.exists(),
        "audit setup must precede the mutation"
    );
}
