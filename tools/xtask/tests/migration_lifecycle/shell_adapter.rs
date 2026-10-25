use super::*;

#[test]
fn readiness_shell_adapter_uses_rust_owner_and_cleans_fixture() {
    let case = Case::new(Behavior::Clean, vec![ready()]);
    let output = std::process::Command::new("bash")
        .arg(repository().join("scripts/ci-client-readiness-smoke.sh"))
        .args([&case.binary, &case.native])
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .env("MESH_LLM_CLIENT_STATE_PARENT", &case.state)
        .env("MESH_LLM_CLIENT_READY_MAX_WAIT", "1")
        .env("MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        case.audit().arguments[5..],
        ["client", "--mesh-discovery-mode", "mdns"]
    );
    case.assert_removed();
}
