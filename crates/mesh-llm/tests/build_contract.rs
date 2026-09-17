use std::process::Command;

#[test]
fn compiled_build_contract_needs_no_runtime_or_user_state() {
    let home = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args(["--log-format", "json", "--print-build-contract"])
        .env("HOME", home.path())
        .env("MESH_LLM_DATA_DIR", home.path().join("data"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            home.path().join("absent-cache"),
        )
        .env(
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
            home.path().join("absent-bundle"),
        )
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["skippy_abi"],
        mesh_llm_system::native_runtime_install::current_skippy_abi_version()
    );
    assert!(
        value["product_version"]
            .as_str()
            .is_some_and(|v| !v.is_empty())
    );
    assert!(
        value["runtime_release"]
            .as_str()
            .is_some_and(|v| !v.is_empty())
    );
    assert_eq!(std::fs::read_dir(home.path()).unwrap().count(), 0);
}
