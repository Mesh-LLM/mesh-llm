//! Exercise the standalone command's output streams without loading a model.
#[test]
fn example_config_is_one_json_document_on_stdout() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_skippy-server"))
        .arg("example-config")
        .output()
        .expect("start standalone command");
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    let config: skippy_protocol::StageConfig =
        serde_json::from_slice(&output.stdout).expect("stdout contains only stage config JSON");
    assert!(!config.stage_id.is_empty());
}
