fn tensor_split_diagnostics(config: &str) -> Vec<crate::ConfigDiagnostic> {
    let config: MeshConfig = toml::from_str(config).expect("tensor split config should parse");
    validate_config_diagnostics(&config)
}

#[test]
fn defaults_tensor_split_rejects_an_explicit_empty_ratio_list() {
    let diagnostics = tensor_split_diagnostics(
        r#"
[defaults.hardware]
tensor_split = []
"#,
    );

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic
                .path
                .as_ref()
                .is_some_and(|path| path.render() == "defaults.hardware.tensor_split")
                && diagnostic
                    .message
                    .contains("must contain at least one ratio")
        }),
        "explicit empty defaults tensor_split should fail static validation: {diagnostics:#?}"
    );
}

#[test]
fn model_tensor_split_rejects_an_explicit_empty_ratio_list() {
    let diagnostics = tensor_split_diagnostics(
        r#"
[[models]]
model = "Qwen/Qwen3-4B-GGUF:Q4_K_M"

[models.hardware]
tensor_split = []
"#,
    );

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic
                .path
                .as_ref()
                .is_some_and(|path| path.render() == "models[0].hardware.tensor_split")
                && diagnostic
                    .message
                    .contains("must contain at least one ratio")
        }),
        "explicit empty model tensor_split should fail static validation: {diagnostics:#?}"
    );
}

#[test]
fn non_empty_tensor_split_keeps_the_unwired_policy_diagnostic() {
    let diagnostics = tensor_split_diagnostics(
        r#"
[defaults.hardware]
tensor_split = [0.6, 0.4]
"#,
    );

    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code == crate::ConfigDiagnosticCode::InvalidValue
                && diagnostic
                    .path
                    .as_ref()
                    .is_some_and(|path| path.render() == "defaults.hardware.tensor_split")
                && diagnostic.message.contains("fails downstream today")
        }),
        "non-empty tensor_split must retain its unwired-policy diagnostic: {diagnostics:#?}"
    );
}
