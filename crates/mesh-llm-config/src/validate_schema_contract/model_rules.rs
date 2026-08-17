use super::{
    assert_range, assert_requires, diagnostic_for_canonical, diagnostics_from_toml, rendered,
    schema_setting,
};
use crate::{
    ConfigConditionOperator, ConfigConditionValue, ConfigDiagnosticCode, ConfigDisabledWritePolicy,
    ConfigOptionsSource,
};

#[test]
fn validate_schema_contract_keeps_gpu_assignment_validation_authoritative() {
    let setting = schema_setting("models.<model-ref>.hardware.device");
    let behavior = setting
        .control_behavior
        .as_ref()
        .expect("device control behavior should exist");

    assert_eq!(
        behavior.options_source,
        Some(ConfigOptionsSource::RuntimeGpus)
    );
    assert_eq!(behavior.enable_when.len(), 1);
    assert_eq!(behavior.enable_when[0].path.render(), "gpu.assignment");
    assert_eq!(
        behavior.enable_when[0].operator,
        ConfigConditionOperator::Equals
    );
    assert_eq!(
        behavior.enable_when[0].values,
        vec![ConfigConditionValue::String("pinned".into())]
    );
    assert_eq!(behavior.disable_when.len(), 1);
    assert_eq!(
        behavior.disable_when[0].condition.path.render(),
        "gpu.assignment"
    );
    assert_eq!(
        behavior.disable_when[0].condition.values,
        vec![ConfigConditionValue::String("auto".into())]
    );
    assert_eq!(
        behavior.disable_when[0].write_policy,
        ConfigDisabledWritePolicy::OmitWhenDisabled
    );

    let diagnostics = diagnostics_from_toml(
        r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-4B-Q4_K_M"

[models.hardware]
device = "metal:0"
"#,
    );
    let diagnostic = diagnostic_for_canonical(&diagnostics, "models.<model-ref>.hardware.device");

    assert_eq!(diagnostic.code, ConfigDiagnosticCode::InvalidValue);
    assert_eq!(
        rendered(&diagnostic.path).as_deref(),
        Some("models[0].hardware.device")
    );
    assert_eq!(
        diagnostic.message,
        "models[0].hardware.device must not be set when gpu.assignment = \"auto\""
    );

    let default_diagnostics = diagnostics_from_toml(
        r#"
[gpu]
assignment = "auto"

[defaults.hardware]
device = "metal:0"
"#,
    );
    let default_diagnostic =
        diagnostic_for_canonical(&default_diagnostics, "defaults.hardware.device");
    assert_eq!(default_diagnostic.code, ConfigDiagnosticCode::InvalidValue);
    assert_eq!(
        rendered(&default_diagnostic.path).as_deref(),
        Some("defaults.hardware.device")
    );
}

#[test]
fn validate_schema_contract_aligns_pairing_and_relative_bound_rules() {
    assert_range(
        &schema_setting("defaults.model_fit.ubatch"),
        None,
        Some("defaults.model_fit.batch"),
    );
    assert_requires(
        &schema_setting("models.<model-ref>.hardware.stage_layer_end"),
        "models.<model-ref>.hardware.stage_layer_start",
    );
    assert_range(
        &schema_setting("models.<model-ref>.hardware.stage_layer_end"),
        Some("models.<model-ref>.hardware.stage_layer_start"),
        None,
    );
    assert_requires(
        &schema_setting("models.<model-ref>.hardware.hf_file"),
        "models.<model-ref>.hardware.hf_repo",
    );
    assert_requires(
        &schema_setting("defaults.speculative.draft_hf_file"),
        "defaults.speculative.draft_hf_repo",
    );
    assert_range(
        &schema_setting("defaults.speculative.draft_min_tokens"),
        None,
        Some("defaults.speculative.draft_max_tokens"),
    );
    assert_range(
        &schema_setting("defaults.speculative.ngram_max"),
        Some("defaults.speculative.ngram_min"),
        None,
    );
    assert_range(
        &schema_setting("defaults.multimodal.image_min_tokens"),
        None,
        Some("defaults.multimodal.image_max_tokens"),
    );

    for (raw, canonical_path) in [
        (
            "[defaults.model_fit]\nbatch = 4\nubatch = 8\n",
            "defaults.model_fit.ubatch",
        ),
        (
            "[[models]]\nmodel = \"Qwen3-4B-Q4_K_M\"\n[models.hardware]\nstage_layer_start = 8\n",
            "models.<model-ref>.hardware.stage_layer_end",
        ),
        (
            "[[models]]\nmodel = \"Qwen3-4B-Q4_K_M\"\n[models.hardware]\nhf_repo = \"mesh/test\"\n",
            "models.<model-ref>.hardware.hf_file",
        ),
        (
            "[defaults.speculative]\ndraft_hf_repo = \"mesh/test\"\n",
            "defaults.speculative.draft_hf_file",
        ),
        (
            "[defaults.speculative]\ndraft_max_tokens = 4\ndraft_min_tokens = 8\n",
            "defaults.speculative.draft_min_tokens",
        ),
        (
            "[defaults.speculative]\nngram_min = 4\nngram_max = 2\n",
            "defaults.speculative.ngram_max",
        ),
        (
            "[defaults.multimodal]\nimage_min_tokens = 400\nimage_max_tokens = 200\n",
            "defaults.multimodal.image_min_tokens",
        ),
    ] {
        let diagnostics = diagnostics_from_toml(raw);
        let diagnostic = diagnostic_for_canonical(&diagnostics, canonical_path);
        assert_eq!(
            diagnostic.code,
            ConfigDiagnosticCode::InvalidValue,
            "{canonical_path}"
        );
    }
}
