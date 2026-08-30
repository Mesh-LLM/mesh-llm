//! Permanent contract test for issue #1462 PR 1: static
//! `mesh-llm config validate` must produce a diagnostic for every manifest
//! entry whose wiring status is not `Wired`.
//!
//! `model_fit.keep_tokens` is a `basic_setting` (fully `Supported`) on the
//! built-in config schema, and static validation accepts any positive
//! integer for it. But the embedded staged runtime resolver
//! (`reject_unsupported_model_fit_controls` in
//! `crates/mesh-llm-host-runtime/src/inference/skippy/resolver/support.rs`)
//! bails at model load for any positive value:
//! `if config.keep_tokens.unwrap_or(0) > 0 { bail!(...) }`. This test
//! guards that gap: as long as `model_fit.keep_tokens` remains `Unwired`
//! with a `BailsDownstream` behavior in the manifest, static
//! `mesh-llm config validate` must flag `model_fit.keep_tokens = 128` as an
//! error rather than reporting the config valid.

use mesh_llm_config::{ConfigDiagnosticSeverity, MeshConfig, validate_config_diagnostics};

#[test]
fn static_validation_flags_a_field_that_bails_at_model_load() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.model_fit]
keep_tokens = 128
"#,
    )
    .expect("config should parse");

    let diagnostics = validate_config_diagnostics(&config);
    let has_keep_tokens_error = diagnostics.iter().any(|diagnostic| {
        diagnostic
            .path
            .as_ref()
            .is_some_and(|path| path.render() == "defaults.model_fit.keep_tokens")
            && diagnostic.severity == ConfigDiagnosticSeverity::Error
    });

    assert!(
        has_keep_tokens_error,
        "model_fit.keep_tokens > 0 fails at model load in the embedded staged runtime \
         (reject_unsupported_model_fit_controls in \
         mesh-llm-host-runtime/src/inference/skippy/resolver/support.rs), but static \
         `mesh-llm config validate` reported no error for it: {diagnostics:#?}"
    );
}

#[test]
fn pr4_supported_fields_do_not_emit_unsupported_diagnostics() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.throughput]
continuous_batching = false
tuning_profile = "saver"

[defaults.skippy]
lifecycle_startup_timeout_ms = 120000
lifecycle_readiness_interval_ms = 125
lifecycle_health_interval_ms = 5000
"#,
    )
    .expect("config should parse");

    let diagnostics = validate_config_diagnostics(&config);

    assert!(
        diagnostics.is_empty(),
        "wired PR4 fields must pass static validation without unsupported warnings: {diagnostics:#?}"
    );
}

#[test]
fn pr4_fields_without_runtime_consumers_are_rejected() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.throughput]
priority = "normal"
poll = "busy"
cpu_affinity = "0-3"
numa = "distribute"
slot_prompt_similarity = 0.75

[defaults.skippy]
stage_model_path = "/models/stage.gguf"
stage_role = "middle"
stage_topology = "legacy-lock"
binary_stage_transport = "binary"
"#,
    )
    .expect("config should parse");

    let diagnostics = validate_config_diagnostics(&config);
    let rejected = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.as_ref().map(|path| path.render()))
        .collect::<std::collections::BTreeSet<_>>();

    let expected = std::collections::BTreeSet::from([
        "defaults.skippy.binary_stage_transport".to_string(),
        "defaults.skippy.stage_model_path".to_string(),
        "defaults.skippy.stage_role".to_string(),
        "defaults.skippy.stage_topology".to_string(),
        "defaults.throughput.cpu_affinity".to_string(),
        "defaults.throughput.numa".to_string(),
        "defaults.throughput.poll".to_string(),
        "defaults.throughput.priority".to_string(),
        "defaults.throughput.slot_prompt_similarity".to_string(),
    ]);
    assert_eq!(rejected, expected);
}
