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

use mesh_llm_config::{
    ConfigDiagnosticSeverity, ConfigSupportState, MeshConfig, WIRING_MANIFEST, WiringBehavior,
    WiringStatus, built_in_config_schema, validate_config_diagnostics,
};

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
fn fit_target_mib_manifest_matches_its_tuner_and_live_resolver_consumers() {
    let entry = WIRING_MANIFEST
        .iter()
        .find(|entry| entry.path == "hardware.fit_target_mib")
        .expect("fit_target_mib must remain in the exhaustive wiring manifest");

    assert_eq!(entry.status, WiringStatus::Wired);
    assert_eq!(entry.behavior, WiringBehavior::None);
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

#[test]
fn prompt_shape_metrics_are_accepted_for_bounded_otlp_export() {
    let config: MeshConfig = toml::from_str(
        r#"
[telemetry]
prompt_shape_metrics = true
"#,
    )
    .expect("config should parse");

    let diagnostics = validate_config_diagnostics(&config);

    assert!(
        diagnostics.is_empty(),
        "prompt-shape metrics must pass static validation once their bounded exporter is wired: {diagnostics:#?}"
    );
}

#[test]
fn wiring_manifest_covers_every_builtin_schema_path_in_both_directions() {
    let manifest = WIRING_MANIFEST
        .iter()
        .map(|entry| entry.path.trim_end_matches(".*"))
        .collect::<std::collections::BTreeSet<_>>();
    let schema = built_in_config_schema();
    let mut normalized = schema
        .settings
        .iter()
        .map(|setting| {
            let path = setting
                .path
                .render()
                .replace("plugin.<plugin-name>", "plugin.<name>");
            path.strip_prefix("defaults.")
                .or_else(|| path.strip_prefix("models.<model-ref>."))
                .unwrap_or(&path)
                .trim_end_matches(".*")
                .to_string()
        })
        .collect::<std::collections::BTreeSet<_>>();
    normalized.insert("plugin.<name>.settings".to_string());

    let missing = normalized
        .iter()
        .filter(|path| !manifest.contains(path.as_str()))
        .collect::<Vec<_>>();
    let stale = manifest
        .iter()
        .filter(|path| !normalized.contains(**path))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "schema paths missing from wiring manifest: {missing:?}"
    );
    assert!(
        stale.is_empty(),
        "wiring manifest paths missing from schema: {stale:?}"
    );
}

#[test]
fn builtin_schema_support_matches_structured_wiring_status() {
    let schema = built_in_config_schema();
    let mismatches = schema
        .settings
        .iter()
        .filter_map(|setting| {
            let rendered = setting
                .path
                .render()
                .replace("plugin.<plugin-name>", "plugin.<name>");
            let path = rendered
                .strip_prefix("defaults.")
                .or_else(|| rendered.strip_prefix("models.<model-ref>."))
                .unwrap_or(&rendered);
            let entry = WIRING_MANIFEST
                .iter()
                .find(|entry| entry.path.trim_end_matches(".*") == path)
                .unwrap_or_else(|| panic!("schema path {rendered} must have a wiring entry"));
            let expected = match (entry.status, entry.behavior) {
                (WiringStatus::Wired, WiringBehavior::None) => ConfigSupportState::Supported,
                (WiringStatus::Partial, WiringBehavior::None) => ConfigSupportState::Experimental,
                (WiringStatus::Partial, WiringBehavior::SilentNoOp) => {
                    ConfigSupportState::Experimental
                }
                (WiringStatus::Unwired, WiringBehavior::BailsDownstream) => {
                    ConfigSupportState::Unwired
                }
                (WiringStatus::Unwired, WiringBehavior::SilentNoOp) => {
                    ConfigSupportState::Unsupported
                }
                (WiringStatus::Rejected, WiringBehavior::Rejected) => ConfigSupportState::Rejected,
                combination => panic!(
                    "wiring entry {path} has inconsistent status and behavior: {combination:?}"
                ),
            };
            (setting.support != expected).then_some((rendered, setting.support, expected))
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "built-in schema support disagrees with structured wiring status: {mismatches:#?}"
    );
}

#[test]
fn native_mmap_controls_and_partial_tensor_split_are_rejected() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
use_mmap_prefetch = true
use_mmap_buffer = true
split_mode = "tensor"
"#,
    )
    .expect("documented native controls must parse");

    let diagnostics = validate_config_diagnostics(&config);

    let rejected = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.as_ref().map(|path| path.render()))
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        rejected,
        std::collections::BTreeSet::from([
            "defaults.hardware.use_mmap_buffer".to_string(),
            "defaults.hardware.use_mmap_prefetch".to_string(),
            "defaults.hardware.split_mode".to_string(),
        ])
    );
}

#[test]
fn every_silent_no_op_field_is_an_early_validation_error() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
model_runtime = "cuda"
fit_target_mib = 8192
split_mode = "row"
main_gpu = 0
fit_context = true
lora_adapters = ["adapter.gguf"]
control_vectors = ["control.gguf"]
check_tensors = true
direct_io = true
repack = true
op_offload = true
no_host_buffer = true
use_mmap_prefetch = true
use_mmap_buffer = true
warmup = true
"#,
    )
    .expect("unsupported controls must still parse before validation");

    let diagnostics = validate_config_diagnostics(&config);
    let rejected = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.as_ref().map(|path| path.render()))
        .collect::<std::collections::BTreeSet<_>>();
    let expected = WIRING_MANIFEST
        .iter()
        .filter(|entry| entry.behavior == WiringBehavior::SilentNoOp)
        .map(|entry| format!("defaults.{}", entry.path))
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(rejected, expected);
}

#[test]
fn closeout_audit_names_every_manifest_row_and_required_boundary() {
    let audit_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/CONFIGURATION_PR8_CLOSEOUT_AUDIT.md");
    let audit = std::fs::read_to_string(&audit_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", audit_path.display()));

    let reverse_audit = audit
        .split_once("## Reverse audit")
        .and_then(|(_, suffix)| suffix.split_once("## Closeout defects"))
        .map(|(section, _)| section)
        .expect("audit must contain a bounded reverse-audit section");
    for entry in WIRING_MANIFEST {
        assert!(
            reverse_audit.contains(&format!("`{}`", entry.path)),
            "reverse audit inventory is missing manifest path {}",
            entry.path
        );
    }
    for entry in WIRING_MANIFEST.iter().filter(|entry| entry.owner == "PR8") {
        let row_prefix = format!("| `{}` |", entry.path);
        let row = audit
            .lines()
            .find(|line| line.starts_with(&row_prefix))
            .unwrap_or_else(|| panic!("forward audit is missing row {}", entry.path));
        assert!(
            row.matches('|').count() >= 6 && row.contains("::tests::"),
            "forward audit row lacks executable evidence: {row}"
        );
    }
    for boundary in [
        "parsed",
        "validated",
        "final consumer",
        "reverse audit",
        "hardware limitation",
    ] {
        assert!(
            audit.to_ascii_lowercase().contains(boundary),
            "closeout audit is missing required evidence boundary {boundary:?}"
        );
    }
}
