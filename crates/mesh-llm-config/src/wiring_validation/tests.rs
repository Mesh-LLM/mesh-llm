use super::*;
use crate::MeshConfig;

#[test]
fn rejected_request_defaults_field_produces_rejected_field_error() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.request_defaults]
grammar = "root ::= \"a\""
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    let diagnostic = diagnostics
        .iter()
        .find(|d| {
            d.path.as_ref().map(ConfigPath::render)
                == Some("defaults.request_defaults.grammar".to_string())
        })
        .expect("grammar should produce a diagnostic");
    assert_eq!(diagnostic.code, ConfigDiagnosticCode::RejectedField);
    assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
}

#[test]
fn empty_dry_defaults_produce_no_diagnostic_at_path() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.request_defaults.dry]
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().all(|diagnostic| {
        diagnostic.path.as_ref().map(ConfigPath::render)
            != Some("defaults.request_defaults.dry".to_string())
    }));
}

#[test]
fn empty_xtc_defaults_produce_no_diagnostic_at_path() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.request_defaults.xtc]
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().all(|diagnostic| {
        diagnostic.path.as_ref().map(ConfigPath::render)
            != Some("defaults.request_defaults.xtc".to_string())
    }));
}

#[test]
fn empty_adaptive_defaults_retain_rejected_field_error() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.request_defaults.adaptive]
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    let matching = diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic.path.as_ref().map(ConfigPath::render)
                == Some("defaults.request_defaults.adaptive".to_string())
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0].code, ConfigDiagnosticCode::RejectedField);
    assert_eq!(matching[0].severity, ConfigDiagnosticSeverity::Error);
}

#[test]
fn bails_downstream_field_produces_error_before_model_load() {
    let config: MeshConfig = toml::from_str(
        r#"
[[models]]
model = "test-model"

[models.model_fit]
keep_tokens = 32
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().any(|d| {
        d.path.as_ref().map(ConfigPath::render)
            == Some("models[0].model_fit.keep_tokens".to_string())
            && d.severity == ConfigDiagnosticSeverity::Error
    }));
}

#[test]
fn silent_no_op_field_produces_warning() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
warmup = true
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().any(|d| {
        d.path.as_ref().map(ConfigPath::render) == Some("defaults.hardware.warmup".to_string())
            && d.severity == ConfigDiagnosticSeverity::Warning
    }));
}

#[test]
fn spec_default_true_produces_no_diagnostic_at_path() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.speculative]
spec_default = true
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().all(|diagnostic| {
        diagnostic.path.as_ref().map(ConfigPath::render)
            != Some("defaults.speculative.spec_default".to_string())
    }));
}

#[test]
fn spec_default_false_produces_no_diagnostic_at_path() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.speculative]
spec_default = false
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().all(|diagnostic| {
        diagnostic.path.as_ref().map(ConfigPath::render)
            != Some("defaults.speculative.spec_default".to_string())
    }));
}

#[test]
fn spec_default_auto_produces_no_diagnostic_at_path() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.speculative]
spec_default = "auto"
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.iter().all(|diagnostic| {
        diagnostic.path.as_ref().map(ConfigPath::render)
            != Some("defaults.speculative.spec_default".to_string())
    }));
}

#[test]
fn wired_fields_produce_no_manifest_diagnostics() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.model_fit]
ctx_size = 4096
"#,
    )
    .expect("config should parse");

    let diagnostics = wiring_manifest_diagnostics(&config);
    assert!(diagnostics.is_empty());
}

#[test]
fn unset_fields_produce_no_diagnostics() {
    let config = MeshConfig::default();
    assert!(wiring_manifest_diagnostics(&config).is_empty());
}
