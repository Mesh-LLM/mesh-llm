use super::{diagnostic_for_canonical, diagnostics_from_toml, rendered};
use crate::{ConfigDiagnosticCode, ConfigDiagnosticSeverity, validate_config};
use std::collections::BTreeSet;

const VALID_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_valid.toml"
));
const INVALID_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_invalid.toml"
));

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DiagnosticSignature {
    path: String,
    canonical_path: String,
    severity: &'static str,
    code: &'static str,
}

impl DiagnosticSignature {
    fn new(path: &str, canonical_path: &str, severity: &'static str, code: &'static str) -> Self {
        Self {
            path: path.to_string(),
            canonical_path: canonical_path.to_string(),
            severity,
            code,
        }
    }
}

fn severity_label(severity: ConfigDiagnosticSeverity) -> &'static str {
    match severity {
        ConfigDiagnosticSeverity::Error => "error",
        ConfigDiagnosticSeverity::Warning => "warning",
        ConfigDiagnosticSeverity::Info => "info",
    }
}

fn code_label(code: ConfigDiagnosticCode) -> &'static str {
    match code {
        ConfigDiagnosticCode::InvalidValue => "invalid_value",
        ConfigDiagnosticCode::MissingRequiredValue => "missing_required_value",
        ConfigDiagnosticCode::UnknownField => "unknown_field",
        ConfigDiagnosticCode::UnsupportedField => "unsupported_field",
        ConfigDiagnosticCode::RejectedField => "rejected_field",
        ConfigDiagnosticCode::MisplacedField => "misplaced_field",
        ConfigDiagnosticCode::SchemaUnavailable => "schema_unavailable",
        ConfigDiagnosticCode::LegacyUnvalidatedConfig => "legacy_unvalidated_config",
        ConfigDiagnosticCode::AliasApplied => "alias_applied",
        ConfigDiagnosticCode::UnsupportedSchemaVersion => "unsupported_schema_version",
    }
}

fn signatures_from_toml(raw: &str) -> BTreeSet<DiagnosticSignature> {
    diagnostics_from_toml(raw)
        .into_iter()
        .map(|diagnostic| {
            DiagnosticSignature::new(
                rendered(&diagnostic.path)
                    .as_deref()
                    .expect("fixture diagnostics should carry a rendered path"),
                rendered(&diagnostic.canonical_path)
                    .as_deref()
                    .expect("fixture diagnostics should carry a canonical path"),
                severity_label(diagnostic.severity),
                code_label(diagnostic.code),
            )
        })
        .collect()
}

#[test]
fn validate_schema_contract_fixture_accepts_full_surface_valid_controls() {
    let config = toml::from_str(VALID_FIXTURE).expect("valid fixture should deserialize");

    validate_config(&config).expect("valid schema-driven control fixture should validate");
    assert!(diagnostics_from_toml(VALID_FIXTURE).is_empty());
}

#[test]
fn validate_schema_contract_fixture_reports_stable_canonical_signatures() {
    let signatures = signatures_from_toml(INVALID_FIXTURE);

    assert_eq!(
        signatures,
        BTreeSet::from([
            DiagnosticSignature::new(
                "defaults.model_fit.ubatch",
                "defaults.model_fit.ubatch",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "mesh_requirements.require_release_attestation",
                "mesh_requirements.require_release_attestation",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[0].hardware.device",
                "models.<model-ref>.hardware.device",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[1].hardware.hf_file",
                "models.<model-ref>.hardware.hf_file",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[2].hardware.stage_layer_start",
                "models.<model-ref>.hardware.stage_layer_start",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[3].model_fit.keep_tokens",
                "models.<model-ref>.model_fit.keep_tokens",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[4].model_fit.cache_idle_slots",
                "models.<model-ref>.model_fit.cache_idle_slots",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[5].skippy.prefill_chunk_schedule",
                "models.<model-ref>.skippy.prefill_chunk_schedule",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[6].speculative.draft_hf_file",
                "models.<model-ref>.speculative.draft_hf_file",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[7].speculative.draft_min_tokens",
                "models.<model-ref>.speculative.draft_min_tokens",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[8].speculative.ngram_max",
                "models.<model-ref>.speculative.ngram_max",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[9].request_defaults.mirostat_mode",
                "models.<model-ref>.request_defaults.mirostat_mode",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[10].multimodal.mmproj",
                "models.<model-ref>.multimodal.mmproj",
                "error",
                "invalid_value",
            ),
            DiagnosticSignature::new(
                "models[11].hardware.rpc_backend",
                "models.<model-ref>.hardware.rpc_backend",
                "error",
                "rejected_field",
            ),
            DiagnosticSignature::new(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr",
                "error",
                "invalid_value",
            ),
        ])
    );

    let diagnostics = diagnostics_from_toml(INVALID_FIXTURE);
    let rejected =
        diagnostic_for_canonical(&diagnostics, "models.<model-ref>.hardware.rpc_backend");
    assert_eq!(rejected.severity, ConfigDiagnosticSeverity::Error);
    assert_eq!(rejected.code, ConfigDiagnosticCode::RejectedField);
}
