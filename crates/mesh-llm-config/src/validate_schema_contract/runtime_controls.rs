use super::{
    assert_present_enable_when, assert_reject_when_disabled, assert_requires,
    assert_socket_addr_schema, diagnostic_for_canonical, diagnostics_from_toml, rendered,
    schema_setting,
};
use crate::{
    ConfigConditionValue, ConfigConstraint, ConfigDiagnosticCode, ConfigDisabledWritePolicy,
    ConfigOptionsSource, ConfigTextFormat, ConfigValueSchema,
};

#[test]
fn validate_schema_contract_covers_owner_control_attestation_and_plugin_timeouts() {
    let advertise = schema_setting("owner_control.advertise_addr");
    assert_socket_addr_schema(&advertise);
    assert_requires(&advertise, "owner_control.bind");
    assert_present_enable_when(&advertise);

    let signer_keys = schema_setting("mesh_requirements.release_signer_keys");
    let signer_behavior = signer_keys
        .control_behavior
        .as_ref()
        .expect("signer-key control behavior should exist");
    assert_eq!(
        signer_behavior.text_format,
        Some(ConfigTextFormat::Ed25519Key)
    );
    assert_eq!(
        signer_behavior.enable_when[0].values,
        vec![ConfigConditionValue::Bool(true)]
    );
    assert_eq!(
        signer_behavior.disable_when[0].write_policy,
        ConfigDisabledWritePolicy::OmitWhenDisabled
    );

    let plugin_timeout = schema_setting("plugin.<plugin-name>.startup.connect_timeout_secs");
    let plugin_behavior = plugin_timeout
        .control_behavior
        .as_ref()
        .expect("plugin timeout control behavior should exist");
    assert_eq!(
        plugin_behavior
            .numeric
            .as_ref()
            .and_then(|numeric| numeric.min),
        Some(1.0)
    );
    assert_eq!(
        plugin_behavior
            .numeric
            .as_ref()
            .and_then(|numeric| numeric.unit.as_deref()),
        Some("sec")
    );

    let telemetry_service_name = schema_setting("telemetry.service_name");
    assert!(telemetry_service_name.constraints.iter().any(|constraint| {
        matches!(
            constraint,
            ConfigConstraint::AllowedPattern { pattern }
                if pattern == "^[A-Za-z0-9_-]+$"
        )
    }));

    let advertise_diagnostics =
        diagnostics_from_toml("[owner_control]\nadvertise_addr = \"127.0.0.1:17001\"\n");
    let advertise_diagnostic =
        diagnostic_for_canonical(&advertise_diagnostics, "owner_control.advertise_addr");
    assert_eq!(
        rendered(&advertise_diagnostic.path).as_deref(),
        Some("owner_control.advertise_addr")
    );

    let attestation_diagnostics =
        diagnostics_from_toml("[mesh_requirements]\nrequire_release_attestation = true\n");
    let attestation_diagnostic = diagnostic_for_canonical(
        &attestation_diagnostics,
        "mesh_requirements.require_release_attestation",
    );
    assert_eq!(
        attestation_diagnostic.code,
        ConfigDiagnosticCode::InvalidValue
    );

    let timeout_diagnostics = diagnostics_from_toml(
        r#"
[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 0
"#,
    );
    let timeout_diagnostic = diagnostic_for_canonical(
        &timeout_diagnostics,
        "plugin.<plugin-name>.startup.connect_timeout_secs",
    );
    assert_eq!(timeout_diagnostic.code, ConfigDiagnosticCode::InvalidValue);

    let service_name_diagnostics = diagnostics_from_toml(
        r#"
[telemetry]
service_name = "@@*(!111---aa"
"#,
    );
    let service_name_diagnostic =
        diagnostic_for_canonical(&service_name_diagnostics, "telemetry.service_name");
    assert_eq!(
        service_name_diagnostic.code,
        ConfigDiagnosticCode::InvalidValue
    );
}

#[test]
fn validate_schema_contract_aligns_choices_formats_and_rejected_fields() {
    let tuning_profile = schema_setting("defaults.throughput.tuning_profile");
    assert!(matches!(
        tuning_profile.value_schema,
        ConfigValueSchema::Enum { ref values } if values == &vec!["throughput".to_string(), "balanced".to_string(), "saver".to_string()]
    ));
    assert_eq!(
        tuning_profile
            .control_behavior
            .as_ref()
            .and_then(|behavior| behavior.options_source),
        Some(ConfigOptionsSource::Static)
    );

    let schedule = schema_setting("defaults.skippy.prefill_chunk_schedule");
    assert_eq!(
        schedule
            .control_behavior
            .as_ref()
            .and_then(|behavior| behavior.text_format),
        Some(ConfigTextFormat::CsvPositiveInts)
    );

    assert_reject_when_disabled(&schema_setting(
        "defaults.request_defaults.backend_sampling",
    ));
    assert_reject_when_disabled(&schema_setting("defaults.advanced.server.host"));

    for (raw, canonical_path, expected_code) in [
        (
            "[defaults.throughput]\nparallel = 0\n",
            "defaults.throughput.parallel",
            ConfigDiagnosticCode::InvalidValue,
        ),
        (
            "[defaults.skippy]\nprefill_chunk_schedule = \"1,0\"\n",
            "defaults.skippy.prefill_chunk_schedule",
            ConfigDiagnosticCode::InvalidValue,
        ),
        (
            "[defaults.request_defaults.backend_sampling]\nfoo = 1\n",
            "defaults.request_defaults.backend_sampling",
            ConfigDiagnosticCode::RejectedField,
        ),
        (
            "[defaults.advanced.server]\nhost = \"127.0.0.1\"\n",
            "defaults.advanced.server.host",
            ConfigDiagnosticCode::RejectedField,
        ),
    ] {
        let diagnostics = diagnostics_from_toml(raw);
        let diagnostic = diagnostic_for_canonical(&diagnostics, canonical_path);
        assert_eq!(diagnostic.code, expected_code, "{canonical_path}");
    }
}
