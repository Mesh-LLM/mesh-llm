use crate::{
    ConfigConditionOperator, ConfigConstraint, ConfigDiagnostic, ConfigDisabledWritePolicy,
    ConfigPath, ConfigSettingSchema, ConfigValueSchema, MeshConfig,
    built_in_config_schema_descriptor, validate_config_diagnostics,
};

mod fixture_contract;
mod model_rules;
mod runtime_controls;

fn schema_setting(path: &str) -> ConfigSettingSchema {
    let path = ConfigPath::parse_rendered(path).expect("schema path should parse");
    built_in_config_schema_descriptor(&path).expect("schema setting should exist")
}

fn diagnostics_from_toml(raw: &str) -> Vec<ConfigDiagnostic> {
    let config: MeshConfig = toml::from_str(raw).expect("config should parse before validation");
    validate_config_diagnostics(&config)
}

fn rendered(path: &Option<ConfigPath>) -> Option<String> {
    path.as_ref().map(ConfigPath::render)
}

fn diagnostic_for_canonical<'a>(
    diagnostics: &'a [ConfigDiagnostic],
    canonical_path: &str,
) -> &'a ConfigDiagnostic {
    diagnostics
        .iter()
        .find(|diagnostic| rendered(&diagnostic.canonical_path).as_deref() == Some(canonical_path))
        .unwrap_or_else(|| panic!("missing diagnostic for {canonical_path}"))
}

fn assert_requires(setting: &ConfigSettingSchema, required_path: &str) {
    let required_path =
        ConfigPath::parse_rendered(required_path).expect("required path should parse");
    assert!(
        setting.constraints.iter().any(|constraint| {
            matches!(constraint, ConfigConstraint::Requires { path } if path == &required_path)
        }),
        "expected requires constraint on {}",
        setting.path.render()
    );
}

fn assert_range(setting: &ConfigSettingSchema, min: Option<&str>, max: Option<&str>) {
    assert!(
        setting.constraints.iter().any(|constraint| {
            matches!(constraint, ConfigConstraint::Range { min: current_min, max: current_max }
                if current_min.as_deref() == min && current_max.as_deref() == max)
        }),
        "expected range constraint on {}",
        setting.path.render()
    );
}

fn assert_socket_addr_schema(setting: &ConfigSettingSchema) {
    assert_eq!(setting.value_schema, ConfigValueSchema::SocketAddr);
}

fn assert_present_enable_when(setting: &ConfigSettingSchema) {
    let behavior = setting
        .control_behavior
        .as_ref()
        .expect("control behavior should exist");
    assert_eq!(
        behavior.enable_when[0].operator,
        ConfigConditionOperator::Present
    );
    assert_eq!(
        behavior.disable_when[0].condition.operator,
        ConfigConditionOperator::Absent
    );
}

fn assert_reject_when_disabled(setting: &ConfigSettingSchema) {
    assert_eq!(
        setting.default_disabled_write_policy(None),
        Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
    );
}
