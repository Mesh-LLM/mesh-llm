use super::*;
mod control_behavior;
mod presentation;
use self::control_behavior::apply_built_in_control_behavior;
use self::presentation::apply_built_in_presentation_metadata;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltInConfigPathResolution {
    pub requested_path: ConfigPath,
    pub normalized_path: ConfigPath,
    pub canonical_path: ConfigPath,
    pub matched_alias: Option<ConfigPath>,
    pub support: ConfigSupportState,
}

impl BuiltInConfigPathResolution {
    pub fn canonical_identifier(&self) -> String {
        self.canonical_path.render()
    }

    pub fn used_legacy_alias(&self) -> bool {
        self.matched_alias.is_some()
    }
}

pub fn built_in_config_settings() -> Vec<ConfigSettingSchema> {
    built_in_config_schema_cache().settings.clone()
}

pub fn built_in_config_schema_descriptor(path: &ConfigPath) -> Option<ConfigSettingSchema> {
    let normalized = path.normalize_builtin_layout();
    built_in_config_schema_cache()
        .settings
        .iter()
        .find(|setting| setting.path == normalized)
        .cloned()
}

pub fn resolve_built_in_config_path(path: &ConfigPath) -> Option<BuiltInConfigPathResolution> {
    let requested_path = path.clone();
    let normalized_path = path.normalize_builtin_layout();

    for setting in &built_in_config_schema_cache().settings {
        if setting.path == normalized_path {
            return Some(BuiltInConfigPathResolution {
                requested_path,
                normalized_path,
                canonical_path: setting.path.clone(),
                matched_alias: None,
                support: setting.support,
            });
        }
        if let Some(alias) = setting
            .alias_policy
            .aliases
            .iter()
            .find(|alias| alias.path == normalized_path)
        {
            return Some(BuiltInConfigPathResolution {
                requested_path,
                normalized_path,
                canonical_path: setting.path.clone(),
                matched_alias: Some(alias.path.clone()),
                support: setting.support,
            });
        }
    }

    None
}

pub fn resolve_built_in_config_identifier(rendered: &str) -> Option<BuiltInConfigPathResolution> {
    let parsed = ConfigPath::parse_rendered(rendered).ok()?;
    resolve_built_in_config_path(&parsed)
}

pub fn canonicalize_built_in_config_path(path: &ConfigPath) -> Option<ConfigPath> {
    resolve_built_in_config_path(path).map(|resolution| resolution.canonical_path)
}

pub fn canonicalize_built_in_config_identifier(rendered: &str) -> Option<String> {
    resolve_built_in_config_identifier(rendered).map(|resolution| resolution.canonical_identifier())
}

fn built_in_config_schema_cache() -> &'static ConfigSchema {
    static SCHEMA: OnceLock<ConfigSchema> = OnceLock::new();
    SCHEMA.get_or_init(build_built_in_config_schema)
}

include!("built_in_schema/declarations.rs");
include!("built_in_schema/setting_schema.rs");

#[cfg(test)]
#[path = "built_in_schema/logging_contract_tests.rs"]
mod logging_contract_tests;

#[cfg(test)]
#[path = "built_in_schema/validation.rs"]
mod tests;
