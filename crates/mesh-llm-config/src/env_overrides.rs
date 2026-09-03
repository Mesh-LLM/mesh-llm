use std::ffi::OsStr;

use anyhow::{Result, bail};

use crate::{LifecycleLogParserMode, MeshConfig};

pub const MESH_LLM_LIFECYCLE_LOG_PARSER_ENV: &str = "MESH_LLM_LIFECYCLE_LOG_PARSER";
pub const MESH_LLM_CONFIG_ENV: &str = "MESH_LLM_CONFIG";

pub const CONFIG_OVERRIDE_ENV_NAMES: &[&str] =
    &[MESH_LLM_CONFIG_ENV, MESH_LLM_LIFECYCLE_LOG_PARSER_ENV];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfigValueSource {
    Env,
    Config,
    #[default]
    Default,
}

impl ConfigValueSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Config => "config",
            Self::Default => "default",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LifecycleLogParserSelection {
    pub mode: LifecycleLogParserMode,
    pub source: ConfigValueSource,
}

pub fn resolve_lifecycle_log_parser_override(
    configured: LifecycleLogParserMode,
    configured_source: ConfigValueSource,
    environment: Option<&OsStr>,
) -> Result<LifecycleLogParserSelection> {
    let Some(environment) = environment else {
        return Ok(LifecycleLogParserSelection {
            mode: configured,
            source: configured_source,
        });
    };
    let Some(environment) = environment.to_str() else {
        bail!("invalid {MESH_LLM_LIFECYCLE_LOG_PARSER_ENV}; expected auto, enabled, or disabled");
    };
    let mode = environment.parse().map_err(|_| {
        anyhow::anyhow!(
            "invalid {MESH_LLM_LIFECYCLE_LOG_PARSER_ENV}; expected auto, enabled, or disabled"
        )
    })?;
    Ok(LifecycleLogParserSelection {
        mode,
        source: ConfigValueSource::Env,
    })
}

pub(crate) fn config_path_override() -> Option<std::path::PathBuf> {
    std::env::var_os(MESH_LLM_CONFIG_ENV).map(std::path::PathBuf::from)
}

pub(crate) fn apply_env_overrides(config: &mut MeshConfig) -> Result<()> {
    let environment = std::env::var_os(MESH_LLM_LIFECYCLE_LOG_PARSER_ENV);
    if environment.is_none() {
        return Ok(());
    }
    let selection = resolve_lifecycle_log_parser_override(
        config.runtime.lifecycle_log_parser,
        config.runtime.lifecycle_log_parser_source,
        environment.as_deref(),
    )?;
    config.runtime.lifecycle_log_parser = selection.mode;
    config.runtime.lifecycle_log_parser_source = selection.source;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_overrides_owner_lists_every_config_override_name() {
        assert_eq!(
            CONFIG_OVERRIDE_ENV_NAMES,
            &["MESH_LLM_CONFIG", "MESH_LLM_LIFECYCLE_LOG_PARSER"]
        );
    }
}
