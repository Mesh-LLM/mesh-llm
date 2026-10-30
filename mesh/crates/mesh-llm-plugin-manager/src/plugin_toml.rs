//! The static `plugin.toml` at a plugin archive's root.
//!
//! The file marks the archive root and names the plugin and version it
//! packages. It is hand-written packaging metadata; the generated
//! `plugin-manifest.json` describes the executable.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub const PLUGIN_TOML_FILE: &str = "plugin.toml";

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct PluginToml {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

pub fn read_plugin_toml(plugin_dir: &Path, plugin_name: &str) -> Result<PluginToml> {
    let path = plugin_dir.join(PLUGIN_TOML_FILE);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("read plugin manifest {}", path.display()))?;
    let parsed = parse_plugin_toml(&contents, plugin_name)
        .with_context(|| format!("parse plugin manifest {}", path.display()))?;
    Ok(parsed)
}

pub fn parse_plugin_toml(contents: &str, plugin_name: &str) -> Result<PluginToml> {
    let parsed: PluginToml = toml::from_str(contents)?;
    if let Some(name) = parsed.name.as_deref()
        && name != plugin_name
    {
        bail!("plugin.toml names '{name}' but the installed plugin is '{plugin_name}'");
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marker_file_with_name_and_version_has_no_requirements() {
        let parsed = parse_plugin_toml("name = \"demo\"\nversion = \"0.1.0\"\n", "demo").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("demo"));
        assert_eq!(parsed.version.as_deref(), Some("0.1.0"));
    }

    #[test]
    fn an_empty_file_is_accepted() {
        assert_eq!(
            parse_plugin_toml("", "demo").unwrap(),
            PluginToml::default()
        );
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let parsed = parse_plugin_toml("name = \"demo\"\nauthor = \"x\"\n", "demo").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("demo"));
    }

    #[test]
    fn a_name_for_another_plugin_is_refused() {
        let error = parse_plugin_toml("name = \"other\"\n", "demo").unwrap_err();
        assert!(error.to_string().contains("names 'other'"), "{error}");
    }

    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse_plugin_toml("name = \n", "demo").is_err());
    }
}
