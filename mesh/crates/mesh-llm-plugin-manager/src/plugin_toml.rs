//! The static `plugin.toml` at a plugin archive's root.
//!
//! The file marks the archive root and, with `requires`, names the installed
//! plugins this one needs alongside it. It is hand-written packaging
//! metadata, so a plugin declares a requirement without a code change or
//! rebuild; the generated `plugin-manifest.json` describes the executable.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::source_ref::is_valid_name;

pub const PLUGIN_TOML_FILE: &str = "plugin.toml";

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct PluginToml {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    /// Catalog names of the plugins this one requires to be installed.
    /// `plugins install` installs a missing one from the catalog.
    #[serde(default)]
    pub requires: Vec<String>,
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
    validate_requires(&parsed.requires, plugin_name)?;
    Ok(parsed)
}

fn validate_requires(requires: &[String], plugin_name: &str) -> Result<()> {
    for (index, required) in requires.iter().enumerate() {
        if !is_valid_name(required) {
            bail!(
                "plugin.toml requires '{required}', which is not a plugin name (lowercase ASCII \
                 letters, numbers, and single '-' separators)"
            );
        }
        if required == plugin_name {
            bail!("plugin.toml says '{plugin_name}' requires itself");
        }
        if requires[..index].contains(required) {
            bail!("plugin.toml requires '{required}' more than once");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marker_file_with_name_and_version_has_no_requirements() {
        let parsed = parse_plugin_toml("name = \"demo\"\nversion = \"0.1.0\"\n", "demo").unwrap();
        assert_eq!(parsed.name.as_deref(), Some("demo"));
        assert_eq!(parsed.version.as_deref(), Some("0.1.0"));
        assert!(parsed.requires.is_empty());
    }

    #[test]
    fn requirements_are_read_in_order() {
        let parsed = parse_plugin_toml(
            "name = \"demo-app\"\nrequires = [\"toolkit\", \"capsules\"]\n",
            "demo-app",
        )
        .unwrap();
        assert_eq!(parsed.requires, vec!["toolkit", "capsules"]);
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
    fn invalid_self_and_duplicate_requirements_are_refused() {
        let invalid = parse_plugin_toml("requires = [\"Tool Kit\"]\n", "demo").unwrap_err();
        assert!(
            invalid.to_string().contains("not a plugin name"),
            "{invalid}"
        );
        let own = parse_plugin_toml("requires = [\"demo\"]\n", "demo").unwrap_err();
        assert!(own.to_string().contains("requires itself"), "{own}");
        let twice = parse_plugin_toml("requires = [\"a\", \"a\"]\n", "demo").unwrap_err();
        assert!(twice.to_string().contains("more than once"), "{twice}");
    }

    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse_plugin_toml("name = \n", "demo").is_err());
    }
}
