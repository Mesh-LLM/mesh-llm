use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::source_ref::is_valid_name;

const METADATA_FILE: &str = "plugin-install.json";
pub const SUPPORTED_PLUGIN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginManifestMetadata {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub config_schema: Option<InstalledPluginConfigSchema>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginConfigSchema {
    pub plugin_name: String,
    pub schema_version: u32,
    #[serde(default)]
    pub allow_unvalidated_config: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<InstalledPluginSettingSchema>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginSettingSchema {
    pub key: String,
    pub value_schema: InstalledPluginValueSchema,
    #[serde(default)]
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub default_json: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<InstalledPluginConstraint>,
    pub apply_mode: InstalledPluginApplyMode,
    pub restart_scope: InstalledPluginRestartScope,
    pub visibility: InstalledPluginVisibility,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginValueSchema {
    pub kind: InstalledPluginValueKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub items: Option<Box<InstalledPluginValueSchema>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_properties: Vec<InstalledPluginObjectProperty>,
    #[serde(default)]
    pub allow_additional_properties: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginObjectProperty {
    pub key: String,
    pub value_schema: InstalledPluginValueSchema,
    #[serde(default)]
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginValueKind {
    Boolean,
    Integer,
    Float,
    String,
    Path,
    Url,
    Enum,
    Array,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginApplyMode {
    StaticOnLoad,
    DynamicValidationOnly,
    DynamicApply,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginRestartScope {
    None,
    ModelReload,
    ProcessRestart,
    MeshRestart,
    PluginProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginVisibility {
    User,
    Advanced,
    Hidden,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InstalledPluginConstraint {
    NonEmpty,
    Positive,
    Range {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        min: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        max: Option<String>,
    },
    AllowedValues {
        values: Vec<String>,
    },
    Requires {
        key: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginMetadata {
    pub name: String,
    pub source_repository: String,
    pub installed_version: String,
    pub target_triple: String,
    pub downloaded_asset_name: String,
    pub install_path: PathBuf,
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub manifest: Option<InstalledPluginManifestMetadata>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_protocol_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_error: Option<String>,
}

impl InstalledPluginMetadata {
    pub fn executable_path(&self) -> PathBuf {
        self.install_path
            .join(format!("{}{}", self.name, std::env::consts::EXE_SUFFIX))
    }
}

#[derive(Debug, Clone)]
pub struct PluginStore {
    root: PathBuf,
}

impl PluginStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn save(&self, metadata: &InstalledPluginMetadata) -> Result<()> {
        validate_plugin_name(&metadata.name)?;
        let plugin_dir = self.plugin_dir(&metadata.name);
        fs::create_dir_all(&plugin_dir).with_context(|| {
            format!("create plugin metadata directory {}", plugin_dir.display())
        })?;
        let metadata_path = self.metadata_path(&metadata.name);
        let temp_path = metadata_path.with_extension("json.tmp");
        let contents = serde_json::to_vec_pretty(metadata)?;
        fs::write(&temp_path, contents)
            .with_context(|| format!("write plugin metadata {}", temp_path.display()))?;
        fs::rename(&temp_path, &metadata_path).with_context(|| {
            format!(
                "replace plugin metadata {} with {}",
                metadata_path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    }

    pub fn load(&self, name: &str) -> Result<InstalledPluginMetadata> {
        self.try_load(name)?
            .with_context(|| format!("plugin '{name}' is not installed"))
    }

    pub fn try_load(&self, name: &str) -> Result<Option<InstalledPluginMetadata>> {
        validate_plugin_name(name)?;
        let metadata_path = self.metadata_path(name);
        if !metadata_path.exists() {
            return Ok(None);
        }
        let contents = fs::read(&metadata_path)
            .with_context(|| format!("read plugin metadata {}", metadata_path.display()))?;
        Ok(Some(serde_json::from_slice(&contents).with_context(
            || format!("parse plugin metadata {}", metadata_path.display()),
        )?))
    }

    pub fn load_optional(&self, name: &str) -> Result<Option<InstalledPluginMetadata>> {
        self.try_load(name)
    }

    pub fn list(&self) -> Result<Vec<InstalledPluginMetadata>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }

        let mut plugins = Vec::new();
        for entry in fs::read_dir(&self.root)
            .with_context(|| format!("read plugin store {}", self.root.display()))?
        {
            let entry = entry
                .with_context(|| format!("read plugin store entry {}", self.root.display()))?;
            if !entry
                .file_type()
                .with_context(|| format!("read file type for {}", entry.path().display()))?
                .is_dir()
            {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if is_valid_name(&name) && self.metadata_path(&name).exists() {
                plugins.push(self.load(&name)?);
            }
        }
        plugins.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(plugins)
    }

    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<InstalledPluginMetadata> {
        let mut metadata = self.load(name)?;
        metadata.enabled = enabled;
        self.save(&metadata)?;
        Ok(metadata)
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        validate_plugin_name(name)?;
        let metadata = self.load(name).ok();
        if let Some(metadata) = metadata
            && metadata.install_path.exists()
        {
            fs::remove_dir_all(&metadata.install_path).with_context(|| {
                format!("delete plugin install {}", metadata.install_path.display())
            })?;
        }
        let plugin_dir = self.plugin_dir(name);
        if plugin_dir.exists() {
            fs::remove_dir_all(&plugin_dir)
                .with_context(|| format!("delete plugin metadata {}", plugin_dir.display()))?;
        }
        Ok(())
    }

    fn plugin_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    fn metadata_path(&self, name: &str) -> PathBuf {
        self.plugin_dir(name).join(METADATA_FILE)
    }
}

pub fn default_store_root() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("MESH_LLM_PLUGIN_DIR") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("plugins"))
}

fn validate_plugin_name(name: &str) -> Result<()> {
    if is_valid_name(name) {
        Ok(())
    } else {
        bail!("invalid plugin name: {name}")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn metadata(name: &str) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: "https://github.com/mesh-llm/blackboard".to_string(),
            installed_version: "v1.0.0".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            downloaded_asset_name: "blackboard-v1.0.0-aarch64-apple-darwin.tar.gz".to_string(),
            install_path: PathBuf::from("/tmp/plugins/blackboard"),
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: Some(InstalledPluginConfigSchema {
                    plugin_name: name.to_string(),
                    schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
                    allow_unvalidated_config: false,
                    settings: vec![InstalledPluginSettingSchema {
                        key: "retention_days".to_string(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Integer,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: true,
                        default_json: Some("14".to_string()),
                        constraints: vec![InstalledPluginConstraint::Range {
                            min: Some("1".to_string()),
                            max: Some("365".to_string()),
                        }],
                        apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::User,
                        description: Some("How long to retain entries.".to_string()),
                    }],
                }),
            }),
            last_protocol_version: Some(2),
            last_status: Some("running".to_string()),
            last_error: None,
        }
    }

    #[test]
    fn saves_loads_and_lists_metadata() {
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());

        store.save(&metadata("blackboard")).unwrap();
        store.save(&metadata("notes")).unwrap();

        let loaded = store.load("blackboard").unwrap();
        assert_eq!(loaded.name, "blackboard");
        assert!(loaded.enabled);
        assert_eq!(
            loaded
                .manifest
                .as_ref()
                .and_then(|manifest| manifest.config_schema.as_ref())
                .map(|schema| schema.schema_version),
            Some(SUPPORTED_PLUGIN_SCHEMA_VERSION)
        );
        assert_eq!(loaded.last_protocol_version, Some(2));

        let listed = store.list().unwrap();
        assert_eq!(
            listed
                .iter()
                .map(|plugin| plugin.name.as_str())
                .collect::<Vec<_>>(),
            vec!["blackboard", "notes"]
        );
    }

    #[test]
    fn updates_enabled_state() {
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());
        store.save(&metadata("blackboard")).unwrap();

        let disabled = store.set_enabled("blackboard", false).unwrap();
        assert!(!disabled.enabled);
        assert!(!store.load("blackboard").unwrap().enabled);
    }

    #[test]
    fn load_optional_distinguishes_missing_metadata() {
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());

        assert!(store.load_optional("blackboard").unwrap().is_none());

        store.save(&metadata("blackboard")).unwrap();
        assert_eq!(
            store.load_optional("blackboard").unwrap().unwrap().name,
            "blackboard"
        );
    }

    #[test]
    fn deletes_metadata_directory() {
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());
        let install_temp = TempDir::new().unwrap();
        let install_path = install_temp.path().join("blackboard");
        std::fs::create_dir_all(&install_path).unwrap();
        let mut metadata = metadata("blackboard");
        metadata.install_path = install_path.clone();
        store.save(&metadata).unwrap();

        store.delete("blackboard").unwrap();
        assert!(store.list().unwrap().is_empty());
        assert!(!install_path.exists());
    }

    #[test]
    fn list_ignores_non_metadata_directories() {
        let temp = TempDir::new().unwrap();
        let store = PluginStore::new(temp.path());
        std::fs::create_dir_all(temp.path().join("installed").join("blackboard")).unwrap();
        store.save(&metadata("blackboard")).unwrap();

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "blackboard");
    }
}
