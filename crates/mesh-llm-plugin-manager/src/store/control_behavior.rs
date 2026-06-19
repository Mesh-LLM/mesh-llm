use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InstalledPluginControlBehavior {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub numeric: Option<InstalledPluginNumericControl>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text_format: Option<InstalledPluginTextFormat>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub options_source: Option<InstalledPluginOptionsSource>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub availability: Option<InstalledPluginControlAvailability>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enable_when: Vec<InstalledPluginControlCondition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disable_when: Vec<InstalledPluginConditionalDisable>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<InstalledPluginConflictRule>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub write_policy: Option<InstalledPluginDisabledWritePolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InstalledPluginNumericControl {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub soft_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub soft_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginTextFormat {
    Plain,
    Path,
    Url,
    SocketAddr,
    Semver,
    Ed25519Key,
    CsvPositiveInts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginOptionsSource {
    Static,
    RuntimeGpus,
    RuntimeNativeBackends,
    RuntimeLocalModels,
    RuntimeInstalledPlugins,
    RuntimeMeshPeers,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledPluginControlAvailability {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub note: Option<String>,
    pub source: InstalledPluginControlAvailabilitySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginControlAvailabilitySource {
    Static,
    Runtime,
    Dependency,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledPluginControlCondition {
    pub key: String,
    pub operator: InstalledPluginConditionOperator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<InstalledPluginConditionValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginConditionOperator {
    Equals,
    NotEquals,
    In,
    NotIn,
    Present,
    Absent,
    Truthy,
    Falsy,
    Range,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum InstalledPluginConditionValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledPluginConditionalDisable {
    pub condition: InstalledPluginControlCondition,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub note: Option<String>,
    pub write_policy: InstalledPluginDisabledWritePolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstalledPluginConflictRule {
    pub group: String,
    pub condition: InstalledPluginControlCondition,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub preferred_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstalledPluginDisabledWritePolicy {
    PreserveExisting,
    OmitWhenDisabled,
    RejectWhenDisabled,
}
