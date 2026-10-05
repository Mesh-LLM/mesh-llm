#[derive(Clone, Debug, PartialEq, Default)]
pub struct PluginControlBehavior {
    pub numeric: Option<PluginNumericControl>,
    pub text_format: Option<PluginTextFormat>,
    pub options_source: Option<PluginOptionsSource>,
    pub availability: Option<PluginControlAvailability>,
    pub enable_when: Vec<PluginControlCondition>,
    pub disable_when: Vec<PluginConditionalDisable>,
    pub conflicts: Vec<PluginConflictRule>,
    pub write_policy: Option<PluginDisabledWritePolicy>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PluginNumericControl {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub soft_min: Option<f64>,
    pub soft_max: Option<f64>,
    pub unit: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginTextFormat {
    Plain,
    Path,
    Url,
    SocketAddr,
    Semver,
    Ed25519Key,
    CsvPositiveInts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginOptionsSource {
    Static,
    RuntimeGpus,
    RuntimeNativeBackends,
    RuntimeLocalModels,
    RuntimeInstalledPlugins,
    RuntimeMeshPeers,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginControlAvailability {
    pub enabled: bool,
    pub reason: Option<String>,
    pub note: Option<String>,
    pub source: PluginControlAvailabilitySource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginControlAvailabilitySource {
    Static,
    Runtime,
    Dependency,
    Conflict,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PluginControlCondition {
    pub key: String,
    pub operator: PluginConditionOperator,
    pub values: Vec<PluginConditionValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginConditionOperator {
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

#[derive(Clone, Debug, PartialEq)]
pub enum PluginConditionValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PluginConditionalDisable {
    pub condition: PluginControlCondition,
    pub reason: String,
    pub note: Option<String>,
    pub write_policy: PluginDisabledWritePolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PluginConflictRule {
    pub group: String,
    pub condition: PluginControlCondition,
    pub reason: String,
    pub preferred_key: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginDisabledWritePolicy {
    PreserveExisting,
    OmitWhenDisabled,
    RejectWhenDisabled,
}
