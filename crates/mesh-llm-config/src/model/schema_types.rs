use serde::{Deserialize, Serialize};
use std::iter::Peekable;
use std::str::Chars;

pub const CANONICAL_MODEL_REF_SEGMENT: &str = "<model-ref>";
pub const CANONICAL_PLUGIN_NAME_SEGMENT: &str = "<plugin-name>";

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ConfigSchema {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<ConfigSettingSchema>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ConfigSettingSchema {
    pub path: ConfigPath,
    #[serde(default)]
    pub alias_policy: ConfigAliasPolicy,
    pub owner: ConfigSettingOwner,
    pub value_schema: ConfigValueSchema,
    pub support: ConfigSupportState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub control_surfaces: Vec<ConfigControlSurface>,
    pub apply_mode: ConfigApplyMode,
    pub restart_scope: ConfigRestartScope,
    pub visibility: ConfigVisibility,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<ConfigConstraint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<ConfigPresentationMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_behavior: Option<ConfigControlBehavior>,
}

impl ConfigSettingSchema {
    pub fn default_disabled_write_policy(
        &self,
        availability_source: Option<ConfigControlAvailabilitySource>,
    ) -> Option<ConfigDisabledWritePolicy> {
        match self
            .control_behavior
            .as_ref()
            .and_then(|behavior| behavior.write_policy)
        {
            Some(policy) => Some(policy),
            None => match self.support.default_disabled_write_policy() {
                Some(policy) => Some(policy),
                None => match availability_source {
                    Some(source) => source.default_disabled_write_policy(),
                    None => None,
                },
            },
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ConfigControlBehavior {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric: Option<ConfigNumericControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_format: Option<ConfigTextFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options_source: Option<ConfigOptionsSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability: Option<ConfigControlAvailability>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enable_when: Vec<ConfigControlCondition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disable_when: Vec<ConfigConditionalDisable>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<ConfigConflictRule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_policy: Option<ConfigDisabledWritePolicy>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ConfigNumericControl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigTextFormat {
    Plain,
    Path,
    Url,
    SocketAddr,
    Semver,
    Ed25519Key,
    CsvPositiveInts,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigOptionsSource {
    Static,
    RuntimeGpus,
    RuntimeNativeBackends,
    RuntimeLocalModels,
    RuntimeInstalledPlugins,
    RuntimeMeshPeers,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigControlAvailability {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub source: ConfigControlAvailabilitySource,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigControlAvailabilitySource {
    Static,
    Runtime,
    Dependency,
    Conflict,
}

impl ConfigControlAvailabilitySource {
    pub const fn default_disabled_write_policy(self) -> Option<ConfigDisabledWritePolicy> {
        match self {
            Self::Static | Self::Runtime => Some(ConfigDisabledWritePolicy::PreserveExisting),
            Self::Dependency => Some(ConfigDisabledWritePolicy::OmitWhenDisabled),
            Self::Conflict => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ConfigControlCondition {
    pub path: ConfigPath,
    pub operator: ConfigConditionOperator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ConfigConditionValue>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigConditionOperator {
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ConfigConditionValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ConfigConditionalDisable {
    pub condition: ConfigControlCondition,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub write_policy: ConfigDisabledWritePolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ConfigConflictRule {
    pub group: String,
    pub condition: ConfigControlCondition,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_path: Option<ConfigPath>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDisabledWritePolicy {
    PreserveExisting,
    OmitWhenDisabled,
    RejectWhenDisabled,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigPresentationMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category_order: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setting_order: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renderer_id: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConfigPath {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<ConfigPathSegment>,
}

impl ConfigPath {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn field(name: impl Into<String>) -> Self {
        let mut path = Self::root();
        path.push_field(name);
        path
    }

    pub fn from_fields<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut path = Self::root();
        for field in fields {
            path.push_field(field);
        }
        path
    }

    pub fn push_field(&mut self, name: impl Into<String>) -> &mut Self {
        self.segments
            .push(ConfigPathSegment::Field { name: name.into() });
        self
    }

    pub fn push_index(&mut self, index: usize) -> &mut Self {
        self.segments.push(ConfigPathSegment::Index { index });
        self
    }

    pub fn push_key(&mut self, name: impl Into<String>) -> &mut Self {
        self.segments
            .push(ConfigPathSegment::Key { name: name.into() });
        self
    }

    pub fn render(&self) -> String {
        let mut rendered = String::new();
        for segment in &self.segments {
            match segment {
                ConfigPathSegment::Field { name } => {
                    if !rendered.is_empty() {
                        rendered.push('.');
                    }
                    rendered.push_str(name);
                }
                ConfigPathSegment::Index { index } => {
                    rendered.push('[');
                    rendered.push_str(&index.to_string());
                    rendered.push(']');
                }
                ConfigPathSegment::Key { name } => {
                    rendered.push('[');
                    rendered.push_str(&format!("{name:?}"));
                    rendered.push(']');
                }
            }
        }
        rendered
    }

    pub fn parse_rendered(rendered: &str) -> Result<Self, String> {
        let mut path = Self::root();
        let mut chars = rendered.chars().peekable();
        let mut field = String::new();

        while let Some(ch) = chars.next() {
            match ch {
                '.' => {
                    if field.is_empty() {
                        if path.segments.is_empty() {
                            return Err(format!("invalid config path `{rendered}`"));
                        }
                        continue;
                    }
                    path.push_field(std::mem::take(&mut field));
                }
                '[' => {
                    if !field.is_empty() {
                        path.push_field(std::mem::take(&mut field));
                    }
                    match chars.peek().copied() {
                        Some('"') => {
                            path.push_key(parse_rendered_key(&mut chars, rendered)?);
                        }
                        Some(next) if next.is_ascii_digit() => {
                            let mut index = String::new();
                            while let Some(next) = chars.peek().copied() {
                                if next == ']' {
                                    break;
                                }
                                if !next.is_ascii_digit() {
                                    return Err(format!("invalid config path `{rendered}`"));
                                }
                                index.push(next);
                                chars.next();
                            }
                            if chars.next() != Some(']') || index.is_empty() {
                                return Err(format!("invalid config path `{rendered}`"));
                            }
                            let index = index
                                .parse::<usize>()
                                .map_err(|_| format!("invalid config path `{rendered}`"))?;
                            path.push_index(index);
                        }
                        _ => return Err(format!("invalid config path `{rendered}`")),
                    }
                }
                other => field.push(other),
            }
        }

        if !field.is_empty() {
            path.push_field(field);
        }

        Ok(path)
    }

    pub fn normalize_builtin_layout(&self) -> Self {
        let mut normalized = Self::root();
        let root_field = self.segments.first().and_then(|segment| match segment {
            ConfigPathSegment::Field { name } => Some(name.as_str()),
            _ => None,
        });

        for (index, segment) in self.segments.iter().enumerate() {
            match (root_field, index, segment) {
                (Some("models"), 1, ConfigPathSegment::Index { .. }) => {
                    normalized.push_field(CANONICAL_MODEL_REF_SEGMENT);
                }
                (Some("plugin"), 1, ConfigPathSegment::Index { .. }) => {
                    normalized.push_field(CANONICAL_PLUGIN_NAME_SEGMENT);
                }
                _ => normalized.segments.push(segment.clone()),
            }
        }

        normalized
    }
}

fn parse_rendered_key(chars: &mut Peekable<Chars<'_>>, rendered: &str) -> Result<String, String> {
    if chars.next() != Some('"') {
        return Err(format!("invalid config path `{rendered}`"));
    }

    let mut key = String::new();
    while let Some(next) = chars.next() {
        match next {
            '"' => {
                if chars.next() != Some(']') {
                    return Err(format!("invalid config path `{rendered}`"));
                }
                return Ok(key);
            }
            '\\' => key.push(parse_rendered_escape(chars, rendered)?),
            other => key.push(other),
        }
    }

    Err(format!("invalid config path `{rendered}`"))
}

fn parse_rendered_escape(chars: &mut Peekable<Chars<'_>>, rendered: &str) -> Result<char, String> {
    match chars.next() {
        Some('"') => Ok('"'),
        Some('\\') => Ok('\\'),
        Some('n') => Ok('\n'),
        Some('r') => Ok('\r'),
        Some('t') => Ok('\t'),
        Some('0') => Ok('\0'),
        Some('u') => parse_rendered_unicode_escape(chars, rendered),
        _ => Err(format!("invalid config path `{rendered}`")),
    }
}

fn parse_rendered_unicode_escape(
    chars: &mut Peekable<Chars<'_>>,
    rendered: &str,
) -> Result<char, String> {
    if chars.next() != Some('{') {
        return Err(format!("invalid config path `{rendered}`"));
    }

    let mut codepoint = String::new();
    for next in chars.by_ref() {
        match next {
            '}' => {
                let codepoint = u32::from_str_radix(&codepoint, 16)
                    .map_err(|_| format!("invalid config path `{rendered}`"))?;
                return char::from_u32(codepoint)
                    .ok_or_else(|| format!("invalid config path `{rendered}`"));
            }
            hex if hex.is_ascii_hexdigit() => codepoint.push(hex),
            _ => return Err(format!("invalid config path `{rendered}`")),
        }
    }

    Err(format!("invalid config path `{rendered}`"))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigPathSegment {
    Field { name: String },
    Index { index: usize },
    Key { name: String },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigAliasPolicy {
    #[serde(default)]
    pub mode: ConfigAliasMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<ConfigPathAlias>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigPathAlias {
    pub path: ConfigPath,
    pub kind: ConfigPathAliasKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigAliasMode {
    #[default]
    CanonicalOnly,
    CanonicalWithLegacyAliases,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigPathAliasKind {
    #[default]
    LegacyKey,
    LegacyLayout,
    LegacySection,
    LegacyShim,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSettingOwner {
    #[default]
    BuiltIn,
    Engine,
    Plugin,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigValueSchema {
    Boolean,
    Integer,
    Float,
    String,
    Path,
    Url,
    SocketAddr,
    Enum { values: Vec<String> },
    OneOf { variants: Vec<ConfigValueSchema> },
    Array { items: Box<ConfigValueSchema> },
    Object,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSupportState {
    #[default]
    Supported,
    Experimental,
    DeprecatedAlias,
    Unwired,
    Unsupported,
    Rejected,
}

impl ConfigSupportState {
    pub const fn default_disabled_write_policy(self) -> Option<ConfigDisabledWritePolicy> {
        match self {
            Self::Unsupported | Self::Rejected => {
                Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
            }
            Self::Supported | Self::Experimental | Self::DeprecatedAlias | Self::Unwired => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigControlSurface {
    ConfigFile,
    Cli,
    OwnerControl,
    Api,
    Ui,
    PluginManifest,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigApplyMode {
    #[default]
    StaticOnLoad,
    DynamicValidationOnly,
    DynamicApply,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigRestartScope {
    #[default]
    None,
    ModelReload,
    ProcessRestart,
    MeshRestart,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigVisibility {
    #[default]
    User,
    Advanced,
    Hidden,
    Internal,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigConstraint {
    NonEmpty,
    Positive,
    Range {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<String>,
    },
    AllowedPattern {
        pattern: String,
    },
    Requires {
        path: ConfigPath,
    },
    AllowedValues {
        values: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use toml::Value;

    fn legacy_setting_value() -> Value {
        Value::Table(toml::map::Map::from_iter([
            (
                "path".to_string(),
                config_path_value("defaults.hardware.device"),
            ),
            ("owner".to_string(), Value::String("built_in".to_string())),
            (
                "value_schema".to_string(),
                Value::Table(toml::map::Map::from_iter([(
                    "kind".to_string(),
                    Value::String("string".to_string()),
                )])),
            ),
            (
                "support".to_string(),
                Value::String("supported".to_string()),
            ),
            (
                "control_surfaces".to_string(),
                Value::Array(vec![Value::String("config_file".to_string())]),
            ),
            (
                "apply_mode".to_string(),
                Value::String("static_on_load".to_string()),
            ),
            (
                "restart_scope".to_string(),
                Value::String("none".to_string()),
            ),
            ("visibility".to_string(), Value::String("user".to_string())),
        ]))
    }

    fn config_path_value(rendered: &str) -> Value {
        let path = ConfigPath::parse_rendered(rendered).expect("path should parse");
        Value::try_from(path).expect("path should serialize")
    }

    #[test]
    fn parse_rendered_accepts_canonical_placeholder_path() {
        let rendered = "models.<model-ref>.hardware.device";
        let path = ConfigPath::parse_rendered(rendered).expect("canonical path should parse");

        assert_eq!(path.render(), rendered);
    }

    #[test]
    fn parse_rendered_roundtrips_rendered_key_escapes() {
        let mut path = ConfigPath::field("plugin");
        path.push_key("plugin.with\nquote\"backslash\\escape\u{1b}");
        path.push_field("settings");

        let rendered = path.render();
        let parsed = ConfigPath::parse_rendered(&rendered).expect("rendered key should parse");

        assert_eq!(parsed, path);
        assert_eq!(parsed.render(), rendered);
    }

    #[test]
    fn setting_without_control_behavior_deserializes_and_omits_optional_field() {
        let legacy_setting = legacy_setting_value();

        let setting: ConfigSettingSchema = legacy_setting
            .try_into()
            .expect("legacy setting should deserialize");

        assert!(setting.control_behavior.is_none());
        assert_eq!(setting.default_disabled_write_policy(None), None);

        let serialized = Value::try_from(setting).expect("setting should serialize");
        let table = serialized
            .as_table()
            .expect("setting should serialize to a table");

        assert!(!table.contains_key("control_behavior"));
    }

    #[test]
    fn numeric_control_behavior_roundtrips() {
        let setting = ConfigSettingSchema {
            path: ConfigPath::parse_rendered("defaults.request.max_tokens")
                .expect("path should parse"),
            alias_policy: ConfigAliasPolicy::default(),
            owner: ConfigSettingOwner::BuiltIn,
            value_schema: ConfigValueSchema::Integer,
            support: ConfigSupportState::Supported,
            control_surfaces: vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Ui],
            apply_mode: ConfigApplyMode::DynamicValidationOnly,
            restart_scope: ConfigRestartScope::None,
            visibility: ConfigVisibility::User,
            constraints: Vec::new(),
            description: Some("Request max tokens".to_string()),
            presentation: None,
            control_behavior: Some(ConfigControlBehavior {
                numeric: Some(ConfigNumericControl {
                    min: Some(1.0),
                    max: Some(8192.0),
                    step: Some(1.0),
                    soft_min: Some(16.0),
                    soft_max: Some(4096.0),
                    unit: Some("tokens".to_string()),
                }),
                text_format: None,
                options_source: None,
                availability: None,
                enable_when: Vec::new(),
                disable_when: Vec::new(),
                conflicts: Vec::new(),
                write_policy: None,
            }),
        };

        let serialized = Value::try_from(setting.clone()).expect("setting should serialize");
        let roundtrip: ConfigSettingSchema =
            serialized.try_into().expect("setting should deserialize");

        assert_eq!(roundtrip, setting);
    }

    #[test]
    fn value_schema_roundtrips_explicit_path_kind() {
        let schema = Value::Table(toml::map::Map::from_iter([(
            "kind".to_string(),
            Value::String("path".to_string()),
        )]));

        let parsed: ConfigValueSchema = schema.clone().try_into().expect(
            "path kind should deserialize as a distinct value schema instead of collapsing",
        );
        let serialized = Value::try_from(parsed).expect("path schema should serialize");

        assert_eq!(serialized, schema);
    }

    #[test]
    fn value_schema_roundtrips_explicit_url_kind() {
        let schema = Value::Table(toml::map::Map::from_iter([(
            "kind".to_string(),
            Value::String("url".to_string()),
        )]));

        let parsed: ConfigValueSchema = schema
            .clone()
            .try_into()
            .expect("url kind should deserialize as a distinct value schema instead of collapsing");
        let serialized = Value::try_from(parsed).expect("url schema should serialize");

        assert_eq!(serialized, schema);
    }

    #[test]
    fn value_schema_roundtrips_array_item_path_kind() {
        let schema = Value::Table(toml::map::Map::from_iter([
            ("kind".to_string(), Value::String("array".to_string())),
            (
                "items".to_string(),
                Value::Table(toml::map::Map::from_iter([(
                    "kind".to_string(),
                    Value::String("path".to_string()),
                )])),
            ),
        ]));

        let parsed: ConfigValueSchema = schema.clone().try_into().expect(
            "array items should preserve explicit item value kinds for schema-driven controls",
        );
        let serialized = Value::try_from(parsed).expect("array schema should serialize");

        assert_eq!(serialized, schema);
    }

    #[test]
    fn value_schema_preserves_existing_string_socket_addr_and_enum_json() {
        let string_schema = Value::Table(toml::map::Map::from_iter([(
            "kind".to_string(),
            Value::String("string".to_string()),
        )]));
        let socket_addr_schema = Value::Table(toml::map::Map::from_iter([(
            "kind".to_string(),
            Value::String("socket_addr".to_string()),
        )]));
        let enum_schema = Value::Table(toml::map::Map::from_iter([
            ("kind".to_string(), Value::String("enum".to_string())),
            (
                "values".to_string(),
                Value::Array(vec![
                    Value::String("auto".to_string()),
                    Value::String("metal".to_string()),
                ]),
            ),
        ]));

        for schema in [string_schema, socket_addr_schema, enum_schema] {
            let parsed: ConfigValueSchema = schema
                .clone()
                .try_into()
                .expect("existing schema JSON should stay backward compatible");
            let serialized = Value::try_from(parsed).expect("schema should serialize");
            assert_eq!(serialized, schema);
        }
    }

    #[test]
    fn control_condition_roundtrips_canonical_path_serialization() {
        let condition = ConfigControlCondition {
            path: ConfigPath::parse_rendered("models.<model-ref>.hardware.device")
                .expect("path should parse"),
            operator: ConfigConditionOperator::In,
            values: vec![ConfigConditionValue::String("auto".to_string())],
        };

        let serialized = Value::try_from(condition.clone()).expect("condition should serialize");
        let table = serialized
            .as_table()
            .expect("condition should serialize to a table");
        let path = table
            .get("path")
            .and_then(Value::as_table)
            .expect("condition path should serialize as a table");
        let segments = path
            .get("segments")
            .and_then(Value::as_array)
            .expect("condition path should serialize segments");

        assert_eq!(segments.len(), 4);
        assert_eq!(
            condition.path.render(),
            "models.<model-ref>.hardware.device"
        );

        let roundtrip: ConfigControlCondition =
            serialized.try_into().expect("condition should deserialize");
        assert_eq!(
            roundtrip.path.render(),
            "models.<model-ref>.hardware.device"
        );
    }

    #[test]
    fn disabled_write_policy_defaults_follow_spec() {
        let mut setting: ConfigSettingSchema = legacy_setting_value()
            .try_into()
            .expect("legacy setting should deserialize");

        assert_eq!(setting.default_disabled_write_policy(None), None);

        setting.control_behavior = Some(ConfigControlBehavior::default());
        assert_eq!(
            setting.default_disabled_write_policy(Some(ConfigControlAvailabilitySource::Static)),
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(
            setting.default_disabled_write_policy(Some(ConfigControlAvailabilitySource::Runtime)),
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(
            setting
                .default_disabled_write_policy(Some(ConfigControlAvailabilitySource::Dependency)),
            Some(ConfigDisabledWritePolicy::OmitWhenDisabled)
        );

        setting.support = ConfigSupportState::Unsupported;
        assert_eq!(
            setting.default_disabled_write_policy(Some(ConfigControlAvailabilitySource::Static)),
            Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
        );

        setting.support = ConfigSupportState::Rejected;
        assert_eq!(
            setting
                .default_disabled_write_policy(Some(ConfigControlAvailabilitySource::Dependency)),
            Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
        );
    }

    #[test]
    fn unknown_and_missing_optional_control_behavior_fields_remain_compatible() {
        let mut setting = legacy_setting_value();
        let table = setting
            .as_table_mut()
            .expect("legacy setting should serialize as a table");
        table.insert(
            "unknown_top_level".to_string(),
            Value::String("ignored".to_string()),
        );
        table.insert(
            "control_behavior".to_string(),
            Value::Table(toml::map::Map::from_iter([
                (
                    "numeric".to_string(),
                    Value::Table(toml::map::Map::from_iter([(
                        "min".to_string(),
                        Value::Float(1.0),
                    )])),
                ),
                (
                    "unknown_nested".to_string(),
                    Value::String("ignored".to_string()),
                ),
            ])),
        );

        let parsed: ConfigSettingSchema = setting
            .try_into()
            .expect("setting with unknown and missing optional fields should deserialize");

        let behavior = parsed
            .control_behavior
            .expect("control behavior should deserialize");
        let numeric = behavior
            .numeric
            .expect("numeric control should deserialize");

        assert_eq!(numeric.min, Some(1.0));
        assert_eq!(numeric.max, None);
        assert!(behavior.enable_when.is_empty());
        assert!(behavior.disable_when.is_empty());
        assert!(behavior.conflicts.is_empty());
        assert_eq!(behavior.write_policy, None);
    }

    #[test]
    fn range_condition_without_numeric_values_remains_representable() {
        let condition = ConfigControlCondition {
            path: ConfigPath::parse_rendered("defaults.request.temperature")
                .expect("path should parse"),
            operator: ConfigConditionOperator::Range,
            values: vec![ConfigConditionValue::String("not-a-number".to_string())],
        };

        let serialized = Value::try_from(condition.clone()).expect("condition should serialize");
        let roundtrip: ConfigControlCondition =
            serialized.try_into().expect("condition should deserialize");

        assert_eq!(roundtrip, condition);
    }
}
