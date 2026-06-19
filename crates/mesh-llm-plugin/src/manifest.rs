use crate::{helpers::json_string, json_schema_for, proto};
use anyhow::{Context, Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod control_behavior;

use self::control_behavior::PackagedPluginControlBehavior;

#[cfg(test)]
use self::control_behavior::{
    PackagedPluginDisabledWritePolicy, PackagedPluginOptionsSource, PackagedPluginTextFormat,
};

#[derive(Clone, Debug)]
pub enum ManifestEntry {
    Capability(String),
    ConfigSchema(proto::PluginConfigSchemaManifest),
    Operation(proto::OperationManifest),
    Resource(proto::ResourceManifest),
    ResourceTemplate(proto::ResourceTemplateManifest),
    Prompt(proto::PromptManifest),
    Completion(proto::CompletionManifest),
    HttpBinding(proto::HttpBindingManifest),
    Endpoint(proto::EndpointManifest),
    MeshChannel(proto::MeshChannelManifest),
    MeshEventSubscription(proto::MeshEventSubscriptionManifest),
}

#[derive(Clone, Debug, Default)]
pub struct PluginManifestBuilder {
    manifest: proto::PluginManifest,
}

impl PluginManifestBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn item<T: Into<ManifestEntry>>(mut self, item: T) -> Self {
        self.push(item.into());
        self
    }

    pub fn build(self) -> proto::PluginManifest {
        self.manifest
    }

    pub fn push_item<T: Into<ManifestEntry>>(&mut self, item: T) {
        self.push(item.into());
    }

    fn push(&mut self, item: ManifestEntry) {
        match item {
            ManifestEntry::Capability(capability) => self.manifest.capabilities.push(capability),
            ManifestEntry::ConfigSchema(schema) => self.manifest.config_schema = Some(schema),
            ManifestEntry::Operation(operation) => self.manifest.operations.push(operation),
            ManifestEntry::Resource(resource) => self.manifest.resources.push(resource),
            ManifestEntry::ResourceTemplate(template) => {
                self.manifest.resource_templates.push(template);
            }
            ManifestEntry::Prompt(prompt) => self.manifest.prompts.push(prompt),
            ManifestEntry::Completion(completion) => {
                self.manifest.completions.push(completion);
            }
            ManifestEntry::HttpBinding(binding) => self.manifest.http_bindings.push(binding),
            ManifestEntry::Endpoint(endpoint) => self.manifest.endpoints.push(endpoint),
            ManifestEntry::MeshChannel(channel) => self.manifest.mesh_channels.push(channel),
            ManifestEntry::MeshEventSubscription(subscription) => {
                self.manifest.mesh_event_subscriptions.push(subscription);
            }
        }
    }
}

pub fn plugin_manifest() -> PluginManifestBuilder {
    PluginManifestBuilder::new()
}

pub fn capability(name: impl Into<String>) -> ManifestEntry {
    ManifestEntry::Capability(name.into())
}

pub fn config_schema(plugin_name: impl Into<String>) -> PluginConfigSchemaBuilder {
    PluginConfigSchemaBuilder {
        inner: proto::PluginConfigSchemaManifest {
            plugin_name: plugin_name.into(),
            schema_version: 1,
            allow_unvalidated_config: false,
            settings: Vec::new(),
        },
    }
}

pub fn config_setting(
    key: impl Into<String>,
    value_schema: proto::PluginConfigValueSchema,
) -> PluginConfigSettingBuilder {
    PluginConfigSettingBuilder {
        inner: proto::PluginConfigSettingManifest {
            key: key.into(),
            value_schema: Some(value_schema),
            required: false,
            default_json: None,
            constraints: Vec::new(),
            apply_mode: proto::PluginConfigApplyMode::StaticOnLoad as i32,
            restart_scope: proto::PluginConfigRestartScope::None as i32,
            visibility: proto::PluginConfigVisibility::User as i32,
            description: None,
            presentation: None,
            control_behavior: None,
        },
    }
}

pub fn config_boolean() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::Boolean)
}

pub fn config_integer() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::Integer)
}

pub fn config_float() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::Float)
}

pub fn config_string() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::String)
}

pub fn config_path() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::Path)
}

pub fn config_url() -> proto::PluginConfigValueSchema {
    value_schema(proto::PluginConfigValueKind::Url)
}

pub fn config_enum<I, S>(values: I) -> proto::PluginConfigValueSchema
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut schema = value_schema(proto::PluginConfigValueKind::Enum);
    schema.enum_values = values.into_iter().map(Into::into).collect();
    schema
}

pub fn config_array(items: proto::PluginConfigValueSchema) -> proto::PluginConfigValueSchema {
    let mut schema = value_schema(proto::PluginConfigValueKind::Array);
    schema.items = Some(Box::new(items));
    schema
}

pub fn config_object<I>(properties: I) -> proto::PluginConfigValueSchema
where
    I: IntoIterator<Item = proto::PluginConfigObjectProperty>,
{
    let mut schema = value_schema(proto::PluginConfigValueKind::Object);
    schema.object_properties = properties.into_iter().collect();
    schema
}

pub fn config_object_property(
    key: impl Into<String>,
    value_schema: proto::PluginConfigValueSchema,
) -> PluginConfigObjectPropertyBuilder {
    PluginConfigObjectPropertyBuilder {
        inner: proto::PluginConfigObjectProperty {
            key: key.into(),
            value_schema: Some(value_schema),
            required: false,
            description: None,
        },
    }
}

pub fn constraint_non_empty() -> proto::PluginConfigConstraintManifest {
    proto::PluginConfigConstraintManifest {
        constraint: Some(
            proto::plugin_config_constraint_manifest::Constraint::NonEmpty(
                proto::PluginConfigNonEmptyConstraint {},
            ),
        ),
    }
}

pub fn constraint_positive() -> proto::PluginConfigConstraintManifest {
    proto::PluginConfigConstraintManifest {
        constraint: Some(
            proto::plugin_config_constraint_manifest::Constraint::Positive(
                proto::PluginConfigPositiveConstraint {},
            ),
        ),
    }
}

pub fn constraint_range(
    min: Option<impl Into<String>>,
    max: Option<impl Into<String>>,
) -> proto::PluginConfigConstraintManifest {
    proto::PluginConfigConstraintManifest {
        constraint: Some(proto::plugin_config_constraint_manifest::Constraint::Range(
            proto::PluginConfigRangeConstraint {
                min: min.map(Into::into),
                max: max.map(Into::into),
            },
        )),
    }
}

pub fn constraint_allowed_values<I, S>(values: I) -> proto::PluginConfigConstraintManifest
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    proto::PluginConfigConstraintManifest {
        constraint: Some(
            proto::plugin_config_constraint_manifest::Constraint::AllowedValues(
                proto::PluginConfigAllowedValuesConstraint {
                    values: values.into_iter().map(Into::into).collect(),
                },
            ),
        ),
    }
}

pub fn constraint_requires(key: impl Into<String>) -> proto::PluginConfigConstraintManifest {
    proto::PluginConfigConstraintManifest {
        constraint: Some(
            proto::plugin_config_constraint_manifest::Constraint::Requires(
                proto::PluginConfigRequiresConstraint { key: key.into() },
            ),
        ),
    }
}

pub fn package_manifest_json(manifest: &proto::PluginManifest) -> Result<String> {
    let packaged = PackagedPluginManifest::try_from(manifest)?;
    Ok(serde_json::to_string_pretty(&packaged)?)
}

fn value_schema(kind: proto::PluginConfigValueKind) -> proto::PluginConfigValueSchema {
    proto::PluginConfigValueSchema {
        kind: kind as i32,
        enum_values: Vec::new(),
        items: None,
        object_properties: Vec::new(),
        allow_additional_properties: false,
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
struct PackagedPluginManifest {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    config_schema: Option<PackagedPluginConfigSchema>,
}

impl TryFrom<&proto::PluginManifest> for PackagedPluginManifest {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginManifest) -> Result<Self> {
        let config_schema = value
            .config_schema
            .as_ref()
            .map(PackagedPluginConfigSchema::try_from)
            .transpose()?;

        Ok(Self { config_schema })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct PackagedPluginConfigSchema {
    plugin_name: String,
    schema_version: u32,
    #[serde(default)]
    allow_unvalidated_config: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    settings: Vec<PackagedPluginSetting>,
}

impl TryFrom<&proto::PluginConfigSchemaManifest> for PackagedPluginConfigSchema {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigSchemaManifest) -> Result<Self> {
        let settings = value
            .settings
            .iter()
            .map(PackagedPluginSetting::try_from)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            plugin_name: value.plugin_name.clone(),
            schema_version: value.schema_version,
            allow_unvalidated_config: value.allow_unvalidated_config,
            settings,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct PackagedPluginSetting {
    key: String,
    value_schema: PackagedPluginValueSchema,
    #[serde(default)]
    required: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    default_json: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    constraints: Vec<PackagedPluginConstraint>,
    apply_mode: PackagedPluginApplyMode,
    restart_scope: PackagedPluginRestartScope,
    visibility: PackagedPluginVisibility,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    presentation: Option<PackagedPluginPresentation>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    control_behavior: Option<PackagedPluginControlBehavior>,
}

impl TryFrom<&proto::PluginConfigSettingManifest> for PackagedPluginSetting {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigSettingManifest) -> Result<Self> {
        let value_schema = value.value_schema.as_ref().ok_or_else(|| {
            anyhow!(
                "plugin config setting `{}` is missing value_schema",
                value.key
            )
        })?;
        let constraints = value
            .constraints
            .iter()
            .enumerate()
            .map(|(index, constraint)| {
                PackagedPluginConstraint::try_from(constraint).with_context(|| {
                    format!(
                        "plugin config setting `{}` has invalid constraint #{}",
                        value.key,
                        index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            key: value.key.clone(),
            value_schema: PackagedPluginValueSchema::try_from(value_schema).with_context(|| {
                format!(
                    "plugin config setting `{}` has invalid value_schema",
                    value.key
                )
            })?,
            required: value.required,
            default_json: value.default_json.clone(),
            constraints,
            apply_mode: PackagedPluginApplyMode::try_from_i32(value.apply_mode).with_context(
                || {
                    format!(
                        "plugin config setting `{}` has invalid apply_mode",
                        value.key
                    )
                },
            )?,
            restart_scope: PackagedPluginRestartScope::try_from_i32(value.restart_scope)
                .with_context(|| {
                    format!(
                        "plugin config setting `{}` has invalid restart_scope",
                        value.key
                    )
                })?,
            visibility: PackagedPluginVisibility::try_from_i32(value.visibility).with_context(
                || {
                    format!(
                        "plugin config setting `{}` has invalid visibility",
                        value.key
                    )
                },
            )?,
            description: value.description.clone(),
            presentation: value
                .presentation
                .as_ref()
                .map(PackagedPluginPresentation::from),
            control_behavior: value
                .control_behavior
                .as_ref()
                .map(PackagedPluginControlBehavior::try_from)
                .transpose()
                .with_context(|| {
                    format!(
                        "plugin config setting `{}` has invalid control_behavior",
                        value.key
                    )
                })?,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
struct PackagedPluginPresentation {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    help: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    category_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    category_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    category_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    category_order: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    setting_order: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    placeholder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    control_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    renderer_id: Option<String>,
}

impl From<&proto::PluginConfigPresentationManifest> for PackagedPluginPresentation {
    fn from(value: &proto::PluginConfigPresentationManifest) -> Self {
        Self {
            label: value.label.clone(),
            help: value.help.clone(),
            category_id: value.category_id.clone(),
            category_label: value.category_label.clone(),
            category_summary: value.category_summary.clone(),
            category_order: value.category_order,
            setting_order: value.setting_order,
            unit: value.unit.clone(),
            placeholder: value.placeholder.clone(),
            control_hint: value.control_hint.clone(),
            renderer_id: value.renderer_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PackagedPluginValueSchema {
    kind: PackagedPluginValueKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    enum_values: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    items: Option<Box<PackagedPluginValueSchema>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    object_properties: Vec<PackagedPluginObjectProperty>,
    #[serde(default)]
    allow_additional_properties: bool,
}

impl TryFrom<&proto::PluginConfigValueSchema> for PackagedPluginValueSchema {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigValueSchema) -> Result<Self> {
        let items = value
            .items
            .as_ref()
            .map(|items| PackagedPluginValueSchema::try_from(items.as_ref()).map(Box::new))
            .transpose()
            .context("array items schema is invalid")?;
        let object_properties = value
            .object_properties
            .iter()
            .map(PackagedPluginObjectProperty::try_from)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            kind: PackagedPluginValueKind::try_from_i32(value.kind)?,
            enum_values: value.enum_values.clone(),
            items,
            object_properties,
            allow_additional_properties: value.allow_additional_properties,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PackagedPluginObjectProperty {
    key: String,
    value_schema: PackagedPluginValueSchema,
    #[serde(default)]
    required: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    description: Option<String>,
}

impl TryFrom<&proto::PluginConfigObjectProperty> for PackagedPluginObjectProperty {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigObjectProperty) -> Result<Self> {
        let value_schema = value.value_schema.as_ref().ok_or_else(|| {
            anyhow!(
                "plugin config object property `{}` is missing value_schema",
                value.key
            )
        })?;

        Ok(Self {
            key: value.key.clone(),
            value_schema: PackagedPluginValueSchema::try_from(value_schema).with_context(|| {
                format!(
                    "plugin config object property `{}` has invalid value_schema",
                    value.key
                )
            })?,
            required: value.required,
            description: value.description.clone(),
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackagedPluginValueKind {
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

impl PackagedPluginValueKind {
    fn try_from_i32(value: i32) -> Result<Self> {
        let kind = match proto::PluginConfigValueKind::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config value kind `{value}`"))?
        {
            proto::PluginConfigValueKind::Boolean => Self::Boolean,
            proto::PluginConfigValueKind::Integer => Self::Integer,
            proto::PluginConfigValueKind::Float => Self::Float,
            proto::PluginConfigValueKind::String => Self::String,
            proto::PluginConfigValueKind::Path => Self::Path,
            proto::PluginConfigValueKind::Url => Self::Url,
            proto::PluginConfigValueKind::Enum => Self::Enum,
            proto::PluginConfigValueKind::Array => Self::Array,
            proto::PluginConfigValueKind::Object => Self::Object,
            proto::PluginConfigValueKind::Unspecified => {
                return Err(anyhow!("plugin config value kind is unspecified"));
            }
        };
        Ok(kind)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackagedPluginApplyMode {
    StaticOnLoad,
    DynamicValidationOnly,
    DynamicApply,
}

impl PackagedPluginApplyMode {
    fn try_from_i32(value: i32) -> Result<Self> {
        let mode = match proto::PluginConfigApplyMode::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config apply mode `{value}`"))?
        {
            proto::PluginConfigApplyMode::StaticOnLoad => Self::StaticOnLoad,
            proto::PluginConfigApplyMode::DynamicValidationOnly => Self::DynamicValidationOnly,
            proto::PluginConfigApplyMode::DynamicApply => Self::DynamicApply,
            proto::PluginConfigApplyMode::Unspecified => {
                return Err(anyhow!("plugin config apply mode is unspecified"));
            }
        };
        Ok(mode)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackagedPluginRestartScope {
    None,
    ModelReload,
    ProcessRestart,
    MeshRestart,
    PluginProcess,
}

impl PackagedPluginRestartScope {
    fn try_from_i32(value: i32) -> Result<Self> {
        let scope = match proto::PluginConfigRestartScope::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config restart scope `{value}`"))?
        {
            proto::PluginConfigRestartScope::None => Self::None,
            proto::PluginConfigRestartScope::ModelReload => Self::ModelReload,
            proto::PluginConfigRestartScope::ProcessRestart => Self::ProcessRestart,
            proto::PluginConfigRestartScope::MeshRestart => Self::MeshRestart,
            proto::PluginConfigRestartScope::PluginProcess => Self::PluginProcess,
            proto::PluginConfigRestartScope::Unspecified => {
                return Err(anyhow!("plugin config restart scope is unspecified"));
            }
        };
        Ok(scope)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackagedPluginVisibility {
    User,
    Advanced,
    Hidden,
    Internal,
}

impl PackagedPluginVisibility {
    fn try_from_i32(value: i32) -> Result<Self> {
        let visibility = match proto::PluginConfigVisibility::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config visibility `{value}`"))?
        {
            proto::PluginConfigVisibility::User => Self::User,
            proto::PluginConfigVisibility::Advanced => Self::Advanced,
            proto::PluginConfigVisibility::Hidden => Self::Hidden,
            proto::PluginConfigVisibility::Internal => Self::Internal,
            proto::PluginConfigVisibility::Unspecified => {
                return Err(anyhow!("plugin config visibility is unspecified"));
            }
        };
        Ok(visibility)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PackagedPluginConstraint {
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

impl PackagedPluginConstraint {
    fn try_from(value: &proto::PluginConfigConstraintManifest) -> Result<Self> {
        match value
            .constraint
            .as_ref()
            .ok_or_else(|| anyhow!("plugin config constraint is empty"))?
        {
            proto::plugin_config_constraint_manifest::Constraint::NonEmpty(_) => Ok(Self::NonEmpty),
            proto::plugin_config_constraint_manifest::Constraint::Positive(_) => Ok(Self::Positive),
            proto::plugin_config_constraint_manifest::Constraint::Range(range) => Ok(Self::Range {
                min: range.min.clone(),
                max: range.max.clone(),
            }),
            proto::plugin_config_constraint_manifest::Constraint::AllowedValues(values) => {
                Ok(Self::AllowedValues {
                    values: values.values.clone(),
                })
            }
            proto::plugin_config_constraint_manifest::Constraint::Requires(requires) => {
                Ok(Self::Requires {
                    key: requires.key.clone(),
                })
            }
        }
    }
}

pub fn mesh_channel(name: impl Into<String>) -> ManifestEntry {
    ManifestEntry::MeshChannel(proto::MeshChannelManifest { name: name.into() })
}

pub fn mesh_event_subscription(kind: proto::mesh_event::Kind) -> ManifestEntry {
    ManifestEntry::MeshEventSubscription(proto::MeshEventSubscriptionManifest { kind: kind as i32 })
}

pub fn mesh_event_peer_up() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerUp)
}

pub fn mesh_event_peer_down() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerDown)
}

pub fn mesh_event_peer_updated() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::PeerUpdated)
}

pub fn mesh_event_local_accepting() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::LocalAccepting)
}

pub fn mesh_event_local_standby() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::LocalStandby)
}

pub fn mesh_event_mesh_id_updated() -> ManifestEntry {
    mesh_event_subscription(proto::mesh_event::Kind::MeshIdUpdated)
}

impl From<proto::MeshChannelManifest> for ManifestEntry {
    fn from(value: proto::MeshChannelManifest) -> Self {
        Self::MeshChannel(value)
    }
}

impl From<proto::PluginConfigSchemaManifest> for ManifestEntry {
    fn from(value: proto::PluginConfigSchemaManifest) -> Self {
        Self::ConfigSchema(value)
    }
}

#[derive(Clone, Debug)]
pub struct PluginConfigSchemaBuilder {
    inner: proto::PluginConfigSchemaManifest,
}

impl PluginConfigSchemaBuilder {
    pub fn schema_version(mut self, schema_version: u32) -> Self {
        self.inner.schema_version = schema_version;
        self
    }

    pub fn allow_unvalidated_config(mut self, allow_unvalidated_config: bool) -> Self {
        self.inner.allow_unvalidated_config = allow_unvalidated_config;
        self
    }

    pub fn setting<T: Into<proto::PluginConfigSettingManifest>>(mut self, setting: T) -> Self {
        self.inner.settings.push(setting.into());
        self
    }
}

impl From<PluginConfigSchemaBuilder> for ManifestEntry {
    fn from(value: PluginConfigSchemaBuilder) -> Self {
        Self::ConfigSchema(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct PluginConfigSettingBuilder {
    inner: proto::PluginConfigSettingManifest,
}

impl PluginConfigSettingBuilder {
    pub fn required(mut self, required: bool) -> Self {
        self.inner.required = required;
        self
    }

    pub fn default_value<T: Serialize>(mut self, value: &T) -> Self {
        self.inner.default_json = json_string(value).ok();
        self
    }

    pub fn constraint(mut self, constraint: proto::PluginConfigConstraintManifest) -> Self {
        self.inner.constraints.push(constraint);
        self
    }

    pub fn apply_mode(mut self, apply_mode: proto::PluginConfigApplyMode) -> Self {
        self.inner.apply_mode = apply_mode as i32;
        self
    }

    pub fn restart_scope(mut self, restart_scope: proto::PluginConfigRestartScope) -> Self {
        self.inner.restart_scope = restart_scope as i32;
        self
    }

    pub fn visibility(mut self, visibility: proto::PluginConfigVisibility) -> Self {
        self.inner.visibility = visibility as i32;
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.presentation_mut().label = Some(label.into());
        self
    }

    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.presentation_mut().help = Some(help.into());
        self
    }

    pub fn category(
        mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        summary: impl Into<String>,
        order: u32,
    ) -> Self {
        let presentation = self.presentation_mut();
        presentation.category_id = Some(id.into());
        presentation.category_label = Some(label.into());
        presentation.category_summary = Some(summary.into());
        presentation.category_order = Some(order);
        self
    }

    pub fn order(mut self, order: u32) -> Self {
        self.presentation_mut().setting_order = Some(order);
        self
    }

    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.presentation_mut().unit = Some(unit.into());
        self
    }

    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.presentation_mut().placeholder = Some(placeholder.into());
        self
    }

    pub fn control_hint(mut self, control_hint: impl Into<String>) -> Self {
        self.presentation_mut().control_hint = Some(control_hint.into());
        self
    }

    pub fn renderer_id(mut self, renderer_id: impl Into<String>) -> Self {
        self.presentation_mut().renderer_id = Some(renderer_id.into());
        self
    }

    fn presentation_mut(&mut self) -> &mut proto::PluginConfigPresentationManifest {
        self.inner
            .presentation
            .get_or_insert_with(proto::PluginConfigPresentationManifest::default)
    }
}

impl From<PluginConfigSettingBuilder> for proto::PluginConfigSettingManifest {
    fn from(value: PluginConfigSettingBuilder) -> Self {
        value.inner
    }
}

#[derive(Clone, Debug)]
pub struct PluginConfigObjectPropertyBuilder {
    inner: proto::PluginConfigObjectProperty,
}

impl PluginConfigObjectPropertyBuilder {
    pub fn required(mut self, required: bool) -> Self {
        self.inner.required = required;
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }
}

impl From<PluginConfigObjectPropertyBuilder> for proto::PluginConfigObjectProperty {
    fn from(value: PluginConfigObjectPropertyBuilder) -> Self {
        value.inner
    }
}

impl From<proto::MeshEventSubscriptionManifest> for ManifestEntry {
    fn from(value: proto::MeshEventSubscriptionManifest) -> Self {
        Self::MeshEventSubscription(value)
    }
}

#[derive(Clone, Debug)]
pub struct OperationBuilder {
    inner: proto::OperationManifest,
}

pub fn operation<Input: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> OperationBuilder {
    OperationBuilder {
        inner: proto::OperationManifest {
            name: name.into(),
            description: description.into(),
            input_schema_json: schema_json::<Input>(),
            output_schema_json: None,
            title: None,
        },
    }
}

impl OperationBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.inner.title = Some(title.into());
        self
    }

    pub fn output_schema<Output: JsonSchema>(mut self) -> Self {
        self.inner.output_schema_json = Some(schema_json::<Output>());
        self
    }
}

impl From<OperationBuilder> for ManifestEntry {
    fn from(value: OperationBuilder) -> Self {
        Self::Operation(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct ResourceBuilder {
    inner: proto::ResourceManifest,
}

pub fn resource(uri: impl Into<String>, name: impl Into<String>) -> ResourceBuilder {
    ResourceBuilder {
        inner: proto::ResourceManifest {
            uri: uri.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        },
    }
}

impl ResourceBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.inner.mime_type = Some(mime_type.into());
        self
    }
}

impl From<ResourceBuilder> for ManifestEntry {
    fn from(value: ResourceBuilder) -> Self {
        Self::Resource(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct ResourceTemplateBuilder {
    inner: proto::ResourceTemplateManifest,
}

pub fn resource_template_service(
    uri_template: impl Into<String>,
    name: impl Into<String>,
) -> ResourceTemplateBuilder {
    ResourceTemplateBuilder {
        inner: proto::ResourceTemplateManifest {
            uri_template: uri_template.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        },
    }
}

impl ResourceTemplateBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }

    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.inner.mime_type = Some(mime_type.into());
        self
    }
}

impl From<ResourceTemplateBuilder> for ManifestEntry {
    fn from(value: ResourceTemplateBuilder) -> Self {
        Self::ResourceTemplate(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct PromptBuilder {
    inner: proto::PromptManifest,
}

pub fn prompt_service(name: impl Into<String>) -> PromptBuilder {
    PromptBuilder {
        inner: proto::PromptManifest {
            name: name.into(),
            description: None,
        },
    }
}

impl PromptBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }
}

impl From<PromptBuilder> for ManifestEntry {
    fn from(value: PromptBuilder) -> Self {
        Self::Prompt(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct CompletionBuilder {
    inner: proto::CompletionManifest,
}

pub fn completion(argument_ref: impl Into<String>) -> CompletionBuilder {
    CompletionBuilder {
        inner: proto::CompletionManifest {
            argument_ref: argument_ref.into(),
            description: None,
        },
    }
}

impl CompletionBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner.description = Some(description.into());
        self
    }
}

impl From<CompletionBuilder> for ManifestEntry {
    fn from(value: CompletionBuilder) -> Self {
        Self::Completion(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct HttpBindingBuilder {
    inner: proto::HttpBindingManifest,
}

pub fn http_binding(
    method: proto::HttpMethod,
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    let path = normalize_path(path.into());
    let operation_name = operation_name.into();
    HttpBindingBuilder {
        inner: proto::HttpBindingManifest {
            binding_id: default_binding_id(&path, &operation_name),
            method: method as i32,
            path,
            operation_name: Some(operation_name),
            request_body_mode: proto::HttpBodyMode::Buffered as i32,
            response_body_mode: proto::HttpBodyMode::Buffered as i32,
            request_schema_json: None,
            response_schema_json: None,
        },
    }
}

pub fn http_get(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Get, path, operation_name)
}

pub fn http_post(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Post, path, operation_name)
}

pub fn http_put(path: impl Into<String>, operation_name: impl Into<String>) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Put, path, operation_name)
}

pub fn http_patch(
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Patch, path, operation_name)
}

pub fn http_delete(
    path: impl Into<String>,
    operation_name: impl Into<String>,
) -> HttpBindingBuilder {
    http_binding(proto::HttpMethod::Delete, path, operation_name)
}

impl HttpBindingBuilder {
    pub fn binding_id(mut self, binding_id: impl Into<String>) -> Self {
        self.inner.binding_id = binding_id.into();
        self
    }

    pub fn request_schema<Request: JsonSchema>(mut self) -> Self {
        self.inner.request_schema_json = Some(schema_json::<Request>());
        self
    }

    pub fn response_schema<Response: JsonSchema>(mut self) -> Self {
        self.inner.response_schema_json = Some(schema_json::<Response>());
        self
    }

    pub fn streamed_request(mut self) -> Self {
        self.inner.request_body_mode = proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn streamed_response(mut self) -> Self {
        self.inner.response_body_mode = proto::HttpBodyMode::Streamed as i32;
        self
    }

    pub fn buffered_request(mut self) -> Self {
        self.inner.request_body_mode = proto::HttpBodyMode::Buffered as i32;
        self
    }

    pub fn buffered_response(mut self) -> Self {
        self.inner.response_body_mode = proto::HttpBodyMode::Buffered as i32;
        self
    }
}

impl From<HttpBindingBuilder> for ManifestEntry {
    fn from(value: HttpBindingBuilder) -> Self {
        Self::HttpBinding(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct EndpointBuilder {
    inner: proto::EndpointManifest,
}

pub fn openai_http_inference_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_stdio_endpoint(
    endpoint_id: impl Into<String>,
    command: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportStdio as i32,
            protocol: None,
            address: Some(command.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_http_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("streamable_http".into()),
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_tcp_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportTcp as i32,
            protocol: None,
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

pub fn mcp_unix_socket_endpoint(
    endpoint_id: impl Into<String>,
    address: impl Into<String>,
) -> EndpointBuilder {
    EndpointBuilder {
        inner: proto::EndpointManifest {
            endpoint_id: endpoint_id.into(),
            kind: proto::EndpointKind::Mcp as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportUnixSocket as i32,
            protocol: None,
            address: Some(address.into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
        },
    }
}

impl EndpointBuilder {
    pub fn protocol(mut self, protocol: impl Into<String>) -> Self {
        self.inner.protocol = Some(protocol.into());
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.inner.namespace = Some(namespace.into());
        self
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.inner.args.push(arg.into());
        self
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn supports_streaming(mut self, supports_streaming: bool) -> Self {
        self.inner.supports_streaming = supports_streaming;
        self
    }

    pub fn managed_by_plugin(mut self, managed_by_plugin: bool) -> Self {
        self.inner.managed_by_plugin = managed_by_plugin;
        self
    }
}

impl From<EndpointBuilder> for ManifestEntry {
    fn from(value: EndpointBuilder) -> Self {
        Self::Endpoint(value.inner)
    }
}

fn schema_json<T: JsonSchema>() -> String {
    json_string(&json_schema_for::<T>()).unwrap_or_else(|_| "{}".into())
}

fn normalize_path(path: String) -> String {
    if path.is_empty() {
        "/".into()
    } else if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    }
}

fn default_binding_id(path: &str, operation_name: &str) -> String {
    let candidate = if !operation_name.trim().is_empty() {
        operation_name
    } else {
        path.trim_matches('/')
    };
    let sanitized = candidate
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('_');
    if sanitized.is_empty() {
        "root".into()
    } else {
        sanitized.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Plugin, PluginMetadata, inference, mcp, plugin_server_info};

    #[allow(dead_code)]
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct DemoInput {
        value: String,
    }

    #[allow(dead_code)]
    #[derive(serde::Serialize, schemars::JsonSchema)]
    struct DemoOutput {
        echoed: String,
    }

    fn manifest_with_setting(setting: proto::PluginConfigSettingManifest) -> proto::PluginManifest {
        proto::PluginManifest {
            config_schema: Some(proto::PluginConfigSchemaManifest {
                plugin_name: "demo".into(),
                schema_version: 1,
                settings: vec![setting],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn error_chain_contains(error: &anyhow::Error, needle: &str) -> bool {
        error
            .chain()
            .any(|cause| cause.to_string().contains(needle))
    }

    #[test]
    fn macro_builds_manifest_entries() {
        let manifest = crate::plugin_manifest![
            capability("demo.v1"),
            mesh_channel("demo.v1"),
            mesh_event_peer_up(),
            operation::<DemoInput>("echo", "Echo input").title("Echo"),
            http_post("/echo", "echo")
                .request_schema::<DemoInput>()
                .response_schema::<DemoOutput>(),
            mcp_stdio_endpoint("notes", "demo-mcp").arg("--serve"),
        ];

        assert_eq!(manifest.capabilities, vec!["demo.v1"]);
        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(manifest.http_bindings.len(), 1);
        assert_eq!(manifest.endpoints.len(), 1);
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert_eq!(manifest.mesh_event_subscriptions.len(), 1);
        assert_eq!(manifest.http_bindings[0].binding_id, "echo");
        assert_eq!(manifest.endpoints[0].args, vec!["--serve"]);
    }

    #[test]
    fn streaming_http_builder_sets_modes() {
        let entry: ManifestEntry = http_post("/upload", "upload")
            .streamed_request()
            .streamed_response()
            .into();
        let ManifestEntry::HttpBinding(binding) = entry else {
            panic!("expected http binding");
        };
        assert_eq!(
            binding.request_body_mode,
            proto::HttpBodyMode::Streamed as i32
        );
        assert_eq!(
            binding.response_body_mode,
            proto::HttpBodyMode::Streamed as i32
        );
    }

    #[test]
    fn plugin_macro_builds_simple_plugin_with_manifest() {
        let plugin = crate::plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            provides: [capability("demo.v1")],
            mesh: [mesh_channel("demo.v1")],
            events: [mesh_event_peer_up()],
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
                mcp::external_stdio("stdio", "demo-mcp"),
            ],
            http: [
                crate::http::post("/echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .output::<DemoOutput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
            ],
            inference: [
                inference::openai_http("local", "http://127.0.0.1:8080/v1"),
            ],
        };

        let manifest = plugin.manifest().expect("manifest");
        assert_eq!(plugin.capabilities(), vec!["demo.v1"]);
        assert_eq!(manifest.capabilities, vec!["demo.v1"]);
        assert_eq!(manifest.operations.len(), 2);
        assert_eq!(manifest.http_bindings.len(), 1);
        assert_eq!(manifest.endpoints.len(), 2);
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert_eq!(manifest.mesh_event_subscriptions.len(), 1);
    }

    #[test]
    fn declarative_macro_builds_local_mcp_entries() {
        let plugin = crate::plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            provides: [capability("demo.v1")],
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoInput>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(DemoOutput { echoed: args.value })
                    })),
                mcp::resource("demo://snapshot")
                    .name("Snapshot")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::read_resource_result(vec![
                            rmcp::model::ResourceContents::text("snapshot", request.uri),
                        ]))
                    })),
                mcp::prompt("brief")
                    .description("Brief prompt")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::get_prompt_result(vec![
                            rmcp::model::PromptMessage::new(
                                rmcp::model::PromptMessageRole::User,
                                rmcp::model::PromptMessageContent::text(request.name),
                            ),
                        ]))
                    })),
                mcp::completion("prompt.brief.topic")
                    .description("Topic completion")
                    .handle(|_request, _context| Box::pin(async move {
                        crate::complete_result(vec!["alpha".into()])
                    })),
            ],
        };

        let manifest = plugin.manifest().expect("manifest");
        assert_eq!(manifest.operations.len(), 1);
        assert_eq!(manifest.resources.len(), 1);
        assert_eq!(manifest.prompts.len(), 1);
        assert_eq!(manifest.completions.len(), 1);
    }

    #[test]
    fn manifest_can_embed_packaged_config_schema() {
        let manifest = crate::plugin_manifest![
            config_schema("demo")
                .setting(
                    config_setting("retention_days", config_integer())
                        .required(true)
                        .default_value(&14)
                        .constraint(constraint_range(Some("1"), Some("365")))
                        .apply_mode(proto::PluginConfigApplyMode::DynamicValidationOnly)
                        .restart_scope(proto::PluginConfigRestartScope::PluginProcess)
                        .description("How long to retain entries."),
                )
                .setting(
                    config_setting("mode", config_enum(["strict", "relaxed"]))
                        .default_value(&"strict")
                        .constraint(constraint_allowed_values(["strict", "relaxed"])),
                )
        ];

        let schema = manifest.config_schema.expect("config schema");
        assert_eq!(schema.plugin_name, "demo");
        assert_eq!(schema.schema_version, 1);
        assert_eq!(schema.settings.len(), 2);
        assert_eq!(schema.settings[0].default_json.as_deref(), Some("14"));
    }

    #[test]
    fn packaged_manifest_json_includes_config_schema() {
        let manifest = crate::plugin_manifest![
            config_schema("demo")
                .allow_unvalidated_config(true)
                .setting(
                    config_setting("legacy", config_boolean())
                        .default_value(&true)
                        .label("Legacy mode")
                        .help("Enable the legacy compatibility path.")
                        .category("compat", "Compatibility", "Compatibility settings", 20)
                        .order(10)
                        .control_hint("toggle"),
                )
        ];

        let encoded = package_manifest_json(&manifest).expect("manifest json");
        let decoded: PackagedPluginManifest =
            serde_json::from_str(&encoded).expect("manifest should deserialize");

        let schema = decoded.config_schema.expect("config schema");
        assert!(schema.allow_unvalidated_config);
        assert_eq!(schema.settings[0].key, "legacy");
        assert_eq!(
            schema.settings[0]
                .presentation
                .as_ref()
                .and_then(|presentation| presentation.label.as_deref()),
            Some("Legacy mode")
        );
        assert!(schema.settings[0].control_behavior.is_none());
    }

    #[test]
    fn packaged_manifest_json_omits_control_behavior_for_old_manifests() {
        let manifest = crate::plugin_manifest![config_schema("demo").setting(
            config_setting("legacy", config_string()).description("Legacy free-form setting."),
        )];

        let encoded = package_manifest_json(&manifest).expect("manifest json");
        let decoded: serde_json::Value =
            serde_json::from_str(&encoded).expect("manifest should deserialize");

        assert!(
            !decoded["config_schema"]["settings"][0]
                .as_object()
                .expect("setting object")
                .contains_key("control_behavior")
        );
    }

    #[test]
    fn packaged_manifest_json_roundtrips_control_behavior_metadata() {
        let manifest = crate::plugin_manifest![
            config_schema("demo").setting(
                config_setting("service_url", config_url())
                    .control_text_format(proto::PluginConfigTextFormat::Url)
                    .control_options_runtime_local_models()
                    .control_availability(
                        false,
                        proto::PluginConfigControlAvailabilitySource::Runtime
                    )
                    .control_availability_reason("Waiting for runtime discovery")
                    .control_availability_note("The current value will be preserved.")
                    .control_enable_when(proto::PluginConfigControlCondition {
                        key: "mode".into(),
                        operator: proto::PluginConfigConditionOperator::Equals as i32,
                        values: vec![proto::PluginConfigConditionValue {
                            value: Some(proto::plugin_config_condition_value::Value::StringValue(
                                "remote".into(),
                            ),),
                        }],
                    })
                    .control_disable_when(proto::PluginConfigConditionalDisable {
                        condition: Some(proto::PluginConfigControlCondition {
                            key: "mode".into(),
                            operator: proto::PluginConfigConditionOperator::NotEquals as i32,
                            values: vec![proto::PluginConfigConditionValue {
                                value: Some(
                                    proto::plugin_config_condition_value::Value::StringValue(
                                        "remote".into(),
                                    ),
                                ),
                            }],
                        }),
                        reason: "Remote mode is required".into(),
                        note: Some("Switch mode back to remote to edit this setting.".into()),
                        write_policy: proto::PluginConfigDisabledWritePolicy::PreserveExisting
                            as i32,
                    })
                    .control_conflict(proto::PluginConfigConflictRule {
                        group: "transport".into(),
                        condition: Some(proto::PluginConfigControlCondition {
                            key: "socket_path".into(),
                            operator: proto::PluginConfigConditionOperator::Present as i32,
                            values: Vec::new(),
                        }),
                        reason: "Use either a URL or a socket path.".into(),
                        preferred_key: Some("service_url".into()),
                    })
                    .control_write_policy(proto::PluginConfigDisabledWritePolicy::PreserveExisting),
            )
        ];

        let encoded = package_manifest_json(&manifest).expect("manifest json");
        let decoded: PackagedPluginManifest =
            serde_json::from_str(&encoded).expect("manifest should deserialize");
        let schema = decoded.config_schema.expect("config schema");
        let setting = &schema.settings[0];
        let control_behavior = setting
            .control_behavior
            .as_ref()
            .expect("control behavior should be present");

        assert_eq!(setting.value_schema.kind, PackagedPluginValueKind::Url);
        assert_eq!(
            control_behavior.text_format,
            Some(PackagedPluginTextFormat::Url)
        );
        assert_eq!(
            control_behavior.options_source,
            Some(PackagedPluginOptionsSource::RuntimeLocalModels)
        );
        assert_eq!(
            control_behavior
                .availability
                .as_ref()
                .map(|availability| availability.enabled),
            Some(false)
        );
        assert_eq!(
            control_behavior.write_policy,
            Some(PackagedPluginDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(control_behavior.enable_when.len(), 1);
        assert_eq!(control_behavior.disable_when.len(), 1);
        assert_eq!(control_behavior.conflicts.len(), 1);
    }

    #[test]
    fn packaged_manifest_json_rejects_missing_setting_value_schema() {
        let setting = proto::PluginConfigSettingManifest {
            key: "broken".into(),
            value_schema: None,
            apply_mode: proto::PluginConfigApplyMode::StaticOnLoad as i32,
            restart_scope: proto::PluginConfigRestartScope::None as i32,
            visibility: proto::PluginConfigVisibility::User as i32,
            ..Default::default()
        };

        let error = package_manifest_json(&manifest_with_setting(setting))
            .expect_err("missing value_schema should fail packaging");

        assert!(
            error_chain_contains(&error, "missing value_schema"),
            "{error}"
        );
        assert!(error_chain_contains(&error, "broken"), "{error}");
    }

    #[test]
    fn packaged_manifest_json_rejects_empty_constraint_payload() {
        let mut setting: proto::PluginConfigSettingManifest =
            config_setting("mode", config_string()).into();
        setting
            .constraints
            .push(proto::PluginConfigConstraintManifest { constraint: None });

        let error = package_manifest_json(&manifest_with_setting(setting))
            .expect_err("empty constraint should fail packaging");

        assert!(
            error_chain_contains(&error, "invalid constraint #1"),
            "{error}"
        );
        assert!(
            error_chain_contains(&error, "constraint is empty"),
            "{error}"
        );
    }

    #[test]
    fn packaged_manifest_json_rejects_unknown_enum_discriminants() {
        let mut setting: proto::PluginConfigSettingManifest =
            config_setting("mode", config_string()).into();
        setting.apply_mode = 99_999;

        let error = package_manifest_json(&manifest_with_setting(setting))
            .expect_err("unknown apply mode should fail packaging");

        assert!(
            error_chain_contains(&error, "invalid apply_mode"),
            "{error}"
        );
        assert!(
            error_chain_contains(&error, "unknown plugin config apply mode"),
            "{error}"
        );
    }
}
