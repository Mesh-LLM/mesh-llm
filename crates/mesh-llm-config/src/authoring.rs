use crate::{
    ConfigAliasMode, ConfigApplyMode, ConfigConditionalDisable, ConfigConflictRule,
    ConfigConstraint, ConfigControlAvailability, ConfigControlAvailabilitySource,
    ConfigControlBehavior, ConfigControlCondition, ConfigControlSurface, ConfigDisabledWritePolicy,
    ConfigNumericControl, ConfigOptionsSource, ConfigPath, ConfigPathAlias,
    ConfigPresentationMetadata, ConfigRestartScope, ConfigSchema, ConfigSettingOwner,
    ConfigSettingSchema, ConfigSupportState, ConfigTextFormat, ConfigValueSchema, ConfigVisibility,
    GpuAssignment, HardwareConfig, MeshConfig, ModelConfigDefaults, ModelConfigEntry,
    ModelFitConfig, MultimodalConfig, PluginConfigEntry, RequestDefaultsConfig, ThroughputConfig,
};
use anyhow::{Result, bail};
use mesh_llm_types::runtime::ModelRuntimeKind;
use std::net::SocketAddr;

#[derive(Clone, Debug, Default)]
pub struct LocalServingNodeConfig {
    pub model: String,
    pub runtime: Option<ModelRuntimeKind>,
    pub device: Option<String>,
    pub context_size: Option<u32>,
    pub parallel: Option<usize>,
    pub mmproj: Option<String>,
    pub owner_control_bind: Option<SocketAddr>,
    pub owner_control_advertise_addr: Option<SocketAddr>,
    pub gpu_assignment: Option<GpuAssignment>,
}

#[derive(Clone, Debug, Default)]
pub struct ConfigSchemaBuilder {
    settings: Vec<ConfigSettingSchema>,
}

impl ConfigSchemaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn setting(&mut self, setting: ConfigSettingSchema) -> &mut Self {
        self.settings.push(setting);
        self
    }

    pub fn build(self) -> ConfigSchema {
        ConfigSchema {
            settings: self.settings,
        }
    }
}

pub fn built_in_config_schema() -> ConfigSchema {
    ConfigSchema {
        settings: crate::built_in_config_settings(),
    }
}

#[derive(Clone, Debug)]
pub struct ConfigSettingSchemaBuilder {
    setting: ConfigSettingSchema,
}

impl ConfigSettingSchemaBuilder {
    pub fn new(path: ConfigPath, value_schema: ConfigValueSchema) -> Self {
        Self {
            setting: ConfigSettingSchema {
                path,
                alias_policy: Default::default(),
                owner: ConfigSettingOwner::BuiltIn,
                value_schema,
                support: ConfigSupportState::Supported,
                control_surfaces: Vec::new(),
                apply_mode: ConfigApplyMode::StaticOnLoad,
                restart_scope: ConfigRestartScope::None,
                visibility: ConfigVisibility::User,
                constraints: Vec::new(),
                description: None,
                presentation: None,
                control_behavior: None,
            },
        }
    }

    pub fn owner(&mut self, owner: ConfigSettingOwner) -> &mut Self {
        self.setting.owner = owner;
        self
    }

    pub fn support(&mut self, support: ConfigSupportState) -> &mut Self {
        self.setting.support = support;
        self
    }

    pub fn control_surface(&mut self, surface: ConfigControlSurface) -> &mut Self {
        self.setting.control_surfaces.push(surface);
        self
    }

    pub fn apply_mode(&mut self, apply_mode: ConfigApplyMode) -> &mut Self {
        self.setting.apply_mode = apply_mode;
        self
    }

    pub fn restart_scope(&mut self, restart_scope: ConfigRestartScope) -> &mut Self {
        self.setting.restart_scope = restart_scope;
        self
    }

    pub fn visibility(&mut self, visibility: ConfigVisibility) -> &mut Self {
        self.setting.visibility = visibility;
        self
    }

    pub fn description(&mut self, description: impl Into<String>) -> &mut Self {
        self.setting.description = Some(description.into());
        self
    }

    pub fn presentation(&mut self, presentation: ConfigPresentationMetadata) -> &mut Self {
        self.setting.presentation = Some(presentation);
        self
    }

    pub fn control_behavior(&mut self, control_behavior: ConfigControlBehavior) -> &mut Self {
        self.setting.control_behavior = Some(control_behavior);
        self
    }

    pub fn control_numeric(&mut self, numeric: ConfigNumericControl) -> &mut Self {
        self.control_behavior_mut().numeric = Some(numeric);
        self
    }

    pub fn control_numeric_min(&mut self, min: f64) -> &mut Self {
        self.control_numeric_mut().min = Some(min);
        self
    }

    pub fn control_numeric_max(&mut self, max: f64) -> &mut Self {
        self.control_numeric_mut().max = Some(max);
        self
    }

    pub fn control_numeric_step(&mut self, step: f64) -> &mut Self {
        self.control_numeric_mut().step = Some(step);
        self
    }

    pub fn control_numeric_soft_min(&mut self, soft_min: f64) -> &mut Self {
        self.control_numeric_mut().soft_min = Some(soft_min);
        self
    }

    pub fn control_numeric_soft_max(&mut self, soft_max: f64) -> &mut Self {
        self.control_numeric_mut().soft_max = Some(soft_max);
        self
    }

    pub fn control_numeric_unit(&mut self, unit: impl Into<String>) -> &mut Self {
        self.control_numeric_mut().unit = Some(unit.into());
        self
    }

    pub fn control_text_format(&mut self, text_format: ConfigTextFormat) -> &mut Self {
        self.control_behavior_mut().text_format = Some(text_format);
        self
    }

    pub fn control_options_source(&mut self, options_source: ConfigOptionsSource) -> &mut Self {
        self.control_behavior_mut().options_source = Some(options_source);
        self
    }

    pub fn control_options_static(&mut self) -> &mut Self {
        self.control_options_source(ConfigOptionsSource::Static)
    }

    pub fn control_options_runtime_gpus(&mut self) -> &mut Self {
        self.control_options_source(ConfigOptionsSource::RuntimeGpus)
    }

    pub fn control_availability(&mut self, availability: ConfigControlAvailability) -> &mut Self {
        self.control_behavior_mut().availability = Some(availability);
        self
    }

    pub fn control_availability_enabled(&mut self, enabled: bool) -> &mut Self {
        self.control_availability_mut().enabled = enabled;
        self
    }

    pub fn control_availability_source(
        &mut self,
        source: ConfigControlAvailabilitySource,
    ) -> &mut Self {
        self.control_availability_mut().source = source;
        self
    }

    pub fn control_availability_reason(&mut self, reason: impl Into<String>) -> &mut Self {
        self.control_availability_mut().reason = Some(reason.into());
        self
    }

    pub fn control_availability_note(&mut self, note: impl Into<String>) -> &mut Self {
        self.control_availability_mut().note = Some(note.into());
        self
    }

    pub fn control_enable_when(&mut self, condition: ConfigControlCondition) -> &mut Self {
        self.control_behavior_mut().enable_when.push(condition);
        self
    }

    pub fn control_disable_when(&mut self, disable: ConfigConditionalDisable) -> &mut Self {
        self.control_behavior_mut().disable_when.push(disable);
        self
    }

    pub fn control_conflict(&mut self, conflict: ConfigConflictRule) -> &mut Self {
        self.control_behavior_mut().conflicts.push(conflict);
        self
    }

    pub fn control_write_policy(&mut self, policy: ConfigDisabledWritePolicy) -> &mut Self {
        self.control_behavior_mut().write_policy = Some(policy);
        self
    }

    pub fn presentation_label(&mut self, label: impl Into<String>) -> &mut Self {
        self.presentation_mut().label = Some(label.into());
        self
    }

    pub fn presentation_help(&mut self, help: impl Into<String>) -> &mut Self {
        self.presentation_mut().help = Some(help.into());
        self
    }

    pub fn presentation_category(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        summary: impl Into<String>,
        order: u32,
    ) -> &mut Self {
        let presentation = self.presentation_mut();
        presentation.category_id = Some(id.into());
        presentation.category_label = Some(label.into());
        presentation.category_summary = Some(summary.into());
        presentation.category_order = Some(order);
        self
    }

    pub fn presentation_order(&mut self, order: u32) -> &mut Self {
        self.presentation_mut().setting_order = Some(order);
        self
    }

    pub fn presentation_unit(&mut self, unit: impl Into<String>) -> &mut Self {
        self.presentation_mut().unit = Some(unit.into());
        self
    }

    pub fn presentation_placeholder(&mut self, placeholder: impl Into<String>) -> &mut Self {
        self.presentation_mut().placeholder = Some(placeholder.into());
        self
    }

    pub fn presentation_control_hint(&mut self, control_hint: impl Into<String>) -> &mut Self {
        self.presentation_mut().control_hint = Some(control_hint.into());
        self
    }

    pub fn presentation_renderer_id(&mut self, renderer_id: impl Into<String>) -> &mut Self {
        self.presentation_mut().renderer_id = Some(renderer_id.into());
        self
    }

    pub fn alias(&mut self, alias: ConfigPathAlias) -> &mut Self {
        self.setting.alias_policy.mode = ConfigAliasMode::CanonicalWithLegacyAliases;
        self.setting.alias_policy.aliases.push(alias);
        self
    }

    pub fn constraint(&mut self, constraint: ConfigConstraint) -> &mut Self {
        self.setting.constraints.push(constraint);
        self
    }

    pub fn build(self) -> ConfigSettingSchema {
        self.setting
    }

    fn presentation_mut(&mut self) -> &mut ConfigPresentationMetadata {
        self.setting
            .presentation
            .get_or_insert_with(ConfigPresentationMetadata::default)
    }

    fn control_behavior_mut(&mut self) -> &mut ConfigControlBehavior {
        self.setting
            .control_behavior
            .get_or_insert_with(ConfigControlBehavior::default)
    }

    fn control_numeric_mut(&mut self) -> &mut ConfigNumericControl {
        self.control_behavior_mut()
            .numeric
            .get_or_insert_with(ConfigNumericControl::default)
    }

    fn control_availability_mut(&mut self) -> &mut ConfigControlAvailability {
        self.control_behavior_mut()
            .availability
            .get_or_insert(ConfigControlAvailability {
                enabled: true,
                reason: None,
                note: None,
                source: ConfigControlAvailabilitySource::Static,
            })
    }
}

#[derive(Clone, Debug)]
pub struct ConfigEditor {
    config: MeshConfig,
}

impl ConfigEditor {
    pub fn new(config: MeshConfig) -> Self {
        Self { config }
    }

    pub fn into_config(self) -> MeshConfig {
        self.config
    }

    pub fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub fn set_version(&mut self, version: Option<u32>) -> &mut Self {
        self.config.version = version;
        self
    }

    pub fn set_gpu_assignment(&mut self, assignment: GpuAssignment) -> &mut Self {
        self.config.gpu.assignment = assignment;
        self
    }

    pub fn set_gpu_parallel(&mut self, parallel: Option<usize>) -> &mut Self {
        self.config.gpu.parallel = parallel;
        self
    }

    pub fn set_owner_control_bind(&mut self, bind: Option<SocketAddr>) -> &mut Self {
        self.config.owner_control.bind = bind;
        self
    }

    pub fn set_owner_control_advertise_addr(
        &mut self,
        advertise_addr: Option<SocketAddr>,
    ) -> &mut Self {
        self.config.owner_control.advertise_addr = advertise_addr;
        self
    }

    pub fn defaults(&mut self) -> ModelDefaultsEditor<'_> {
        ModelDefaultsEditor {
            defaults: self.config.defaults.get_or_insert_with(Default::default),
        }
    }

    pub fn set_default_runtime(&mut self, runtime: ModelRuntimeKind) -> &mut Self {
        self.defaults().runtime(runtime);
        self
    }

    pub fn clear_default_runtime(&mut self) -> &mut Self {
        self.defaults().clear_runtime();
        self
    }

    pub fn set_default_device(&mut self, device: impl Into<String>) -> &mut Self {
        self.defaults().device(device);
        self
    }

    pub fn clear_default_device(&mut self) -> &mut Self {
        self.defaults().clear_device();
        self
    }

    pub fn set_default_context_size(&mut self, context_size: Option<u32>) -> &mut Self {
        self.defaults().context_size(context_size);
        self
    }

    pub fn configure_local_serving_node(
        &mut self,
        node: LocalServingNodeConfig,
    ) -> Result<&mut Self> {
        self.set_version(Some(1));
        if let Some(assignment) = node.gpu_assignment {
            self.set_gpu_assignment(assignment);
        }
        if node.owner_control_bind.is_some() {
            self.set_owner_control_bind(node.owner_control_bind);
        }
        if node.owner_control_advertise_addr.is_some() {
            self.set_owner_control_advertise_addr(node.owner_control_advertise_addr);
        }
        let mut model = self.upsert_model(node.model, None)?;
        if let Some(runtime) = node.runtime {
            model.runtime(runtime);
        }
        if let Some(device) = node.device {
            model.device(device);
        }
        if let Some(context_size) = node.context_size {
            model.context_size(context_size);
        }
        if let Some(parallel) = node.parallel {
            model.parallel(parallel);
        }
        if let Some(mmproj) = node.mmproj {
            model.mmproj(mmproj);
        }
        Ok(self)
    }

    pub fn upsert_model(
        &mut self,
        model_ref: impl AsRef<str>,
        profile: Option<String>,
    ) -> Result<ModelConfigEditor<'_>> {
        let model_ref = normalize_non_empty(model_ref.as_ref(), "model ref")?;
        let index = match self
            .config
            .models
            .iter()
            .position(|entry| entry.model == model_ref && entry.profile == profile)
        {
            Some(index) => index,
            None => {
                self.config.models.push(ModelConfigEntry {
                    model: model_ref,
                    profile,
                    ..ModelConfigEntry::default()
                });
                self.config.models.len() - 1
            }
        };
        Ok(ModelConfigEditor {
            model: &mut self.config.models[index],
        })
    }

    pub fn remove_model(
        &mut self,
        model_ref: impl AsRef<str>,
        profile: Option<String>,
    ) -> Result<&mut Self> {
        let model_ref = normalize_non_empty(model_ref.as_ref(), "model ref")?;
        self.config
            .models
            .retain(|entry| !(entry.model == model_ref && entry.profile == profile));
        Ok(self)
    }

    pub fn model_refs(&self) -> Vec<String> {
        self.config
            .models
            .iter()
            .map(|entry| entry.model.clone())
            .collect()
    }

    pub fn upsert_plugin(&mut self, name: impl AsRef<str>) -> Result<PluginConfigEditor<'_>> {
        let name = normalize_non_empty(name.as_ref(), "plugin name")?;
        let index = match self
            .config
            .plugins
            .iter()
            .position(|entry| entry.name == name)
        {
            Some(index) => index,
            None => {
                self.config.plugins.push(PluginConfigEntry {
                    name,
                    enabled: None,
                    command: None,
                    args: Vec::new(),
                    url: None,
                    settings: Default::default(),
                    startup: Default::default(),
                });
                self.config.plugins.len() - 1
            }
        };
        Ok(PluginConfigEditor {
            plugin: &mut self.config.plugins[index],
        })
    }

    pub fn enable_builtin_plugin(&mut self, name: impl AsRef<str>) -> Result<&mut Self> {
        self.upsert_plugin(name)?.enabled(true);
        Ok(self)
    }

    pub fn disable_plugin(&mut self, name: impl AsRef<str>) -> Result<&mut Self> {
        self.upsert_plugin(name)?.enabled(false);
        Ok(self)
    }

    pub fn upsert_external_plugin(
        &mut self,
        name: impl AsRef<str>,
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<&mut Self> {
        self.upsert_plugin(name)?
            .enabled(true)
            .command(command)
            .args(args);
        Ok(self)
    }
}

impl From<MeshConfig> for ConfigEditor {
    fn from(config: MeshConfig) -> Self {
        Self::new(config)
    }
}

pub struct ModelDefaultsEditor<'a> {
    defaults: &'a mut ModelConfigDefaults,
}

impl ModelDefaultsEditor<'_> {
    pub fn runtime(&mut self, runtime: ModelRuntimeKind) -> &mut Self {
        self.hardware().model_runtime = Some(runtime);
        self
    }

    pub fn clear_runtime(&mut self) -> &mut Self {
        self.hardware().model_runtime = None;
        self
    }

    pub fn device(&mut self, device: impl Into<String>) -> &mut Self {
        self.hardware().device = Some(device.into());
        self
    }

    pub fn clear_device(&mut self) -> &mut Self {
        self.hardware().device = None;
        self
    }

    pub fn context_size(&mut self, context_size: Option<u32>) -> &mut Self {
        self.model_fit().ctx_size = context_size;
        self
    }

    pub fn parallel(&mut self, parallel: Option<usize>) -> &mut Self {
        self.throughput().parallel = parallel;
        self
    }

    fn hardware(&mut self) -> &mut HardwareConfig {
        self.defaults.hardware.get_or_insert_with(Default::default)
    }

    fn model_fit(&mut self) -> &mut ModelFitConfig {
        self.defaults.model_fit.get_or_insert_with(Default::default)
    }

    fn throughput(&mut self) -> &mut ThroughputConfig {
        self.defaults
            .throughput
            .get_or_insert_with(Default::default)
    }
}

pub struct ModelConfigEditor<'a> {
    model: &'a mut ModelConfigEntry,
}

impl ModelConfigEditor<'_> {
    pub fn model_ref(&self) -> &str {
        &self.model.model
    }

    pub fn runtime(&mut self, runtime: ModelRuntimeKind) -> &mut Self {
        self.hardware().model_runtime = Some(runtime);
        self
    }

    pub fn clear_runtime(&mut self) -> &mut Self {
        self.hardware().model_runtime = None;
        self
    }

    pub fn device(&mut self, device: impl Into<String>) -> &mut Self {
        self.hardware().device = Some(device.into());
        self
    }

    pub fn clear_device(&mut self) -> &mut Self {
        self.hardware().device = None;
        self
    }

    pub fn context_size(&mut self, context_size: u32) -> &mut Self {
        self.model_fit().ctx_size = Some(context_size);
        self
    }

    pub fn parallel(&mut self, parallel: usize) -> &mut Self {
        self.throughput().parallel = Some(parallel);
        self
    }

    pub fn cache_types(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        let model_fit = self.model_fit();
        model_fit.cache_type_k = Some(key.into());
        model_fit.cache_type_v = Some(value.into());
        self
    }

    pub fn max_tokens(&mut self, max_tokens: u32) -> &mut Self {
        self.request_defaults().max_tokens = Some(max_tokens);
        self
    }

    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        self.request_defaults().temperature = Some(temperature);
        self
    }

    pub fn mmproj(&mut self, mmproj: impl Into<String>) -> &mut Self {
        self.multimodal().mmproj = Some(mmproj.into());
        self
    }

    fn hardware(&mut self) -> &mut HardwareConfig {
        self.model.hardware.get_or_insert_with(Default::default)
    }

    fn model_fit(&mut self) -> &mut ModelFitConfig {
        self.model.model_fit.get_or_insert_with(Default::default)
    }

    fn throughput(&mut self) -> &mut ThroughputConfig {
        self.model.throughput.get_or_insert_with(Default::default)
    }

    fn request_defaults(&mut self) -> &mut RequestDefaultsConfig {
        self.model
            .request_defaults
            .get_or_insert_with(Default::default)
    }

    fn multimodal(&mut self) -> &mut MultimodalConfig {
        self.model.multimodal.get_or_insert_with(Default::default)
    }
}

pub struct PluginConfigEditor<'a> {
    plugin: &'a mut PluginConfigEntry,
}

impl PluginConfigEditor<'_> {
    pub fn name(&self) -> &str {
        &self.plugin.name
    }

    pub fn enabled(&mut self, enabled: bool) -> &mut Self {
        self.plugin.enabled = Some(enabled);
        self
    }

    pub fn command(&mut self, command: impl Into<String>) -> &mut Self {
        self.plugin.command = Some(command.into());
        self
    }

    pub fn args(&mut self, args: impl IntoIterator<Item = impl Into<String>>) -> &mut Self {
        self.plugin.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn url(&mut self, url: impl Into<String>) -> &mut Self {
        self.plugin.url = Some(url.into());
        self
    }

    pub fn connect_timeout_secs(&mut self, seconds: u64) -> &mut Self {
        self.plugin.startup.connect_timeout_secs = Some(seconds);
        self
    }

    pub fn init_timeout_secs(&mut self, seconds: u64) -> &mut Self {
        self.plugin.startup.init_timeout_secs = Some(seconds);
        self
    }

    pub fn optional(&mut self, optional: bool) -> &mut Self {
        self.plugin.startup.optional = optional;
        self
    }

    pub fn lazy_start(&mut self, lazy_start: bool) -> &mut Self {
        self.plugin.startup.lazy_start = lazy_start;
        self
    }
}

fn normalize_non_empty(value: &str, label: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} cannot be empty");
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    use crate::{
        ConfigAliasPolicy, ConfigConditionOperator, ConfigConditionValue, ConfigPathAliasKind,
        ConfigVisibility, config_to_toml, parse_config_toml,
    };
    use toml::Value;

    #[test]
    fn schema_setting_builder_populates_control_surface_metadata() {
        let mut setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["owner_control", "bind"]),
            ConfigValueSchema::SocketAddr,
        );
        setting
            .owner(ConfigSettingOwner::BuiltIn)
            .support(ConfigSupportState::Supported)
            .control_surface(ConfigControlSurface::ConfigFile)
            .control_surface(ConfigControlSurface::OwnerControl)
            .apply_mode(ConfigApplyMode::DynamicApply)
            .restart_scope(ConfigRestartScope::ProcessRestart)
            .visibility(ConfigVisibility::Advanced)
            .description("Owner control listener bind address")
            .constraint(ConfigConstraint::NonEmpty)
            .alias(ConfigPathAlias {
                path: ConfigPath::from_fields(["owner_control", "listen"]),
                kind: ConfigPathAliasKind::LegacyKey,
                note: Some("legacy naming preserved for diagnostics".into()),
            });

        let built = setting.build();

        assert_eq!(built.path.render(), "owner_control.bind");
        assert_eq!(
            built.alias_policy.mode,
            ConfigAliasMode::CanonicalWithLegacyAliases
        );
        assert_eq!(built.alias_policy.aliases.len(), 1);
        assert_eq!(built.control_surfaces.len(), 2);
        assert_eq!(built.apply_mode, ConfigApplyMode::DynamicApply);
        assert_eq!(built.restart_scope, ConfigRestartScope::ProcessRestart);
        assert_eq!(built.visibility, ConfigVisibility::Advanced);
    }

    #[test]
    fn schema_builder_collects_settings() {
        let mut schema = ConfigSchemaBuilder::new();
        let mut setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["telemetry", "endpoint"]),
            ConfigValueSchema::String,
        );
        setting
            .owner(ConfigSettingOwner::BuiltIn)
            .control_surface(ConfigControlSurface::ConfigFile);
        schema.setting(setting.build());

        let built = schema.build();

        assert_eq!(built.settings.len(), 1);
        assert_eq!(built.settings[0].path.render(), "telemetry.endpoint");
    }

    #[test]
    fn schema_setting_builder_control_behavior_matches_hand_constructed_json() {
        let enable_condition = ConfigControlCondition {
            path: ConfigPath::from_fields(["gpu", "assignment"]),
            operator: ConfigConditionOperator::Equals,
            values: vec![ConfigConditionValue::String("pinned".to_string())],
        };
        let disable_condition = ConfigConditionalDisable {
            condition: ConfigControlCondition {
                path: ConfigPath::from_fields(["owner_control", "bind"]),
                operator: ConfigConditionOperator::Absent,
                values: Vec::new(),
            },
            reason: "Owner control bind is required".to_string(),
            note: Some("Preserve the existing value until bind is configured".to_string()),
            write_policy: ConfigDisabledWritePolicy::OmitWhenDisabled,
        };
        let conflict = ConfigConflictRule {
            group: "gpu-selection".to_string(),
            condition: ConfigControlCondition {
                path: ConfigPath::from_fields(["defaults", "hardware", "gpu_id"]),
                operator: ConfigConditionOperator::Present,
                values: Vec::new(),
            },
            reason: "Choose either a runtime GPU selector or a pinned GPU id".to_string(),
            preferred_path: Some(ConfigPath::from_fields(["gpu", "assignment"])),
        };
        let expected_behavior = ConfigControlBehavior {
            numeric: Some(ConfigNumericControl {
                min: Some(1.0),
                max: Some(8.0),
                step: Some(1.0),
                soft_min: Some(1.0),
                soft_max: Some(4.0),
                unit: Some("gpus".to_string()),
            }),
            text_format: Some(ConfigTextFormat::Path),
            options_source: Some(ConfigOptionsSource::RuntimeGpus),
            availability: Some(ConfigControlAvailability {
                enabled: false,
                reason: Some("GPU inventory is unavailable".to_string()),
                note: Some(
                    "The current value is preserved until runtime inventory returns".to_string(),
                ),
                source: ConfigControlAvailabilitySource::Runtime,
            }),
            enable_when: vec![enable_condition.clone()],
            disable_when: vec![disable_condition.clone()],
            conflicts: vec![conflict.clone()],
            write_policy: Some(ConfigDisabledWritePolicy::RejectWhenDisabled),
        };
        let hand_constructed = ConfigSettingSchema {
            path: ConfigPath::from_fields(["gpu", "parallel"]),
            alias_policy: ConfigAliasPolicy::default(),
            owner: ConfigSettingOwner::BuiltIn,
            value_schema: ConfigValueSchema::Integer,
            support: ConfigSupportState::Supported,
            control_surfaces: Vec::new(),
            apply_mode: ConfigApplyMode::StaticOnLoad,
            restart_scope: ConfigRestartScope::None,
            visibility: ConfigVisibility::User,
            constraints: Vec::new(),
            description: None,
            presentation: None,
            control_behavior: Some(expected_behavior.clone()),
        };
        let mut builder = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["gpu", "parallel"]),
            ConfigValueSchema::Integer,
        );
        builder
            .control_numeric_min(1.0)
            .control_numeric_max(8.0)
            .control_numeric_step(1.0)
            .control_numeric_soft_min(1.0)
            .control_numeric_soft_max(4.0)
            .control_numeric_unit("gpus")
            .control_text_format(ConfigTextFormat::Path)
            .control_options_runtime_gpus()
            .control_availability_enabled(false)
            .control_availability_source(ConfigControlAvailabilitySource::Runtime)
            .control_availability_reason("GPU inventory is unavailable")
            .control_availability_note(
                "The current value is preserved until runtime inventory returns",
            )
            .control_enable_when(enable_condition)
            .control_disable_when(disable_condition)
            .control_conflict(conflict)
            .control_write_policy(ConfigDisabledWritePolicy::RejectWhenDisabled);

        let built = builder.build();

        assert_eq!(built.control_behavior, Some(expected_behavior));
        assert_eq!(
            Value::try_from(built).expect("built setting should serialize"),
            Value::try_from(hand_constructed).expect("hand-constructed setting should serialize")
        );
    }

    #[test]
    fn schema_setting_builder_runtime_gpu_option_helper_sets_runtime_source() {
        let mut setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "hardware", "device"]),
            ConfigValueSchema::String,
        );
        setting.control_options_runtime_gpus();

        let built = setting.build();

        assert_eq!(
            built
                .control_behavior
                .and_then(|behavior| behavior.options_source),
            Some(ConfigOptionsSource::RuntimeGpus)
        );
    }

    #[test]
    fn schema_setting_builder_no_helper_serialization_omits_control_behavior() {
        let setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["telemetry", "endpoint"]),
            ConfigValueSchema::String,
        )
        .build();

        let serialized = Value::try_from(setting).expect("setting should serialize");
        let table = serialized
            .as_table()
            .expect("setting should serialize to a table");

        assert!(!table.contains_key("control_behavior"));
    }

    #[test]
    fn schema_setting_builder_direct_control_behavior_can_be_extended_deterministically() {
        let mut setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["defaults", "request_defaults", "temperature"]),
            ConfigValueSchema::Float,
        );
        setting
            .control_behavior(ConfigControlBehavior {
                numeric: None,
                text_format: Some(ConfigTextFormat::Plain),
                options_source: None,
                availability: None,
                enable_when: Vec::new(),
                disable_when: Vec::new(),
                conflicts: Vec::new(),
                write_policy: None,
            })
            .control_numeric(ConfigNumericControl {
                min: Some(0.0),
                max: Some(2.0),
                step: Some(0.1),
                soft_min: None,
                soft_max: None,
                unit: None,
            })
            .control_options_static();

        let built = setting.build();
        let behavior = built
            .control_behavior
            .expect("control behavior should be present");

        assert_eq!(behavior.text_format, Some(ConfigTextFormat::Plain));
        assert_eq!(
            behavior.numeric,
            Some(ConfigNumericControl {
                min: Some(0.0),
                max: Some(2.0),
                step: Some(0.1),
                soft_min: None,
                soft_max: None,
                unit: None,
            })
        );
        assert_eq!(behavior.options_source, Some(ConfigOptionsSource::Static));
    }

    #[test]
    fn schema_setting_builder_static_availability_and_dependency_disable_are_deterministic() {
        let dependency_disable = ConfigConditionalDisable {
            condition: ConfigControlCondition {
                path: ConfigPath::from_fields(["owner_control", "bind"]),
                operator: ConfigConditionOperator::Absent,
                values: Vec::new(),
            },
            reason: "Owner control bind is required".to_string(),
            note: None,
            write_policy: ConfigDisabledWritePolicy::OmitWhenDisabled,
        };
        let mut setting = ConfigSettingSchemaBuilder::new(
            ConfigPath::from_fields(["owner_control", "advertise_addr"]),
            ConfigValueSchema::SocketAddr,
        );
        setting
            .control_availability_enabled(false)
            .control_availability_source(ConfigControlAvailabilitySource::Static)
            .control_availability_reason("Owner control is disabled for this build")
            .control_disable_when(dependency_disable.clone());

        let built = setting.build();
        let behavior = built
            .control_behavior
            .as_ref()
            .expect("control behavior should be present");
        let availability = behavior
            .availability
            .as_ref()
            .expect("availability metadata should be present");

        assert!(!availability.enabled);
        assert_eq!(availability.source, ConfigControlAvailabilitySource::Static);
        assert_eq!(
            built.default_disabled_write_policy(Some(availability.source)),
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(behavior.disable_when, vec![dependency_disable]);
        assert_eq!(
            behavior.disable_when[0].write_policy,
            ConfigDisabledWritePolicy::OmitWhenDisabled
        );
    }

    #[test]
    fn model_config_entry_roundtrips_with_profile() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M", Some("low-ctx".into()))
            .unwrap()
            .context_size(4096);
        editor
            .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M", Some("high-ctx".into()))
            .unwrap()
            .context_size(16384);

        let config = editor.into_config();
        let serialized = config_to_toml(&config).expect("should serialize");
        let deserialized = parse_config_toml(&serialized).expect("should deserialize");

        assert_eq!(deserialized.models.len(), 2);
        let low_ctx = deserialized
            .models
            .iter()
            .find(|e| e.profile.as_deref() == Some("low-ctx"))
            .expect("should have low-ctx entry");
        let high_ctx = deserialized
            .models
            .iter()
            .find(|e| e.profile.as_deref() == Some("high-ctx"))
            .expect("should have high-ctx entry");

        assert_eq!(low_ctx.model, "Qwen/Qwen3-8B-GGUF:Q4_K_M");
        assert_eq!(low_ctx.model_fit.as_ref().unwrap().ctx_size, Some(4096));
        assert_eq!(high_ctx.model, "Qwen/Qwen3-8B-GGUF:Q4_K_M");
        assert_eq!(high_ctx.model_fit.as_ref().unwrap().ctx_size, Some(16384));
    }

    #[test]
    fn model_config_entry_without_profile_omits_profile_key() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen/Qwen3-8B-GGUF:Q4_K_M", None)
            .unwrap()
            .context_size(8192);

        let config = editor.into_config();
        let serialized = config_to_toml(&config).expect("should serialize");
        let toml_str = serialized.to_string();

        assert!(!toml_str.contains("profile"));
        let deserialized = parse_config_toml(&serialized).expect("should deserialize");
        assert_eq!(deserialized.models.len(), 1);
        assert_eq!(deserialized.models[0].model, "Qwen/Qwen3-8B-GGUF:Q4_K_M");
        assert_eq!(deserialized.models[0].profile, None);
    }

    #[test]
    fn upsert_model_dedup_by_profile() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap()
            .context_size(4096);
        editor
            .upsert_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap()
            .context_size(8192);

        let config = editor.into_config();
        assert_eq!(config.models.len(), 1);
        assert_eq!(
            config.models[0].model_fit.as_ref().unwrap().ctx_size,
            Some(8192)
        );
    }

    #[test]
    fn upsert_model_coexists_with_different_profiles() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap();
        editor
            .upsert_model("Qwen3-8B", Some("high-ctx".into()))
            .unwrap();

        let config = editor.into_config();
        assert_eq!(config.models.len(), 2);
    }

    #[test]
    fn upsert_model_none_profile_is_distinct() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap();
        editor.upsert_model("Qwen3-8B", None).unwrap();

        let config = editor.into_config();
        assert_eq!(config.models.len(), 2);
    }

    #[test]
    fn remove_model_by_profile() {
        let mut editor = ConfigEditor::new(MeshConfig::default());
        editor
            .upsert_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap();
        editor
            .upsert_model("Qwen3-8B", Some("high-ctx".into()))
            .unwrap();
        editor.upsert_model("Qwen3-8B", None).unwrap();

        editor
            .remove_model("Qwen3-8B", Some("low-ctx".into()))
            .unwrap();

        let config = editor.into_config();
        assert_eq!(config.models.len(), 2);
        assert!(
            config
                .models
                .iter()
                .any(|e| e.profile.as_deref() == Some("high-ctx"))
        );
        assert!(config.models.iter().any(|e| e.profile.is_none()));
    }

    #[test]
    fn backwards_compat_parse_model_without_profile_field() {
        let toml_str = r#"
version = 1

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
runtime = "metal"

[models.model_fit]
ctx_size = 8192
"#;

        let config = parse_config_toml(toml_str).expect("should parse");
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].model, "Qwen/Qwen3-8B-GGUF:Q4_K_M");
        assert_eq!(config.models[0].profile, None);
        assert_eq!(
            config.models[0].model_fit.as_ref().unwrap().ctx_size,
            Some(8192)
        );

        let serialized = config_to_toml(&config).expect("should serialize");
        let deserialized = parse_config_toml(&serialized).expect("should re-parse");
        assert_eq!(deserialized.models.len(), 1);
        assert_eq!(deserialized.models[0].profile, None);
    }
}
