use crate::{
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
        let mut model = self.upsert_model(node.model)?;
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

    pub fn upsert_model(&mut self, model_ref: impl AsRef<str>) -> Result<ModelConfigEditor<'_>> {
        let model_ref = normalize_non_empty(model_ref.as_ref(), "model ref")?;
        let index = match self
            .config
            .models
            .iter()
            .position(|entry| entry.model == model_ref)
        {
            Some(index) => index,
            None => {
                self.config.models.push(ModelConfigEntry {
                    model: model_ref,
                    ..ModelConfigEntry::default()
                });
                self.config.models.len() - 1
            }
        };
        Ok(ModelConfigEditor {
            model: &mut self.config.models[index],
        })
    }

    pub fn remove_model(&mut self, model_ref: impl AsRef<str>) -> Result<&mut Self> {
        let model_ref = normalize_non_empty(model_ref.as_ref(), "model ref")?;
        self.config.models.retain(|entry| entry.model != model_ref);
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
