pub mod blobstore;
mod config;
pub(crate) mod mcp;
mod runtime;
pub(crate) mod stapler;
mod support;
mod transport;

use anyhow::{anyhow, bail, Context, Result};
pub use mesh_llm_plugin::proto;
use rmcp::model::ServerInfo;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use url::Url;

#[allow(unused_imports)]
pub use self::config::ExternalPluginSpec;
pub use self::config::{
    config_path, load_config, resolve_plugins, PluginHostMode, ResolvedPlugins,
};
use self::runtime::ExternalPlugin;
pub(crate) use self::support::parse_optional_json;
use self::support::{format_args_for_log, format_slice_for_log, format_tool_names_for_log};
#[cfg(all(test, unix))]
use self::transport::unix_socket_path;
use self::transport::{connect_side_stream, make_instance_id, LocalStream};
#[cfg(test)]
use mesh_llm_plugin::MeshVisibility;

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
pub const BLOBSTORE_PLUGIN_ID: &str = "blobstore";
pub const LEMONADE_PLUGIN_ID: &str = "lemonade";
pub const BLACKBOARD_CAPABILITY: &str = "blackboard.v1";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;
const ENDPOINT_STARTUP_GRACE_SECS: u64 = 30;
const ENDPOINT_FAILURE_THRESHOLD: u32 = 2;

#[derive(Clone, Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
#[cfg(test)]
type TestStreamFuture = Pin<Box<dyn Future<Output = Result<LocalStream>> + Send>>;
#[cfg(test)]
type TestStreamHandler = Arc<dyn Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PluginManifestOverview>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginManifestOverview {
    pub mcp_tools: usize,
    pub mcp_resources: usize,
    pub mcp_resource_templates: usize,
    pub mcp_prompts: usize,
    pub mcp_completions: usize,
    pub http_bindings: usize,
    pub endpoints: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginEndpointSummary {
    pub plugin_name: String,
    pub plugin_status: String,
    pub endpoint_id: String,
    pub state: String,
    pub available: bool,
    pub kind: String,
    pub transport_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub supports_streaming: bool,
    pub managed_by_plugin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginCapabilityProvider {
    pub capability: String,
    pub plugin_name: String,
    pub plugin_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct EndpointHealthRecord {
    state: String,
    available: bool,
    detail: Option<String>,
    models: Vec<String>,
}

#[derive(Clone, Debug)]
struct EndpointHealthState {
    record: EndpointHealthRecord,
    first_checked_at: Instant,
    consecutive_failures: u32,
}

#[derive(Clone, Debug)]
pub struct InferenceEndpointRoute {
    pub plugin_name: String,
    pub endpoint_id: String,
    pub address: String,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedInferenceEndpoint {
    pub plugin_name: String,
    pub endpoint_id: String,
    pub protocol: Option<String>,
    pub address: Option<String>,
    pub supports_streaming: bool,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
    inactive: BTreeMap<String, PluginSummary>,
    endpoint_health: Arc<Mutex<BTreeMap<String, EndpointHealthState>>>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    #[cfg(test)]
    bridged_plugins: BTreeSet<String>,
    #[cfg(test)]
    test_endpoints: Arc<Mutex<Vec<PluginEndpointSummary>>>,
    #[cfg(test)]
    test_inference_endpoints: Arc<Mutex<Vec<InferenceEndpointRoute>>>,
    #[cfg(test)]
    test_manifests: Arc<Mutex<BTreeMap<String, proto::PluginManifest>>>,
    #[cfg(test)]
    test_stream_handlers: Arc<Mutex<BTreeMap<String, TestStreamHandler>>>,
}

impl PluginManager {
    pub async fn start(
        specs: &ResolvedPlugins,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
    ) -> Result<Self> {
        if specs.externals.is_empty() {
            tracing::info!("Plugin manager: no plugins enabled");
        } else {
            let names = specs
                .externals
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                "Plugin manager: loading {} plugin(s): {}",
                specs.externals.len(),
                names
            );
        }

        let rpc_bridge = Arc::new(Mutex::new(None));
        let instance_id = make_instance_id();
        let mut plugins = BTreeMap::new();
        for spec in &specs.externals {
            tracing::info!(
                plugin = %spec.name,
                command = %spec.command,
                args = %format_args_for_log(&spec.args),
                "Loading plugin"
            );
            let plugin = match ExternalPlugin::spawn(
                spec,
                instance_id.clone(),
                host_mode,
                mesh_tx.clone(),
                rpc_bridge.clone(),
            )
            .await
            {
                Ok(plugin) => plugin,
                Err(err) => {
                    tracing::error!(
                        plugin = %spec.name,
                        error = %err,
                        "Plugin failed to load"
                    );
                    return Err(err);
                }
            };
            let summary = plugin.summary().await;
            tracing::info!(
                plugin = %summary.name,
                version = %summary.version.as_deref().unwrap_or("unknown"),
                capabilities = %format_slice_for_log(&summary.capabilities),
                tools = %format_tool_names_for_log(&summary.tools),
                "Plugin loaded successfully"
            );
            plugins.insert(spec.name.clone(), plugin);
        }
        let manager = Self {
            inner: Arc::new(PluginManagerInner {
                plugins,
                inactive: specs
                    .inactive
                    .iter()
                    .cloned()
                    .map(|summary| (summary.name.clone(), summary))
                    .collect(),
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                rpc_bridge,
                #[cfg(test)]
                bridged_plugins: BTreeSet::new(),
                #[cfg(test)]
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                #[cfg(test)]
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        };
        manager.start_supervisor();
        Ok(manager)
    }

    #[cfg(test)]
    pub fn for_test_bridge(plugin_names: &[&str], bridge: Arc<dyn PluginRpcBridge>) -> Self {
        Self {
            inner: Arc::new(PluginManagerInner {
                plugins: BTreeMap::new(),
                inactive: BTreeMap::new(),
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                rpc_bridge: Arc::new(Mutex::new(Some(bridge))),
                bridged_plugins: plugin_names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let manifests = self.inner.test_manifests.lock().await.clone();
            if !manifests.is_empty() {
                let mut summaries = manifests
                    .into_iter()
                    .map(|(name, manifest)| PluginSummary {
                        name,
                        kind: "bridge".into(),
                        enabled: true,
                        status: "running".into(),
                        version: None,
                        capabilities: manifest.capabilities.clone(),
                        command: None,
                        args: Vec::new(),
                        tools: Vec::new(),
                        manifest: Some(plugin_manifest_overview(&manifest)),
                        error: None,
                    })
                    .collect::<Vec<_>>();
                summaries.sort_by(|a, b| a.name.cmp(&b.name));
                return summaries;
            }
        }
        let mut summaries =
            Vec::with_capacity(self.inner.plugins.len() + self.inner.inactive.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary().await);
        }
        summaries.extend(self.inner.inactive.values().cloned());
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn endpoints(&self) -> Result<Vec<PluginEndpointSummary>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut endpoints = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };
            for endpoint in manifest.endpoints {
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                endpoints.push(PluginEndpointSummary {
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: endpoint.endpoint_id,
                    state: health.state,
                    available: health.available,
                    kind: endpoint_kind_name(endpoint.kind).to_string(),
                    transport_kind: endpoint_transport_kind_name(endpoint.transport_kind)
                        .to_string(),
                    protocol: endpoint.protocol,
                    address: endpoint.address,
                    args: endpoint.args,
                    namespace: endpoint.namespace,
                    supports_streaming: endpoint.supports_streaming,
                    managed_by_plugin: endpoint.managed_by_plugin,
                    detail: health.detail,
                    models: health.models,
                });
            }
        }
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints)
    }

    #[cfg(test)]
    pub async fn set_test_endpoints(&self, endpoints: Vec<PluginEndpointSummary>) {
        *self.inner.test_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub async fn set_test_inference_endpoints(&self, endpoints: Vec<InferenceEndpointRoute>) {
        *self.inner.test_inference_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub async fn set_test_manifests(&self, manifests: BTreeMap<String, proto::PluginManifest>) {
        *self.inner.test_manifests.lock().await = manifests;
    }

    pub async fn tools(&self, name: &str) -> Result<Vec<ToolSummary>> {
        if let Some(summary) = self.inner.inactive.get(name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(name)
            .with_context(|| format!("Unknown plugin '{name}'"))?;
        plugin.list_tools().await
    }

    pub async fn call_tool(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let arguments = parse_optional_json(arguments_json)?;
            let params_json = serde_json::to_string(&serde_json::json!({
                "name": tool_name,
                "arguments": arguments,
            }))
            .with_context(|| format!("Serialize tool call for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), "tools/call".into(), params_json)
                .await
                .map_err(|err| anyhow!("{}", err.message))?;
            let decoded: rmcp::model::CallToolResult = serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode tool result from test plugin '{plugin_name}'"))?;
            return Ok(ToolCallResult {
                content_json: normalize_test_tool_result_content(&decoded)?,
                is_error: decoded.is_error.unwrap_or(false),
            });
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.call_tool(tool_name, arguments_json).await
    }

    pub async fn inference_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for endpoint in self.inference_endpoints().await? {
            models.extend(endpoint.models);
        }
        models.sort();
        models.dedup();
        Ok(models)
    }

    pub async fn inference_endpoint_for_model(
        &self,
        model: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let mut endpoints = self.inference_endpoints().await?;
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints
            .into_iter()
            .find(|endpoint| endpoint.models.iter().any(|candidate| candidate == model)))
    }

    pub async fn managed_inference_endpoints(&self) -> Result<Vec<ManagedInferenceEndpoint>> {
        let summaries = self.list().await;
        let mut endpoints = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };
            for endpoint in manifest.endpoints {
                if proto::EndpointKind::try_from(endpoint.kind)
                    .unwrap_or(proto::EndpointKind::Unspecified)
                    != proto::EndpointKind::Inference
                {
                    continue;
                }
                if !endpoint.managed_by_plugin {
                    continue;
                }
                endpoints.push(ManagedInferenceEndpoint {
                    plugin_name: summary.name.clone(),
                    endpoint_id: endpoint.endpoint_id,
                    protocol: endpoint.protocol,
                    address: endpoint.address,
                    supports_streaming: endpoint.supports_streaming,
                });
            }
        }
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints)
    }

    pub async fn ensure_managed_inference_endpoint(
        &self,
        plugin_name: &str,
        endpoint_id: &str,
        model_path: &std::path::Path,
        requested_port: Option<u16>,
        ctx_size_override: Option<u32>,
    ) -> Result<mesh_llm_plugin::EnsureInferenceEndpointResponse> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.bridged_plugins.contains(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .context("No active test RPC bridge")?;
            let params_json =
                serde_json::to_string(&mesh_llm_plugin::EnsureInferenceEndpointRequest {
                    endpoint_id: endpoint_id.to_string(),
                    model_path: model_path.display().to_string(),
                    requested_port,
                    ctx_size_override,
                })?;
            let result = bridge
                .handle_request(
                    plugin_name.to_string(),
                    "inference/ensure_endpoint".into(),
                    params_json,
                )
                .await
                .map_err(|err| {
                    self::support::plugin_error(plugin_name, "inference/ensure_endpoint", &err)
                })?;
            return serde_json::from_str(&result.result_json).with_context(|| {
                format!("Decode ensure endpoint result from test plugin '{plugin_name}'")
            });
        }

        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin
            .mcp_request(
                "inference/ensure_endpoint",
                mesh_llm_plugin::EnsureInferenceEndpointRequest {
                    endpoint_id: endpoint_id.to_string(),
                    model_path: model_path.display().to_string(),
                    requested_port,
                    ctx_size_override,
                },
            )
            .await
    }

    pub async fn ensure_managed_inference_worker(
        &self,
        plugin_name: &str,
        model_path: Option<&std::path::Path>,
        device_hint: Option<&str>,
    ) -> Result<mesh_llm_plugin::EnsureInferenceWorkerResponse> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.bridged_plugins.contains(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .context("No active test RPC bridge")?;
            let params_json =
                serde_json::to_string(&mesh_llm_plugin::EnsureInferenceWorkerRequest {
                    model_path: model_path.map(|path| path.display().to_string()),
                    device_hint: device_hint.map(str::to_string),
                })?;
            let result = bridge
                .handle_request(
                    plugin_name.to_string(),
                    "inference/ensure_worker".into(),
                    params_json,
                )
                .await
                .map_err(|err| {
                    self::support::plugin_error(plugin_name, "inference/ensure_worker", &err)
                })?;
            return serde_json::from_str(&result.result_json).with_context(|| {
                format!("Decode ensure worker result from test plugin '{plugin_name}'")
            });
        }

        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin
            .mcp_request(
                "inference/ensure_worker",
                mesh_llm_plugin::EnsureInferenceWorkerRequest {
                    model_path: model_path.map(|path| path.display().to_string()),
                    device_hint: device_hint.map(str::to_string),
                },
            )
            .await
    }

    pub async fn prepare_managed_moe_shard(
        &self,
        plugin_name: &str,
        model_path: &std::path::Path,
        output_path: &std::path::Path,
        assignment: &crate::inference::moe::NodeAssignment,
    ) -> Result<()> {
        let request = mesh_llm_plugin::PrepareMoeShardRequest {
            model_path: model_path.display().to_string(),
            output_path: output_path.display().to_string(),
            experts: assignment.experts.clone(),
            n_shared: assignment.n_shared as u32,
            n_unique: assignment.n_unique as u32,
        };

        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.bridged_plugins.contains(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .context("No active test RPC bridge")?;
            let params_json = serde_json::to_string(&request)?;
            bridge
                .handle_request(
                    plugin_name.to_string(),
                    "inference/prepare_moe_shard".into(),
                    params_json,
                )
                .await
                .map_err(|err| {
                    self::support::plugin_error(plugin_name, "inference/prepare_moe_shard", &err)
                })?;
            return Ok(());
        }

        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        let _: mesh_llm_plugin::PrepareMoeShardResponse = plugin
            .mcp_request("inference/prepare_moe_shard", request)
            .await?;
        Ok(())
    }

    pub async fn capability_providers(&self) -> Result<Vec<PluginCapabilityProvider>> {
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut providers = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };

            let plugin_default = endpoint_record_from_plugin_status(&summary);
            for capability in &manifest.capabilities {
                providers.push(PluginCapabilityProvider {
                    capability: capability.clone(),
                    plugin_name: summary.name.clone(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: None,
                    available: plugin_default.available,
                    detail: plugin_default.detail.clone(),
                });
            }

            for endpoint in &manifest.endpoints {
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                let endpoint_capabilities = endpoint_declared_capabilities(endpoint);
                for capability in endpoint_capabilities {
                    providers.push(PluginCapabilityProvider {
                        capability,
                        plugin_name: summary.name.clone(),
                        plugin_status: summary.status.clone(),
                        endpoint_id: Some(endpoint.endpoint_id.clone()),
                        available: health.available,
                        detail: health.detail.clone(),
                    });
                }
            }
        }
        providers.sort_by(|a, b| {
            a.capability
                .cmp(&b.capability)
                .then_with(|| a.plugin_name.cmp(&b.plugin_name))
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(providers)
    }

    pub async fn provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        let mut providers = self.capability_providers().await?;
        providers.sort_by(|a, b| {
            b.available
                .cmp(&a.available)
                .then_with(|| a.plugin_name.cmp(&b.plugin_name))
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(providers
            .into_iter()
            .find(|provider| provider.capability == capability))
    }

    pub async fn available_provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        Ok(self
            .provider_for_capability(capability)
            .await?
            .filter(|provider| provider.available))
    }

    pub async fn is_capability_available(&self, capability: &str) -> bool {
        self.available_provider_for_capability(capability)
            .await
            .ok()
            .flatten()
            .is_some()
    }

    pub async fn call_tool_by_capability(
        &self,
        capability: &str,
        tool_name: &str,
        payload_json: &str,
    ) -> Result<ToolCallResult> {
        let provider = self
            .available_provider_for_capability(capability)
            .await?
            .ok_or_else(|| anyhow!("No provider for capability '{capability}'"))?;
        self.call_tool(&provider.plugin_name, tool_name, payload_json)
            .await
    }

    pub async fn mcp_request<T, P>(&self, plugin_name: &str, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), method.to_string(), params_json)
                .await
                .map_err(|err| anyhow!("{}", err.message))?;
            return serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode response from test plugin '{plugin_name}'"));
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_request(method, params).await
    }

    pub async fn mcp_notify<P>(&self, plugin_name: &str, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            bridge
                .handle_notification(plugin_name.to_string(), method.to_string(), params_json)
                .await;
            return Ok(());
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_notify(method, params).await
    }

    fn is_test_bridge_enabled(&self, _plugin_name: &str) -> bool {
        #[cfg(test)]
        {
            return self.inner.bridged_plugins.contains(_plugin_name);
        }
        #[allow(unreachable_code)]
        false
    }

    pub async fn list_server_infos(&self) -> Vec<(String, ServerInfo)> {
        let mut infos = Vec::new();
        for (name, plugin) in &self.inner.plugins {
            if let Ok(info) = plugin.server_info().await {
                infos.push((name.clone(), info));
            }
        }
        infos
    }

    pub async fn manifest(&self, plugin_name: &str) -> Result<Option<proto::PluginManifest>> {
        if self.is_test_bridge_enabled(plugin_name) {
            #[cfg(test)]
            if let Some(manifest) = self
                .inner
                .test_manifests
                .lock()
                .await
                .get(plugin_name)
                .cloned()
            {
                return Ok(Some(manifest));
            }
            bail!(
                "Plugin '{}' does not expose a manifest in bridge mode",
                plugin_name
            );
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.manifest().await
    }

    pub async fn manifest_json(&self, plugin_name: &str) -> Result<Option<Value>> {
        Ok(self
            .manifest(plugin_name)
            .await?
            .as_ref()
            .map(plugin_manifest_to_json))
    }

    pub async fn set_rpc_bridge(&self, bridge: Option<Arc<dyn PluginRpcBridge>>) {
        *self.inner.rpc_bridge.lock().await = bridge;
    }

    #[cfg(test)]
    pub async fn set_test_stream_handler<F>(&self, plugin_name: &str, handler: F)
    where
        F: Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync + 'static,
    {
        self.inner
            .test_stream_handlers
            .lock()
            .await
            .insert(plugin_name.to_string(), Arc::new(handler));
    }

    pub async fn dispatch_channel_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::Channel { plugin_id, message } = event else {
            bail!("expected plugin channel event");
        };
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping channel message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_channel_message(message).await
    }

    pub async fn dispatch_bulk_transfer_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::BulkTransfer { plugin_id, message } = event else {
            bail!("expected plugin bulk transfer event");
        };
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping bulk transfer message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_bulk_transfer_message(message).await
    }

    pub async fn broadcast_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        for plugin in self.inner.plugins.values() {
            plugin.send_mesh_event(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn open_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        if self.is_test_bridge_enabled(plugin_name) {
            bail!(
                "Plugin '{}' does not support stream control in bridge mode",
                plugin_name
            );
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.open_stream(request).await
    }

    pub(crate) async fn connect_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<LocalStream> {
        #[cfg(test)]
        if let Some(handler) = self
            .inner
            .test_stream_handlers
            .lock()
            .await
            .get(plugin_name)
            .cloned()
        {
            return handler(request).await;
        }
        let response = self.open_stream(plugin_name, request).await?;
        if !response.accepted {
            bail!(
                "Plugin '{}' rejected stream request: {}",
                plugin_name,
                response.message.as_deref().unwrap_or("no reason provided")
            );
        }
        let endpoint = response.endpoint.as_deref().with_context(|| {
            format!(
                "Plugin '{}' accepted stream request without an endpoint",
                plugin_name
            )
        })?;
        connect_side_stream(endpoint, response.transport_kind).await
    }

    fn start_supervisor(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS));
            loop {
                ticker.tick().await;
                let plugin_names = manager.inner.plugins.keys().cloned().collect::<Vec<_>>();
                for plugin_name in plugin_names {
                    let Some(plugin) = manager.inner.plugins.get(&plugin_name) else {
                        continue;
                    };
                    if let Err(err) = plugin.supervise().await {
                        tracing::warn!(
                            plugin = %plugin.name(),
                            error = %err,
                            "Plugin supervision round failed"
                        );
                    }
                    if let Err(err) = manager.refresh_plugin_endpoints(&plugin_name).await {
                        tracing::warn!(
                            plugin = %plugin_name,
                            error = %err,
                            "Endpoint supervision round failed"
                        );
                    }
                }
            }
        });
    }

    async fn refresh_plugin_endpoints(&self, plugin_name: &str) -> Result<()> {
        let summary = if let Some(plugin) = self.inner.plugins.get(plugin_name) {
            plugin.summary().await
        } else if let Some(summary) = self.inner.inactive.get(plugin_name) {
            summary.clone()
        } else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };

        let manifest = self.manifest(plugin_name).await.ok().flatten();
        let Some(manifest) = manifest else {
            self.clear_plugin_endpoint_health(plugin_name).await;
            return Ok(());
        };

        let now = Instant::now();
        let prefix = format!("{plugin_name}:");
        let previous = self
            .inner
            .endpoint_health
            .lock()
            .await
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|endpoint_id| (endpoint_id.to_string(), value.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let mut endpoint_states = BTreeMap::new();
        for endpoint in manifest.endpoints {
            let key = endpoint.endpoint_id.clone();
            let health =
                endpoint_health_for_summary(&summary, &endpoint, previous.get(&key), now).await;
            endpoint_states.insert(endpoint_key(plugin_name, &key), health);
        }

        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
        registry.extend(endpoint_states);
        Ok(())
    }

    async fn clear_plugin_endpoint_health(&self, plugin_name: &str) {
        let mut registry = self.inner.endpoint_health.lock().await;
        registry.retain(|key, _| !key.starts_with(&format!("{plugin_name}:")));
    }

    async fn inference_endpoints(&self) -> Result<Vec<InferenceEndpointRoute>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_inference_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        let summaries = self.list().await;
        let endpoint_health = self.inner.endpoint_health.lock().await.clone();
        let mut endpoints = Vec::new();
        for summary in summaries {
            let Ok(Some(manifest)) = self.manifest(&summary.name).await else {
                continue;
            };
            for endpoint in manifest.endpoints {
                if proto::EndpointKind::try_from(endpoint.kind)
                    .unwrap_or(proto::EndpointKind::Unspecified)
                    != proto::EndpointKind::Inference
                {
                    continue;
                }
                let Some(address) = endpoint.address.clone() else {
                    continue;
                };
                let health = endpoint_health
                    .get(&endpoint_key(&summary.name, &endpoint.endpoint_id))
                    .map(|state| state.record.clone())
                    .unwrap_or_else(|| endpoint_record_from_plugin_status(&summary));
                if !health.available {
                    continue;
                }
                endpoints.push(InferenceEndpointRoute {
                    plugin_name: summary.name.clone(),
                    endpoint_id: endpoint.endpoint_id,
                    address,
                    models: health.models,
                });
            }
        }
        Ok(endpoints)
    }
}

#[cfg(test)]
pub(crate) async fn connect_test_side_stream(
    endpoint: &str,
    transport_kind: i32,
) -> Result<LocalStream> {
    connect_side_stream(endpoint, transport_kind).await
}

pub(crate) fn plugin_manifest_overview(manifest: &proto::PluginManifest) -> PluginManifestOverview {
    PluginManifestOverview {
        mcp_tools: manifest.mcp_tools.len(),
        mcp_resources: manifest.mcp_resources.len(),
        mcp_resource_templates: manifest.mcp_resource_templates.len(),
        mcp_prompts: manifest.mcp_prompts.len(),
        mcp_completions: manifest.mcp_completions.len(),
        http_bindings: manifest.http_bindings.len(),
        endpoints: manifest.endpoints.len(),
        capabilities: manifest.capabilities.clone(),
    }
}

pub(crate) fn plugin_manifest_to_json(manifest: &proto::PluginManifest) -> Value {
    json!({
        "mcp_tools": manifest.mcp_tools.iter().map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema_json": tool.input_schema_json,
                "output_schema_json": tool.output_schema_json,
                "title": tool.title,
            })
        }).collect::<Vec<_>>(),
        "mcp_resources": manifest.mcp_resources.iter().map(|resource| {
            json!({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "mcp_resource_templates": manifest.mcp_resource_templates.iter().map(|resource| {
            json!({
                "uri_template": resource.uri_template,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "mcp_prompts": manifest.mcp_prompts.iter().map(|prompt| {
            json!({
                "name": prompt.name,
                "description": prompt.description,
            })
        }).collect::<Vec<_>>(),
        "mcp_completions": manifest.mcp_completions.iter().map(|completion| {
            json!({
                "argument_ref": completion.argument_ref,
                "description": completion.description,
            })
        }).collect::<Vec<_>>(),
        "http_bindings": manifest.http_bindings.iter().map(|binding| {
            json!({
                "binding_id": binding.binding_id,
                "method": http_method_name(binding.method),
                "path": binding.path,
                "operation_name": binding.operation_name,
                "request_body_mode": http_body_mode_name(binding.request_body_mode),
                "response_body_mode": http_body_mode_name(binding.response_body_mode),
                "request_schema_json": binding.request_schema_json,
                "response_schema_json": binding.response_schema_json,
            })
        }).collect::<Vec<_>>(),
        "endpoints": manifest.endpoints.iter().map(|endpoint| {
            json!({
                "endpoint_id": endpoint.endpoint_id,
                "kind": endpoint_kind_name(endpoint.kind),
                "transport_kind": endpoint_transport_kind_name(endpoint.transport_kind),
                "protocol": endpoint.protocol,
                "address": endpoint.address,
                "args": endpoint.args,
                "namespace": endpoint.namespace,
                "supports_streaming": endpoint.supports_streaming,
                "managed_by_plugin": endpoint.managed_by_plugin,
            })
        }).collect::<Vec<_>>(),
        "capabilities": manifest.capabilities,
    })
}

fn http_method_name(value: i32) -> &'static str {
    match proto::HttpMethod::try_from(value).unwrap_or(proto::HttpMethod::Unspecified) {
        proto::HttpMethod::Get => "GET",
        proto::HttpMethod::Post => "POST",
        proto::HttpMethod::Put => "PUT",
        proto::HttpMethod::Patch => "PATCH",
        proto::HttpMethod::Delete => "DELETE",
        proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn http_body_mode_name(value: i32) -> &'static str {
    match proto::HttpBodyMode::try_from(value).unwrap_or(proto::HttpBodyMode::Unspecified) {
        proto::HttpBodyMode::Buffered => "buffered",
        proto::HttpBodyMode::Streamed => "streamed",
        proto::HttpBodyMode::Unspecified => "unspecified",
    }
}

fn endpoint_kind_name(value: i32) -> &'static str {
    match proto::EndpointKind::try_from(value).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => "inference",
        proto::EndpointKind::Mcp => "mcp",
        proto::EndpointKind::Unspecified => "unspecified",
    }
}

fn endpoint_transport_kind_name(value: i32) -> &'static str {
    match proto::EndpointTransportKind::try_from(value)
        .unwrap_or(proto::EndpointTransportKind::Unspecified)
    {
        proto::EndpointTransportKind::EndpointTransportHttp => "http",
        proto::EndpointTransportKind::EndpointTransportUnixSocket => "unix_socket",
        proto::EndpointTransportKind::EndpointTransportStdio => "stdio",
        proto::EndpointTransportKind::EndpointTransportNamedPipe => "named_pipe",
        proto::EndpointTransportKind::EndpointTransportTcp => "tcp",
        proto::EndpointTransportKind::Unspecified => "unspecified",
    }
}

fn endpoint_record_from_plugin_status(summary: &PluginSummary) -> EndpointHealthRecord {
    if !summary.enabled || summary.status == "disabled" {
        return EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        };
    }

    match summary.status.as_str() {
        "running" => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        },
        "starting" | "restarting" => EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        "degraded" => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
        _ => EndpointHealthRecord {
            state: "unavailable".into(),
            available: false,
            detail: summary.error.clone(),
            models: Vec::new(),
        },
    }
}

fn endpoint_state_from_plugin_status(summary: &PluginSummary, now: Instant) -> EndpointHealthState {
    EndpointHealthState {
        record: endpoint_record_from_plugin_status(summary),
        first_checked_at: now,
        consecutive_failures: 0,
    }
}

fn endpoint_key(plugin_name: &str, endpoint_id: &str) -> String {
    format!("{plugin_name}:{endpoint_id}")
}

fn endpoint_declared_capabilities(endpoint: &proto::EndpointManifest) -> Vec<String> {
    match proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => {
            let mut capabilities = vec!["endpoint:inference".into()];
            if let Some(protocol) = endpoint.protocol.as_deref() {
                capabilities.push(format!("endpoint:inference/{protocol}"));
            }
            capabilities
        }
        proto::EndpointKind::Mcp => {
            let mut capabilities = vec!["endpoint:mcp".into()];
            if let Some(namespace) = endpoint.namespace.as_deref() {
                capabilities.push(format!("endpoint:mcp/{namespace}"));
            }
            capabilities
        }
        proto::EndpointKind::Unspecified => Vec::new(),
    }
}

fn normalize_test_tool_result_content(result: &rmcp::model::CallToolResult) -> Result<String> {
    if let Some(value) = &result.structured_content {
        return serde_json::to_string(value).map_err(Into::into);
    }
    if let Some(text) = result.content.first().and_then(|content| content.as_text()) {
        return Ok(text.text.clone());
    }
    serde_json::to_string(&result.content).map_err(Into::into)
}

async fn endpoint_health_for_summary(
    summary: &PluginSummary,
    endpoint: &proto::EndpointManifest,
    previous: Option<&EndpointHealthState>,
    now: Instant,
) -> EndpointHealthState {
    if summary.status != "running" {
        return endpoint_state_from_plugin_status(summary, now);
    }

    let probe = probe_endpoint(endpoint)
        .await
        .unwrap_or(EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: None,
            models: Vec::new(),
        });
    apply_endpoint_probe(previous, probe, now)
}

fn apply_endpoint_probe(
    previous: Option<&EndpointHealthState>,
    probe: EndpointHealthRecord,
    now: Instant,
) -> EndpointHealthState {
    let first_checked_at = previous.map(|state| state.first_checked_at).unwrap_or(now);

    if probe.available {
        return EndpointHealthState {
            record: probe,
            first_checked_at,
            consecutive_failures: 0,
        };
    }

    let failure_streak = previous
        .map(|state| state.consecutive_failures.saturating_add(1))
        .unwrap_or(1);
    let within_startup_grace =
        now.duration_since(first_checked_at) < Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS);
    let was_available = previous
        .map(|state| state.record.available)
        .unwrap_or(false);

    let record = if !was_available && within_startup_grace {
        EndpointHealthRecord {
            state: "starting".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else if was_available && failure_streak < ENDPOINT_FAILURE_THRESHOLD {
        EndpointHealthRecord {
            state: "degraded".into(),
            available: true,
            detail: probe.detail,
            models: Vec::new(),
        }
    } else {
        EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: probe.detail,
            models: Vec::new(),
        }
    };

    EndpointHealthState {
        record,
        first_checked_at,
        consecutive_failures: failure_streak,
    }
}

async fn probe_endpoint(endpoint: &proto::EndpointManifest) -> Option<EndpointHealthRecord> {
    match (
        proto::EndpointKind::try_from(endpoint.kind).unwrap_or(proto::EndpointKind::Unspecified),
        proto::EndpointTransportKind::try_from(endpoint.transport_kind)
            .unwrap_or(proto::EndpointTransportKind::Unspecified),
    ) {
        (proto::EndpointKind::Inference, proto::EndpointTransportKind::EndpointTransportHttp) => {
            let protocol = endpoint.protocol.as_deref().unwrap_or_default();
            if protocol.eq_ignore_ascii_case("openai_compatible") {
                return Some(
                    probe_openai_compatible_http_endpoint(endpoint.address.as_deref()?).await,
                );
            }
            None
        }
        _ => None,
    }
}

async fn probe_openai_compatible_http_endpoint(address: &str) -> EndpointHealthRecord {
    let models_url = match endpoint_models_url(address) {
        Some(url) => url,
        None => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("invalid endpoint address '{address}'")),
                models: Vec::new(),
            };
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            return EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some(format!("build health probe client: {err}")),
                models: Vec::new(),
            };
        }
    };

    match client.get(models_url.clone()).send().await {
        Ok(response) if response.status().is_success() => EndpointHealthRecord {
            state: "healthy".into(),
            available: true,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: parse_models_response(response).await.unwrap_or_default(),
        },
        Ok(response) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} -> {}", models_url, response.status())),
            models: Vec::new(),
        },
        Err(err) => EndpointHealthRecord {
            state: "unhealthy".into(),
            available: false,
            detail: Some(format!("GET {} failed: {}", models_url, err)),
            models: Vec::new(),
        },
    }
}

async fn parse_models_response(response: reqwest::Response) -> Result<Vec<String>> {
    let body = response.json::<Value>().await?;
    let models = body
        .get("data")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("id").and_then(|id| id.as_str()))
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    Ok(models)
}

fn endpoint_models_url(address: &str) -> Option<Url> {
    let mut url = Url::parse(address).ok()?;
    let mut path = url.path().trim_end_matches('/').to_string();
    if path.is_empty() {
        path = "/v1".into();
    }
    if !path.ends_with("/models") {
        if path.ends_with("/v1") || path.ends_with("/api/v1") {
            path.push_str("/models");
        } else {
            path.push_str("/v1/models");
        }
    }
    url.set_path(&path);
    url.set_query(None);
    Some(url)
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => crate::plugins::blackboard::run_plugin(name).await,
        BLOBSTORE_PLUGIN_ID => crate::plugins::blobstore::run_plugin(name).await,
        LEMONADE_PLUGIN_ID => crate::plugins::lemonade::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

#[cfg(test)]
mod tests {
    use super::config::{MeshConfig, PluginConfigEntry};
    use super::*;
    use crate::inference::provider;
    use crate::inference::provider::InferenceProvider;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct NoopTestBridge;

    impl PluginRpcBridge for NoopTestBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async {
                Ok(RpcResult {
                    result_json: "{}".into(),
                })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    struct EnsureEndpointTestBridge;

    impl PluginRpcBridge for EnsureEndpointTestBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            method: String,
            params_json: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(async move {
                match method.as_str() {
                    "inference/ensure_endpoint" => {
                        let request: mesh_llm_plugin::EnsureInferenceEndpointRequest =
                            serde_json::from_str(&params_json).unwrap();
                        Ok(RpcResult {
                            result_json: serde_json::to_string(
                                &mesh_llm_plugin::EnsureInferenceEndpointResponse {
                                    address: format!(
                                        "http://127.0.0.1:{}/v1",
                                        request.requested_port.unwrap_or(8091)
                                    ),
                                    backend_label: "mlx".into(),
                                    context_length: 32768,
                                },
                            )
                            .unwrap(),
                        })
                    }
                    "inference/ensure_worker" => {
                        let request: mesh_llm_plugin::EnsureInferenceWorkerRequest =
                            serde_json::from_str(&params_json).unwrap();
                        assert_eq!(request.device_hint.as_deref(), Some("cuda:0"));
                        Ok(RpcResult {
                            result_json: serde_json::to_string(
                                &mesh_llm_plugin::EnsureInferenceWorkerResponse { port: 19091 },
                            )
                            .unwrap(),
                        })
                    }
                    "inference/prepare_moe_shard" => {
                        let request: mesh_llm_plugin::PrepareMoeShardRequest =
                            serde_json::from_str(&params_json).unwrap();
                        assert_eq!(request.experts, vec![1, 2, 5]);
                        assert_eq!(request.n_shared, 1);
                        assert_eq!(request.n_unique, 2);
                        Ok(RpcResult {
                            result_json: serde_json::to_string(
                                &mesh_llm_plugin::PrepareMoeShardResponse::default(),
                            )
                            .unwrap(),
                        })
                    }
                    _ => Ok(RpcResult {
                        result_json: "{}".into(),
                    }),
                }
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    async fn spawn_fake_models_server(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_seen = requests.clone();
        let handle = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let _ = stream.read(&mut buf).await.unwrap();
                requests_seen.fetch_add(1, Ordering::SeqCst);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            }
        });
        (format!("http://{addr}/api/v1"), handle, requests)
    }

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blobstore_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLOBSTORE_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn lemonade_can_be_enabled_explicitly() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: LEMONADE_PLUGIN_ID.into(),
                enabled: Some(true),
                command: None,
                args: Vec::new(),
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 3);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, LEMONADE_PLUGIN_ID);
        assert_eq!(resolved.externals[2].name, BLOBSTORE_PLUGIN_ID);
    }

    #[test]
    fn blackboard_is_resolved_on_public_meshes() {
        let resolved = resolve_plugins(
            &MeshConfig::default(),
            PluginHostMode {
                mesh_visibility: MeshVisibility::Public,
            },
        )
        .unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 3);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert_eq!(resolved.externals[1].name, "demo");
        assert_eq!(resolved.externals[2].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn instance_ids_include_pid_and_random_suffix() {
        let instance_id = make_instance_id();
        let prefix = format!("p{}-", std::process::id());
        assert!(instance_id.starts_with(&prefix));
        assert_eq!(instance_id.len(), prefix.len() + 8);
        assert!(instance_id[prefix.len()..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit()));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_path_is_namespaced_by_instance_id() {
        let path = unix_socket_path("p1234-deadbeef", "Pipes").unwrap();
        assert_eq!(
            path.file_name().and_then(|value| value.to_str()),
            Some("p1234-deadbeef-Pipes.sock")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_pipe_name_is_namespaced_by_instance_id() {
        assert_eq!(
            windows_pipe_name("p1234-deadbeef", "Pipes"),
            r"\\.\pipe\mesh-llm-p1234-deadbeef-Pipes"
        );
    }

    fn running_summary() -> PluginSummary {
        PluginSummary {
            name: "demo".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            version: None,
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            error: None,
        }
    }

    #[test]
    fn running_plugin_endpoints_are_healthy() {
        let summary = running_summary();
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn restarting_plugin_endpoints_are_not_available() {
        let summary = PluginSummary {
            status: "restarting".into(),
            error: Some("timed out".into()),
            ..running_summary()
        };
        assert_eq!(
            endpoint_record_from_plugin_status(&summary),
            EndpointHealthRecord {
                state: "starting".into(),
                available: false,
                detail: Some("timed out".into()),
                models: Vec::new(),
            }
        );
    }

    #[test]
    fn first_probe_failure_stays_in_startup_grace() {
        let now = Instant::now();
        let state = apply_endpoint_probe(
            None,
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("GET /models failed".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(state.record.state, "starting");
        assert!(!state.record.available);
        assert_eq!(state.consecutive_failures, 1);
    }

    #[test]
    fn healthy_endpoint_degrades_before_becoming_unhealthy() {
        let now = Instant::now();
        let healthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: 0,
        };

        let degraded = apply_endpoint_probe(
            Some(&healthy),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now,
        );
        assert_eq!(degraded.record.state, "degraded");
        assert!(degraded.record.available);
        assert_eq!(degraded.consecutive_failures, 1);

        let unhealthy = apply_endpoint_probe(
            Some(&degraded),
            EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            now + Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS),
        );
        assert_eq!(unhealthy.record.state, "unhealthy");
        assert!(!unhealthy.record.available);
        assert_eq!(unhealthy.consecutive_failures, 2);
    }

    #[test]
    fn unhealthy_endpoint_recovers_immediately_on_success() {
        let now = Instant::now();
        let unhealthy = EndpointHealthState {
            record: EndpointHealthRecord {
                state: "unhealthy".into(),
                available: false,
                detail: Some("503".into()),
                models: Vec::new(),
            },
            first_checked_at: now - Duration::from_secs(ENDPOINT_STARTUP_GRACE_SECS + 1),
            consecutive_failures: ENDPOINT_FAILURE_THRESHOLD,
        };

        let recovered = apply_endpoint_probe(
            Some(&unhealthy),
            EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models: vec!["demo".into()],
            },
            now,
        );
        assert_eq!(recovered.record.state, "healthy");
        assert!(recovered.record.available);
        assert_eq!(recovered.record.models, vec!["demo".to_string()]);
        assert_eq!(recovered.consecutive_failures, 0);
    }

    #[test]
    fn models_probe_url_extends_openai_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/v1/models");
    }

    #[test]
    fn models_probe_url_extends_api_v1_base() {
        let url = endpoint_models_url("http://localhost:8000/api/v1").unwrap();
        assert_eq!(url.as_str(), "http://localhost:8000/api/v1/models");
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_extracts_models_from_fake_server() {
        let (address, handle, requests) = spawn_fake_models_server(vec![(
            "200 OK",
            r#"{"data":[{"id":"lemonade-small"},{"id":"lemonade-large"}]}"#,
        )])
        .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(health.available);
        assert_eq!(health.state, "healthy");
        assert_eq!(
            health.models,
            vec!["lemonade-small".to_string(), "lemonade-large".to_string()]
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_marks_503_unavailable() {
        let (address, handle, requests) =
            spawn_fake_models_server(vec![("503 Service Unavailable", r#"{"error":"warming"}"#)])
                .await;

        let health = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!health.available);
        assert_eq!(health.state, "unhealthy");
        assert!(health.models.is_empty());
        assert!(health
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("503 Service Unavailable"));
        assert_eq!(requests.load(Ordering::SeqCst), 1);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn openai_http_endpoint_probe_recovers_when_fake_server_recovers() {
        let (address, handle, requests) = spawn_fake_models_server(vec![
            ("503 Service Unavailable", r#"{"error":"warming"}"#),
            ("200 OK", r#"{"data":[{"id":"lemonade-recovered"}]}"#),
        ])
        .await;

        let first = probe_openai_compatible_http_endpoint(&address).await;
        assert!(!first.available);
        assert_eq!(first.state, "unhealthy");

        let second = probe_openai_compatible_http_endpoint(&address).await;
        assert!(second.available);
        assert_eq!(second.state, "healthy");
        assert_eq!(second.models, vec!["lemonade-recovered".to_string()]);
        assert_eq!(requests.load(Ordering::SeqCst), 2);

        handle.await.unwrap();
    }

    #[test]
    fn endpoint_declares_inference_capabilities() {
        let endpoint = proto::EndpointManifest {
            endpoint_id: "demo".into(),
            kind: proto::EndpointKind::Inference as i32,
            transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
            protocol: Some("openai_compatible".into()),
            address: Some("http://localhost:8000/api/v1".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: false,
        };
        assert_eq!(
            endpoint_declared_capabilities(&endpoint),
            vec![
                "endpoint:inference".to_string(),
                "endpoint:inference/openai_compatible".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn managed_inference_endpoints_only_export_plugin_managed_inference_entries() {
        let plugin_manager = PluginManager::for_test_bridge(&["mlx"], Arc::new(NoopTestBridge));
        let mut manifests = BTreeMap::new();
        manifests.insert(
            "mlx".into(),
            proto::PluginManifest {
                endpoints: vec![
                    proto::EndpointManifest {
                        endpoint_id: "local-mlx".into(),
                        kind: proto::EndpointKind::Inference as i32,
                        transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
                        protocol: Some("openai_compatible".into()),
                        address: Some("http://127.0.0.1:8091/v1".into()),
                        args: Vec::new(),
                        namespace: None,
                        supports_streaming: true,
                        managed_by_plugin: true,
                    },
                    proto::EndpointManifest {
                        endpoint_id: "attached".into(),
                        kind: proto::EndpointKind::Inference as i32,
                        transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
                        protocol: Some("openai_compatible".into()),
                        address: Some("http://127.0.0.1:8000/api/v1".into()),
                        args: Vec::new(),
                        namespace: None,
                        supports_streaming: true,
                        managed_by_plugin: false,
                    },
                ],
                ..Default::default()
            },
        );
        plugin_manager.set_test_manifests(manifests).await;

        let endpoints = plugin_manager.managed_inference_endpoints().await.unwrap();
        assert_eq!(
            endpoints,
            vec![ManagedInferenceEndpoint {
                plugin_name: "mlx".into(),
                endpoint_id: "local-mlx".into(),
                protocol: Some("openai_compatible".into()),
                address: Some("http://127.0.0.1:8091/v1".into()),
                supports_streaming: true,
            }]
        );
    }

    #[tokio::test]
    async fn managed_inference_endpoints_can_sync_into_provider_registry() {
        provider::clear_registered_providers_for_tests();

        let plugin_manager = PluginManager::for_test_bridge(&["mlx"], Arc::new(NoopTestBridge));
        let mut manifests = BTreeMap::new();
        manifests.insert(
            "mlx".into(),
            proto::PluginManifest {
                endpoints: vec![proto::EndpointManifest {
                    endpoint_id: "local-mlx".into(),
                    kind: proto::EndpointKind::Inference as i32,
                    transport_kind: proto::EndpointTransportKind::EndpointTransportHttp as i32,
                    protocol: Some("openai_compatible".into()),
                    address: Some("http://127.0.0.1:8091/v1".into()),
                    args: Vec::new(),
                    namespace: None,
                    supports_streaming: true,
                    managed_by_plugin: true,
                }],
                ..Default::default()
            },
        );
        plugin_manager.set_test_manifests(manifests).await;

        let registrations = plugin_manager
            .managed_inference_endpoints()
            .await
            .unwrap()
            .into_iter()
            .map(|endpoint| {
                let provider_id =
                    provider::plugin_provider_id(&endpoint.plugin_name, &endpoint.endpoint_id);
                (
                    provider::PluginInferenceProviderRegistration::new(
                        provider_id.clone(),
                        endpoint.plugin_name.clone(),
                        provider::InferenceProviderCapabilities {
                            supports_local_runtime: true,
                            supports_distributed_host_runtime: false,
                            requires_worker_runtime: false,
                            supports_moe_shard_runtime: false,
                        },
                    ),
                    Arc::new(provider::PluginManagedEndpointProvider::new(
                        provider_id,
                        endpoint.plugin_name,
                        endpoint.endpoint_id,
                        plugin_manager.clone(),
                    )) as Arc<dyn provider::InferenceProvider>,
                )
            })
            .collect();
        provider::sync_plugin_providers(registrations);

        let selection = provider::select_local_endpoint_provider(
            &provider::InferenceEndpointRequest::local("/tmp/model.gguf", 8080, 1, 1)
                .with_preferred_provider_id(Some("plugin.mlx.local-mlx")),
        );
        assert_eq!(selection.provider_id(), "plugin.mlx.local-mlx");
        assert_eq!(selection.backend_label(), "mlx");

        provider::clear_registered_providers_for_tests();
    }

    #[tokio::test]
    async fn plugin_managed_provider_can_ensure_endpoint_via_plugin_rpc() {
        let plugin_manager =
            PluginManager::for_test_bridge(&["mlx"], Arc::new(EnsureEndpointTestBridge));
        let provider = provider::PluginManagedEndpointProvider::new(
            "plugin.mlx.local-mlx",
            "mlx",
            "local-mlx",
            plugin_manager,
        );

        let process = provider
            .start_endpoint(
                std::path::Path::new("."),
                None,
                &provider::InferenceEndpointRequest::local("/tmp/model.gguf", 8123, 1, 1),
            )
            .await
            .unwrap();

        assert_eq!(process.listen_port, 8123);
        assert_eq!(process.context_length, 32768);
    }

    #[tokio::test]
    async fn plugin_managed_provider_can_ensure_worker_via_plugin_rpc() {
        let plugin_manager =
            PluginManager::for_test_bridge(&["mlx"], Arc::new(EnsureEndpointTestBridge));
        let provider = provider::PluginManagedEndpointProvider::new(
            "plugin.mlx.local-mlx",
            "mlx",
            "local-mlx",
            plugin_manager,
        );

        let port = provider
            .start_worker(
                std::path::Path::new("."),
                None,
                &provider::InferenceWorkerRequest::default()
                    .with_model_path(Some("/tmp/model.gguf"))
                    .with_device_hint(Some("cuda:0")),
            )
            .await
            .unwrap();

        assert_eq!(port, 19091);
    }

    #[tokio::test]
    async fn plugin_managed_provider_can_prepare_moe_shard_via_plugin_rpc() {
        let plugin_manager =
            PluginManager::for_test_bridge(&["mlx"], Arc::new(EnsureEndpointTestBridge));
        let provider = provider::PluginManagedEndpointProvider::new(
            "plugin.mlx.local-mlx",
            "mlx",
            "local-mlx",
            plugin_manager,
        );

        let root = std::env::temp_dir().join(format!(
            "mesh-llm-plugin-managed-shard-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let model_path = root.join("model.gguf");
        let output_path = root.join("node-0.gguf");
        std::fs::write(&model_path, b"gguf").unwrap();

        provider
            .prepare_moe_shard(
                std::path::Path::new("."),
                &model_path,
                &crate::inference::moe::NodeAssignment {
                    experts: vec![1, 2, 5],
                    n_shared: 1,
                    n_unique: 2,
                },
                &output_path,
            )
            .await
            .unwrap();

        let _ = std::fs::remove_dir_all(&root);
    }
}
