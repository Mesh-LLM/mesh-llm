mod default_plugin_policy;
mod request;

use super::config::{ExternalPluginSpec, PluginHostMode};
use super::plugin_manifest_overview;
use super::support::{PluginTimeout, plugin_error, serialize_params, summarize_capabilities};
use super::transport::{
    LocalListener, LocalStream, bind_local_listener, connection_loop, reject_remote_control,
};
use super::{
    PROTOCOL_VERSION, PluginMeshEvent, PluginRpcBridge, PluginSummary, PluginWebUiState,
    PluginWebUiStateInput, REQUEST_TIMEOUT_SECS, ToolCallResult, ToolSummary,
    derive_plugin_web_ui_state, proto,
};
use crate::runtime_data::RuntimeDataProducer;
use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{MeshVisibility, STARTUP_DISABLED_ERROR_CODE};
use mesh_llm_plugin_manager::defaults::{UNLISTED_DEFAULT_ALLOWS, default_plugin};
use rmcp::model::{InitializeRequestParams, ServerConfig};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};

mod exchange_request;
#[cfg(test)]
mod exchange_request_tests;
#[cfg(test)]
mod exit_tests;
mod identity_artifact;
mod lifecycle_readiness;
#[cfg(test)]
mod restart_tests;
#[cfg(test)]
mod settings_tests;
mod shutdown;
mod socket_auth;
mod start_announce;
#[cfg(test)]
mod start_announce_tests;
pub(crate) mod status_publication;

use self::default_plugin_policy::restrict_default_plugin;
use self::status_publication::StatusDeclaration;

/// How long a stopping plugin waits to tell the node to withdraw its key.
const PLUGIN_STOPPED_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

pub(crate) struct ExternalPlugin {
    spec: ExternalPluginSpec,
    installed_artifact_sha256: Option<String>,
    authenticated_peer: Arc<AtomicBool>,
    web_ui_enabled: Arc<Mutex<Option<bool>>>,
    web_ui_primary_tab: Arc<Mutex<Option<bool>>>,
    instance_id: String,
    host_mode: PluginHostMode,
    summary: Arc<Mutex<PluginSummary>>,
    server_info: Arc<Mutex<Option<ServerConfig>>>,
    manifest: Arc<Mutex<Option<proto::PluginManifest>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<super::PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    runtime_data_producer: RuntimeDataProducer,
    restart_lock: Arc<Mutex<()>>,
    stopping: AtomicBool,
    /// Set when a node takes this plugin's manager
    /// ([`Self::expect_mesh_snapshots`]). The node sends every plugin its
    /// mesh snapshot then, which covers the starts before it; any later start
    /// (a restart, or a lazy plugin's first start) tells the node, which
    /// sends that plugin the snapshot again. Without a node (the plugin CLI)
    /// there is no snapshot to wait for.
    announce_starts: AtomicBool,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
    /// Serves this plugin as a task in this process instead of a child.
    in_process: Option<super::InProcessPluginRunner>,
}

pub(crate) struct PluginRuntime {
    pub(crate) managed_models: Vec<mesh_llm_plugin::models::ManagedModelConfig>,
    pub(crate) generation: u64,
    /// Its start finished: initialized and, after `spawn`, given the mesh
    /// snapshot. Only a started runtime takes `ensure_running`'s fast path.
    pub(crate) started: bool,
    pub(crate) authenticated_peer: bool,
    pub(crate) initialized_lifecycle: Option<Box<proto::OpenAiExchangeHookManifest>>,
    /// What this generation's manifest says about `/api/status`.
    pub(crate) status_declaration: StatusDeclaration,
    /// Whether this generation's manifest declares `inference_observation`;
    /// meaningful once `status_declaration` is no longer pending.
    pub(crate) declares_inference_observation: bool,
    pub(crate) _child: Option<Child>,
    connection_task: tokio::task::JoinHandle<()>,
    pub(crate) outbound_tx: mpsc::Sender<proto::Envelope>,
    pub(crate) pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
}

type PendingResponses = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>;

async fn stop_runtime(runtime: PluginRuntime, reason: &str) {
    runtime.connection_task.abort();
    let _ = runtime.connection_task.await;
    let mut pending = runtime.pending.lock().await;
    for (_, response) in pending.drain() {
        let _ = response.send(Err(anyhow::anyhow!(reason.to_string())));
    }
}

impl ExternalPlugin {
    pub(crate) fn installed_identity_executable(&self) -> Option<PathBuf> {
        identity_artifact::installed_executable(&self.spec)
    }
    pub(crate) fn installed_artifact_sha256(&self) -> Option<&str> {
        self.installed_artifact_sha256.as_deref()
    }
    pub fn exchange_grant(&self) -> Option<&mesh_llm_config::OpenAiExchangeGrant> {
        self.spec.openai_exchange_grant.as_deref()
    }

    pub(crate) fn installed_metadata(
        &self,
    ) -> Option<&mesh_llm_plugin_manager::InstalledPluginMetadata> {
        self.spec.installed_metadata.as_ref()
    }

    pub(crate) async fn spawn(
        spec: &ExternalPluginSpec,
        instance_id: String,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
        runtime_data_producer: RuntimeDataProducer,
        in_process: Option<super::InProcessPluginRunner>,
    ) -> Result<Self> {
        let in_process = in_process.filter(|_| spec.command.is_empty());
        let installed_artifact_sha256 = identity_artifact::capture_startup_digest(spec).await?;
        let plugin = Self {
            spec: spec.clone(),
            installed_artifact_sha256,
            authenticated_peer: Arc::new(AtomicBool::new(false)),
            web_ui_enabled: Arc::new(Mutex::new(spec.web_ui_enabled)),
            web_ui_primary_tab: Arc::new(Mutex::new(spec.web_ui_primary_tab)),
            instance_id,
            host_mode,
            summary: Arc::new(Mutex::new(PluginSummary {
                name: spec.name.clone(),
                kind: "external".into(),
                enabled: true,
                status: "starting".into(),
                pid: None,
                version: None,
                capabilities: Vec::new(),
                command: (!spec.command.is_empty()).then(|| spec.command.clone()),
                args: spec.args.clone(),
                tools: Vec::new(),
                manifest: None,
                web_ui: PluginWebUiState::default(),
                startup: Some(spec.startup.summary()),
                error: None,
            })),
            server_info: Arc::new(Mutex::new(None)),
            manifest: Arc::new(Mutex::new(None)),
            runtime: Arc::new(Mutex::new(None)),
            mesh_tx,
            rpc_bridge,
            runtime_data_producer,
            restart_lock: Arc::new(Mutex::new(())),
            stopping: AtomicBool::new(false),
            announce_starts: AtomicBool::new(false),
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
            in_process,
        };
        if spec.startup.lazy_start {
            plugin.mark_deferred().await;
        } else if let Err(err) = plugin.ensure_running().await
            && !plugin.is_disabled().await
        {
            return Err(err);
        }
        Ok(plugin)
    }

    pub(crate) fn name(&self) -> &str {
        &self.spec.name
    }

    pub(crate) async fn summary(&self) -> PluginSummary {
        let mut summary = self.summary.lock().await.clone();
        let manifest = self.manifest.lock().await.clone();
        summary.manifest = manifest.as_ref().map(plugin_manifest_overview);
        summary.web_ui = derive_plugin_web_ui_state(PluginWebUiStateInput {
            plugin_name: &self.spec.name,
            live_manifest: manifest.as_ref(),
            installed_metadata: self.spec.installed_metadata.as_ref(),
            web_ui_enabled: *self.web_ui_enabled.lock().await,
            web_ui_primary_tab: self.web_ui_primary_tab.lock().await.unwrap_or(false),
            runtime_available: summary.status == "running",
            runtime_unavailable_reason: summary.error.as_deref(),
        });
        summary
    }

    pub(crate) async fn set_web_ui_enabled(&self, enabled: bool) -> PluginWebUiState {
        *self.web_ui_enabled.lock().await = Some(enabled);
        self.publish_summary().await;
        self.summary().await.web_ui
    }

    pub(crate) async fn set_web_ui_primary_tab(
        &self,
        primary_tab_enabled: bool,
    ) -> PluginWebUiState {
        *self.web_ui_primary_tab.lock().await = Some(primary_tab_enabled);
        self.publish_summary().await;
        self.summary().await.web_ui
    }

    pub(crate) fn web_ui_asset_root(&self) -> Option<PathBuf> {
        plugin_web_ui_asset_root(&self.spec)
    }

    async fn publish_summary(&self) {
        let _ = self
            .runtime_data_producer
            .publish_plugin_summary(self.summary().await);
    }

    async fn publish_starting_summary(&self) {
        {
            let mut summary = self.summary.lock().await;
            summary.status = "starting".into();
            summary.pid = None;
            summary.error = None;
        }
        self.publish_summary().await;
    }

    async fn mark_deferred(&self) {
        {
            let mut summary = self.summary.lock().await;
            summary.status = "deferred".into();
            summary.pid = None;
            summary.error =
                Some("lazy start enabled; plugin will start on first direct use".to_string());
        }
        self.publish_summary().await;
    }

    fn log_waiting_for_connection(&self, listener: &LocalListener) {
        let endpoint = listener.endpoint();
        let transport = listener.transport_name();
        tracing::debug!(
            plugin = %self.spec.name,
            endpoint = %endpoint,
            transport,
            "Waiting for plugin connection"
        );
    }

    fn configured_child_command(&self, endpoint: &str, transport: &str) -> Command {
        let mut child = Command::new(&self.spec.command);
        child.args(&self.spec.args);
        child.env("MESH_LLM_PLUGIN_ENDPOINT", endpoint);
        child.env("MESH_LLM_PLUGIN_TRANSPORT", transport);
        child.env("MESH_LLM_PLUGIN_NAME", &self.spec.name);
        if let Some(ref url) = self.spec.url {
            child.env("MESH_LLM_PLUGIN_URL", url);
        }
        for (key, value) in &self.spec.env {
            child.env(key, value);
        }
        child.env_remove("MESH_LLM_PLUGIN_WEB_UI_DIR");
        if let Some(asset_root) = plugin_web_ui_asset_root(&self.spec) {
            child.env("MESH_LLM_PLUGIN_WEB_UI_DIR", asset_root);
        }
        child.stdin(std::process::Stdio::null());
        child.stdout(std::process::Stdio::null());
        child.stderr(std::process::Stdio::inherit());
        child.kill_on_drop(true);
        child
    }

    fn spawn_child_process(&self, endpoint: &str, transport: &str) -> Result<Child> {
        self.configured_child_command(endpoint, transport)
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to launch plugin '{}' via {}",
                    self.spec.name, self.spec.command
                )
            })
    }

    async fn await_plugin_connection(
        &self,
        listener: LocalListener,
        child_pid: Option<u32>,
    ) -> Result<LocalStream> {
        let stream = tokio::time::timeout(self.spec.startup.connect_timeout(), listener.accept())
            .await
            .with_context(|| format!("Timed out waiting for plugin '{}'", self.spec.name))??;
        if self.spec.openai_exchange_grant.is_some() {
            socket_auth::authenticate(&stream, child_pid)?;
        }
        Ok(stream)
    }

    async fn install_runtime(
        &self,
        child: Option<Child>,
        stream: LocalStream,
    ) -> Result<(u64, mpsc::Sender<proto::Envelope>, PendingResponses)> {
        let managed_models = super::model_assets::managed_models_from_env(&self.spec.env)?;
        let authenticated_peer = socket_auth::is_authenticated(
            &stream,
            child.as_ref().and_then(Child::id),
            self.in_process.is_some(),
        );
        self.authenticated_peer
            .store(authenticated_peer, Ordering::Release);
        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let outbound_tx_for_runtime = outbound_tx.clone();
        let outbound_tx_for_init = outbound_tx.clone();
        let connection_task = tokio::spawn(connection_loop(
            stream,
            outbound_rx,
            pending.clone(),
            self.mesh_tx.clone(),
            self.spec.name.clone(),
            self.summary.clone(),
            self.rpc_bridge.clone(),
            self.runtime.clone(),
            outbound_tx_for_runtime,
            generation,
            self.authenticated_peer.clone(),
        ));
        *self.runtime.lock().await = Some(PluginRuntime {
            managed_models,
            generation,
            started: false,
            authenticated_peer,
            initialized_lifecycle: None,
            status_declaration: StatusDeclaration::Pending,
            declares_inference_observation: false,
            _child: child,
            connection_task,
            outbound_tx,
            pending: pending.clone(),
        });
        Ok((generation, outbound_tx_for_init, pending))
    }

    async fn request_initialize(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
    ) -> Result<proto::InitializeResponse> {
        let host_info_json = serde_json::to_string(&InitializeRequestParams::default())?;
        let settings_json = serde_json::to_string(&self.spec.settings)?;
        let response = self
            .request_once(
                generation,
                outbound_tx,
                pending,
                proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    host_version: crate::VERSION.to_string(),
                    host_info_json,
                    mesh_visibility: proto_mesh_visibility(self.host_mode.mesh_visibility),
                    settings_json,
                    host_capabilities: vec![
                        mesh_llm_plugin::host_capabilities::INVOCATION_CALLBACKS.to_string(),
                        mesh_llm_plugin::models::ASSET_CAPABILITY.to_string(),
                        mesh_llm_plugin::host_capabilities::PEER_BLOCKS.to_string(),
                        mesh_llm_plugin::host_capabilities::OPENAI_EXCHANGE.to_string(),
                        mesh_llm_plugin::host_capabilities::PLUGIN_KEYS.to_string(),
                        mesh_llm_plugin::host_capabilities::PLUGIN_STATUS.to_string(),
                        mesh_llm_plugin::host_capabilities::LOCAL_INFERENCE_OBSERVATION.to_string(),
                    ],
                }),
                Some(self.spec.startup.init_timeout()),
            )
            .await?;
        self.parse_initialize_response(generation, response).await
    }

    async fn parse_initialize_response(
        &self,
        generation: u64,
        response: proto::Envelope,
    ) -> Result<proto::InitializeResponse> {
        let mut init = match response.payload {
            Some(proto::envelope::Payload::InitializeResponse(resp)) => resp,
            Some(proto::envelope::Payload::ErrorResponse(err))
                if err.code == STARTUP_DISABLED_ERROR_CODE =>
            {
                self.mark_disabled(generation, err.message).await;
                bail!("Plugin '{}' is disabled", self.spec.name);
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                bail!(
                    "Plugin '{}' rejected initialize: {}",
                    self.spec.name,
                    err.message
                )
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected initialize payload",
                self.spec.name
            ),
        };
        if self
            .spec
            .installed_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.default_managed)
        {
            let (allows, capabilities) = default_plugin(&self.spec.name)
                .map_or((UNLISTED_DEFAULT_ALLOWS, &[][..]), |default| {
                    (default.allows, default.capabilities)
                });
            let removed = restrict_default_plugin(&mut init, allows, capabilities);
            if !removed.is_empty() {
                tracing::warn!(
                    plugin = %self.spec.name,
                    removed = %removed.join(", "),
                    "Removed what this default plugin is not allowed to use"
                );
            }
        }
        self.validate_initialize_response(&init)?;
        Ok(init)
    }

    fn validate_initialize_response(&self, init: &proto::InitializeResponse) -> Result<()> {
        if init.plugin_id != self.spec.name {
            bail!(
                "Plugin '{}' identified itself as '{}'",
                self.spec.name,
                init.plugin_id
            );
        }
        if init.plugin_protocol_version != PROTOCOL_VERSION {
            bail!(
                "Plugin '{}' uses protocol {}, host uses {}",
                self.spec.name,
                init.plugin_protocol_version,
                PROTOCOL_VERSION
            );
        }
        if let Some(manifest) = &init.manifest {
            socket_auth::validate_lifecycle_declaration(
                manifest.openai_exchange_hook.is_some(),
                self.authenticated_peer.load(Ordering::Acquire),
            )?;
            if manifest.openai_exchange_hook.as_ref().is_some_and(|hook| {
                hook.request_body || hook.effective_request_body || hook.response_body
            }) {
                tracing::warn!(plugin = %self.spec.name,
                    "plugin requests OpenAI prompt or response body access; only explicit host grants authorize observation");
            }
            mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
                manifest.openai_exchange_hook.as_deref(),
                self.spec.openai_exchange_grant.as_deref(),
            )?;
            status_publication::validate_declaration(&self.spec.name, manifest)?;
        }
        Ok(())
    }

    async fn initialize_runtime(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
    ) -> Result<proto::InitializeResponse> {
        match self
            .request_initialize(generation, outbound_tx, pending)
            .await
        {
            Ok(init) => Ok(init),
            Err(err) => {
                if self.is_disabled().await {
                    return Err(err);
                }
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' failed initialize: {err}", self.spec.name),
                )
                .await;
                Err(err)
            }
        }
    }

    pub(crate) async fn supervise(&self) -> Result<()> {
        if self.is_disabled().await {
            return Ok(());
        }
        if self.is_deferred().await {
            return Ok(());
        }
        if self.is_stopping().await {
            return Ok(());
        }
        self.ensure_running().await?;
        let response = self
            .request(proto::envelope::Payload::HealthRequest(
                proto::HealthRequest {},
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::HealthResponse(resp))
                if resp.status == proto::health_response::Status::Ok as i32 =>
            {
                let mut summary = self.summary.lock().await;
                summary.status = "running".into();
                summary.error = None;
                drop(summary);
                self.publish_summary().await;
                Ok(())
            }
            Some(proto::envelope::Payload::HealthResponse(resp)) => {
                self.handle_runtime_failure(
                    None,
                    format!("health check reported status {}", resp.status),
                )
                .await;
                self.ensure_running().await
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                self.handle_runtime_failure(None, err.message).await;
                self.ensure_running().await
            }
            _ => {
                self.handle_runtime_failure(None, "unexpected health payload".into())
                    .await;
                self.ensure_running().await
            }
        }
    }

    async fn ensure_running(&self) -> Result<()> {
        self.reject_shutdown()?;
        if let Some(reason) = self.disabled_reason().await {
            bail!("Plugin '{}' is disabled: {}", self.spec.name, reason);
        }
        // A start in progress holds the restart lock until its process has
        // the mesh snapshot (`announce_started`): a caller that finds the
        // lock taken waits for it rather than reach the process first. The
        // guard is held while the runtime is checked, so no start can begin
        // in between (the restart lock is always taken before the runtime's).
        if let Ok(_no_start_in_progress) = self.restart_lock.try_lock()
            && self.runtime_started().await == Some(true)
        {
            return Ok(());
        }
        let _guard = self.restart_lock.lock().await;
        self.reject_shutdown()?;
        match self.runtime_started().await {
            Some(true) => return Ok(()),
            Some(false) => self.discard_unfinished_start().await,
            None => {}
        }

        self.authenticated_peer.store(false, Ordering::Release);
        self.publish_starting_summary().await;

        #[cfg(test)]
        if let Some(address) = self
            .spec
            .url
            .as_deref()
            .and_then(|url| url.strip_prefix("test+tcp://"))
        {
            let stream = tokio::net::TcpStream::connect(address).await?;
            let (generation, outbound_tx, pending) =
                self.install_runtime(None, LocalStream::Tcp(stream)).await?;
            return self.finish_startup(generation, outbound_tx, pending).await;
        }

        if let Some(runner) = &self.in_process {
            let stream = super::in_process::start_in_process(&self.spec.name, runner);
            let (generation, outbound_tx, pending) = self.install_runtime(None, stream).await?;
            return self.finish_startup(generation, outbound_tx, pending).await;
        }

        if let Some(remote_url) = self.spec.url.as_deref()
            && url::Url::parse(remote_url).is_ok_and(|url| url.scheme() == "tcp")
        {
            return reject_remote_control(remote_url);
        }

        let listener = bind_local_listener(&self.instance_id, &self.spec.name).await?;
        let endpoint = listener.endpoint();
        let transport = listener.transport_name();
        self.log_waiting_for_connection(&listener);

        let child = self.spawn_child_process(&endpoint, transport)?;
        let pid = child.id();
        self.summary.lock().await.pid = pid;

        let stream = self.await_plugin_connection(listener, pid).await?;
        let (generation, outbound_tx, pending) = self.install_runtime(Some(child), stream).await?;
        self.finish_startup(generation, outbound_tx, pending).await
    }

    async fn finish_startup(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
    ) -> Result<()> {
        match self
            .finish_initialize(generation, outbound_tx, pending)
            .await
        {
            Ok(()) if !self.announce_starts.load(Ordering::Acquire) => {
                self.mark_started(generation).await;
                Ok(())
            }
            Ok(()) => match self.announce_started(generation).await {
                Ok(()) => {
                    self.mark_started(generation).await;
                    Ok(())
                }
                Err(error) => {
                    self.handle_runtime_failure(Some(generation), error.to_string())
                        .await;
                    Err(error)
                }
            },
            Err(error) => {
                if self.is_disabled().await {
                    return Ok(());
                }
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' failed startup: {error}", self.spec.name),
                )
                .await;
                Err(error)
            }
        }
    }

    async fn finish_initialize(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
    ) -> Result<()> {
        let init = self
            .initialize_runtime(generation, outbound_tx, pending)
            .await?;

        let server_info: ServerConfig =
            serde_json::from_str(&init.server_info_json).with_context(|| {
                format!(
                    "Plugin '{}' returned invalid server_info_json",
                    self.spec.name
                )
            })?;
        self.publish_initialized_state(generation, &init, server_info.clone())
            .await?;

        let tools = init
            .manifest
            .as_ref()
            .map(manifest_tool_summaries)
            .unwrap_or_default();
        let mut summary = self.summary.lock().await;
        summary.status = "running".into();
        summary.version = Some(init.plugin_version);
        let mut declared_capabilities = init.capabilities;
        if let Some(manifest) = init.manifest {
            declared_capabilities.extend(manifest.capabilities);
        }
        summary.capabilities = summarize_capabilities(&server_info, &declared_capabilities);
        summary.tools = tools;
        summary.error = None;
        drop(summary);
        self.publish_summary().await;
        Ok(())
    }

    pub(crate) async fn server_info(&self) -> Result<ServerConfig> {
        self.ensure_running().await?;
        self.server_info
            .lock()
            .await
            .clone()
            .with_context(|| format!("Plugin '{}' did not publish server info", self.spec.name))
    }

    pub(crate) async fn manifest(&self) -> Result<Option<proto::PluginManifest>> {
        self.ensure_running().await?;
        Ok(self.manifest.lock().await.clone())
    }

    pub(crate) async fn manifest_snapshot(&self) -> Option<proto::PluginManifest> {
        self.manifest.lock().await.clone()
    }

    /// How long one operation may take before the host gives up on it: the
    /// deadline the plugin declares for it, `None` when the operation declares
    /// it may run as long as the caller does, or the host default.
    pub(crate) async fn operation_timeout(&self, operation: &str) -> Option<std::time::Duration> {
        let declared = self.manifest_snapshot().await.and_then(|manifest| {
            manifest
                .operations
                .into_iter()
                .find(|declared| declared.name == operation)
                .and_then(|declared| declared.timeout_secs)
        });
        match declared {
            Some(0) => None,
            Some(secs) => Some(std::time::Duration::from_secs(u64::from(secs))),
            None => Some(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS)),
        }
    }

    pub(crate) async fn list_tools(&self) -> Result<Vec<ToolSummary>> {
        Ok(self
            .manifest
            .lock()
            .await
            .clone()
            .map(|manifest| manifest_tool_summaries(&manifest))
            .unwrap_or_default())
    }

    pub(crate) async fn call_tool(
        &self,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        // The deadline an operation declares lives in its manifest, so the
        // plugin has to be running before the call can be given one.
        self.ensure_running().await?;
        let timeout = self.operation_timeout(tool_name).await;
        let response = self
            .invoke_service(
                proto::ServiceKind::Operation,
                tool_name,
                arguments_json,
                timeout,
            )
            .await?;
        Ok(ToolCallResult {
            content_json: response.output_json,
            is_error: response.is_error,
        })
    }

    pub(crate) async fn mcp_request<T, P>(&self, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        let params_json = serialize_params(params)?;
        let response = self
            .request(proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                invocation_id: super::mcp::invocation::forwarded_id(),
                method: method.to_string(),
                params_json,
            }))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::RpcResponse(resp)) => {
                serde_json::from_str(&resp.result_json).with_context(|| {
                    format!(
                        "Plugin '{}' returned invalid result for '{}'",
                        self.spec.name, method
                    )
                })
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, method, &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected RPC payload for '{}'",
                self.spec.name,
                method
            ),
        }
    }

    pub(crate) async fn mcp_notify<P>(&self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.send_unsolicited(
            proto::envelope::Payload::RpcNotification(proto::RpcNotification {
                invocation_id: None,
                method: method.to_string(),
                params_json: serialize_params(params)?,
            }),
            method,
        )
        .await
    }

    pub(crate) async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::ChannelMessage(message),
            "messages",
        )
        .await
    }

    /// Fire-and-forget variant of [`Self::send_channel_message`] for
    /// delivery paths that must never block the caller on plugin
    /// backpressure — backs `openai_exchange::OpenAiExchangeChannel::publish`,
    /// whose own doc comment already promises fire-and-forget delivery.
    /// `send_channel_message` (via `send_unsolicited`) awaits
    /// `outbound_tx.send()` on the bounded(256) per-plugin channel, which
    /// blocks the caller once that channel is full; this uses `try_send`
    /// instead, so a plugin that is behind on its queue — or not running at
    /// all — is skipped rather than stalling whatever request triggered the
    /// broadcast. Deliberately skips [`Self::ensure_running`] too: this is a
    /// broadcast to whatever is already up, not a reason to lazily start a
    /// plugin.
    pub(crate) async fn try_send_channel_message(
        &self,
        message: proto::ChannelMessage,
    ) -> Result<()> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_ref()
            .with_context(|| format!("Plugin '{}' is not running", self.spec.name))?;
        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::ChannelMessage(message)),
        };
        runtime
            .outbound_tx
            .try_send(envelope)
            .with_context(|| format!("Plugin '{}' outbound queue is full", self.spec.name))
    }

    pub(crate) async fn send_bulk_transfer_message(
        &self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::BulkTransferMessage(message),
            "bulk transfers",
        )
        .await
    }

    pub(crate) async fn send_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        self.send_unsolicited(proto::envelope::Payload::MeshEvent(event), "mesh events")
            .await
    }

    async fn request(&self, payload: proto::envelope::Payload) -> Result<proto::Envelope> {
        self.request_with_timeout(
            payload,
            Some(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS)),
        )
        .await
    }

    async fn send_unsolicited(&self, payload: proto::envelope::Payload, kind: &str) -> Result<()> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, _) = self.runtime_handles().await?;
            let envelope = proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id: 0,
                payload: Some(payload.clone()),
            };
            if outbound_tx.send(envelope).await.is_ok() {
                return Ok(());
            }
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting {kind}", self.spec.name),
            )
            .await;
            if attempt == 1 {
                break;
            }
        }
        bail!("Plugin '{}' is not accepting {}", self.spec.name, kind)
    }

    async fn runtime_handles(
        &self,
    ) -> Result<(
        u64,
        mpsc::Sender<proto::Envelope>,
        Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    )> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_ref()
            .with_context(|| format!("Plugin '{}' is not running", self.spec.name))?;
        Ok((
            runtime.generation,
            runtime.outbound_tx.clone(),
            runtime.pending.clone(),
        ))
    }

    async fn request_once(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::Envelope> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(request_id, tx);

        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id,
            payload: Some(payload),
        };

        if let Err(_send_err) = outbound_tx.send(envelope).await {
            pending.lock().await.remove(&request_id);
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting requests", self.spec.name),
            )
            .await;
            bail!("Plugin '{}' is not accepting requests", self.spec.name);
        }

        let response = match timeout {
            Some(timeout) => match tokio::time::timeout(timeout, rx).await {
                Ok(response) => response,
                Err(_) => {
                    pending.lock().await.remove(&request_id);
                    self.handle_runtime_failure(
                        Some(generation),
                        format!("Plugin '{}' timed out", self.spec.name),
                    )
                    .await;
                    return Err(anyhow::Error::new(PluginTimeout {
                        plugin_name: self.spec.name.clone(),
                    }));
                }
            },
            None => rx.await,
        };

        match response {
            Ok(resp) => resp,
            Err(_recv_err) => {
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' dropped the response channel", self.spec.name),
                )
                .await;
                bail!("Plugin '{}' dropped the response channel", self.spec.name);
            }
        }
    }

    async fn handle_runtime_failure(&self, generation: Option<u64>, reason: String) {
        let reason = crate::logging::policy::redact_urls_in_text(&reason);
        let failed_runtime = {
            let mut runtime = self.runtime.lock().await;
            let is_current_generation = generation
                .map(|generation| runtime.as_ref().map(|r| r.generation) == Some(generation))
                .unwrap_or(true);
            if !is_current_generation {
                return;
            }

            self.authenticated_peer.store(false, Ordering::Release);

            let failed_runtime = runtime.take();
            *self.server_info.lock().await = None;
            *self.manifest.lock().await = None;
            let mut summary = self.summary.lock().await;
            summary.status = "restarting".into();
            summary.pid = None;
            summary.version = None;
            summary.capabilities.clear();
            summary.tools.clear();
            summary.error = Some(reason.clone());
            drop(summary);
            self.runtime_data_producer
                .clear_plugin_reports(&self.spec.name);
            failed_runtime
        };
        if let Some(runtime) = failed_runtime {
            let generation = runtime.generation;
            stop_runtime(runtime, &reason).await;
            status_publication::announce_disconnected(&self.mesh_tx, &self.spec.name, generation);
        }
        self.publish_summary().await;
    }

    async fn disabled_reason(&self) -> Option<String> {
        let summary = self.summary.lock().await;
        if summary.status == "disabled" {
            Some(
                summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "disabled".to_string()),
            )
        } else {
            None
        }
    }

    async fn is_disabled(&self) -> bool {
        self.disabled_reason().await.is_some()
    }

    async fn is_deferred(&self) -> bool {
        if !self.spec.startup.lazy_start || self.runtime.lock().await.is_some() {
            return false;
        }
        self.summary.lock().await.status == "deferred"
    }

    async fn is_stopping(&self) -> bool {
        let summary = self.summary.lock().await;
        matches!(summary.status.as_str(), "shutting down" | "stopped")
    }

    /// Tell the node this plugin stopped for good, so it withdraws the key it
    /// announced for it. Bounded: a node that is itself stopping may not read it.
    async fn announce_stopped(&self) {
        let _ = self
            .mesh_tx
            .send_timeout(
                PluginMeshEvent::PluginStopped {
                    plugin_id: self.spec.name.clone(),
                },
                PLUGIN_STOPPED_SEND_TIMEOUT,
            )
            .await;
    }

    async fn mark_disabled(&self, generation: u64, reason: String) {
        let mut runtime = self.runtime.lock().await;
        let disabled_runtime = (runtime.as_ref().map(|runtime| runtime.generation)
            == Some(generation))
        .then(|| runtime.take())
        .flatten();
        if disabled_runtime.is_some() {
            self.authenticated_peer.store(false, Ordering::Release);
        }
        drop(runtime);
        if let Some(runtime) = disabled_runtime {
            stop_runtime(runtime, "plugin disabled").await;
        }

        let mut server_info = self.server_info.lock().await;
        *server_info = None;
        drop(server_info);

        let mut manifest = self.manifest.lock().await;
        *manifest = None;
        drop(manifest);

        let mut summary = self.summary.lock().await;
        summary.enabled = false;
        summary.status = "disabled".into();
        summary.pid = None;
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = Some(crate::logging::policy::redact_urls_in_text(&reason));
        drop(summary);
        self.publish_summary().await;
        self.announce_stopped().await;
    }
}

fn manifest_tool_summaries(manifest: &proto::PluginManifest) -> Vec<ToolSummary> {
    manifest
        .operations
        .iter()
        .map(|operation| ToolSummary {
            name: operation.name.clone(),
            description: operation.description.clone(),
            input_schema_json: operation.input_schema_json.clone(),
        })
        .collect()
}

fn proto_mesh_visibility(mesh_visibility: MeshVisibility) -> i32 {
    match mesh_visibility {
        MeshVisibility::Private => proto::MeshVisibility::Private as i32,
        MeshVisibility::Public => proto::MeshVisibility::Public as i32,
    }
}

fn plugin_web_ui_asset_root(spec: &ExternalPluginSpec) -> Option<PathBuf> {
    spec.installed_metadata
        .as_ref()
        .and_then(mesh_llm_plugin_manager::InstalledPluginMetadata::web_ui_asset_root_path)
        .filter(|path| path.is_dir())
}

#[cfg(test)]
#[path = "runtime/tests.rs"]
pub(crate) mod tests;

#[cfg(all(test, feature = "payments"))]
#[path = "runtime/payment_failure_tests.rs"]
mod payment_failure_tests;
