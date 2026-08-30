use super::config::{ExternalPluginSpec, PluginHostMode};
use super::plugin_manifest_overview;
use super::support::{plugin_error, serialize_params, summarize_capabilities};
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
use rmcp::model::{InitializeRequestParams, ServerInfo};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};

pub(crate) struct ExternalPlugin {
    spec: ExternalPluginSpec,
    web_ui_enabled: Arc<Mutex<Option<bool>>>,
    instance_id: String,
    host_mode: PluginHostMode,
    summary: Arc<Mutex<PluginSummary>>,
    server_info: Arc<Mutex<Option<ServerInfo>>>,
    manifest: Arc<Mutex<Option<proto::PluginManifest>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<super::PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    runtime_data_producer: RuntimeDataProducer,
    restart_lock: Arc<Mutex<()>>,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
}

pub(crate) struct PluginRuntime {
    pub(crate) generation: u64,
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
    pub(crate) async fn spawn(
        spec: &ExternalPluginSpec,
        instance_id: String,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
        runtime_data_producer: RuntimeDataProducer,
    ) -> Result<Self> {
        let plugin = Self {
            spec: spec.clone(),
            web_ui_enabled: Arc::new(Mutex::new(spec.web_ui_enabled)),
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
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
        };
        if spec.startup.lazy_start {
            plugin.mark_deferred().await;
            return Ok(plugin);
        }
        if let Err(err) = plugin.ensure_running().await {
            if plugin.is_disabled().await {
                return Ok(plugin);
            }
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

    async fn await_plugin_connection(&self, listener: LocalListener) -> Result<LocalStream> {
        tokio::time::timeout(self.spec.startup.connect_timeout(), listener.accept())
            .await
            .with_context(|| format!("Timed out waiting for plugin '{}'", self.spec.name))?
    }

    async fn install_runtime(
        &self,
        child: Option<Child>,
        stream: LocalStream,
    ) -> (u64, mpsc::Sender<proto::Envelope>, PendingResponses) {
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
        ));
        *self.runtime.lock().await = Some(PluginRuntime {
            generation,
            _child: child,
            connection_task,
            outbound_tx,
            pending: pending.clone(),
        });
        (generation, outbound_tx_for_init, pending)
    }

    async fn request_initialize(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
    ) -> Result<proto::InitializeResponse> {
        let host_info_json = serde_json::to_string(&InitializeRequestParams::default())?;
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
        let init = match response.payload {
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
        if let Some(reason) = self.disabled_reason().await {
            bail!("Plugin '{}' is disabled: {}", self.spec.name, reason);
        }
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }
        let _guard = self.restart_lock.lock().await;
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }

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
                self.install_runtime(None, LocalStream::Tcp(stream)).await;
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

        let stream = self.await_plugin_connection(listener).await?;
        let (generation, outbound_tx, pending) = self.install_runtime(Some(child), stream).await;
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
            Ok(()) => Ok(()),
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

        let server_info: ServerInfo =
            serde_json::from_str(&init.server_info_json).with_context(|| {
                format!(
                    "Plugin '{}' returned invalid server_info_json",
                    self.spec.name
                )
            })?;
        *self.server_info.lock().await = Some(server_info.clone());
        *self.manifest.lock().await = init.manifest.clone();

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

    pub(crate) async fn server_info(&self) -> Result<ServerInfo> {
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

    pub(crate) async fn open_stream(
        &self,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        let response = self
            .request(proto::envelope::Payload::OpenStreamRequest(request))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::OpenStreamResponse(resp)) => Ok(resp),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, "open_stream", &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected payload for 'open_stream'",
                self.spec.name
            ),
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

    pub(crate) async fn shutdown(&self) {
        {
            let mut summary = self.summary.lock().await;
            summary.status = "shutting down".into();
            summary.error = None;
        }

        let runtime = self.runtime.lock().await.take();
        if let Some(runtime) = runtime {
            stop_runtime(runtime, "plugin shutting down").await;
        }

        *self.server_info.lock().await = None;
        *self.manifest.lock().await = None;

        let mut summary = self.summary.lock().await;
        summary.status = "stopped".into();
        summary.pid = None;
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = None;
        drop(summary);
        self.publish_summary().await;
    }

    pub(crate) async fn call_tool(
        &self,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let response = self
            .invoke_service(
                proto::ServiceKind::Operation,
                tool_name,
                arguments_json,
                Some(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS)),
            )
            .await?;
        Ok(ToolCallResult {
            content_json: response.output_json,
            is_error: response.is_error,
        })
    }

    pub(crate) async fn call_tool_without_timeout(
        &self,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let response = self
            .invoke_service(
                proto::ServiceKind::Operation,
                tool_name,
                arguments_json,
                None,
            )
            .await?;
        Ok(ToolCallResult {
            content_json: response.output_json,
            is_error: response.is_error,
        })
    }

    pub(crate) async fn invoke_service(
        &self,
        kind: proto::ServiceKind,
        service_name: &str,
        input_json: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::InvokeServiceResponse> {
        let response = self
            .request_with_timeout(
                proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
                    kind: kind as i32,
                    service_name: service_name.to_string(),
                    input_json: input_json.to_string(),
                }),
                timeout,
            )
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::InvokeServiceResponse(resp)) => Ok(resp),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, "invoke_service", &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected payload for 'invoke_service'",
                self.spec.name
            ),
        }
    }

    pub(crate) async fn mcp_request<T, P>(&self, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        let params_json = serialize_params(params)?;
        let response = self
            .request(proto::envelope::Payload::RpcRequest(proto::RpcRequest {
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

    async fn request_with_timeout(
        &self,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::Envelope> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, pending) = self.runtime_handles().await?;
            match self
                .request_once(generation, outbound_tx, pending, payload.clone(), timeout)
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt == 0 => {
                    tracing::debug!(
                        plugin = %self.spec.name,
                        error = %err,
                        "Retrying plugin request after restart"
                    );
                }
                Err(err) => return Err(err),
            }
        }
        bail!("Plugin '{}' request failed after restart", self.spec.name)
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
                    bail!("Plugin '{}' timed out", self.spec.name);
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
            stop_runtime(runtime, &reason).await;
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

    async fn mark_disabled(&self, generation: u64, reason: String) {
        let mut runtime = self.runtime.lock().await;
        let disabled_runtime = (runtime.as_ref().map(|runtime| runtime.generation)
            == Some(generation))
        .then(|| runtime.take())
        .flatten();
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
mod tests {
    use super::super::config::{MeshConfig, PluginConfigEntry, resolve_plugins};
    use super::super::transport::{read_envelope, write_envelope};
    use super::super::{PluginCapabilityProvider, PluginEndpointSummary};
    use super::*;
    use crate::runtime_data::{
        PluginDataKey, PluginEndpointKey, RuntimeDataCollector, RuntimeDataSource,
    };
    use mesh_llm_plugin::MeshVisibility;
    use mesh_llm_plugin_manager::store::{
        InstalledPluginManifestMetadata, InstalledPluginMetadata,
        InstalledPluginWebUiBundleMetadata, InstalledPluginWebUiConfigSectionMetadata,
        InstalledPluginWebUiMetadata, InstalledPluginWebUiPageMetadata,
        InstalledPluginWebUiValidation, InstalledPluginWebUiValidationStatus,
    };
    use std::collections::BTreeMap;
    use std::ffi::OsStr;
    use std::time::Duration;
    use tempfile::TempDir;

    fn test_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    fn installed_metadata_with_web_ui(
        install_path: PathBuf,
        validation_status: InstalledPluginWebUiValidationStatus,
        asset_root: Option<&str>,
    ) -> InstalledPluginMetadata {
        if let Some(asset_root) = asset_root {
            std::fs::create_dir_all(install_path.join(asset_root)).unwrap();
        }
        InstalledPluginMetadata {
            name: "demo".into(),
            source_repository: "https://github.com/mesh-llm/demo".into(),
            installed_version: "v1.0.0".into(),
            target_triple: "test-target".into(),
            downloaded_asset_name: "demo.tar.gz".into(),
            install_path,
            enabled: true,
            manifest: Some(InstalledPluginManifestMetadata {
                config_schema: None,
                web_ui: Some(InstalledPluginWebUiMetadata {
                    pages: vec![InstalledPluginWebUiPageMetadata {
                        id: "home".into(),
                        label: "Home".into(),
                        icon: Some("icons/home.svg".into()),
                        route: "index.html".into(),
                        bundle_id: "main".into(),
                        entry_script: "assets/app.js".into(),
                    }],
                    config_sections: vec![InstalledPluginWebUiConfigSectionMetadata {
                        id: "settings".into(),
                        title: "Settings".into(),
                        entry_script: "assets/settings.js".into(),
                        parent_tab: Some("integrations".into()),
                        bundle_id: "main".into(),
                    }],
                    bundles: vec![InstalledPluginWebUiBundleMetadata {
                        id: "main".into(),
                        root_path: "web".into(),
                    }],
                    asset_root: asset_root.map(PathBuf::from),
                    validation: InstalledPluginWebUiValidation {
                        status: validation_status,
                        reason: Some("bundle failed validation".into()),
                    },
                }),
            }),
            last_protocol_version: Some(1),
            last_status: Some("running".into()),
            last_error: None,
        }
    }

    fn plugin_spec(
        temp_dir: &TempDir,
        web_ui_enabled: Option<bool>,
        validation_status: InstalledPluginWebUiValidationStatus,
        asset_root: Option<&str>,
    ) -> ExternalPluginSpec {
        ExternalPluginSpec {
            name: "demo".into(),
            command: "mesh-llm-plugin-demo".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: Default::default(),
            web_ui_enabled,
            installed_metadata: Some(installed_metadata_with_web_ui(
                temp_dir.path().to_path_buf(),
                validation_status,
                asset_root,
            )),
        }
    }

    fn plugin_for_spec(spec: ExternalPluginSpec) -> ExternalPlugin {
        plugin_for_spec_with_runtime_data(spec).0
    }

    fn plugin_for_spec_with_runtime_data(
        spec: ExternalPluginSpec,
    ) -> (ExternalPlugin, RuntimeDataCollector) {
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let runtime_data = RuntimeDataCollector::new();
        let plugin_name = spec.name.clone();
        let web_ui_enabled = spec.web_ui_enabled;
        let plugin = ExternalPlugin {
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
            spec,
            web_ui_enabled: Arc::new(Mutex::new(web_ui_enabled)),
            instance_id: "test-instance".into(),
            host_mode: test_host_mode(),
            server_info: Arc::new(Mutex::new(None)),
            manifest: Arc::new(Mutex::new(None)),
            runtime: Arc::new(Mutex::new(None)),
            mesh_tx,
            rpc_bridge: Arc::new(Mutex::new(None)),
            runtime_data_producer: runtime_data.producer(RuntimeDataSource {
                scope: "test",
                plugin_data_key: Some(PluginDataKey {
                    plugin_name,
                    data_key: "summary".into(),
                }),
                plugin_endpoint_key: None,
            }),
            restart_lock: Arc::new(Mutex::new(())),
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
        };
        (plugin, runtime_data)
    }

    fn command_env_value(command: &Command, key: &str) -> Option<String> {
        command
            .as_std()
            .get_envs()
            .find(|(name, _value)| *name == OsStr::new(key))
            .and_then(|(_name, value)| value.map(|value| value.to_string_lossy().into_owned()))
    }

    fn command_env_is_removed(command: &Command, key: &str) -> bool {
        command
            .as_std()
            .get_envs()
            .any(|(name, value)| name == OsStr::new(key) && value.is_none())
    }

    #[cfg(unix)]
    fn sleeping_test_command() -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 60"]);
        command
    }

    #[cfg(windows)]
    fn sleeping_test_command() -> Command {
        let mut command = Command::new("cmd");
        command.args(["/C", "ping -n 61 127.0.0.1 >NUL"]);
        command
    }

    async fn mark_plugin_running(plugin: &ExternalPlugin, generation: u64) {
        let child = sleeping_test_command()
            .kill_on_drop(true)
            .spawn()
            .expect("sleep process should start for runtime lifecycle test");
        let pid = child.id();
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        *plugin.runtime.lock().await = Some(PluginRuntime {
            generation,
            _child: Some(child),
            connection_task: tokio::spawn(std::future::pending::<()>()),
            outbound_tx,
            pending: Arc::new(Mutex::new(HashMap::new())),
        });
        let mut summary = plugin.summary.lock().await;
        summary.status = "running".into();
        summary.pid = pid;
        summary.version = Some("v1.0.0".into());
        summary.capabilities = vec!["operation:echo".into()];
        summary.error = None;
    }

    fn generation_manifest(label: &str) -> proto::PluginManifest {
        proto::PluginManifest {
            capabilities: vec![format!("capability:{label}")],
            operations: vec![proto::OperationManifest {
                name: format!("{label}.echo"),
                description: format!("{label} echo operation"),
                input_schema_json: r#"{"type":"object"}"#.into(),
                output_schema_json: None,
                title: None,
            }],
            ..proto::PluginManifest::default()
        }
    }

    async fn install_test_generation(plugin: &ExternalPlugin, generation: u64, label: &str) {
        if let Some(runtime) = plugin.runtime.lock().await.take() {
            stop_runtime(runtime, "test runtime replaced").await;
        }
        mark_plugin_running(plugin, generation).await;

        let manifest = generation_manifest(label);
        *plugin.server_info.lock().await = Some(ServerInfo::default());
        *plugin.manifest.lock().await = Some(manifest.clone());
        {
            let mut summary = plugin.summary.lock().await;
            summary.version = Some(format!("{label}-version"));
            summary.capabilities = vec![format!("capability:{label}")];
            summary.tools = manifest_tool_summaries(&manifest);
            summary.error = None;
        }
        plugin.publish_summary().await;
        plugin
            .runtime_data_producer
            .publish_plugin_manifest(plugin_manifest_overview(&manifest));
        plugin
            .runtime_data_producer
            .publish_plugin_providers(vec![PluginCapabilityProvider {
                capability: format!("capability:{label}"),
                plugin_name: plugin.name().into(),
                plugin_status: "running".into(),
                endpoint_id: Some(format!("{label}-endpoint")),
                available: true,
                detail: None,
            }]);
        plugin
            .runtime_data_producer
            .publish_plugin_payload("generation", serde_json::json!(label));
        plugin
            .runtime_data_producer
            .collector()
            .producer(RuntimeDataSource {
                scope: "test",
                plugin_data_key: None,
                plugin_endpoint_key: Some(PluginEndpointKey {
                    plugin_name: plugin.name().into(),
                    endpoint_id: format!("{label}-endpoint"),
                }),
            })
            .publish_plugin_endpoint(PluginEndpointSummary {
                plugin_name: plugin.name().into(),
                plugin_status: "running".into(),
                endpoint_id: format!("{label}-endpoint"),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "http".into(),
                protocol: Some("http".into()),
                address: Some(format!("http://127.0.0.1/{label}")),
                args: Vec::new(),
                namespace: Some(format!("demo.{label}")),
                supports_streaming: true,
                managed_by_plugin: true,
                detail: None,
                models: Vec::new(),
            });
    }

    #[tokio::test]
    async fn stale_runtime_failure_preserves_replacement_generation_metadata() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let (plugin, runtime_data) = plugin_for_spec_with_runtime_data(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        ));
        install_test_generation(&plugin, 41, "generation-a").await;
        install_test_generation(&plugin, 42, "generation-b").await;

        let replacement_summary = plugin.summary().await;
        let replacement_server_info = plugin.server_info.lock().await.clone();
        let replacement_manifest = plugin.manifest.lock().await.clone();
        let replacement_reports = runtime_data.plugin_snapshot(plugin.name());

        plugin
            .handle_runtime_failure(Some(41), "generation A failed late".into())
            .await;

        assert_eq!(
            plugin
                .runtime
                .lock()
                .await
                .as_ref()
                .map(|runtime| runtime.generation),
            Some(42)
        );
        assert_eq!(*plugin.server_info.lock().await, replacement_server_info);
        assert_eq!(*plugin.manifest.lock().await, replacement_manifest);
        assert_eq!(plugin.summary().await, replacement_summary);
        assert_eq!(
            runtime_data.plugin_snapshot(plugin.name()),
            replacement_reports
        );

        plugin.shutdown().await;
    }

    #[tokio::test]
    async fn current_runtime_failure_clears_its_generation_metadata() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let (plugin, runtime_data) = plugin_for_spec_with_runtime_data(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        ));
        install_test_generation(&plugin, 51, "current-generation").await;

        plugin
            .handle_runtime_failure(Some(51), "current generation failed".into())
            .await;

        assert!(plugin.runtime.lock().await.is_none());
        assert!(plugin.server_info.lock().await.is_none());
        assert!(plugin.manifest.lock().await.is_none());
        let summary = plugin.summary().await;
        assert_eq!(summary.status, "restarting");
        assert!(summary.pid.is_none());
        assert!(summary.version.is_none());
        assert!(summary.capabilities.is_empty());
        assert!(summary.tools.is_empty());
        assert_eq!(summary.error.as_deref(), Some("current generation failed"));

        let reports = runtime_data.plugin_snapshot(plugin.name());
        assert_eq!(reports.summary, Some(summary));
        assert!(reports.manifest.is_none());
        assert!(reports.providers.is_empty());
        assert!(reports.payloads.is_empty());
        assert!(reports.endpoints.is_empty());
    }

    #[test]
    fn child_env_includes_valid_package_web_ui_asset_root() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let asset_root = temp_dir.path().join("web");
        std::fs::create_dir(&asset_root).expect("asset root should be created");
        let plugin = plugin_for_spec(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        ));

        let command = plugin.configured_child_command("endpoint", "unix");

        assert_eq!(
            command_env_value(&command, "MESH_LLM_PLUGIN_WEB_UI_DIR"),
            Some(asset_root.display().to_string())
        );
    }

    #[test]
    fn configured_plugin_url_is_normalized_before_child_env_export() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("mesh-llm-plugin-demo".into()),
                args: Vec::new(),
                url: Some("\u{2003}https://plugin.example.test/v1\u{2003}".into()),
                settings: BTreeMap::new(),
                startup: Default::default(),
            }],
            ..MeshConfig::default()
        };
        let spec = resolve_plugins(&config, test_host_mode())
            .expect("configured plugin should resolve")
            .externals
            .into_iter()
            .find(|spec| spec.name == "demo")
            .expect("configured plugin spec should be stored");

        assert_eq!(spec.url.as_deref(), Some("https://plugin.example.test/v1"));
        let plugin = plugin_for_spec(spec);

        let command = plugin.configured_child_command("endpoint", "unix");

        assert_eq!(
            command_env_value(&command, "MESH_LLM_PLUGIN_URL"),
            Some("https://plugin.example.test/v1".into())
        );
    }

    #[tokio::test]
    async fn locally_connected_stream_completes_plugin_initialize_without_child_process() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("remote plugin listener");
        let address = listener.local_addr().expect("listener address");
        let (release_tx, release_rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("remote connection");
            let mut stream = LocalStream::Tcp(stream);
            let request = read_envelope(&mut stream)
                .await
                .expect("initialize request");
            assert!(matches!(
                request.payload,
                Some(proto::envelope::Payload::InitializeRequest(_))
            ));
            write_envelope(
                &mut stream,
                &proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: request.request_id,
                    plugin_id: "remote-demo".into(),
                    payload: Some(proto::envelope::Payload::InitializeResponse(
                        proto::InitializeResponse {
                            plugin_id: "remote-demo".into(),
                            plugin_protocol_version: PROTOCOL_VERSION,
                            plugin_version: "v1.0.0".into(),
                            server_info_json: serde_json::to_string(&ServerInfo::default())
                                .expect("server info"),
                            capabilities: Vec::new(),
                            manifest: None,
                        },
                    )),
                },
            )
            .await
            .expect("initialize response");
            let _ = release_rx.await;
        });
        let mut spec = plugin_spec(
            &TempDir::new().expect("temp dir"),
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        );
        spec.name = "remote-demo".into();
        spec.command.clear();
        let plugin = plugin_for_spec(spec);
        let stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("local test connection");
        let (generation, outbound_tx, pending) =
            plugin.install_runtime(None, LocalStream::Tcp(stream)).await;
        plugin
            .finish_startup(generation, outbound_tx, pending)
            .await
            .expect("locally connected plugin should initialize");
        let summary = plugin.summary().await;

        assert_eq!(summary.status, "running");
        assert_eq!(summary.pid, None);
        assert_eq!(summary.command, None);
        let _ = release_tx.send(());
        server.await.expect("remote server task");
    }

    #[tokio::test]
    async fn startup_disabled_response_remains_a_successful_disabled_plugin() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("remote plugin listener");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("remote connection");
            let mut stream = LocalStream::Tcp(stream);
            let request = read_envelope(&mut stream)
                .await
                .expect("initialize request");
            write_envelope(
                &mut stream,
                &proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: request.request_id,
                    plugin_id: "remote-disabled".into(),
                    payload: Some(proto::envelope::Payload::ErrorResponse(
                        proto::ErrorResponse {
                            code: STARTUP_DISABLED_ERROR_CODE,
                            message: "operator disabled this plugin".into(),
                            data_json: String::new(),
                        },
                    )),
                },
            )
            .await
            .expect("disabled response");
            let mut byte = [0_u8; 1];
            let read = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut byte))
                .await
                .expect("self-disable must close the remote control socket")
                .unwrap_or(0);
            assert_eq!(read, 0);
        });
        let mut spec = plugin_spec(
            &TempDir::new().expect("temp dir"),
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        );
        spec.name = "remote-disabled".into();
        spec.command.clear();
        let plugin = plugin_for_spec(spec);
        let stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("local test connection");
        let (generation, outbound_tx, pending) =
            plugin.install_runtime(None, LocalStream::Tcp(stream)).await;
        plugin
            .finish_startup(generation, outbound_tx, pending)
            .await
            .expect("plugin-owned disable is a successful startup outcome");
        let summary = plugin.summary().await;

        assert!(!summary.enabled);
        assert_eq!(summary.status, "disabled");
        assert_eq!(
            summary.error.as_deref(),
            Some("operator disabled this plugin")
        );
        assert!(plugin.runtime.lock().await.is_none());
        server.await.expect("remote server task");
    }

    #[tokio::test]
    async fn malformed_initialize_metadata_rolls_back_remote_runtime() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("remote plugin listener");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("remote connection");
            let mut stream = LocalStream::Tcp(stream);
            let request = read_envelope(&mut stream)
                .await
                .expect("initialize request");
            write_envelope(
                &mut stream,
                &proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    request_id: request.request_id,
                    plugin_id: "remote-malformed".into(),
                    payload: Some(proto::envelope::Payload::InitializeResponse(
                        proto::InitializeResponse {
                            plugin_id: "remote-malformed".into(),
                            plugin_protocol_version: PROTOCOL_VERSION,
                            plugin_version: "v1.0.0".into(),
                            server_info_json: "not-json".into(),
                            capabilities: Vec::new(),
                            manifest: None,
                        },
                    )),
                },
            )
            .await
            .expect("initialize response");
            let mut byte = [0_u8; 1];
            let read =
                tokio::time::timeout(std::time::Duration::from_secs(2), stream.read(&mut byte))
                    .await
                    .expect("failed startup should close connection")
                    .unwrap_or(0);
            assert_eq!(read, 0);
        });
        let mut spec = plugin_spec(
            &TempDir::new().expect("temp dir"),
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        );
        spec.name = "remote-malformed".into();
        spec.command.clear();
        let plugin = plugin_for_spec(spec);
        let stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("local test connection");
        let (generation, outbound_tx, pending) =
            plugin.install_runtime(None, LocalStream::Tcp(stream)).await;
        let error = plugin
            .finish_startup(generation, outbound_tx, pending)
            .await
            .expect_err("malformed initialize metadata must fail startup");

        assert!(error.to_string().contains("invalid server_info_json"));
        server.await.expect("remote server task");
    }

    #[test]
    fn child_env_removes_web_ui_dir_without_valid_asset_root() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let plugin = plugin_for_spec(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Invalid,
            Some("web"),
        ));

        let command = plugin.configured_child_command("endpoint", "unix");

        assert!(command_env_is_removed(
            &command,
            "MESH_LLM_PLUGIN_WEB_UI_DIR"
        ));
    }

    #[tokio::test]
    async fn invalid_web_ui_assets_do_not_clear_running_plugin_capabilities() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let plugin = plugin_for_spec(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Invalid,
            Some("web"),
        ));
        mark_plugin_running(&plugin, 1).await;

        let summary = plugin.summary().await;

        assert_eq!(summary.status, "running");
        assert_eq!(summary.capabilities, vec!["operation:echo".to_string()]);
        assert_eq!(
            summary.web_ui.state,
            super::super::PluginWebUiStateKind::Invalid
        );
        assert!(summary.web_ui.declared);
        assert!(summary.web_ui.enabled);
    }

    #[tokio::test]
    async fn disabled_web_ui_preference_does_not_stop_running_plugin() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let plugin = plugin_for_spec(plugin_spec(
            &temp_dir,
            Some(false),
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        ));
        mark_plugin_running(&plugin, 7).await;

        let summary = plugin.summary().await;
        let runtime_generation = plugin
            .runtime
            .lock()
            .await
            .as_ref()
            .map(|runtime| runtime.generation);

        assert_eq!(summary.status, "running");
        assert_eq!(runtime_generation, Some(7));
        assert!(summary.pid.is_some());
        assert_eq!(
            summary.web_ui.state,
            super::super::PluginWebUiStateKind::Disabled
        );
        assert!(!summary.web_ui.available);
    }

    #[tokio::test]
    async fn stopped_plugin_keeps_web_ui_metadata_as_plugin_not_running() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let plugin = plugin_for_spec(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        ));
        mark_plugin_running(&plugin, 11).await;

        plugin.shutdown().await;
        let summary = plugin.summary().await;

        assert_eq!(summary.status, "stopped");
        assert!(summary.pid.is_none());
        assert!(summary.web_ui.declared);
        assert_eq!(
            summary.web_ui.state,
            super::super::PluginWebUiStateKind::PluginNotRunning
        );
        assert_eq!(summary.web_ui.pages.len(), 1);
        assert_eq!(summary.web_ui.config_sections.len(), 1);
    }

    #[tokio::test]
    async fn stopped_plugin_publishes_plugin_not_running_to_runtime_data() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let (plugin, runtime_data) = plugin_for_spec_with_runtime_data(plugin_spec(
            &temp_dir,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            Some("web"),
        ));
        mark_plugin_running(&plugin, 17).await;
        plugin.publish_summary().await;

        plugin.shutdown().await;
        let summary = runtime_data
            .plugins_snapshot()
            .plugins
            .into_iter()
            .find(|summary| summary.name == "demo")
            .expect("stopped summary should remain visible");

        assert_eq!(summary.status, "stopped");
        assert_eq!(
            summary.web_ui.state,
            super::super::PluginWebUiStateKind::PluginNotRunning
        );
        assert_eq!(summary.web_ui.pages.len(), 1);
    }
}
