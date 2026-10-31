use super::super::config::{MeshConfig, PluginConfigEntry, resolve_plugins};
use super::super::transport::{read_envelope, write_envelope};
use super::super::{PluginCapabilityProvider, PluginEndpointSummary};
use super::*;

use crate::runtime_data::{
    PluginDataKey, PluginEndpointKey, RuntimeDataCollector, RuntimeDataSource,
};
use mesh_llm_plugin::MeshVisibility;
use mesh_llm_plugin_manager::store::{
    InstalledPluginManifestMetadata, InstalledPluginMetadata, InstalledPluginWebUiBundleMetadata,
    InstalledPluginWebUiConfigSectionMetadata, InstalledPluginWebUiMetadata,
    InstalledPluginWebUiPageMetadata, InstalledPluginWebUiPagePlacement,
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
        default_managed: false,
        manifest: Some(InstalledPluginManifestMetadata {
            openai_exchange_hook: None,
            config_schema: None,
            web_ui: Some(InstalledPluginWebUiMetadata {
                pages: vec![InstalledPluginWebUiPageMetadata {
                    id: "home".into(),
                    label: "Home".into(),
                    icon: Some("icons/home.svg".into()),
                    route: "index.html".into(),
                    bundle_id: "main".into(),
                    entry_script: "assets/app.js".into(),
                    placement: InstalledPluginWebUiPagePlacement::Auxiliary,
                    host_header: None,
                }],
                config_sections: vec![InstalledPluginWebUiConfigSectionMetadata {
                    id: "settings".into(),
                    title: "Settings".into(),
                    entry_script: "assets/settings.js".into(),
                    parent_tab: Some("integrations".into()),
                    bundle_id: "main".into(),
                }],
                contributions: Vec::new(),
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
        verified_sha256: None,
    }
}

pub(super) fn plugin_spec(
    temp_dir: &TempDir,
    web_ui_enabled: Option<bool>,
    validation_status: InstalledPluginWebUiValidationStatus,
    asset_root: Option<&str>,
) -> ExternalPluginSpec {
    ExternalPluginSpec {
        openai_exchange_grant: None,
        name: "demo".into(),
        command: "mesh-llm-plugin-demo".into(),
        args: Vec::new(),
        url: None,
        env: BTreeMap::new(),
        settings: Default::default(),
        startup: Default::default(),
        web_ui_enabled,
        web_ui_primary_tab: None,
        installed_metadata: Some(installed_metadata_with_web_ui(
            temp_dir.path().to_path_buf(),
            validation_status,
            asset_root,
        )),
    }
}

fn stopped_events(rx: &mut mpsc::Receiver<PluginMeshEvent>) -> Vec<String> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let PluginMeshEvent::PluginStopped { plugin_id } = event {
            out.push(plugin_id);
        }
    }
    out
}

#[tokio::test]
async fn a_plugin_that_is_disabled_or_shut_down_asks_the_node_to_withdraw_its_key() {
    let temp = tempfile::tempdir().unwrap();
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let (plugin, _runtime_data) = plugin_for_spec_with_mesh_tx(
        plugin_spec(
            &temp,
            None,
            InstalledPluginWebUiValidationStatus::Valid,
            None,
        ),
        mesh_tx,
    );

    // A failure restarts the plugin: its key stays announced.
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    assert!(stopped_events(&mut mesh_rx).is_empty());

    // Disabled: withdrawn.
    plugin.mark_disabled(0, "turned off".into()).await;
    assert_eq!(stopped_events(&mut mesh_rx), vec!["demo".to_string()]);

    // Shut down (removed, or the host stopping): withdrawn.
    plugin.shutdown().await;
    assert_eq!(stopped_events(&mut mesh_rx), vec!["demo".to_string()]);
}

/// A builtin spec with no command, as resolved for an in-process plugin.
pub(crate) fn in_process_plugin(
    name: &str,
    runner: crate::plugin::InProcessPluginRunner,
) -> ExternalPlugin {
    let mut plugin = plugin_for_spec(ExternalPluginSpec {
        openai_exchange_grant: None,
        name: name.into(),
        command: String::new(),
        args: Vec::new(),
        url: None,
        env: BTreeMap::new(),
        settings: Default::default(),
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
    });
    plugin.in_process = Some(runner);
    plugin
}

pub(super) fn plugin_for_spec(spec: ExternalPluginSpec) -> ExternalPlugin {
    plugin_for_spec_with_runtime_data(spec).0
}

fn plugin_for_spec_with_runtime_data(
    spec: ExternalPluginSpec,
) -> (ExternalPlugin, RuntimeDataCollector) {
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);
    plugin_for_spec_with_mesh_tx(spec, mesh_tx)
}

pub(super) fn plugin_for_spec_with_mesh_tx(
    spec: ExternalPluginSpec,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
) -> (ExternalPlugin, RuntimeDataCollector) {
    let runtime_data = RuntimeDataCollector::new();
    let plugin_name = spec.name.clone();
    let web_ui_enabled = spec.web_ui_enabled;
    let plugin = ExternalPlugin {
        installed_artifact_sha256: None,
        // These helpers construct trusted in-memory/mock test hosts.
        authenticated_peer: Arc::new(AtomicBool::new(true)),
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
        web_ui_primary_tab: Arc::new(Mutex::new(spec.web_ui_primary_tab)),
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
        stopping: AtomicBool::new(false),
        next_request_id: AtomicU64::new(1),
        next_generation: AtomicU64::new(1),
        in_process: None,
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

pub(super) async fn mark_plugin_running(plugin: &ExternalPlugin, generation: u64) {
    let child = sleeping_test_command()
        .kill_on_drop(true)
        .spawn()
        .expect("sleep process should start for runtime lifecycle test");
    let pid = child.id();
    let (outbound_tx, _outbound_rx) = mpsc::channel(1);
    *plugin.runtime.lock().await = Some(PluginRuntime {
        managed_models: Vec::new(),
        generation,
        authenticated_peer: true,
        initialized_lifecycle: None,
        status_declaration: StatusDeclaration::Pending,
        declares_inference_observation: false,
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
            timeout_secs: None,
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
    *plugin.server_info.lock().await = Some(ServerConfig::default());
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
    assert!(plugin.authenticated_peer.load(Ordering::Acquire));
    assert_eq!(*plugin.manifest.lock().await, replacement_manifest);
    assert_eq!(plugin.summary().await, replacement_summary);
    assert_eq!(
        runtime_data.plugin_snapshot(plugin.name()),
        replacement_reports
    );

    plugin.shutdown().await;
    assert!(!plugin.authenticated_peer.load(Ordering::Acquire));
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
#[serial_test::serial]
fn configured_plugin_url_is_normalized_before_child_env_export() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "demo".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            allow_peer_blocks: None,
            allow_inference_observation: None,
            openai_exchange_grant: None,
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
                        server_info_json: serde_json::to_string(&ServerConfig::default())
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
    let (generation, outbound_tx, pending) = plugin
        .install_runtime(None, LocalStream::Tcp(stream))
        .await
        .expect("valid managed model configuration");
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
    let (generation, outbound_tx, pending) = plugin
        .install_runtime(None, LocalStream::Tcp(stream))
        .await
        .expect("valid managed model configuration");
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
        let read = tokio::time::timeout(std::time::Duration::from_secs(2), stream.read(&mut byte))
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
    let (generation, outbound_tx, pending) = plugin
        .install_runtime(None, LocalStream::Tcp(stream))
        .await
        .expect("valid managed model configuration");
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

/// `send_channel_message` awaits capacity on the bounded outbound
/// channel and so blocks the caller once it's full — the exact failure
/// mode `try_send_channel_message` exists to avoid for fire-and-forget
/// delivery (see `plugin::openai_exchange::OpenAiExchangeChannel::publish`).
/// This fills a bounded(1) queue, keeps the receiver alive (so the
/// channel is genuinely full, not closed), and confirms the blocking
/// sibling still blocks (sanity-checking the fixture) while the
/// `try_send` variant returns immediately with an error instead.
#[tokio::test]
async fn try_send_channel_message_does_not_block_on_a_full_outbound_queue() {
    let temp_dir = TempDir::new().expect("temp dir should be created");
    let plugin = plugin_for_spec(plugin_spec(
        &temp_dir,
        None,
        InstalledPluginWebUiValidationStatus::Valid,
        Some("web"),
    ));
    let child = sleeping_test_command()
        .kill_on_drop(true)
        .spawn()
        .expect("sleep process should start for outbound-queue test");
    let (outbound_tx, outbound_rx) = mpsc::channel(1);
    *plugin.runtime.lock().await = Some(PluginRuntime {
        managed_models: Vec::new(),
        generation: 1,
        authenticated_peer: true,
        initialized_lifecycle: None,
        status_declaration: StatusDeclaration::Pending,
        declares_inference_observation: false,
        _child: Some(child),
        connection_task: tokio::spawn(std::future::pending::<()>()),
        outbound_tx: outbound_tx.clone(),
        pending: Arc::new(Mutex::new(HashMap::new())),
    });
    outbound_tx
        .try_send(proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".to_string(),
            request_id: 0,
            payload: None,
        })
        .expect("first send fills the bounded(1) queue");

    let blocking_sibling_still_pending = tokio::time::timeout(
        std::time::Duration::from_millis(50),
        plugin.send_channel_message(proto::ChannelMessage::default()),
    )
    .await
    .is_err();
    assert!(
        blocking_sibling_still_pending,
        "fixture sanity check: send_channel_message must still block on a full queue"
    );

    // Bounded by an outer timeout so a regression to blocking behavior
    // fails this test instead of hanging the whole suite forever —
    // nothing ever drains `outbound_rx` in this test.
    let started = std::time::Instant::now();
    let outcome = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        plugin.try_send_channel_message(proto::ChannelMessage::default()),
    )
    .await;
    let elapsed = started.elapsed();

    let result = outcome.unwrap_or_else(|_| {
        panic!(
            "try_send_channel_message must return immediately instead of awaiting \
             queue capacity; still pending after {elapsed:?}"
        )
    });
    assert!(
        result.is_err(),
        "a full outbound queue must be reported, not silently dropped"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(50),
        "try_send_channel_message must return immediately instead of awaiting \
         queue capacity, took {elapsed:?}"
    );

    drop(outbound_rx);
}

#[test]
fn backend_pricing_is_listed_exactly_when_the_build_can_send_it() {
    assert_eq!(
        mesh_llm_plugin::host_capabilities::OPENAI_EXCHANGE_BACKEND_PRICING,
        "openai_exchange.backend_pricing.v1"
    );
    let capabilities = initialize_host_capabilities();
    assert_eq!(
        capabilities
            .iter()
            .any(|capability| capability == "openai_exchange.backend_pricing.v1"),
        cfg!(feature = "payments")
    );
    assert!(
        capabilities
            .iter()
            .any(|capability| capability == mesh_llm_plugin::host_capabilities::OPENAI_EXCHANGE)
    );
}
