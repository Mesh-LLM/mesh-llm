use super::startup_identity::{emit_private_mesh_name_warning, handle_public_identity_transition};
use super::status::mesh_guardrail_mode_to_openai;
use super::{
    AutoRuntimeNodeSetup, BootstrapProxyStopTx, DashboardContextUsage, ManagedModelController,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, OpenAiGuardrailPolicyHandle,
    PreparedRuntimeStartup, RunAutoAdditionalModelsContext, RunAutoConsoleStateContext,
    RunAutoRuntimeLifecycleContext, RunAutoServingSurface, RunAutoServingSurfaceContext,
    RuntimeCapacityLedger, RuntimeDashboardSnapshotProvider, RuntimeEvent, RuntimeInstanceRegistry,
    RuntimeModelHandleEntry, RuntimeOptions, RuntimeResourcePlanningProfile, RuntimeSurface,
    SkippyNativeLogForwardingGuard, StartupLocalModelTask, StartupMeshCreationState,
    StartupModelPlan, StartupReadyReporter, bridge_skippy_native_logs, build_serving_list,
    cli_has_explicit_models, configure_skippy_native_logging, emit_configuration_ui_read_only_hint,
    initialize_embedded_runtime_entrypoint, initialize_runtime_entrypoint,
    maybe_discover_join_candidates, next_runtime_instance_id, nostr_rediscovery, nostr_relays,
    openai_guardrail_policy_handle, owner_runtime_config, prepare_runtime_startup,
    publish_initial_openai_guardrails_status, record_first_joined_mesh_ts,
    resolve_runtime_owner_key_path, resolve_startup_mesh_creation_state, run_auto_join_mesh_phase,
    run_auto_model_identity, run_auto_model_path_or_shutdown, run_auto_runtime_loop_and_shutdown,
    runtime_data_producer_for_console, runtime_startup_requirements, setup_run_auto_console_state,
    setup_run_auto_serving_surface, spawn_embedded_runtime_control_forwarder,
    spawn_run_auto_additional_model_tasks, spawn_run_auto_discovery_publisher,
    start_run_auto_bootstrap_proxy, startup_default_backend_device, startup_launch_plan,
    startup_local_model_loop, swarm_capture_observer_requested,
};
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::models;
use crate::network::{
    affinity, discovery as mesh_discovery,
    lan_bootstrap::{effective_quic_bind_ip, spawn_mdns_reverse_dial},
    nostr, tunnel,
};
use crate::plugin;
use crate::runtime::release_attestation;
use crate::runtime::survey;
use crate::system::{autoupdate, benchmark, hardware};
use anyhow::Result;
use mesh_llm_events::{LogFormat, OutputEvent, emit_event, output_sink};
use skippy_protocol::FlashAttentionType;
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};

pub(super) enum RunAutoModelSelection {
    Model(PathBuf),
    Shutdown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RuntimeUnloadOwner {
    Runtime,
    Managed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeUnloadCandidate {
    pub(super) owner: RuntimeUnloadOwner,
    pub(super) instance_id: String,
    pub(super) model_name: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmbeddedRuntimeMode {
    Serve,
    Client,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmbeddedRuntimeDiscoveryMode {
    Nostr,
    Mdns,
}

pub(crate) struct EmbeddedRuntimeOptions {
    pub(crate) mode: EmbeddedRuntimeMode,
    pub(crate) models: Vec<String>,
    pub(crate) join: Vec<String>,
    pub(crate) auto: bool,
    pub(crate) api_port: u16,
    pub(crate) console_port: u16,
    pub(crate) mesh_name: Option<String>,
    pub(crate) max_vram_gb: Option<f64>,
    pub(crate) publish: bool,
    pub(crate) discovery_mode: EmbeddedRuntimeDiscoveryMode,
    pub(crate) relay: Vec<String>,
    pub(crate) relay_auth: Vec<(String, String)>,
    pub(crate) disable_iroh_relays: bool,
    pub(crate) nostr_relay: Vec<String>,
    pub(crate) region: Option<String>,
    pub(crate) node_name: Option<String>,
    pub(crate) bind_ip: Option<IpAddr>,
    pub(crate) bind_port: Option<u16>,
    pub(crate) listen_all: bool,
    pub(crate) enumerate_host: bool,
    pub(crate) owner_key: Option<PathBuf>,
    pub(crate) owner_required: bool,
    pub(crate) node_label: Option<String>,
    pub(crate) trust_policy: Option<crate::crypto::TrustPolicy>,
    pub(crate) trust_owner: Vec<String>,
    pub(crate) mesh_requirements: crate::plugin::MeshRequirementsConfig,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) log_format: LogFormat,
    pub(crate) headless: bool,
    pub(crate) control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

impl EmbeddedRuntimeOptions {
    pub(super) fn runtime_surface(&self) -> RuntimeSurface {
        match self.mode {
            EmbeddedRuntimeMode::Serve => RuntimeSurface::Serve,
            EmbeddedRuntimeMode::Client => RuntimeSurface::Client,
        }
    }
}

pub(super) fn acquire_instance_runtime(
    options: &RuntimeOptions,
) -> Option<Arc<crate::runtime::instance::InstanceRuntime>> {
    if options.client && !swarm_capture_observer_requested(options) {
        return None;
    }

    match crate::runtime::instance::InstanceRuntime::acquire(std::process::id()) {
        Ok(rt) => Some(Arc::new(rt)),
        Err(err) => {
            tracing::warn!("failed to acquire instance runtime: {err}");
            None
        }
    }
}

pub(super) fn write_runtime_owner_metadata(
    runtime: Option<&Arc<crate::runtime::instance::InstanceRuntime>>,
    console_port: u16,
) {
    let Some(rt) = runtime else {
        return;
    };

    let started_at =
        crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
    let owner_meta = serde_json::json!({
        "pid": std::process::id(),
        "api_port": console_port,
        "version": crate::BUILD_VERSION,
        "started_at_unix": started_at,
        "mesh_llm_binary": std::env::current_exe()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
    });
    let owner_path = rt.dir().join("owner.json");
    if let Ok(json) = serde_json::to_string_pretty(&owner_meta) {
        let _ = crate::runtime::instance::write_text_file_atomic(&owner_path, &json);
    }
}

pub(crate) async fn run() -> Result<()> {
    initialize_runtime_entrypoint()?;
    run_runtime_cli(RuntimeOptions::default(), None, None, None).await
}

pub(crate) async fn run_cli(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_runtime_entrypoint()?;
    run_runtime_cli(options, explicit_surface, legacy_warning, None).await
}

pub(crate) async fn run_embedded_runtime(mut options: EmbeddedRuntimeOptions) -> Result<()> {
    initialize_embedded_runtime_entrypoint()?;

    let surface = options.runtime_surface();
    let control_rx = options.control_rx.take();
    let options = options_from_embedded_options(options);
    run_runtime_cli(options, Some(surface), None, control_rx).await
}

pub(super) fn options_from_embedded_options(embedded: EmbeddedRuntimeOptions) -> RuntimeOptions {
    RuntimeOptions {
        log_format: embedded.log_format,
        client: matches!(embedded.mode, EmbeddedRuntimeMode::Client),
        model: embedded.models.into_iter().map(PathBuf::from).collect(),
        join: embedded.join,
        auto: embedded.auto,
        port: embedded.api_port,
        console: embedded.console_port,
        headless: embedded.headless,
        publish: embedded.publish,
        mesh_name: embedded.mesh_name,
        max_vram: embedded.max_vram_gb,
        mesh_discovery_mode: match embedded.discovery_mode {
            EmbeddedRuntimeDiscoveryMode::Nostr => mesh_discovery::MeshDiscoveryMode::Nostr,
            EmbeddedRuntimeDiscoveryMode::Mdns => mesh_discovery::MeshDiscoveryMode::Mdns,
        },
        relay: embedded.relay,
        relay_auth: embedded.relay_auth,
        disable_iroh_relays: embedded.disable_iroh_relays,
        nostr_relay: embedded.nostr_relay,
        region: embedded.region,
        name: embedded.node_name,
        bind_ip: embedded.bind_ip,
        bind_port: embedded.bind_port,
        listen_all: embedded.listen_all,
        no_enumerate_host: !embedded.enumerate_host,
        owner_key: embedded.owner_key,
        owner_required: embedded.owner_required,
        node_label: embedded.node_label,
        trust_policy: embedded.trust_policy,
        trust_owner: embedded.trust_owner,
        min_node_version: embedded.mesh_requirements.min_node_version,
        max_node_version: embedded.mesh_requirements.max_node_version,
        min_protocol_version: embedded.mesh_requirements.min_protocol_version,
        max_protocol_version: embedded.mesh_requirements.max_protocol_version,
        require_release_attestation: embedded.mesh_requirements.require_release_attestation,
        release_signer_key: embedded.mesh_requirements.release_signer_keys,
        config: embedded.config_path,
        ..RuntimeOptions::default()
    }
}

pub(super) async fn run_runtime_cli(
    mut options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
    embedded_control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
) -> Result<()> {
    options.validate_discovery_mode_args()?;

    if let Some(warning) = legacy_warning {
        let _ = emit_event(OutputEvent::Warning {
            message: warning,
            context: None,
        });
    }

    if let Some(name) = options.plugin.clone() {
        return plugin::run_plugin_process(name).await;
    }

    let checked_updates = autoupdate::maybe_auto_update(autoupdate::AutoUpdateOptions {
        auto_update: options.auto_update,
        plugin_requested: options.plugin.is_some(),
        command_is_update: options.command_is_update,
        llama_flavor: options.llama_flavor,
        current_version: crate::BUILD_VERSION,
    })
    .await?;

    // Finish the release check before startup continues.
    if !checked_updates && !options.command_is_update && !options.command_uses_machine_output {
        autoupdate::check_for_update(crate::BUILD_VERSION).await;
    }

    let mut config = plugin::load_config(options.config.as_deref())?;
    apply_runtime_cli_speculative_overrides(&mut config, options.speculative_overrides.as_ref());
    apply_runtime_config_options(&mut options, &config);
    let startup_mesh_creation_state = resolve_startup_mesh_creation_state(&options, &config)?;
    let cli_has_explicit_models = cli_has_explicit_models(&options);
    let has_config_models = !config.models.is_empty();
    let has_startup_models = cli_has_explicit_models || has_config_models;

    // Acquire the per-instance runtime directory and flock. Plain --client still
    // skips this, but capture observers register so detached runs can be found
    // and stopped by `mesh-llm stop`.
    // Wrap in Arc so it can be cheaply shared with local model tasks.
    let runtime = acquire_instance_runtime(&options);

    // Write owner.json into the runtime dir so sibling-instance discovery can find us.
    write_runtime_owner_metadata(runtime.as_ref(), options.console);

    // Publication intent is now explicit only: --publish gates Nostr discovery.
    // --mesh-name alone never implies publication (Issue #240).

    // Warn users who set --mesh-name without --publish — but only when they
    // are creating a new mesh, not when they are joining one via --discover
    // or --auto (where --mesh-name is just a filter for which mesh to join).
    emit_private_mesh_name_warning(&options);

    // --- Public-to-private identity transition ---
    // If the previous run was public (--auto or --publish) but this run is
    // private, clear the stored identity so the private mesh gets a fresh key
    // that isn't associated with the old public listing.
    handle_public_identity_transition(&options)?;

    let mut auto_join_candidates: Vec<(String, Option<String>)> = Vec::new();
    maybe_discover_join_candidates(&mut options, has_startup_models, &mut auto_join_candidates)
        .await?;
    let Some(PreparedRuntimeStartup {
        startup_models,
        requested_model_names,
        bin_dir,
    }) = prepare_runtime_startup(&options, &config, explicit_surface).await?
    else {
        return Ok(());
    };

    run_auto(RunAutoContext {
        options,
        config,
        startup_mesh_creation_state,
        startup_models,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
        embedded_control_rx,
    })
    .await
}

pub(super) fn apply_runtime_config_options(
    options: &mut RuntimeOptions,
    config: &plugin::MeshConfig,
) {
    options.debug |= config.runtime.debug;
    options.listen_all |= config.runtime.listen_all;
}

pub(in crate::runtime) fn apply_runtime_cli_speculative_overrides(
    config: &mut plugin::MeshConfig,
    overrides: Option<&plugin::SpeculativeConfig>,
) {
    let Some(overrides) = overrides else {
        return;
    };
    let defaults = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.speculative.as_ref())
        .cloned();
    let resolved_defaults =
        plugin::SpeculativeConfig::with_precedence(Some(overrides), None, defaults.as_ref());
    config
        .defaults
        .get_or_insert_with(plugin::ModelConfigDefaults::default)
        .speculative = Some(resolved_defaults);
    for model in &mut config.models {
        model.speculative = Some(plugin::SpeculativeConfig::with_precedence(
            Some(overrides),
            model.speculative.as_ref(),
            defaults.as_ref(),
        ));
    }
}

pub fn load_resolved_plugins(options: &RuntimeOptions) -> Result<plugin::ResolvedPlugins> {
    let config = plugin::load_config(options.config.as_deref())?;
    resolve_plugins_from_config(&config, options)
}

pub(super) fn resolve_plugins_from_config(
    config: &plugin::MeshConfig,
    options: &RuntimeOptions,
) -> Result<plugin::ResolvedPlugins> {
    plugin::resolve_plugins(config, plugin_host_mode(options))
}

pub(super) fn plugin_host_mode(options: &RuntimeOptions) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if options.publish || options.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

pub(super) fn node_display_name(options: &RuntimeOptions, node: &mesh::Node) -> String {
    options
        .name
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("USERNAME").ok())
        .unwrap_or_else(|| node.id().fmt_short().to_string())
}

#[allow(dead_code)]
pub(super) async fn store_benchmark_metrics(
    mem_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp32_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp16_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    result: Option<&benchmark::BenchmarkResult>,
) {
    *mem_arc.lock().await = result.map(|r| r.mem_bandwidth_gbps.clone());
    *fp32_arc.lock().await = result.and_then(|r| r.compute_tflops_fp32.clone());
    *fp16_arc.lock().await = result.and_then(|r| r.compute_tflops_fp16.clone());
}

#[expect(
    clippy::cognitive_complexity,
    reason = "release attestation loading logs missing, valid, and invalid embedded states before advertising the result"
)]
pub(super) async fn attach_local_release_attestation(node: &mesh::Node) -> Result<()> {
    let loaded = match release_attestation::load_for_current_binary() {
        Ok(loaded) => loaded,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to load local embedded release attestation; continuing without advertising one"
            );
            return Ok(());
        }
    };
    node.set_release_attestation_report(loaded.summary.clone(), loaded.attestation.clone())
        .await;
    match loaded.summary.status {
        crate::ReleaseAttestationStatus::Missing => {
            tracing::info!(
                path = %loaded.binary_path.display(),
                "no embedded release attestation found for local binary"
            );
            return Ok(());
        }
        crate::ReleaseAttestationStatus::Valid => {}
        crate::ReleaseAttestationStatus::Invalid => {
            tracing::warn!(
                path = %loaded.binary_path.display(),
                error = %loaded.summary.error.as_deref().unwrap_or("unknown release attestation error"),
                "local binary has an invalid embedded release attestation; continuing without advertising one"
            );
            return Ok(());
        }
    }
    let Some(attestation) = loaded.attestation else {
        tracing::warn!(
            path = %loaded.binary_path.display(),
            "embedded release attestation verified but no release attestation payload was produced"
        );
        return Ok(());
    };
    let attestation_hash = attestation.canonical_hash_hex().ok();
    if loaded.summary.verified {
        tracing::info!(
            path = %loaded.binary_path.display(),
            signer_key_id = %attestation.signer_key_id,
            attestation_hash = attestation_hash.as_deref().unwrap_or("unknown"),
            "loaded local embedded release attestation"
        );
    }
    node.set_release_attestation_report(loaded.summary, Some(attestation))
        .await;
    Ok(())
}

pub(super) fn skippy_telemetry_options(options: &RuntimeOptions) -> skippy::SkippyTelemetryOptions {
    let endpoint = options
        .skippy_metrics_otlp_grpc
        .as_deref()
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(str::to_owned);
    match (endpoint, options.debug) {
        (Some(endpoint), true) => skippy::SkippyTelemetryOptions::debug(Some(endpoint)),
        (Some(endpoint), false) => skippy::SkippyTelemetryOptions::summary(endpoint),
        (None, _) => skippy::SkippyTelemetryOptions::off(),
    }
}

pub(super) fn configure_run_auto_process_state(
    options: &RuntimeOptions,
    runtime: Option<&std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
) {
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("MESH_API_PORT", options.console.to_string()) };

    let verbose_native_debug = options.debug
        && std::env::var("MESH_LLM_DEBUG_NATIVE_VERBOSE")
            .ok()
            .as_deref()
            == Some("1");
    if verbose_native_debug {
        skippy_runtime::enable_verbose_native_logs();
    } else {
        skippy_runtime::disable_verbose_native_logs();
    }

    let native_log_rx = skippy_runtime::register_filtered_native_logs();
    skippy_runtime::set_filtered_native_logs_enabled(true);
    bridge_skippy_native_logs(native_log_rx);
    skippy::configure_materialized_stage_cache();
    configure_skippy_native_logging(runtime.as_ref().map(|runtime| runtime.dir()));
}

pub(super) fn spawn_node_benchmark_task(node: &mesh::Node, bin_dir: &Path) {
    let mem_arc = node.gpu_mem_bandwidth_gbps.clone();
    let compute_fp32_arc = node.gpu_compute_tflops_fp32.clone();
    let compute_fp16_arc = node.gpu_compute_tflops_fp16.clone();
    let bin_dir_clone = bin_dir.to_path_buf();
    let node_bench = node.clone();
    tokio::spawn(async move {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || {
                let hw = hardware::survey();
                if hw.gpu_count == 0 {
                    tracing::debug!("no GPUs detected — skipping memory bandwidth benchmark");
                    return None;
                }
                benchmark::run_or_load(&hw, &bin_dir_clone, benchmark::BENCHMARK_TIMEOUT)
            }),
        )
        .await
        .map_err(|_| {
            tracing::warn!("benchmark timed out after 30s — bandwidth will not be gossiped")
        })
        .ok()
        .and_then(|r| r.ok())
        .flatten();

        if let Some(ref run) = result {
            let total: f64 = run.mem_bandwidth_gbps.iter().sum();
            tracing::info!(
                "Memory bandwidth fingerprint: {} GPUs, {:.1} GB/s total",
                run.mem_bandwidth_gbps.len(),
                total
            );
            for (i, gbps) in run.mem_bandwidth_gbps.iter().enumerate() {
                tracing::debug!("  GPU {}: {:.1} GB/s", i, gbps);
            }
            if let Some(fp32s) = &run.compute_tflops_fp32 {
                let total_fp32: f64 = fp32s.iter().sum();
                tracing::info!(
                    "Compute FP32 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp32s.len(),
                    total_fp32
                );
                for (i, tf) in fp32s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF32", i, tf);
                }
            }
            if let Some(fp16s) = &run.compute_tflops_fp16 {
                let total_fp16: f64 = fp16s.iter().sum();
                tracing::info!(
                    "Compute FP16 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp16s.len(),
                    total_fp16
                );
                for (i, tf) in fp16s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF16", i, tf);
                }
            }
        }
        store_benchmark_metrics(
            mem_arc.clone(),
            compute_fp32_arc.clone(),
            compute_fp16_arc.clone(),
            result.as_ref(),
        )
        .await;
        node_bench.regossip().await;
    });
}

pub(super) async fn start_run_auto_node_and_plugins(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    resolved_plugins: &plugin::ResolvedPlugins,
    swarm_capture: Option<crate::capture::SwarmCaptureRecorder>,
    startup_mesh_creation_state: &StartupMeshCreationState,
) -> Result<(mesh::Node, mesh::TunnelChannels, plugin::PluginManager)> {
    let role = if options.client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    let owner_config = owner_runtime_config(options, config)?;
    if !options.headless && owner_config.keypair.is_none() {
        emit_configuration_ui_read_only_hint();
    }
    let max_vram = if options.client {
        Some(0.0)
    } else {
        options.max_vram
    };
    let relay_auths: std::collections::HashMap<String, String> =
        options.relay_auth.iter().cloned().collect();
    let (node, channels) = mesh::Node::start(
        role,
        mesh::RelayConfig {
            urls: &options.relay,
            auths: &relay_auths,
            policy: relay_policy_for_runtime_options(options),
        },
        mesh::QuicBindSelection {
            ip: effective_quic_bind_ip(options),
            port: options.bind_port,
        },
        max_vram,
        !options.no_enumerate_host,
        Some(owner_config),
        options.config.as_deref(),
        startup_mesh_creation_state.requirements.clone(),
    )
    .await?;
    node.set_swarm_capture_recorder(swarm_capture);
    attach_local_release_attestation(&node).await?;
    node.set_stage_control_sender(skippy::spawn_stage_control_loop(Some(Arc::new(
        node.clone(),
    ))))
    .await;
    node.start_accepting();
    node.set_display_name(node_display_name(options, &node))
        .await;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(resolved_plugins, plugin_host_mode(options), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);
    Ok((node, channels, plugin_manager))
}

pub(super) fn relay_policy_for_runtime_options(options: &RuntimeOptions) -> mesh::RelayPolicy {
    if options.disable_iroh_relays {
        mesh::RelayPolicy::ExplicitlyDisabled
    } else {
        relay_policy_for_mesh_discovery_mode(options.mesh_discovery_mode)
    }
}

pub(super) fn relay_policy_for_mesh_discovery_mode(
    mode: mesh_discovery::MeshDiscoveryMode,
) -> mesh::RelayPolicy {
    match mode {
        mesh_discovery::MeshDiscoveryMode::Nostr => mesh::RelayPolicy::DefaultPublic,
        mesh_discovery::MeshDiscoveryMode::Mdns => mesh::RelayPolicy::Disabled,
    }
}

pub(super) fn runtime_resource_planning_profile(
    options: &RuntimeOptions,
) -> RuntimeResourcePlanningProfile {
    if options.auto || options.publish || options.discover.is_some() || !options.join.is_empty() {
        RuntimeResourcePlanningProfile::SharedMesh
    } else {
        RuntimeResourcePlanningProfile::DedicatedLocal
    }
}

pub(super) fn runtime_model_ctx_size_override(
    options: &RuntimeOptions,
    model_overrides: Option<&plugin::ModelConfigEntry>,
) -> Option<u32> {
    options
        .ctx_size
        .or_else(|| model_overrides.and_then(|model| model.ctx_size))
}

pub(super) fn should_start_relay_health_monitor(mode: mesh_discovery::MeshDiscoveryMode) -> bool {
    matches!(
        relay_policy_for_mesh_discovery_mode(mode),
        mesh::RelayPolicy::DefaultPublic
    )
}

pub(super) fn should_start_lan_rediscovery(
    mode: mesh_discovery::MeshDiscoveryMode,
    join_tokens: &[String],
) -> bool {
    mode == mesh_discovery::MeshDiscoveryMode::Mdns
        && join_tokens.iter().any(|token| !token.trim().is_empty())
}

pub(super) fn start_relay_health_monitor_for_discovery_mode(
    node: &mesh::Node,
    mode: mesh_discovery::MeshDiscoveryMode,
) {
    if should_start_relay_health_monitor(mode) {
        node.start_relay_health_monitor();
    } else {
        tracing::debug!("Relay health monitor disabled for LAN-only mesh discovery");
    }
}

pub(super) fn run_auto_survey_hardware(is_client: bool) -> hardware::HardwareSurvey {
    if is_client {
        hardware::HardwareSurvey::default()
    } else {
        hardware::query(&[
            hardware::Metric::GpuName,
            hardware::Metric::GpuCount,
            hardware::Metric::IsSoc,
            hardware::Metric::GpuFacts,
        ])
    }
}

pub(super) async fn build_run_auto_node_setup(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    resolved_plugins: &plugin::ResolvedPlugins,
    bin_dir: &Path,
    swarm_capture: Option<crate::capture::SwarmCaptureRecorder>,
    startup_mesh_creation_state: &StartupMeshCreationState,
) -> Result<AutoRuntimeNodeSetup> {
    let console_port = Some(options.console);
    let is_client = options.client;
    let skippy_telemetry = skippy_telemetry_options(options);
    let local_models = if is_client {
        vec![]
    } else {
        models::scan_local_models()
    };
    tracing::info!("Local models on disk: {:?}", local_models);
    let (node, channels, plugin_manager) = start_run_auto_node_and_plugins(
        options,
        config,
        resolved_plugins,
        swarm_capture,
        startup_mesh_creation_state,
    )
    .await?;
    let survey_hardware = run_auto_survey_hardware(is_client);
    let survey_telemetry = survey::SurveyTelemetry::start(
        config,
        survey_hardware,
        survey::SurveyTelemetrySource {
            node_id: node.id().fmt_short().to_string(),
            node_role: if is_client { "client" } else { "worker" }.into(),
        },
    );
    node.set_routing_telemetry_sink(survey_telemetry.routing_sink());
    node.set_available_models(local_models.clone()).await;
    node.start_heartbeat();
    node.start_rtt_refresh();
    node.start_direct_path_maintenance();
    start_relay_health_monitor_for_discovery_mode(&node, options.mesh_discovery_mode);
    let lan_bootstrap_tasks = spawn_mdns_reverse_dial(options, &node);

    if !is_client {
        spawn_node_benchmark_task(&node, bin_dir);
    } else {
        tracing::debug!("client node — skipping memory bandwidth benchmark");
    }

    Ok(AutoRuntimeNodeSetup {
        is_client,
        console_port,
        skippy_telemetry,
        local_models,
        node,
        channels,
        plugin_manager,
        survey_telemetry,
        lan_bootstrap_tasks,
    })
}

pub(super) async fn run_auto_start_new_mesh(
    options: &RuntimeOptions,
    node: &mesh::Node,
) -> Result<()> {
    let nostr_pubkey = if options.publish
        && options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
    {
        nostr::load_or_create_keys()
            .ok()
            .map(|k| k.public_key().to_hex())
    } else {
        None
    };
    let mesh_id = node
        .initialize_mesh_identity_as_originator(
            options.mesh_name.as_deref(),
            nostr_pubkey.as_deref(),
        )
        .await?;
    record_first_joined_mesh_ts(node).await;
    mesh::save_last_mesh_id(&mesh_id)?;
    tracing::info!("Mesh ID: {mesh_id}");
    let _ = emit_event(OutputEvent::InviteToken {
        token: node.invite_token().await,
        mesh_id: mesh_id.clone(),
        mesh_name: options.mesh_name.clone(),
    });
    let _ = emit_event(OutputEvent::WaitingForPeers { detail: None });

    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
    {
        let rediscover_node = node.clone();
        let rediscover_relays = nostr_relays(&options.nostr_relay);
        let rediscover_relay_urls = options.relay.clone();
        let rediscover_mesh_name = options.mesh_name.clone();
        tokio::spawn(Box::pin(nostr_rediscovery(
            rediscover_node,
            rediscover_relays,
            rediscover_relay_urls,
            rediscover_mesh_name,
        )));
    }

    Ok(())
}

/// Returns true if `run_auto` should spawn the bootstrap proxy.
///
/// The bootstrap proxy binds the API port and tunnels OpenAI requests to
/// whichever mesh peer can serve them, so the local API stays usable while
/// this node's GPU loads its model.
///
/// Historically this gated solely on `options.join` being non-empty, which worked
/// because both `--client --auto` and `serve --auto` pushed their discovered
/// token into `options.join`. Commit 1bd62389 changed the serve path to stage
/// candidates in `auto_join_candidates` instead, leaving `options.join` empty and
/// silently disabling the bootstrap proxy for `serve --auto`. Accepting either
/// signal restores the original contract without changing any other path:
///
/// - `--join <token>` (any mode): `options.join` non-empty → fires (unchanged).
/// - `--client --auto` with discovery hit: `options.join` populated by
///   `handle_auto_decision` → fires (unchanged).
/// - `serve --auto` with discovery hit: `auto_join_candidates` non-empty,
///   `options.join` empty → **now fires** (the fix).
/// - Anything with no candidates and no join token (bare `mesh-llm`, bare
///   `--client`, `--auto` with zero discovery results): both empty → does
///   not fire (unchanged — there is nowhere to tunnel to).
pub(super) async fn advertise_run_auto_models(
    node: &mesh::Node,
    startup_models: &[StartupModelPlan],
    model_name: &str,
    model_source: String,
) {
    node.set_model_source(model_source).await;
    let all_declared = build_serving_list(startup_models, model_name);
    node.set_serving_models(all_declared.clone()).await;
    node.set_hosted_models(Vec::new()).await;
    node.set_models(all_declared).await;
    node.regossip().await;
}

pub(super) struct RunAutoRuntimeLoopContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) primary_model_name: &'a str,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_models: &'a mut HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: &'a mut HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) next_runtime_instance_sequence: &'a mut u64,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) openai_guardrail_policy: &'a OpenAiGuardrailPolicyHandle,
    pub(super) model_target_reconciliation_policy: ModelTargetReconciliationPolicy,
    pub(super) model_target_reconciliation_state: ModelTargetReconciliationState,
}

pub(super) struct RunAutoRuntimeState {
    pub(super) runtime_models: HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: HashMap<String, ManagedModelController>,
    pub(super) runtime_instance_registry: RuntimeInstanceRegistry,
    pub(super) runtime_capacity_ledger: RuntimeCapacityLedger,
    pub(super) next_runtime_instance_sequence: u64,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: DashboardContextUsage,
    pub(super) input_handler_enabled: bool,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
}

pub(super) struct RunAutoStartupTasksContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) tunnel_mgr: &'a tunnel::Manager,
    pub(super) startup_models: &'a [StartupModelPlan],
    pub(super) primary_startup_model: Option<&'a StartupModelPlan>,
    pub(super) model_name: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) api_ready_url: String,
    pub(super) ready_console_url: Option<String>,
    pub(super) ready_api_port: u16,
    pub(super) ready_console_port: Option<u16>,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) runtime_state: &'a mut RunAutoRuntimeState,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) api_port: u16,
    pub(super) interactive_started: Arc<AtomicBool>,
}

pub(super) fn initialize_run_auto_runtime_state(options: &RuntimeOptions) -> RunAutoRuntimeState {
    RunAutoRuntimeState {
        runtime_models: HashMap::new(),
        runtime_survey_models: HashMap::new(),
        managed_models: HashMap::new(),
        runtime_instance_registry: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        runtime_capacity_ledger: RuntimeCapacityLedger::default(),
        next_runtime_instance_sequence: 1_u64,
        dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        dashboard_context_usage: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        input_handler_enabled: output_sink()
            .and_then(|sink| sink.console_session_mode())
            .is_some(),
        openai_guardrail_policy: openai_guardrail_policy_handle(mesh_guardrail_mode_to_openai(
            options.mesh_guardrails,
        )),
    }
}

pub(super) async fn spawn_run_auto_startup_model_tasks(
    ctx: RunAutoStartupTasksContext<'_>,
) -> StartupReadyReporter {
    let RunAutoStartupTasksContext {
        options,
        config,
        node,
        tunnel_mgr,
        startup_models,
        primary_startup_model,
        model_name,
        model_path,
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
        target_tx,
        runtime_state,
        console_state,
        control_tx,
        survey_telemetry,
        skippy_telemetry,
        api_port,
        interactive_started,
    } = ctx;

    let startup_model_names: Vec<String> = startup_models
        .iter()
        .map(|model| model.declared_ref.clone())
        .collect();
    let startup_ready_reporter = StartupReadyReporter::new(
        &startup_model_names,
        model_name.to_string(),
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
    );
    let startup_load_gate = Arc::new(tokio::sync::Mutex::new(()));
    let primary_parallel_override = super::startup_models::resolve_model_parallel_override(
        primary_startup_model.and_then(|m| m.parallel),
        &config.gpu,
    );
    let resource_planning_profile = runtime_resource_planning_profile(options);
    let console_state_for_election = console_state.cloned();
    let interactive_console_state = console_state.cloned();
    let primary_mmproj = primary_startup_model.and_then(|model| model.mmproj_path.clone());
    let primary_ctx_size = primary_startup_model.and_then(|model| model.ctx_size);
    let primary_pinned_gpu = primary_startup_model.and_then(|model| model.pinned_gpu.clone());
    let primary_cache_type_k = primary_startup_model.and_then(|model| model.cache_type_k.clone());
    let primary_cache_type_v = primary_startup_model.and_then(|model| model.cache_type_v.clone());
    let primary_n_batch = primary_startup_model.and_then(|model| model.n_batch);
    let primary_n_ubatch = primary_startup_model.and_then(|model| model.n_ubatch);
    let primary_flash_attention = primary_startup_model
        .map(|model| model.flash_attention)
        .unwrap_or(FlashAttentionType::Auto);
    let primary_model_ref = primary_startup_model
        .map(|model| model.declared_ref.clone())
        .unwrap_or_else(|| model_name.to_string());
    let (primary_stop_tx, primary_stop_rx) = tokio::sync::watch::channel(false);
    let primary_instance_id =
        next_runtime_instance_id(&mut runtime_state.next_runtime_instance_sequence);
    let primary_task = tokio::spawn(Box::pin(startup_local_model_loop(StartupLocalModelTask {
        node: node.clone(),
        config: config.clone(),
        tunnel_mgr: tunnel_mgr.clone(),
        target_tx: target_tx.clone(),
        model_path: model_path.to_path_buf(),
        model_ref: primary_model_ref,
        model_name: model_name.to_string(),
        instance_id: primary_instance_id.clone(),
        primary_model_name: model_name.to_string(),
        mmproj_path: primary_mmproj,
        ctx_size: primary_ctx_size,
        pinned_gpu: primary_pinned_gpu,
        runtime_capacity_ledger: runtime_state.runtime_capacity_ledger.clone(),
        cache_type_k: primary_cache_type_k,
        cache_type_v: primary_cache_type_v,
        n_batch: primary_n_batch,
        n_ubatch: primary_n_ubatch,
        flash_attention: primary_flash_attention,
        parallel_override: primary_parallel_override,
        resource_planning_profile,
        openai_guardrail_policy: runtime_state.openai_guardrail_policy.clone(),
        split: options.split,
        skippy_telemetry: skippy_telemetry.clone(),
        survey_telemetry: survey_telemetry.clone(),
        survey_launch_kind: survey::SurveyLaunchKind::Startup,
        stop_rx: primary_stop_rx,
        dashboard_processes: runtime_state.dashboard_processes.clone(),
        dashboard_context_usage: runtime_state.dashboard_context_usage.clone(),
        runtime_instance_registry: runtime_state.runtime_instance_registry.clone(),
        console_state: console_state_for_election,
        api_port,
        startup_ready_reporter: startup_ready_reporter.clone(),
        startup_load_gate: startup_load_gate.clone(),
        input_handler_enabled: runtime_state.input_handler_enabled,
        interactive_started,
        interactive_control_tx: control_tx.clone(),
        interactive_console_state,
    })));
    runtime_state.managed_models.insert(
        primary_instance_id,
        ManagedModelController {
            model_name: model_name.to_string(),
            stop_tx: primary_stop_tx,
            task: primary_task,
        },
    );

    spawn_run_auto_additional_model_tasks(RunAutoAdditionalModelsContext {
        options,
        config,
        node,
        tunnel_mgr,
        startup_models,
        primary_model_name: model_name,
        target_tx,
        managed_models: &mut runtime_state.managed_models,
        next_runtime_instance_sequence: &mut runtime_state.next_runtime_instance_sequence,
        dashboard_processes: &runtime_state.dashboard_processes,
        dashboard_context_usage: &runtime_state.dashboard_context_usage,
        runtime_instance_registry: &runtime_state.runtime_instance_registry,
        runtime_capacity_ledger: &runtime_state.runtime_capacity_ledger,
        console_state,
        startup_ready_reporter: &startup_ready_reporter,
        startup_load_gate: &startup_load_gate,
        control_tx,
        survey_telemetry,
        skippy_telemetry,
        openai_guardrail_policy: &runtime_state.openai_guardrail_policy,
    })
    .await;

    startup_ready_reporter
}

pub(super) fn configure_swarm_capture(
    options: &RuntimeOptions,
) -> Result<Option<crate::capture::SwarmCaptureRecorder>> {
    let recorder =
        crate::capture::SwarmCaptureRecorder::from_cli_or_env(options.swarm_capture.as_deref())?;
    if let Some(recorder) = recorder.as_ref() {
        tracing::info!(
            path = %recorder.path().display(),
            "passive swarm capture enabled; writing local debug capture JSONL"
        );
    }
    Ok(recorder)
}

pub(super) struct RunAutoModelSelectionContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) startup_models: &'a [StartupModelPlan],
    pub(super) local_models: &'a [String],
    pub(super) is_client: bool,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) bootstrap_listener_tx: &'a mut Option<BootstrapProxyStopTx>,
    pub(super) primary_startup_model: Option<&'a StartupModelPlan>,
    pub(super) embedded_control_rx:
        &'a mut Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

pub(super) async fn select_advertised_run_auto_model(
    mut ctx: RunAutoModelSelectionContext<'_>,
) -> Result<Option<(PathBuf, String)>> {
    let Some(model) = run_auto_model_path_or_shutdown(&mut ctx).await? else {
        return Ok(None);
    };

    let (model_name, model_source) = run_auto_model_identity(ctx.primary_startup_model, &model);
    advertise_run_auto_models(ctx.node, ctx.startup_models, &model_name, model_source).await;
    Ok(Some((model, model_name)))
}

/// Serve mode: join the mesh and serve local models through the embedded runtime.
pub(super) struct RunAutoContext {
    pub(super) options: RuntimeOptions,
    pub(super) config: plugin::MeshConfig,
    pub(super) startup_mesh_creation_state: StartupMeshCreationState,
    pub(super) startup_models: Vec<StartupModelPlan>,
    pub(super) requested_model_names: Vec<String>,
    pub(super) bin_dir: PathBuf,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
    pub(super) auto_join_candidates: Vec<(String, Option<String>)>,
    pub(super) embedded_control_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

#[expect(
    clippy::cognitive_complexity,
    reason = "run_auto is the top-level runtime orchestration path and preserves startup/shutdown ordering"
)]
pub(super) async fn run_auto(ctx: RunAutoContext) -> Result<()> {
    let RunAutoContext {
        mut options,
        config,
        startup_mesh_creation_state,
        startup_models,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
        mut embedded_control_rx,
    } = ctx;
    let resolved_plugins = resolve_plugins_from_config(&config, &options)?;
    let swarm_capture = configure_swarm_capture(&options)?;
    tracing::debug!(
        mesh_requirements = ?runtime_startup_requirements(&startup_mesh_creation_state),
        "loaded creation-time mesh requirements into runtime startup state"
    );
    let api_port = options.port;
    configure_run_auto_process_state(&options, runtime.as_ref());
    let _native_log_forwarding = SkippyNativeLogForwardingGuard;
    // Embedded native logs are process-global and are redirected to the runtime log
    // file before model load. We also forward the filtered, aggregated model-loading
    // summaries through OutputEvent/JSONL so structured startup progress remains visible
    // without streaming every raw native line through the dashboard.
    let AutoRuntimeNodeSetup {
        is_client,
        console_port,
        skippy_telemetry,
        local_models,
        node,
        channels,
        plugin_manager,
        survey_telemetry,
        lan_bootstrap_tasks,
    } = build_run_auto_node_setup(
        &options,
        &config,
        &resolved_plugins,
        &bin_dir,
        swarm_capture,
        &startup_mesh_creation_state,
    )
    .await?;

    // Advertise what we have on disk and what we want the mesh to serve
    node.set_requested_models(requested_model_names.clone())
        .await;

    run_auto_join_mesh_phase(&mut options, &node, &auto_join_candidates).await?;

    let affinity_router = affinity::AffinityRouter::new();

    // Start bootstrap proxy if we have somewhere to tunnel to. This gives
    // instant API access via tunnel while our GPU loads.
    let mut bootstrap_listener_tx = start_run_auto_bootstrap_proxy(
        &options,
        &node,
        api_port,
        &affinity_router,
        &auto_join_candidates,
    );

    let primary_startup_model = startup_models.first().cloned();

    let Some((model, model_name)) =
        select_advertised_run_auto_model(RunAutoModelSelectionContext {
            options: &options,
            node: &node,
            startup_models: &startup_models,
            local_models: &local_models,
            is_client,
            plugin_manager: &plugin_manager,
            bootstrap_listener_tx: &mut bootstrap_listener_tx,
            primary_startup_model: primary_startup_model.as_ref(),
            embedded_control_rx: &mut embedded_control_rx,
        })
        .await?
    else {
        return Ok(());
    };

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), channels.rpc, channels.http, channels.stage).await?;

    // Election publishes per-model targets
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

    // Runtime control for local load/unload of extra models.
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    spawn_embedded_runtime_control_forwarder(embedded_control_rx.take(), control_tx.clone());
    let (runtime_event_tx, mut runtime_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
    let mut runtime_state = initialize_run_auto_runtime_state(&options);

    let model_name_for_console = model_name.clone();
    let runtime_owner_key_path = resolve_runtime_owner_key_path(&options)?;
    let console_state = setup_run_auto_console_state(RunAutoConsoleStateContext {
        options: &options,
        node: &node,
        console_enabled: console_port.is_some(),
        model_name: &model_name_for_console,
        model_path: &model,
        api_port,
        plugin_manager: &plugin_manager,
        affinity_router: &affinity_router,
        control_tx: &control_tx,
        owner_key_path: &runtime_owner_key_path,
    })
    .await?;
    publish_initial_openai_guardrails_status(
        console_state.as_ref(),
        &runtime_state.openai_guardrail_policy,
    )
    .await;

    if let Some(sink) = output_sink() {
        sink.register_dashboard_snapshot_provider(Arc::new(RuntimeDashboardSnapshotProvider::new(
            node.clone(),
            runtime_state.dashboard_processes.clone(),
            runtime_state.dashboard_context_usage.clone(),
            Some(plugin_manager.clone()),
            api_port,
            console_port,
            options.headless,
        )));
    }

    let _ = emit_event(OutputEvent::LaunchPlan {
        plan: startup_launch_plan(
            &startup_models,
            &model_name,
            api_port,
            console_port,
            options.headless,
            config.gpu.parallel,
            startup_default_backend_device(options.llama_flavor),
        ),
    });

    let interactive_started = Arc::new(AtomicBool::new(false));
    let RunAutoServingSurface {
        api_proxy_handle,
        console_server_handle,
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
    } = setup_run_auto_serving_surface(RunAutoServingSurfaceContext {
        options: &options,
        node: &node,
        api_port,
        console_port,
        is_client,
        target_rx: &target_rx,
        control_tx: &control_tx,
        affinity_router: &affinity_router,
        bootstrap_listener_tx,
        input_handler_enabled: runtime_state.input_handler_enabled,
        interactive_started: &interactive_started,
        console_state: console_state.as_ref(),
        model_name_for_console: &model_name_for_console,
    })
    .await?;

    tracing::info!("Starting embedded runtime for model: {model_name}");
    let startup_ready_reporter = spawn_run_auto_startup_model_tasks(RunAutoStartupTasksContext {
        options: &options,
        config: &config,
        node: &node,
        tunnel_mgr: &tunnel_mgr,
        startup_models: &startup_models,
        primary_startup_model: primary_startup_model.as_ref(),
        model_name: &model_name,
        model_path: &model,
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
        target_tx: &target_tx,
        runtime_state: &mut runtime_state,
        console_state: console_state.as_ref(),
        control_tx: &control_tx,
        survey_telemetry: &survey_telemetry,
        skippy_telemetry: &skippy_telemetry,
        api_port,
        interactive_started,
    })
    .await;

    // Discovery publish loop (if --publish) or Nostr watchdog (if --auto, to take over if publisher dies).
    let discovery_publisher =
        spawn_run_auto_discovery_publisher(&options, &node, console_state.as_ref()).await;

    let runtime_data_producer = runtime_data_producer_for_console(console_state.as_ref()).await;
    run_auto_runtime_loop_and_shutdown(RunAutoRuntimeLifecycleContext {
        options: &options,
        config: &config,
        node: &node,
        primary_model_name: &model_name,
        target_tx: &target_tx,
        control_rx: &mut control_rx,
        control_tx: &control_tx,
        runtime_event_rx: &mut runtime_event_rx,
        runtime_state: &mut runtime_state,
        console_state: console_state.as_ref(),
        runtime_data_producer: runtime_data_producer.as_ref(),
        runtime_event_tx: &runtime_event_tx,
        survey_telemetry: &survey_telemetry,
        startup_ready_reporter: &startup_ready_reporter,
        plugin_manager: &plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        lan_bootstrap_tasks,
        runtime,
    })
    .await;
    Ok(())
}
