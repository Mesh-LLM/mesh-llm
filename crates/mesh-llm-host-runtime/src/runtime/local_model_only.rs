use super::{
    LocalOpenAiModelStartSpec, RuntimeOptions, RuntimeResourcePlanningProfile,
    SkippyNativeLogForwardingGuard, acquire_instance_runtime,
    apply_runtime_cli_speculative_overrides, apply_runtime_config_options,
    build_startup_model_specs, cleanup_run_auto_runtime_dir, configure_run_auto_process_state,
    emit_shutdown, openai_guardrail_policy_handle, preflight_config_owned_startup_models,
    resolve_local_model_only_startup_models, runtime_model_required_bytes,
    skippy_telemetry_options, start_local_openai_model, wait_shutdown_signal,
};
use crate::inference::election;
use crate::plugin;
use crate::runtime::survey;
use crate::system::hardware;
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_server::EmbeddedState;
use skippy_server::serving_hooks::SharedModelServingHooksFactory;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

const OPENAI_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const OPENAI_STATUS_POLL_INTERVAL: Duration = Duration::from_millis(25);

pub(super) fn validate_local_model_only_options(options: &RuntimeOptions) -> Result<()> {
    anyhow::ensure!(!options.client, "--local-model-only cannot run as a client");
    anyhow::ensure!(
        !options.auto && options.discover.is_none() && options.join.is_empty(),
        "--local-model-only cannot discover or join a mesh"
    );
    anyhow::ensure!(
        !options.publish
            && options.mesh_name.is_none()
            && options.region.is_none()
            && options.name.is_none(),
        "--local-model-only cannot publish or describe a mesh"
    );
    anyhow::ensure!(
        !options.split && options.split_topology_lock.is_none() && options.tensor_split.is_none(),
        "--local-model-only does not support split serving"
    );
    anyhow::ensure!(
        options.relay.is_empty()
            && options.relay_auth.is_empty()
            && options.nostr_relay.is_empty()
            && !options.disable_iroh_relays
            && options.bind_ip.is_none()
            && options.bind_port.is_none()
            && options.max_clients.is_none(),
        "--local-model-only does not accept mesh transport options"
    );
    anyhow::ensure!(
        options.min_node_version.is_none()
            && options.max_node_version.is_none()
            && options.min_protocol_version.is_none()
            && options.max_protocol_version.is_none()
            && !options.require_release_attestation
            && options.release_signer_key.is_empty(),
        "--local-model-only does not accept mesh admission options"
    );
    anyhow::ensure!(
        options.owner_key.is_none()
            && options.control_bind.is_none()
            && options.control_advertise_addr.is_none()
            && !options.owner_required
            && options.node_label.is_none()
            && options.trust_policy.is_none()
            && options.trust_owner.is_empty(),
        "--local-model-only does not start owner control or management APIs"
    );
    anyhow::ensure!(
        options.plugin.is_none() && options.swarm_capture.is_none(),
        "--local-model-only does not start plugins or swarm capture"
    );
    anyhow::ensure!(
        !options.auto_update,
        "--local-model-only does not perform release updates"
    );
    anyhow::ensure!(
        !options.headless,
        "--local-model-only never starts a console; remove --headless"
    );
    if let Some(max_vram) = options.max_vram {
        anyhow::ensure!(
            max_vram.is_finite() && max_vram > 0.0,
            "--max-vram must be a finite positive number"
        );
    }
    match options.native_serving_plugin.as_ref() {
        Some(_) => {
            anyhow::ensure!(
                options.native_serving_plugin_config.is_some()
                    && options.native_serving_plugin_state.is_some()
                    && options.native_serving_plugin_deadline_ms.is_some(),
                "--native-serving-plugin requires config, state, and deadline options"
            );
        }
        None => {
            anyhow::ensure!(
                options.native_serving_plugin_config.is_none()
                    && options.native_serving_plugin_state.is_none()
                    && options.native_serving_plugin_deadline_ms.is_none(),
                "native serving plugin config, state, and deadline require --native-serving-plugin"
            );
        }
    }
    Ok(())
}

pub(super) async fn run_local_model_only(mut options: RuntimeOptions) -> Result<()> {
    validate_local_model_only_options(&options)?;
    let serving_hooks_factory = native_serving_plugin_factory(&options)?;
    let mut config = plugin::load_config(options.config.as_deref())?;
    apply_runtime_cli_speculative_overrides(&mut config, options.speculative_overrides.as_ref());
    apply_runtime_config_options(&mut options, &config);

    let startup_specs = build_startup_model_specs(&options, &config)?;
    anyhow::ensure!(
        startup_specs.len() == 1,
        "--local-model-only requires exactly one startup model"
    );
    let mut startup_models = resolve_local_model_only_startup_models(&startup_specs).await?;
    preflight_config_owned_startup_models(
        &config,
        &startup_specs,
        &mut startup_models,
        options.llama_flavor,
        None,
    )?;
    let model = startup_models
        .pop()
        .context("local model resolution produced no startup model")?;
    anyhow::ensure!(
        model.resolved_path.is_file(),
        "--local-model-only requires one complete local model file: {}",
        model.resolved_path.display()
    );

    let model_bytes = election::total_model_bytes(&model.resolved_path);
    anyhow::ensure!(
        model_bytes > 0,
        "could not determine local model size: {}",
        model.resolved_path.display()
    );
    let local_capacity_bytes = local_capacity_bytes(&options, model.pinned_gpu.as_ref());
    let required_bytes = runtime_model_required_bytes(model_bytes);
    anyhow::ensure!(
        local_capacity_bytes >= required_bytes,
        "local model requires {:.2} GB but this process has {:.2} GB; local model-only serving never falls back to a split",
        required_bytes as f64 / 1e9,
        local_capacity_bytes as f64 / 1e9
    );

    let bind_addr = local_openai_bind_addr(&options);
    let runtime = acquire_instance_runtime(&options);
    configure_run_auto_process_state(&options, runtime.as_ref());
    let _native_log_forwarding = SkippyNativeLogForwardingGuard;

    let model_name = model.declared_ref.clone();
    let survey_telemetry = survey::SurveyTelemetry::start(
        &config,
        hardware::survey(),
        survey::SurveyTelemetrySource {
            node_id: "local-model-only".into(),
            node_role: "worker".into(),
        },
    );
    let launch = LocalOpenAiModelStartSpec {
        mesh_config: &config,
        config_model_id: model.config_model_id.as_deref(),
        model_path: &model.resolved_path,
        model_bytes,
        mmproj_override: model.mmproj_path.as_deref(),
        ctx_size_override: model.ctx_size,
        pinned_gpu: model.pinned_gpu.as_ref(),
        capacity_budget_bytes: local_capacity_bytes,
        cache_type_k_override: model.cache_type_k.as_deref(),
        cache_type_v_override: model.cache_type_v.as_deref(),
        n_batch_override: model.n_batch,
        n_ubatch_override: model.n_ubatch,
        flash_attention_override: model.flash_attention,
        parallel_override: model.parallel,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            super::status::mesh_guardrail_mode_to_openai(options.mesh_guardrails),
        ),
        skippy_telemetry: skippy_telemetry_options(&options),
        survey_telemetry,
        hook_policy: None,
        serving_hooks_factory,
        http_bind_addr: bind_addr,
    };

    let result = run_loaded_local_model(launch, &model_name, bind_addr).await;
    cleanup_run_auto_runtime_dir(runtime);
    result
}

fn native_serving_plugin_factory(
    options: &RuntimeOptions,
) -> Result<Option<SharedModelServingHooksFactory>> {
    let Some(library_path) = options.native_serving_plugin.as_deref() else {
        return Ok(None);
    };
    let config_path = options
        .native_serving_plugin_config
        .clone()
        .context("--native-serving-plugin-config is required")?;
    let state_directory = options
        .native_serving_plugin_state
        .clone()
        .context("--native-serving-plugin-state is required")?;
    let deadline_ms = options
        .native_serving_plugin_deadline_ms
        .context("--native-serving-plugin-deadline-ms is required")?;
    anyhow::ensure!(
        deadline_ms > 0,
        "--native-serving-plugin-deadline-ms must be greater than zero"
    );
    let factory = mesh_native_serving_plugin_host::NativeServingPluginFactory::load(
        library_path,
        config_path,
        state_directory,
        Duration::from_millis(deadline_ms),
    )?;
    Ok(Some(std::sync::Arc::new(factory)))
}

fn local_capacity_bytes(
    options: &RuntimeOptions,
    pinned_gpu: Option<&super::StartupPinnedGpuTarget>,
) -> u64 {
    let detected = pinned_gpu
        .map(super::StartupPinnedGpuTarget::allocatable_vram_bytes)
        .unwrap_or_else(|| hardware::survey().vram_bytes);
    options
        .max_vram
        .map(|gb| (gb * 1e9) as u64)
        .map_or(detected, |cap| detected.min(cap))
}

fn local_openai_bind_addr(options: &RuntimeOptions) -> SocketAddr {
    let ip = if options.listen_all {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    } else {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    };
    SocketAddr::new(ip, options.port)
}

async fn run_loaded_local_model(
    launch: LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    bind_addr: SocketAddr,
) -> Result<()> {
    let (_, model, _death_rx) = start_local_openai_model(launch, model_name).await?;
    if let Err(error) = wait_for_openai_ready(&model, bind_addr).await {
        model.shutdown().await;
        return Err(error);
    }

    let ready_url = format!("http://{}:{}/v1", connect_ip(bind_addr), bind_addr.port());
    let _ = emit_event(OutputEvent::ApiReady {
        url: ready_url.clone(),
    });
    let _ = emit_event(OutputEvent::RuntimeReady {
        api_url: ready_url,
        console_url: None,
        api_port: bind_addr.port(),
        console_port: None,
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    });

    let outcome = wait_for_openai_exit_or_shutdown(&model).await;
    let reason = outcome
        .as_ref()
        .map_or_else(|error| error.to_string(), |signal| signal.to_string());
    emit_shutdown(Some(reason)).await;
    model.shutdown().await;
    outcome.map(|_| ())
}

async fn wait_for_openai_ready(
    model: &super::LocalRuntimeModelHandle,
    bind_addr: SocketAddr,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + OPENAI_STARTUP_TIMEOUT;
    loop {
        let status = model.openai_server_status();
        if status.state == EmbeddedState::Failed {
            anyhow::bail!(
                "local OpenAI API failed during startup: {}",
                status.last_error.as_deref().unwrap_or("unknown error")
            );
        }
        if status.state == EmbeddedState::Ready && status.bind_addr == bind_addr {
            return Ok(());
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "local OpenAI API did not bind {bind_addr}"
        );
        tokio::time::sleep(OPENAI_STATUS_POLL_INTERVAL).await;
    }
}

async fn wait_for_openai_exit_or_shutdown(
    model: &super::LocalRuntimeModelHandle,
) -> Result<&'static str> {
    let mut interval = tokio::time::interval(OPENAI_STATUS_POLL_INTERVAL);
    loop {
        tokio::select! {
            signal = wait_shutdown_signal() => return Ok(signal),
            _ = interval.tick() => {
                let status = model.openai_server_status();
                match status.state {
                    EmbeddedState::Failed => anyhow::bail!(
                        "local OpenAI API stopped: {}",
                        status.last_error.as_deref().unwrap_or("unknown error")
                    ),
                    EmbeddedState::Stopped => anyhow::bail!("local OpenAI API stopped unexpectedly"),
                    EmbeddedState::Starting | EmbeddedState::Ready | EmbeddedState::Stopping => {}
                }
            }
        }
    }
}

fn connect_ip(bind_addr: SocketAddr) -> IpAddr {
    if bind_addr.ip().is_unspecified() {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        bind_addr.ip()
    }
}
