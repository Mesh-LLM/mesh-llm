use super::status::current_time_unix_ms;
use super::status::single_quote_shell_arg;
use super::{
    DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL, DashboardContextUsage, InitialPromptMode,
    LocalRuntimeModelHandle, LocalRuntimeModelStartSpec, OpenAiGuardrailPolicyHandle,
    RuntimeCapacityLedger, RuntimeCapacityReservation, RuntimeInstanceRegistry,
    RuntimeResourcePlanningProfile, SPLIT_STANDBY_RETRY_INTERVAL, SplitCoordinatorAck,
    SplitCoordinatorEvent, SplitRuntimeReason, SplitRuntimeStart, StartupModelPlan,
    StartupPinnedGpuTarget, StartupRuntimePlan, add_runtime_local_target, local_process_payload,
    publish_runtime_llama_slots, publish_runtime_llama_unavailable,
    refresh_dashboard_context_usage, register_runtime_instance, remove_dashboard_context_usage,
    remove_dashboard_process, remove_runtime_local_target, reserve_runtime_capacity_for_model,
    runtime_model_planning_bytes, runtime_model_required_bytes,
    runtime_process_payload_with_status, start_runtime_local_model, start_runtime_split_model,
    startup_runtime_plan, stop_split_generation_cleanup, unregister_runtime_instance,
    update_pi_models_json, upsert_dashboard_process,
};
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::network::tunnel;
use crate::plugin;
use crate::runtime::interactive;
use crate::runtime::local;
use crate::runtime::survey;
use anyhow::Context;
use mesh_llm_events::{OutputEvent, emit_event, output_sink, schedule_ready_prompt};
use skippy_protocol::FlashAttentionType;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

pub(super) type BootstrapProxyStopTx =
    tokio::sync::mpsc::Sender<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>;

pub(super) struct StartupLaunchHandles {
    pub(super) loaded_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) split_cleanup: Option<local::SplitGenerationCleanup>,
    pub(super) split_event_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) coordinator_task: Option<tokio::task::JoinHandle<()>>,
    pub(super) capacity_reservation: Option<RuntimeCapacityReservation>,
}

pub(super) struct StartupLocalModelTask {
    pub(super) node: mesh::Node,
    pub(super) config: plugin::MeshConfig,
    pub(super) tunnel_mgr: tunnel::Manager,
    pub(super) target_tx: Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: PathBuf,
    pub(super) model_ref: String,
    pub(super) model_name: String,
    pub(super) instance_id: String,
    pub(super) primary_model_name: String,
    pub(super) mmproj_path: Option<PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<StartupPinnedGpuTarget>,
    pub(super) runtime_capacity_ledger: RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) split: bool,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) survey_launch_kind: survey::SurveyLaunchKind,
    pub(super) stop_rx: tokio::sync::watch::Receiver<bool>,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: DashboardContextUsage,
    pub(super) runtime_instance_registry: RuntimeInstanceRegistry,
    pub(super) console_state: Option<api::MeshApi>,
    pub(super) api_port: u16,
    pub(super) startup_ready_reporter: StartupReadyReporter,
    pub(super) startup_load_gate: Arc<tokio::sync::Mutex<()>>,
    pub(super) input_handler_enabled: bool,
    pub(super) interactive_started: Arc<AtomicBool>,
    pub(super) interactive_control_tx:
        tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) interactive_console_state: Option<api::MeshApi>,
}

pub(super) struct StartupLaunchFailureContext<'a> {
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
}

pub(super) struct StartupSplitRuntimeLoopParams<'a, F, G>
where
    F: Fn() -> LocalRuntimeModelStartSpec<'a>,
    G: Fn() -> survey::SurveyModelSpec<'a> + Copy,
{
    make_start_spec: F,
    model_ref: &'a str,
    model_name: &'a str,
    local_capacity: u64,
    model_bytes: u64,
    node: &'a mesh::Node,
    startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    launch_failure: StartupLaunchFailureContext<'a>,
    make_survey_spec: G,
    announce_capacity_fallback: bool,
}

pub(super) struct StartupLocalRuntimeOnceParams<'a, F>
where
    F: Fn() -> survey::SurveyModelSpec<'a>,
{
    make_start_spec: LocalRuntimeModelStartSpec<'a>,
    runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    instance_id: &'a str,
    model_name: &'a str,
    pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    local_capacity: u64,
    model_bytes: u64,
    startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    launch_failure: StartupLaunchFailureContext<'a>,
    make_survey_spec: F,
    model_ref: &'a str,
}

pub(super) struct StartupLoopContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) tunnel_mgr: &'a tunnel::Manager,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: &'a PathBuf,
    pub(super) model_ref: &'a str,
    pub(super) instance_id: &'a str,
    pub(super) primary_model_name: &'a str,
    pub(super) mmproj_path: Option<&'a PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<&'a str>,
    pub(super) cache_type_v: Option<&'a str>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) launch_kind: survey::SurveyLaunchKind,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) api_port: u16,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
}

pub(super) struct StartupLoopState {
    pub(super) loaded_name: String,
    pub(super) handle: Option<LocalRuntimeModelHandle>,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) split_cleanup: Option<local::SplitGenerationCleanup>,
    pub(super) split_event_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) survey_loaded_model: survey::SurveyLoadedModel,
    pub(super) capacity_reservation: Option<RuntimeCapacityReservation>,
    pub(super) survey_exited_unexpectedly: bool,
}

pub(super) struct StartupLoopEventContext<'a> {
    pub(super) context_usage_tick: &'a mut tokio::time::Interval,
    pub(super) stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
}

pub(super) enum StartupLoopControl {
    Continue,
    Break,
    Return,
}

pub(super) struct StartupPreparedLaunch {
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
    pub(super) runtime_plan: StartupRuntimePlan,
    pub(super) launch_kind: survey::SurveyLaunchKind,
}

pub(super) struct StartupPrepareLaunchContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) model_path: &'a Path,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_name: &'a str,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) split: bool,
    pub(super) survey_launch_kind: survey::SurveyLaunchKind,
}

pub(super) struct StartupLaunchRuntimeContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: &'a PathBuf,
    pub(super) model_ref: &'a str,
    pub(super) model_name: &'a str,
    pub(super) instance_id: &'a str,
    pub(super) mmproj_path: Option<&'a PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<&'a str>,
    pub(super) cache_type_v: Option<&'a str>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    pub(super) stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
    pub(super) runtime_plan: StartupRuntimePlan,
    pub(super) launch_kind: survey::SurveyLaunchKind,
}

pub(super) struct PreparedRuntimeStartup {
    pub(super) startup_models: Vec<StartupModelPlan>,
    pub(super) requested_model_names: Vec<String>,
    pub(super) bin_dir: PathBuf,
}

pub(super) struct RunAutoJoinOutcome {
    pub(super) joined: bool,
    pub(super) last_join_error: Option<String>,
    pub(super) successful_join: Option<(String, Option<String>)>,
}

pub(super) struct ShutdownRuntimeLoadedModelsContext<'a> {
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) node: &'a mesh::Node,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
}

pub(super) async fn startup_reset_model_target(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    console_state: Option<&api::MeshApi>,
) {
    update_startup_target(target_tx, model_name, election::InferenceTarget::None);
    if let Some(cs) = console_state {
        cs.update(false, false).await;
    }
}

pub(super) async fn startup_emit_model_inspection_failure(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    err: &anyhow::Error,
    console_state: Option<&api::MeshApi>,
) {
    let _ = emit_event(OutputEvent::Error {
        message: format!("Failed to inspect model {model_name}: {err:#}"),
        context: Some(format!("model={model_name}")),
    });
    startup_reset_model_target(target_tx, model_name, console_state).await;
}

pub(super) async fn startup_emit_launch_failure(
    survey_telemetry: &survey::SurveyTelemetry,
    survey_spec: survey::SurveyModelSpec<'_>,
    launch_started: Instant,
    err: anyhow::Error,
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    console_state: Option<&api::MeshApi>,
) {
    survey_telemetry.record_launch_failure(
        survey_spec,
        launch_started.elapsed(),
        survey::classify_launch_failure(&err),
    );
    let _ = emit_event(OutputEvent::Error {
        message: format!("Failed to start model {model_name}: {err:#}"),
        context: Some(format!("model={model_name}")),
    });
    startup_reset_model_target(target_tx, model_name, console_state).await;
}

pub(super) async fn startup_start_split_runtime_loop<'a, F, G>(
    params: StartupSplitRuntimeLoopParams<'a, F, G>,
) -> Option<(StartupLaunchHandles, Instant)>
where
    F: Fn() -> LocalRuntimeModelStartSpec<'a>,
    G: Fn() -> survey::SurveyModelSpec<'a> + Copy,
{
    let StartupSplitRuntimeLoopParams {
        make_start_spec,
        model_ref,
        model_name,
        local_capacity,
        model_bytes,
        node,
        startup_load_gate,
        stop_rx,
        launch_failure,
        make_survey_spec,
        announce_capacity_fallback,
    } = params;
    let StartupLaunchFailureContext {
        target_tx,
        console_state,
        survey_telemetry,
    } = launch_failure;

    if announce_capacity_fallback {
        let required_bytes = runtime_model_required_bytes(model_bytes);
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Model {model_name} exceeds local runtime capacity; attempting split runtime"
            ),
            context: Some(format!(
                "model={model_name} local_capacity_gb={:.1} required_capacity_gb={:.1} model_size_gb={:.1}",
                local_capacity as f64 / 1e9,
                required_bytes as f64 / 1e9,
                model_bytes as f64 / 1e9
            )),
        });
    }

    let mut peer_rx = node.peer_change_rx.clone();
    loop {
        let startup_load_guard = startup_load_gate.lock().await;
        let launch_started = Instant::now();
        match start_runtime_split_model(make_start_spec(), model_ref).await {
            Ok(SplitRuntimeStart::Started(loaded)) => {
                drop(startup_load_guard);
                let mut loaded = *loaded;
                return Some((
                    StartupLaunchHandles {
                        loaded_name: loaded.loaded_name,
                        handle: loaded.handle,
                        death_rx: loaded.death_rx,
                        split_cleanup: loaded.cleanup.take(),
                        split_event_rx: loaded.coordinator_rx.take(),
                        coordinator_task: loaded.coordinator_task.take(),
                        capacity_reservation: None,
                    },
                    launch_started,
                ));
            }
            Ok(SplitRuntimeStart::Standby { coordinator }) => {
                drop(startup_load_guard);
                let _ = emit_event(OutputEvent::Info {
                    message: format!(
                        "Split runtime coordinator is {}; standing by for stage assignment",
                        coordinator.fmt_short()
                    ),
                    context: Some(format!("model={model_ref}")),
                });
                startup_reset_model_target(target_tx, model_name, console_state).await;
            }
            Err(err) => {
                drop(startup_load_guard);
                let err_msg = format!("{err:#}");
                let is_participant_shortage = err_msg.contains("at least two participating nodes")
                    || err_msg.contains("at least two stage participants");
                if is_participant_shortage {
                    let _ = emit_event(OutputEvent::Info {
                        message: format!("Split waiting for peers: {err_msg}"),
                        context: Some(format!("model={model_name}")),
                    });
                } else {
                    startup_emit_launch_failure(
                        survey_telemetry,
                        make_survey_spec(),
                        launch_started,
                        err,
                        target_tx,
                        model_name,
                        console_state,
                    )
                    .await;
                    return None;
                }
            }
        }

        tokio::select! {
            result = peer_rx.changed() => {
                if result.is_err() {
                    return None;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {}
                    result = stop_rx.changed() => {
                        if result.is_err() || *stop_rx.borrow() {
                            return None;
                        }
                    }
                }
            }
            _ = tokio::time::sleep(SPLIT_STANDBY_RETRY_INTERVAL) => {}
            result = stop_rx.changed() => {
                if result.is_err() || *stop_rx.borrow() {
                    return None;
                }
            }
        }
    }
}

pub(super) async fn startup_start_local_runtime_once<'a, F>(
    params: StartupLocalRuntimeOnceParams<'a, F>,
) -> Option<(StartupLaunchHandles, Instant)>
where
    F: Fn() -> survey::SurveyModelSpec<'a>,
{
    let StartupLocalRuntimeOnceParams {
        mut make_start_spec,
        runtime_capacity_ledger,
        instance_id,
        model_name,
        pinned_gpu,
        local_capacity,
        model_bytes,
        startup_load_gate,
        launch_failure,
        make_survey_spec,
        model_ref,
    } = params;
    let StartupLaunchFailureContext {
        target_tx,
        console_state,
        survey_telemetry,
    } = launch_failure;

    let startup_load_guard = startup_load_gate.lock().await;
    let launch_started = Instant::now();
    let reservation = match reserve_runtime_capacity_for_model(
        runtime_capacity_ledger,
        instance_id,
        model_name,
        pinned_gpu,
        local_capacity,
        model_bytes,
    ) {
        Ok(reservation) => reservation,
        Err(err) => {
            drop(startup_load_guard);
            startup_emit_launch_failure(
                survey_telemetry,
                make_survey_spec(),
                launch_started,
                err,
                target_tx,
                model_name,
                console_state,
            )
            .await;
            return None;
        }
    };

    make_start_spec.capacity_budget_bytes = Some(reservation.capacity_budget_bytes());
    let start_result = start_runtime_local_model(make_start_spec, model_ref).await;
    drop(startup_load_guard);

    match start_result {
        Ok((loaded_name, handle, death_rx)) => Some((
            StartupLaunchHandles {
                loaded_name,
                handle,
                death_rx,
                split_cleanup: None,
                split_event_rx: None,
                coordinator_task: None,
                capacity_reservation: Some(reservation),
            },
            launch_started,
        )),
        Err(err) => {
            drop(reservation);
            startup_emit_launch_failure(
                survey_telemetry,
                make_survey_spec(),
                launch_started,
                err,
                target_tx,
                model_name,
                console_state,
            )
            .await;
            None
        }
    }
}

pub(super) fn startup_split_unavailable_stage_nodes(nodes: &[iroh::EndpointId]) -> String {
    nodes
        .iter()
        .map(|node| node.fmt_short().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) async fn startup_unregister_runtime_instance(
    ctx: &StartupLoopContext<'_>,
    model_name: &str,
) -> bool {
    unregister_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        model_name,
        ctx.instance_id,
    )
    .await
}

pub(super) async fn startup_remove_runtime_instance_artifacts(
    ctx: &StartupLoopContext<'_>,
    model_name: &str,
) {
    if startup_unregister_runtime_instance(ctx, model_name).await {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            model_name,
            Some(ctx.instance_id),
        );
    }
    remove_dashboard_process(ctx.dashboard_processes, ctx.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(ctx.instance_id).await;
        cs.update(false, false).await;
    }
}

pub(super) async fn startup_register_loaded_runtime(
    ctx: &StartupLoopContext<'_>,
    loaded_name: &str,
    handle: &LocalRuntimeModelHandle,
) -> api::RuntimeProcessPayload {
    add_runtime_local_target(ctx.target_tx, loaded_name, handle.port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    register_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        ctx.primary_model_name,
        loaded_name,
        ctx.instance_id,
        Some(handle.context_length),
        handle.capabilities,
    )
    .await;
    let payload = local_process_payload(
        loaded_name,
        Some(ctx.instance_id),
        "",
        &handle.backend,
        handle.port,
        handle.pid(),
        handle.slots,
        handle.context_length,
    );
    upsert_dashboard_process(ctx.dashboard_processes, payload.clone()).await;
    payload
}

pub(super) fn startup_fallback_survey_spec<'a>(
    ctx: &'a StartupLoopContext<'a>,
    model_name: &'a str,
    backend: Option<&'a str>,
    context_length: Option<u32>,
) -> survey::SurveyModelSpec<'a> {
    survey::SurveyModelSpec {
        model: model_name,
        model_path: Some(ctx.model_path),
        launch_kind: survey::SurveyLaunchKind::MoeFallback,
        pinned_gpu: ctx.pinned_gpu,
        backend,
        context_length: context_length.map(u64::from),
    }
}

pub(super) async fn startup_handle_fallback_failure(
    ctx: &StartupLoopContext<'_>,
    event: &local::SplitCoordinatorLocalFallbackEvent,
    model_name: &str,
    launch_started: Instant,
    err: &anyhow::Error,
    unavailable_stage_nodes: &str,
) -> StartupLoopControl {
    ctx.survey_telemetry.record_launch_failure(
        startup_fallback_survey_spec(ctx, model_name, None, ctx.ctx_size),
        launch_started.elapsed(),
        survey::classify_launch_failure(err),
    );
    let _ = emit_event(OutputEvent::Warning {
        message: format!(
            "Split runtime topology '{}' lost required stage peer(s); local fallback failed, withdrawing model '{}'",
            event.topology_id, model_name
        ),
        context: Some(format!(
            "reason={} generation={} unavailable_stage_nodes=[{}] error={err:#}",
            event.reason, event.generation, unavailable_stage_nodes
        )),
    });
    startup_remove_runtime_instance_artifacts(ctx, model_name).await;
    StartupLoopControl::Return
}

pub(super) async fn startup_handle_local_fallback_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: local::SplitCoordinatorLocalFallbackEvent,
    local_capacity: u64,
    model_bytes: u64,
) -> StartupLoopControl {
    let unavailable_stage_nodes =
        startup_split_unavailable_stage_nodes(&event.unavailable_stage_nodes);
    let old_loaded_name = state.loaded_name.clone();
    let withdrew_topology = ctx
        .node
        .withdraw_stage_topology(&event.topology_id, &event.run_id)
        .await;
    let Some(old_handle) = state.handle.take() else {
        let _ = event.ack.send(SplitCoordinatorAck::Accepted);
        return StartupLoopControl::Break;
    };

    let old_port = old_handle.port;
    remove_runtime_local_target(ctx.target_tx, &old_loaded_name, old_port);
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &old_loaded_name, &old_handle)
        .await;
    old_handle.shutdown().await;
    ctx.survey_telemetry
        .record_unload(&state.survey_loaded_model);
    if let Some(cleanup) = state.split_cleanup.take() {
        stop_split_generation_cleanup(ctx.node, cleanup, event.generation.saturating_add(1)).await;
    }

    let launch_started = Instant::now();
    let reservation = match reserve_runtime_capacity_for_model(
        ctx.runtime_capacity_ledger,
        ctx.instance_id,
        &old_loaded_name,
        ctx.pinned_gpu,
        local_capacity,
        model_bytes,
    ) {
        Ok(reservation) => reservation,
        Err(err) => {
            let result = startup_handle_fallback_failure(
                ctx,
                &event,
                &old_loaded_name,
                launch_started,
                &err,
                &unavailable_stage_nodes,
            )
            .await;
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            return result;
        }
    };

    let start_result = start_runtime_local_model(
        LocalRuntimeModelStartSpec {
            node: ctx.node,
            mesh_config: ctx.config,
            config_model_id: Some(ctx.model_ref),
            model_path: ctx.model_path,
            model_bytes,
            mmproj_override: ctx.mmproj_path.map(PathBuf::as_path),
            ctx_size_override: ctx.ctx_size,
            pinned_gpu: ctx.pinned_gpu,
            capacity_budget_bytes: Some(reservation.capacity_budget_bytes()),
            cache_type_k_override: ctx.cache_type_k,
            cache_type_v_override: ctx.cache_type_v,
            n_batch_override: ctx.n_batch,
            n_ubatch_override: ctx.n_ubatch,
            flash_attention_override: ctx.flash_attention,
            parallel_override: ctx.parallel_override,
            planning_profile: ctx.resource_planning_profile,
            openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
            skippy_telemetry: ctx.skippy_telemetry.clone(),
            survey_telemetry: ctx.survey_telemetry.clone(),
        },
        ctx.model_ref,
    )
    .await;

    let (next_loaded_name, next_handle, next_death_rx) = match start_result {
        Ok(result) => result,
        Err(err) => {
            drop(reservation);
            let result = startup_handle_fallback_failure(
                ctx,
                &event,
                &old_loaded_name,
                launch_started,
                &err,
                &unavailable_stage_nodes,
            )
            .await;
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            return result;
        }
    };

    state.capacity_reservation = Some(reservation);
    state.loaded_name = next_loaded_name;
    let payload = startup_register_loaded_runtime(ctx, &state.loaded_name, &next_handle).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    state.survey_loaded_model = ctx.survey_telemetry.model(startup_fallback_survey_spec(
        ctx,
        &state.loaded_name,
        Some(&next_handle.backend),
        Some(next_handle.context_length),
    ));
    ctx.survey_telemetry
        .record_launch_success(&state.survey_loaded_model, launch_started.elapsed());
    refresh_dashboard_context_usage(
        ctx.dashboard_context_usage,
        &state.loaded_name,
        &next_handle,
    )
    .await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &state.loaded_name,
        Some(ctx.instance_id),
        &next_handle,
    );
    let new_port = next_handle.port;
    let new_context_length = next_handle.context_length;
    state.handle = Some(next_handle);
    state.death_rx = next_death_rx;
    state.split_event_rx = None;
    let _ = event.ack.send(SplitCoordinatorAck::Accepted);
    let _ = emit_event(OutputEvent::Warning {
        message: format!(
            "Split runtime topology '{}' lost required stage peer(s); recovered model '{}' locally",
            event.topology_id, state.loaded_name
        ),
        context: Some(format!(
            "reason={} generation={} run_id={} topology_withdrawn={} unavailable_stage_nodes=[{}] previous_port={} new_port={} new_ctx={}",
            event.reason,
            event.generation,
            event.run_id,
            withdrew_topology,
            unavailable_stage_nodes,
            old_port,
            new_port,
            new_context_length
        )),
    });
    StartupLoopControl::Continue
}

pub(super) async fn startup_handle_replace_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: local::SplitCoordinatorReplaceEvent,
) -> StartupLoopControl {
    let mut next = event.loaded;
    let old_loaded_name = state.loaded_name.clone();
    let Some(old_handle) = state.handle.take() else {
        let _ = event.ack.send(SplitCoordinatorAck::Accepted);
        return StartupLoopControl::Break;
    };

    let old_port = old_handle.port;
    let old_context_length = old_handle.context_length;
    remove_runtime_local_target(ctx.target_tx, &old_loaded_name, old_port);
    add_runtime_local_target(ctx.target_tx, &next.loaded_name, next.handle.port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    if old_loaded_name != next.loaded_name
        && startup_unregister_runtime_instance(ctx, &old_loaded_name).await
    {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &old_loaded_name,
            Some(ctx.instance_id),
        );
    }
    let payload = startup_register_loaded_runtime(ctx, &next.loaded_name, &next.handle).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &old_loaded_name, &old_handle)
        .await;
    ctx.survey_telemetry
        .record_unload(&state.survey_loaded_model);
    state.loaded_name = next.loaded_name;
    state.survey_loaded_model = ctx.survey_telemetry.model(survey::SurveyModelSpec {
        model: &state.loaded_name,
        model_path: Some(ctx.model_path),
        launch_kind: ctx.launch_kind,
        pinned_gpu: ctx.pinned_gpu,
        backend: Some(&next.handle.backend),
        context_length: Some(u64::from(next.handle.context_length)),
    });
    ctx.survey_telemetry
        .record_launch_success(&state.survey_loaded_model, Duration::from_secs(0));
    refresh_dashboard_context_usage(
        ctx.dashboard_context_usage,
        &state.loaded_name,
        &next.handle,
    )
    .await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &state.loaded_name,
        Some(ctx.instance_id),
        &next.handle,
    );
    let new_port = next.handle.port;
    let new_context_length = next.handle.context_length;
    state.death_rx = next.death_rx;
    state.split_cleanup = next.cleanup.take();
    state.handle = Some(next.handle);
    let _ = event.ack.send(SplitCoordinatorAck::Accepted);
    old_handle.shutdown().await;
    drop(state.capacity_reservation.take());
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Split runtime cut over model '{}' from :{} to :{}",
            state.loaded_name, old_port, new_port
        ),
        context: Some(format!(
            "reason={} generation={} previous_ctx={} new_ctx={}",
            event.reason, event.generation, old_context_length, new_context_length
        )),
    });
    StartupLoopControl::Continue
}

pub(super) async fn startup_handle_split_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: SplitCoordinatorEvent,
    local_capacity: u64,
    model_bytes: u64,
) -> StartupLoopControl {
    match event {
        SplitCoordinatorEvent::Replace(event) => {
            startup_handle_replace_event(ctx, state, *event).await
        }
        SplitCoordinatorEvent::LocalFallback(event) => {
            startup_handle_local_fallback_event(ctx, state, event, local_capacity, model_bytes)
                .await
        }
        SplitCoordinatorEvent::Withdraw(event) => {
            let unavailable_stage_nodes =
                startup_split_unavailable_stage_nodes(&event.unavailable_stage_nodes);
            let withdrew_topology = ctx
                .node
                .withdraw_stage_topology(&event.topology_id, &event.run_id)
                .await;
            let _ = emit_event(OutputEvent::Warning {
                message: format!(
                    "Split runtime topology '{}' lost required stage peer(s); withdrawing model '{}'",
                    event.topology_id, state.loaded_name
                ),
                context: Some(format!(
                    "reason={} generation={} run_id={} topology_withdrawn={} unavailable_stage_nodes=[{}]",
                    event.reason,
                    event.generation,
                    event.run_id,
                    withdrew_topology,
                    unavailable_stage_nodes
                )),
            });
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            StartupLoopControl::Break
        }
    }
}

pub(super) async fn startup_shutdown_local_model_loop(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    coordinator_task: &mut Option<tokio::task::JoinHandle<()>>,
) {
    if let Some(task) = coordinator_task.take() {
        task.abort();
        let _ = task.await;
    }
    if !state.survey_exited_unexpectedly {
        ctx.survey_telemetry
            .record_unload(&state.survey_loaded_model);
    }
    let Some(handle) = state.handle.take() else {
        drop(state.capacity_reservation.take());
        return;
    };
    let port = handle.port;
    remove_runtime_local_target(ctx.target_tx, &state.loaded_name, port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    if startup_unregister_runtime_instance(ctx, &state.loaded_name).await {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &state.loaded_name,
            Some(ctx.instance_id),
        );
    }
    let shutting_down_payload = runtime_process_payload_with_status(
        &state.loaded_name,
        Some(ctx.instance_id),
        &handle,
        "shutting down",
    );
    upsert_dashboard_process(ctx.dashboard_processes, shutting_down_payload.clone()).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(shutting_down_payload).await;
    }
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &state.loaded_name, &handle).await;
    handle.shutdown().await;
    drop(state.capacity_reservation.take());
    if let Some(cleanup) = state.split_cleanup.take() {
        stop_split_generation_cleanup(ctx.node, cleanup, u64::MAX).await;
    }
    remove_dashboard_process(ctx.dashboard_processes, ctx.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(ctx.instance_id).await;
        cs.update(false, false).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Stopped startup model '{}' from :{}",
            state.loaded_name, port
        ),
        context: None,
    });
}

pub(super) async fn startup_prepare_launch(
    ctx: StartupPrepareLaunchContext<'_>,
) -> Option<StartupPreparedLaunch> {
    let local_capacity = ctx
        .pinned_gpu
        .map(|gpu| gpu.allocatable_vram_bytes())
        .unwrap_or_else(|| ctx.node.vram_bytes());
    let model_bytes = startup_planning_model_bytes(&ctx).await?;
    let runtime_plan = startup_runtime_plan(ctx.split, local_capacity, model_bytes);
    let launch_kind = startup_launch_kind(runtime_plan, ctx.survey_launch_kind);
    Some(StartupPreparedLaunch {
        local_capacity,
        model_bytes,
        runtime_plan,
        launch_kind,
    })
}

pub(super) async fn startup_planning_model_bytes(
    ctx: &StartupPrepareLaunchContext<'_>,
) -> Option<u64> {
    let model_path_for_sizing = ctx.model_path.to_path_buf();
    match tokio::task::spawn_blocking(move || runtime_model_planning_bytes(&model_path_for_sizing))
        .await
        .context("join runtime model sizing task")
        .and_then(|result| result)
    {
        Ok(model_bytes) => Some(model_bytes),
        Err(err) => {
            startup_emit_model_inspection_failure(
                ctx.target_tx,
                ctx.model_name,
                &err,
                ctx.console_state,
            )
            .await;
            None
        }
    }
}

pub(super) fn startup_launch_kind(
    runtime_plan: StartupRuntimePlan,
    survey_launch_kind: survey::SurveyLaunchKind,
) -> survey::SurveyLaunchKind {
    match runtime_plan {
        StartupRuntimePlan::Local => survey_launch_kind,
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::Forced,
        } => survey::SurveyLaunchKind::MoeShard,
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity,
        } => survey::SurveyLaunchKind::MoeFallback,
    }
}

pub(super) async fn startup_launch_runtime(
    ctx: StartupLaunchRuntimeContext<'_>,
) -> Option<(StartupLaunchHandles, Instant)> {
    let StartupLaunchRuntimeContext {
        node,
        config,
        target_tx,
        model_path,
        model_ref,
        model_name,
        instance_id,
        mmproj_path,
        ctx_size,
        pinned_gpu,
        runtime_capacity_ledger,
        cache_type_k,
        cache_type_v,
        n_batch,
        n_ubatch,
        flash_attention,
        parallel_override,
        resource_planning_profile,
        openai_guardrail_policy,
        skippy_telemetry,
        survey_telemetry,
        console_state,
        startup_load_gate,
        stop_rx,
        local_capacity,
        model_bytes,
        runtime_plan,
        launch_kind,
    } = ctx;
    let make_start_spec = || LocalRuntimeModelStartSpec {
        node,
        mesh_config: config,
        config_model_id: Some(model_ref),
        model_path,
        model_bytes,
        mmproj_override: mmproj_path.map(PathBuf::as_path),
        ctx_size_override: ctx_size,
        pinned_gpu,
        capacity_budget_bytes: None,
        cache_type_k_override: cache_type_k,
        cache_type_v_override: cache_type_v,
        n_batch_override: n_batch,
        n_ubatch_override: n_ubatch,
        flash_attention_override: flash_attention,
        parallel_override,
        planning_profile: resource_planning_profile,
        openai_guardrail_policy: openai_guardrail_policy.clone(),
        skippy_telemetry: skippy_telemetry.clone(),
        survey_telemetry: survey_telemetry.clone(),
    };
    let make_launch_failure_spec = || survey::SurveyModelSpec {
        model: model_name,
        model_path: Some(model_path),
        launch_kind,
        pinned_gpu,
        backend: None,
        context_length: ctx_size.map(u64::from),
    };
    match runtime_plan {
        StartupRuntimePlan::Split { reason } => {
            startup_start_split_runtime_loop(StartupSplitRuntimeLoopParams {
                make_start_spec,
                model_ref,
                model_name,
                local_capacity,
                model_bytes,
                node,
                startup_load_gate,
                stop_rx,
                launch_failure: StartupLaunchFailureContext {
                    target_tx,
                    console_state,
                    survey_telemetry,
                },
                make_survey_spec: make_launch_failure_spec,
                announce_capacity_fallback: reason == SplitRuntimeReason::LocalCapacity,
            })
            .await
        }
        StartupRuntimePlan::Local => {
            startup_start_local_runtime_once(StartupLocalRuntimeOnceParams {
                make_start_spec: make_start_spec(),
                runtime_capacity_ledger,
                instance_id,
                model_name,
                pinned_gpu,
                local_capacity,
                model_bytes,
                startup_load_gate,
                launch_failure: StartupLaunchFailureContext {
                    target_tx,
                    console_state,
                    survey_telemetry,
                },
                make_survey_spec: make_launch_failure_spec,
                model_ref,
            })
            .await
        }
    }
}

pub(super) fn maybe_spawn_startup_interactive_handler(
    input_handler_enabled: bool,
    loaded_name: &str,
    primary_model_name: &str,
    interactive_started: &AtomicBool,
    interactive_control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    interactive_console_state: Option<api::MeshApi>,
) {
    if !input_handler_enabled || loaded_name != primary_model_name {
        return;
    }
    if interactive_started.swap(true, Ordering::AcqRel) || !std::io::stdin().is_terminal() {
        return;
    }
    if let Some(cs) = interactive_console_state {
        let Some(sink) = output_sink() else {
            return;
        };
        interactive::spawn_handler(
            interactive_control_tx,
            cs,
            sink,
            InitialPromptMode::Deferred,
        );
    }
}

pub(super) async fn runtime_data_producer_for_console(
    console_state: Option<&api::MeshApi>,
) -> Option<crate::runtime_data::RuntimeDataProducer> {
    match console_state {
        Some(cs) => Some(cs.runtime_data_producer().await),
        None => None,
    }
}

pub(super) async fn startup_local_model_loop(params: StartupLocalModelTask) {
    let StartupLocalModelTask {
        node,
        config,
        tunnel_mgr,
        target_tx,
        model_path,
        model_ref,
        model_name,
        instance_id,
        primary_model_name,
        mmproj_path,
        ctx_size,
        pinned_gpu,
        runtime_capacity_ledger,
        cache_type_k,
        cache_type_v,
        n_batch,
        n_ubatch,
        flash_attention,
        parallel_override,
        resource_planning_profile,
        openai_guardrail_policy,
        split,
        skippy_telemetry,
        survey_telemetry,
        survey_launch_kind,
        mut stop_rx,
        dashboard_processes,
        dashboard_context_usage,
        runtime_instance_registry,
        console_state,
        api_port,
        startup_ready_reporter,
        startup_load_gate,
        input_handler_enabled,
        interactive_started,
        interactive_control_tx,
        interactive_console_state,
    } = params;

    let runtime_data_producer = runtime_data_producer_for_console(console_state.as_ref()).await;

    let Some(StartupPreparedLaunch {
        local_capacity,
        model_bytes,
        runtime_plan,
        launch_kind,
    }) = startup_prepare_launch(StartupPrepareLaunchContext {
        node: &node,
        pinned_gpu: pinned_gpu.as_ref(),
        model_path: &model_path,
        target_tx: &target_tx,
        model_name: &model_name,
        console_state: console_state.as_ref(),
        split,
        survey_launch_kind,
    })
    .await
    else {
        return;
    };
    let Some((launch_handles, launch_started)) =
        startup_launch_runtime(StartupLaunchRuntimeContext {
            node: &node,
            config: &config,
            target_tx: &target_tx,
            model_path: &model_path,
            model_ref: &model_ref,
            model_name: &model_name,
            instance_id: &instance_id,
            mmproj_path: mmproj_path.as_ref(),
            ctx_size,
            pinned_gpu: pinned_gpu.as_ref(),
            runtime_capacity_ledger: &runtime_capacity_ledger,
            cache_type_k: cache_type_k.as_deref(),
            cache_type_v: cache_type_v.as_deref(),
            n_batch,
            n_ubatch,
            flash_attention,
            parallel_override,
            resource_planning_profile,
            openai_guardrail_policy: openai_guardrail_policy.clone(),
            skippy_telemetry: &skippy_telemetry,
            survey_telemetry: &survey_telemetry,
            console_state: console_state.as_ref(),
            startup_load_gate: &startup_load_gate,
            stop_rx: &mut stop_rx,
            local_capacity,
            model_bytes,
            runtime_plan,
            launch_kind,
        })
        .await
    else {
        return;
    };
    let StartupLaunchHandles {
        loaded_name,
        handle,
        death_rx,
        split_cleanup,
        split_event_rx,
        mut coordinator_task,
        capacity_reservation,
    } = launch_handles;

    let survey_loaded_model = survey_telemetry.model(survey::SurveyModelSpec {
        model: &loaded_name,
        model_path: Some(&model_path),
        launch_kind,
        pinned_gpu: pinned_gpu.as_ref(),
        backend: Some(&handle.backend),
        context_length: Some(u64::from(handle.context_length)),
    });
    survey_telemetry.record_launch_success(&survey_loaded_model, launch_started.elapsed());

    let ctx = StartupLoopContext {
        node: &node,
        config: &config,
        tunnel_mgr: &tunnel_mgr,
        target_tx: &target_tx,
        model_path: &model_path,
        model_ref: &model_ref,
        instance_id: &instance_id,
        primary_model_name: &primary_model_name,
        mmproj_path: mmproj_path.as_ref(),
        ctx_size,
        pinned_gpu: pinned_gpu.as_ref(),
        runtime_capacity_ledger: &runtime_capacity_ledger,
        cache_type_k: cache_type_k.as_deref(),
        cache_type_v: cache_type_v.as_deref(),
        n_batch,
        n_ubatch,
        flash_attention,
        parallel_override,
        resource_planning_profile,
        openai_guardrail_policy,
        skippy_telemetry: &skippy_telemetry,
        survey_telemetry: &survey_telemetry,
        launch_kind,
        dashboard_processes: &dashboard_processes,
        dashboard_context_usage: &dashboard_context_usage,
        runtime_instance_registry: &runtime_instance_registry,
        console_state: console_state.as_ref(),
        api_port,
        runtime_data_producer: runtime_data_producer.as_ref(),
    };
    startup_publish_loaded_runtime(&ctx, &loaded_name, &handle, &startup_ready_reporter).await;

    maybe_spawn_startup_interactive_handler(
        input_handler_enabled,
        &loaded_name,
        &primary_model_name,
        &interactive_started,
        interactive_control_tx,
        interactive_console_state,
    );

    let mut state = StartupLoopState {
        loaded_name,
        handle: Some(handle),
        death_rx,
        split_cleanup,
        split_event_rx,
        survey_loaded_model,
        capacity_reservation,
        survey_exited_unexpectedly: false,
    };
    let mut context_usage_tick = tokio::time::interval(DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL);
    context_usage_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    if !startup_run_local_model_event_loop(
        &ctx,
        &mut state,
        StartupLoopEventContext {
            context_usage_tick: &mut context_usage_tick,
            stop_rx: &mut stop_rx,
            local_capacity,
            model_bytes,
        },
    )
    .await
    {
        return;
    }

    startup_shutdown_local_model_loop(&ctx, &mut state, &mut coordinator_task).await;
}

pub(super) async fn startup_publish_loaded_runtime(
    ctx: &StartupLoopContext<'_>,
    loaded_name: &str,
    handle: &LocalRuntimeModelHandle,
    startup_ready_reporter: &StartupReadyReporter,
) {
    let payload = startup_register_loaded_runtime(ctx, loaded_name, handle).await;
    ctx.node
        .set_role(NodeRole::Host {
            http_port: ctx.api_port,
        })
        .await;
    refresh_dashboard_context_usage(ctx.dashboard_context_usage, loaded_name, handle).await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        loaded_name,
        Some(ctx.instance_id),
        handle,
    );
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    update_pi_models_json(loaded_name, ctx.api_port);
    startup_ready_reporter.mark_ready_and_maybe_emit(loaded_name);
    let _ = emit_event(OutputEvent::ModelReady {
        model: loaded_name.to_string(),
        internal_port: Some(handle.port),
        role: Some(handle.backend.clone()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: format!("Startup-loaded model '{}' on :{}", loaded_name, handle.port),
        context: None,
    });
}

pub(super) async fn startup_run_local_model_event_loop(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event_ctx: StartupLoopEventContext<'_>,
) -> bool {
    let StartupLoopEventContext {
        context_usage_tick,
        stop_rx,
        local_capacity,
        model_bytes,
    } = event_ctx;
    loop {
        tokio::select! {
            _ = context_usage_tick.tick() => {
                if let Some(handle) = state.handle.as_ref() {
                    refresh_dashboard_context_usage(ctx.dashboard_context_usage, &state.loaded_name, handle).await;
                    publish_runtime_llama_slots(ctx.runtime_data_producer, &state.loaded_name, Some(ctx.instance_id), handle);
                }
            }
            _ = &mut state.death_rx => {
                state.survey_exited_unexpectedly = true;
                ctx.survey_telemetry.record_unexpected_exit(&state.survey_loaded_model);
                let port = state.handle.as_ref().map(|handle| handle.port).unwrap_or_default();
                let _ = emit_event(OutputEvent::Warning {
                    message: format!("Startup model '{}' exited unexpectedly", state.loaded_name),
                    context: Some(format!("model={} port={port}", state.loaded_name)),
                });
                return true;
            }
            event = async {
                if let Some(rx) = state.split_event_rx.as_mut() {
                    rx.recv().await
                } else {
                    std::future::pending().await
                }
            } => {
                let Some(event) = event else {
                    state.split_event_rx = None;
                    continue;
                };
                match startup_handle_split_event(ctx, state, event, local_capacity, model_bytes).await {
                    StartupLoopControl::Continue => continue,
                    StartupLoopControl::Break => return true,
                    StartupLoopControl::Return => return false,
                }
            }
            res = stop_rx.changed() => {
                let _ = res;
                return true;
            }
        }
    }
}

pub(super) fn update_startup_target(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    target: election::InferenceTarget,
) {
    let mut targets = target_tx.borrow().clone();
    targets.targets.insert(model_name.to_string(), vec![target]);
    target_tx.send_replace(targets);
}

#[derive(Clone)]
pub(super) struct StartupReadyReporter {
    pub(super) ready_by_model: Arc<Mutex<HashMap<String, bool>>>,
    pub(super) emitted: Arc<AtomicBool>,
    pub(super) shutdown_requested: Arc<AtomicBool>,
    pub(super) primary_model: String,
    pub(super) api_url: String,
    pub(super) console_url: Option<String>,
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
}

impl StartupReadyReporter {
    pub(super) fn new(
        models: &[String],
        primary_model: String,
        api_url: String,
        console_url: Option<String>,
        api_port: u16,
        console_port: Option<u16>,
    ) -> Self {
        let ready_by_model = models.iter().cloned().map(|model| (model, false)).collect();
        Self {
            ready_by_model: Arc::new(Mutex::new(ready_by_model)),
            emitted: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            primary_model,
            api_url,
            console_url,
            api_port,
            console_port,
        }
    }

    pub(super) fn mark_shutdown_requested(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    pub(super) fn mark_ready_and_build_event(&self, model_name: &str) -> Option<OutputEvent> {
        let models_count = {
            let mut ready_by_model = self
                .ready_by_model
                .lock()
                .expect("startup readiness mutex poisoned");
            if let Some(entry) = ready_by_model.get_mut(model_name) {
                *entry = true;
            }
            if ready_by_model.values().all(|ready| *ready) {
                Some(ready_by_model.len())
            } else {
                None
            }
        };

        let models_count = models_count?;

        if self.shutdown_requested.load(Ordering::SeqCst) {
            return None;
        };

        if self.emitted.swap(true, Ordering::SeqCst) {
            return None;
        }

        let pi_command = Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            self.api_port,
            single_quote_shell_arg(&self.primary_model)
        ));
        let goose_command = Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST={} OPENAI_API_KEY=mesh GOOSE_MODEL={} goose session",
            self.api_url, self.primary_model
        ));
        Some(OutputEvent::RuntimeReady {
            api_url: self.api_url.clone(),
            console_url: self.console_url.clone(),
            api_port: self.api_port,
            console_port: self.console_port,
            models_count: Some(models_count),
            pi_command,
            goose_command,
        })
    }

    fn mark_ready_and_maybe_emit(&self, model_name: &str) {
        let Some(event) = self.mark_ready_and_build_event(model_name) else {
            return;
        };
        let _ = emit_event(event);
        let _ = schedule_ready_prompt();
    }
}

pub(super) async fn record_first_joined_mesh_ts(node: &mesh::Node) {
    let now_ms = current_time_unix_ms();
    node.set_first_joined_mesh_ts_if_absent(now_ms).await;
}
