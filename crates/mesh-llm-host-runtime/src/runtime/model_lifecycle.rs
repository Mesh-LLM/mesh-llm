use super::{
    DashboardContextUsage, LocalRuntimeModelHandle, LocalRuntimeModelStartSpec,
    ManagedModelController, ModelTargetReconciliationAction, ModelTargetReconciliationCandidate,
    ModelTargetReconciliationCapacityState, ModelTargetReconciliationInput,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, RunAutoRuntimeLoopContext,
    RunAutoRuntimeState, RuntimeCapacityReservation, RuntimeEvent, RuntimeInstanceRegistry,
    RuntimeOptions, RuntimeUnloadCandidate, RuntimeUnloadOwner, ShutdownRuntimeLoadedModelsContext,
    StartupReadyReporter, add_runtime_local_target, add_serving_assignment,
    find_remote_catalog_model_exact_blocking, local_process_payload, next_runtime_instance_id,
    plan_model_target_reconciliation, publish_runtime_llama_slots,
    publish_runtime_llama_unavailable, refresh_dashboard_context_usage, register_runtime_instance,
    remove_dashboard_context_usage, remove_dashboard_process, remove_runtime_local_target,
    remove_serving_assignment, reserve_runtime_capacity_for_model, resolve_model,
    runtime_model_ctx_size_override, runtime_model_planning_bytes,
    runtime_process_payload_with_status, runtime_registry_has_model,
    runtime_resource_planning_profile, set_advertised_model_context, skippy_telemetry_options,
    start_runtime_local_model, unregister_runtime_instance, upsert_dashboard_process,
    withdraw_advertised_model,
};
use crate::api;
use crate::inference::election;
use crate::mesh;
use crate::models;
use crate::network::lan_bootstrap::LanBootstrapTasks;
use crate::plugin;
use crate::runtime::survey;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};
use skippy_protocol::FlashAttentionType;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(super) fn model_target_reconciliation_policy(
    config: &plugin::MeshConfig,
) -> ModelTargetReconciliationPolicy {
    ModelTargetReconciliationPolicy {
        enabled: config.runtime.reconcile_model_targets,
        demand_upgrades_enabled: config.runtime.reconcile_model_target_demand_upgrades,
        demand_upgrade_min_request_count: config.runtime.model_target_demand_upgrade_min_requests,
        demand_upgrade_max_age_secs: config.runtime.model_target_demand_upgrade_max_age_secs,
        ..ModelTargetReconciliationPolicy::default()
    }
}

pub(super) struct ReconcileModelTargetsContext<'a> {
    pub(super) policy: &'a ModelTargetReconciliationPolicy,
    pub(super) state: &'a mut ModelTargetReconciliationState,
    pub(super) node: &'a mesh::Node,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_models: &'a HashMap<String, RuntimeModelHandleEntry>,
    pub(super) managed_models: &'a HashMap<String, ManagedModelController>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
}

pub(super) async fn reconcile_model_targets_once(ctx: ReconcileModelTargetsContext<'_>) {
    let ReconcileModelTargetsContext {
        policy,
        state,
        node,
        console_state,
        runtime_models,
        managed_models,
        control_tx,
        runtime_event_tx,
    } = ctx;
    if !policy.enabled {
        return;
    }
    let Some(console_state) = console_state else {
        return;
    };
    let local_interest_model_refs = node
        .explicit_model_interests()
        .await
        .into_iter()
        .collect::<BTreeSet<_>>();
    let loaded_model_refs = runtime_loaded_model_refs(runtime_models, managed_models);
    if local_interest_model_refs.is_empty() && loaded_model_refs.is_empty() {
        state.prune_expired(runtime_unix_secs());
        return;
    }

    let target_lookup = console_state.model_target_lookup().await;
    let local_vram_bytes = node.vram_bytes();
    let targets = target_lookup
        .targets
        .into_iter()
        .map(|target| {
            let demand_upgrade_target = model_target_reconciliation_demand_upgrade_candidate(
                policy,
                &loaded_model_refs,
                &target,
            );
            let local_path = if target.wanted
                && target.serving_node_count == 0
                && (local_interest_model_refs.contains(&target.model_ref) || demand_upgrade_target)
                && target.capacity_advice.state
                    == api::status::ModelTargetCapacityAdviceState::SingleNodeFit
                && model_target_reconciliation_local_fit(&target, local_vram_bytes)
            {
                local_model_path_for_reconciliation_target(&target)
            } else {
                None
            };
            ModelTargetReconciliationCandidate {
                rank: target.rank,
                model_ref: target.model_ref,
                profile: target.profile,
                model_name: target.model_name,
                wanted: target.wanted,
                wanted_reason: target.wanted_reason,
                request_count: target.request_count,
                last_active_secs_ago: target.last_active_secs_ago,
                serving_node_count: target.serving_node_count,
                capacity_state: ModelTargetReconciliationCapacityState::from(
                    target.capacity_advice.state,
                ),
                local_path,
            }
        })
        .collect::<Vec<_>>();

    let now_secs = runtime_unix_secs();
    let actions = plan_model_target_reconciliation(
        policy,
        state,
        ModelTargetReconciliationInput {
            now_secs,
            local_role: node.role().await,
            local_interest_model_refs: &local_interest_model_refs,
            loaded_model_refs: &loaded_model_refs,
            targets: &targets,
        },
    );

    for action in actions {
        let load_spec = action.load_spec.to_string_lossy().to_string();
        let profile = action.profile.clone();
        state.mark_load_started(&action.model_ref, &profile);
        let event_tx = runtime_event_tx.clone();
        let model_ref = action.model_ref.clone();
        let control_tx = control_tx.clone();
        let replace_model_ref = action.replace_model_ref.clone();
        let event_profile = action.profile.clone();
        tokio::spawn(async move {
            let result = run_model_target_reconciliation_action(
                control_tx,
                load_spec,
                replace_model_ref,
                profile,
            )
            .await;
            let _ = event_tx.send(RuntimeEvent::ModelTargetReconciliationLoadFinished {
                model_ref,
                profile: event_profile,
                result,
            });
        });
        emit_model_target_reconciliation_queued(&action);
    }
}

pub(super) async fn run_model_target_reconciliation_action(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    load_spec: String,
    replace_model_ref: Option<String>,
    profile: String,
) -> std::result::Result<api::RuntimeLoadResponse, String> {
    if let Some(replace_model_ref) = replace_model_ref {
        run_model_target_reconciliation_unload(control_tx.clone(), replace_model_ref).await?;
    }
    run_model_target_reconciliation_load(control_tx, load_spec, profile).await
}

pub(super) async fn run_model_target_reconciliation_unload(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    model_ref: String,
) -> std::result::Result<api::RuntimeUnloadResponse, String> {
    let (resp, response) = tokio::sync::oneshot::channel();
    control_tx
        .send(api::RuntimeControlRequest::Unload {
            target: UnloadTarget::Model(model_ref.clone()),
            options: UnloadOptions::default(),
            resp,
        })
        .map_err(|_| format!("runtime unload queue closed for replacement target '{model_ref}'"))?;
    response
        .await
        .map_err(|err| format!("runtime unload response channel closed: {err}"))?
        .map_err(|err| err.to_string())
}

pub(super) async fn run_model_target_reconciliation_load(
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    load_spec: String,
    profile: String,
) -> std::result::Result<api::RuntimeLoadResponse, String> {
    let (resp, response) = tokio::sync::oneshot::channel();
    control_tx
        .send(api::RuntimeControlRequest::Load {
            spec: load_spec.clone(),
            profile: profile.clone(),
            resp,
        })
        .map_err(|_| format!("runtime load queue closed for '{load_spec}'"))?;
    response
        .await
        .map_err(|err| format!("runtime load response channel closed: {err}"))?
        .map_err(|err| err.to_string())
}

pub(super) fn emit_model_target_reconciliation_queued(action: &ModelTargetReconciliationAction) {
    let context = match action.replace_model_ref.as_deref() {
        Some(replace_model_ref) => Some(format!("replace={replace_model_ref}")),
        None => Some(format!("path={}", action.load_spec.display())),
    };
    let verb = if action.replace_model_ref.is_some() {
        "upgrading to"
    } else {
        "loading"
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!("Model target reconciliation {verb} '{}'", action.model_ref),
        context,
    });
}

pub(super) fn runtime_loaded_model_refs(
    runtime_models: &HashMap<String, RuntimeModelHandleEntry>,
    managed_models: &HashMap<String, ManagedModelController>,
) -> BTreeSet<String> {
    runtime_models
        .values()
        .map(|entry| entry.model_name.clone())
        .chain(
            managed_models
                .values()
                .map(|controller| controller.model_name.clone()),
        )
        .collect()
}

pub(super) fn local_model_path_for_reconciliation_target(
    target: &api::status::ModelTargetPayload,
) -> Option<PathBuf> {
    [
        Some(target.model_ref.as_str()),
        target.model_name.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(models::find_model_path)
    .find(|path| path.exists())
}

pub(super) fn model_target_reconciliation_local_fit(
    target: &api::status::ModelTargetPayload,
    local_vram_bytes: u64,
) -> bool {
    target
        .capacity_advice
        .required_bytes
        .is_some_and(|required| local_vram_bytes >= required)
}

pub(super) fn model_target_reconciliation_demand_upgrade_candidate(
    policy: &ModelTargetReconciliationPolicy,
    loaded_model_refs: &BTreeSet<String>,
    target: &api::status::ModelTargetPayload,
) -> bool {
    policy.demand_upgrades_enabled
        && !loaded_model_refs.is_empty()
        && target.wanted_reason == Some("active_demand")
        && target.request_count >= policy.demand_upgrade_min_request_count
        && target
            .last_active_secs_ago
            .is_some_and(|age| age <= policy.demand_upgrade_max_age_secs)
}

pub(super) fn runtime_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

pub(super) struct RunAutoShutdownContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) api_proxy_handle: tokio::task::JoinHandle<()>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) discovery_publisher: Option<tokio::task::JoinHandle<()>>,
    pub(super) lan_bootstrap_tasks: LanBootstrapTasks,
    pub(super) runtime_models: &'a mut HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: &'a mut HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
}

pub(super) struct RunAutoRuntimeLifecycleContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) primary_model_name: &'a str,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) control_rx: &'a mut tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_rx: &'a mut tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
    pub(super) runtime_state: &'a mut RunAutoRuntimeState,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) api_proxy_handle: tokio::task::JoinHandle<()>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) discovery_publisher: Option<tokio::task::JoinHandle<()>>,
    pub(super) lan_bootstrap_tasks: LanBootstrapTasks,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
}

pub(super) async fn run_auto_load_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    spec: String,
    profile: String,
) -> Result<api::RuntimeLoadResponse> {
    let model_path = resolve_model(&PathBuf::from(&spec)).await?;
    let runtime_model_name = find_remote_catalog_model_exact_blocking(spec.clone())
        .await
        .map(|model| models::remote_catalog_model_ref(&model))
        .unwrap_or_else(|| models::model_ref_for_path(&model_path));
    let requested_model = spec.clone();
    let model_bytes = {
        let p = model_path.clone();
        tokio::task::spawn_blocking(move || runtime_model_planning_bytes(&p))
            .await
            .unwrap_or_else(|err| {
                Err(anyhow::anyhow!(
                    "join runtime model byte planning task: {err}"
                ))
            })
            .unwrap_or_else(|err| {
                let fallback = election::total_model_bytes(&model_path);
                tracing::warn!(
                    model = %requested_model,
                    error = %err,
                    fallback_bytes = fallback,
                    "failed to resolve runtime model planning bytes; using filesystem size fallback"
                );
                fallback
            })
    };
    let model_overrides = ctx
        .config
        .models
        .iter()
        .find(|m| m.model == spec && m.derived_profile() == *profile);
    let ctx_size_override = runtime_model_ctx_size_override(ctx.options, model_overrides);
    let parallel_override = super::startup_models::resolve_model_parallel_override(
        model_overrides.and_then(|m| m.parallel),
        &ctx.config.gpu,
    );
    let instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
    let capacity_reservation = reserve_runtime_capacity_for_model(
        ctx.runtime_capacity_ledger,
        &instance_id,
        &runtime_model_name,
        None,
        ctx.node.vram_bytes(),
        model_bytes,
    )?;
    add_serving_assignment(ctx.node, ctx.primary_model_name, &runtime_model_name).await;
    let launch_started = Instant::now();
    let capacity_budget_bytes = capacity_reservation.capacity_budget_bytes();
    let (loaded_name, handle, death_rx) = match start_runtime_local_model(
        LocalRuntimeModelStartSpec {
            node: ctx.node,
            mesh_config: ctx.config,
            config_model_id: Some(&spec),
            model_path: &model_path,
            model_bytes,
            mmproj_override: None,
            ctx_size_override,
            pinned_gpu: None,
            capacity_budget_bytes: Some(capacity_budget_bytes),
            cache_type_k_override: model_overrides.and_then(|m| m.cache_type_k.as_deref()),
            cache_type_v_override: model_overrides.and_then(|m| m.cache_type_v.as_deref()),
            n_batch_override: model_overrides.and_then(|m| m.batch),
            n_ubatch_override: model_overrides.and_then(|m| m.ubatch),
            flash_attention_override: model_overrides
                .and_then(|m| m.flash_attention)
                .unwrap_or(FlashAttentionType::Auto),
            parallel_override,
            planning_profile: runtime_resource_planning_profile(ctx.options),
            openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
            skippy_telemetry: skippy_telemetry_options(ctx.options),
            survey_telemetry: ctx.survey_telemetry.clone(),
        },
        &runtime_model_name,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            drop(capacity_reservation);
            remove_serving_assignment(ctx.node, &runtime_model_name).await;
            ctx.survey_telemetry.record_launch_failure(
                survey::SurveyModelSpec {
                    model: &requested_model,
                    model_path: Some(&model_path),
                    launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
                    pinned_gpu: None,
                    backend: None,
                    context_length: ctx_size_override.map(u64::from),
                },
                launch_started.elapsed(),
                survey::classify_launch_failure(&err),
            );
            return Err(err);
        }
    };
    let survey_loaded_model = ctx.survey_telemetry.model(survey::SurveyModelSpec {
        model: &loaded_name,
        model_path: Some(&model_path),
        launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
        pinned_gpu: None,
        backend: Some(&handle.backend),
        context_length: Some(u64::from(handle.context_length)),
    });
    ctx.survey_telemetry
        .record_launch_success(&survey_loaded_model, launch_started.elapsed());
    add_runtime_local_target(ctx.target_tx, &loaded_name, handle.port);
    register_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        ctx.primary_model_name,
        &loaded_name,
        &instance_id,
        Some(handle.context_length),
        handle.capabilities,
    )
    .await;
    ctx.node
        .set_available_models(models::scan_local_models())
        .await;
    let payload = local_process_payload(
        &loaded_name,
        Some(&instance_id),
        &profile,
        &handle.backend,
        handle.port,
        handle.pid(),
        handle.slots,
        handle.context_length,
    );
    upsert_dashboard_process(ctx.dashboard_processes, payload.clone()).await;
    if let Some(cs) = ctx.console_state {
        cs.set_openai_guardrails(
            handle
                .openai_guardrails()
                .map(crate::api::status::OpenAiGuardrailsPayload::from),
        )
        .await;
        cs.upsert_local_process(payload).await;
    }

    let event_tx = ctx.runtime_event_tx.clone();
    let event_instance_id = instance_id.clone();
    let event_name = loaded_name.clone();
    let event_port = handle.port;
    tokio::spawn(async move {
        let _ = death_rx.await;
        let _ = event_tx.send(RuntimeEvent::Exited {
            instance_id: event_instance_id,
            model: event_name,
            port: event_port,
        });
    });

    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Runtime-loaded {} model '{}' on :{}",
            handle.backend, loaded_name, handle.port
        ),
        context: None,
    });
    refresh_dashboard_context_usage(ctx.dashboard_context_usage, &loaded_name, &handle).await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &loaded_name,
        Some(&instance_id),
        &handle,
    );
    ctx.runtime_survey_models
        .insert(instance_id.clone(), survey_loaded_model);
    let loaded_backend = handle.backend.clone();
    let loaded_context_length = handle.context_length;
    ctx.runtime_models.insert(
        instance_id.clone(),
        RuntimeModelHandleEntry {
            model_name: loaded_name.clone(),
            handle,
            capacity_reservation,
        },
    );
    Ok(api::RuntimeLoadResponse {
        model_ref: requested_model,
        model: loaded_name,
        instance_id,
        profile: profile.clone(),
        backend: Some(loaded_backend),
        context_length: Some(loaded_context_length),
    })
}

pub(super) async fn run_auto_unload_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    target: UnloadTarget,
    options: UnloadOptions,
) -> Result<api::RuntimeUnloadResponse> {
    let unload = resolve_runtime_unload_target(
        target.as_runtime_target(),
        runtime_unload_candidates(ctx.runtime_models, ctx.managed_models),
    )?;
    let drain_delay = if options.force {
        Duration::ZERO
    } else {
        options.drain_timeout
    };
    match unload.owner {
        RuntimeUnloadOwner::Runtime => {
            run_auto_unload_runtime_entry(ctx, unload, drain_delay).await
        }
        RuntimeUnloadOwner::Managed => {
            let Some(controller) = ctx.managed_models.remove(&unload.instance_id) else {
                anyhow::bail!(
                    "model or runtime instance '{}' is not loaded",
                    unload.instance_id
                );
            };
            let model = controller.model_name.clone();
            let _ = controller.stop_tx.send(true);
            await_managed_model_stop(controller.task, drain_delay, options.force, &model).await;
            if !runtime_registry_has_model(ctx.runtime_instance_registry, &model).await {
                publish_runtime_llama_unavailable(
                    ctx.runtime_data_producer,
                    &model,
                    Some(&unload.instance_id),
                );
                withdraw_advertised_model(ctx.node, &model, "").await;
                set_advertised_model_context(ctx.node, &model, None).await;
                remove_serving_assignment(ctx.node, &model).await;
            }
            remove_dashboard_process(ctx.dashboard_processes, &unload.instance_id).await;
            if let Some(cs) = ctx.console_state {
                cs.remove_local_process(&unload.instance_id).await;
            }
            let _ = emit_event(OutputEvent::Info {
                message: format!("Unloaded managed model '{}'", model),
                context: None,
            });
            Ok(api::RuntimeUnloadResponse {
                model,
                instance_id: unload.instance_id,
                unloaded: true,
            })
        }
    }
}

pub(super) async fn await_managed_model_stop(
    mut task: tokio::task::JoinHandle<()>,
    drain_timeout: Duration,
    force: bool,
    model: &str,
) {
    if force {
        task.abort();
        let _ = task.await;
        return;
    }

    match tokio::time::timeout(drain_timeout, &mut task).await {
        Ok(join_result) => {
            let _ = join_result;
        }
        Err(_) => {
            tracing::warn!(
                model,
                drain_timeout_ms = drain_timeout.as_millis(),
                "managed model task did not stop within unload drain timeout; aborting"
            );
            task.abort();
            let _ = task.await;
        }
    }
}

pub(super) async fn run_auto_unload_runtime_entry(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    unload: RuntimeUnloadCandidate,
    drain_delay: Duration,
) -> Result<api::RuntimeUnloadResponse> {
    let Some(entry) = ctx.runtime_models.remove(&unload.instance_id) else {
        anyhow::bail!(
            "model or runtime instance '{}' is not loaded",
            unload.instance_id
        );
    };
    let RuntimeModelHandleEntry {
        model_name: model,
        handle,
        capacity_reservation,
    } = entry;
    let port = handle.port;
    if let Some(survey_model) = ctx.runtime_survey_models.remove(&unload.instance_id) {
        ctx.survey_telemetry.record_unload(&survey_model);
    }
    remove_runtime_local_target(ctx.target_tx, &model, port);
    if unregister_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        &model,
        &unload.instance_id,
    )
    .await
    {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &model,
            Some(&unload.instance_id),
        );
    }
    upsert_dashboard_process(
        ctx.dashboard_processes,
        runtime_process_payload_with_status(
            &model,
            Some(&unload.instance_id),
            &handle,
            "shutting down",
        ),
    )
    .await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(runtime_process_payload_with_status(
            &model,
            Some(&unload.instance_id),
            &handle,
            "shutting down",
        ))
        .await;
    }
    if !drain_delay.is_zero() {
        tokio::time::sleep(drain_delay).await;
    }
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &model, &handle).await;
    handle.shutdown().await;
    drop(capacity_reservation);
    remove_dashboard_process(ctx.dashboard_processes, &unload.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(&unload.instance_id).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: format!("Unloaded local model '{}' from :{}", model, port),
        context: None,
    });
    Ok(api::RuntimeUnloadResponse {
        model,
        instance_id: unload.instance_id,
        unloaded: true,
    })
}

pub(super) async fn run_auto_handle_runtime_exit(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    instance_id: String,
    model: String,
    port: u16,
) {
    let matches = ctx
        .runtime_models
        .get(&instance_id)
        .map(|entry| entry.model_name == model && entry.handle.port == port)
        .unwrap_or(false);
    if !matches {
        return;
    }
    if let Some(entry) = ctx.runtime_models.remove(&instance_id) {
        let RuntimeModelHandleEntry {
            handle,
            capacity_reservation,
            ..
        } = entry;
        if let Some(survey_model) = ctx.runtime_survey_models.remove(&instance_id) {
            ctx.survey_telemetry.record_unexpected_exit(&survey_model);
        }
        if unregister_runtime_instance(
            ctx.runtime_instance_registry,
            ctx.node,
            &model,
            &instance_id,
        )
        .await
        {
            publish_runtime_llama_unavailable(
                ctx.runtime_data_producer,
                &model,
                Some(&instance_id),
            );
        }
        upsert_dashboard_process(
            ctx.dashboard_processes,
            runtime_process_payload_with_status(&model, Some(&instance_id), &handle, "exited"),
        )
        .await;
        if let Some(cs) = ctx.console_state {
            cs.upsert_local_process(runtime_process_payload_with_status(
                &model,
                Some(&instance_id),
                &handle,
                "exited",
            ))
            .await;
        }
        remove_dashboard_context_usage(ctx.dashboard_context_usage, &model, &handle).await;
        handle.shutdown().await;
        drop(capacity_reservation);
    }
    remove_runtime_local_target(ctx.target_tx, &model, port);
    let _ = emit_event(OutputEvent::Warning {
        message: format!("Runtime model '{model}' exited unexpectedly"),
        context: Some(format!("model={model} port={port}")),
    });
}

pub(super) async fn run_auto_reconcile_model_targets(ctx: &mut RunAutoRuntimeLoopContext<'_>) {
    reconcile_model_targets_once(ReconcileModelTargetsContext {
        policy: &ctx.model_target_reconciliation_policy,
        state: &mut ctx.model_target_reconciliation_state,
        node: ctx.node,
        console_state: ctx.console_state,
        runtime_models: ctx.runtime_models,
        managed_models: ctx.managed_models,
        control_tx: ctx.control_tx,
        runtime_event_tx: ctx.runtime_event_tx,
    })
    .await;
}

pub(super) fn run_auto_record_model_target_manual_unload(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    requested_target: &str,
    result: &Result<api::RuntimeUnloadResponse>,
) {
    let Ok(response) = result else {
        return;
    };
    let now_secs = runtime_unix_secs();
    ctx.model_target_reconciliation_state.record_manual_unload(
        requested_target,
        "",
        now_secs,
        &ctx.model_target_reconciliation_policy,
    );
    if response.model != requested_target {
        ctx.model_target_reconciliation_state.record_manual_unload(
            &response.model,
            "",
            now_secs,
            &ctx.model_target_reconciliation_policy,
        );
    }
}

pub(super) fn run_auto_handle_model_target_reconciliation_result(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    model_ref: String,
    profile: String,
    result: std::result::Result<api::RuntimeLoadResponse, String>,
) {
    match result {
        Ok(response) => {
            let load_profile = if response.profile.is_empty() {
                profile.clone()
            } else {
                response.profile.clone()
            };
            ctx.model_target_reconciliation_state
                .record_load_success(&model_ref, &load_profile);
            if !load_profile.is_empty() && load_profile != profile {
                tracing::warn!(
                    model_ref = %model_ref,
                    requested_profile = %profile,
                    loaded_profile = %load_profile,
                    "model target reconciliation load response profile differs from requested profile"
                );
            }
            let _ = emit_event(OutputEvent::Info {
                message: format!("Model target reconciliation loaded '{}'", response.model),
                context: Some(format!(
                    "model_ref={} instance={}",
                    model_ref, response.instance_id
                )),
            });
        }
        Err(error) => {
            ctx.model_target_reconciliation_state.record_load_failure(
                &model_ref,
                &profile,
                runtime_unix_secs(),
                &ctx.model_target_reconciliation_policy,
            );
            let _ = emit_event(OutputEvent::Warning {
                message: format!("Model target reconciliation failed for '{model_ref}'"),
                context: Some(error),
            });
        }
    }
}

pub(super) async fn shutdown_runtime_loaded_models(
    runtime_models: &mut HashMap<String, RuntimeModelHandleEntry>,
    runtime_survey_models: &mut HashMap<String, survey::SurveyLoadedModel>,
    ctx: ShutdownRuntimeLoadedModelsContext<'_>,
) {
    let ShutdownRuntimeLoadedModelsContext {
        survey_telemetry,
        dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry,
        node,
        runtime_data_producer,
        dashboard_context_usage,
    } = ctx;

    for (instance_id, entry) in runtime_models.drain() {
        let RuntimeModelHandleEntry {
            model_name: name,
            handle,
            capacity_reservation,
        } = entry;
        if let Some(survey_model) = runtime_survey_models.remove(&instance_id) {
            survey_telemetry.record_unload(&survey_model);
        }
        let shutting_down_payload = runtime_process_payload_with_status(
            &name,
            Some(&instance_id),
            &handle,
            "shutting down",
        );
        upsert_dashboard_process(dashboard_processes, shutting_down_payload.clone()).await;
        if let Some(cs) = console_state {
            cs.upsert_local_process(shutting_down_payload).await;
        }
        remove_runtime_local_target(target_tx, &name, handle.port);
        if unregister_runtime_instance(runtime_instance_registry, node, &name, &instance_id).await {
            publish_runtime_llama_unavailable(runtime_data_producer, &name, Some(&instance_id));
        }
        remove_dashboard_context_usage(dashboard_context_usage, &name, &handle).await;
        let _ = emit_event(OutputEvent::ModelUnloading {
            model: name.clone(),
        });
        let stopped_payload =
            runtime_process_payload_with_status(&name, Some(&instance_id), &handle, "stopped");
        handle.shutdown().await;
        drop(capacity_reservation);
        let _ = emit_event(OutputEvent::ModelUnloaded {
            model: name.clone(),
        });
        upsert_dashboard_process(dashboard_processes, stopped_payload.clone()).await;
        if let Some(cs) = console_state {
            cs.upsert_local_process(stopped_payload).await;
        }
    }
}

pub(super) async fn shutdown_runtime_managed_models(
    managed_models: &mut HashMap<String, ManagedModelController>,
) {
    for (_, controller) in managed_models.drain() {
        let _ = emit_event(OutputEvent::ModelUnloading {
            model: controller.model_name.clone(),
        });
        let _ = controller.stop_tx.send(true);
        let mut task = controller.task;
        match tokio::time::timeout(std::time::Duration::from_secs(3), &mut task).await {
            Ok(join_result) => {
                let _ = join_result;
            }
            Err(_) => {
                tracing::warn!("local model task did not stop within 3s during shutdown");
                task.abort();
                let _ = task.await;
            }
        }
        let _ = emit_event(OutputEvent::ModelUnloaded {
            model: controller.model_name,
        });
    }
}

pub(super) struct RuntimeModelHandleEntry {
    pub(super) model_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) capacity_reservation: RuntimeCapacityReservation,
}

pub(super) fn runtime_unload_candidates(
    runtime_models: &HashMap<String, RuntimeModelHandleEntry>,
    managed_models: &HashMap<String, ManagedModelController>,
) -> Vec<RuntimeUnloadCandidate> {
    runtime_models
        .iter()
        .map(|(instance_id, entry)| RuntimeUnloadCandidate {
            owner: RuntimeUnloadOwner::Runtime,
            instance_id: instance_id.clone(),
            model_name: entry.model_name.clone(),
        })
        .chain(
            managed_models
                .iter()
                .map(|(instance_id, controller)| RuntimeUnloadCandidate {
                    owner: RuntimeUnloadOwner::Managed,
                    instance_id: instance_id.clone(),
                    model_name: controller.model_name.clone(),
                }),
        )
        .collect()
}

pub(super) fn resolve_runtime_unload_target(
    target: &str,
    candidates: Vec<RuntimeUnloadCandidate>,
) -> Result<RuntimeUnloadCandidate> {
    let mut instance_matches = candidates
        .iter()
        .filter(|candidate| candidate.instance_id == target);
    if let Some(candidate) = instance_matches.next() {
        return Ok(candidate.clone());
    }

    let model_matches: Vec<_> = candidates
        .into_iter()
        .filter(|candidate| candidate.model_name == target)
        .collect();
    match model_matches.len() {
        0 => Err(anyhow::anyhow!(
            "model or runtime instance '{target}' is not loaded"
        )),
        1 => Ok(model_matches.into_iter().next().expect("one model match")),
        _ => {
            let ids = model_matches
                .iter()
                .map(|candidate| candidate.instance_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Err(anyhow::anyhow!(
                "model '{target}' has multiple loaded instances ({ids}); unload by runtime instance id"
            ))
        }
    }
}
