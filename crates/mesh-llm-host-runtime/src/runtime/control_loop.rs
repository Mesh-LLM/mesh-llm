/// Wait for either SIGINT (ctrl-c) or SIGTERM. Without this, an unhandled
/// SIGTERM aborts the process before runtime cleanup can run.
use super::{
    DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL, MODEL_TARGET_RECONCILIATION_INTERVAL,
    ModelTargetReconciliationState, OpenAiGuardrailPolicyHandle, RunAutoRuntimeLifecycleContext,
    RunAutoRuntimeLoopContext, RunAutoShutdownContext, RuntimeEvent,
    ShutdownRuntimeLoadedModelsContext, cleanup_run_auto_runtime_dir,
    dashboard_context_usage_source, emit_shutdown, model_target_reconciliation_policy,
    publish_runtime_llama_slots, refresh_dashboard_context_usage_batch,
    run_auto_handle_model_target_reconciliation_result, run_auto_handle_runtime_exit,
    run_auto_load_runtime_model, run_auto_reconcile_model_targets,
    run_auto_record_model_target_manual_unload, run_auto_unload_runtime_model,
    set_openai_guardrail_policy_mode, shutdown_run_auto_services, shutdown_runtime_loaded_models,
    shutdown_runtime_managed_models, unpublish_run_auto_nostr_listing,
};
use crate::api;
use crate::inference::skippy;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event, flush_output};

pub(super) async fn wait_shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return "SIGINT";
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = term.recv() => "SIGTERM",
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "CTRL-C"
    }
}

pub(super) async fn run_auto_runtime_loop_and_shutdown(ctx: RunAutoRuntimeLifecycleContext<'_>) {
    let RunAutoRuntimeLifecycleContext {
        options,
        config,
        node,
        primary_model_name,
        target_tx,
        control_rx,
        control_tx,
        runtime_event_rx,
        runtime_state,
        console_state,
        runtime_data_producer,
        runtime_event_tx,
        survey_telemetry,
        startup_ready_reporter,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        lan_bootstrap_tasks,
        runtime,
    } = ctx;
    let mut loop_ctx = RunAutoRuntimeLoopContext {
        options,
        config,
        node,
        primary_model_name,
        target_tx,
        control_tx,
        runtime_models: &mut runtime_state.runtime_models,
        runtime_survey_models: &mut runtime_state.runtime_survey_models,
        managed_models: &mut runtime_state.managed_models,
        runtime_capacity_ledger: &runtime_state.runtime_capacity_ledger,
        next_runtime_instance_sequence: &mut runtime_state.next_runtime_instance_sequence,
        runtime_instance_registry: &runtime_state.runtime_instance_registry,
        dashboard_processes: &runtime_state.dashboard_processes,
        dashboard_context_usage: &runtime_state.dashboard_context_usage,
        console_state,
        runtime_data_producer,
        runtime_event_tx,
        survey_telemetry,
        startup_ready_reporter,
        openai_guardrail_policy: &runtime_state.openai_guardrail_policy,
        model_target_reconciliation_policy: model_target_reconciliation_policy(config),
        model_target_reconciliation_state: ModelTargetReconciliationState::default(),
    };
    run_auto_runtime_event_loop(&mut loop_ctx, control_rx, runtime_event_rx).await;

    shutdown_run_auto_runtime(RunAutoShutdownContext {
        options,
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        lan_bootstrap_tasks,
        runtime_models: &mut runtime_state.runtime_models,
        runtime_survey_models: &mut runtime_state.runtime_survey_models,
        managed_models: &mut runtime_state.managed_models,
        survey_telemetry,
        dashboard_processes: &runtime_state.dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry: &runtime_state.runtime_instance_registry,
        runtime_data_producer,
        dashboard_context_usage: &runtime_state.dashboard_context_usage,
        runtime,
    })
    .await;
}

pub(super) async fn shutdown_run_auto_runtime(ctx: RunAutoShutdownContext<'_>) {
    let RunAutoShutdownContext {
        options,
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        lan_bootstrap_tasks,
        runtime_models,
        runtime_survey_models,
        managed_models,
        survey_telemetry,
        dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry,
        runtime_data_producer,
        dashboard_context_usage,
        runtime,
    } = ctx;
    node.broadcast_leaving().await;

    unpublish_run_auto_nostr_listing(options).await;
    if let Some(handle) = discovery_publisher {
        handle.abort();
    }
    // Stop the relay-less LAN bootstrap loops (mDNS publisher, reverse-dial,
    // and beacon) so they release their sockets and stop dialing on shutdown.
    lan_bootstrap_tasks.abort();

    shutdown_run_auto_services(
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
    )
    .await;

    shutdown_runtime_loaded_models(
        runtime_models,
        runtime_survey_models,
        ShutdownRuntimeLoadedModelsContext {
            survey_telemetry,
            dashboard_processes,
            console_state,
            target_tx,
            runtime_instance_registry,
            node,
            runtime_data_producer,
            dashboard_context_usage,
        },
    )
    .await;
    shutdown_runtime_managed_models(managed_models).await;

    node.set_serving_models(Vec::new()).await;
    node.set_hosted_models(Vec::new()).await;
    cleanup_run_auto_runtime_dir(runtime);
}

pub(super) async fn run_auto_handle_control_request(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    cmd: api::RuntimeControlRequest,
) -> bool {
    match cmd {
        api::RuntimeControlRequest::Join { invite_token, resp } => {
            let result = ctx.node.join_with_retry(&invite_token).await;
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::Load {
            spec,
            profile,
            resp,
        } => {
            let result = run_auto_load_runtime_model(ctx, spec, profile).await;
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::Unload {
            target,
            options,
            resp,
        } => {
            let result = run_auto_unload_runtime_model(ctx, target.clone(), options).await;
            run_auto_record_model_target_manual_unload(ctx, target.as_runtime_target(), &result);
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::SetOpenAiGuardrailMode { mode, resp } => {
            let result = run_auto_set_openai_guardrail_mode(ctx, mode).await;
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::Shutdown { source } => {
            let _ = emit_event(OutputEvent::ShutdownRequested { signal: source });
            ctx.startup_ready_reporter.mark_shutdown_requested();
            let _ = flush_output().await;
            emit_shutdown(None).await;
            true
        }
    }
}

pub(super) async fn run_auto_set_openai_guardrail_mode(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    mode: openai_frontend::GuardrailMode,
) -> Result<api::OpenAiGuardrailModeUpdateResponse> {
    set_openai_guardrail_policy_mode(ctx.openai_guardrail_policy, mode);
    let mut updated_models = 0_usize;
    let mut latest_status = None;
    for entry in ctx.runtime_models.values() {
        if let Some(status) = entry.handle.set_openai_guardrail_mode(mode) {
            updated_models += 1;
            latest_status = Some(status);
        }
    }

    let status_payload = Some(
        latest_status
            .map(api::status::OpenAiGuardrailsPayload::from)
            .unwrap_or_else(|| openai_guardrails_payload_from_policy(ctx.openai_guardrail_policy)),
    );
    if let Some(console_state) = ctx.console_state {
        console_state
            .set_openai_guardrails(status_payload.clone())
            .await;
    }

    Ok(api::OpenAiGuardrailModeUpdateResponse {
        mode: guardrail_mode_status_label(mode),
        updated_models,
        status: status_payload,
    })
}

pub(super) fn guardrail_mode_status_label(mode: openai_frontend::GuardrailMode) -> &'static str {
    match mode {
        openai_frontend::GuardrailMode::Disabled => "disabled",
        openai_frontend::GuardrailMode::MetricsOnly => "metrics",
        openai_frontend::GuardrailMode::Enforce => "enforce",
    }
}

pub(super) fn openai_guardrails_payload_from_policy(
    policy: &OpenAiGuardrailPolicyHandle,
) -> api::status::OpenAiGuardrailsPayload {
    api::status::OpenAiGuardrailsPayload::from(
        skippy::skippy_openai_guardrails_for_policy_handle(policy.clone()).status(),
    )
}

pub(super) async fn publish_initial_openai_guardrails_status(
    console_state: Option<&api::MeshApi>,
    policy: &OpenAiGuardrailPolicyHandle,
) {
    let Some(console_state) = console_state else {
        return;
    };
    console_state
        .set_openai_guardrails(Some(openai_guardrails_payload_from_policy(policy)))
        .await;
}

pub(super) async fn run_auto_runtime_event_loop(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    control_rx: &mut tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    runtime_event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) {
    let mut dashboard_context_usage_tick =
        tokio::time::interval(DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL);
    dashboard_context_usage_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut model_target_reconciliation_tick =
        tokio::time::interval(MODEL_TARGET_RECONCILIATION_INTERVAL);
    model_target_reconciliation_tick
        .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = dashboard_context_usage_tick.tick() => {
                let updates = ctx.runtime_models
                    .iter()
                    .map(|(instance_id, entry)| {
                        publish_runtime_llama_slots(
                            ctx.runtime_data_producer,
                            &entry.model_name,
                            Some(instance_id.as_str()),
                            &entry.handle,
                        );
                        (
                            entry.model_name.clone(),
                            dashboard_context_usage_source(&entry.handle),
                            entry.handle.ctx_used_tokens(),
                        )
                    })
                    .collect();
                refresh_dashboard_context_usage_batch(ctx.dashboard_context_usage, updates).await;
            }
            _ = model_target_reconciliation_tick.tick() => {
                run_auto_reconcile_model_targets(ctx).await;
            }
            signal = wait_shutdown_signal() => {
                let _ = emit_event(OutputEvent::ShutdownRequested { signal });
                ctx.startup_ready_reporter.mark_shutdown_requested();
                let _ = flush_output().await;
                emit_shutdown(None).await;
                break;
            }
            Some(cmd) = control_rx.recv() => {
                if run_auto_handle_control_request(ctx, cmd).await {
                    break;
                }
            }
            Some(event) = runtime_event_rx.recv() => {
                match event {
                    RuntimeEvent::ModelTargetReconciliationLoadFinished {
                        model_ref,
                        profile,
                        result,
                    } => {
                        run_auto_handle_model_target_reconciliation_result(
                            ctx,
                            model_ref,
                            profile,
                            result,
                        );
                    }
                    RuntimeEvent::Exited { instance_id, model, port } => {
                        run_auto_handle_runtime_exit(ctx, instance_id, model, port).await;
                    }
                }
            }
        }
    }
}

pub(super) fn spawn_embedded_runtime_control_forwarder(
    embedded_control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
) {
    let Some(mut embedded_control_rx) = embedded_control_rx else {
        return;
    };
    tokio::spawn(async move {
        while let Some(command) = embedded_control_rx.recv().await {
            if control_tx.send(command).is_err() {
                break;
            }
        }
    });
}
