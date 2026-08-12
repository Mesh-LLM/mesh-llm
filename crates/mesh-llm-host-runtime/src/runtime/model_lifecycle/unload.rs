use super::*;
use crate::runtime::{DrainCoordinator, DrainResult};

/// Unload a runtime model (entry point for both runtime and managed models).
#[expect(
    clippy::cognitive_complexity,
    reason = "unload dispatch keeps runtime and managed ownership cleanup symmetric and auditable"
)]
pub(crate) async fn run_auto_unload_runtime_model(
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
            let ManagedModelController {
                model_name: model,
                stop_tx,
                task,
                lifecycle,
                port,
                ..
            } = controller;
            let active_port = port.load(std::sync::atomic::Ordering::Acquire);
            if active_port != 0 {
                let draining = {
                    let mut record = lifecycle.lock().await;
                    match record.mark_draining(Instant::now() + drain_delay) {
                        Ok(()) => true,
                        Err(error) => {
                            tracing::warn!(
                                model,
                                instance_id = unload.instance_id,
                                %error,
                                "managed instance could not enter draining; forcing unload"
                            );
                            false
                        }
                    }
                };
                if draining {
                    let result = DrainCoordinator::default()
                        .wait_for_unload_ready(&lifecycle)
                        .await;
                    if result == DrainResult::ForceCancelled {
                        tracing::warn!(
                            model,
                            instance_id = unload.instance_id,
                            "managed instance drain deadline expired; forcing unload"
                        );
                    }
                    if let Err(error) = lifecycle.lock().await.transition_to_unloading() {
                        tracing::warn!(
                            model,
                            instance_id = unload.instance_id,
                            %error,
                            "managed instance could not enter unloading; forcing unload"
                        );
                    }
                }
            }
            let _ = stop_tx.send(true);
            await_managed_model_stop(task, drain_delay, options.force, &model).await;
            ctx.node.unregister_runtime_instance_lifecycle(active_port);
            {
                let mut record = lifecycle.lock().await;
                if record.state() == InstanceLifecycleState::Unloading
                    && let Err(error) = record.transition_to(InstanceLifecycleState::Stopped)
                {
                    tracing::warn!(
                        model,
                        instance_id = unload.instance_id,
                        %error,
                        "managed instance stopped with a stale lifecycle state"
                    );
                }
            }
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
            record_runtime_operational_event(RuntimeOperationalEvent::ModelUnloaded);
            Ok(api::RuntimeUnloadResponse {
                model,
                instance_id: unload.instance_id,
                unloaded: true,
            })
        }
    }
}

/// Await a managed model's task to stop, with optional timeout and force.
pub(crate) async fn await_managed_model_stop(
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

/// Unload a runtime-managed model entry (not externally managed).
#[expect(
    clippy::cognitive_complexity,
    reason = "runtime entry unload keeps drain, registry, telemetry, and capacity cleanup in one ordered transaction"
)]
pub(crate) async fn run_auto_unload_runtime_entry(
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
        lifecycle,
        ..
    } = entry;
    let port = handle.port;
    let draining = {
        let mut record = lifecycle.lock().await;
        match record.mark_draining(std::time::Instant::now() + drain_delay) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    model,
                    instance_id = unload.instance_id,
                    %error,
                    "runtime instance could not enter draining; forcing unload"
                );
                false
            }
        }
    };
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
    if draining {
        let result = DrainCoordinator::default()
            .wait_for_unload_ready(&lifecycle)
            .await;
        if result == DrainResult::ForceCancelled {
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                "instance drain deadline expired; forcing unload"
            );
        }
        if let Err(error) = lifecycle.lock().await.transition_to_unloading() {
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                %error,
                "runtime instance could not enter unloading; forcing unload"
            );
        }
    }
    ctx.node.unregister_runtime_instance_lifecycle(port);
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &model, &handle).await;
    handle.shutdown().await;
    {
        let mut record = lifecycle.lock().await;
        if record.state() == InstanceLifecycleState::Unloading
            && let Err(error) = record.transition_to(InstanceLifecycleState::Stopped)
        {
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                %error,
                "runtime instance stopped with a stale lifecycle state"
            );
        }
    }
    drop(capacity_reservation);
    remove_dashboard_process(ctx.dashboard_processes, &unload.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(&unload.instance_id).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: format!("Unloaded local model '{}' from :{}", model, port),
        context: None,
    });
    record_runtime_operational_event(RuntimeOperationalEvent::ModelUnloaded);
    Ok(api::RuntimeUnloadResponse {
        model,
        instance_id: unload.instance_id,
        unloaded: true,
    })
}

/// Handle unexpected exit of a runtime-loaded model.
pub(crate) async fn run_auto_handle_runtime_exit(
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
            lifecycle,
            ..
        } = entry;
        ctx.node.unregister_runtime_instance_lifecycle(port);
        let _ = lifecycle
            .lock()
            .await
            .transition_to(InstanceLifecycleState::Failed);
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
