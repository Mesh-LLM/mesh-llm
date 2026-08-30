use super::*;
use std::path::{Path, PathBuf};

fn audit_runtime_model_load_result<T>(
    result: Result<T>,
    context: OperationalAuditContext,
    load_started: Instant,
) -> Result<T> {
    result.inspect_err(|_| {
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelLoadFailed,
            context
                .outcome("failed")
                .duration_ms(u64::try_from(load_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
        );
    })
}

fn record_runtime_model_load_terminal(
    event: RuntimeOperationalEvent,
    model: &str,
    instance_id: &str,
    outcome: &'static str,
    load_started: Instant,
) {
    record_runtime_operational_event_with_context(
        event,
        runtime_model_audit_context(Some(model), instance_id)
            .outcome(outcome)
            .duration_ms(u64::try_from(load_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
    );
}

fn spawn_runtime_model_exit_watcher(
    event_tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    instance_id: String,
    model: String,
    port: u16,
    death_rx: tokio::sync::oneshot::Receiver<()>,
) {
    tokio::spawn(async move {
        let _ = death_rx.await;
        let _ = event_tx.send(RuntimeEvent::Exited {
            instance_id,
            model,
            port,
        });
    });
}

async fn plan_runtime_model_bytes(model_path: &Path, requested_model: &str) -> u64 {
    let planning_path = model_path.to_path_buf();
    tokio::task::spawn_blocking(move || runtime_model_planning_bytes(&planning_path))
        .await
        .unwrap_or_else(|err| {
            Err(anyhow::anyhow!(
                "join runtime model byte planning task: {err}"
            ))
        })
        .unwrap_or_else(|err| {
            let fallback = election::total_model_bytes(model_path);
            tracing::warn!(
                model = %requested_model,
                error = %err,
                fallback_bytes = fallback,
                "failed to resolve runtime model planning bytes; using filesystem size fallback"
            );
            fallback
        })
}

fn find_profile_model_overrides<'a>(
    config: &'a plugin::MeshConfig,
    model_ref: &str,
    profile: &str,
) -> Option<&'a plugin::ModelConfigEntry> {
    config.models.iter().find(|model| {
        model.model == model_ref
            && model
                .with_profile_defaults(config.defaults.as_ref())
                .derived_profile()
                == profile
    })
}

/// Run auto-load for a runtime model.
pub(crate) async fn run_auto_load_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    spec: String,
    config_model_id: Option<String>,
    profile: String,
) -> Result<api::RuntimeLoadResponse> {
    let load_started = Instant::now();
    let instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelLoadStarted,
        runtime_model_audit_context(None, &instance_id).outcome("started"),
    );
    let model_path = audit_runtime_model_load_result(
        resolve_model(&PathBuf::from(&spec)).await,
        runtime_model_audit_context(None, &instance_id),
        load_started,
    )?;
    let runtime_model_name = find_remote_catalog_model_exact_blocking(spec.clone())
        .await
        .map(|model| models::remote_catalog_model_ref(&model))
        .unwrap_or_else(|| models::model_ref_for_path(&model_path));
    let requested_model = spec.clone();
    let model_bytes = plan_runtime_model_bytes(&model_path, &requested_model).await;
    let config_selector = config_model_id.as_deref().unwrap_or(&spec);
    let model_overrides = find_profile_model_overrides(ctx.config, config_selector, &profile);
    let ctx_size_override = runtime_model_ctx_size_override(ctx.options, model_overrides);
    let parallel_override = crate::runtime::startup_models::resolve_model_parallel_override(
        model_overrides.and_then(|m| m.parallel),
        &ctx.config.gpu,
    );
    let capacity_reservation = audit_runtime_model_load_result(
        reserve_runtime_capacity_for_model(
            ctx.runtime_capacity_ledger,
            &instance_id,
            &runtime_model_name,
            None,
            ctx.node.local_runtime_capacity_bytes(),
            model_bytes,
        ),
        runtime_model_audit_context(Some(&runtime_model_name), &instance_id),
        load_started,
    )?;
    add_serving_assignment(ctx.node, ctx.primary_model_name, &runtime_model_name).await;
    let launch_started = Instant::now();
    let capacity_budget_bytes = capacity_reservation.capacity_budget_bytes();
    let (loaded_name, handle, death_rx) = match start_runtime_local_model(
        LocalRuntimeModelStartSpec {
            node: ctx.node,
            mesh_config: ctx.config,
            config_model_id: model_overrides.map(|model| model.model.as_str()),
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
            split_topology_lock: None,
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
                    configured_model_selector: config_model_id.as_deref(),
                    model_path: Some(&model_path),
                    launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
                    pinned_gpu: None,
                    backend: None,
                    context_length: ctx_size_override.map(u64::from),
                },
                launch_started.elapsed(),
                survey::classify_launch_failure(&err),
            );
            record_runtime_model_load_terminal(
                RuntimeOperationalEvent::ModelLoadFailed,
                &runtime_model_name,
                &instance_id,
                "failed",
                load_started,
            );
            return Err(err);
        }
    };
    let survey_loaded_model = ctx.survey_telemetry.model(survey::SurveyModelSpec {
        model: &loaded_name,
        configured_model_selector: config_model_id.as_deref(),
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

    spawn_runtime_model_exit_watcher(
        ctx.runtime_event_tx.clone(),
        instance_id.clone(),
        loaded_name.clone(),
        handle.port,
        death_rx,
    );

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
    let lifecycle = std::sync::Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        50,
    )));
    ctx.node
        .register_runtime_instance_lifecycle(handle.port, lifecycle.clone());
    ctx.runtime_models.insert(
        instance_id.clone(),
        RuntimeModelHandleEntry {
            model_name: loaded_name.clone(),
            profile: profile.clone(),
            handle,
            capacity_reservation,
            lifecycle,
        },
    );
    record_runtime_model_load_terminal(
        RuntimeOperationalEvent::ModelReady,
        &loaded_name,
        &instance_id,
        "ready",
        load_started,
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
