use super::*;
use std::path::PathBuf;

fn audit_runtime_model_load_result<T>(result: Result<T>) -> Result<T> {
    result.inspect_err(|_| {
        record_runtime_operational_event(RuntimeOperationalEvent::ModelLoadFailed);
    })
}

/// Run auto-load for a runtime model.
pub(crate) async fn run_auto_load_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    spec: String,
    profile: String,
) -> Result<api::RuntimeLoadResponse> {
    record_runtime_operational_event(RuntimeOperationalEvent::ModelLoadStarted);
    let model_path = audit_runtime_model_load_result(resolve_model(&PathBuf::from(&spec)).await)?;
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
    let parallel_override = crate::runtime::startup_models::resolve_model_parallel_override(
        model_overrides.and_then(|m| m.parallel),
        &ctx.config.gpu,
    );
    let instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
    let capacity_reservation =
        audit_runtime_model_load_result(reserve_runtime_capacity_for_model(
            ctx.runtime_capacity_ledger,
            &instance_id,
            &runtime_model_name,
            None,
            ctx.node.local_runtime_capacity_bytes(),
            model_bytes,
        ))?;
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
                    model_path: Some(&model_path),
                    launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
                    pinned_gpu: None,
                    backend: None,
                    context_length: ctx_size_override.map(u64::from),
                },
                launch_started.elapsed(),
                survey::classify_launch_failure(&err),
            );
            record_runtime_operational_event(RuntimeOperationalEvent::ModelLoadFailed);
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
    record_runtime_operational_event(RuntimeOperationalEvent::ModelReady);
    Ok(api::RuntimeLoadResponse {
        model_ref: requested_model,
        model: loaded_name,
        instance_id,
        profile: profile.clone(),
        backend: Some(loaded_backend),
        context_length: Some(loaded_context_length),
    })
}
