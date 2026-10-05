//! Translate Mesh configuration and observers into the Skippy lifecycle API.
use super::*;

struct NativeSkippyStartupAudit {
    ready: bool,
}

impl NativeSkippyStartupAudit {
    fn new() -> Self {
        record_native_skippy_operational_event(NativeSkippyOperationalEvent::RuntimeStartupStarted);
        Self { ready: false }
    }

    fn mark_ready(&mut self) {
        self.ready = true;
        record_native_skippy_operational_event(NativeSkippyOperationalEvent::RuntimeReady);
    }
}

impl Drop for NativeSkippyStartupAudit {
    fn drop(&mut self) {
        if !self.ready {
            record_native_skippy_operational_event(
                NativeSkippyOperationalEvent::RuntimeStartupFailed,
            );
        }
    }
}

impl SkippyModelHandle {
    pub(crate) fn output_activation_boundary(
        &self,
    ) -> Option<skippy_runtime::ActivationBoundaryDesc> {
        self.runtime.output_activation_boundary()
    }

    pub(super) fn resolved_mtp_source(
        native_mtp_enabled: bool,
        native_mtp_draft_model_path: Option<&Path>,
    ) -> MtpSource {
        if !native_mtp_enabled {
            MtpSource::Disabled
        } else if native_mtp_draft_model_path.is_some() {
            MtpSource::External
        } else {
            MtpSource::Integrated
        }
    }

    pub(crate) fn load(options: SkippyModelLoadOptions) -> Result<Self> {
        Self::load_with_hooks(options, None, survey::SurveyTelemetry::disabled())
    }

    pub(crate) fn load_with_hooks(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        guardrail_telemetry: survey::SurveyTelemetry,
    ) -> Result<Self> {
        audit_load(|| Self::load_local(options, hook_policy, guardrail_telemetry, None))
    }

    pub(crate) fn load_with_hooks_and_open_events(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        model_open_events: Option<NativeModelOpenEvents>,
        guardrail_telemetry: survey::SurveyTelemetry,
    ) -> Result<Self> {
        audit_load(|| {
            Self::load_local(options, hook_policy, guardrail_telemetry, model_open_events)
        })
    }

    fn load_local(
        options: SkippyModelLoadOptions,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        guardrail_telemetry: survey::SurveyTelemetry,
        model_open_events: Option<NativeModelOpenEvents>,
    ) -> Result<Self> {
        let stage_config = single_stage_config(&options)?;
        let mtp_source = Self::resolved_mtp_source(
            options.native_mtp_enabled,
            options
                .embedded_openai
                .as_ref()
                .and_then(|args| args.native_mtp_draft_model_path.as_deref()),
        );
        let runtime_options = EmbeddedRuntimeOptions {
            config: stage_config,
            topology: None,
            n_threads: options.n_threads,
            n_threads_batch: options.n_threads_batch,
            mtp_source,
            metrics_otlp_grpc: options.telemetry.metrics_otlp_grpc.clone(),
            telemetry_queue_capacity: options.telemetry.queue_capacity,
            telemetry_level: options.telemetry.level,
            session_lifecycle_observer: Some(Arc::new(
                runtime_events::SkippySessionRuntimeEventObserver::new(),
            )),
        };
        let embedded_args = options.embedded_openai.unwrap_or_else(|| {
            resolver::ResolvedEmbeddedOpenAiArgs::direct_single_stage_defaults(
                options.model_id,
                options.default_max_tokens,
                options.generation_concurrency,
                options.native_mtp_enabled,
            )
        });
        Self::load_prepared(
            runtime_options,
            embedded_args,
            hook_policy,
            SkippyOpenAiGuardrailOptions::new(options.openai_guardrails, guardrail_telemetry),
            options.serving_hooks_factory,
            model_open_events,
        )
    }

    pub(crate) fn load_stage0_config(
        config: StageConfig,
        activation_width: i32,
        generation_concurrency: usize,
        default_max_tokens: u32,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
    ) -> Result<Self> {
        let model_id = config.model_id.clone();
        let native_mtp_enabled = config.native_mtp_enabled;
        Self::load_stage0_config_with_openai_args(
            config,
            resolver::ResolvedEmbeddedOpenAiArgs::embedded_stage_defaults(
                Some(model_id),
                default_max_tokens,
                generation_concurrency,
                activation_width,
                native_mtp_enabled,
            ),
            hook_policy,
            telemetry,
            guardrails,
        )
    }

    pub(crate) fn load_stage0_config_with_openai_args(
        config: StageConfig,
        embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
    ) -> Result<Self> {
        let mtp_source = Self::resolved_mtp_source(
            config.native_mtp_enabled,
            embedded_args.native_mtp_draft_model_path.as_deref(),
        );
        let session_observer: Arc<dyn skippy_serving::runtime_state::SessionLifecycleObserver> =
            Arc::new(runtime_events::SkippySessionRuntimeEventObserver::new());
        Self::load_stage0_runtime_options_with_openai_args(
            EmbeddedRuntimeOptions {
                config,
                topology: None,
                n_threads: None,
                n_threads_batch: None,
                mtp_source,
                metrics_otlp_grpc: telemetry.metrics_otlp_grpc.clone(),
                telemetry_queue_capacity: telemetry.queue_capacity,
                telemetry_level: telemetry.level,
                session_lifecycle_observer: Some(session_observer),
            },
            embedded_args,
            hook_policy,
            telemetry,
            guardrails,
            None,
        )
    }

    pub(crate) fn load_stage0_runtime_options_with_openai_args(
        runtime_options: EmbeddedRuntimeOptions,
        embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    ) -> Result<Self> {
        audit_load(|| {
            Self::load_stage0_prepared(
                runtime_options,
                embedded_args,
                hook_policy,
                telemetry,
                guardrails,
                serving_hooks_factory,
                None,
            )
        })
    }

    pub(crate) fn load_stage0_runtime_options_with_openai_args_and_open_events(
        runtime_options: EmbeddedRuntimeOptions,
        embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        model_open_events: Option<NativeModelOpenEvents>,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    ) -> Result<Self> {
        audit_load(|| {
            Self::load_stage0_prepared(
                runtime_options,
                embedded_args,
                hook_policy,
                telemetry,
                guardrails,
                serving_hooks_factory,
                model_open_events,
            )
        })
    }

    fn load_stage0_prepared(
        mut runtime_options: EmbeddedRuntimeOptions,
        mut embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        telemetry: SkippyTelemetryOptions,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
        model_open_events: Option<NativeModelOpenEvents>,
    ) -> Result<Self> {
        configure_materialized_stage_cache();
        let config = &mut runtime_options.config;
        anyhow::ensure!(
            config.load_mode != LoadMode::LayerPackage,
            "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
        );
        if config.kv_cache.is_none() {
            config.kv_cache =
                family_policy_for_stage_config(config).stage_kv_cache_config_for_stage(config);
        }
        if config.downstream.is_none() {
            embedded_args.activation_width = 0;
        }
        // The existing stage path supplies its serving telemetry separately.
        let mut request = Self::model_load_request(
            runtime_options,
            embedded_args,
            hook_policy,
            &guardrails,
            serving_hooks_factory,
            skippy_api::serving::ModelOpenEvents::Disabled,
        )?;
        request.serving_telemetry = Some(telemetry);
        let model =
            Self::finish_load_with_open_events(request, guardrails.config, model_open_events)?;
        register_stage0_compute_meter(&model.config.run_id, &model.runtime);
        Ok(model)
    }

    fn load_prepared(
        runtime_options: EmbeddedRuntimeOptions,
        embedded_args: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        guardrails: SkippyOpenAiGuardrailOptions,
        serving_hooks_factory: Option<SharedModelServingHooksFactory>,
        model_open_events: Option<NativeModelOpenEvents>,
    ) -> Result<Self> {
        let request = Self::model_load_request(
            runtime_options,
            embedded_args,
            hook_policy,
            &guardrails,
            serving_hooks_factory,
            skippy_api::serving::ModelOpenEvents::Disabled,
        )?;
        Self::finish_load_with_open_events(request, guardrails.config, model_open_events)
    }

    fn model_load_request(
        runtime: EmbeddedRuntimeOptions,
        openai: resolver::ResolvedEmbeddedOpenAiArgs,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
        guardrails: &SkippyOpenAiGuardrailOptions,
        hooks_factory: Option<SharedModelServingHooksFactory>,
        open_events: skippy_api::serving::ModelOpenEvents,
    ) -> Result<skippy_api::serving::ModelLoadRequest> {
        Ok(skippy_api::serving::ModelLoadRequest {
            runtime,
            openai,
            open_events,
            hooks_factory,
            hook_policy,
            generation_observer: Some(Arc::new(
                runtime_events::SkippyGenerationRuntimeEventAdapter::new(),
            )),
            kv_observer: Some(Arc::new(runtime_events::SkippyKvRuntimeEventObserver::new())),
            guardrails: guardrails.config.clone(),
            guardrail_telemetry: guardrails.telemetry.guardrail_sink(),
            downstream_wire_condition: benchmark_downstream_wire_condition()?,
            serving_telemetry: None,
            l3_manager: crate::runtime::kv_disk_config::node_kv_disk_manager(),
        })
    }

    fn finish_load(
        request: skippy_api::serving::ModelLoadRequest,
        openai_guardrails: Option<OpenAiGuardrailsConfig>,
    ) -> Result<Self> {
        let loaded = request.load()?;
        Ok(Self {
            runtime: loaded.runtime,
            backend: loaded.backend,
            config: loaded.config,
            openai_guardrails,
            started_at_unix_nanos: now_unix_nanos(),
            status: Arc::new(Mutex::new(HandleState {
                state: SkippyModelState::Ready,
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            _prediction_return_listener: loaded.prediction_return_listener,
        })
    }

    fn finish_load_with_open_events(
        mut request: skippy_api::serving::ModelLoadRequest,
        openai_guardrails: Option<OpenAiGuardrailsConfig>,
        model_open_events: Option<NativeModelOpenEvents>,
    ) -> Result<Self> {
        model_open_drain::observe_model_open(
            skippy_runtime::next_operation_id(),
            model_open_events,
            |queue| {
                request.open_events = skippy_api::serving::ModelOpenEvents::Enabled(queue);
                Self::finish_load(request, openai_guardrails)
            },
        )
    }
}

fn audit_load(load: impl FnOnce() -> Result<SkippyModelHandle>) -> Result<SkippyModelHandle> {
    let mut audit = NativeSkippyStartupAudit::new();
    let model = load()?;
    audit.mark_ready();
    Ok(model)
}
