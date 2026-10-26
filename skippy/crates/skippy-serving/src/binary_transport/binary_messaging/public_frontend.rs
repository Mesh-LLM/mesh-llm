//! Launch the public API for a binary stage with operator request and compatibility settings.
use super::*;
use crate::runtime_state::RuntimeState;

pub(super) struct PublicFrontendLaunch {
    pub config: StageConfig,
    pub runtime: Arc<Mutex<RuntimeState>>,
    pub iteration_scheduler: IterationScheduler,
    pub telemetry: Telemetry,
    pub prediction_returns: Arc<PredictionReturnHub>,
    pub shutdown_requested: Arc<AtomicBool>,
    pub openai: Option<super::super::EmbeddedOpenAiStageOptions>,
    pub continuous_batching: bool,
    pub native_mtp_enabled: bool,
    pub output_activation_width: i32,
    pub reply_credit_limit: Option<usize>,
    pub downstream_connect_timeout_secs: u64,
    pub downstream_wire_condition: super::super::WireCondition,
    pub l3_manager: Option<skippy_cache::L3CacheManager>,
    pub tuning: crate::settings::ServingTuning,
}

pub(super) fn start(launch: PublicFrontendLaunch) -> Result<Option<EmbeddedFrontendTask>> {
    let PublicFrontendLaunch {
        config,
        runtime,
        iteration_scheduler,
        telemetry,
        prediction_returns,
        shutdown_requested,
        openai,
        continuous_batching,
        native_mtp_enabled,
        output_activation_width,
        reply_credit_limit,
        downstream_connect_timeout_secs,
        downstream_wire_condition,
        l3_manager,
        tuning,
    } = launch;
    let mut frontend_task = None;
    if let Some(openai_options) = openai {
        if config.stage_index != 0 || config.layer_start != 0 {
            bail!("--bind-addr is only supported on stage 0");
        }
        let openai_config = config.clone();
        let openai_runtime = runtime.clone();
        let openai_iteration_scheduler = iteration_scheduler.clone();
        let openai_telemetry = telemetry.clone();
        let openai_prediction_returns = prediction_returns.clone();
        frontend_task = Some(EmbeddedFrontendTask(Some(tokio::spawn(async move {
            frontend::serve_embedded_openai_with_scheduler(
                EmbeddedOpenAiArgs {
                    bind_addr: openai_options.bind_addr,
                    config: openai_config,
                    runtime: openai_runtime,
                    model_id: openai_options.model_id,
                    default_max_tokens: openai_options.default_max_tokens,
                    request_defaults: tuning.request_defaults,
                    generation_concurrency: openai_options.generation_concurrency,
                    continuous_batching,
                    pipeline_decode_groups: openai_options.pipeline_decode_groups,
                    adaptive_generation_min_concurrency: openai_options
                        .adaptive_generation_min_concurrency,
                    generation_queue_capacity: openai_options.generation_queue_capacity,
                    generation_admission_timeout_secs: openai_options
                        .generation_admission_timeout_secs,
                    prefill_chunk_size: openai_options.prefill_chunk_size,
                    prefill_chunk_policy: openai_options.prefill_chunk_policy,
                    prefill_chunk_schedule: openai_options.prefill_chunk_schedule,
                    prefill_adaptive_start: openai_options.prefill_adaptive_start,
                    prefill_adaptive_step: openai_options.prefill_adaptive_step,
                    prefill_adaptive_max: openai_options.prefill_adaptive_max,
                    prefill_adaptive_target_ms: openai_options.prefill_adaptive_target_ms,
                    draft_model_path: openai_options.draft_model_path,
                    speculative_window: openai_options.speculative_window,
                    adaptive_speculative_window: openai_options.adaptive_speculative_window,
                    draft_n_gpu_layers: openai_options.draft_n_gpu_layers,
                    speculative: openai_options.speculative.clone(),
                    native_mtp_enabled: native_mtp_enabled
                        && openai_options.speculative.native_mtp.enabled,
                    native_mtp_draft_model_path: openai_options.native_mtp_draft_model_path,
                    native_mtp_max_tokens: openai_options.native_mtp_max_tokens,
                    native_mtp_min_tokens: openai_options.native_mtp_min_tokens,
                    activation_width: output_activation_width,
                    reply_credit_limit,
                    downstream_connect_timeout_secs,
                    downstream_wire_condition,
                    prediction_returns: Some(openai_prediction_returns),
                    telemetry: openai_telemetry,
                    hook_policy: None,
                    generation_receipt: None,
                    generation_lifecycle: None,
                    linear_proposal_ingress: None,
                    kv_lifecycle_observer: None,
                    openai_guardrails: tuning.guardrails.or_else(|| {
                        Some(frontend::InferenceGuardrailsConfig::disabled_for_skippy())
                    }),
                    l3_manager,
                },
                openai_iteration_scheduler,
                wait_for_shutdown(shutdown_requested),
            )
            .await
        }))));
    }
    Ok(frontend_task)
}
