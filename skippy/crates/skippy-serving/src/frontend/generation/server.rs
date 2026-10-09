use super::request_defaults::EmbeddedOpenAiRequestDefaults;
use crate::binary_transport::PredictionReturnHub;
use crate::binary_transport::WireCondition;
use crate::frontend::GenerationLifecycleConfig;
use crate::frontend::GenerationReceiptConfig;
use crate::frontend::InferenceGuardrailsConfig;
use crate::frontend::InferenceGuardrailsStatus;
use crate::frontend::LinearProposalIngressConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::generation::GenerationConcurrencyController;
use crate::frontend::generation::GenerationServiceEstimator;
use crate::frontend::generation::InferenceBackendMode;
use crate::frontend::generation::PersistentStageLanePool;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::attach_native_mtp_draft_model;
use crate::frontend::generation::ensure_generation_concurrency_fits_lanes;
use crate::frontend::generation::open_draft_runner;
use crate::frontend::generation::prewarm_generation_sessions;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::prefill::PrefillChunkPolicy;
use crate::frontend::prefill::PrefillChunkPolicyArgs;
use crate::frontend::speculative::{SpeculativeDecodeConfig, standalone_ngram_proposal_limit};
use crate::kv_integration::KvStageIntegration;
use crate::listener::bind_serve_listener;
use crate::runtime_state::RuntimeState;
use crate::runtime_state::{loaded_memory_cache_capabilities, loaded_model_state_kind};
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use crate::thinking_probe::ThinkingProbeInputs;
use crate::thinking_probe::emit_probe_status;
use crate::thinking_probe::native_renderer_identity;
use crate::thinking_probe::probe_loaded_model;
use crate::tokenizer::{TokenizerCapability, tokenizer_http_router};
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::middleware;
use axum::middleware::Next;
use axum::response::Response;
use serde_json::Value;
use serde_json::json;
use skippy_inference_api::InferenceBackend;
use skippy_inference_api::InferenceHookPolicy;
use skippy_inference_api::ModelId;
use skippy_inference_api::thinking::ThinkingControls;
use skippy_protocol::StageConfig;
use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

/// Serve a caller-owned backend and tokenizer, awaiting all accepted HTTP work on shutdown.
pub async fn serve_openai_backend_with_shutdown(
    bind_addr: SocketAddr,
    backend: Arc<dyn InferenceBackend>,
    tokenizer: TokenizerCapability,
    telemetry: Telemetry,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let app = instrumented_openai_router(backend, tokenizer, telemetry);
    let listener = bind_serve_listener(bind_addr)?;
    skippy_events::diagnostics::emit(skippy_events::diagnostics::ServingDiagnostic::Status {
        message: format!("skippy-serving listening: openai={bind_addr}"),
    })?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

#[derive(Clone)]
pub struct EmbeddedOpenAiArgs {
    pub bind_addr: SocketAddr,
    pub config: StageConfig,
    pub runtime: Arc<Mutex<RuntimeState>>,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub request_defaults: EmbeddedOpenAiRequestDefaults,
    pub generation_concurrency: usize,
    pub continuous_batching: bool,
    /// Decode-wave groups for this frontend's dispatcher. `None` keeps the
    /// ungrouped default; `SKIPPY_PIPELINE_DECODE_GROUPS` still overrides it.
    pub pipeline_decode_groups: Option<usize>,
    pub adaptive_generation_min_concurrency: Option<usize>,
    pub generation_queue_capacity: usize,
    pub generation_admission_timeout_secs: u64,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub prefill_adaptive_target_ms: f64,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub speculative: SpeculativeDecodeConfig,
    pub native_mtp_enabled: bool,
    pub native_mtp_draft_model_path: Option<PathBuf>,
    pub native_mtp_max_tokens: usize,
    pub native_mtp_min_tokens: usize,
    pub activation_width: i32,
    pub reply_credit_limit: Option<usize>,
    pub downstream_connect_timeout_secs: u64,
    pub downstream_wire_condition: WireCondition,
    pub prediction_returns: Option<Arc<PredictionReturnHub>>,
    pub telemetry: Telemetry,
    pub hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
    pub generation_receipt: Option<GenerationReceiptConfig>,
    pub generation_lifecycle: Option<GenerationLifecycleConfig>,
    pub linear_proposal_ingress: Option<LinearProposalIngressConfig>,
    pub openai_guardrails: Option<InferenceGuardrailsConfig>,
    pub kv_lifecycle_observer: Option<Arc<dyn crate::kv_integration::KvLifecycleObserver>>,
    /// Node-scoped durable disk-cache owner supplied by the embedding host.
    /// `None` keeps standalone and cache-disabled launches in-memory only.
    pub l3_manager: Option<skippy_cache::L3CacheManager>,
}

pub async fn serve_embedded_openai(args: EmbeddedOpenAiArgs) -> Result<()> {
    serve_embedded_openai_with_shutdown(args, std::future::pending::<()>()).await
}

pub(crate) async fn serve_embedded_openai_with_scheduler(
    args: EmbeddedOpenAiArgs,
    iteration_scheduler: IterationScheduler,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_embedded_openai_with_shutdown_and_scheduler(args, shutdown, Some(iteration_scheduler))
        .await
}

pub async fn serve_embedded_openai_with_shutdown(
    args: EmbeddedOpenAiArgs,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_embedded_openai_with_shutdown_and_scheduler(args, shutdown, None).await
}

async fn serve_embedded_openai_with_shutdown_and_scheduler(
    args: EmbeddedOpenAiArgs,
    shutdown: impl Future<Output = ()> + Send + 'static,
    iteration_scheduler: Option<IterationScheduler>,
) -> Result<()> {
    let bind_addr = args.bind_addr;
    let binding = embedded_openai_router_with_scheduler(args, iteration_scheduler)?;

    skippy_events::diagnostics::emit(skippy_events::diagnostics::ServingDiagnostic::Status {
        message: format!(
            "skippy-serving listening: openai={} model_id={} backend=embedded-stage0 generation_concurrency={} generation_queue_capacity={} generation_admission_timeout_secs={}",
            bind_addr,
            binding.model_id,
            binding.generation_concurrency,
            binding.generation_queue_capacity,
            binding.generation_admission_timeout_secs,
        ),
    })?;

    let listener = bind_serve_listener(bind_addr)?;
    axum::serve(listener, binding.router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub struct EmbeddedOpenAiRouter {
    pub router: Router,
    pub model_id: String,
    pub generation_concurrency: usize,
    pub generation_queue_capacity: usize,
    pub generation_admission_timeout_secs: u64,
}

pub struct EmbeddedOpenAiBackend {
    pub backend: Arc<dyn InferenceBackend>,
    pub model_id: String,
    pub thinking: Option<ThinkingControls>,
    pub generation_concurrency: usize,
    pub generation_queue_capacity: usize,
    pub generation_admission_timeout_secs: u64,
    pub openai_guardrails: Option<InferenceGuardrailsStatus>,
}

pub fn embedded_openai_router(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiRouter> {
    embedded_openai_router_with_scheduler(args, None)
}

fn embedded_openai_router_with_scheduler(
    args: EmbeddedOpenAiArgs,
    iteration_scheduler: Option<IterationScheduler>,
) -> Result<EmbeddedOpenAiRouter> {
    let telemetry = args.telemetry.clone();
    let tokenizer = TokenizerCapability::from_stage_zero(&args.config, args.runtime.clone())
        .context("construct stage-0 tokenizer capability for embedded OpenAI serving")?;
    let binding = embedded_openai_backend_with_scheduler(args, iteration_scheduler)?;
    let router = instrumented_openai_router(binding.backend.clone(), tokenizer, telemetry);

    Ok(EmbeddedOpenAiRouter {
        router,
        model_id: binding.model_id,
        generation_concurrency: binding.generation_concurrency,
        generation_queue_capacity: binding.generation_queue_capacity,
        generation_admission_timeout_secs: binding.generation_admission_timeout_secs,
    })
}

pub fn embedded_openai_backend(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiBackend> {
    embedded_openai_backend_with_scheduler(args, None)
}

fn embedded_openai_backend_with_scheduler(
    args: EmbeddedOpenAiArgs,
    iteration_scheduler: Option<IterationScheduler>,
) -> Result<EmbeddedOpenAiBackend> {
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--generation-concurrency must be greater than zero");
    }
    ensure_generation_concurrency_fits_lanes(
        args.generation_concurrency,
        args.config.lane_count,
        "--generation-concurrency",
    )?;
    if args.draft_model_path.is_some() && args.speculative_window == 0 {
        bail!("--speculative-window must be greater than zero when a draft model is set");
    }
    if args.native_mtp_draft_model_path.is_some() && !args.native_mtp_enabled {
        bail!("native MTP must be enabled when an MTP draft model is set");
    }
    validate_generation_receipt_topology(
        args.generation_receipt.is_some(),
        args.config.upstream.is_some(),
        args.config.downstream.is_some(),
    )?;
    // Recurrent verify windows are supported by the native runtime's bounded
    // recurrent checkpoints and accepted-prefix replay. Keep admission aligned
    // with that recovery contract instead of rejecting these models up front.
    if args.config.stage_index != 0
        || args.config.layer_start != 0
        || args.config.upstream.is_some()
    {
        bail!("embedded OpenAI serving is only supported on stage 0");
    }
    attach_native_mtp_draft_model(
        args.native_mtp_draft_model_path.as_deref(),
        &args.runtime,
        &args.config,
        args.draft_n_gpu_layers,
        &args.speculative,
    )?;
    let draft = open_draft_runner(
        args.draft_model_path.as_deref(),
        &args.config,
        args.draft_n_gpu_layers,
        args.speculative_window,
        &args.speculative,
    )?;
    // Render-only probe of the selected chat template, run once at load. This is
    // where every input is in hand: the loaded runtime, the stage config, and the
    // selected template. `/v1/models` publishes the result as `thinking`.
    let thinking = probe_thinking_controls(&args);
    let model_id = ModelId::new(
        args.model_id
            .unwrap_or_else(|| args.config.model_id.clone()),
    )
    .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
    .into_string();
    let prefill_chunk_policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: &args.prefill_chunk_policy,
        schedule: args.prefill_chunk_schedule.as_deref(),
        fixed_chunk_size: args.prefill_chunk_size,
        adaptive_start: args.prefill_adaptive_start,
        adaptive_step: args.prefill_adaptive_step,
        adaptive_max: args.prefill_adaptive_max,
        adaptive_target_ms: args.prefill_adaptive_target_ms,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })?;
    let mode = if args.config.downstream.is_none() {
        InferenceBackendMode::LocalRuntime
    } else {
        let lane_pool = PersistentStageLanePool::new(
            &args.config,
            args.generation_concurrency,
            args.downstream_connect_timeout_secs,
            args.telemetry.clone(),
        )
        .context("create embedded OpenAI persistent downstream lanes")?;
        let prefill_reply_credit_limit = args.reply_credit_limit.unwrap_or(3);
        InferenceBackendMode::EmbeddedStageZero {
            config: args.config.clone(),
            prefill_chunk_policy,
            activation_width: args.activation_width,
            downstream_wire_condition: args.downstream_wire_condition,
            prefill_reply_credit_limit,
            lane_pool,
            prediction_returns: args.prediction_returns.clone(),
        }
    };
    let mut server_start_attrs = lifecycle_attrs(&args.config);
    insert_generation_admission_config_attrs(
        &mut server_start_attrs,
        args.generation_concurrency,
        args.adaptive_generation_min_concurrency,
        args.generation_queue_capacity,
        args.generation_admission_timeout_secs,
    );
    args.telemetry
        .emit("stage.openai_server_start", server_start_attrs);
    prewarm_generation_sessions(
        &args.runtime,
        args.generation_concurrency,
        &args.telemetry,
        &args.config,
        "stage.openai_runtime_prewarm",
    )
    .context("prewarm embedded OpenAI runtime sessions")?;
    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &args.config,
        loaded_model_state_kind(Some(&args.runtime)),
        loaded_memory_cache_capabilities(Some(&args.runtime)),
        args.l3_manager.clone(),
        args.kv_lifecycle_observer.clone(),
    )?
    .map(Arc::new);
    if let Some(kv) = kv.as_ref() {
        let mut attrs = lifecycle_attrs(&args.config);
        attrs.extend(
            kv.attrs()
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value)),
        );
        args.telemetry.emit("stage.kv_payload_selected", attrs);
    }
    let ctx_size = usize::try_from(args.config.ctx_size).unwrap_or(usize::MAX);
    let iteration_scheduler = match iteration_scheduler {
        Some(iteration_scheduler) => iteration_scheduler,
        None => IterationScheduler::new(
            args.runtime.clone(),
            &args.config,
            args.generation_concurrency,
            args.continuous_batching,
            args.pipeline_decode_groups,
            args.telemetry.clone(),
        )?,
    };
    let backend: Arc<dyn InferenceBackend> = Arc::new(StageOpenAiBackend {
        runtime: args.runtime,
        workload: Default::default(),
        config: args.config.clone(),
        telemetry: args.telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        request_defaults: args.request_defaults,
        thinking: thinking.clone(),
        ctx_size,
        mode,
        draft,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        ngram_max: standalone_ngram_proposal_limit(&args.speculative),
        // Opt-in, and only when the plan actually speculates. With speculation
        // off there is nothing to stand down, and trialling it on would turn a
        // deliberate `strategy = "disabled"` into something that flips back on
        // by itself.
        speculation_governor: (crate::frontend::speculation_gate::speculation_gate_enabled(
            args.speculative.gate,
        ) && crate::frontend::speculation_gate::speculation_plan_is_active(
            &args.speculative,
        ))
        .then(|| {
            std::sync::Arc::new(crate::frontend::speculation_gate::SpeculationGovernor::new(
                args.speculative.gate.into(),
                true,
            ))
        }),
        // Only when the plan declines to state a number. A stated budget,
        // including a stated zero, is the operator's and is not searched.
        runahead_governor: args.speculative.verify_window.runahead_auto.then(|| {
            std::sync::Arc::new(crate::frontend::runahead_search::RunaheadGovernor::new(
                crate::frontend::runahead_search::RunaheadSearchConfig::default(),
            ))
        }),
        speculative: args.speculative,
        generation_limit: Arc::new(match args.adaptive_generation_min_concurrency {
            Some(initial_limit) => GenerationConcurrencyController::adaptive(
                args.generation_concurrency,
                initial_limit,
            ),
            None => GenerationConcurrencyController::fixed(args.generation_concurrency),
        }),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_queue_capacity,
        generation_admission_timeout: Duration::from_secs(args.generation_admission_timeout_secs),
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(
            args.generation_concurrency,
        )),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: args.hook_policy,
        generation_receipt: args.generation_receipt,
        generation_lifecycle: args.generation_lifecycle,
        linear_proposal_ingress: args.linear_proposal_ingress,
        kv,
        iteration_scheduler,
    });
    let openai_guardrails = args
        .openai_guardrails
        .as_ref()
        .map(InferenceGuardrailsConfig::status);
    let backend = args
        .openai_guardrails
        .as_ref()
        .map_or(backend.clone(), |guardrails| {
            guardrails.wrap_backend_with_context_limit(backend, Some(ctx_size))
        });

    Ok(EmbeddedOpenAiBackend {
        backend,
        model_id,
        thinking,
        generation_concurrency: args.generation_concurrency,
        generation_queue_capacity: args.generation_queue_capacity,
        generation_admission_timeout_secs: args.generation_admission_timeout_secs,
        openai_guardrails,
    })
}

/// Renders the selected template to learn which reasoning controls it reacts to.
///
/// Never generates: see [`crate::thinking_probe`] for what the observations can
/// and cannot claim. A poisoned runtime lock leaves the model unprobed, which is
/// reported as absent rather than as a default.
fn probe_thinking_controls(args: &EmbeddedOpenAiArgs) -> Option<ThinkingControls> {
    let artifact = args
        .config
        .source_model_sha256
        .clone()
        .or_else(|| args.config.package_ref.clone());
    let report = probe_loaded_model(
        &args.runtime,
        &ThinkingProbeInputs {
            defaults: &args.request_defaults,
            model_id: &args.config.model_id,
            artifact: artifact.as_deref(),
            template_override: args.request_defaults.chat_template.as_deref(),
            renderer: &native_renderer_identity(),
        },
    )?;
    let _ = emit_probe_status(&report);
    Some(report.controls().clone())
}

fn validate_generation_receipt_topology(
    receipt_enabled: bool,
    has_upstream: bool,
    has_downstream: bool,
) -> Result<()> {
    if receipt_enabled && (has_upstream || has_downstream) {
        bail!("generation receipts are supported only for local single-stage execution");
    }
    Ok(())
}

pub fn resolve_adaptive_generation_min_concurrency(
    enabled: bool,
    configured_minimum: Option<usize>,
    hard_limit: usize,
    minimum_arg: &str,
) -> Result<Option<usize>> {
    if !enabled {
        if configured_minimum.is_some() {
            bail!("{minimum_arg} requires adaptive generation concurrency to be enabled");
        }
        return Ok(None);
    }
    let minimum = configured_minimum.unwrap_or(1);
    if minimum == 0 {
        bail!("{minimum_arg} must be greater than zero");
    }
    if minimum > hard_limit {
        bail!("{minimum_arg} ({minimum}) exceeds generation concurrency ({hard_limit})");
    }
    Ok(Some(minimum))
}

fn insert_generation_admission_config_attrs(
    attrs: &mut BTreeMap<String, Value>,
    generation_concurrency: usize,
    adaptive_generation_min_concurrency: Option<usize>,
    generation_queue_capacity: usize,
    generation_admission_timeout_secs: u64,
) {
    attrs.insert(
        "llama_stage.generation_concurrency".to_string(),
        json!(generation_concurrency),
    );
    attrs.insert(
        "llama_stage.adaptive_generation_concurrency".to_string(),
        json!(adaptive_generation_min_concurrency.is_some()),
    );
    if let Some(minimum) = adaptive_generation_min_concurrency {
        attrs.insert(
            "llama_stage.adaptive_generation_min_concurrency".to_string(),
            json!(minimum),
        );
    }
    attrs.insert(
        "llama_stage.generation_queue_capacity".to_string(),
        json!(generation_queue_capacity),
    );
    attrs.insert(
        "llama_stage.generation_admission_timeout_secs".to_string(),
        json!(generation_admission_timeout_secs),
    );
}

pub(in crate::frontend) fn instrumented_openai_router(
    backend: Arc<dyn InferenceBackend>,
    tokenizer: TokenizerCapability,
    telemetry: Telemetry,
) -> Router {
    skippy_inference_api::router_for(backend)
        .merge(tokenizer_http_router(tokenizer))
        .layer(middleware::from_fn_with_state(
            telemetry,
            openai_http_telemetry,
        ))
}

pub(in crate::frontend) async fn openai_http_telemetry(
    State(telemetry): State<Telemetry>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let timer = PhaseTimer::start();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    let mut attrs = BTreeMap::from([
        ("llama_stage.http_method".to_string(), json!(method)),
        ("llama_stage.http_path".to_string(), json!(path)),
        ("llama_stage.http_status".to_string(), json!(status)),
    ]);
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        "stage.openai_http_request",
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    response
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_adaptive_generation_min_concurrency, validate_generation_receipt_topology,
    };
    use crate::frontend::CompositeGenerationLifecycleIngress;
    use crate::frontend::GenerationLifecycleConfig;
    use crate::frontend::GenerationReceiptConfig;
    use crate::serving_hooks::ModelServingHooks;
    use std::sync::Arc;

    #[test]
    fn generation_receipts_require_local_single_stage_topology() {
        assert!(validate_generation_receipt_topology(true, false, false).is_ok());
        assert!(validate_generation_receipt_topology(false, true, true).is_ok());
        assert!(validate_generation_receipt_topology(true, true, false).is_err());
        assert!(validate_generation_receipt_topology(true, false, true).is_err());
        assert!(validate_generation_receipt_topology(true, true, true).is_err());
    }

    #[test]
    fn lifecycle_only_hooks_do_not_enable_split_receipt_validation() {
        let hooks = ModelServingHooks::default().with_generation_lifecycle(
            GenerationLifecycleConfig::from_ingress(Arc::new(
                CompositeGenerationLifecycleIngress::new(Vec::new()),
            )),
        );

        assert!(hooks.generation_receipt().is_none());
        assert!(
            validate_generation_receipt_topology(hooks.generation_receipt().is_some(), true, true,)
                .is_ok()
        );

        let exact_hooks = ModelServingHooks::default().with_generation_receipt(
            GenerationReceiptConfig::from_lifecycle_ingress(Arc::new(
                CompositeGenerationLifecycleIngress::new(Vec::new()),
            ))
            .with_full_state_digest(true),
        );
        let exact_config = exact_hooks
            .generation_receipt()
            .expect("exact receipt config");
        assert!(exact_config.exports_full_state());
        assert!(
            validate_generation_receipt_topology(
                exact_hooks.generation_receipt().is_some(),
                true,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn adaptive_generation_minimum_is_explicit_and_bounded() {
        assert_eq!(
            resolve_adaptive_generation_min_concurrency(true, None, 8, "--minimum")
                .expect("default minimum"),
            Some(1)
        );
        assert_eq!(
            resolve_adaptive_generation_min_concurrency(false, None, 8, "--minimum")
                .expect("fixed mode"),
            None
        );
        assert!(
            resolve_adaptive_generation_min_concurrency(false, Some(2), 8, "--minimum").is_err()
        );
        assert!(
            resolve_adaptive_generation_min_concurrency(true, Some(9), 8, "--minimum").is_err()
        );
    }
}
