use crate::binary_transport::PredictionReturnHub;
use crate::binary_transport::WireCondition;
use crate::cli::ServeOpenAiArgs;
use crate::config::load_json;
use crate::config::validate_config;
use crate::frontend::GenerationReceiptConfig;
use crate::frontend::LinearProposalIngressConfig;
use crate::frontend::OpenAiGuardrailsConfig;
use crate::frontend::OpenAiGuardrailsStatus;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::generation::OpenAiBackendMode;
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
use crate::frontend::speculative::{
    SpeculativeDecodeConfig, load_standalone_speculative_config, standalone_ngram_proposal_limit,
};
use crate::kv_integration::KvStageIntegration;
use crate::runtime_state::RuntimeState;
use crate::runtime_state::load_runtime;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
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
use openai_frontend::ModelId;
use openai_frontend::OpenAiBackend;
use openai_frontend::OpenAiHookPolicy;
use openai_frontend::ReasoningEffort;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::StageTopology;
use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

pub async fn serve_openai(args: ServeOpenAiArgs) -> Result<()> {
    let config = load_json::<StageConfig>(&args.config)
        .with_context(|| format!("load stage config {}", args.config.display()))?;
    let topology = match args.topology.as_ref() {
        Some(path) => Some(
            load_json::<StageTopology>(path)
                .with_context(|| format!("load topology {}", path.display()))?,
        ),
        None => None,
    };
    validate_config(&config, topology.as_ref())?;
    if args.first_stage_addr.is_none() && config.downstream.is_some() {
        bail!("serve-openai local backend requires a final/single-stage config with no downstream");
    }
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--generation-concurrency must be greater than zero");
    }
    let speculative = load_standalone_speculative_config(args.speculative_config.as_ref())?;

    let runtime = load_runtime(&config)?.ok_or_else(|| {
        anyhow!("serve-openai requires a stage config with model_path for tokenization and decode")
    })?;
    let model_id = ModelId::new(args.model_id.unwrap_or_else(|| config.model_id.clone()))
        .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
        .into_string();
    if args.first_stage_addr.is_some() {
        bail!(
            "--first-stage-addr is no longer supported; direct prediction return requires embedded stage-0 OpenAI serving via serve-binary --openai-bind-addr"
        );
    }
    let mode = OpenAiBackendMode::LocalRuntime;
    let mode_label = mode.label();
    let telemetry = Telemetry::new(
        args.metrics_otlp_grpc,
        args.telemetry_queue_capacity,
        config.clone(),
        args.telemetry_level,
    );
    telemetry.emit("stage.openai_server_start", lifecycle_attrs(&config));
    if matches!(&mode, OpenAiBackendMode::LocalRuntime) {
        ensure_generation_concurrency_fits_lanes(
            args.generation_concurrency,
            config.lane_count,
            "--generation-concurrency",
        )?;
        prewarm_generation_sessions(
            &runtime,
            args.generation_concurrency,
            &telemetry,
            &config,
            "stage.openai_runtime_prewarm",
        )
        .context("prewarm OpenAI runtime sessions")?;
    }
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    let iteration_scheduler = IterationScheduler::new(
        runtime.clone(),
        &config,
        args.generation_concurrency,
        true,
        telemetry.clone(),
    )?;
    let tokenizer = TokenizerCapability::from_stage_zero(&config, runtime.clone())
        .context("construct stage-0 tokenizer capability for OpenAI serving")?;
    let backend: Arc<dyn OpenAiBackend> = Arc::new(StageOpenAiBackend {
        runtime,
        config,
        telemetry: telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: standalone_ngram_proposal_limit(&speculative),
        speculative,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        generation_receipt: None,
        linear_proposal_ingress: None,
        kv,
        iteration_scheduler,
    });
    let backend = OpenAiGuardrailsConfig::for_standalone_mode(args.openai_guardrails)
        .wrap_backend_with_context_limit(backend, Some(ctx_size));
    let app: Router = instrumented_openai_router(backend, tokenizer, telemetry.clone());

    println!(
        "skippy-server listening: openai={} model_id={} backend={} generation_concurrency={}",
        args.bind_addr, model_id, mode_label, args.generation_concurrency,
    );

    let listener = TcpListener::bind(args.bind_addr).await?;
    axum::serve(listener, app).await?;
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
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
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
    pub hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    pub generation_receipt: Option<GenerationReceiptConfig>,
    pub linear_proposal_ingress: Option<LinearProposalIngressConfig>,
    pub openai_guardrails: Option<OpenAiGuardrailsConfig>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmbeddedOpenAiRequestDefaults {
    pub stop: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub reasoning_format: Option<EmbeddedReasoningFormat>,
    pub reasoning_enabled: Option<EmbeddedReasoningEnabled>,
    pub reasoning_budget: Option<EmbeddedReasoningBudget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningFormat {
    Auto,
    None,
    Deepseek,
    DeepseekLegacy,
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningEnabled {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningBudget {
    Auto,
    Tokens(u32),
    Effort(ReasoningEffort),
}

pub async fn serve_embedded_openai(args: EmbeddedOpenAiArgs) -> Result<()> {
    serve_embedded_openai_with_shutdown(args, std::future::pending::<()>()).await
}

pub(crate) async fn serve_embedded_openai_with_scheduler(
    args: EmbeddedOpenAiArgs,
    iteration_scheduler: IterationScheduler,
) -> Result<()> {
    serve_embedded_openai_with_shutdown_and_scheduler(
        args,
        std::future::pending::<()>(),
        Some(iteration_scheduler),
    )
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

    println!(
        "skippy-server listening: openai={} model_id={} backend=embedded-stage0 generation_concurrency={}",
        bind_addr, binding.model_id, binding.generation_concurrency,
    );

    let listener = TcpListener::bind(bind_addr).await?;
    axum::serve(listener, binding.router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub struct EmbeddedOpenAiRouter {
    pub router: Router,
    pub model_id: String,
    pub generation_concurrency: usize,
}

pub struct EmbeddedOpenAiBackend {
    pub backend: Arc<dyn OpenAiBackend>,
    pub model_id: String,
    pub generation_concurrency: usize,
    pub openai_guardrails: Option<OpenAiGuardrailsStatus>,
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
        bail!("--openai-prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--openai-generation-concurrency must be greater than zero");
    }
    ensure_generation_concurrency_fits_lanes(
        args.generation_concurrency,
        args.config.lane_count,
        "--openai-generation-concurrency",
    )?;
    if args.draft_model_path.is_some() && args.speculative_window == 0 {
        bail!("--openai-speculative-window must be greater than zero when a draft model is set");
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
    if args.config.stage_index != 0 || args.config.layer_start != 0 {
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
    let model_id = ModelId::new(
        args.model_id
            .unwrap_or_else(|| args.config.model_id.clone()),
    )
    .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
    .into_string();
    let lane_pool = PersistentStageLanePool::new(
        &args.config,
        args.generation_concurrency,
        args.downstream_connect_timeout_secs,
        args.telemetry.clone(),
    )
    .context("create embedded OpenAI persistent downstream lanes")?;
    let prefill_reply_credit_limit = args.reply_credit_limit.unwrap_or(3);
    let mode = OpenAiBackendMode::EmbeddedStageZero {
        config: args.config.clone(),
        prefill_chunk_policy: PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
            policy: &args.prefill_chunk_policy,
            schedule: args.prefill_chunk_schedule.as_deref(),
            fixed_chunk_size: args.prefill_chunk_size,
            adaptive_start: args.prefill_adaptive_start,
            adaptive_step: args.prefill_adaptive_step,
            adaptive_max: args.prefill_adaptive_max,
            schedule_arg: "--openai-prefill-chunk-schedule",
            policy_arg: "--openai-prefill-chunk-policy",
        })?,
        activation_width: args.activation_width,
        downstream_wire_condition: args.downstream_wire_condition,
        prefill_reply_credit_limit,
        lane_pool,
        prediction_returns: args.prediction_returns.clone(),
    };
    args.telemetry
        .emit("stage.openai_server_start", lifecycle_attrs(&args.config));
    prewarm_generation_sessions(
        &args.runtime,
        args.generation_concurrency,
        &args.telemetry,
        &args.config,
        "stage.openai_runtime_prewarm",
    )
    .context("prewarm embedded OpenAI runtime sessions")?;
    let kv = KvStageIntegration::from_config(&args.config)?.map(Arc::new);
    let ctx_size = usize::try_from(args.config.ctx_size).unwrap_or(usize::MAX);
    let iteration_scheduler = match iteration_scheduler {
        Some(iteration_scheduler) => iteration_scheduler,
        None => IterationScheduler::new(
            args.runtime.clone(),
            &args.config,
            args.generation_concurrency,
            args.continuous_batching,
            args.telemetry.clone(),
        )?,
    };
    let backend: Arc<dyn OpenAiBackend> = Arc::new(StageOpenAiBackend {
        runtime: args.runtime,
        config: args.config.clone(),
        telemetry: args.telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        request_defaults: args.request_defaults,
        ctx_size,
        mode,
        draft,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        ngram_max: standalone_ngram_proposal_limit(&args.speculative),
        speculative: args.speculative,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: args.hook_policy,
        generation_receipt: args.generation_receipt,
        linear_proposal_ingress: args.linear_proposal_ingress,
        kv,
        iteration_scheduler,
    });
    let openai_guardrails = args
        .openai_guardrails
        .as_ref()
        .map(OpenAiGuardrailsConfig::status);
    let backend = args
        .openai_guardrails
        .as_ref()
        .map_or(backend.clone(), |guardrails| {
            guardrails.wrap_backend_with_context_limit(backend, Some(ctx_size))
        });

    Ok(EmbeddedOpenAiBackend {
        backend,
        model_id,
        generation_concurrency: args.generation_concurrency,
        openai_guardrails,
    })
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

pub(in crate::frontend) fn instrumented_openai_router(
    backend: Arc<dyn OpenAiBackend>,
    tokenizer: TokenizerCapability,
    telemetry: Telemetry,
) -> Router {
    openai_frontend::router_for(backend)
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
    use super::validate_generation_receipt_topology;

    #[test]
    fn generation_receipts_require_local_single_stage_topology() {
        assert!(validate_generation_receipt_topology(true, false, false).is_ok());
        assert!(validate_generation_receipt_topology(false, true, true).is_ok());
        assert!(validate_generation_receipt_topology(true, true, false).is_err());
        assert!(validate_generation_receipt_topology(true, false, true).is_err());
        assert!(validate_generation_receipt_topology(true, true, true).is_err());
    }
}
