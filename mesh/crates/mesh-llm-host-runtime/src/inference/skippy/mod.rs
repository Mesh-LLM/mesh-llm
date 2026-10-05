#![allow(dead_code)]

#[cfg(test)]
use std::path::PathBuf;

mod certification;
mod deployment;
pub(crate) mod diagnostics;
mod hooks;
mod loading;
mod local_source;
mod materialization;
pub(crate) mod metal_pipeline_cache;
mod model_capabilities;
mod model_open_drain;
mod package;
mod projector;
#[cfg(test)]
mod resolver_host_tests;
use mesh_llm_skippy_adapter::config as resolver;
pub(crate) use projector::materialize_projector_url;
pub(crate) mod runtime_events;
mod split_certification;
mod stage;
mod topology;

use crate::runtime::{
    NativeSkippyOperationalEvent, record_native_skippy_operational_event, survey,
};
use std::{
    env,
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use skippy_inference_api::CompactionConfig;
#[cfg(test)]
use skippy_serving::InferenceGuardrailsTarget;

use anyhow::{Context, Result};
use async_trait::async_trait;
use skippy_inference_api::{
    AudioResponse, AudioSpeechRequest, AudioTranscriptionRequest, AudioTranscriptionResponse,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, CompletionRequest,
    CompletionResponse, CompletionStream, EmbeddingResponse, EmbeddingsRequest, GuardrailMode,
    GuardrailPolicy, GuardrailPolicyHandle, InferenceBackend, InferenceHookPolicy,
    InferenceRequestContext, InferenceResult, ModelObject, RerankRequest, RerankResponse,
};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};
use skippy_runtime::{ModelInfo, MtpSource};
use skippy_serving::serving_hooks::SharedModelServingHooksFactory;
use skippy_serving::{
    EmbeddedRuntimeOptions, EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedState,
    InferenceGuardrailsConfig, InferenceGuardrailsStatus, SkippyRuntimeHandle,
    binary_transport::PredictionReturnListener, binary_transport::WireCondition,
};

pub use certification::{
    CertificationGateStatus, SkippyCertificationRequest, certify_layer_package,
};
pub(crate) use hooks::MeshAutoHookPolicy;
#[cfg(test)]
pub(crate) use local_source::local_source_required_for_model;
pub(crate) use local_source::{
    apply_verified_local_source, effective_local_source_required, into_content_addressed_identity,
    is_content_addressed_gguf_ref, register_local_source_policy, unregister_local_source_policy,
    verify_registered_content_source,
};
#[cfg(test)]
pub(crate) use materialization::resolve_package_v2_stage_to_local;
pub use materialization::{
    configure_materialized_stage_cache, download_package_v2_to_local, is_layer_package_ref,
    materialized_stage_cache_dir, materialized_stages_for_sources,
    prune_unpinned_materialized_stages, remove_materialized_stages_for_sources,
    resolve_hf_package_to_local, resolve_package_v2_full_model_to_local,
    resolve_stage_load_package,
};
pub(crate) use mesh_llm_skippy_adapter::{
    KvCachePolicy, SkippyDeviceDescriptor, SkippyModelLoadOptions, SkippyTelemetryOptions,
    single_stage_config,
};
#[cfg(test)]
pub(crate) use package::write_test_package_v2_fixture;
pub use package::{
    SkippyPackageIdentity, identity_from_layer_package, identity_from_package_v2,
    synthetic_direct_gguf_package,
};
pub(crate) use package::{
    direct_gguf_planning_manifest_from_identity, direct_gguf_source_paths, is_package_v2_ref,
    synthetic_content_addressed_gguf_package, synthetic_huggingface_gguf_package,
};
pub(crate) use resolver::{
    ResolvedEmbeddedOpenAiArgs, ResolvedSkippyConfig, SkippyConfigResolveRequest,
    effective_safety_margin_bytes, resolve_skippy_config_for_selector,
    resolve_skippy_config_for_selector_with_publisher_defaults,
};
pub(crate) use skippy_api::family_policy;
pub(crate) use skippy_api::family_policy::family_policy_for_stage_config;
pub(crate) use skippy_serving::InferenceGuardrailsStatus as SkippyOpenAiGuardrailsStatus;
pub(crate) use split_certification::{SplitCertificationAdmission, require_split_certification};
#[cfg(test)]
pub(crate) use stage::test_stage_admission;
pub(crate) use stage::{
    LayerRange, SourceModelKind, StageControlCommand, StageControlHandle, StageControlRequest,
    StageControlResponse, StageCoordinatorClaim, StageCoordinatorClaimAck, StageInventoryRequest,
    StageLayerInventory, StageLoadRequest, StageLoadRuntimeSettings, StagePeerDescriptor,
    StageReadyResponse, StageRuntimeState, StageStatusFilter, StageStatusSnapshot,
    StageStopRequest, StageTopologyStageDescriptor, spawn_stage_control_loop, stage_load_timeout,
};
pub(crate) use stage::{admitted_activation_frontier, admitted_resident_tensor_names};
#[cfg(test)]
pub(crate) use topology::{StageTopologyParticipant, plan_package_identity_topology};

const BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV: &str = "MESH_LLM_BENCH_DOWNSTREAM_WIRE_DELAY_MS";
const BENCH_DOWNSTREAM_WIRE_JITTER_MS_ENV: &str = "MESH_LLM_BENCH_DOWNSTREAM_WIRE_JITTER_MS";
const BENCH_DOWNSTREAM_WIRE_STALL_MS_ENV: &str = "MESH_LLM_BENCH_DOWNSTREAM_WIRE_STALL_MS";
const BENCH_DOWNSTREAM_WIRE_STALL_P_ENV: &str = "MESH_LLM_BENCH_DOWNSTREAM_WIRE_STALL_P";

fn benchmark_downstream_wire_condition() -> Result<WireCondition> {
    let delay_ms = parse_benchmark_wire_env(BENCH_DOWNSTREAM_WIRE_DELAY_MS_ENV)?;
    let jitter_ms = parse_benchmark_wire_env(BENCH_DOWNSTREAM_WIRE_JITTER_MS_ENV)?;
    let stall_ms = parse_benchmark_wire_env(BENCH_DOWNSTREAM_WIRE_STALL_MS_ENV)?;
    let stall_p = parse_benchmark_wire_env(BENCH_DOWNSTREAM_WIRE_STALL_P_ENV)?;
    WireCondition::with_jitter(delay_ms, None, jitter_ms, stall_ms, stall_p)
}

fn parse_benchmark_wire_env(name: &'static str) -> Result<f64> {
    match env::var(name) {
        Ok(value) => parse_benchmark_downstream_wire_value(name, &value),
        Err(env::VarError::NotPresent) => Ok(0.0),
        Err(env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} must be valid UTF-8")
        }
    }
}

fn parse_benchmark_downstream_wire_value(name: &str, value: &str) -> Result<f64> {
    let parsed = value
        .parse::<f64>()
        .with_context(|| format!("{name} must be a finite non-negative number"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        anyhow::bail!("{name} must be a finite non-negative number");
    }
    Ok(parsed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SkippyModelState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub(crate) struct SkippyModelStatus {
    pub(crate) state: SkippyModelState,
    pub(crate) model_id: String,
    pub(crate) backend: &'static str,
    pub(crate) runtime_loaded: bool,
    pub(crate) package_ref: Option<String>,
    pub(crate) manifest_sha256: Option<String>,
    pub(crate) source_model_path: Option<String>,
    pub(crate) source_model_sha256: Option<String>,
    pub(crate) source_model_bytes: Option<u64>,
    pub(crate) materialized_path: Option<String>,
    pub(crate) materialized_pinned: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    /// Per-lane session state, possibly a frozen snapshot. Display only.
    pub(crate) lanes: Vec<SkippySessionLaneStatus>,
    pub(crate) max_session_tokens: u64,
    /// When [`Self::lanes`] and [`Self::max_session_tokens`] were read, which
    /// may be arbitrarily earlier than this status.
    pub(crate) sessions_captured_at_unix_nanos: i64,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) selected_device: Option<SkippyDeviceDescriptor>,
    pub(crate) openai_guardrails: Option<InferenceGuardrailsStatus>,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) stage_id: String,
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) started_at_unix_nanos: i64,
    pub(crate) stopped_at_unix_nanos: Option<i64>,
    pub(crate) last_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SkippySessionLaneStatus {
    pub(crate) index: usize,
    pub(crate) active: bool,
    pub(crate) session_id: Option<String>,
    pub(crate) token_count: Option<u64>,
}

pub(crate) fn default_skippy_openai_guardrails() -> InferenceGuardrailsConfig {
    InferenceGuardrailsConfig::for_standalone_mode(
        skippy_serving::frontend::InferenceGuardrailsMode::default(),
    )
}

pub(crate) fn skippy_openai_guardrails_for_mode(mode: GuardrailMode) -> InferenceGuardrailsConfig {
    // v1 only wraps hosted Skippy OpenAI backends constructed at the local/staged
    // seams below. MoA `model:"mesh"` arbitration and Virtual LLM consult paths
    // stay unwrapped until they adopt the backend-free guardrail core directly.
    let policy = GuardrailPolicy {
        mode,
        ..GuardrailPolicy::default()
    };
    skippy_openai_guardrails_for_policy_handle(GuardrailPolicyHandle::new(policy))
}

pub(crate) fn skippy_openai_guardrails_for_policy_handle(
    policy: GuardrailPolicyHandle,
) -> InferenceGuardrailsConfig {
    InferenceGuardrailsConfig::with_policy(policy)
}

#[derive(Debug)]
struct HandleState {
    state: SkippyModelState,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

pub(crate) struct SkippyModelHandle {
    runtime: SkippyRuntimeHandle,
    backend: Arc<dyn InferenceBackend>,
    openai_guardrails: Option<InferenceGuardrailsConfig>,
    config: StageConfig,
    started_at_unix_nanos: i64,
    status: Arc<Mutex<HandleState>>,
    _prediction_return_listener: Option<PredictionReturnListener>,
}

pub(crate) struct SkippyHttpHandle {
    port: u16,
    server: EmbeddedServerHandle,
}

pub(crate) struct SkippyOpenAiGuardrailOptions {
    config: Option<InferenceGuardrailsConfig>,
    telemetry: survey::SurveyTelemetry,
}

/// Host consumer for native model-open events. Runs on a host drain thread,
/// never on the native callback thread.
pub(crate) type NativeModelOpenEventReporter = Box<dyn FnMut(skippy_runtime::RuntimeEvent) + Send>;
pub(crate) use model_open_drain::{ModelOpenObservation, ModelOpenReturn, NativeModelOpenEvents};

impl SkippyOpenAiGuardrailOptions {
    pub(crate) fn new(
        config: Option<InferenceGuardrailsConfig>,
        telemetry: survey::SurveyTelemetry,
    ) -> Self {
        Self { config, telemetry }
    }
}

pub(crate) fn load_laya_model(
    path: &Path,
    device: Option<&str>,
) -> Result<Arc<skippy_runtime::LayaModel>> {
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    Ok(Arc::new(skippy_runtime::LayaModel::open(
        path, threads, device,
    )?))
}

pub(crate) fn start_laya_http_on(
    model_id: &str,
    model: Arc<skippy_runtime::LayaModel>,
    bind_addr: std::net::SocketAddr,
) -> SkippyHttpHandle {
    let lifecycle_observer = crate::network::openai::runtime_events::compose_lifecycle_observer(
        crate::logging_runtime_state().and_then(|state| state.openai_lifecycle_observer()),
    );
    let server = skippy_serving::start_openai_backend_with_lifecycle_observer(
        bind_addr,
        Arc::new(skippy_serving::LayaSystemOneBackend::new(model_id, model)),
        lifecycle_observer,
    );
    SkippyHttpHandle {
        port: bind_addr.port(),
        server,
    }
}

impl SkippyHttpHandle {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn status(&self) -> skippy_serving::EmbeddedServerStatus {
        self.server.status()
    }

    pub(crate) async fn shutdown(self) -> Result<()> {
        self.server.shutdown().await
    }
}

impl SkippyModelHandle {
    pub(crate) fn backend(&self) -> Arc<dyn InferenceBackend> {
        self.backend.clone()
    }

    pub(crate) fn openai_guardrails(&self) -> Option<InferenceGuardrailsStatus> {
        self.openai_guardrails
            .as_ref()
            .map(InferenceGuardrailsConfig::status)
    }

    pub(crate) fn set_openai_guardrail_mode(
        &self,
        mode: GuardrailMode,
    ) -> Option<InferenceGuardrailsStatus> {
        let guardrails = self.openai_guardrails.as_ref()?;
        guardrails.policy.set_mode(mode);
        Some(guardrails.status())
    }

    pub(crate) fn start_http(&self, port: u16) -> Result<SkippyHttpHandle> {
        self.start_http_on(([127, 0, 0, 1], port).into())
    }

    pub(crate) fn start_http_on(
        &self,
        bind_addr: std::net::SocketAddr,
    ) -> Result<SkippyHttpHandle> {
        let port = bind_addr.port();
        let tokenizer = self
            .runtime
            .tokenizer_capability()
            .context("loaded Skippy runtime cannot provide its stage-0 tokenizer capability")?;
        let lifecycle_observer = crate::network::openai::runtime_events::compose_lifecycle_observer(
            crate::logging_runtime_state().and_then(|state| state.openai_lifecycle_observer()),
        );
        let server = skippy_serving::start_openai_backend_with_tokenizer_and_lifecycle_observer(
            bind_addr,
            self.backend(),
            tokenizer,
            lifecycle_observer,
        );
        Ok(SkippyHttpHandle { port, server })
    }

    pub(crate) fn status(&self) -> SkippyModelStatus {
        let embedded = self.runtime.status();
        let local = self.status.lock().expect("skippy status lock poisoned");
        status_from_parts(
            &self.config,
            &embedded,
            &local,
            self.started_at_unix_nanos,
            self.openai_guardrails(),
        )
    }

    pub(crate) fn shutdown(&self) {
        {
            let mut state = self.status.lock().expect("skippy status lock poisoned");
            if matches!(state.state, SkippyModelState::Stopped) {
                return;
            }
            state.state = SkippyModelState::Stopping;
        }
        record_native_skippy_operational_event(
            NativeSkippyOperationalEvent::RuntimeShutdownStarted,
        );
        self.runtime.shutdown();
        let mut state = self.status.lock().expect("skippy status lock poisoned");
        state.state = SkippyModelState::Stopped;
        state.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for SkippyModelHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
fn wrap_host_guardrail_backend(
    backend: Arc<dyn InferenceBackend>,
    openai_guardrails: Option<&InferenceGuardrailsConfig>,
    context_limit_tokens: Option<usize>,
    telemetry: Option<Arc<dyn skippy_inference_api::GuardrailTelemetrySink>>,
) -> Arc<dyn InferenceBackend> {
    match openai_guardrails {
        Some(config) => {
            config.wrap_backend_with_telemetry(backend, context_limit_tokens, telemetry)
        }
        None => backend,
    }
}

#[async_trait]
impl InferenceBackend for SkippyModelHandle {
    async fn count_chat_tokens(&self, request: ChatCompletionRequest) -> InferenceResult<u32> {
        self.backend.count_chat_tokens(request).await
    }

    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        self.backend.models().await
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        self.backend.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        self.backend.chat_completion_stream(request, context).await
    }

    async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        self.backend.completion(request).await
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        self.backend.completion_stream(request, context).await
    }

    /// Forward embeddings and request context without chat processing.
    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        self.backend.embeddings(request, context).await
    }

    /// Forward reranking and request context without chat processing.
    async fn rerank(
        &self,
        request: RerankRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        self.backend.rerank(request, context).await
    }

    /// Forward speech generation and request context unchanged.
    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        self.backend.audio_speech(request, context).await
    }

    /// Forward multipart transcription and request context unchanged.
    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.backend.audio_transcription(request, context).await
    }

    /// Forward multipart translation and request context unchanged.
    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.backend.audio_translation(request, context).await
    }
}

pub(crate) fn infer_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model metadata {}", path.display()))?;
    let layer_count = info
        .tensors()
        .with_context(|| format!("read model tensors {}", path.display()))?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .with_context(|| format!("infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

fn status_from_parts(
    config: &StageConfig,
    embedded: &EmbeddedRuntimeStatus,
    local: &HandleState,
    started_at_unix_nanos: i64,
    openai_guardrails: Option<InferenceGuardrailsStatus>,
) -> SkippyModelStatus {
    SkippyModelStatus {
        state: match local.state {
            SkippyModelState::Starting => SkippyModelState::Starting,
            SkippyModelState::Ready => map_embedded_state(embedded.state),
            SkippyModelState::Stopping => SkippyModelState::Stopping,
            SkippyModelState::Stopped => SkippyModelState::Stopped,
            SkippyModelState::Failed => SkippyModelState::Failed,
        },
        model_id: config.model_id.clone(),
        backend: "skippy",
        runtime_loaded: embedded.runtime_loaded,
        package_ref: config.package_ref.clone(),
        manifest_sha256: config.manifest_sha256.clone(),
        source_model_path: config.source_model_path.clone(),
        source_model_sha256: config.source_model_sha256.clone(),
        source_model_bytes: config.source_model_bytes,
        materialized_path: config.materialized_path.clone(),
        materialized_pinned: config.materialized_pinned,
        projector_path: config.projector_path.clone(),
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        lanes: embedded
            .sessions
            .lanes
            .iter()
            .map(|lane| SkippySessionLaneStatus {
                index: lane.index,
                active: lane.active,
                session_id: lane.session_id.clone(),
                token_count: lane.token_count,
            })
            .collect(),
        max_session_tokens: embedded.sessions.max_session_tokens,
        sessions_captured_at_unix_nanos: embedded.sessions_captured_at_unix_nanos,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_gpu_layers: config.n_gpu_layers,
        flash_attn_type: config.flash_attn_type,
        selected_device: config.selected_device.clone().map(Into::into),
        openai_guardrails,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        stage_id: config.stage_id.clone(),
        topology_id: config.topology_id.clone(),
        run_id: config.run_id.clone(),
        started_at_unix_nanos,
        stopped_at_unix_nanos: local
            .stopped_at_unix_nanos
            .or(embedded.stopped_at_unix_nanos),
        last_error: local
            .last_error
            .clone()
            .or_else(|| embedded.last_error.clone()),
    }
}

fn map_embedded_state(state: EmbeddedState) -> SkippyModelState {
    match state {
        EmbeddedState::Starting => SkippyModelState::Starting,
        EmbeddedState::Ready => SkippyModelState::Ready,
        EmbeddedState::Stopping => SkippyModelState::Stopping,
        EmbeddedState::Stopped => SkippyModelState::Stopped,
        EmbeddedState::Failed => SkippyModelState::Failed,
    }
}

fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

/// Stage-0 compute meters of running split generations, keyed by run id, so
/// the split coordinator can read the local stage's busy time alongside the
/// peer stages' reported status.
static STAGE0_COMPUTE_METERS: std::sync::LazyLock<
    std::sync::Mutex<
        std::collections::HashMap<
            String,
            std::sync::Arc<skippy_serving::compute_meter::StageComputeMeter>,
        >,
    >,
> = std::sync::LazyLock::new(Default::default);

fn register_stage0_compute_meter(run_id: &str, runtime: &SkippyRuntimeHandle) {
    let Ok(state) = runtime.runtime().lock().map(|state| state.compute_meter()) else {
        return;
    };
    if let Ok(mut meters) = STAGE0_COMPUTE_METERS.lock() {
        meters.insert(run_id.to_string(), state);
    }
}

pub(crate) fn stage0_compute_meter(
    run_id: &str,
) -> Option<std::sync::Arc<skippy_serving::compute_meter::StageComputeMeter>> {
    STAGE0_COMPUTE_METERS.lock().ok()?.get(run_id).cloned()
}

pub(crate) fn forget_stage0_compute_meter(run_id: &str) {
    if let Ok(mut meters) = STAGE0_COMPUTE_METERS.lock() {
        meters.remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use skippy_inference_api::{InferenceError, MESH_COMPACT_FIELD};
    use skippy_serving::runtime_state::RuntimeSessionStats;
    use skippy_serving::telemetry::TelemetryStats;

    #[test]
    fn lifecycle_only_hooks_leave_exact_receipts_unset() {
        let ingress: Arc<dyn skippy_serving::frontend::GenerationLifecycleIngress> =
            Arc::new(runtime_events::SkippyGenerationRuntimeEventAdapter::new());
        let hooks = skippy_serving::serving_hooks::ModelServingHooks::default()
            .with_generation_lifecycle(
                skippy_serving::frontend::GenerationLifecycleConfig::from_ingress(ingress),
            );

        assert!(hooks.generation_lifecycle().is_some());
        assert!(hooks.generation_receipt().is_none());
    }

    #[test]
    fn benchmark_wire_delay_accepts_finite_non_negative_values() {
        assert_eq!(
            parse_benchmark_downstream_wire_value("test", "0").unwrap(),
            0.0
        );
        assert_eq!(
            parse_benchmark_downstream_wire_value("test", "25.5").unwrap(),
            25.5
        );
    }

    #[test]
    fn benchmark_wire_delay_rejects_invalid_values() {
        for value in ["-1", "NaN", "inf", "not-a-number"] {
            assert!(parse_benchmark_downstream_wire_value("test", value).is_err());
        }
    }

    #[derive(Default)]
    struct RecordingHostBackend {
        seen_chat: Mutex<Option<ChatCompletionRequest>>,
    }

    #[async_trait]
    impl InferenceBackend for RecordingHostBackend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("host-skippy")])
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> InferenceResult<ChatCompletionResponse> {
            *self.seen_chat.lock().expect("seen chat lock poisoned") = Some(request.clone());
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                skippy_inference_api::Usage::new(0, 0),
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: InferenceRequestContext,
        ) -> InferenceResult<ChatCompletionStream> {
            Err(InferenceError::unsupported(
                "streaming is not needed by this host wrapper test",
            ))
        }
    }

    fn fake_package_identity(layer_count: u32) -> SkippyPackageIdentity {
        SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_string(),
            source_model_bytes: 1234,
            source_files: Vec::new(),
            layer_weight_bytes: Vec::new(),
            layer_count,
            activation_width: 4096,
            tensor_count: 100,
            generation: None,
            publisher_defaults: None,
        }
    }

    fn fake_stage_config() -> StageConfig {
        single_stage_config(
            &SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36)),
        )
        .expect("fake stage config")
    }

    fn fake_embedded_runtime_status(config: &StageConfig) -> EmbeddedRuntimeStatus {
        EmbeddedRuntimeStatus {
            state: EmbeddedState::Ready,
            run_id: config.run_id.clone(),
            topology_id: config.topology_id.clone(),
            model_id: config.model_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            runtime_loaded: true,
            started_at_unix_nanos: 111,
            stopped_at_unix_nanos: None,
            last_error: None,
            sessions: RuntimeSessionStats {
                lane_count: 1,
                active_sessions: 0,
                idle_sessions: 1,
                idle_resident_prefixes: 0,
                tracked_token_counts: 0,
                max_session_tokens: 2048,
                total_session_tokens: 0,
                graphs_reused: 0,
                tokens_evaluated: 0,
                lanes: vec![],
            },
            sessions_captured_at_unix_nanos: 111,
            telemetry: TelemetryStats {
                queued: 0,
                sent: 0,
                dropped: 0,
                export_errors: 0,
            },
        }
    }

    #[test]
    fn single_stage_config_materializes_direct_gguf_runtime_slice() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36));

        let config = single_stage_config(&options).unwrap();

        assert_eq!(config.model_id, "Qwen3-8B-Q4_K_M");
        assert_eq!(config.model_path.as_deref(), Some("/models/qwen.gguf"));
        assert_eq!(
            config.package_ref.as_deref(),
            Some("gguf:///models/qwen.gguf")
        );
        assert_eq!(
            config.manifest_sha256.as_deref(),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        );
        assert_eq!(
            config.source_model_path.as_deref(),
            Some("/models/qwen.gguf")
        );
        assert_eq!(config.source_model_bytes, Some(1234));
        assert!(config.materialized_path.is_none());
        assert!(!config.materialized_pinned);
        assert_eq!(config.stage_id, "stage-0");
        assert_eq!(config.stage_index, 0);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 36);
        assert_eq!(config.ctx_size, 8192);
        assert_eq!(config.n_gpu_layers, -1);
        assert!(config.selected_device.is_none());
        assert_eq!(config.load_mode, LoadMode::RuntimeSlice);
        assert!(config.upstream.is_none());
        assert!(config.downstream.is_none());
    }

    #[test]
    fn single_stage_config_canonicalizes_checkpoint_quantization_aliases() {
        let mut options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36));
        options.checkpoint_quantization = Some("Q4_K".to_string());

        let config = single_stage_config(&options).unwrap();

        assert_eq!(config.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn single_stage_config_preserves_projector_path() {
        let options = SkippyModelLoadOptions::for_direct_gguf("Qwen2.5-VL", "/models/qwen-vl.gguf")
            .with_layer_end(36)
            .with_package_identity(fake_package_identity(36))
            .with_projector_path("/models/mmproj-qwen-vl.gguf");

        let config = single_stage_config(&options).unwrap();

        assert_eq!(
            config.projector_path.as_deref(),
            Some("/models/mmproj-qwen-vl.gguf")
        );
    }

    #[test]
    fn single_stage_config_preserves_selected_device_descriptor() {
        let options =
            SkippyModelLoadOptions::for_direct_gguf("Qwen3-8B-Q4_K_M", "/models/qwen.gguf")
                .with_ctx_size(8192)
                .with_generation_concurrency(3)
                .with_layer_end(36)
                .with_package_identity(fake_package_identity(36))
                .with_selected_device(SkippyDeviceDescriptor {
                    backend_device: "CUDA3".into(),
                    stable_id: Some("uuid:GPU-123".into()),
                    index: Some(3),
                    vram_bytes: Some(24_000_000_000),
                });

        let config = single_stage_config(&options).unwrap();
        let device = config.selected_device.expect("device descriptor");

        assert_eq!(device.backend_device, "CUDA3");
        assert_eq!(device.stable_id.as_deref(), Some("uuid:GPU-123"));
        assert_eq!(device.index, Some(3));
        assert_eq!(device.vram_bytes, Some(24_000_000_000));
    }

    #[test]
    fn single_stage_config_rejects_empty_selected_backend_device() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(1)
            .with_selected_device(SkippyDeviceDescriptor {
                backend_device: String::new(),
                stable_id: Some("uuid:GPU-123".into()),
                index: Some(0),
                vram_bytes: Some(24_000_000_000),
            });

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("selected backend device"));
    }

    #[test]
    fn single_stage_config_rejects_empty_layer_range() {
        let options = SkippyModelLoadOptions::for_direct_gguf("bad", "/models/bad.gguf")
            .with_layer_end(0)
            .with_package_identity(fake_package_identity(1));

        let err = single_stage_config(&options).unwrap_err().to_string();

        assert!(err.contains("layer_end"));
    }

    #[test]
    fn embedded_state_maps_to_mesh_skippy_state() {
        assert_eq!(
            map_embedded_state(EmbeddedState::Starting),
            SkippyModelState::Starting
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Ready),
            SkippyModelState::Ready
        );
        assert_eq!(
            map_embedded_state(EmbeddedState::Failed),
            SkippyModelState::Failed
        );
    }

    #[test]
    fn status_includes_guardrail_policy_without_private_content() {
        let config = fake_stage_config();
        let embedded = fake_embedded_runtime_status(&config);
        let local = HandleState {
            state: SkippyModelState::Ready,
            stopped_at_unix_nanos: None,
            last_error: None,
        };
        let status = status_from_parts(
            &config,
            &embedded,
            &local,
            222,
            Some(InferenceGuardrailsStatus {
                mode: "disabled",
                target: "skippy",
                streaming: "pass_through",
                retry_exhaustion: "error",
                small_model_policy: "small_models_only",
                small_param_threshold_b: 9.0,
                max_tool_retries: 1,
                max_structured_retries: 2,
            }),
        );

        let guardrails = serde_json::to_value(
            status
                .openai_guardrails
                .expect("skippy status should include guardrails policy"),
        )
        .expect("guardrails serialize");
        let guardrails = guardrails
            .as_object()
            .expect("guardrails should serialize as an object");

        assert_eq!(guardrails.len(), 8);
        assert_eq!(guardrails.get("mode"), Some(&serde_json::json!("disabled")));
        assert_eq!(guardrails.get("target"), Some(&serde_json::json!("skippy")));
        assert_eq!(
            guardrails.get("streaming"),
            Some(&serde_json::json!("pass_through"))
        );
        assert_eq!(
            guardrails.get("retry_exhaustion"),
            Some(&serde_json::json!("error"))
        );
        assert_eq!(
            guardrails.get("small_model_policy"),
            Some(&serde_json::json!("small_models_only"))
        );
        assert_eq!(
            guardrails.get("small_param_threshold_b"),
            Some(&serde_json::json!(9.0))
        );
        assert_eq!(
            guardrails.get("max_tool_retries"),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            guardrails.get("max_structured_retries"),
            Some(&serde_json::json!(2))
        );

        for forbidden in [
            "prompt",
            "schema",
            "tool_args",
            "tool_names",
            "reserved_tool_prefix",
            "sentinels",
            "raw_tool_names",
            "sentinel_definitions",
        ] {
            assert!(
                guardrails.get(forbidden).is_none(),
                "privacy-safe status should omit {forbidden}"
            );
        }
    }

    #[test]
    fn guardrail_config_status_tracks_shared_policy_handle() {
        let policy = GuardrailPolicyHandle::default();
        let config = skippy_openai_guardrails_for_policy_handle(policy.clone());

        assert_eq!(config.status().mode, "disabled");

        policy.set_mode(GuardrailMode::MetricsOnly);
        assert_eq!(config.status().mode, "metrics");

        policy.set_mode(GuardrailMode::Enforce);
        let status = config.status();
        assert_eq!(status.mode, "enforce");
        assert_eq!(status.streaming, "pass_through");
        assert_eq!(status.retry_exhaustion, "error");
        assert_eq!(status.max_tool_retries, 1);
        assert_eq!(status.max_structured_retries, 2);
    }

    #[tokio::test]
    async fn host_guardrail_wrapper_applies_compaction_when_guardrails_are_disabled() {
        let backend = Arc::new(RecordingHostBackend::default());
        let wrapped = wrap_host_guardrail_backend(
            backend.clone(),
            Some(&InferenceGuardrailsConfig {
                target: InferenceGuardrailsTarget::Skippy,
                policy: GuardrailPolicyHandle::default(),
                compaction: Some(CompactionConfig::default()),
            }),
            Some(8),
            None,
        );
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [
                {"role": "tool", "content": "large intermediate result", "tool_call_id": "call_1"},
                {"role": "user", "content": "continue"}
            ],
            (MESH_COMPACT_FIELD): true
        }))
        .expect("valid compacting request");

        wrapped
            .chat_completion(request)
            .await
            .expect("wrapped chat completion");

        let seen = backend
            .seen_chat
            .lock()
            .expect("seen chat lock poisoned")
            .clone()
            .expect("inner backend should see compacted request");
        assert_eq!(
            seen.messages.first().map(|message| message.role.as_str()),
            Some("system")
        );
        assert!(
            seen.messages.iter().all(|message| message.role != "tool"),
            "host-runtime wrapper should run compacting before the embedded backend sees the request"
        );
    }

    #[tokio::test]
    async fn host_guardrail_wrapper_uses_live_policy_mode() {
        let backend = Arc::new(RecordingHostBackend::default());
        let policy = GuardrailPolicyHandle::default();
        let wrapped = wrap_host_guardrail_backend(
            backend.clone(),
            Some(&InferenceGuardrailsConfig {
                target: InferenceGuardrailsTarget::Skippy,
                policy: policy.clone(),
                compaction: None,
            }),
            Some(8192),
            None,
        );
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen3-8B-Q4_K_M",
            "messages": [{"role": "user", "content": "look this up"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"
        }))
        .expect("valid tool request");

        wrapped.chat_completion(request.clone()).await.unwrap();
        assert_eq!(
            backend
                .seen_chat
                .lock()
                .expect("seen chat lock poisoned")
                .clone()
                .unwrap()
                .tools,
            request.tools
        );

        policy.update(GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        });
        let _ = wrapped.chat_completion(request).await;

        let seen = backend
            .seen_chat
            .lock()
            .expect("seen chat lock poisoned")
            .clone()
            .unwrap();
        let tool_names = seen
            .tools
            .as_ref()
            .and_then(|tools| tools.as_array())
            .unwrap()
            .iter()
            .filter_map(|tool| tool.get("function"))
            .filter_map(|function| function.get("name"))
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>();
        assert!(tool_names.contains(&skippy_inference_api::MESH_RESPOND_TOOL_NAME));
    }
}
