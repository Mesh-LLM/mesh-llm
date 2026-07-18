use super::*;
use crate::kv_integration::PrefillKvIdentity;
use async_trait::async_trait;
use std::io::Cursor;
use std::{
    env, fs,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{http::StatusCode, response::IntoResponse};
use openai_frontend::{AssistantMessage, ChatCompletionChoice};
use serde_json::json;
use skippy_protocol::{
    LoadMode, PeerConfig, SCHEMA_VERSION, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use tokio::sync::Semaphore;

const MM_MODEL_ENV: &str = "SKIPPY_MM_MODEL";
const MM_PROJECTOR_ENV: &str = "SKIPPY_MM_PROJECTOR";
const MM_IMAGE_ENV: &str = "SKIPPY_MM_IMAGE";
const MM_ACTIVATION_WIDTH_ENV: &str = "SKIPPY_MM_ACTIVATION_WIDTH";
const MM_SPLIT_LAYER_ENV: &str = "SKIPPY_MM_SPLIT_LAYER";
const MM_CTX_SIZE_ENV: &str = "SKIPPY_MM_CTX_SIZE";
const MM_MAX_TOKENS_ENV: &str = "SKIPPY_MM_MAX_TOKENS";
const MM_N_GPU_LAYERS_ENV: &str = "SKIPPY_MM_N_GPU_LAYERS";

#[test]
fn proactive_eviction_attrs_are_bounded_and_request_free() {
    let attrs = proactive_eviction_attrs("error", Some("inactive_session"), 1024, 2, 768);

    assert_eq!(
        attrs.get("skippy.kv.decision"),
        Some(&json!("proactive_eviction"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_STATUS),
        Some(&json!("error"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_ERROR_KIND),
        Some(&json!("inactive_session"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTION_TARGET_TOKENS),
        Some(&json!(1024))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTED_ENTRIES),
        Some(&json!(2))
    );
    assert_eq!(
        attrs.get(attr_key::KV_PROACTIVE_EVICTED_TOKENS),
        Some(&json!(768))
    );
    assert!(!attrs.contains_key(attr_key::REQUEST_ID));
    assert!(!attrs.contains_key(attr_key::SESSION_ID));
    assert!(!attrs.contains_key("openai.prompt_cache_key"));
    assert!(!attrs.contains_key("openai.prompt_cache_retention"));
}

fn prefix_cache_test_config() -> StageConfig {
    StageConfig {
        run_id: "run".to_string(),
        topology_id: "topology".to_string(),
        model_id: "org/model:Q4_K_M".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: None,
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 4,
        ctx_size: 8192,
        lane_count: 2,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 0,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        filter_tensors_on_load: false,
        selected_device: None,
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::ResidentKv,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 256,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        }),
        native_mtp_enabled: true,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "127.0.0.1:0".to_string(),
        }),
    }
}

fn prefix_cache_test_base() -> MessageBase {
    MessageBase {
        schema_version: SCHEMA_VERSION,
        run_id: "run".to_string(),
        request_id: "request".to_string(),
        session_id: "session".to_string(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        topology_id: "topology".to_string(),
        model_id: Some("org/model:Q4_K_M".to_string()),
        tokenizer_id: None,
        chat_template_id: Some("template".to_string()),
        seq: Some(1),
    }
}

fn prefix_cache_base_with_request(request_id: &str, session_id: &str) -> MessageBase {
    MessageBase {
        request_id: request_id.to_string(),
        session_id: session_id.to_string(),
        ..prefix_cache_test_base()
    }
}

fn seed_resident_prefix(kv: &KvStageIntegration, identity: &PrefillKvIdentity) {
    let token_count = identity.identity.token_count;
    let mut cache = kv.resident.lock().expect("resident cache lock poisoned");
    let allocation = cache
        .allocate_for_record(&identity.page_id, token_count, token_count, |_| Ok(()))
        .expect("synthetic resident prefix should allocate");
    assert!(allocation.should_retain);
    cache.commit_record(
        identity.page_id.clone(),
        allocation.seq_id,
        token_count,
        token_count,
    );
}

#[test]
fn openai_cache_stats_default_to_disabled() {
    let stats = GenerationCacheStats::default();

    assert_eq!(stats.status, "disabled");
    assert_eq!(stats.cached_prompt_tokens, 0);
    assert_eq!(stats.matched_prefix_tokens, 0);
    assert_eq!(stats.suffix_prefill_tokens, 0);
    assert_eq!(stats.hit_kind, None);
}

#[test]
fn cache_identity_reuses_repeated_prompts_without_client_cache_key() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let first_request = prefix_cache_base_with_request("request-a", "session-a");
    let second_request = prefix_cache_base_with_request("request-b", "session-b");
    let tokens = (0..1024).collect::<Vec<_>>();

    let recorded = kv.prefill_identity(&config, &first_request, 0, &tokens);
    let looked_up = kv.prefill_identity(&config, &second_request, 0, &tokens);

    assert_eq!(recorded.page_id, looked_up.page_id);
    assert_eq!(
        recorded.identity.prefix_hash,
        looked_up.identity.prefix_hash
    );
    assert_ne!(recorded.identity.session_id, looked_up.identity.session_id);

    seed_resident_prefix(&kv, &recorded);
    let hit = kv
        .probe_resident_prefix(&looked_up)
        .expect("repeated prompt should hit without a client cache key");
    assert_eq!(hit.page_id, recorded.page_id);
    assert_eq!(hit.token_count, tokens.len());
}

#[test]
fn cache_identity_namespaces_explicit_prompt_cache_keys() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let tokens = (0..1024).collect::<Vec<_>>();
    let mut default_namespace = prefix_cache_test_base();
    default_namespace.chat_template_id = None;
    let mut explicit_namespace = prefix_cache_test_base();
    explicit_namespace.chat_template_id = Some("openai:prompt_cache_key:tenant-a".to_string());

    let default_identity = kv.prefill_identity(&config, &default_namespace, 0, &tokens);
    let explicit_identity = kv.prefill_identity(&config, &explicit_namespace, 0, &tokens);

    assert_ne!(default_identity.page_id, explicit_identity.page_id);
    assert_ne!(
        default_identity.identity.prefix_hash,
        explicit_identity.identity.prefix_hash
    );
}

#[test]
fn disabled_cache_config_has_no_stage_integration() {
    let config = StageConfig {
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::Disabled,
            ..prefix_cache_test_config()
                .kv_cache
                .expect("test cache config")
        }),
        ..prefix_cache_test_config()
    };

    let kv = KvStageIntegration::from_config(&config).unwrap();

    assert!(kv.is_none());
}

#[test]
fn cold_resident_prefix_lookup_misses_before_recording() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let identity = kv.prefill_identity(
        &config,
        &prefix_cache_test_base(),
        0,
        &(0..1024).collect::<Vec<_>>(),
    );

    assert!(kv.probe_resident_prefix(&identity).is_none());
}

#[test]
fn resident_prefix_cache_hits_shared_prefix_grid() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);
    let record_plan = super::prefix_cache::stage0_full_prefill_record_identities(
        &kv,
        &config,
        &base,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);
    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");

    seed_resident_prefix(&kv, recorded_shared);
    let hit = kv
        .probe_resident_prefix(lookup_shared)
        .expect("different-tail prompt should hit shared prefix grid");

    assert_eq!(hit.page_id, recorded_shared.page_id);
    assert_eq!(hit.token_count, 2176);
}

#[test]
fn stage0_full_prefill_record_plan_includes_shared_prefix_candidate() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);

    let record_plan = super::prefix_cache::stage0_full_prefill_record_identities(
        &kv,
        &config,
        &base,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

    let record_counts = record_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();
    let lookup_counts = lookup_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();

    assert_eq!(record_counts, vec![2214, 2176]);
    assert!(lookup_counts.contains(&2176));

    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");
    let recorded_exact = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2214)
        .expect("record plan should keep exact first prompt");
    let lookup_exact = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2231)
        .expect("lookup plan should probe exact second prompt");

    assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
    assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
}

#[test]
fn stage0_chunked_prefill_record_plan_includes_shared_prefix_candidate() {
    let config = prefix_cache_test_config();
    let kv = KvStageIntegration::from_config(&config)
        .unwrap()
        .expect("resident prefix cache enabled");
    let base = prefix_cache_test_base();
    let recorded_tokens = (0..2214).collect::<Vec<_>>();
    let mut lookup_tokens = recorded_tokens.clone();
    lookup_tokens.extend(100_000..100_017);

    let record_plan = super::prefix_cache::stage0_prefill_record_identities(
        &kv,
        &config,
        &base,
        0,
        &recorded_tokens,
    );
    let lookup_plan = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

    let record_counts = record_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();
    let lookup_counts = lookup_plan
        .iter()
        .map(|identity| identity.identity.token_count)
        .collect::<Vec<_>>();

    assert_eq!(record_counts, vec![2214, 2176]);
    assert!(lookup_counts.contains(&2176));

    let recorded_shared = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("chunked record plan should include shared grid prefix");
    let lookup_shared = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2176)
        .expect("lookup plan should probe shared grid prefix");
    let recorded_exact = record_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2214)
        .expect("chunked record plan should keep exact first prompt");
    let lookup_exact = lookup_plan
        .iter()
        .find(|identity| identity.identity.token_count == 2231)
        .expect("lookup plan should probe exact second prompt");

    assert_eq!(recorded_shared.page_id, lookup_shared.page_id);
    assert_ne!(recorded_exact.page_id, lookup_exact.page_id);
}

struct MultimodalSmokeFixture {
    model_path: PathBuf,
    projector_path: PathBuf,
    image_path: PathBuf,
    layer_end: u32,
    activation_width: i32,
    ctx_size: u32,
    max_tokens: u32,
    n_gpu_layers: i32,
}

fn multimodal_smoke_fixture() -> Result<Option<MultimodalSmokeFixture>> {
    let model_path = match env::var_os(MM_MODEL_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!(
                "skipping real multimodal smoke: set {MM_MODEL_ENV}, {MM_PROJECTOR_ENV}, and {MM_IMAGE_ENV}"
            );
            return Ok(None);
        }
    };
    let projector_path = match env::var_os(MM_PROJECTOR_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_PROJECTOR_ENV}");
            return Ok(None);
        }
    };
    let image_path = match env::var_os(MM_IMAGE_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_IMAGE_ENV}");
            return Ok(None);
        }
    };
    if !model_path.is_file() {
        bail!(
            "{MM_MODEL_ENV} does not point at a file: {}",
            model_path.display()
        );
    }
    if !projector_path.is_file() {
        bail!(
            "{MM_PROJECTOR_ENV} does not point at a file: {}",
            projector_path.display()
        );
    }
    if !image_path.is_file() {
        bail!(
            "{MM_IMAGE_ENV} does not point at a file: {}",
            image_path.display()
        );
    }
    let layer_end = model_layer_count(&model_path)?;
    let activation_width = env_i32(MM_ACTIVATION_WIDTH_ENV)?
        .map(Ok)
        .unwrap_or_else(|| infer_activation_width(&model_path))?;
    let ctx_size = env_u32(MM_CTX_SIZE_ENV)?.unwrap_or(2048);
    let max_tokens = env_u32(MM_MAX_TOKENS_ENV)?.unwrap_or(16);
    let n_gpu_layers = env_i32(MM_N_GPU_LAYERS_ENV)?.unwrap_or(0);
    Ok(Some(MultimodalSmokeFixture {
        model_path,
        projector_path,
        image_path,
        layer_end,
        activation_width,
        ctx_size,
        max_tokens,
        n_gpu_layers,
    }))
}

fn env_i32(name: &str) -> Result<Option<i32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<i32>()
                .with_context(|| format!("parse {name}={value:?} as i32"))
        })
        .transpose()
}

fn env_u32(name: &str) -> Result<Option<u32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("parse {name}={value:?} as u32"))
        })
        .transpose()
}

fn infer_activation_width(path: &Path) -> Result<i32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let candidates = [
        "attn_norm.weight",
        "attention_norm.weight",
        "input_layernorm.weight",
        "ln_1.weight",
    ];
    let width = info
        .tensors()?
        .into_iter()
        .filter(|tensor| tensor.layer_index == Some(0))
        .find(|tensor| {
            candidates
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| tensor.element_count)
        .ok_or_else(|| {
            anyhow!(
                "could not infer activation width for {}; set {MM_ACTIVATION_WIDTH_ENV}",
                path.display()
            )
        })?;
    i32::try_from(width).context("activation width exceeds i32")
}

fn unsupported_code(error: OpenAiError) -> Option<String> {
    error.body().error.code
}

fn test_request_defaults() -> EmbeddedOpenAiRequestDefaults {
    EmbeddedOpenAiRequestDefaults {
        stop: Some(vec!["</stop>".to_string()]),
        temperature: Some(0.2),
        top_p: Some(0.9),
        presence_penalty: Some(1.25),
        frequency_penalty: Some(0.5),
        seed: Some(77),
        logit_bias: Some(std::collections::BTreeMap::from([
            ("123".to_string(), json!(-50.0)),
            ("456".to_string(), json!(12.5)),
        ])),
        top_k: Some(12),
        min_p: Some(0.1),
        repeat_penalty: Some(1.2),
        repeat_last_n: Some(64),
        reasoning_format: Some(EmbeddedReasoningFormat::Hidden),
        reasoning_enabled: Some(EmbeddedReasoningEnabled::Enabled),
        reasoning_budget: Some(EmbeddedReasoningBudget::Tokens(256)),
    }
}

fn assert_generation_rate_limit(error: OpenAiError, message_fragment: &str) {
    assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = error.body();
    assert_eq!(body.error.code.as_deref(), Some("rate_limit_exceeded"));
    assert!(
        body.error.message.contains(message_fragment),
        "expected {:?} to contain {:?}",
        body.error.message,
        message_fragment
    );

    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
}

#[tokio::test]
async fn generation_admission_uses_open_lane_without_queueing() {
    let generation_limit = Arc::new(Semaphore::new(1));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));

    let permit = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(10),
    )
    .await
    .unwrap();

    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
    drop(permit);
}

#[tokio::test]
async fn generation_admission_rejects_when_queue_full() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(1));

    let error = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(10),
    )
    .await
    .unwrap_err();

    assert_generation_rate_limit(error, "queue is full");
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn generation_admission_times_out_and_releases_queue_slot() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));

    let error = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(5),
    )
    .await
    .unwrap_err();

    assert_generation_rate_limit(error, "timed out waiting");
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn generation_admission_waits_for_released_lane() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));
    let release_limit = generation_limit.clone();
    let release_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        release_limit.add_permits(1);
    });

    let permit = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_secs(1),
    )
    .await
    .unwrap();

    release_task.await.unwrap();
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
    drop(permit);
}

#[test]
fn chat_runtime_feature_guard_allows_noop_parity_fields() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [],
        "tool_choice": null,
        "parallel_tool_calls": false,
        "response_format": {"type": "text"}
    }))
    .unwrap();

    ensure_chat_runtime_features_supported(&request).unwrap();
}

#[test]
fn chat_runtime_feature_guard_allows_structured_output_for_guardrails() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}}
        }
    }))
    .unwrap();

    ensure_chat_runtime_features_supported(&request).unwrap();
}

#[derive(Default)]
struct StructuredGuardrailRecordingBackend {
    seen: Mutex<Option<ChatCompletionRequest>>,
}

#[async_trait]
impl OpenAiBackend for StructuredGuardrailRecordingBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("test")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        ensure_chat_runtime_features_supported(&request)
            .expect("guarded wrapper should downgrade backend-facing structured requests");
        *self.seen.lock().unwrap() = Some(request);
        Ok(ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 123,
            model: "test".to_string(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(json!([{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "_mesh_emit_structured",
                            "arguments": "{\"answer\":\"ok\"}"
                        }
                    }])),
                },
                logprobs: None,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::new(1, 1),
            timings: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        unreachable!("streaming is not used in this test")
    }

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        unreachable!("completions are not used in this test")
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        unreachable!("completions are not used in this test")
    }
}

#[tokio::test]
async fn guarded_structured_output_is_not_rejected_by_runtime_feature_guard() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        }
        .into(),
        compaction: None,
    };
    let guarded = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .unwrap();

    let response = guarded.chat_completion(request).await.unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":\"ok\"}")
    );

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert!(
        seen.response_format.is_none(),
        "guarded backend should clear backend-facing response_format"
    );
}

#[test]
fn standalone_guardrail_modes_have_expected_policies() {
    let metrics =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Metrics)
            .status();
    assert_eq!(metrics.mode, "metrics");
    assert_eq!(metrics.retry_exhaustion, "pass_last_text");
    assert_eq!(metrics.small_model_policy, "all");

    let enforce =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Enforce)
            .status();
    assert_eq!(enforce.mode, "enforce");
    assert_eq!(enforce.retry_exhaustion, "error");
    assert_eq!(enforce.small_model_policy, "all");

    let disabled =
        OpenAiGuardrailsConfig::for_standalone_mode(crate::cli::OpenAiGuardrailsCliMode::Disabled)
            .status();
    assert_eq!(disabled.mode, "disabled");
    assert_eq!(disabled.small_model_policy, "small_models_only");
}

#[tokio::test]
async fn compaction_wraps_skippy_backend_even_when_guardrails_are_disabled() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy::default().into(),
        compaction: Some(CompactionConfig::default()),
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ],
        "mesh_compact": true
    }))
    .unwrap();

    let _ = wrapped.chat_completion(request).await;

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn disabled_skippy_guardrail_wrapper_can_be_enabled_live() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let policy: openai_frontend::GuardrailPolicyHandle = GuardrailPolicy::default().into();
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: policy.clone(),
        compaction: None,
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request = tool_request();

    wrapped.chat_completion(request.clone()).await.unwrap();
    assert_eq!(
        backend.seen.lock().unwrap().clone().unwrap().tools,
        request.tools
    );

    policy.update(GuardrailPolicy {
        mode: GuardrailMode::Enforce,
        apply_to_all_models: true,
        ..GuardrailPolicy::default()
    });
    let _ = wrapped.chat_completion(request).await;

    let seen = backend.seen.lock().unwrap().clone().unwrap();
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
    assert!(tool_names.contains(&"_mesh_respond"));
    assert_eq!(guardrails.status().mode, "enforce");
}

#[tokio::test]
async fn compaction_wraps_skippy_backend_with_runtime_context_limit() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy::default().into(),
        compaction: Some(CompactionConfig {
            enabled: true,
            ..CompactionConfig::default()
        }),
    };
    let wrapped = guardrails.wrap_backend_with_context_limit(backend.clone(), Some(1));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ]
    }))
    .unwrap();

    wrapped.chat_completion(request).await.unwrap();

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[tokio::test]
async fn compaction_and_guardrails_can_stack() {
    let backend = Arc::new(StructuredGuardrailRecordingBackend::default());
    let guardrails = OpenAiGuardrailsConfig {
        target: OpenAiGuardrailsTarget::Skippy,
        policy: GuardrailPolicy {
            mode: GuardrailMode::Enforce,
            apply_to_all_models: true,
            ..GuardrailPolicy::default()
        }
        .into(),
        compaction: Some(CompactionConfig::default()),
    };
    let wrapped = guardrails.wrap_backend(backend.clone());
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "tool", "content": "stale tool result"},
            {"role": "user", "content": "continue"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        },
        "mesh_compact": true
    }))
    .unwrap();

    let response = wrapped.chat_completion(request).await.unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("{\"answer\":\"ok\"}")
    );

    let seen = backend.seen.lock().unwrap().clone().unwrap();
    assert!(seen.response_format.is_none());
    assert_eq!(seen.messages[0].role, "system");
    assert!(seen.messages.iter().all(|message| message.role != "tool"));
}

#[test]
fn chat_runtime_feature_guard_allows_tool_calls() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}]
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto"
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "parallel_tool_calls": true
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        ensure_chat_runtime_features_supported(&request).unwrap();
    }
}

fn tool_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }]
    }))
    .unwrap()
}

#[test]
fn plain_chat_does_not_require_chat_output_parser() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();

    assert!(!chat_output_parser_required(
        &request,
        &ChatTemplateOptions::default(),
    ));
}

#[test]
fn hidden_reasoning_format_requires_chat_output_parser_for_plain_chat() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Hidden),
        ..ChatTemplateOptions::default()
    };

    assert!(chat_output_parser_required(&request, &template_options));
}

#[test]
fn reasoning_format_none_leaves_plain_chat_unparsed() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::None),
        ..ChatTemplateOptions::default()
    };

    assert!(!chat_output_parser_required(&request, &template_options));
}

#[test]
fn tools_require_chat_output_parser() {
    assert!(chat_output_parser_required(
        &tool_request(),
        &ChatTemplateOptions::default(),
    ));
}

#[test]
fn parses_llama_message_tool_calls() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");

    assert_eq!(parsed.content, None);
    assert_eq!(parsed.tool_calls[0]["id"], "call_123");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn parses_llama_message_reasoning_content() {
    let parsed = parsed_chat_message_from_json(
        r#"{"role":"assistant","reasoning_content":"Checked facts first.","content":"Final answer."}"#,
        &ChatCompletionRequest {
            tools: None,
            ..tool_request()
        },
    )
    .expect("parsed message");

    assert_eq!(parsed.content.as_deref(), Some("Final answer."));
    assert_eq!(
        parsed.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
    assert_eq!(parsed.tool_calls, None);
}

#[test]
fn chat_response_from_parsed_message_separates_reasoning_content() {
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 7,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats {
            drafted_tokens: 7,
            accepted_tokens: 5,
            rejected_tokens: 2,
            verification_count: 7,
            proposal_compute_us: 100,
            verification_compute_us: 200,
            ..NativeMtpStats::default()
        },
        native_mtp_decode_telemetry: None,
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "Checked facts first.</think>Final answer.".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };

    let response = chat_response_from_generated_text("qwen".to_string(), &output, Some(parsed));

    let message = &response.choices[0].message;
    assert_eq!(message.content.as_deref(), Some("Final answer."));
    let timings = response.timings.as_ref().expect("native MTP timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(7)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(5)));
    assert_eq!(
        message.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
    assert_eq!(message.tool_calls, None);
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));

    let completion = completion_response_from_generated_text("qwen".to_string(), &output);
    let timings = completion
        .timings
        .as_ref()
        .expect("completion native MTP timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(7)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(5)));
    assert_eq!(timings.get("predicted_per_second"), Some(&json!(70.0)));
}

#[test]
fn generated_text_timings_are_present_without_native_mtp() {
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 8,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats::default(),
        native_mtp_decode_telemetry: None,
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "Paris".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };

    let timings = output.timings().expect("standard timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(0)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(0)));
    assert_eq!(timings.get("prompt_per_second"), Some(&json!(200.0)));
    assert_eq!(timings.get("predicted_per_second"), Some(&json!(80.0)));
}

#[test]
fn generated_text_timings_prefer_composite_proposal_totals() {
    let mut counters = NativeMtpDecodeCounters::default();
    let proposal = CompositeProposalProvider::from_options(NativeMtpDecodeOptions {
        max_draft_tokens: 1,
        min_draft_tokens: 0,
        reject_cooldown_tokens: 0,
        suppress_cooldown_drafts: false,
        suppress_cooldown_draft_limit: 0,
        ngram_hybrid: true,
        ngram_size: 2,
        ngram_initial_extension_tokens: 2,
        ngram_max_proposal_tokens: 4,
        ngram_tail_backoff_proposals: 2,
        verify_window_min_tokens: 1,
        verify_window_max_tokens: 4,
    })
    .propose(&[], &[0, 0, 2, 3, 9, 1, 7, 8, 2, 3], 4);
    counters.observe_hybrid_proposal(&proposal, 4);
    let output = GeneratedText {
        prompt_tokens: 4,
        completion_tokens: 8,
        cache_status: "disabled",
        cached_prompt_tokens: 0,
        matched_prefix_tokens: 0,
        suffix_prefill_tokens: 0,
        cache_hit_kind: None,
        native_mtp_stats: NativeMtpStats::default(),
        native_mtp_decode_telemetry: Some(NativeMtpDecodeTelemetry::new(
            NativeMtpDecodeOptions {
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
                ngram_hybrid: true,
                ngram_size: 2,
                ngram_initial_extension_tokens: 2,
                ngram_max_proposal_tokens: 4,
                ngram_tail_backoff_proposals: 2,
                verify_window_min_tokens: 1,
                verify_window_max_tokens: 4,
            },
            counters,
        )),
        verify_window_pipeline_stats: None,
        speculative_stats: None,
        prompt_ms: 20.0,
        predicted_ms: 100.0,
        text: "ok".to_string(),
        finish_reason: FinishReason::Stop,
        detokenize_ms: 0.0,
        text_emit_ms: 0.0,
        eog_check_ms: 0.0,
    };

    let timings = output.timings().expect("timings");
    assert_eq!(timings.get("draft_n"), Some(&json!(4)));
    assert_eq!(timings.get("draft_n_accepted"), Some(&json!(4)));
}

#[test]
fn hidden_reasoning_visibility_removes_reasoning_content() {
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Hidden),
        ..ChatTemplateOptions::default()
    };

    let visible = apply_reasoning_visibility(Some(parsed), &template_options)
        .expect("parsed message should remain");

    assert_eq!(visible.content.as_deref(), Some("Final answer."));
    assert_eq!(visible.reasoning_content, None);
}

#[test]
fn auto_reasoning_visibility_keeps_reasoning_content() {
    let parsed = ParsedChatMessage {
        content: Some("Final answer.".to_string()),
        reasoning_content: Some("Checked facts first.".to_string()),
        tool_calls: None,
    };
    let template_options = ChatTemplateOptions {
        reasoning_format: Some(ChatReasoningFormat::Auto),
        ..ChatTemplateOptions::default()
    };

    let visible = apply_reasoning_visibility(Some(parsed), &template_options)
        .expect("parsed message should remain");

    assert_eq!(
        visible.reasoning_content.as_deref(),
        Some("Checked facts first.")
    );
}

#[test]
fn generation_event_to_chat_chunk_emits_reasoning_delta() {
    let chunk = generation_event_to_chat_chunk(
        Ok(GenerationStreamEvent::ReasoningDelta(
            "Checking the premise.".to_string(),
        )),
        "qwen",
    )
    .unwrap();

    let delta = &chunk.choices[0].delta;
    assert_eq!(delta.content, None);
    assert_eq!(
        delta.reasoning_content.as_deref(),
        Some("Checking the premise.")
    );
    assert_eq!(delta.tool_calls, None);
}

#[test]
fn llama_message_tool_parser_rejects_unknown_tool() {
    let request = tool_request();
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}}]}"#,
        &request,
    )
    .expect("tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "lookup");

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"shell","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());
}

#[test]
fn tool_choice_limits_allowed_tool_name() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "search"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}
    }))
    .unwrap();

    assert!(parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request
    )
    .is_none());

    request.tool_choice = Some(json!("search"));
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"search","arguments":"{}"}}]}"#,
        &request,
    )
    .expect("selected tool call");
    assert_eq!(parsed.tool_calls[0]["function"]["name"], "search");
}

#[test]
fn parallel_tool_calls_false_keeps_first_call() {
    let mut request = tool_request();
    request.parallel_tool_calls = Some(false);
    let parsed = parsed_tool_calls_from_message_json(
        r#"{"role":"assistant","content":null,"tool_calls":[
            {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Sydney\"}"}},
            {"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Melbourne\"}"}}
        ]}"#,
        &request,
    )
    .expect("tool calls");

    assert_eq!(parsed.tool_calls.as_array().unwrap().len(), 1);
    assert_eq!(
        parsed.tool_calls[0]["function"]["arguments"],
        "{\"city\":\"Sydney\"}"
    );
}

#[test]
fn emulated_output_final_parses_tool_call() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output(
        "Let me check.\nTOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Sydney\"}}",
        &request,
        false,
    )
    .expect("emulated parse");

    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
    let calls = parsed.tool_calls.expect("tool calls");
    assert_eq!(calls[0]["function"]["name"], "lookup");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Sydney");
}

#[test]
fn emulated_output_partial_withholds_tool_calls_and_incomplete_line() {
    let request = tool_request();
    // Streaming: the TOOL_CALL line is not yet newline-terminated.
    let parsed = parse_emulated_chat_output(
        "Let me check.\nTOOL_CALL {\"name\": \"lookup\", \"argumen",
        &request,
        true,
    )
    .expect("emulated parse");

    // Tool calls are withheld while streaming.
    assert!(parsed.tool_calls.is_none());
    // Only the completed prose line is exposed; the partial marker line is held.
    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
}

#[test]
fn emulated_output_partial_streams_single_line_prose() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output("The capital is Canber", &request, true)
        .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The capital is Canber"));
}

#[test]
fn emulated_output_partial_withholds_possible_marker_suffix() {
    let request = tool_request();
    let parsed =
        parse_emulated_chat_output("Let me check. TOOL_", &request, true).expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("Let me check."));
}

#[test]
fn emulated_output_partial_ignores_marker_inside_completed_thinking() {
    let request = tool_request();
    let parsed = parse_emulated_chat_output(
        "<think>TOOL_CALL {\"name\":\"lookup\"}</think>The answer is Syd",
        &request,
        true,
    )
    .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The answer is Syd"));
}

#[test]
fn emulated_output_respects_allowed_tool_names() {
    let request = tool_request();
    // "search" is not an allowed tool for this request (only "lookup" is).
    let parsed = parse_emulated_chat_output(
        "TOOL_CALL {\"name\": \"search\", \"arguments\": {}}",
        &request,
        false,
    )
    .expect("emulated parse");

    assert!(parsed.tool_calls.is_none());
}

#[test]
fn emulated_output_parallel_false_keeps_first_call() {
    let mut request = tool_request();
    request.parallel_tool_calls = Some(false);
    let parsed = parse_emulated_chat_output(
        "TOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Sydney\"}}\n\
         TOOL_CALL {\"name\": \"lookup\", \"arguments\": {\"city\": \"Melbourne\"}}",
        &request,
        false,
    )
    .expect("emulated parse");

    let calls = parsed.tool_calls.expect("tool calls");
    assert_eq!(calls.as_array().unwrap().len(), 1);
}

#[test]
fn emulated_output_plain_prose_has_no_tool_calls() {
    let request = tool_request();
    let parsed =
        parse_emulated_chat_output("The capital is Canberra.", &request, false).expect("parse");
    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.content.as_deref(), Some("The capital is Canberra."));
}

#[test]
fn tool_call_stream_delta_adds_indexes() {
    let delta = tool_calls_stream_delta(json!([
        {"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{}"}},
        {"id":"call_b","type":"function","function":{"name":"lookup","arguments":"{}"}}
    ]));

    assert_eq!(delta[0]["index"], 0);
    assert_eq!(delta[1]["index"], 1);
}

#[test]
fn chat_message_generation_value_preserves_tool_history() {
    let message: openai_frontend::ChatMessage = serde_json::from_value(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"city\":\"Sydney\"}"}
        }]
    }))
    .unwrap();
    let mut media = Vec::new();

    let value = chat_message_generation_value(&message, "<__media__>", &mut media).unwrap();

    assert_eq!(value["content"], Value::Null);
    assert_eq!(value["tool_calls"][0]["id"], "call_123");
    assert_eq!(value["tool_calls"][0]["function"]["name"], "lookup");
}

#[test]
fn chat_runtime_feature_guard_rejects_logprobs() {
    for payload in [
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        }),
        json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": false,
            "top_logprobs": 1
        }),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(payload).unwrap();
        let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
        assert_eq!(
            unsupported_code(error),
            Some("unsupported_model_feature".to_string())
        );
    }
}

#[test]
fn completion_runtime_feature_guard_rejects_logprobs() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "prompt": "hi",
        "logprobs": 2
    }))
    .unwrap();

    let error = ensure_completion_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

fn multimodal_stage_config(
    fixture: &MultimodalSmokeFixture,
    stage_id: &str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
) -> StageConfig {
    StageConfig {
        run_id: "mm-smoke-run".to_string(),
        topology_id: "mm-smoke-topology".to_string(),
        model_id: "mm-smoke".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(fixture.model_path.to_string_lossy().to_string()),
        projector_path: (stage_index == 0)
            .then(|| fixture.projector_path.to_string_lossy().to_string()),
        stage_id: stage_id.to_string(),
        stage_index,
        layer_start,
        layer_end,
        ctx_size: fixture.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: fixture.n_gpu_layers,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        filter_tensors_on_load: layer_start != 0 || layer_end != fixture.layer_end,
        selected_device: None,
        kv_cache: None,
        native_mtp_enabled: true,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        bind_addr: bind_addr.to_string(),
        upstream: None,
        downstream: None,
    }
}

fn local_openai_backend(config: StageConfig) -> Result<StageOpenAiBackend> {
    let runtime = load_runtime(&config)?.context("load smoke runtime")?;
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    let decode_batcher = DecodeBatcher::new(runtime.clone(), 1);
    let decode_frame_batcher = DecodeFrameBatcher::new(runtime.clone(), 1);
    Ok(StageOpenAiBackend {
        runtime,
        telemetry: Telemetry::new(
            None,
            1,
            config.clone(),
            crate::telemetry::TelemetryLevel::Off,
        ),
        config,
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_min: 0,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig::default(),
        generation_limit: Arc::new(Semaphore::new(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        kv: None,
        decode_batcher,
        decode_frame_batcher,
    })
}

fn multimodal_chat_request(fixture: &MultimodalSmokeFixture) -> Result<ChatCompletionRequest> {
    let image = fs::read(&fixture.image_path)
        .with_context(|| format!("read smoke image {}", fixture.image_path.display()))?;
    let mime_type = match fixture
        .image_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    serde_json::from_value(json!({
            "model": "mm-smoke",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": format!("data:{mime_type};base64,{encoded}")}}
                ]
            }],
            "max_tokens": fixture.max_tokens,
            "temperature": 0.0
        }))
        .context("build multimodal smoke request")
}

fn assert_nonempty_chat_response(response: ChatCompletionResponse) {
    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or_default()
        .trim();
    assert!(
        !content.is_empty(),
        "expected non-empty multimodal response"
    );
}

fn available_loopback_addr() -> Result<SocketAddr> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn split_layer_for_fixture(fixture: &MultimodalSmokeFixture) -> Result<Option<u32>> {
    if fixture.layer_end < 2 {
        eprintln!("skipping split multimodal smoke: model has fewer than two layers");
        return Ok(None);
    }
    let split = env_u32(MM_SPLIT_LAYER_ENV)?.unwrap_or(fixture.layer_end / 2);
    if split == 0 || split >= fixture.layer_end {
        bail!(
            "{MM_SPLIT_LAYER_ENV} must be in 1..{} for this model",
            fixture.layer_end
        );
    }
    Ok(Some(split))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_local_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let config = multimodal_stage_config(
        &fixture,
        "stage-0",
        0,
        0,
        fixture.layer_end,
        available_loopback_addr()?,
    );
    let backend = local_openai_backend(config)?;
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await?;

    assert_nonempty_chat_response(response);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_split_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let Some(split_layer) = split_layer_for_fixture(&fixture)? else {
        return Ok(());
    };
    let stage1_addr = available_loopback_addr()?;
    let stage0_addr = available_loopback_addr()?;
    let mut stage1_config = multimodal_stage_config(
        &fixture,
        "stage-1",
        1,
        split_layer,
        fixture.layer_end,
        stage1_addr,
    );
    stage1_config.upstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        endpoint: stage0_addr.to_string(),
    });
    let mut stage0_config =
        multimodal_stage_config(&fixture, "stage-0", 0, 0, split_layer, stage0_addr);
    stage0_config.downstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        endpoint: stage1_addr.to_string(),
    });

    let stage1_handle =
        crate::embedded::start_binary_stage(crate::binary_transport::BinaryStageOptions {
            config: stage1_config,
            topology: None,
            bind_addr: stage1_addr,
            activation_width: fixture.activation_width,
            wire_dtype: WireActivationDType::F16,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 1,
            telemetry_level: crate::telemetry::TelemetryLevel::Off,
            max_inflight: 4,
            reply_credit_limit: None,
            async_prefill_forward: false,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            downstream_connect_timeout_secs: 5,
            native_mtp_enabled: true,
            openai: None,
        });
    let ready = connect_endpoint_ready(&stage1_addr.to_string(), 120);
    if let Err(error) = ready {
        let status = stage1_handle.status();
        stage1_handle.abort();
        return Err(error.context(format!(
            "wait for stage-1 binary server; status={:?} last_error={:?}",
            status.state, status.last_error
        )));
    }

    let telemetry = Telemetry::new(
        None,
        1,
        stage0_config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let lane_pool = PersistentStageLanePool::new(&stage0_config, 1, 5, telemetry.clone())?
        .context("create split smoke lane pool")?;
    let runtime = load_runtime(&stage0_config)?.context("load stage-0 smoke runtime")?;
    let ctx_size = usize::try_from(stage0_config.ctx_size).unwrap_or(usize::MAX);
    let decode_batcher = DecodeBatcher::new(runtime.clone(), 1);
    let decode_frame_batcher = DecodeFrameBatcher::new(runtime.clone(), 1);
    let backend = StageOpenAiBackend {
        runtime,
        telemetry,
        config: stage0_config.clone(),
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode: OpenAiBackendMode::EmbeddedStageZero {
            config: stage0_config,
            wire_dtype: WireActivationDType::F16,
            prefill_chunk_policy: PrefillChunkPolicy::Fixed { chunk_size: 64 },
            activation_width: fixture.activation_width,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            prefill_reply_credit_limit: 0,
            lane_pool: Some(lane_pool),
            prediction_returns: None,
        },
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_min: 0,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig {
            native_mtp: NativeMtpProposalConfig {
                enabled: true,
                max_draft_tokens: 3,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            effective_strategy: "native-mtp".to_string(),
            ..SpeculativeDecodeConfig::default()
        },
        generation_limit: Arc::new(Semaphore::new(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        kv: None,
        decode_batcher,
        decode_frame_batcher,
    };
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await;
    stage1_handle.shutdown().await?;

    assert_nonempty_chat_response(response?);
    Ok(())
}

#[test]
fn trims_at_first_stop_sequence() {
    assert_eq!(trim_at_stop("hello END world", &["END"]), "hello ");
    assert_eq!(trim_at_stop("abc xyz def", &["def", "xyz"]), "abc ");
    assert_eq!(trim_at_stop("abc", &[""]), "abc");
}

#[test]
fn generation_stop_values_include_chat_template_stops() {
    let request_stop = openai_frontend::StopSequence::One("</stop>".to_string());
    let metadata = json!({
        "additional_stops": ["<|user|>", "<|observation|>", ""],
    })
    .to_string();

    let stops = generation_stop_values(Some(&request_stop), Some(&metadata));

    assert_eq!(stops, vec!["</stop>", "<|user|>", "<|observation|>"]);
}

#[test]
fn valid_utf8_prefix_skips_incomplete_suffix() {
    assert_eq!(valid_utf8_prefix_len("hello".as_bytes()), 5);
    assert_eq!(valid_utf8_prefix_len(&[b'h', b'i', 0xE2, 0x82]), 2);
    assert_eq!(valid_utf8_prefix_len(&[0xF0, 0x9F, 0x98]), 0);
}

#[test]
fn message_content_to_generation_text_inserts_media_markers() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let text = message_content_to_generation_text(&content, "<__media__>", &mut media)
        .expect("media text");

    assert_eq!(text, "what is this?\n<__media__>");
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].bytes, b"hello");
}

#[test]
fn message_content_to_generation_text_rejects_remote_media_urls() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "input_image", "image_url": "https://example.com/image.png"}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let error =
        message_content_to_generation_text(&content, "<__media__>", &mut media).unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn rescued_audio_media_becomes_text_only_before_prompt_media_extraction() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "please transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": "YWJj",
                    "format": "wav"
                }}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();
    let media = openai_frontend::first_chat_media(&request.messages).expect("media");

    apply_chat_hook_outcome(
        &mut request,
        &ChatHookOutcome::injected_with_consumed_media("[Audio context: hello]\n\n", media),
    );

    let content = request.messages[0].content.as_ref().expect("content");
    let mut media = Vec::new();
    let text = message_content_to_generation_text(content, "<__media__>", &mut media)
        .expect("generation text");

    assert!(media.is_empty());
    assert!(!text.contains("<__media__>"));
    assert!(text.contains("[Audio context: hello]"));
    assert!(text.contains("please transcribe this"));
}

#[test]
fn rescued_media_leaves_unhandled_second_media_in_prompt_media() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these"},
                {"type": "input_audio", "input_audio": {
                    "data": "YXVkaW8=",
                    "format": "wav"
                }},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,aW1hZ2U="
                }}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();
    let media = openai_frontend::first_chat_media(&request.messages).expect("media");

    apply_chat_hook_outcome(
        &mut request,
        &ChatHookOutcome::injected_with_consumed_media("[Audio context: hello]\n\n", media),
    );

    let content = request.messages[0].content.as_ref().expect("content");
    let mut media = Vec::new();
    let text = message_content_to_generation_text(content, "<__media__>", &mut media)
        .expect("generation text");

    assert_eq!(media.len(), 1);
    assert_eq!(media[0].bytes, b"image");
    assert_eq!(
        text,
        "[Audio context: hello]\n\n\ncompare these\n<__media__>"
    );
}

#[test]
fn multimodal_final_prefill_message_requests_downstream_prediction() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = multimodal_prefill_message(
        WireActivationDType::F16,
        MultimodalPrefillArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 17,
            pos_start: 0,
            token_count: 17,
            positions: Vec::new(),
            sampling: Some(sampling.clone()),
            final_chunk: true,
        },
    )
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::PrefillFinalEmbd);
    assert!(message.kind.requires_predicted_reply());
    assert_eq!(message.token_count, 17);
    assert_eq!(message.state.current_token, LLAMA_TOKEN_NULL);
    assert_eq!(message.sampling, Some(sampling));
}

#[test]
fn restore_prefill_decode_message_carries_chat_sampling_metadata() {
    let metadata = r#"{"grammar":"chat","prompt_tokens":4}"#;
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };

    let message = embedded_restore_prefill_decode_message(
        WireActivationDType::F16,
        RestorePrefillDecodeMessageArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 4,
            pos_start: 3,
            decode_step: 0,
            prefix_tokens: &[101, 102, 103],
            current: 104,
            sampling: Some(sampling.clone()),
            chat_sampling_metadata: Some(metadata),
        },
    )
    .unwrap();

    assert_eq!(message.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(message.tokens, vec![101, 102, 103, 104]);
    assert_eq!(message.sampling, Some(sampling.clone()));
    assert_eq!(message.chat_sampling_metadata.as_deref(), Some(metadata));

    let mut encoded = Vec::new();
    write_stage_message(&mut encoded, &message, WireActivationDType::F16).unwrap();
    let decoded = skippy_protocol::binary::read_stage_message(Cursor::new(encoded), 2816).unwrap();
    assert_eq!(decoded.kind, WireMessageKind::TryRestorePrefillDecode);
    assert_eq!(decoded.tokens, vec![101, 102, 103, 104]);
    assert_eq!(decoded.sampling, Some(sampling));
    assert_eq!(decoded.chat_sampling_metadata.as_deref(), Some(metadata));
}

#[test]
fn reusable_decode_message_updates_hot_path_fields() {
    let sampling = WireSamplingConfig {
        flags: 1,
        seed: 7,
        ..WireSamplingConfig::default()
    };
    let mut message = ReusableDecodeMessage::new(
        WireActivationDType::F16,
        ReusableDecodeMessageArgs {
            request_id: 11,
            session_id: 13,
            prompt_token_count: 4,
            base_pos_start: 4,
            sampling: Some(sampling.clone()),
            sideband_capacity: 4,
        },
    )
    .unwrap();

    let first = message.update(0, 104).unwrap();
    assert_eq!(first.kind, WireMessageKind::DecodeEmbd);
    assert_eq!(first.request_id, 11);
    assert_eq!(first.session_id, 13);
    assert_eq!(first.sampling, Some(sampling.clone()));
    assert_eq!(first.pos_start, 4);
    assert_eq!(first.token_count, 1);
    assert_eq!(first.state.prompt_token_count, 4);
    assert_eq!(first.state.decode_step, 0);
    assert_eq!(first.state.current_token, 104);
    assert_eq!(first.tokens, vec![104]);

    let second = message
        .update_with_tokens(1, 105, &[101, 102, 104, 105])
        .unwrap();
    assert_eq!(second.request_id, 11);
    assert_eq!(second.session_id, 13);
    assert_eq!(second.sampling, Some(sampling));
    assert_eq!(second.pos_start, 5);
    assert_eq!(second.token_count, 1);
    assert_eq!(second.state.prompt_token_count, 4);
    assert_eq!(second.state.decode_step, 1);
    assert_eq!(second.state.current_token, 105);
    assert_eq!(second.tokens, vec![101, 102, 104, 105]);
    assert!(second.positions.is_empty());
    assert!(second.activation.is_empty());
    assert!(second.raw_bytes.is_empty());
}

#[test]
fn hook_injected_text_concatenates_injection_actions() {
    let outcome = ChatHookOutcome {
        actions: vec![
            ChatHookAction::InjectText {
                text: "[first]\n".to_string(),
            },
            ChatHookAction::None,
            ChatHookAction::InjectText {
                text: "[second]\n".to_string(),
            },
        ],
    };

    assert_eq!(
        hook_injected_text(&outcome),
        Some("[first]\n[second]\n".to_string())
    );
}

#[test]
fn mid_generation_window_requires_minimum_tokens_and_cooldown() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 4.5,
        max_entropy: 5.0,
        mean_margin: 0.02,
        min_margin: 0.01,
        high_entropy_count: 12,
        repetition_count: 0,
    };

    assert!(!mid_generation_window_should_fire(11, &None, &window));
    assert!(!mid_generation_window_should_fire(20, &Some(0), &window));
    assert!(mid_generation_window_should_fire(33, &Some(0), &window));
}

#[test]
fn mid_generation_window_fires_on_repetition_even_with_low_entropy() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 0.3,
        max_entropy: 0.7,
        mean_margin: 0.7,
        min_margin: 0.4,
        high_entropy_count: 0,
        repetition_count: 3,
    };

    assert!(mid_generation_window_should_fire(16, &None, &window));
}

#[test]
fn default_sampling_controls_are_allowed() {
    // When no sampling params are specified, the server applies its own
    // defaults (temp=0.8, top_k=40, top_p=0.95, min_p=0.05) which enable
    // the sampling chain automatically.
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.temperature, 0.8);
    assert_eq!(sampling.top_p, 0.95);
    assert_eq!(sampling.top_k, 40);
    assert_eq!(sampling.min_p, 0.05);
}

#[test]
fn non_default_sampling_controls_are_enabled() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    }))
    .unwrap();

    let sampling = completion_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.seed, 42);
    assert_eq!(sampling.temperature, 0.7);
}

#[test]
fn typed_sampling_penalties_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "presence_penalty": 1.0
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.presence_penalty, 1.0);
}

#[test]
fn extra_sampling_fields_are_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "top_k": 40
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.top_k, 40);
}

#[test]
fn request_defaults_fill_omitted_chat_fields_only() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let sampling = chat_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.presence_penalty, Some(1.25));
    assert_eq!(request.frequency_penalty, Some(0.5));
    assert_eq!(request.seed, Some(77));
    assert_eq!(request.logit_bias, test_request_defaults().logit_bias);
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::One("</stop>".to_string()))
    );
    assert_eq!(sampling.temperature, 0.2);
    assert_eq!(sampling.top_p, 0.9);
    assert_eq!(sampling.presence_penalty, 1.25);
    assert_eq!(sampling.frequency_penalty, 0.5);
    assert_eq!(sampling.seed, 77);
    assert_eq!(sampling.top_k, 12);
    assert_eq!(sampling.min_p, 0.1);
    assert_eq!(sampling.repeat_penalty, 1.2);
    assert_eq!(sampling.penalty_last_n, 64);
    assert_eq!(sampling.logit_bias.len(), 2);
    let template_options = chat_template_options(&request, &test_request_defaults()).unwrap();
    assert_eq!(template_options.enable_thinking, None);
    assert_eq!(
        template_options.reasoning_format,
        Some(ChatReasoningFormat::Hidden)
    );
    assert_eq!(request.reasoning, None);
    assert_eq!(
        GenerationTokenLimit::from_request(request.effective_max_tokens(), 64),
        GenerationTokenLimit::Default(64)
    );
}

#[test]
fn request_defaults_fill_omitted_completion_fields_and_nulls() {
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "top_k": null,
        "repeat_last_n": null,
        "min_p": null
    }))
    .unwrap();

    apply_completion_request_defaults(&mut request, &test_request_defaults());

    let sampling = completion_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.2));
    assert_eq!(request.top_p, Some(0.9));
    assert_eq!(request.presence_penalty, Some(1.25));
    assert_eq!(request.frequency_penalty, Some(0.5));
    assert_eq!(request.seed, Some(77));
    assert_eq!(request.logit_bias, test_request_defaults().logit_bias);
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::One("</stop>".to_string()))
    );
    assert_eq!(sampling.seed, 77);
    assert_eq!(sampling.presence_penalty, 1.25);
    assert_eq!(sampling.frequency_penalty, 0.5);
    assert_eq!(sampling.top_k, 12);
    assert_eq!(sampling.min_p, 0.1);
    assert_eq!(sampling.repeat_penalty, 1.2);
    assert_eq!(sampling.penalty_last_n, 64);
    assert_eq!(sampling.logit_bias.len(), 2);
    assert_eq!(
        GenerationTokenLimit::from_request(request.max_tokens, 48),
        GenerationTokenLimit::Default(48)
    );
}

#[test]
fn explicit_chat_request_values_override_request_defaults() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "temperature": 0.8,
        "top_p": 0.7,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 9,
        "logit_bias": {"7": 1.0},
        "stop": ["USER"],
        "repetition_penalty": 1.8,
        "repeat_last_n": 24,
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let sampling = chat_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.8));
    assert_eq!(request.top_p, Some(0.7));
    assert_eq!(request.presence_penalty, Some(0.1));
    assert_eq!(request.frequency_penalty, Some(0.2));
    assert_eq!(request.seed, Some(9));
    assert_eq!(request.effective_max_tokens(), Some(32));
    assert_eq!(
        request.stop,
        Some(openai_frontend::StopSequence::Many(vec![
            "USER".to_string()
        ]))
    );
    assert_eq!(sampling.top_p, 0.7);
    assert_eq!(sampling.presence_penalty, 0.1);
    assert_eq!(sampling.frequency_penalty, 0.2);
    assert_eq!(sampling.seed, 9);
    assert_eq!(sampling.repeat_penalty, 1.8);
    assert_eq!(sampling.penalty_last_n, 24);
    assert_eq!(sampling.logit_bias.len(), 1);
    let template_options = chat_template_options(&request, &test_request_defaults()).unwrap();
    assert_eq!(template_options.enable_thinking, Some(false));
    assert_eq!(
        template_options.reasoning_format,
        Some(ChatReasoningFormat::Hidden)
    );
    assert_eq!(
        GenerationTokenLimit::from_request(request.effective_max_tokens(), 64),
        GenerationTokenLimit::Explicit(32)
    );
}

#[test]
fn explicit_completion_request_values_override_request_defaults() {
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "prompt": "hello",
        "max_tokens": 12,
        "temperature": 0.6,
        "top_p": 0.4,
        "presence_penalty": 0.25,
        "frequency_penalty": 0.75,
        "seed": 12,
        "logit_bias": {"8": -3.0},
        "stop": ["DONE"],
        "repeat_penalty": 1.4,
        "repeat_last_n": 16,
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    apply_completion_request_defaults(&mut request, &test_request_defaults());

    let sampling = completion_sampling_config(&request).unwrap();
    assert_eq!(request.temperature, Some(0.6));
    assert_eq!(request.top_p, Some(0.4));
    assert_eq!(request.presence_penalty, Some(0.25));
    assert_eq!(request.frequency_penalty, Some(0.75));
    assert_eq!(request.seed, Some(12));
    assert_eq!(request.max_tokens, Some(12));
    assert_eq!(sampling.repeat_penalty, 1.4);
    assert_eq!(sampling.penalty_last_n, 16);
    assert_eq!(sampling.logit_bias.len(), 1);
    assert_eq!(
        GenerationTokenLimit::from_request(request.max_tokens, 48),
        GenerationTokenLimit::Explicit(12)
    );
}

#[test]
fn request_defaults_do_not_make_logprobs_executable() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}}
        },
        "logprobs": true,
        "top_logprobs": 2
    }))
    .unwrap();

    apply_chat_request_defaults(&mut request, &test_request_defaults());

    let error = ensure_chat_runtime_features_supported(&request).unwrap_err();
    assert_eq!(
        unsupported_code(error),
        Some("unsupported_model_feature".to_string())
    );
}

#[test]
fn canonical_reasoning_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn chat_template_options_default_to_hidden_reasoning_parser() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();

    let options = chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
        .expect("template options");

    assert_eq!(options.enable_thinking, None);
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn request_default_reasoning_format_controls_chat_parser_mode() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    let defaults = EmbeddedOpenAiRequestDefaults {
        reasoning_format: Some(EmbeddedReasoningFormat::None),
        ..EmbeddedOpenAiRequestDefaults::default()
    };

    let options = chat_template_options(&request, &defaults).expect("template options");

    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::None));
    assert!(!chat_output_parser_required(&request, &options));
}

#[test]
fn reasoning_effort_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "none"}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn top_level_reasoning_effort_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "none"
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn provider_enable_thinking_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": false},
        "enable_thinking": true
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(true));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn chat_template_kwargs_enable_thinking_overrides_template() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }))
    .unwrap();

    let options =
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default()).unwrap();
    assert_eq!(options.enable_thinking, Some(false));
    assert_eq!(options.reasoning_format, Some(ChatReasoningFormat::Hidden));
}

#[test]
fn thinking_boolean_aliases_override_chat_template_thinking() {
    for field in openai_frontend::THINKING_BOOLEAN_ALIASES {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            (*field): false
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
                .unwrap()
                .enable_thinking,
            Some(false),
            "top-level alias {field}"
        );

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {(*field): false}
        }))
        .unwrap();
        assert_eq!(
            chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
                .unwrap()
                .enable_thinking,
            Some(false),
            "chat_template_kwargs alias {field}"
        );
    }
}

#[test]
fn reasoning_budget_overrides_chat_template_thinking() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"max_tokens": 1024}
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
            .unwrap()
            .enable_thinking,
        Some(true)
    );

    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"enabled": true},
        "thinking_budget": 0
    }))
    .unwrap();
    assert_eq!(
        chat_template_options(&request, &EmbeddedOpenAiRequestDefaults::default())
            .unwrap()
            .enable_thinking,
        Some(false)
    );
}

#[test]
fn logit_bias_is_enabled() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"123": -50.0, "456": 12.5}
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.logit_bias.len(), 2);
    assert_eq!(sampling.logit_bias[0].token_id, 123);
    assert_eq!(sampling.logit_bias[0].bias, -50.0);
    assert_eq!(sampling.logit_bias[1].token_id, 456);
    assert_eq!(sampling.logit_bias[1].bias, 12.5);
}

#[test]
fn invalid_logit_bias_returns_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "logit_bias": {"not-a-token": 1.0}
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("invalid_value"));
}

#[test]
fn unsupported_extra_generation_fields_return_openai_error() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "typical_p": 0.5
    }))
    .unwrap();

    let error = chat_sampling_config(&request).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn min_p_is_accepted_and_forwarded() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}],
        "min_p": 0.1
    }))
    .unwrap();

    let sampling = chat_sampling_config(&request).unwrap();
    assert!(sampling.enabled);
    assert_eq!(sampling.min_p, 0.1);
}

#[test]
fn maps_generation_exhaustion_to_length_finish_reason() {
    assert_eq!(finish_reason_for_generation(true), FinishReason::Length);
    assert_eq!(finish_reason_for_generation(false), FinishReason::Stop);
}

#[test]
fn generation_ids_are_unique_under_fast_creation() {
    let ids = (0..1024)
        .map(|_| OpenAiGenerationIds::new(OpenAiCacheHints::default()))
        .collect::<Vec<_>>();
    let mut sessions = std::collections::BTreeSet::new();
    let mut requests = std::collections::BTreeSet::new();
    for id in ids {
        assert!(sessions.insert(id.session_id));
        assert!(requests.insert(id.request_id));
    }
}

#[test]
fn prefill_chunk_schedule_parses_and_repeats_last_size() {
    let schedule = PrefillChunkSchedule::parse(Some("128, 256,512"))
        .unwrap()
        .unwrap();
    assert_eq!(schedule.label(), "128,256,512");
    assert_eq!(schedule.chunk_size_for(0), 128);
    assert_eq!(schedule.chunk_size_for(1), 256);
    assert_eq!(schedule.chunk_size_for(2), 512);
    assert_eq!(schedule.chunk_size_for(3), 512);
}

#[test]
fn prefill_chunk_schedule_rejects_bad_sizes() {
    assert!(PrefillChunkSchedule::parse(Some("128,0")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("128,,256")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("abc")).is_err());
}

#[test]
fn prefill_chunk_policy_keeps_legacy_schedule_behavior() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "fixed",
        schedule: Some("128,256,384"),
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    assert_eq!(planner.chunk_size_for(2, 512), 384);
    assert_eq!(planner.chunk_size_for(3, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_grows_when_downstream_wait_is_hidden() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_can_advance_without_observations() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(2, 512), 384);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(3, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_backs_off_when_wait_is_exposed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 10.0,
    });
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 150.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 128);
}

#[test]
fn prefill_adaptive_ramp_backs_off_when_write_is_exposed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 90.0,
        downstream_wait_ms: 0.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 128);
}

#[test]
fn prefill_adaptive_ramp_keeps_short_prompts_fixed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();

    assert_eq!(planner.chunk_size_for(0, 256), 256);
    assert_eq!(planner.chunk_size_for(0, 257), 128);
}

#[test]
fn prefill_transport_ewma_seeds_adaptive_ramp() {
    let config = prefix_cache_test_config();
    let pool = PersistentStageLanePool {
        config: config.clone(),
        timeout_secs: 5,
        telemetry: Telemetry::new(None, 1, config, crate::telemetry::TelemetryLevel::Off),
        lanes: Mutex::new(Vec::new()),
        prefill_transport: Mutex::new(PrefillTransportEstimate::default()),
        next_lane_id: AtomicU64::new(0),
        capacity: 1,
    };
    let mut stats = StageReplyStats::default();
    stats.observe_prefill_edge_transport(1, 1_000, 0, 1_048_576);
    pool.observe_prefill_transport(&stats, 10.0, 1);

    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.observe(pool.prefill_transport_seed().unwrap());

    assert_eq!(planner.chunk_size_for(0, 512), 256);
}

#[test]
fn persistent_lane_ready_handshake_times_out_for_silent_downstream() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(200));
    });
    let mut client = TcpStream::connect(address).unwrap();

    let error = receive_persistent_lane_ready(&mut client, Duration::from_millis(25)).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("persistent downstream lane did not become ready")
    );
    server.join().unwrap();
}

#[test]
fn steady_state_lane_reconnect_deadline_is_much_shorter_than_warmup() {
    // A mid-life reconnect to an already-serving mesh must fail fast so a new
    // request routed to a dead stage errors quickly instead of stalling for the
    // full warmup deadline. Guard the invariant that the steady-state deadline
    // stays well under the warmup deadline.
    assert!(
        LANE_STEADY_CONNECT_TIMEOUT < LANE_READY_READ_TIMEOUT,
        "steady-state reconnect deadline must be shorter than the warmup deadline"
    );
    assert!(
        LANE_STEADY_CONNECT_TIMEOUT <= Duration::from_secs(5),
        "steady-state reconnect deadline must stay small enough to fail fast on a dead stage"
    );
}

#[test]
fn steady_state_ready_handshake_times_out_fast_for_silent_downstream() {
    // A downstream that accepts the TCP connection but never sends the ready
    // frame must be bounded by the supplied deadline, not hang. This mirrors the
    // silent-downstream case a dead split stage produces on a new request.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(200));
    });
    let mut client = TcpStream::connect(address).unwrap();

    let start = std::time::Instant::now();
    let error = receive_persistent_lane_ready(&mut client, Duration::from_millis(25)).unwrap_err();
    let elapsed = start.elapsed();

    assert!(
        error
            .to_string()
            .contains("persistent downstream lane did not become ready")
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "handshake read must fail within the supplied deadline, took {elapsed:?}"
    );
    server.join().unwrap();
}

#[test]
fn model_matching_is_exact_for_mesh_style_ids() {
    ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
    )
    .unwrap();

    let error = ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "org/repo:Q5_K_M",
    )
    .unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn model_matching_normalizes_default_revision() {
    // Advertised with @main, requested without (public display form)
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF:UD-Q4_K_XL",
    )
    .unwrap();

    // Advertised without, requested with @main
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
    )
    .unwrap();

    // Both with @main — exact match still works
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
    )
    .unwrap();

    // Bare repo@main without selector
    ensure_requested_model("org/repo@main", "org/repo").unwrap();

    // Different quants still rejected
    let error = ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF:Q5_K_M",
    )
    .unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn rejects_requests_that_exceed_context_window() {
    ensure_context_capacity(4, 4, 8).unwrap();

    let error = ensure_context_capacity(5, 4, 8).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn omitted_max_tokens_can_use_remaining_context_budget() {
    let limit = GenerationTokenLimit::from_request(None, CONTEXT_BUDGET_MAX_TOKENS);
    assert_eq!(limit.resolve(5, 8).unwrap(), 3);
}

#[test]
fn omitted_max_tokens_with_embedded_default_is_bounded() {
    // Server picked DEFAULT_EMBEDDED_MAX_TOKENS as the cap because the
    // client omitted max_tokens. With a large ctx window the cap is
    // the binding limit.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let ctx_size = 32_000;
    let resolved = limit.resolve(128, ctx_size).unwrap();
    assert_eq!(resolved, DEFAULT_EMBEDDED_MAX_TOKENS);
    assert!((resolved as usize) < ctx_size);
}

#[test]
fn omitted_max_tokens_clamps_to_remaining_budget_in_small_ctx() {
    // When the configured ctx_size is smaller than the server-picked
    // default, the omitted-max_tokens path must clamp to remaining
    // budget rather than reject the request. The client didn't ask
    // for the specific number; the server picked it.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let ctx_size = 1024;
    let prompt_tokens = 128;
    let resolved = limit.resolve(prompt_tokens, ctx_size).unwrap();
    assert_eq!(resolved, (ctx_size - prompt_tokens) as u32);
    assert!(resolved < DEFAULT_EMBEDDED_MAX_TOKENS);
}

#[test]
fn omitted_max_tokens_errors_only_when_prompt_already_exceeds_ctx() {
    // Even on the silently-clamping default path, a prompt that
    // already overflows the context window is an error the client
    // needs to see.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let error = limit.resolve(2048, 1024).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn explicit_max_tokens_still_errors_when_too_large_for_ctx() {
    // Client-asserted max_tokens that won't fit is still a hard error.
    // The clamping behavior applies only to the server-picked default.
    let limit = GenerationTokenLimit::from_request(Some(4), 999);
    assert_eq!(limit.resolve(4, 8).unwrap(), 4);

    let error = limit.resolve(5, 8).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn strip_default_revision_removes_at_main_before_quant() {
    assert_eq!(
        super::strip_default_revision("org/repo@main:Q4"),
        "org/repo:Q4"
    );
}

#[test]
fn strip_default_revision_removes_at_main_at_end() {
    assert_eq!(super::strip_default_revision("org/repo@main"), "org/repo");
}

#[test]
fn strip_default_revision_preserves_mainland() {
    assert_eq!(
        super::strip_default_revision("org/repo@mainland:Q4"),
        "org/repo@mainland:Q4"
    );
}

#[test]
fn strip_default_revision_preserves_no_revision() {
    assert_eq!(super::strip_default_revision("org/repo:Q4"), "org/repo:Q4");
}
