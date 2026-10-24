//! Explicit real-model HTTP proof; no synthetic fixture can certify readout scores.

use super::*;
use crate::runtime_state::{loaded_memory_cache_capabilities, loaded_model_state_kind};
use axum::body::{Body, to_bytes};
use axum::http::Request;
use tower::ServiceExt;

mod benchmark;
mod scheduling;
mod split;

/// Configure a real causal GGUF with bounded CPU lanes and prefix caching.
fn model_config() -> Result<StageConfig> {
    skippy_runtime::test_support::require_native_runtime();
    let path = env::var("SKIPPY_DECISION_MODEL").context("SKIPPY_DECISION_MODEL is required")?;
    let layers = ModelInfo::open(&path)?
        .tensors()?
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .context("model needs transformer layers")?
        + 1;
    Ok(StageConfig {
        run_id: "decision-proof".into(),
        topology_id: "decision-local".into(),
        model_id: "decision-model".into(),
        model_path: Some(path),
        stage_id: "stage-0".into(),
        layer_end: layers,
        ctx_size: 4096,
        lane_count: 4,
        // Match the local and split prefill shapes so the comparison measures
        // transport/readout correctness rather than different matrix kernels.
        n_batch: Some(128),
        n_ubatch: Some(128),
        n_gpu_layers: 0,
        kv_offload: Some(false),
        op_offload: Some(false),
        selected_device: Some(StageDevice {
            backend_device: "CPU".into(),
            stable_id: None,
            index: None,
            vram_bytes: None,
        }),
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".into(),
        kv_cache: Some(serde_json::from_value(json!({
            "mode": "lookup-record",
            "min_tokens": 32, "shared_prefix_stride_tokens": 32,
            "shared_prefix_record_limit": 4, "max_entries": 16
        }))?),
        ..StageConfig::default()
    })
}

/// Open the real-model decision backend used by HTTP and scheduling tests.
fn backend() -> Result<Arc<StageOpenAiBackend>> {
    Ok(Arc::new(backend_with_config(model_config()?)?))
}

/// Fix arithmetic and batching shape for local-versus-split score comparisons.
fn reference_config() -> Result<StageConfig> {
    Ok(StageConfig {
        // Hold native matrix shape fixed across cache hits and stage splits.
        // Larger-shape numerical and concurrent-wave tests are separate.
        lane_count: 1,
        n_batch: Some(1),
        n_ubatch: Some(1),
        flash_attn_type: skippy_protocol::FlashAttentionType::Enabled,
        ..model_config()?
    })
}

/// Build a serving backend with explicit admission limits, scheduler and cache state.
fn backend_with_config(config: StageConfig) -> Result<StageOpenAiBackend> {
    let mut backend = support::local_openai_backend(config.clone(), config.model_id.clone())?;
    backend.generation_limit = Arc::new(GenerationConcurrencyController::fixed(
        config.lane_count as usize,
    ));
    backend.generation_queue_limit = 8;
    backend.generation_token_budget = Arc::new(GenerationTokenBudget::new(16384));
    backend.iteration_scheduler = IterationScheduler::new(
        backend.runtime.clone(),
        &config,
        8,
        true,
        None,
        backend.telemetry.clone(),
    )?;
    backend.kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &config,
        loaded_model_state_kind(Some(&backend.runtime)),
        loaded_memory_cache_capabilities(Some(&backend.runtime)),
        None,
        None,
    )?
    .map(Arc::new);
    if config
        .kv_cache
        .as_ref()
        .is_some_and(|cache| cache.mode == StageKvCacheMode::LookupRecord)
    {
        anyhow::ensure!(backend.kv.is_some(), "model must admit prefix caching");
    }
    Ok(backend)
}

/// Construct shared-state questions exercising both causal scoring methods.
fn request() -> Value {
    json!({
        "model": "decision-model",
        "state": "The customer requests a refund for an item delivered yesterday. Billing owns refunds; shipping owns delivery tracking.".repeat(4),
        "questions": {
            "department": {"type": "choice", "criteria": {"B": "billing", "S": "shipping"}},
            "action": {"type": "choice", "criteria": {"contact billing support": "refund", "track shipment": "delivery"}}
        }
    })
}

/// Submit a System One request through the real HTTP router.
async fn post(backend: Arc<StageOpenAiBackend>, body: Value) -> Result<(StatusCode, Value)> {
    post_route(backend, "/systemone", body).await
}

/// Submit JSON to a selected route and decode its status and response body.
async fn post_route(
    backend: Arc<StageOpenAiBackend>,
    route: &str,
    body: Value,
) -> Result<(StatusCode, Value)> {
    let reply = skippy_inference_api::router(backend)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(route)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body)?))?,
        )
        .await?;
    let status = reply.status();
    let json = serde_json::from_slice(&to_bytes(reply.into_body(), 1024 * 1024).await?)?;
    Ok((status, json))
}

/// Compare per-candidate scores within the native arithmetic tolerance.
fn assert_same_scores(left: &Value, right: &Value) {
    for key in ["department", "action"] {
        let left = left["readouts"][key]["candidate_log_probabilities"]
            .as_array()
            .unwrap();
        let right = right["readouts"][key]["candidate_log_probabilities"]
            .as_array()
            .unwrap();
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!(
                (left.as_f64().unwrap() - right.as_f64().unwrap()).abs() < 1e-3,
                "question {key}: expected {left}, actual {right}"
            );
        }
    }
}

/// HTTP readouts reuse prefixes and release all sessions, including invalid requests.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real causal GGUF"]
async fn causal_decisions_http_reuses_prefixes_and_releases_all_lanes() -> Result<()> {
    let backend = Arc::new(backend_with_config(reference_config()?)?);
    assert!(
        backend
            .runtime
            .lock()
            .unwrap()
            .supports_system_one_endpoint()
    );
    let started = std::time::Instant::now();
    let (status, cold) = post(backend.clone(), request()).await?;
    let cold_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(status, StatusCode::OK, "{cold}");
    let started = std::time::Instant::now();
    let (status, warm) = post(backend.clone(), request()).await?;
    let warm_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(status, StatusCode::OK, "{warm}");
    assert_same_scores(&cold, &warm);
    assert_eq!(warm["usage"]["output_tokens"], 0);
    assert!(
        cold["readout_usage"]["cached_tokens"].as_u64().unwrap() > 0,
        "questions and teacher-forced branches must share their prepared prefix"
    );
    assert!(
        warm["readout_usage"]["evaluated_tokens"].as_u64().unwrap()
            < cold["readout_usage"]["evaluated_tokens"].as_u64().unwrap()
    );
    assert_eq!(
        backend
            .runtime
            .lock()
            .unwrap()
            .session_stats()
            .active_sessions,
        0
    );
    let mut invalid = request();
    invalid["questions"]["department"]["criteria"] = json!({"one": "only one choice"});
    let (status, _) = post(backend.clone(), invalid).await?;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        backend
            .runtime
            .lock()
            .unwrap()
            .session_stats()
            .active_sessions,
        0
    );
    if let Some(path) = env::var_os("SKIPPY_DECISION_REPORT") {
        fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "cold_ms": cold_ms, "warm_ms": warm_ms, "cold": cold, "warm": warm
            }))?,
        )?;
    }
    Ok(())
}

/// The public Decisions API preserves question and candidate ordering.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real causal GGUF"]
async fn causal_decisions_openai_answers_keep_question_and_candidate_order() -> Result<()> {
    let backend = backend()?;
    let body = json!({"model": "decision-model", "input": "A delivered order was charged twice.",
    "questions": [
        {"type": "choice", "name": "z_department", "instructions": "Select a department.",
         "choices": [{"value":"S","description":"shipping"}, {"value":"B","description":"billing"}]},
        {"type": "choice", "name": "a_action", "instructions": "Select an action.",
         "choices": [{"value":"contact billing support"}, {"value":"track shipment"}]}
    ]});
    let (status, response) = post_route(backend.clone(), "/v1/decisions", body).await?;
    assert_eq!(status, StatusCode::OK, "{response}");
    let answers = response["answers"].as_array().context("ordered answers")?;
    assert_eq!(answers.len(), 2);
    assert_eq!(answers[0]["name"], "z_department");
    assert_eq!(answers[1]["name"], "a_action");
    assert_eq!(answers[0]["probabilities"][0]["value"], "S");
    assert_eq!(answers[0]["probabilities"][1]["value"], "B");
    assert_eq!(answers[0]["readout"]["method"], "selected_token_logits");
    assert_eq!(answers[1]["readout"]["method"], "candidate_log_likelihood");
    assert_eq!(response["usage"]["output_tokens"], 0);
    assert_eq!(
        backend
            .runtime
            .lock()
            .unwrap()
            .session_stats()
            .active_sessions,
        0
    );
    Ok(())
}
