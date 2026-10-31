//! Explicit real-model recurrent cache fixture, including admitted lane capacity.
use super::*;

const RECURRENT_TEST_CHECKPOINT_RAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub(super) fn recurrent_test_backend(
    run_id: &str,
    backend_model_id: &str,
    ctx_size: usize,
    batch_size: u32,
    default_max_tokens: u32,
    token_budget: usize,
) -> Result<(StageOpenAiBackend, SpeculativeDecodeConfig)> {
    recurrent_test_backend_with_lanes(
        run_id,
        backend_model_id,
        ctx_size,
        batch_size,
        default_max_tokens,
        token_budget,
        1,
    )
}

pub(super) fn recurrent_test_backend_with_lanes(
    run_id: &str,
    backend_model_id: &str,
    ctx_size: usize,
    batch_size: u32,
    default_max_tokens: u32,
    token_budget: usize,
    lane_count: u32,
) -> Result<(StageOpenAiBackend, SpeculativeDecodeConfig)> {
    skippy_runtime::test_support::require_native_runtime();
    let model_path = std::env::var_os("SKIPPY_RECURRENT_CACHE_TEST_MODEL").ok_or_else(|| {
        anyhow::anyhow!("SKIPPY_RECURRENT_CACHE_TEST_MODEL is required for this ignored test")
    })?;
    let layer_count =
        std::env::var_os("SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS").ok_or_else(|| {
            anyhow::anyhow!(
                "SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS is required for this ignored test"
            )
        })?;
    let layer_count = layer_count
        .to_string_lossy()
        .parse::<u32>()
        .map_err(|error| anyhow::anyhow!("invalid recurrent cache test layer count: {error}"))?;
    let config = StageConfig {
        run_id: run_id.to_string(),
        topology_id: run_id.to_string(),
        model_id: std::env::var("SKIPPY_RECURRENT_CACHE_TEST_MODEL_ID")
            .unwrap_or_else(|_| "unsloth/Qwen3.5-0.8B-GGUF:Q6_K".to_string()),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(model_path.to_string_lossy().into_owned()),
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: u32::try_from(ctx_size)
            .map_err(|error| anyhow::anyhow!("invalid recurrent test context size: {error}"))?,
        lane_count,
        n_batch: Some(batch_size),
        n_ubatch: Some(batch_size),
        n_gpu_layers: std::env::var("SKIPPY_RECURRENT_CACHE_TEST_GPU_LAYERS")
            .ok()
            .map(|value| value.parse::<i32>())
            .transpose()
            .map_err(|error| anyhow::anyhow!("invalid recurrent cache GPU layer count: {error}"))?
            .unwrap_or(0),
        mmap: Some(true),
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_protocol::SplitMode::Auto,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: Default::default(),
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
        cache_idle_slots: None,
        resident_tensor_names: Vec::new(),
        selected_device: None,
        kv_cache: Some(StageKvCacheConfig {
            exact_max_bytes: None,
            mode: StageKvCacheMode::LookupRecord,
            max_entries: 8,
            max_bytes: 0,
            l2_max_bytes: RECURRENT_TEST_CHECKPOINT_RAM_BYTES,
            codec: skippy_protocol::StageKvCacheCodec::Native,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
        }),
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
        ..StageConfig::default()
    };
    let runtime = load_runtime(&config)?
        .ok_or_else(|| anyhow::anyhow!("recurrent cache test runtime was not loaded"))?;
    let kv = KvStageIntegration::from_loaded_model(
        &config,
        crate::runtime_state::loaded_model_state_kind(Some(&runtime)),
        crate::runtime_state::loaded_memory_cache_capabilities(Some(&runtime)),
        None,
    )?
    .map(Arc::new)
    .ok_or_else(|| anyhow::anyhow!("recurrent cache test did not enable KV integration"))?;
    assert!(
        kv.records_resident_prefixes(),
        "fixture must exercise native prefix ownership"
    );
    assert!(
        kv.resident_requires_checkpoint(),
        "fixture requires recurrent native memory"
    );
    assert_eq!(
        kv.resident_checkpoint_max_bytes,
        RECURRENT_TEST_CHECKPOINT_RAM_BYTES
    );
    assert_eq!(kv.resident_config.max_entries, 8);
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    let speculative = SpeculativeDecodeConfig::default();
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, None, telemetry.clone())?;
    let backend = StageOpenAiBackend {
        runtime: runtime.clone(),
        workload: Default::default(),
        config,
        telemetry,
        model_id: backend_model_id.to_string(),
        default_max_tokens,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        thinking: None,
        ctx_size,
        mode: InferenceBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: speculative.clone(),
        speculation_governor: None,
        runahead_governor: None,
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(lane_count as usize)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: std::time::Duration::from_secs(10),
        generation_service_estimator: Arc::new(crate::frontend::GenerationServiceEstimator::new(
            lane_count as usize,
        )),
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(token_budget)),
        hook_policy: None,
        generation_receipt: None,
        generation_lifecycle: None,
        linear_proposal_ingress: None,
        kv: Some(kv),
        iteration_scheduler,
    };
    Ok((backend, speculative))
}
