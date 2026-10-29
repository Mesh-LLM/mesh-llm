//! A real two-stage binary exchange, including warm prefix restoration.

use super::*;
use crate::embedded::{EmbeddedServerHandle, EmbeddedState};
use crate::frontend::generation::PersistentStageLanePool;

/// Reserve a loopback address for one split-stage test server.
fn address() -> Result<SocketAddr> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?)
}

/// Construct the two-stage partition used for causal split comparisons.
pub(super) fn partition(config: &StageConfig) -> Result<[StageConfig; 2]> {
    let split = config.layer_end / 2;
    anyhow::ensure!(split > 0, "model must have at least two layers");
    let ranges = [(0, split), (split, config.layer_end)];
    let plans = skippy_runtime::plan_gguf_stage_runtime_plans(
        Path::new(config.model_path.as_deref().context("model path")?),
        &ranges,
        config.ctx_size,
        config.lane_count,
    )?;
    anyhow::ensure!(plans.len() == 2, "planner must admit both stages");
    let mut stages = [config.clone(), config.clone()];
    for (index, (stage, plan)) in stages.iter_mut().zip(plans).enumerate() {
        stage.stage_id = format!("stage-{index}");
        stage.stage_index = index as u32;
        stage.topology_id = "decision-split".into();
        (stage.layer_start, stage.layer_end) = ranges[index];
        stage.bind_addr = address()?.to_string();
        stage.resident_tensor_names = plan.resident_tensor_names;
        stage.execution_contract = plan.execution_contract;
        stage.activation_import_identities = plan.activation_import_identities;
        stage.activation_import_bindings = plan.activation_import_bindings;
        stage.activation_export_identities = plan.activation_export_identities;
        stage.activation_export_bindings = plan.activation_export_bindings;
        // Preserve the caller's chunk shape across both stage partitions.
    }
    stages[0].downstream = Some(PeerConfig {
        persistent_lane_marker: false,
        stage_id: stages[1].stage_id.clone(),
        stage_index: 1,
        endpoint: stages[1].bind_addr.clone(),
    });
    stages[1].upstream = Some(PeerConfig {
        persistent_lane_marker: false,
        stage_id: stages[0].stage_id.clone(),
        stage_index: 0,
        endpoint: stages[0].bind_addr.clone(),
    });
    Ok(stages)
}

/// Start the terminal stage and return its address and owned shutdown handle.
pub(super) async fn terminal(config: StageConfig) -> Result<EmbeddedServerHandle> {
    let handle = crate::embedded::start_binary_stage(crate::binary_transport::BinaryStageOptions {
        tuning: Default::default(),
        bind_addr: config.bind_addr.parse()?,
        config,
        topology: None,
        metrics_otlp_grpc: None,
        telemetry_queue_capacity: 1,
        telemetry_level: crate::telemetry::TelemetryLevel::Off,
        max_inflight: 4,
        reply_credit_limit: None,
        async_prefill_forward: false,
        downstream_wire_condition: WireCondition::new(0.0, None)?,
        downstream_connect_timeout_secs: 5,
        native_mtp_enabled: false,
        last_stage_decode_batch: None,
        continuous_batching: true,
        openai: None,
        compute_meter: None,
        split_dflash: None,
        l3_manager: None,
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let status = handle.status();
        match status.state {
            EmbeddedState::Ready => return Ok(handle),
            EmbeddedState::Failed | EmbeddedState::Stopped => {
                bail!("terminal startup failed: {:?}", status.last_error)
            }
            _ => {}
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "terminal startup timed out"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Build the source-stage backend connected to the test terminal stage.
pub(super) fn source(config: StageConfig) -> Result<Arc<StageOpenAiBackend>> {
    let mut backend = backend_with_config(config.clone())?;
    let width = backend
        .runtime
        .lock()
        .unwrap()
        .output_activation_boundary()
        .context("source must export activations")?
        .primary_token_width("decision split test")?;
    let lanes = PersistentStageLanePool::new(
        &config,
        config.lane_count as usize,
        5,
        backend.telemetry.clone(),
    )?;
    backend.mode = InferenceBackendMode::EmbeddedStageZero {
        config,
        prefill_chunk_policy: PrefillChunkPolicy::Fixed { chunk_size: 128 },
        activation_width: width,
        downstream_wire_condition: WireCondition::new(0.0, None)?,
        prefill_reply_credit_limit: 0,
        lane_pool: lanes,
        prediction_returns: None,
    };
    Ok(Arc::new(backend))
}

/// Split readouts match local scores and reuse their distributed prefixes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real split-capable causal GGUF"]
async fn causal_decisions_split_matches_local_and_reuses_prefixes() -> Result<()> {
    let config = reference_config()?;
    let local = Arc::new(backend_with_config(config.clone())?);
    let (status, expected) = post(local, request()).await?;
    assert_eq!(status, StatusCode::OK, "{expected}");
    let [first, last] = partition(&config)?;
    let handle = terminal(last).await?;
    let backend = source(first)?;
    let mut runs = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        let (status, actual) = post(backend.clone(), request()).await?;
        assert_eq!(status, StatusCode::OK, "{actual}");
        assert_same_scores(&expected, &actual);
        assert!(actual["readout_usage"]["cached_tokens"].as_u64().unwrap() > 0);
        assert_eq!(
            backend
                .runtime
                .lock()
                .unwrap()
                .session_stats()
                .active_sessions,
            0
        );
        runs.push(
            json!({"elapsed_ms": started.elapsed().as_secs_f64() * 1000.0, "response": actual}),
        );
    }
    drop(backend);
    handle.shutdown().await?;
    if let Some(path) = env::var_os("SKIPPY_DECISION_SPLIT_REPORT") {
        fs::write(
            path,
            serde_json::to_vec_pretty(&json!({"local": expected, "split": runs}))?,
        )?;
    }
    Ok(())
}

/// A source-only cache hit recomputes state missing on the downstream stage.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real split-capable causal GGUF"]
async fn causal_decisions_split_recomputes_when_only_the_source_has_cached_state() -> Result<()> {
    let [first, mut last] = partition(&model_config()?)?;
    last.kv_cache
        .as_mut()
        .context("terminal cache config")?
        .mode = StageKvCacheMode::Disabled;
    let handle = terminal(last).await?;
    let backend = source(first)?;
    for _ in 0..2 {
        let (status, response) = post(backend.clone(), request()).await?;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(response["usage"]["output_tokens"], 0);
        assert_eq!(
            response["readout_usage"]["cached_tokens"], 0,
            "a local hit must not be counted when a downstream stage misses"
        );
        assert!(
            response["readout_usage"]["evaluated_tokens"]
                .as_u64()
                .unwrap()
                > 0
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
    }
    drop(backend);
    handle.shutdown().await?;
    Ok(())
}
