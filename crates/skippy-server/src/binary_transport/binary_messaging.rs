use std::{
    future::Future,
    io::{self, Write},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use super::stage_execution::{
    consume_optional_client_ready_hello, prepare_binary_stage_connection,
    take_warm_or_connect_downstream, warm_downstream_preconnect_enabled,
};
use super::{
    decode_batcher::DecodeFrameBatcher,
    direct_return::{PredictionReturnHub, PredictionReturnSinks},
    options::BinaryStageOptions,
    preconnect::spawn_downstream_preconnector,
};
use crate::{
    cli::ServeBinaryArgs,
    config::validate_config,
    frontend::{self, EmbeddedOpenAiArgs},
    kv_integration::KvStageIntegration,
    runtime_state::load_runtime,
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use skippy_protocol::binary::{WireMessageKind, read_stage_message, send_ready};

mod async_forwarder;
mod connection;
pub(in crate::binary_transport) mod reply;
mod summary;
mod telemetry;

use self::connection::handle_binary_connection;

pub async fn serve_binary(args: ServeBinaryArgs) -> Result<()> {
    serve_binary_stage(BinaryStageOptions::from_cli_args(args)?).await
}

pub async fn serve_binary_stage(options: BinaryStageOptions) -> Result<()> {
    serve_binary_stage_with_shutdown(options, std::future::pending::<()>()).await
}

pub async fn serve_binary_stage_with_shutdown(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_task = tokio::spawn({
        let stop = stop.clone();
        async move {
            shutdown.await;
            stop.store(true, Ordering::SeqCst);
        }
    });
    let result = run_binary_stage(options, stop);
    stop_task.abort();
    result
}

fn run_binary_stage(options: BinaryStageOptions, shutdown: Arc<AtomicBool>) -> Result<()> {
    let BinaryStageOptions {
        config,
        topology,
        bind_addr,
        activation_width,
        wire_dtype,
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        telemetry_level,
        max_inflight,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        native_mtp_enabled,
        openai,
    } = options;
    let native_mtp_enabled = native_mtp_enabled && config.native_mtp_enabled;
    validate_config(&config, topology.as_ref())?;
    let max_inflight = max_inflight.min(config.lane_count as usize);
    let telemetry = Telemetry::new(
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        config.clone(),
        telemetry_level,
    );
    telemetry.emit("stage.binary_server_start", lifecycle_attrs(&config));
    let warm_downstream = Arc::new(Mutex::new(None));
    if warm_downstream_preconnect_enabled() {
        spawn_downstream_preconnector(config.clone(), warm_downstream.clone(), shutdown.clone());
    }
    let runtime = load_runtime(&config)?.context("binary stage server requires model_path")?;
    let decode_frame_batcher = DecodeFrameBatcher::new(runtime.clone(), max_inflight);
    if max_inflight > 0 {
        let timer = Instant::now();
        let sessions = runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?
            .prewarm_idle_sessions(max_inflight)
            .context("prewarm binary stage runtime sessions")?;
        let mut attrs = lifecycle_attrs(&config);
        attrs.insert("llama_stage.max_inflight".to_string(), json!(max_inflight));
        attrs.insert(
            "llama_stage.lane_count".to_string(),
            json!(sessions.lane_count),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_active".to_string(),
            json!(sessions.active_sessions),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_idle".to_string(),
            json!(sessions.idle_sessions),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed().as_secs_f64() * 1000.0),
        );
        telemetry.emit("stage.binary_runtime_prewarm", attrs);
    }
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
    let prediction_returns = Arc::new(PredictionReturnHub::default());
    let prediction_return_sinks = Arc::new(PredictionReturnSinks::default());
    let listener = TcpListener::bind(bind_addr)?;
    listener.set_nonblocking(true)?;
    if let Some(openai_options) = openai {
        if config.stage_index != 0 || config.layer_start != 0 {
            bail!("--openai-bind-addr is only supported on stage 0");
        }
        let openai_config = config.clone();
        let openai_runtime = runtime.clone();
        let openai_telemetry = telemetry.clone();
        let openai_prediction_returns = prediction_returns.clone();
        tokio::spawn(async move {
            if let Err(error) = frontend::serve_embedded_openai(EmbeddedOpenAiArgs {
                bind_addr: openai_options.bind_addr,
                config: openai_config,
                runtime: openai_runtime,
                model_id: openai_options.model_id,
                default_max_tokens: openai_options.default_max_tokens,
                request_defaults: frontend::EmbeddedOpenAiRequestDefaults::default(),
                generation_concurrency: openai_options.generation_concurrency,
                prefill_chunk_size: openai_options.prefill_chunk_size,
                prefill_chunk_policy: openai_options.prefill_chunk_policy,
                prefill_chunk_schedule: openai_options.prefill_chunk_schedule,
                prefill_adaptive_start: openai_options.prefill_adaptive_start,
                prefill_adaptive_step: openai_options.prefill_adaptive_step,
                prefill_adaptive_max: openai_options.prefill_adaptive_max,
                draft_model_path: openai_options.draft_model_path,
                speculative_window: openai_options.speculative_window,
                adaptive_speculative_window: openai_options.adaptive_speculative_window,
                draft_n_gpu_layers: openai_options.draft_n_gpu_layers,
                speculative: openai_options.speculative.clone(),
                ngram_min: openai_options
                    .speculative
                    .ngram
                    .as_ref()
                    .filter(|ngram| ngram.kind == frontend::NgramProposerKind::Simple)
                    .map_or(0, |ngram| ngram.min_ngram),
                ngram_max: openai_options
                    .speculative
                    .ngram
                    .as_ref()
                    .filter(|ngram| ngram.kind == frontend::NgramProposerKind::Simple)
                    .map_or(0, |ngram| ngram.max_proposal_tokens.min(ngram.max_ngram)),
                native_mtp_enabled: native_mtp_enabled
                    && openai_options.speculative.native_mtp.enabled,
                native_mtp_draft_model_path: None,
                native_mtp_max_tokens: openai_options.native_mtp_max_tokens,
                native_mtp_min_tokens: openai_options.native_mtp_min_tokens,
                activation_width,
                wire_dtype,
                reply_credit_limit,
                downstream_connect_timeout_secs,
                downstream_wire_condition,
                prediction_returns: Some(openai_prediction_returns),
                telemetry: openai_telemetry,
                hook_policy: None,
                openai_guardrails: Some(frontend::OpenAiGuardrailsConfig::disabled_for_skippy()),
            })
            .await
            {
                eprintln!("embedded OpenAI server failed: {error:#}");
            }
        });
    }
    println!(
        "skippy-server listening: binary={} stage_id={} layer_range={}..{} activation_width={} dtype={:?}",
        bind_addr,
        config.stage_id,
        config.layer_start,
        config.layer_end,
        activation_width,
        wire_dtype,
    );

    while !shutdown.load(Ordering::SeqCst) {
        let (mut upstream, _) = match listener.accept() {
            Ok(conn) => conn,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(error) => return Err(error).context("accept binary stage connection"),
        };
        prepare_binary_stage_connection(&upstream)?;
        let peer_addr = upstream.peer_addr().ok();
        eprintln!(
            "binary accepted connection: stage_id={} peer={peer_addr:?}",
            config.stage_id
        );
        let config = config.clone();
        let topology = topology.clone();
        let runtime = runtime.clone();
        let decode_frame_batcher = decode_frame_batcher.clone();
        let kv = kv.clone();
        let telemetry = telemetry.clone();
        let warm_downstream = warm_downstream.clone();
        let prediction_returns = prediction_returns.clone();
        let prediction_return_sinks = prediction_return_sinks.clone();
        thread::spawn(move || {
            let connection_result = (|| -> Result<()> {
                eprintln!(
                    "binary sending ready: stage_id={} peer={peer_addr:?}",
                    config.stage_id
                );
                consume_optional_client_ready_hello(&mut upstream)
                    .context("consume optional client ready hello")?;
                send_ready(&mut upstream).context("failed to send binary ready")?;
                upstream.flush().ok();
                eprintln!(
                    "binary sent ready: stage_id={} peer={peer_addr:?}",
                    config.stage_id
                );
                let first_message = match read_stage_message(&mut upstream, activation_width) {
                    Ok(message) => message,
                    Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                    Err(error) => return Err(error.into()),
                };
                if first_message.kind == WireMessageKind::PredictionReturnOpen {
                    if config.stage_index == 0 {
                        return prediction_returns
                            .handle_return_connection(first_message, upstream);
                    }
                    return prediction_return_sinks.insert_opened_sink(first_message, upstream);
                }
                let downstream = take_warm_or_connect_downstream(
                    &config,
                    &warm_downstream,
                    downstream_connect_timeout_secs,
                )?;
                handle_binary_connection(
                    &config,
                    topology.as_ref(),
                    &runtime,
                    &decode_frame_batcher,
                    kv.as_ref(),
                    &telemetry,
                    &mut upstream,
                    downstream,
                    activation_width,
                    wire_dtype,
                    max_inflight,
                    reply_credit_limit,
                    async_prefill_forward,
                    downstream_wire_condition,
                    downstream_connect_timeout_secs,
                    native_mtp_enabled,
                    &prediction_return_sinks,
                    first_message,
                )
            })()
            .context("binary stage connection failed");
            if let Err(error) = connection_result {
                let mut attrs = lifecycle_attrs(&config);
                if let Some(peer_addr) = peer_addr {
                    attrs.insert("llama_stage.peer_addr".to_string(), json!(peer_addr));
                }
                attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                eprintln!("{error:#}");
                telemetry.emit("stage.binary_connection_error", attrs);
            }
        });
    }
    Ok(())
}
