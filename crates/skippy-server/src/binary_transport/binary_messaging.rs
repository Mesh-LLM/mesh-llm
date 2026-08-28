use std::{
    future::Future,
    io::{self, Write},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use super::stage_execution::{
    consume_optional_client_ready_hello, prepare_binary_stage_connection, take_ready_downstream,
    warm_downstream_preconnect_enabled,
};
use super::{
    direct_return::{PredictionReturnHub, PredictionReturnSinks},
    options::BinaryStageOptions,
    preconnect::DownstreamPreconnector,
};
use crate::{
    cli::ServeBinaryArgs,
    config::validate_config,
    frontend::{self, EmbeddedOpenAiArgs, iteration_scheduler::IterationScheduler},
    kv_integration::KvStageIntegration,
    runtime_state::{RuntimeLaunchOverrides, load_runtime_with_overrides},
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use skippy_protocol::binary::{WireMessageKind, read_stage_message, send_ready};

pub(in crate::binary_transport) mod async_forwarder;
mod connection;
mod control_messages;
mod message_receive;
mod prefill_recording;
pub(in crate::binary_transport) mod reply;
mod session_lifecycle;
mod session_tracker;
mod summary;
mod telemetry;

use self::connection::handle_binary_connection;

#[derive(Default)]
struct ConnectionWorkerControl {
    shutting_down: AtomicBool,
    sockets: Mutex<Vec<std::net::TcpStream>>,
}

impl ConnectionWorkerControl {
    fn track(&self, stream: &std::net::TcpStream) -> io::Result<()> {
        let tracked = stream.try_clone()?;
        let mut sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        if self.shutting_down.load(Ordering::Acquire) {
            let _ = tracked.shutdown(std::net::Shutdown::Both);
        }
        sockets.push(tracked);
        Ok(())
    }

    fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        for socket in sockets.iter() {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    }

    fn clear(&self) {
        self.sockets
            .lock()
            .expect("connection sockets lock poisoned")
            .clear();
    }
}

struct ConnectionWorker {
    control: Arc<ConnectionWorkerControl>,
    task: JoinHandle<()>,
}

#[derive(Default)]
struct ConnectionWorkers(Vec<ConnectionWorker>);

impl ConnectionWorkers {
    fn push(&mut self, worker: ConnectionWorker) {
        self.0.push(worker);
    }

    fn reap_finished(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index].task.is_finished() {
                let worker = self.0.swap_remove(index);
                if worker.task.join().is_err() {
                    bail!("binary stage connection worker panicked");
                }
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    fn shutdown(mut self) -> Result<()> {
        for worker in &self.0 {
            worker.control.shutdown();
        }
        let mut panicked = false;
        for worker in self.0.drain(..) {
            panicked |= worker.task.join().is_err();
        }
        if panicked {
            bail!("binary stage connection worker panicked during shutdown");
        }
        Ok(())
    }
}

fn finish_connection_workers(
    accept_result: Result<()>,
    connection_workers: ConnectionWorkers,
) -> Result<()> {
    let shutdown_result = connection_workers.shutdown();
    match (accept_result, shutdown_result) {
        (Ok(()), result) => result,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(shutdown_error)) => Err(error.context(format!(
            "connection worker shutdown also failed: {shutdown_error:#}"
        ))),
    }
}

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
    let mtp_source = options.resolved_mtp_source();
    let BinaryStageOptions {
        config,
        topology,
        bind_addr,
        activation_width,
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
    let runtime = load_runtime_with_overrides(
        &config,
        &RuntimeLaunchOverrides {
            mtp_source,
            ..RuntimeLaunchOverrides::default()
        },
    )?
    .context("binary stage server requires model_path")?;
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
    let iteration_scheduler = IterationScheduler::new(
        runtime.clone(),
        &config,
        max_inflight.max(1),
        telemetry.clone(),
    )
    .map_err(|error| anyhow!("create binary iteration scheduler: {error}"))?;
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
    let prediction_returns = Arc::new(PredictionReturnHub::default());
    let prediction_return_sinks = Arc::new(PredictionReturnSinks::default());
    let mut connection_workers = ConnectionWorkers::default();
    let listener = TcpListener::bind(bind_addr)?;
    listener.set_nonblocking(true)?;
    if let Some(openai_options) = openai {
        if config.stage_index != 0 || config.layer_start != 0 {
            bail!("--openai-bind-addr is only supported on stage 0");
        }
        let openai_config = config.clone();
        let openai_runtime = runtime.clone();
        let openai_iteration_scheduler = iteration_scheduler.clone();
        let openai_telemetry = telemetry.clone();
        let openai_prediction_returns = prediction_returns.clone();
        tokio::spawn(async move {
            if let Err(error) =
                frontend::serve_embedded_openai_with_scheduler(
                    EmbeddedOpenAiArgs {
                        bind_addr: openai_options.bind_addr,
                        config: openai_config,
                        runtime: openai_runtime,
                        model_id: openai_options.model_id,
                        default_max_tokens: openai_options.default_max_tokens,
                        request_defaults: frontend::EmbeddedOpenAiRequestDefaults::default(),
                        generation_concurrency: openai_options.generation_concurrency,
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
                        activation_width,
                        reply_credit_limit,
                        downstream_connect_timeout_secs,
                        downstream_wire_condition,
                        prediction_returns: Some(openai_prediction_returns),
                        telemetry: openai_telemetry,
                        hook_policy: None,
                        generation_receipt: None,
                        linear_proposal_ingress: None,
                        openai_guardrails: Some(
                            frontend::OpenAiGuardrailsConfig::disabled_for_skippy(),
                        ),
                    },
                    openai_iteration_scheduler,
                )
                .await
            {
                eprintln!("embedded OpenAI server failed: {error:#}");
            }
        });
    }
    let _downstream_preconnector = warm_downstream_preconnect_enabled()
        .then(|| {
            DownstreamPreconnector::spawn(config.clone(), warm_downstream.clone(), shutdown.clone())
        })
        .transpose()
        .context("spawn downstream preconnector")?;
    println!(
        "skippy-server listening: binary={} stage_id={} layer_range={}..{} activation_width={}",
        bind_addr, config.stage_id, config.layer_start, config.layer_end, activation_width,
    );

    let accept_result = (|| -> Result<()> {
        while !shutdown.load(Ordering::SeqCst) {
            connection_workers.reap_finished()?;
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
            let iteration_scheduler = iteration_scheduler.clone();
            let kv = kv.clone();
            let telemetry = telemetry.clone();
            let warm_downstream = warm_downstream.clone();
            let worker_shutdown = shutdown.clone();
            let prediction_returns = prediction_returns.clone();
            let prediction_return_sinks = prediction_return_sinks.clone();
            let worker_control = Arc::new(ConnectionWorkerControl::default());
            worker_control
                .track(&upstream)
                .context("track upstream binary stage connection")?;
            let task_control = worker_control.clone();
            let task = thread::spawn(move || {
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
                    let downstream = take_ready_downstream(
                        &config,
                        &warm_downstream,
                        downstream_connect_timeout_secs,
                        &worker_shutdown,
                    )?;
                    if let Some(stream) = downstream.as_ref() {
                        task_control
                            .track(stream)
                            .context("track downstream binary stage connection")?;
                    }
                    handle_binary_connection(
                        &config,
                        topology.as_ref(),
                        &iteration_scheduler,
                        kv.as_ref(),
                        &telemetry,
                        &mut upstream,
                        downstream,
                        activation_width,
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
                task_control.clear();
            });
            connection_workers.push(ConnectionWorker {
                control: worker_control,
                task,
            });
        }
        Ok(())
    })();
    shutdown.store(true, Ordering::SeqCst);
    finish_connection_workers(accept_result, connection_workers)
}

#[cfg(test)]
mod shutdown_tests {
    use super::{
        ConnectionWorker, ConnectionWorkerControl, ConnectionWorkers, finish_connection_workers,
    };
    use anyhow::anyhow;
    use std::{
        io::Read,
        net::{TcpListener, TcpStream},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    #[test]
    fn shutdown_closes_and_joins_an_active_connection_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let task = thread::spawn(move || {
            let mut byte = [0u8; 1];
            let _ = server.read(&mut byte);
            task_control.clear();
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        let (cleanup_tx, cleanup_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = cleanup_tx.send(workers.shutdown());
        });
        cleanup_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("active worker cleanup must complete within one second")
            .unwrap();
        drop(client);
    }

    #[test]
    fn accept_error_still_closes_and_joins_active_connection_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let task_finished = finished.clone();
        let task = thread::spawn(move || {
            let mut byte = [0u8; 1];
            let _ = server.read(&mut byte);
            task_control.clear();
            task_finished.store(true, Ordering::Release);
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        let (cleanup_tx, cleanup_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let result = finish_connection_workers(Err(anyhow!("accept failed")), workers);
            let _ = cleanup_tx.send(result);
        });

        let result = cleanup_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("active worker cleanup must complete within one second");
        assert!(finished.load(Ordering::Acquire));
        let error = result.expect_err("accept failure must be returned after worker cleanup");
        assert!(format!("{error:#}").contains("accept failed"));
        drop(client);
    }
}
