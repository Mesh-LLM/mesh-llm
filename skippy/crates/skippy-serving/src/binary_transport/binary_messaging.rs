mod public_frontend;
mod server_lifecycle;

pub(crate) use server_lifecycle::serve_binary_stage_with_shutdown_and_boundary_observer;
use server_lifecycle::{EmbeddedFrontendTask, wait_for_shutdown};
pub use server_lifecycle::{serve_binary_stage, serve_binary_stage_with_shutdown};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use super::stage_execution::{
    binary_message_attrs, binary_message_session_id, consume_optional_client_ready_hello,
    prepare_binary_stage_connection, take_ready_downstream, warm_downstream_preconnect_enabled,
};
use super::wire::write_stage_message_conditioned;
use super::{
    WireCondition,
    direct_return::{PredictionReturnHub, PredictionReturnSinks},
    options::BinaryStageOptions,
    preconnect::DownstreamPreconnector,
};
use crate::{
    frontend::{self, EmbeddedOpenAiArgs, iteration_scheduler::IterationScheduler},
    kv_integration::KvStageIntegration,
    runtime_state::{
        RuntimeLaunchOverrides, load_runtime_with_overrides, loaded_memory_cache_capabilities,
        loaded_model_state_kind,
    },
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use skippy_config::validate_config;
use skippy_protocol::PeerConfig;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::{
    StageWireMessage, WireMessageKind, read_stage_message_for_codec_policy, send_ready,
};
use skippy_runtime::ActivationBoundaryDesc;

pub(in crate::binary_transport) mod async_forwarder;
mod connection;
mod control_messages;
mod message_receive;
mod prefill_recording;
pub(in crate::binary_transport) mod reply;
mod session_lifecycle;
mod session_tracker;
mod stale_discard;
mod summary;
mod telemetry;

use self::connection::handle_binary_connection;
use self::session_tracker::ConnectionSessionOwnership;

/// How often a waiting connection worker rechecks the shutdown flag, matching
/// the downstream DOWNSTREAM_SHUTDOWN_POLL cadence in stage_execution.
const WORKER_SHUTDOWN_POLL: Duration = Duration::from_millis(100);

/// Darwin `recv`/`peek` error for an expired `SO_RCVTIMEO` (os error 22).
const EINVAL: i32 = 22;

/// Per-request downstream activation-forward write failure, emitted at both
/// forward sites (the sync write in `connection` and the async writer thread
/// in `async_forwarder`). Normal telemetry level (visible at `Summary` and
/// above) and timed to the write window, carrying the request/session
/// identity of the frame that failed plus the intended downstream identity.
pub(crate) const STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR: &str =
    "stage.binary_downstream_forward_error";

/// Downstream acquisition failure between reading the first upstream message
/// and the first forward. The first message's request/session identity is
/// real (the connection-level session id is not assigned yet, so the wire
/// session wins when present); the configured downstream identity is the
/// *intended* target the connection was trying to acquire, emitted exactly
/// when that acquisition failed.
pub(crate) const STAGE_BINARY_DOWNSTREAM_CONNECT_ERROR: &str =
    "stage.binary_downstream_connect_error";

/// Attributes for [`STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR`]: the caller's
/// identity map (request/session, lifecycle) plus the intended downstream
/// identity and the formatted error chain.
pub(crate) fn downstream_forward_error_attrs(
    mut identity: BTreeMap<String, serde_json::Value>,
    downstream: Option<&PeerConfig>,
    error: &str,
) -> BTreeMap<String, serde_json::Value> {
    if let Some(downstream) = downstream {
        insert_downstream_identity(&mut identity, downstream);
    }
    identity.insert("llama_stage.error".to_string(), json!(error));
    identity
}

/// Attributes for [`STAGE_BINARY_DOWNSTREAM_CONNECT_ERROR`]: the first
/// message's request/session identity plus the intended downstream identity
/// and the formatted error chain. Fallback wire identities (session `0`,
/// request `prompt-<seq>`) are *uncorrelated*: the harness must not promote
/// them to an exact request/UUID join.
pub(crate) fn downstream_connect_error_attrs(
    config: &StageConfig,
    first_message: &StageWireMessage,
    error: &str,
) -> BTreeMap<String, serde_json::Value> {
    let session_id = binary_message_session_id(0, first_message);
    let mut attrs = binary_message_attrs(config, session_id, first_message);
    if let Some(downstream) = &config.downstream {
        insert_downstream_identity(&mut attrs, downstream);
    }
    attrs.insert("llama_stage.error".to_string(), json!(error));
    attrs
}

/// Production downstream-acquisition boundary: run `acquire` and, on failure,
/// emit [`STAGE_BINARY_DOWNSTREAM_CONNECT_ERROR`] exactly once with the first
/// message's identity before propagating the error unchanged. The event
/// proves acquisition failed — including shutdown/cancellation of the
/// acquisition — not necessarily a remote refusal.
pub(crate) fn acquire_downstream_or_emit_connect_error(
    config: &StageConfig,
    first_message: &StageWireMessage,
    telemetry: &Telemetry,
    acquire: impl FnOnce() -> Result<Option<TcpStream>>,
) -> Result<Option<TcpStream>> {
    acquire().map_err(|error| {
        telemetry.emit(
            STAGE_BINARY_DOWNSTREAM_CONNECT_ERROR,
            downstream_connect_error_attrs(config, first_message, &format!("{error:#}")),
        );
        error
    })
}

/// Production sync-forward boundary: write the forwarded frame downstream
/// and, on failure, emit [`STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR`] with the
/// identity of the frame in flight, timed to the write window, before
/// propagating the error. Success emits nothing here — the caller owns the
/// success-path write span.
pub(crate) fn write_forwarded_stage_or_emit_forward_error(
    downstream: &mut TcpStream,
    message: &StageWireMessage,
    condition: WireCondition,
    identity: BTreeMap<String, serde_json::Value>,
    downstream_config: Option<&PeerConfig>,
    telemetry: &Telemetry,
    write_start_unix_nanos: u64,
) -> Result<()> {
    if let Err(error) = write_stage_message_conditioned(downstream, message, condition) {
        telemetry.emit_span(
            STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR,
            downstream_forward_error_attrs(identity, downstream_config, &format!("{error:#}")),
            write_start_unix_nanos,
            crate::telemetry::now_unix_nanos() as u64,
        );
        return Err(anyhow::Error::new(error).context("forward activation frame downstream"));
    }
    Ok(())
}

pub(crate) fn insert_downstream_identity(
    attrs: &mut BTreeMap<String, serde_json::Value>,
    downstream: &PeerConfig,
) {
    attrs.insert(
        "llama_stage.downstream_stage_id".to_string(),
        json!(downstream.stage_id),
    );
    attrs.insert(
        "llama_stage.downstream_stage_index".to_string(),
        json!(downstream.stage_index),
    );
    attrs.insert(
        "llama_stage.downstream_endpoint".to_string(),
        json!(downstream.endpoint),
    );
}

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

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Block until `stream` has readable data (or EOF), returning false when
    /// shutdown is requested instead. Peeks under a short read timeout so no
    /// message bytes are consumed and the worker never sits in an
    /// uninterruptible blocking read: `TcpStream::shutdown` on a tracked
    /// clone does not unblock an in-flight `read` on Windows (#1538).
    ///
    /// Darwin quirk: when a blocking socket has `SO_RCVTIMEO` set, an expired
    /// timeout surfaces from `peek` as `EINVAL` (os error 22), not
    /// `WouldBlock`/`TimedOut`. Treating EINVAL as fatal here killed healthy
    /// stage connections on their first idle poll (two-node split smoke:
    /// "wait for the first binary stage message: Invalid argument (os error
    /// 22)"). We therefore tolerate EINVAL only while our own read timeout is
    /// provably in effect — `timeout_armed` is captured when we set it, so the
    /// guard cannot drift from the socket state — and a timeout-less EINVAL
    /// still fails the connection loudly.
    ///
    /// The tail `set_read_timeout(None)` is best-effort for the same reason:
    /// during teardown Darwin can reject the clear with EINVAL too, and that
    /// cleanup failure must not flip an already-decided poll outcome into a
    /// connection error. For the same teardown window, if even arming the
    /// timeout fails with EINVAL while shutdown is in progress, report
    /// `Ok(false)` — the worker is winding down either way.
    fn wait_for_readable(&self, stream: &TcpStream) -> io::Result<bool> {
        let timeout_armed = match stream.set_read_timeout(Some(WORKER_SHUTDOWN_POLL)) {
            Ok(()) => true,
            // Darwin rejects SO_RCVTIMEO operations with EINVAL (os error 22)
            // on a socket that has been shut down for teardown. If the worker
            // is shutting down, that is the expected outcome, not an error.
            Err(_) if self.is_shutting_down() => return Ok(false),
            Err(error) => return Err(error),
        };
        let mut probe = [0u8; 1];
        let ready = loop {
            if self.is_shutting_down() {
                break false;
            }
            let probe_started = Instant::now();
            match stream.peek(&mut probe) {
                Ok(_) => break true,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Interrupted
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                    ) => {}
                Err(error)
                    if timeout_armed
                        && error.raw_os_error() == Some(EINVAL)
                        && probe_started.elapsed() >= WORKER_SHUTDOWN_POLL => {}
                Err(error) => {
                    let _ = stream.set_read_timeout(None);
                    return Err(error);
                }
            }
        };
        // Best-effort cleanup: on Darwin, clearing SO_RCVTIMEO during
        // connection teardown can itself fail with EINVAL once the peer (or
        // our own shutdown path) has closed the socket. The poll outcome is
        // already established at this point, so a failing timeout clear must
        // not flip a healthy idle shutdown into "binary stage connection
        // failed: Invalid argument (os error 22)".
        let _ = stream.set_read_timeout(None);
        Ok(ready)
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

    fn reap_finished(&mut self) -> usize {
        let mut panicked = 0;
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index].task.is_finished() {
                let worker = self.0.swap_remove(index);
                if worker.task.join().is_err() {
                    // A panicked worker only fails its own connection; the
                    // accept loop must keep serving other clients.
                    panicked += 1;
                }
            } else {
                index += 1;
            }
        }
        panicked
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

fn run_binary_stage(
    options: BinaryStageOptions,
    shutdown: Arc<AtomicBool>,
    boundary_observer: impl FnOnce(Option<ActivationBoundaryDesc>, Option<ActivationBoundaryDesc>),
) -> Result<Option<EmbeddedFrontendTask>> {
    options.tuning.validate()?;
    let mtp_source = options.resolved_mtp_source();
    let BinaryStageOptions {
        tuning,
        config,
        topology,
        bind_addr,
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        telemetry_level,
        max_inflight,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        native_mtp_enabled,
        last_stage_decode_batch,
        continuous_batching,
        openai,
        l3_manager,
        compute_meter,
    } = options;
    // With an OpenAI frontend, a stop request closes HTTP admission first.
    // Keep worker connections and prediction returns alive until Axum drains.
    let shutdown_requested = shutdown;
    let shutdown = if openai.is_some() {
        Arc::new(AtomicBool::new(false))
    } else {
        shutdown_requested.clone()
    };
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
            n_threads: tuning.n_threads,
            n_threads_batch: tuning.n_threads_batch,
        },
        None,
    )?
    .context("binary stage server requires model_path")?;
    let (input_boundary, output_boundary) = {
        let runtime = runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?;
        (
            runtime.input_activation_boundary(),
            runtime.output_activation_boundary(),
        )
    };
    let input_activation_width =
        activation_width_from_graph("input", input_boundary, config.layer_start > 0)?;
    let output_activation_width =
        activation_width_from_graph("output", output_boundary, config.downstream.is_some())?;
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
    if let Some(meter) = compute_meter {
        runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?
            .set_compute_meter(meter);
    }
    let iteration_scheduler = IterationScheduler::new(
        runtime.clone(),
        &config,
        max_inflight.max(1),
        continuous_batching,
        // Grouping is a property of the dispatcher, and the embedded frontend is
        // the only thing that dispatches, so the value rides its options.
        openai
            .as_ref()
            .and_then(|options| options.pipeline_decode_groups),
        telemetry.clone(),
    )
    .map_err(|error| anyhow!("create binary iteration scheduler: {error}"))?;
    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &config,
        loaded_model_state_kind(Some(&runtime)),
        loaded_memory_cache_capabilities(Some(&runtime)),
        l3_manager.clone(),
        None,
    )?
    .map(Arc::new);
    let prediction_returns = Arc::new(PredictionReturnHub::default());
    let prediction_return_sinks = Arc::new(PredictionReturnSinks::default());
    let session_ownership = Arc::new(ConnectionSessionOwnership::default());
    let mut connection_workers = ConnectionWorkers::default();
    let listener = TcpListener::bind(bind_addr)?;
    listener.set_nonblocking(true)?;
    boundary_observer(input_boundary, output_boundary);
    let frontend_task = public_frontend::start(public_frontend::PublicFrontendLaunch {
        config: config.clone(),
        runtime: runtime.clone(),
        iteration_scheduler: iteration_scheduler.clone(),
        telemetry: telemetry.clone(),
        prediction_returns: prediction_returns.clone(),
        shutdown_requested: shutdown_requested.clone(),
        openai,
        continuous_batching,
        native_mtp_enabled,
        output_activation_width,
        reply_credit_limit,
        downstream_connect_timeout_secs,
        downstream_wire_condition,
        l3_manager: l3_manager.clone(),
        tuning,
    })?;
    let _downstream_preconnector = warm_downstream_preconnect_enabled()
        .then(|| {
            DownstreamPreconnector::spawn(config.clone(), warm_downstream.clone(), shutdown.clone())
        })
        .transpose()
        .context("spawn downstream preconnector")?;
    tracing::info!(
        "skippy-serving listening: binary={} stage_id={} layer_range={}..{} input_activation_width={} output_activation_width={}",
        bind_addr,
        config.stage_id,
        config.layer_start,
        config.layer_end,
        input_activation_width,
        output_activation_width,
    );

    let accept_result = (|| -> Result<()> {
        while !shutdown.load(Ordering::SeqCst) {
            if frontend_task
                .as_ref()
                .is_some_and(EmbeddedFrontendTask::is_finished)
            {
                break;
            }
            let panicked_workers = connection_workers.reap_finished();
            if panicked_workers > 0 {
                telemetry.emit(
                    "stage.connection_worker_panic",
                    BTreeMap::from([
                        ("llama_stage.failure_contained".to_string(), json!(true)),
                        (
                            "llama_stage.panicked_workers".to_string(),
                            json!(panicked_workers),
                        ),
                    ]),
                );
            }
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
            tracing::debug!(
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
            let session_ownership = session_ownership.clone();
            let worker_control = Arc::new(ConnectionWorkerControl::default());
            worker_control
                .track(&upstream)
                .context("track upstream binary stage connection")?;
            let task_control = worker_control.clone();
            let task = thread::spawn(move || {
                let connection_result = (|| -> Result<()> {
                    tracing::debug!(
                        "binary sending ready: stage_id={} peer={peer_addr:?}",
                        config.stage_id
                    );
                    consume_optional_client_ready_hello(&mut upstream)
                        .context("consume optional client ready hello")?;
                    send_ready(&mut upstream).context("failed to send binary ready")?;
                    upstream.flush().ok();
                    tracing::debug!(
                        "binary sent ready: stage_id={} peer={peer_addr:?}",
                        config.stage_id
                    );
                    if !task_control
                        .wait_for_readable(&upstream)
                        .context("wait for the first binary stage message")?
                    {
                        return Ok(());
                    }
                    let first_message = match read_stage_message_for_codec_policy(
                        &mut upstream,
                        input_activation_width,
                        config.activation_codec,
                        config.activation_codec_policy,
                    ) {
                        Ok(message) => message,
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                            return Ok(());
                        }
                        Err(error) => return Err(error.into()),
                    };
                    if first_message.kind == WireMessageKind::PredictionReturnOpen {
                        if config.stage_index == 0 {
                            return prediction_returns
                                .handle_return_connection(first_message, upstream);
                        }
                        return prediction_return_sinks.insert_opened_sink(first_message, upstream);
                    }
                    // Dedicated connect-failure event at the fallible
                    // boundary (see acquire_downstream_or_emit_connect_error):
                    // the first message's request/session ids are real, the
                    // configured downstream is the intended target. The outer
                    // connection error below stays generic — it also catches
                    // upstream/protocol/processing failures, so downstream
                    // fields there would imply causality this event does not.
                    let downstream = acquire_downstream_or_emit_connect_error(
                        &config,
                        &first_message,
                        &telemetry,
                        || {
                            take_ready_downstream(
                                &config,
                                &warm_downstream,
                                downstream_connect_timeout_secs,
                                &worker_shutdown,
                            )
                        },
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
                        input_activation_width,
                        output_activation_width,
                        max_inflight,
                        reply_credit_limit,
                        async_prefill_forward,
                        downstream_wire_condition,
                        downstream_connect_timeout_secs,
                        native_mtp_enabled,
                        last_stage_decode_batch,
                        &prediction_return_sinks,
                        session_ownership,
                        task_control.clone(),
                        first_message,
                    )
                })()
                .context("binary stage connection failed");
                if let Err(error) = connection_result {
                    let mut attrs = lifecycle_attrs(&config);
                    if let Some(peer_addr) = peer_addr {
                        attrs.insert("llama_stage.peer_addr".to_string(), json!(peer_addr));
                    }
                    let message = format!("{error:#}");
                    attrs.insert("llama_stage.error".to_string(), json!(message));
                    tracing::warn!("{error:#}");
                    let _ = skippy_events::diagnostics::emit(
                        skippy_events::diagnostics::ServingDiagnostic::Warning {
                            message,
                            context: Some(format!(
                                "run_id={} stage_id={} peer={peer_addr:?}",
                                config.run_id, config.stage_id
                            )),
                        },
                    );
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
    finish_connection_workers(accept_result, connection_workers)?;
    Ok(frontend_task)
}

fn activation_width_from_graph(
    edge: &str,
    descriptor: Option<ActivationBoundaryDesc>,
    required: bool,
) -> Result<i32> {
    let Some(descriptor) = descriptor else {
        if required {
            bail!("stage graph did not expose its {edge} activation boundary");
        }
        return Ok(0);
    };
    descriptor.raw_f32_width(edge)
}

#[cfg(test)]
mod shutdown_tests {
    use super::{
        ConnectionWorker, ConnectionWorkerControl, ConnectionWorkers, activation_width_from_graph,
        finish_connection_workers,
    };
    use crate::test_activation::boundary_f32;
    use anyhow::anyhow;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn graph_boundary_is_the_only_activation_width_authority() {
        assert_eq!(
            activation_width_from_graph("output", Some(boundary_f32(1024)), true)
                .expect("valid graph boundary"),
            1024
        );
    }

    #[test]
    fn required_graph_boundary_cannot_be_omitted() {
        let error = activation_width_from_graph("input", None, true)
            .expect_err("required graph boundary must be present");
        assert!(error.to_string().contains("did not expose"));
    }

    #[test]
    fn absent_unused_graph_boundary_has_no_wire_width() {
        assert_eq!(
            activation_width_from_graph("input", None, false)
                .expect("unused edge may omit a boundary"),
            0
        );
    }

    #[test]
    fn unsupported_graph_boundary_fails_closed() {
        let mut boundary = boundary_f32(1024);
        boundary.parts[0].ggml_type = 1;
        let error = activation_width_from_graph("output", Some(boundary), true)
            .expect_err("non-F32 graph boundary must not use the F32 codec");
        assert!(error.to_string().contains("not token-indexed F32"));

        let mut boundary = boundary_f32(1024);
        boundary.parts[0].token_axis = -1;
        let error = activation_width_from_graph("output", Some(boundary), true)
            .expect_err("non-token-indexed graph boundary must fail");
        assert!(error.to_string().contains("not token-indexed F32"));

        let mut boundary = boundary_f32(1024);
        boundary.part_count = 0;
        let error = activation_width_from_graph("output", Some(boundary), true)
            .expect_err("empty graph boundary must fail");
        assert!(error.to_string().contains("part count is invalid"));
    }

    #[test]
    fn idle_poll_survives_a_full_read_timeout_expiry() {
        // Regression: on Darwin a SO_RCVTIMEO expiry inside `peek` surfaces
        // as EINVAL (os error 22). The worker must keep polling on an idle
        // connection instead of failing it, and must still observe data
        // arriving after several expired polls.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let task = thread::spawn(move || {
            let started = Instant::now();
            let mut ready = false;
            while !ready {
                match task_control.wait_for_readable(&server) {
                    Ok(became_ready) => ready = became_ready,
                    Err(error) => panic!("idle poll must not fail: {error}"),
                }
            }
            assert!(ready, "connection must become readable after data arrives");
            assert!(
                started.elapsed() >= Duration::from_millis(200),
                "at least two timeout polls must have expired before data arrived"
            );
            let mut byte = [0u8; 1];
            server.read_exact(&mut byte).unwrap();
            assert_eq!(byte[0], b'x');
            task_control.clear();
        });
        thread::sleep(Duration::from_millis(250));
        client.write_all(b"x").unwrap();
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = done_tx.send(task.join().is_ok());
        });
        assert!(
            done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("worker must finish")
        );
    }

    #[test]
    fn shutdown_poll_on_a_shut_down_socket_stays_ok() {
        // Regression: during teardown on Darwin, the tail
        // `set_read_timeout(None)` fails with EINVAL once the socket has been
        // shut down ("Invalid argument (os error 22)" at shutdown). The
        // cleanup is best-effort — the poll's already-decided outcome must be
        // reported as Ok instead of surfacing a connection error.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        control.shutdown();
        let result = control.wait_for_readable(&server);
        assert!(
            result.is_ok(),
            "poll after shutdown must not fail: {:?}",
            result.err()
        );
    }

    #[test]
    fn shutdown_closes_and_joins_an_active_connection_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let task = thread::spawn(move || {
            if let Ok(true) = task_control.wait_for_readable(&server) {
                let mut byte = [0u8; 1];
                let _ = server.read(&mut byte);
            }
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
            if let Ok(true) = task_control.wait_for_readable(&server) {
                let mut byte = [0u8; 1];
                let _ = server.read(&mut byte);
            }
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

    #[test]
    fn reap_finished_contains_a_panicked_worker_and_keeps_reaping() {
        let control = Arc::new(ConnectionWorkerControl::default());
        let task = thread::spawn(|| panic!("connection worker exploded"));
        let deadline = Instant::now() + Duration::from_secs(1);
        while !task.is_finished() {
            assert!(
                Instant::now() < deadline,
                "panicking worker thread must finish within one second"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        assert_eq!(
            workers.reap_finished(),
            1,
            "the panicked worker must be counted, not turned into an error"
        );
        assert!(
            workers.0.is_empty(),
            "the panicked worker must still be reaped"
        );
        assert_eq!(workers.reap_finished(), 0);
    }

    #[test]
    fn reap_finished_leaves_running_workers_alone() {
        let (stop_tx, stop_rx) = mpsc::sync_channel::<()>(1);
        let control = Arc::new(ConnectionWorkerControl::default());
        let task = thread::spawn(move || {
            let _ = stop_rx.recv();
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        assert_eq!(workers.reap_finished(), 0);
        assert_eq!(workers.0.len(), 1, "a running worker must not be reaped");
        stop_tx.send(()).unwrap();
        workers.shutdown().unwrap();
    }
}

#[cfg(test)]
mod downstream_error_telemetry_tests {
    use super::super::stage_execution::prefix_cache_test_config;
    use super::*;
    use crate::telemetry::{Telemetry, TelemetryLevel};
    use skippy_protocol::binary::{StageStateHeader, WireMessageKind};

    fn first_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::VerifyWindow,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::VerifyWindow),
            request_id: 77,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn downstream_peer() -> PeerConfig {
        PeerConfig {
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            endpoint: "10.0.0.3:50052".to_string(),
        }
    }

    fn config_with_downstream() -> StageConfig {
        let mut config = prefix_cache_test_config();
        config.downstream = Some(downstream_peer());
        config
    }

    /// The connect-failure event's identity comes from the real first message
    /// (request/session/epoch), carries the intended downstream identity and
    /// the topology/run lifecycle, and reports the error chain.
    #[test]
    fn connect_error_attrs_carry_first_message_identity_and_intended_downstream() {
        let config = config_with_downstream();

        let attrs = downstream_connect_error_attrs(&config, &first_message(), "connect refused");

        assert_eq!(attrs["skippy.request_id"], json!("77"));
        assert_eq!(attrs["skippy.session_id"], json!("9"));
        assert_eq!(attrs["llama_stage.downstream_stage_id"], json!("stage-2"));
        assert_eq!(attrs["llama_stage.downstream_stage_index"], json!(2));
        assert_eq!(
            attrs["llama_stage.downstream_endpoint"],
            json!("10.0.0.3:50052")
        );
        assert_eq!(attrs["llama_stage.error"], json!("connect refused"));
        assert_eq!(attrs["skippy.run_id"], json!("run"));
        assert_eq!(attrs["skippy.topology_id"], json!("topology"));
        assert_eq!(attrs["skippy.stage_id"], json!("stage-0"));
    }

    /// Without a configured downstream there is no downstream identity to
    /// report: the keys must stay absent rather than carry placeholder values.
    #[test]
    fn connect_error_attrs_omit_downstream_identity_without_a_downstream() {
        // The shared fixture may carry a downstream; clear it explicitly so
        // this test pins the no-downstream attribute contract.
        let mut config = prefix_cache_test_config();
        config.downstream = None;

        let attrs = downstream_connect_error_attrs(&config, &first_message(), "connect refused");

        assert!(!attrs.contains_key("llama_stage.downstream_stage_id"));
        assert!(!attrs.contains_key("llama_stage.downstream_stage_index"));
        assert!(!attrs.contains_key("llama_stage.downstream_endpoint"));
        assert_eq!(attrs["skippy.request_id"], json!("77"));
        assert_eq!(attrs["llama_stage.error"], json!("connect refused"));
    }

    /// Fallback wire identities (missing request/session on the wire) must
    /// stay uncorrelated placeholders: a harness must not be able to mistake
    /// them for an exact request join.
    #[test]
    fn fallback_wire_ids_stay_uncorrelated_placeholders() {
        let config = config_with_downstream();
        let mut frame = first_message();
        frame.request_id = 0;
        frame.session_id = 0;

        let attrs = downstream_connect_error_attrs(&config, &frame, "connect refused");

        assert_eq!(attrs["skippy.session_id"], json!("0"));
        let request = attrs["skippy.request_id"].as_str().unwrap().to_string();
        assert!(
            request.starts_with("prompt-"),
            "fallback request id must keep its prompt-* shape: {request}"
        );
        assert!(
            request.parse::<u64>().is_err(),
            "fallback request id must not look like an exact numeric request id: {request}"
        );
    }

    /// The production acquisition boundary emits exactly one connect event on
    /// failure, with the first message's identity, and propagates the error
    /// unchanged.
    #[test]
    fn connect_boundary_emits_once_on_acquisition_failure_and_propagates() {
        let config = config_with_downstream();
        let frame = first_message();
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        let result = acquire_downstream_or_emit_connect_error(&config, &frame, &telemetry, || {
            Err(anyhow!("downstream connect refused"))
        });

        let error = result.unwrap_err();
        assert!(error.to_string().contains("downstream connect refused"));

        let event = rx
            .try_recv()
            .expect("exactly one connect event on acquisition failure");
        assert_eq!(event.event, "stage.binary_downstream_connect_error");
        assert_eq!(event.attributes["skippy.request_id"], json!("77"));
        assert_eq!(event.attributes["skippy.session_id"], json!("9"));
        assert_eq!(
            event.attributes["llama_stage.downstream_stage_id"],
            json!("stage-2")
        );
        assert_eq!(
            event.attributes["llama_stage.error"],
            json!("downstream connect refused")
        );
        assert!(rx.try_recv().is_err(), "no additional events on failure");
    }

    /// Success through the same boundary emits nothing and returns the
    /// acquisition result unchanged.
    #[test]
    fn connect_boundary_emits_nothing_on_acquisition_success() {
        let config = config_with_downstream();
        let frame = first_message();
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        let result =
            acquire_downstream_or_emit_connect_error(&config, &frame, &telemetry, || Ok(None));

        assert!(result.unwrap().is_none());
        assert!(rx.try_recv().is_err(), "success must not emit");
    }

    /// The production sync-forward boundary emits the failure event with the
    /// frame's identity when the write fails against a reset peer.
    #[test]
    fn sync_forward_boundary_emits_error_event_on_write_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let (server, _) = listener.accept().unwrap();
        socket2::SockRef::from(&server)
            .set_linger(Some(Duration::from_secs(0)))
            .unwrap();
        drop(server);
        use std::io::Write as _;
        let mut reset_observed = false;
        for _ in 0..200 {
            if client.write_all(&[0]).is_err() {
                reset_observed = true;
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(reset_observed, "precondition: peer reset must be observed");

        let config = config_with_downstream();
        let frame = first_message();
        let identity = binary_message_attrs(&config, binary_message_session_id(0, &frame), &frame);
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        let result = write_forwarded_stage_or_emit_forward_error(
            &mut client,
            &frame,
            WireCondition::new(0.0, None).unwrap(),
            identity,
            Some(&downstream_peer()),
            &telemetry,
            crate::telemetry::now_unix_nanos() as u64,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("forward activation frame downstream")
        );

        let event = rx
            .try_recv()
            .expect("exactly one forward error event on write failure");
        assert_eq!(event.event, "stage.binary_downstream_forward_error");
        assert_eq!(event.attributes["skippy.request_id"], json!("77"));
        assert_eq!(event.attributes["skippy.session_id"], json!("9"));
        assert_eq!(
            event.attributes["llama_stage.downstream_stage_id"],
            json!("stage-2")
        );
        assert!(
            !event.attributes["llama_stage.error"]
                .as_str()
                .unwrap()
                .is_empty(),
            "the error attribute carries the raw write failure chain; the \
             propagated error adds the boundary context"
        );
        assert!(rx.try_recv().is_err(), "no additional events on failure");
    }

    /// Success through the same boundary emits nothing and delivers the frame
    /// to the peer.
    #[test]
    fn sync_forward_boundary_emits_nothing_on_success() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let config = config_with_downstream();
        let frame = first_message();
        let identity = binary_message_attrs(&config, binary_message_session_id(0, &frame), &frame);
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        let result = write_forwarded_stage_or_emit_forward_error(
            &mut client,
            &frame,
            WireCondition::new(0.0, None).unwrap(),
            identity,
            Some(&downstream_peer()),
            &telemetry,
            crate::telemetry::now_unix_nanos() as u64,
        );

        assert!(result.is_ok());
        let delivered = skippy_protocol::binary::read_stage_message(&mut server, 4).unwrap();
        assert_eq!(delivered.kind, WireMessageKind::VerifyWindow);
        assert!(rx.try_recv().is_err(), "success must not emit");
    }

    /// The shared forward-failure attr builder preserves the caller's
    /// identity map verbatim and adds the intended downstream plus error.
    #[test]
    fn forward_error_attrs_extend_the_caller_identity_map() {
        let config = config_with_downstream();
        let frame = first_message();
        let identity = binary_message_attrs(&config, binary_message_session_id(0, &frame), &frame);

        let attrs =
            downstream_forward_error_attrs(identity, Some(&downstream_peer()), "write reset");

        assert_eq!(attrs["skippy.request_id"], json!("77"));
        assert_eq!(attrs["skippy.session_id"], json!("9"));
        assert_eq!(attrs["llama_stage.downstream_stage_id"], json!("stage-2"));
        assert_eq!(
            attrs["llama_stage.downstream_endpoint"],
            json!("10.0.0.3:50052")
        );
        assert_eq!(attrs["llama_stage.error"], json!("write reset"));
        assert!(attrs.contains_key("skippy.kv_layer_count"));
    }

    /// Emission at normal level: the connect event is observable at
    /// `Summary` telemetry and lands exactly once with its attributes.
    #[test]
    fn connect_error_event_emits_once_at_summary_level() {
        let config = config_with_downstream();
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        telemetry.emit(
            STAGE_BINARY_DOWNSTREAM_CONNECT_ERROR,
            downstream_connect_error_attrs(&config, &first_message(), "connect refused"),
        );

        let event = rx.try_recv().expect("connect event captured");
        assert_eq!(event.event, "stage.binary_downstream_connect_error");
        assert_eq!(
            event.attributes["llama_stage.downstream_stage_id"],
            json!("stage-2")
        );
        assert!(rx.try_recv().is_err());
    }
}
