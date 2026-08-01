use std::{
    net::SocketAddr,
    sync::{Arc, Mutex, TryLockError},
};

use anyhow::{Context, Result};
use openai_frontend::OpenAiBackend;
use skippy_protocol::{StageConfig, StageTopology};
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{
    binary_transport::{BinaryStageOptions, serve_binary_stage_with_shutdown},
    config::validate_config,
    frontend::{EmbeddedOpenAiArgs, serve_embedded_openai_with_shutdown},
    http::{StageHttpOptions, serve_stage_http_with_shutdown},
    runtime_state::{
        RuntimeLaunchOverrides, RuntimeSessionStats, RuntimeState, load_runtime_with_overrides,
        load_runtime_with_overrides_and_open_events,
    },
    telemetry::{Telemetry, TelemetryLevel, TelemetryStats, lifecycle_attrs, now_unix_nanos},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub struct EmbeddedRuntimeStatus {
    pub state: EmbeddedState,
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub runtime_loaded: bool,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
    pub sessions: RuntimeSessionStats,
    pub telemetry: TelemetryStats,
}

#[derive(Clone, Debug)]
pub struct EmbeddedServerStatus {
    pub name: &'static str,
    pub bind_addr: SocketAddr,
    pub state: EmbeddedState,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
}

#[derive(Clone)]
pub struct EmbeddedRuntimeOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
}

pub struct SkippyRuntimeHandle {
    config: Arc<StageConfig>,
    topology: Option<Arc<StageTopology>>,
    runtime: Arc<Mutex<RuntimeState>>,
    telemetry: Telemetry,
    status: Arc<Mutex<RuntimeHandleState>>,
    /// Last session stats read successfully out of [`Self::runtime`].
    ///
    /// `RuntimeState` is held for the entire duration of a decode loop, so a
    /// status read that locks it blocks for as long as the turn runs (tens of
    /// seconds on a large model with a long prompt). Status is observability:
    /// it must never queue behind inference. Reads take the runtime lock
    /// opportunistically and refresh this cache; when inference holds it, they
    /// serve the last published value instead of blocking.
    last_session_stats: Arc<Mutex<RuntimeSessionStats>>,
}

#[derive(Debug)]
struct RuntimeHandleState {
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

impl SkippyRuntimeHandle {
    pub fn load(options: EmbeddedRuntimeOptions) -> Result<Self> {
        validate_config(&options.config, options.topology.as_ref())?;
        let telemetry = Telemetry::new(
            options.metrics_otlp_grpc,
            options.telemetry_queue_capacity,
            options.config.clone(),
            options.telemetry_level,
        );
        telemetry.emit(
            "stage.embedded_runtime_load_start",
            lifecycle_attrs(&options.config),
        );
        let runtime = load_runtime_with_overrides(
            &options.config,
            &RuntimeLaunchOverrides {
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
            },
        )?
        .with_context(|| format!("stage {} requires model_path", options.config.stage_id))?;
        telemetry.emit(
            "stage.embedded_runtime_ready",
            lifecycle_attrs(&options.config),
        );
        // Prime the stats snapshot while the runtime is still unshared and
        // uncontended, so a read that loses the race to the very first
        // generation reports real lanes instead of zeros.
        let initial_session_stats = runtime
            .lock()
            .expect("runtime lock poisoned")
            .session_stats();
        Ok(Self {
            config: Arc::new(options.config),
            topology: options.topology.map(Arc::new),
            runtime,
            telemetry,
            status: Arc::new(Mutex::new(RuntimeHandleState {
                state: EmbeddedState::Ready,
                started_at_unix_nanos: now_unix_nanos(),
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            last_session_stats: Arc::new(Mutex::new(initial_session_stats)),
        })
    }

    pub fn load_with_open_events(
        options: EmbeddedRuntimeOptions,
        mut model_open_event_reporter: Option<Box<dyn FnMut(skippy_runtime::RuntimeEvent) + Send>>,
    ) -> Result<Self> {
        validate_config(&options.config, options.topology.as_ref())?;
        let telemetry = Telemetry::new(
            options.metrics_otlp_grpc,
            options.telemetry_queue_capacity,
            options.config.clone(),
            options.telemetry_level,
        );
        telemetry.emit(
            "stage.embedded_runtime_load_start",
            lifecycle_attrs(&options.config),
        );
        let runtime = load_runtime_with_overrides_and_open_events(
            &options.config,
            &RuntimeLaunchOverrides {
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
            },
            model_open_event_reporter.as_mut().map(|reporter| {
                reporter.as_mut() as &mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)
            }),
        )?
        .with_context(|| format!("stage {} requires model_path", options.config.stage_id))?;
        telemetry.emit(
            "stage.embedded_runtime_ready",
            lifecycle_attrs(&options.config),
        );
        // Prime the stats snapshot while the runtime is still unshared and
        // uncontended, so a read that loses the race to the very first
        // generation reports real lanes instead of zeros.
        let initial_session_stats = runtime
            .lock()
            .expect("runtime lock poisoned")
            .session_stats();
        Ok(Self {
            config: Arc::new(options.config),
            topology: options.topology.map(Arc::new),
            runtime,
            telemetry,
            status: Arc::new(Mutex::new(RuntimeHandleState {
                state: EmbeddedState::Ready,
                started_at_unix_nanos: now_unix_nanos(),
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            last_session_stats: Arc::new(Mutex::new(initial_session_stats)),
        })
    }

    pub fn config(&self) -> &StageConfig {
        &self.config
    }

    pub fn topology(&self) -> Option<&StageTopology> {
        self.topology.as_deref()
    }

    pub fn runtime(&self) -> Arc<Mutex<RuntimeState>> {
        self.runtime.clone()
    }

    pub fn telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }

    /// Session stats without ever waiting on the inference lock.
    ///
    /// `RuntimeState` is held across a whole decode loop, so `lock()` here
    /// would make every status/readiness/dashboard read queue behind the
    /// in-flight turn. Take the lock only if it is free — refreshing the cache
    /// when we get it — and otherwise serve the last published snapshot. Stats
    /// are advisory, so a value that is one turn stale is strictly better than
    /// a caller blocked for the length of a generation.
    fn session_stats_non_blocking(&self) -> RuntimeSessionStats {
        read_without_blocking(&self.runtime, &self.last_session_stats, |runtime| {
            runtime.session_stats()
        })
    }

    pub fn status(&self) -> EmbeddedRuntimeStatus {
        let handle = self.status.lock().expect("runtime status lock poisoned");
        let sessions = self.session_stats_non_blocking();
        EmbeddedRuntimeStatus {
            state: handle.state,
            run_id: self.config.run_id.clone(),
            topology_id: self.config.topology_id.clone(),
            model_id: self.config.model_id.clone(),
            stage_id: self.config.stage_id.clone(),
            stage_index: self.config.stage_index,
            layer_start: self.config.layer_start,
            layer_end: self.config.layer_end,
            runtime_loaded: matches!(handle.state, EmbeddedState::Ready | EmbeddedState::Stopping),
            started_at_unix_nanos: handle.started_at_unix_nanos,
            stopped_at_unix_nanos: handle.stopped_at_unix_nanos,
            last_error: handle.last_error.clone(),
            sessions,
            telemetry: self.telemetry.stats(),
        }
    }

    pub fn shutdown(&self) {
        let mut status = self.status.lock().expect("runtime status lock poisoned");
        if status.state == EmbeddedState::Stopped {
            return;
        }
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
        self.telemetry.emit(
            "stage.embedded_runtime_stopped",
            lifecycle_attrs(&self.config),
        );
    }
}

impl Drop for SkippyRuntimeHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct EmbeddedServerHandle {
    status: Arc<Mutex<ServerHandleState>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<()>>>,
}

#[derive(Debug)]
struct ServerHandleState {
    name: &'static str,
    bind_addr: SocketAddr,
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

impl EmbeddedServerHandle {
    pub fn status(&self) -> EmbeddedServerStatus {
        let status = self.status.lock().expect("server status lock poisoned");
        EmbeddedServerStatus {
            name: status.name,
            bind_addr: status.bind_addr,
            state: status.state,
            started_at_unix_nanos: status.started_at_unix_nanos,
            stopped_at_unix_nanos: status.stopped_at_unix_nanos,
            last_error: status.last_error.clone(),
        }
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let task = self.task.take().expect("server task already taken");
        task.await?
    }

    pub fn abort(mut self) {
        self.shutdown.take();
        if let Some(task) = self.task.take() {
            task.abort();
        }
        let mut status = self.status.lock().expect("server status lock poisoned");
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for EmbeddedServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

pub fn start_stage_http(options: StageHttpOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    spawn_async_server("stage-http", bind_addr, |shutdown| async move {
        serve_stage_http_with_shutdown(options, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_embedded_openai(args: EmbeddedOpenAiArgs) -> EmbeddedServerHandle {
    let bind_addr = args.bind_addr;
    spawn_async_server("openai", bind_addr, |shutdown| async move {
        serve_embedded_openai_with_shutdown(args, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_openai_backend(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
) -> EmbeddedServerHandle {
    spawn_async_server("openai-backend", bind_addr, move |shutdown| async move {
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        axum::serve(listener, openai_frontend::router_for(backend))
            .with_graceful_shutdown(async move {
                let _ = shutdown.await;
            })
            .await?;
        Ok(())
    })
}

pub fn start_binary_stage(options: BinaryStageOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    let status = Arc::new(Mutex::new(ServerHandleState {
        name: "binary-stage",
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let runtime = tokio::runtime::Handle::current();
    let task = tokio::task::spawn_blocking(move || {
        {
            let mut status = task_status.lock().expect("server status lock poisoned");
            status.state = EmbeddedState::Ready;
        }
        let result = runtime.block_on(serve_binary_stage_with_shutdown(options, async move {
            let _ = shutdown_rx.await;
        }));
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn spawn_async_server<F, Fut>(
    name: &'static str,
    bind_addr: SocketAddr,
    serve: F,
) -> EmbeddedServerHandle
where
    F: FnOnce(oneshot::Receiver<()>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    let status = Arc::new(Mutex::new(ServerHandleState {
        name,
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let task = tokio::spawn(async move {
        {
            let mut status = task_status.lock().expect("server status lock poisoned");
            status.state = EmbeddedState::Ready;
        }
        let result = serve(shutdown_rx).await;
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn finish_server_status(status: &Arc<Mutex<ServerHandleState>>, result: &Result<()>) {
    let mut status = status.lock().expect("server status lock poisoned");
    status.stopped_at_unix_nanos = Some(now_unix_nanos());
    match result {
        Ok(()) => {
            status.state = EmbeddedState::Stopped;
        }
        Err(error) => {
            status.state = EmbeddedState::Failed;
            status.last_error = Some(error.to_string());
        }
    }
}

/// Read a value derived from `source` without waiting on its lock.
///
/// `source` here is the inference runtime, whose mutex is held for the whole
/// duration of a decode loop. Blocking on it from an observability path makes
/// status/readiness reads queue behind the in-flight turn — and because those
/// reads happen on async executor threads, a blocking acquire also parks a
/// worker for the length of a generation. Take the lock only when it is free,
/// refreshing `cache`; otherwise serve the last value we published.
///
/// Staleness is deliberately unbounded: under sustained load every
/// opportunistic read can lose the race, so the snapshot may lag by many
/// turns. That is acceptable only because these stats are observability —
/// nothing admits, routes, evicts or limits on them (`lane_count` reaching
/// mesh gossip comes from `StageConfig`, not from here). Do not reuse this
/// helper for a value that gates a decision.
///
/// A poisoned `source` still panics, exactly as the previous blocking
/// `lock().expect(..)` did: a poisoned inference runtime means generation
/// panicked, and reporting healthy cached stats over the top of that would
/// hide a real failure.
fn read_without_blocking<S, V>(source: &Mutex<S>, cache: &Mutex<V>, read: impl FnOnce(&S) -> V) -> V
where
    V: Clone,
{
    match source.try_lock() {
        Ok(guard) => {
            let value = read(&guard);
            drop(guard);
            *cache.lock().expect("stats cache lock poisoned") = value.clone();
            value
        }
        // Inference holds the runtime: serve the last published snapshot
        // instead of blocking for the rest of the turn.
        Err(TryLockError::WouldBlock) => cache.lock().expect("stats cache lock poisoned").clone(),
        Err(TryLockError::Poisoned(error)) => panic!("runtime lock poisoned: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_without_blocking_refreshes_cache_when_lock_is_free() {
        let source = Mutex::new(7u32);
        let cache = Mutex::new(0u32);

        assert_eq!(read_without_blocking(&source, &cache, |v| *v), 7);
        assert_eq!(*cache.lock().unwrap(), 7, "a free lock must refresh cache");
    }

    #[test]
    fn read_without_blocking_serves_cache_instead_of_waiting_on_inference() {
        // The regression: RuntimeState is held for an entire decode loop. A
        // status read must return the last published value immediately rather
        // than block for the length of the turn.
        let source = Mutex::new(7u32);
        let cache = Mutex::new(0u32);

        // Prime the cache while the runtime is idle.
        assert_eq!(read_without_blocking(&source, &cache, |v| *v), 7);

        // Now inference holds the runtime lock for the whole turn.
        let held = source.lock().expect("lock runtime");

        let observed = read_without_blocking(&source, &cache, |v| *v);
        assert_eq!(
            observed, 7,
            "a contended runtime must serve the cached snapshot, never block"
        );

        drop(held);

        // Once the turn ends, reads go live again.
        *source.lock().unwrap() = 9;
        assert_eq!(read_without_blocking(&source, &cache, |v| *v), 9);
        assert_eq!(*cache.lock().unwrap(), 9);
    }

    #[test]
    #[should_panic(expected = "runtime lock poisoned")]
    fn read_without_blocking_still_panics_on_a_poisoned_runtime() {
        // A poisoned runtime means inference panicked. Serving cached stats
        // over the top of that would report a healthy node, so this must keep
        // the pre-existing panic behaviour rather than fall back to the cache.
        let source = Mutex::new(7u32);
        let cache = Mutex::new(0u32);
        let _ = std::panic::catch_unwind(|| {
            let _guard = source.lock().unwrap();
            panic!("inference exploded");
        });
        assert!(source.is_poisoned(), "precondition: source is poisoned");
        let _ = read_without_blocking(&source, &cache, |v| *v);
    }
}
