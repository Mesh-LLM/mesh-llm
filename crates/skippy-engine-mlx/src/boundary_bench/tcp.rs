//! Production Skippy TCP framing measurements for synthetic activation boundaries.

use std::{
    collections::BTreeMap,
    fs,
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use skippy_engine::{
    StageEngine, StageEngineInfo, StageExecutionKind, StageExecutionOutput, StageExecutionRequest,
};
use skippy_metrics::{attr, metric, span};
use skippy_protocol::{
    StageConfig,
    binary::{
        MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_SIDEBAND_VALUES,
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
        activation_wire_bytes, encode_f32_activation_payload, recv_ready, recv_reply,
        write_stage_message,
    },
};
use skippy_server::{
    engine_transport::{EngineStageServerOptions, serve_stage_engine, serve_stage_engine_until},
    telemetry::{Telemetry, TelemetryLevel, TelemetryStats, lifecycle_attrs},
};

use super::{
    MlxBoundaryDurationSummary, PhaseTiming, code_revision, ensure_http_success, max_abs_diff,
    metrics_client, summarize, time_phase, validate_metrics_run_id, validate_roundtrip,
    wait_for_telemetry, wire_dtype_label,
};

const BENCHMARK_SCHEMA: &str = "mlx-tcp-boundary-v2";
const PREDICTED_SENTINEL: i32 = 424_243;
const CONNECT_DEADLINE: Duration = Duration::from_secs(10);
const CONNECT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(1);
const READY_TIMEOUT: Duration = Duration::from_secs(5);
const ROUNDTRIP_TIMEOUT: Duration = Duration::from_secs(30);

/// One explicit, reproducible production-TCP boundary benchmark run.
#[derive(Clone, Debug)]
pub struct MlxTcpBoundaryBenchConfig {
    pub width: usize,
    pub token_count: usize,
    pub wire_dtype: WireActivationDType,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub metrics_http: String,
    pub metrics_otlp_grpc: String,
    pub metrics_run_id: String,
    pub metrics_report_path: PathBuf,
    pub connect_addr: Option<SocketAddr>,
}

/// Foreground production sink for a separate-process or remote benchmark.
#[derive(Clone, Debug)]
pub struct MlxTcpBoundarySinkConfig {
    pub bind_addr: SocketAddr,
    pub width: usize,
    pub token_count: usize,
    pub wire_dtype: WireActivationDType,
}

/// Sender encode through production server decode and predicted reply.
#[derive(Clone, Debug, Serialize)]
pub struct MlxTcpBoundaryBenchReport {
    pub benchmark: &'static str,
    pub code_revision: String,
    pub metrics_run_id: String,
    pub metrics_report_path: PathBuf,
    pub width: usize,
    pub token_count: usize,
    pub wire_dtype: String,
    pub transport: &'static str,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub f32_boundary_bytes: usize,
    pub wire_activation_payload_bytes: usize,
    pub tcp_roundtrip: MlxBoundaryDurationSummary,
    pub warmup_validation_ack_received: bool,
    pub warmup_sink_acknowledged_max_abs_diff: f32,
    pub telemetry: TelemetryStats,
    pub canonical_span_count: u64,
}

/// Measures production activation encoding, Skippy framing, loopback TCP,
/// engine-transport decoding, and the predicted-token reply as one fence.
pub fn benchmark_mlx_tcp_boundary(
    config: &MlxTcpBoundaryBenchConfig,
) -> Result<MlxTcpBoundaryBenchReport> {
    let config = config.clone();
    thread::spawn(move || benchmark_mlx_tcp_boundary_inner(&config))
        .join()
        .map_err(|_| anyhow!("MLX TCP boundary benchmark thread panicked"))?
}

/// Runs the validating final-stage sink in the foreground until interrupted.
pub fn serve_mlx_tcp_boundary_sink(config: &MlxTcpBoundarySinkConfig) -> Result<()> {
    validate_protocol_shape(config.width, config.token_count, config.wire_dtype)?;
    let source = source_bytes(config.width, config.token_count)?;
    let engine = Arc::new(TcpBoundarySink::new(
        config.width,
        config.token_count,
        config.wire_dtype,
        source,
    )?);
    serve_stage_engine(
        engine,
        EngineStageServerOptions {
            bind_addr: config.bind_addr,
            downstream_addr: None,
            wire_dtype: config.wire_dtype,
        },
    )
}

fn benchmark_mlx_tcp_boundary_inner(
    config: &MlxTcpBoundaryBenchConfig,
) -> Result<MlxTcpBoundaryBenchReport> {
    validate_config(config)?;
    let source = source_bytes(config.width, config.token_count)?;
    let (server, engine, mut client) = match config.connect_addr {
        Some(connect_addr) => (None, None, connect_ready(connect_addr)?),
        None => {
            let engine = Arc::new(TcpBoundarySink::new(
                config.width,
                config.token_count,
                config.wire_dtype,
                Arc::clone(&source),
            )?);
            let (server, client) = TcpStageServer::spawn_ready(
                engine.clone(),
                EngineStageServerOptions {
                    bind_addr: "127.0.0.1:0".parse()?,
                    downstream_addr: None,
                    wire_dtype: config.wire_dtype,
                },
            )?;
            (Some(server), Some(engine), client)
        }
    };

    create_metrics_run(config)?;
    let mut metrics_run = MetricsRunGuard::new(config);
    let session_id = benchmark_session_id(&config.metrics_run_id);
    let mut warmup_roundtrip_max_abs_diff = 0.0_f32;
    for iteration in 0..config.warmup_iterations {
        let diff = run_roundtrip(
            config,
            &source,
            &mut client,
            session_id,
            u64::try_from(iteration)?,
        )?;
        warmup_roundtrip_max_abs_diff = warmup_roundtrip_max_abs_diff.max(diff);
    }
    if let Some(engine) = engine.as_ref() {
        let local_diff = engine
            .validation_diff(session_id)?
            .context("TCP boundary warmup acknowledgement has no matching local sink validation")?;
        ensure!(
            local_diff.to_bits() == warmup_roundtrip_max_abs_diff.to_bits(),
            "TCP boundary warmup acknowledgement differs from local sink validation"
        );
    }

    let samples = measure_roundtrips(config, &source, &mut client, session_id)?;
    drop(client);
    if let Some(server) = server {
        server.stop()?;
    }

    let telemetry_runtime = TcpTelemetryRuntime::new(config)?;
    let attrs = benchmark_attrs(config)?;
    for sample in &samples {
        telemetry_runtime.telemetry.emit_span(
            span::MLX_BOUNDARY_TCP_ROUNDTRIP,
            attrs.clone(),
            sample.start_unix_nanos,
            sample.end_unix_nanos,
        );
    }
    let telemetry = wait_for_telemetry(&telemetry_runtime.telemetry, samples.len())?;
    let canonical_report = finalize_metrics_run(config)?;
    metrics_run.mark_finalized();
    let canonical_span_count = canonical_report
        .get("counts")
        .and_then(|counts| counts.get("spans"))
        .and_then(Value::as_u64)
        .context("metrics-server report has no span count")?;
    ensure!(
        canonical_span_count == u64::try_from(samples.len())?,
        "metrics-server stored {canonical_span_count} spans; expected exactly {}",
        samples.len()
    );
    write_metrics_report(config, &canonical_report)?;

    Ok(MlxTcpBoundaryBenchReport {
        benchmark: BENCHMARK_SCHEMA,
        code_revision: code_revision(),
        metrics_run_id: config.metrics_run_id.clone(),
        metrics_report_path: config.metrics_report_path.clone(),
        width: config.width,
        token_count: config.token_count,
        wire_dtype: wire_dtype_label(config.wire_dtype)?.to_string(),
        transport: transport_label(config),
        warmup_iterations: config.warmup_iterations,
        measured_iterations: config.measured_iterations,
        f32_boundary_bytes: activation_bytes(config, size_of::<f32>())?,
        wire_activation_payload_bytes: activation_bytes(
            config,
            match config.wire_dtype {
                WireActivationDType::F32 => size_of::<f32>(),
                WireActivationDType::F16 => size_of::<u16>(),
                _ => unreachable!("validated wire dtype"),
            },
        )?,
        tcp_roundtrip: summarize(samples.iter().map(|sample| sample.elapsed))?,
        warmup_validation_ack_received: true,
        warmup_sink_acknowledged_max_abs_diff: warmup_roundtrip_max_abs_diff,
        telemetry,
        canonical_span_count,
    })
}

fn validate_config(config: &MlxTcpBoundaryBenchConfig) -> Result<()> {
    validate_protocol_shape(config.width, config.token_count, config.wire_dtype)?;
    ensure!(
        config.warmup_iterations > 0,
        "TCP boundary benchmark needs a correctness warmup"
    );
    ensure!(
        config.measured_iterations > 0,
        "TCP boundary benchmark needs measured iterations"
    );
    ensure!(
        !config.metrics_http.trim().is_empty(),
        "metrics-server HTTP endpoint is required"
    );
    ensure!(
        !config.metrics_otlp_grpc.trim().is_empty(),
        "metrics-server OTLP endpoint is required"
    );
    validate_metrics_run_id(&config.metrics_run_id)?;
    Ok(())
}

fn validate_protocol_shape(
    width: usize,
    token_count: usize,
    wire_dtype: WireActivationDType,
) -> Result<()> {
    ensure!(width > 0, "TCP boundary width must be non-zero");
    ensure!(token_count > 0, "TCP boundary token count must be non-zero");
    wire_dtype_label(wire_dtype)?;
    let token_count_i32 = i32::try_from(token_count).context("token count exceeds i32")?;
    let width_i32 = i32::try_from(width).context("activation width exceeds i32")?;
    u32::try_from(width).context("activation width exceeds u32")?;
    ensure!(
        token_count <= MAX_STAGE_SIDEBAND_VALUES,
        "token sideband count exceeds protocol maximum"
    );
    let wire_bytes = activation_wire_bytes(wire_dtype, token_count_i32, width_i32)?;
    ensure!(
        wire_bytes <= MAX_STAGE_ACTIVATION_BYTES,
        "wire activation exceeds protocol maximum"
    );
    let decoded_bytes =
        activation_wire_bytes(WireActivationDType::F32, token_count_i32, width_i32)?;
    ensure!(
        decoded_bytes <= MAX_STAGE_DECODED_ACTIVATION_BYTES,
        "decoded activation exceeds protocol maximum"
    );
    Ok(())
}

fn source_bytes(width: usize, token_count: usize) -> Result<Arc<Vec<u8>>> {
    let elements = width
        .checked_mul(token_count)
        .context("TCP boundary element count overflow")?;
    Ok(Arc::new(
        (0..elements)
            .flat_map(|index| {
                let value = ((index % 257) as f32 - 128.0) / 127.0 + 0.1;
                value.to_le_bytes()
            })
            .collect(),
    ))
}

fn measure_roundtrips(
    config: &MlxTcpBoundaryBenchConfig,
    source: &[u8],
    client: &mut TcpStream,
    session_id: u64,
) -> Result<Vec<PhaseTiming>> {
    (0..config.measured_iterations)
        .map(|iteration| {
            let request_id = u64::try_from(config.warmup_iterations + iteration)?;
            let (_, timing) =
                time_phase(|| run_roundtrip(config, source, client, session_id, request_id))?;
            Ok(timing)
        })
        .collect()
}

fn run_roundtrip(
    config: &MlxTcpBoundaryBenchConfig,
    source: &[u8],
    client: &mut TcpStream,
    session_id: u64,
    request_id: u64,
) -> Result<f32> {
    let token_count = i32::try_from(config.token_count)?;
    let width = i32::try_from(config.width)?;
    let kind = WireMessageKind::PrefillFinalEmbd;
    let mut state = StageStateHeader::new(kind, config.wire_dtype);
    state.prompt_token_count = token_count;
    state.source_stage_index = 0;
    let message = StageWireMessage {
        kind,
        pos_start: 0,
        token_count,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![0; config.token_count],
        positions: (0..token_count).collect(),
        activation: encode_f32_activation_payload(config.wire_dtype, token_count, width, source)?,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *client, &message, config.wire_dtype)?;
    use std::io::Write as _;
    client.flush()?;
    let reply = recv_reply(&mut *client)?;
    ensure!(
        reply.kind == WireReplyKind::PredictedToken
            && reply.predicted == PREDICTED_SENTINEL
            && reply.predicted_tokens.len() == 2
            && reply.predicted_tokens[0] == PREDICTED_SENTINEL,
        "TCP boundary sink returned no validation acknowledgement"
    );
    let diff = validation_diff_from_ack(reply.predicted_tokens[1]);
    validate_roundtrip(config.wire_dtype, diff)
        .context("TCP boundary sink acknowledged an invalid activation")?;
    Ok(diff)
}

struct TcpBoundarySink {
    info: StageEngineInfo,
    token_count: usize,
    wire_dtype: WireActivationDType,
    source: Arc<Vec<u8>>,
    validation: Mutex<Option<(u64, f32)>>,
}

impl TcpBoundarySink {
    fn new(
        width: usize,
        token_count: usize,
        wire_dtype: WireActivationDType,
        source: Arc<Vec<u8>>,
    ) -> Result<Self> {
        let info = StageEngineInfo {
            engine: "mlx-tcp-boundary-sink".to_string(),
            model_id: "synthetic/mlx-tcp-boundary".to_string(),
            stage_index: 1,
            layer_start: 1,
            layer_end: 2,
            total_layers: 2,
            activation_width: u32::try_from(width)?,
        };
        info.validate()?;
        Ok(Self {
            info,
            token_count,
            wire_dtype,
            source,
            validation: Mutex::new(None),
        })
    }

    fn validation_diff(&self, session_id: u64) -> Result<Option<f32>> {
        Ok(self
            .validation
            .lock()
            .map_err(|_| anyhow!("TCP boundary validation lock poisoned"))?
            .filter(|(validated_session, _)| *validated_session == session_id)
            .map(|(_, diff)| diff))
    }

    fn validate_session(&self, session_id: u64, activation: &[u8]) -> Result<f32> {
        if let Some(diff) = self.validation_diff(session_id)? {
            return Ok(diff);
        }
        let diff = max_abs_diff(&self.source, activation)?;
        validate_roundtrip(self.wire_dtype, diff)?;
        *self
            .validation
            .lock()
            .map_err(|_| anyhow!("TCP boundary validation lock poisoned"))? =
            Some((session_id, diff));
        Ok(diff)
    }
}

impl StageEngine for TcpBoundarySink {
    fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    fn execute(&self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        ensure!(
            request.kind == StageExecutionKind::PrefillFinal,
            "TCP boundary sink received the wrong execution kind"
        );
        let activation = request
            .input
            .context("TCP boundary sink requires an activation")?;
        ensure!(
            activation.token_count == self.token_count
                && activation.width == self.info.activation_width as usize,
            "TCP boundary sink received the wrong activation shape"
        );
        let diff = self.validate_session(request.session_id, &activation.f32_le_bytes)?;
        Ok(StageExecutionOutput {
            activation: None,
            predicted_tokens: vec![PREDICTED_SENTINEL, validation_diff_ack(diff)],
        })
    }

    fn reset_session(&self, session_id: u64) -> Result<()> {
        let mut validation = self
            .validation
            .lock()
            .map_err(|_| anyhow!("TCP boundary validation lock poisoned"))?;
        if validation
            .as_ref()
            .is_some_and(|(validated_session, _)| *validated_session == session_id)
        {
            *validation = None;
        }
        Ok(())
    }
}

struct TcpStageServer {
    shutdown: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<Result<()>>>,
}

impl TcpStageServer {
    fn spawn(engine: Arc<dyn StageEngine>, options: EngineStageServerOptions) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = Arc::clone(&shutdown);
        let join =
            thread::spawn(move || serve_stage_engine_until(engine, options, thread_shutdown));
        Self {
            shutdown,
            join: Some(join),
        }
    }

    fn spawn_ready(
        engine: Arc<dyn StageEngine>,
        options: EngineStageServerOptions,
    ) -> Result<(Self, TcpStream)> {
        const BIND_ATTEMPTS: usize = 3;
        let mut last_error = None;
        for _ in 0..BIND_ATTEMPTS {
            let addr = reserve_loopback_addr()?;
            let mut attempt_options = options.clone();
            attempt_options.bind_addr = addr;
            let server = Self::spawn(Arc::clone(&engine), attempt_options);
            match connect_ready(addr) {
                Ok(client) => return Ok((server, client)),
                Err(error) => {
                    let server_error = server.stop().err();
                    last_error = Some(match server_error {
                        Some(server_error) => anyhow!(
                            "connect TCP boundary stage at {addr}: {error:#}; server failed: {server_error:#}"
                        ),
                        None => error,
                    });
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("TCP boundary stage did not start")))
    }

    fn stop(mut self) -> Result<()> {
        self.finish()
    }

    fn finish(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.join()
            .map_err(|_| anyhow!("TCP boundary server panicked"))?
    }
}

impl Drop for TcpStageServer {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn reserve_loopback_addr() -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn connect_ready(addr: SocketAddr) -> Result<TcpStream> {
    let deadline = Instant::now() + CONNECT_DEADLINE;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        ensure!(
            !remaining.is_zero(),
            "connect TCP boundary stage at {addr} timed out"
        );
        let error = match TcpStream::connect_timeout(&addr, remaining.min(CONNECT_ATTEMPT_TIMEOUT))
        {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream.set_read_timeout(Some(remaining.min(READY_TIMEOUT)))?;
                stream.set_write_timeout(Some(ROUNDTRIP_TIMEOUT))?;
                match recv_ready(&mut stream) {
                    Ok(()) => {
                        stream.set_read_timeout(Some(ROUNDTRIP_TIMEOUT))?;
                        return Ok(stream);
                    }
                    Err(error) => anyhow!(error).context("receive TCP boundary ready handshake"),
                }
            }
            Err(error) => anyhow!(error).context("connect TCP boundary socket"),
        };
        if Instant::now() >= deadline {
            return Err(error).with_context(|| format!("connect TCP boundary stage at {addr}"));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn benchmark_session_id(metrics_run_id: &str) -> u64 {
    let digest = Sha256::digest(metrics_run_id.as_bytes());
    let bytes: [u8; size_of::<u64>()] = digest[..size_of::<u64>()]
        .try_into()
        .expect("SHA-256 prefix has the requested length");
    u64::from_le_bytes(bytes)
}

fn validation_diff_ack(diff: f32) -> i32 {
    i32::from_le_bytes(diff.to_bits().to_le_bytes())
}

fn validation_diff_from_ack(ack: i32) -> f32 {
    f32::from_bits(u32::from_le_bytes(ack.to_le_bytes()))
}

struct TcpTelemetryRuntime {
    telemetry: Telemetry,
    _runtime: tokio::runtime::Runtime,
}

impl TcpTelemetryRuntime {
    fn new(config: &MlxTcpBoundaryBenchConfig) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("build TCP boundary telemetry runtime")?;
        let telemetry = {
            let _guard = runtime.enter();
            Telemetry::new(
                Some(config.metrics_otlp_grpc.clone()),
                4096,
                telemetry_stage_config(config)?,
                TelemetryLevel::Summary,
            )
        };
        Ok(Self {
            telemetry,
            _runtime: runtime,
        })
    }
}

fn benchmark_attrs(config: &MlxTcpBoundaryBenchConfig) -> Result<BTreeMap<String, Value>> {
    let mut attrs = lifecycle_attrs(&telemetry_stage_config(config)?);
    attrs.insert(attr::TOKEN_COUNT.to_string(), json!(config.token_count));
    attrs.insert(
        attr::MESSAGE_KIND.to_string(),
        json!("mlx_boundary_tcp_roundtrip"),
    );
    attrs.insert(
        metric::ACTIVATION_BYTES_SENT.to_string(),
        json!(activation_bytes(
            config,
            match config.wire_dtype {
                WireActivationDType::F32 => size_of::<f32>(),
                WireActivationDType::F16 => size_of::<u16>(),
                _ => unreachable!("validated wire dtype"),
            }
        )?),
    );
    Ok(attrs)
}

fn telemetry_stage_config(config: &MlxTcpBoundaryBenchConfig) -> Result<StageConfig> {
    Ok(serde_json::from_value(json!({
        "run_id": config.metrics_run_id,
        "topology_id": BENCHMARK_SCHEMA,
        "model_id": "synthetic/mlx-tcp-boundary",
        "stage_id": "mlx-tcp-boundary",
        "stage_index": 1,
        "layer_start": 1,
        "layer_end": 2,
        "ctx_size": config.token_count,
        "load_mode": "artifact-slice",
        "bind_addr": "127.0.0.1:0"
    }))?)
}

fn create_metrics_run(config: &MlxTcpBoundaryBenchConfig) -> Result<()> {
    let response = metrics_client()?
        .post(format!(
            "{}/v1/runs",
            config.metrics_http.trim_end_matches('/')
        ))
        .json(&metrics_run_body(config)?)
        .send()
        .context("create metrics-server TCP boundary run")?;
    ensure_http_success(response, "create metrics-server TCP boundary run")?;
    Ok(())
}

fn metrics_run_body(config: &MlxTcpBoundaryBenchConfig) -> Result<Value> {
    Ok(json!({
        "run_id": config.metrics_run_id,
        "benchmark": BENCHMARK_SCHEMA,
        "code_revision": code_revision(),
        "width": config.width,
        "token_count": config.token_count,
        "wire_dtype": wire_dtype_label(config.wire_dtype)?,
        "transport": transport_label(config),
        "warmup_iterations": config.warmup_iterations,
        "measured_iterations": config.measured_iterations,
        "stages": [{
            "stage_id": "mlx-tcp-boundary",
            "engine": "skippy-engine-transport",
            "model_id": "synthetic/mlx-tcp-boundary"
        }]
    }))
}

fn finalize_metrics_run(config: &MlxTcpBoundaryBenchConfig) -> Result<Value> {
    let client = metrics_client()?;
    let base = config.metrics_http.trim_end_matches('/');
    let response = client
        .post(format!("{base}/v1/runs/{}/finalize", config.metrics_run_id))
        .send()
        .context("finalize metrics-server TCP boundary run")?;
    ensure_http_success(response, "finalize metrics-server TCP boundary run")?;
    let response = client
        .get(format!(
            "{base}/v1/runs/{}/report.json",
            config.metrics_run_id
        ))
        .send()
        .context("fetch metrics-server TCP boundary report")?;
    ensure_http_success(response, "fetch metrics-server TCP boundary report")?
        .json()
        .context("decode metrics-server TCP boundary report")
}

fn finalize_metrics_run_best_effort(config: &MlxTcpBoundaryBenchConfig) {
    let Ok(client) = metrics_client() else {
        return;
    };
    let base = config.metrics_http.trim_end_matches('/');
    let _ = client
        .post(format!("{base}/v1/runs/{}/finalize", config.metrics_run_id))
        .send();
}

struct MetricsRunGuard<'a> {
    config: &'a MlxTcpBoundaryBenchConfig,
    finalized: bool,
}

impl<'a> MetricsRunGuard<'a> {
    fn new(config: &'a MlxTcpBoundaryBenchConfig) -> Self {
        Self {
            config,
            finalized: false,
        }
    }

    fn mark_finalized(&mut self) {
        self.finalized = true;
    }
}

impl Drop for MetricsRunGuard<'_> {
    fn drop(&mut self) {
        if !self.finalized {
            finalize_metrics_run_best_effort(self.config);
        }
    }
}

fn write_metrics_report(config: &MlxTcpBoundaryBenchConfig, report: &Value) -> Result<()> {
    if let Some(parent) = config
        .metrics_report_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(
        &config.metrics_report_path,
        serde_json::to_vec_pretty(report)?,
    )
    .with_context(|| format!("write {}", config.metrics_report_path.display()))
}

fn activation_bytes(config: &MlxTcpBoundaryBenchConfig, element_bytes: usize) -> Result<usize> {
    config
        .width
        .checked_mul(config.token_count)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .context("TCP boundary byte count overflow")
}

fn transport_label(config: &MlxTcpBoundaryBenchConfig) -> &'static str {
    if config.connect_addr.is_some() {
        "external_tcp"
    } else {
        "loopback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MlxTcpBoundaryBenchConfig {
        MlxTcpBoundaryBenchConfig {
            width: 4096,
            token_count: 32,
            wire_dtype: WireActivationDType::F16,
            warmup_iterations: 1,
            measured_iterations: 2,
            metrics_http: "https://collector.invalid/private".to_string(),
            metrics_otlp_grpc: "https://otlp.invalid/private".to_string(),
            metrics_run_id: "tcp-test-run".to_string(),
            metrics_report_path: PathBuf::from("/private/report.json"),
            connect_addr: None,
        }
    }

    #[test]
    fn telemetry_contains_no_transport_targets_or_local_paths() {
        let mut config = config();
        config.connect_addr = Some("203.0.113.10:1234".parse().unwrap());
        let exported = json!({
            "stage_config": telemetry_stage_config(&config).unwrap(),
            "span_attributes": benchmark_attrs(&config).unwrap(),
            "run_create": metrics_run_body(&config).unwrap(),
        });
        let serialized = serde_json::to_string(&exported).unwrap();
        assert!(!serialized.contains("collector.invalid"));
        assert!(!serialized.contains("otlp.invalid"));
        assert!(!serialized.contains("/private/"));
        assert!(!serialized.contains("203.0.113.10"));
    }

    #[test]
    fn source_values_are_finite_and_in_the_codec_gate_range() {
        let config = config();
        let source = source_bytes(config.width, config.token_count).unwrap();
        assert_eq!(source.len(), 4096 * 32 * size_of::<f32>());
        assert!(
            source
                .chunks_exact(4)
                .all(|chunk| { f32::from_le_bytes(chunk.try_into().unwrap()).is_finite() })
        );
    }

    #[test]
    fn production_tcp_roundtrip_reaches_and_validates_sink() {
        for (wire_dtype, expected_diff) in [
            (WireActivationDType::F32, 0.0),
            (WireActivationDType::F16, 0.001),
        ] {
            let mut config = config();
            config.width = 8;
            config.token_count = 2;
            config.wire_dtype = wire_dtype;
            let source = source_bytes(config.width, config.token_count).unwrap();
            let engine = Arc::new(
                TcpBoundarySink::new(
                    config.width,
                    config.token_count,
                    config.wire_dtype,
                    Arc::clone(&source),
                )
                .unwrap(),
            );
            let (server, mut client) = TcpStageServer::spawn_ready(
                engine.clone(),
                EngineStageServerOptions {
                    bind_addr: "127.0.0.1:0".parse().unwrap(),
                    downstream_addr: None,
                    wire_dtype: config.wire_dtype,
                },
            )
            .unwrap();

            let session_id = benchmark_session_id(&config.metrics_run_id);
            let acknowledged_diff =
                run_roundtrip(&config, &source, &mut client, session_id, 1).unwrap();
            assert!(acknowledged_diff <= expected_diff);
            assert_eq!(
                engine.validation_diff(session_id).unwrap(),
                Some(acknowledged_diff)
            );
            drop(client);
            server.stop().unwrap();
        }
    }

    #[test]
    fn sink_validates_each_session_and_failed_validation_does_not_poison_it() {
        let config = config();
        let source = source_bytes(config.width, config.token_count).unwrap();
        let engine = TcpBoundarySink::new(
            config.width,
            config.token_count,
            config.wire_dtype,
            Arc::clone(&source),
        )
        .unwrap();
        assert!(engine.validate_session(11, &source).unwrap() <= 0.001);

        let mut invalid = source.as_ref().clone();
        invalid[..size_of::<f32>()].copy_from_slice(&10.0_f32.to_le_bytes());
        assert!(engine.validate_session(12, &invalid).is_err());
        assert_eq!(engine.validation_diff(12).unwrap(), None);

        assert!(engine.validate_session(12, &source).unwrap() <= 0.001);
        assert!(engine.validation_diff(12).unwrap().is_some());
    }

    #[test]
    fn validation_ack_and_session_identity_round_trip() {
        let diff = 0.000_452_160_84_f32;
        assert_eq!(validation_diff_from_ack(validation_diff_ack(diff)), diff);
        assert_ne!(benchmark_session_id("run-a"), benchmark_session_id("run-b"));
    }

    #[test]
    fn oversized_protocol_shapes_fail_before_source_allocation() {
        let mut config = config();
        config.token_count = MAX_STAGE_SIDEBAND_VALUES + 1;
        assert!(validate_config(&config).is_err());

        config.token_count = 1;
        config.width = MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 + 1;
        assert!(validate_config(&config).is_err());
    }
}
