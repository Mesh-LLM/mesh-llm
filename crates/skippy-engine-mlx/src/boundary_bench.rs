//! Metrics-server-backed measurements for the MLX activation boundary fence.

mod tcp;

pub use tcp::{
    MlxTcpBoundaryBenchConfig, MlxTcpBoundaryBenchReport, MlxTcpBoundarySinkConfig,
    benchmark_mlx_tcp_boundary, serve_mlx_tcp_boundary_sink,
};

use std::{
    collections::BTreeMap,
    fs,
    mem::size_of,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use safemlx::{
    Array, Device, DeviceType, Dtype, Stream,
    memory::{active_memory, cache_memory, peak_memory, reset_peak_memory},
    transforms::eval,
};
use serde::Serialize;
use serde_json::{Value, json};
use skippy_metrics::{attr, metric, span};
use skippy_protocol::{
    StageConfig,
    binary::{
        StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
        encode_f32_activation_payload,
    },
};
use skippy_server::telemetry::{
    Telemetry, TelemetryLevel, TelemetryStats, lifecycle_attrs, now_unix_nanos,
};

const BENCHMARK_SCHEMA: &str = "mlx-boundary-fence-v2";

/// One explicit, reproducible boundary-fence benchmark run.
#[derive(Clone, Debug)]
pub struct MlxBoundaryBenchConfig {
    pub width: usize,
    pub token_count: usize,
    pub wire_dtype: WireActivationDType,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub metrics_http: String,
    pub metrics_otlp_grpc: String,
    pub metrics_run_id: String,
    pub metrics_report_path: PathBuf,
}

/// Duration distribution in microseconds.
#[derive(Clone, Debug, Serialize)]
pub struct MlxBoundaryDurationSummary {
    pub samples: usize,
    pub min_us: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub max_us: f64,
}

/// Local measurements paired with a canonical metrics-server report.
#[derive(Clone, Debug, Serialize)]
pub struct MlxBoundaryBenchReport {
    pub benchmark: &'static str,
    pub code_revision: String,
    pub metrics_run_id: String,
    pub metrics_report_path: PathBuf,
    pub width: usize,
    pub token_count: usize,
    pub wire_dtype: String,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub f32_boundary_bytes: usize,
    pub wire_activation_payload_bytes: usize,
    pub eval_fence: MlxBoundaryDurationSummary,
    pub host_copy: MlxBoundaryDurationSummary,
    pub eval_and_host_copy_total: MlxBoundaryDurationSummary,
    pub encode: MlxBoundaryDurationSummary,
    pub decode: MlxBoundaryDurationSummary,
    pub codec_total: MlxBoundaryDurationSummary,
    pub max_roundtrip_abs_diff: f32,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
    pub telemetry: TelemetryStats,
    pub canonical_span_count: u64,
}

struct BoundarySample {
    eval_fence: PhaseTiming,
    host_copy: PhaseTiming,
    encode: PhaseTiming,
    decode: PhaseTiming,
    wire_bytes: usize,
    roundtrip_max_abs_diff: f32,
}

struct PhaseTiming {
    start_unix_nanos: u64,
    end_unix_nanos: u64,
    elapsed: Duration,
}

struct TelemetryRuntime {
    telemetry: Telemetry,
    _runtime: tokio::runtime::Runtime,
}

struct MetricsRunGuard<'a> {
    config: &'a MlxBoundaryBenchConfig,
    finalized: bool,
}

/// Measures the lazy MLX eval/readback fence and the existing Skippy activation
/// codecs, emits bounded OTLP spans, and exports the canonical metrics report.
pub fn benchmark_mlx_boundary(config: &MlxBoundaryBenchConfig) -> Result<MlxBoundaryBenchReport> {
    let config = config.clone();
    thread::spawn(move || benchmark_mlx_boundary_inner(&config))
        .join()
        .map_err(|_| anyhow!("MLX boundary benchmark thread panicked"))?
}

fn benchmark_mlx_boundary_inner(config: &MlxBoundaryBenchConfig) -> Result<MlxBoundaryBenchReport> {
    validate_config(config)?;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let (source, offset) = prepared_source(config, &stream)?;
    let attrs = benchmark_attrs(config)?;

    for _ in 0..config.warmup_iterations {
        run_sample(config, &source, &offset, &stream)?;
    }
    reset_peak_memory()?;
    create_metrics_run(config)?;
    let mut metrics_run = MetricsRunGuard::new(config);

    let samples = measure_samples(config, &source, &offset, &stream)?;
    let mlx_active_memory_bytes = active_memory()?;
    let mlx_cache_memory_bytes = cache_memory()?;
    let mlx_peak_memory_bytes = peak_memory()?;

    // Start the exporter only after all timed work has completed. The spans
    // retain their original wall-clock timestamps without perturbing samples.
    let telemetry_runtime = TelemetryRuntime::new(config)?;
    for sample in &samples {
        emit_sample(&telemetry_runtime.telemetry, &attrs, sample);
    }
    let expected_spans = config
        .measured_iterations
        .checked_mul(4)
        .context("telemetry span count overflow")?;
    let telemetry = wait_for_telemetry(&telemetry_runtime.telemetry, expected_spans)?;
    let canonical_report = finalize_metrics_run(config)?;
    metrics_run.mark_finalized();
    let canonical_span_count = canonical_report
        .get("counts")
        .and_then(|counts| counts.get("spans"))
        .and_then(Value::as_u64)
        .context("metrics-server report has no span count")?;
    ensure!(
        canonical_span_count == u64::try_from(expected_spans)?,
        "metrics-server stored {canonical_span_count} spans; expected exactly {expected_spans}"
    );
    write_metrics_report(config, &canonical_report)?;

    let wire_activation_payload_bytes = samples
        .first()
        .context("boundary benchmark produced no samples")?
        .wire_bytes;
    ensure!(
        samples
            .iter()
            .all(|sample| sample.wire_bytes == wire_activation_payload_bytes),
        "wire payload size changed between samples"
    );
    Ok(MlxBoundaryBenchReport {
        benchmark: BENCHMARK_SCHEMA,
        code_revision: code_revision(),
        metrics_run_id: config.metrics_run_id.clone(),
        metrics_report_path: config.metrics_report_path.clone(),
        width: config.width,
        token_count: config.token_count,
        wire_dtype: wire_dtype_label(config.wire_dtype)?.to_string(),
        warmup_iterations: config.warmup_iterations,
        measured_iterations: config.measured_iterations,
        f32_boundary_bytes: boundary_bytes(config, size_of::<f32>())?,
        wire_activation_payload_bytes,
        eval_fence: summarize(samples.iter().map(|sample| sample.eval_fence.elapsed))?,
        host_copy: summarize(samples.iter().map(|sample| sample.host_copy.elapsed))?,
        eval_and_host_copy_total: summarize(
            samples
                .iter()
                .map(|sample| sample.eval_fence.elapsed + sample.host_copy.elapsed),
        )?,
        encode: summarize(samples.iter().map(|sample| sample.encode.elapsed))?,
        decode: summarize(samples.iter().map(|sample| sample.decode.elapsed))?,
        codec_total: summarize(
            samples
                .iter()
                .map(|sample| sample.encode.elapsed + sample.decode.elapsed),
        )?,
        max_roundtrip_abs_diff: samples
            .iter()
            .map(|sample| sample.roundtrip_max_abs_diff)
            .fold(0.0_f32, f32::max),
        mlx_active_memory_bytes,
        mlx_cache_memory_bytes,
        mlx_peak_memory_bytes,
        telemetry,
        canonical_span_count,
    })
}

fn measure_samples(
    config: &MlxBoundaryBenchConfig,
    source: &Array,
    offset: &Array,
    stream: &Stream,
) -> Result<Vec<BoundarySample>> {
    (0..config.measured_iterations)
        .map(|_| run_sample(config, source, offset, stream))
        .collect()
}

fn validate_config(config: &MlxBoundaryBenchConfig) -> Result<()> {
    ensure!(config.width > 0, "boundary width must be non-zero");
    ensure!(
        config.token_count > 0,
        "boundary token count must be non-zero"
    );
    ensure!(
        config.measured_iterations > 0,
        "boundary benchmark needs measured iterations"
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
    wire_dtype_label(config.wire_dtype)?;
    boundary_bytes(config, size_of::<f32>())?;
    Ok(())
}

fn prepared_source(config: &MlxBoundaryBenchConfig, stream: &Stream) -> Result<(Array, Array)> {
    let element_count = config
        .width
        .checked_mul(config.token_count)
        .context("boundary element count overflow")?;
    let values = (0..element_count)
        .map(|index| ((index % 257) as f32 - 128.0) / 127.0)
        .collect::<Vec<_>>();
    let shape = [
        1,
        i32::try_from(config.token_count)?,
        i32::try_from(config.width)?,
    ];
    let source = Array::from_slice(&values, &shape).as_dtype(Dtype::Float32, stream)?;
    let offset = Array::from_slice(&[0.1_f32], &[1]);
    eval([&source, &offset])?;
    stream.synchronize()?;
    Ok((source, offset))
}

fn run_sample(
    config: &MlxBoundaryBenchConfig,
    source: &Array,
    offset: &Array,
    stream: &Stream,
) -> Result<BoundarySample> {
    // Graph construction is intentionally outside the completion fence.
    let output = source
        .add(offset, stream)?
        .as_dtype(Dtype::Float32, stream)?;
    let ((), eval_fence) = time_phase(|| {
        eval([&output])?;
        stream.synchronize()?;
        Ok(())
    })?;
    let (f32_bytes, host_copy) = time_phase(|| {
        let evaluated = output.evaluated()?;
        Ok(bytemuck::cast_slice(evaluated.as_slice::<f32>()).to_vec())
    })?;
    let token_count = i32::try_from(config.token_count)?;
    let width = i32::try_from(config.width)?;
    let (wire_payload, encode) = time_phase(|| {
        Ok(encode_f32_activation_payload(
            config.wire_dtype,
            token_count,
            width,
            &f32_bytes,
        )?)
    })?;
    let wire_bytes = wire_payload.len();
    let (decoded, decode) =
        time_phase(|| Ok(wire_message(config, wire_payload)?.activation_f32_payload(width)?))?;
    let roundtrip_max_abs_diff = max_abs_diff(&f32_bytes, &decoded)?;
    validate_roundtrip(config.wire_dtype, roundtrip_max_abs_diff)?;
    Ok(BoundarySample {
        eval_fence,
        host_copy,
        encode,
        decode,
        wire_bytes,
        roundtrip_max_abs_diff,
    })
}

fn validate_roundtrip(wire_dtype: WireActivationDType, max_abs_diff: f32) -> Result<()> {
    ensure!(
        max_abs_diff.is_finite(),
        "activation codec produced a non-finite difference"
    );
    match wire_dtype {
        WireActivationDType::F32 => {
            ensure!(
                max_abs_diff == 0.0,
                "F32 activation codec was not exact: max abs diff {max_abs_diff}"
            );
        }
        WireActivationDType::F16 => {
            ensure!(
                max_abs_diff <= 0.001,
                "F16 activation codec exceeded synthetic-range error bound: {max_abs_diff} > 0.001"
            );
        }
        _ => bail!("unsupported activation codec dtype"),
    }
    Ok(())
}

fn wire_message(config: &MlxBoundaryBenchConfig, activation: Vec<u8>) -> Result<StageWireMessage> {
    let kind = WireMessageKind::PrefillFinalEmbd;
    let mut state = StageStateHeader::new(kind, config.wire_dtype);
    state.prompt_token_count = i32::try_from(config.token_count)?;
    state.source_stage_index = 0;
    Ok(StageWireMessage {
        kind,
        pos_start: 0,
        token_count: i32::try_from(config.token_count)?,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation,
        raw_bytes: Vec::new(),
    })
}

fn time_phase<T>(work: impl FnOnce() -> Result<T>) -> Result<(T, PhaseTiming)> {
    let start_unix_nanos = u64::try_from(now_unix_nanos())?;
    let start = Instant::now();
    let value = work()?;
    let elapsed = start.elapsed();
    let end_unix_nanos = start_unix_nanos
        .checked_add(u64::try_from(elapsed.as_nanos())?)
        .context("boundary phase timestamp overflow")?;
    Ok((
        value,
        PhaseTiming {
            start_unix_nanos,
            end_unix_nanos,
            elapsed,
        },
    ))
}

fn emit_sample(telemetry: &Telemetry, attrs: &BTreeMap<String, Value>, sample: &BoundarySample) {
    emit_phase(
        telemetry,
        span::MLX_BOUNDARY_EVAL_FENCE,
        attrs,
        &sample.eval_fence,
    );
    emit_phase(
        telemetry,
        span::MLX_BOUNDARY_HOST_COPY,
        attrs,
        &sample.host_copy,
    );
    let mut encode_attrs = attrs.clone();
    encode_attrs.insert(
        metric::ACTIVATION_BYTES_SENT.to_string(),
        json!(sample.wire_bytes),
    );
    emit_phase(
        telemetry,
        span::MLX_BOUNDARY_ENCODE,
        &encode_attrs,
        &sample.encode,
    );
    emit_phase(telemetry, span::MLX_BOUNDARY_DECODE, attrs, &sample.decode);
}

fn emit_phase(
    telemetry: &Telemetry,
    name: &str,
    attrs: &BTreeMap<String, Value>,
    timing: &PhaseTiming,
) {
    telemetry.emit_span(
        name,
        attrs.clone(),
        timing.start_unix_nanos,
        timing.end_unix_nanos,
    );
}

fn benchmark_attrs(config: &MlxBoundaryBenchConfig) -> Result<BTreeMap<String, Value>> {
    let mut attrs = lifecycle_attrs(&telemetry_stage_config(config)?);
    attrs.insert(attr::TOKEN_COUNT.to_string(), json!(config.token_count));
    attrs.insert(attr::MESSAGE_KIND.to_string(), json!("mlx_boundary_fence"));
    Ok(attrs)
}

fn validate_metrics_run_id(run_id: &str) -> Result<()> {
    ensure!(!run_id.is_empty(), "metrics run ID is required");
    ensure!(run_id.len() <= 128, "metrics run ID exceeds 128 bytes");
    ensure!(
        run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "metrics run ID may contain only ASCII letters, digits, '.', '_', and '-'"
    );
    Ok(())
}

fn telemetry_stage_config(config: &MlxBoundaryBenchConfig) -> Result<StageConfig> {
    Ok(serde_json::from_value(json!({
        "run_id": config.metrics_run_id,
        "topology_id": BENCHMARK_SCHEMA,
        "model_id": "synthetic/mlx-boundary-fence",
        "stage_id": "mlx-boundary-fence",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "ctx_size": config.token_count,
        "load_mode": "artifact-slice",
        "bind_addr": "127.0.0.1:0"
    }))?)
}

impl TelemetryRuntime {
    fn new(config: &MlxBoundaryBenchConfig) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("build boundary telemetry runtime")?;
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

impl<'a> MetricsRunGuard<'a> {
    fn new(config: &'a MlxBoundaryBenchConfig) -> Self {
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

fn wait_for_telemetry(telemetry: &Telemetry, expected_spans: usize) -> Result<TelemetryStats> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let stats = telemetry.stats();
        if stats.sent >= u64::try_from(expected_spans)? {
            ensure!(stats.dropped == 0, "boundary telemetry dropped events");
            ensure!(stats.export_errors == 0, "boundary telemetry export failed");
            return Ok(stats);
        }
        ensure!(
            Instant::now() < deadline,
            "metrics-server did not ingest {expected_spans} boundary spans; queued={} sent={} dropped={} errors={}",
            stats.queued,
            stats.sent,
            stats.dropped,
            stats.export_errors,
        );
        thread::sleep(Duration::from_millis(25));
    }
}

fn create_metrics_run(config: &MlxBoundaryBenchConfig) -> Result<()> {
    let response = metrics_client()?
        .post(format!(
            "{}/v1/runs",
            config.metrics_http.trim_end_matches('/')
        ))
        .json(&metrics_run_body(config)?)
        .send()
        .context("create metrics-server boundary run")?;
    ensure_http_success(response, "create metrics-server boundary run")?;
    Ok(())
}

fn metrics_run_body(config: &MlxBoundaryBenchConfig) -> Result<Value> {
    Ok(json!({
        "run_id": config.metrics_run_id,
        "benchmark": BENCHMARK_SCHEMA,
        "code_revision": code_revision(),
        "width": config.width,
        "token_count": config.token_count,
        "wire_dtype": wire_dtype_label(config.wire_dtype)?,
        "warmup_iterations": config.warmup_iterations,
        "measured_iterations": config.measured_iterations,
        "stages": [{
            "stage_id": "mlx-boundary-fence",
            "engine": "mlx",
            "model_id": "synthetic/mlx-boundary-fence"
        }]
    }))
}

fn code_revision() -> String {
    std::env::var("MESH_LLM_BUILD_REVISION")
        .ok()
        .filter(|revision| validate_metrics_run_id(revision).is_ok())
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn finalize_metrics_run(config: &MlxBoundaryBenchConfig) -> Result<Value> {
    let client = metrics_client()?;
    let base = config.metrics_http.trim_end_matches('/');
    let response = client
        .post(format!("{base}/v1/runs/{}/finalize", config.metrics_run_id))
        .send()
        .context("finalize metrics-server boundary run")?;
    ensure_http_success(response, "finalize metrics-server boundary run")?;
    let response = client
        .get(format!(
            "{base}/v1/runs/{}/report.json",
            config.metrics_run_id
        ))
        .send()
        .context("fetch metrics-server boundary report")?;
    ensure_http_success(response, "fetch metrics-server boundary report")?
        .json()
        .context("decode metrics-server boundary report")
}

fn finalize_metrics_run_best_effort(config: &MlxBoundaryBenchConfig) {
    let Ok(client) = metrics_client() else {
        return;
    };
    let base = config.metrics_http.trim_end_matches('/');
    let _ = client
        .post(format!("{base}/v1/runs/{}/finalize", config.metrics_run_id))
        .send();
}

fn metrics_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .build()?)
}

fn ensure_http_success(
    response: reqwest::blocking::Response,
    operation: &str,
) -> Result<reqwest::blocking::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().unwrap_or_default();
    Err(anyhow!("{operation} failed with HTTP {status}: {body}"))
}

fn write_metrics_report(config: &MlxBoundaryBenchConfig, report: &Value) -> Result<()> {
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

fn max_abs_diff(expected_bytes: &[u8], actual_bytes: &[u8]) -> Result<f32> {
    ensure!(
        expected_bytes.len() == actual_bytes.len(),
        "roundtrip activation byte counts differ"
    );
    ensure!(
        expected_bytes.len().is_multiple_of(size_of::<f32>()),
        "activation bytes are not aligned to F32 values"
    );
    expected_bytes
        .chunks_exact(size_of::<f32>())
        .zip(actual_bytes.chunks_exact(size_of::<f32>()))
        .try_fold(0.0_f32, |max_diff, (left, right)| {
            let left = f32::from_le_bytes(left.try_into().expect("four-byte chunk"));
            let right = f32::from_le_bytes(right.try_into().expect("four-byte chunk"));
            ensure!(
                left.is_finite() && right.is_finite(),
                "activation codec produced a non-finite value"
            );
            Ok(max_diff.max((left - right).abs()))
        })
}

fn summarize(durations: impl Iterator<Item = Duration>) -> Result<MlxBoundaryDurationSummary> {
    let mut values = durations
        .map(|duration| duration.as_secs_f64() * 1_000_000.0)
        .collect::<Vec<_>>();
    ensure!(!values.is_empty(), "cannot summarize zero samples");
    values.sort_by(f64::total_cmp);
    let mean_us = values.iter().sum::<f64>() / values.len() as f64;
    Ok(MlxBoundaryDurationSummary {
        samples: values.len(),
        min_us: values[0],
        mean_us,
        p50_us: percentile(&values, 0.50),
        p95_us: percentile(&values, 0.95),
        max_us: values[values.len() - 1],
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn boundary_bytes(config: &MlxBoundaryBenchConfig, element_bytes: usize) -> Result<usize> {
    config
        .width
        .checked_mul(config.token_count)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .context("boundary byte count overflow")
}

fn wire_dtype_label(wire_dtype: WireActivationDType) -> Result<&'static str> {
    match wire_dtype {
        WireActivationDType::F32 => Ok("f32"),
        WireActivationDType::F16 => Ok("f16"),
        other => bail!("MLX boundary benchmark does not support {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_summary_uses_nearest_rank_percentiles() {
        let summary =
            summarize([1_u64, 2, 3, 4, 100].into_iter().map(Duration::from_micros)).unwrap();
        assert_eq!(summary.samples, 5);
        assert_eq!(summary.p50_us, 3.0);
        assert_eq!(summary.p95_us, 100.0);
    }

    #[test]
    fn telemetry_config_contains_no_external_or_local_identifiers() {
        let config = MlxBoundaryBenchConfig {
            width: 4096,
            token_count: 32,
            wire_dtype: WireActivationDType::F16,
            warmup_iterations: 1,
            measured_iterations: 2,
            metrics_http: "https://collector.invalid/private".to_string(),
            metrics_otlp_grpc: "https://otlp.invalid/private".to_string(),
            metrics_run_id: "test-run".to_string(),
            metrics_report_path: PathBuf::from("/private/report.json"),
        };
        let exported = json!({
            "stage_config": telemetry_stage_config(&config).unwrap(),
            "span_attributes": benchmark_attrs(&config).unwrap(),
            "run_create": metrics_run_body(&config).unwrap(),
        });
        let serialized = serde_json::to_string(&exported).unwrap();
        assert!(!serialized.contains("collector.invalid"));
        assert!(!serialized.contains("otlp.invalid"));
        assert!(!serialized.contains("/private/"));
    }

    #[test]
    fn metrics_run_id_is_safe_for_export_and_url_paths() {
        for valid in ["test-run", "mlx.boundary_01", "A1"] {
            validate_metrics_run_id(valid).unwrap();
        }
        for invalid in [
            "",
            "has/slash",
            "has?query",
            "https://collector",
            "has space",
        ] {
            assert!(validate_metrics_run_id(invalid).is_err(), "{invalid}");
        }
        assert!(validate_metrics_run_id(&"x".repeat(129)).is_err());
    }

    #[test]
    fn activation_roundtrip_error_is_gated() {
        validate_roundtrip(WireActivationDType::F32, 0.0).unwrap();
        assert!(validate_roundtrip(WireActivationDType::F32, f32::EPSILON).is_err());
        validate_roundtrip(WireActivationDType::F16, 0.001).unwrap();
        assert!(validate_roundtrip(WireActivationDType::F16, 0.001_1).is_err());
        assert!(validate_roundtrip(WireActivationDType::F16, f32::NAN).is_err());
    }
}
