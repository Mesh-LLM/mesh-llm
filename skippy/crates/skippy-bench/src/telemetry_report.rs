use std::{collections::BTreeMap, fs, path::Path, thread, time::Duration};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const OPENAI_DECODE_TOKEN_SPAN: &str = "stage.openai_decode_token";

#[derive(Default, Serialize)]
pub struct BenchTelemetry {
    pub metrics_http: Option<String>,
    pub metrics_run_id: String,
    pub status: &'static str,
    pub detail: Option<String>,
    pub report_path: Option<String>,
    pub request_count: Option<u64>,
    pub span_count: Option<u64>,
    pub ttft_ms: Option<TelemetryAggregate>,
    pub fttt_ms: Option<TelemetryAggregate>,
    pub request_latency_ms: Option<TelemetryAggregate>,
    pub generation_latency_ms: Option<TelemetryAggregate>,
}

#[derive(Default, Serialize)]
pub struct TelemetryAggregate {
    pub count: usize,
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Deserialize)]
struct CreateRunResponse {
    run_id: String,
}

#[derive(Deserialize)]
struct RunStatusResponse {
    status: String,
    #[serde(default)]
    request_count: Option<u64>,
    #[serde(default)]
    span_count: Option<u64>,
}

pub fn pending(metrics_http: &str, metrics_run_id: &str) -> BenchTelemetry {
    BenchTelemetry {
        metrics_http: Some(metrics_http.to_string()),
        metrics_run_id: metrics_run_id.to_string(),
        status: "pending",
        detail: Some("telemetry collection has not run yet".to_string()),
        ..BenchTelemetry::default()
    }
}

pub fn create_run(metrics_http: &str, metrics_run_id: &str, config: &Value) -> Result<()> {
    let body = create_run_body(metrics_run_id, config);
    let client = reqwest::blocking::Client::new();
    let base = metrics_http.trim_end_matches('/');
    let response = client
        .post(format!("{base}/v1/runs"))
        .json(&body)
        .send()
        .with_context(|| format!("create metrics-server run {metrics_run_id}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if body.contains("UNIQUE constraint failed: runs.run_id") {
            return accept_existing_run(&client, base, metrics_run_id);
        }
        return Err(anyhow!(
            "metrics-server rejected run {metrics_run_id}: HTTP {status}: {body}"
        ));
    }
    let response = response
        .json::<CreateRunResponse>()
        .context("decode metrics-server create-run response")?;
    if response.run_id == metrics_run_id {
        Ok(())
    } else {
        Err(anyhow!(
            "metrics-server returned unexpected run_id {}",
            response.run_id
        ))
    }
}

fn accept_existing_run(
    client: &reqwest::blocking::Client,
    metrics_http: &str,
    metrics_run_id: &str,
) -> Result<()> {
    let status = client
        .get(format!("{metrics_http}/v1/runs/{metrics_run_id}/status"))
        .send()
        .with_context(|| format!("read metrics-server run status {metrics_run_id}"))?
        .error_for_status()
        .with_context(|| format!("metrics-server rejected status for run {metrics_run_id}"))?
        .json::<RunStatusResponse>()
        .context("decode metrics-server run status response")?;
    if status.status == "running" || status.status == "implicit" {
        Ok(())
    } else {
        Err(anyhow!(
            "metrics-server run {metrics_run_id} already exists with status {}",
            status.status
        ))
    }
}

fn create_run_body(metrics_run_id: &str, config: &Value) -> Value {
    let mut body = match config {
        Value::Object(object) => object.clone(),
        value => {
            let mut object = serde_json::Map::new();
            object.insert("config".to_string(), value.clone());
            object
        }
    };
    body.insert(
        "run_id".to_string(),
        Value::String(metrics_run_id.to_string()),
    );
    Value::Object(body)
}

pub fn finalize_and_collect(
    metrics_http: &str,
    metrics_run_id: &str,
    report_path: &Path,
) -> Result<BenchTelemetry> {
    let report = fetch_metrics_report(metrics_http, metrics_run_id, report_path)?;
    Ok(telemetry_from_report(
        metrics_http,
        metrics_run_id,
        report_path,
        report,
    ))
}

pub fn finalize_only(metrics_http: &str, metrics_run_id: &str) -> Result<BenchTelemetry> {
    let client = reqwest::blocking::Client::new();
    let base = metrics_http.trim_end_matches('/');
    let status = client
        .post(format!("{base}/v1/runs/{metrics_run_id}/finalize"))
        .send()
        .with_context(|| format!("finalize metrics-server run {metrics_run_id}"))?
        .error_for_status()
        .with_context(|| format!("metrics-server rejected finalize for run {metrics_run_id}"))?
        .json::<RunStatusResponse>()
        .context("decode finalized metrics-server run status")?;
    Ok(BenchTelemetry {
        metrics_http: Some(metrics_http.to_string()),
        metrics_run_id: metrics_run_id.to_string(),
        status: "finalized",
        detail: Some(
            "metrics finalized server-side without downloading the full raw-span report"
                .to_string(),
        ),
        request_count: status.request_count,
        span_count: status.span_count,
        ..BenchTelemetry::default()
    })
}

pub fn unavailable(
    metrics_http: &str,
    metrics_run_id: &str,
    error: &anyhow::Error,
) -> BenchTelemetry {
    BenchTelemetry {
        metrics_http: Some(metrics_http.to_string()),
        metrics_run_id: metrics_run_id.to_string(),
        status: "unavailable",
        detail: Some(error.to_string()),
        ..BenchTelemetry::default()
    }
}

fn fetch_metrics_report(
    metrics_http: &str,
    metrics_run_id: &str,
    report_path: &Path,
) -> Result<Value> {
    let client = reqwest::blocking::Client::new();
    let base = metrics_http.trim_end_matches('/');
    client
        .post(format!("{base}/v1/runs/{metrics_run_id}/finalize"))
        .send()
        .with_context(|| format!("finalize metrics-server run {metrics_run_id}"))?
        .error_for_status()
        .with_context(|| format!("metrics-server rejected finalize for run {metrics_run_id}"))?;
    thread::sleep(Duration::from_millis(250));
    let report = client
        .get(format!("{base}/v1/runs/{metrics_run_id}/report.json"))
        .send()
        .with_context(|| format!("fetch metrics report for run {metrics_run_id}"))?
        .error_for_status()
        .with_context(|| format!("metrics report request failed for run {metrics_run_id}"))?
        .json::<Value>()
        .context("decode metrics report JSON")?;
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    Ok(report)
}

fn telemetry_from_report(
    metrics_http: &str,
    metrics_run_id: &str,
    report_path: &Path,
    report: Value,
) -> BenchTelemetry {
    let spans = report
        .get("spans")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let requests = summarize_telemetry_requests(spans);
    let ttft_values = requests
        .iter()
        .filter_map(TelemetryRequestSummary::ttft_ms)
        .collect::<Vec<_>>();
    let request_latency_values = requests
        .iter()
        .filter_map(TelemetryRequestSummary::request_latency_ms)
        .collect::<Vec<_>>();
    let generation_latency_values = requests
        .iter()
        .filter_map(|summary| summary.generation_latency_ms)
        .collect::<Vec<_>>();
    let span_count = report
        .get("counts")
        .and_then(|counts| counts.get("spans"))
        .and_then(Value::as_u64)
        .or(Some(spans.len() as u64));
    let status = if spans.is_empty() {
        "no_spans"
    } else if ttft_values.is_empty() {
        "no_decode_token_spans"
    } else {
        "ok"
    };
    let detail = match status {
        "no_spans" => Some(
            "metrics report has no spans; target endpoint may not be emitting this run id"
                .to_string(),
        ),
        "no_decode_token_spans" => Some(
            "metrics report has spans, but no stage.openai_decode_token spans; run endpoint with debug telemetry for server-side TTFT"
                .to_string(),
        ),
        _ => None,
    };
    BenchTelemetry {
        metrics_http: Some(metrics_http.to_string()),
        metrics_run_id: metrics_run_id.to_string(),
        status,
        detail,
        report_path: Some(report_path.display().to_string()),
        request_count: Some(requests.len() as u64),
        span_count,
        ttft_ms: aggregate(&ttft_values),
        fttt_ms: aggregate(&ttft_values),
        request_latency_ms: aggregate(&request_latency_values),
        generation_latency_ms: aggregate(&generation_latency_values),
    }
}

#[derive(Default)]
struct TelemetryRequestSummary {
    first_start_unix_nanos: Option<i64>,
    last_end_unix_nanos: Option<i64>,
    first_decode_token_start_unix_nanos: Option<i64>,
    generation_latency_ms: Option<f64>,
}

impl TelemetryRequestSummary {
    fn observe_span(&mut self, span: &Value) {
        let Some(start) = span.get("start_time_unix_nanos").and_then(Value::as_i64) else {
            return;
        };
        let end = span
            .get("end_time_unix_nanos")
            .and_then(Value::as_i64)
            .unwrap_or(start);
        self.first_start_unix_nanos = Some(
            self.first_start_unix_nanos
                .map(|current| current.min(start))
                .unwrap_or(start),
        );
        self.last_end_unix_nanos = Some(
            self.last_end_unix_nanos
                .map(|current| current.max(end))
                .unwrap_or(end),
        );
        if span.get("name").and_then(Value::as_str) == Some(OPENAI_DECODE_TOKEN_SPAN) {
            self.first_decode_token_start_unix_nanos = Some(
                self.first_decode_token_start_unix_nanos
                    .map(|current| current.min(start))
                    .unwrap_or(start),
            );
        }
        if span.get("name").and_then(Value::as_str) == Some("stage.openai_generation_summary") {
            self.generation_latency_ms = span
                .get("attributes")
                .and_then(|attrs| attrs.get("llama_stage.elapsed_ms"))
                .and_then(Value::as_f64)
                .or_else(|| Some(nanos_to_ms(end.saturating_sub(start))));
        }
    }

    fn ttft_ms(&self) -> Option<f64> {
        Some(nanos_to_ms(
            self.first_decode_token_start_unix_nanos?
                .saturating_sub(self.first_start_unix_nanos?),
        ))
    }

    fn request_latency_ms(&self) -> Option<f64> {
        Some(nanos_to_ms(
            self.last_end_unix_nanos?
                .saturating_sub(self.first_start_unix_nanos?),
        ))
    }
}

fn summarize_telemetry_requests(spans: &[Value]) -> Vec<TelemetryRequestSummary> {
    let mut by_request = BTreeMap::<String, TelemetryRequestSummary>::new();
    for span in spans {
        let request_id = span.get("request_id").and_then(Value::as_str).or_else(|| {
            span.get("attributes")
                .and_then(|attrs| attrs.get("llama_stage.request_id"))
                .and_then(Value::as_str)
        });
        if let Some(request_id) = request_id {
            by_request
                .entry(request_id.to_string())
                .or_default()
                .observe_span(span);
        }
    }
    by_request.into_values().collect()
}

fn aggregate(values: &[f64]) -> Option<TelemetryAggregate> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let sum = values.iter().sum::<f64>();
    Some(TelemetryAggregate {
        count: values.len(),
        min: values[0],
        mean: sum / values.len() as f64,
        p50: percentile(&values, 0.50),
        p95: percentile(&values, 0.95),
        max: values[values.len() - 1],
    })
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn nanos_to_ms(nanos: i64) -> f64 {
    nanos as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        path::PathBuf,
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::*;

    #[test]
    fn finalize_only_posts_once_without_fetching_the_report() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = stream.read(&mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let body = r#"{"status":"finalized","request_count":17,"span_count":2048}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            String::from_utf8(request).unwrap()
        });

        let telemetry = finalize_only(&format!("http://{address}"), "run-finalize-only").unwrap();
        let request = server.join().unwrap();

        assert!(request.starts_with("POST /v1/runs/run-finalize-only/finalize HTTP/1.1\r\n"));
        assert_eq!(telemetry.status, "finalized");
        assert_eq!(telemetry.request_count, Some(17));
        assert_eq!(telemetry.span_count, Some(2048));
        assert!(telemetry.report_path.is_none());
    }

    #[test]
    fn telemetry_report_extracts_ttft_and_request_latency() {
        let run_dir = temp_run_dir("telemetry-metrics");
        fs::create_dir_all(run_dir.join("raw")).unwrap();
        let report = json!({
            "counts": {"spans": 3},
            "spans": [
                {
                    "request_id": "req-1",
                    "name": "stage.openai_tokenize",
                    "start_time_unix_nanos": 1_000_000_000_i64,
                    "end_time_unix_nanos": 1_010_000_000_i64,
                    "attributes": {}
                },
                {
                    "request_id": "req-1",
                    "name": "stage.openai_decode_token",
                    "start_time_unix_nanos": 1_125_000_000_i64,
                    "end_time_unix_nanos": 1_130_000_000_i64,
                    "attributes": {}
                },
                {
                    "request_id": "req-1",
                    "name": "stage.openai_generation_summary",
                    "start_time_unix_nanos": 1_000_000_000_i64,
                    "end_time_unix_nanos": 1_250_000_000_i64,
                    "attributes": {"llama_stage.elapsed_ms": 250.0}
                }
            ]
        });
        let telemetry = telemetry_from_report(
            "http://127.0.0.1:18080",
            "run-telemetry",
            &run_dir.join("raw/metrics-report.json"),
            report,
        );
        assert_eq!(telemetry.status, "ok");
        assert_eq!(telemetry.request_count, Some(1));
        assert_eq!(telemetry.span_count, Some(3));
        assert_eq!(telemetry.ttft_ms.as_ref().unwrap().mean, 125.0);
        assert_eq!(telemetry.request_latency_ms.as_ref().unwrap().mean, 250.0);
        assert_eq!(
            telemetry.generation_latency_ms.as_ref().unwrap().mean,
            250.0
        );
        let _ = fs::remove_dir_all(run_dir);
    }

    fn temp_run_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "skippy-bench-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ))
    }
}
