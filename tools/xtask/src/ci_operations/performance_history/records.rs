//! Version-one competitive history records; independent of agentic replay schema3.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Row {
    pub schema_version: u32,
    pub created_utc: Option<String>,
    pub source_sha: Option<String>,
    pub cohort_key: String,
    pub cohort: Value,
    pub backend_binary_sha256: Option<String>,
    pub observed_gpu_state: Value,
    pub prompt_count: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub output_tokens: u64,
    pub measured_wall_ms: f64,
    pub output_tokens_per_second: f64,
    pub ttft_ms_mean: f64,
    pub complete: bool,
    pub artifact_result: String,
}
impl Row {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1 || self.cohort_key.is_empty() || !self.cohort.is_object() {
            return Err("unsupported competitive history row".into());
        }
        if self.source_sha.as_deref().is_none_or(|source| {
            source.len() != 40
                || !source
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }) || self.created_utc.as_deref().is_none_or(str::is_empty)
        {
            return Err("history row requires created_utc and a40-character source SHA".into());
        }
        for value in [
            self.measured_wall_ms,
            self.output_tokens_per_second,
            self.ttft_ms_mean,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err("history metrics must be finite and nonnegative".into());
            }
        }
        if self.complete
            && (self.failed_requests != 0
                || self.successful_requests != self.prompt_count
                || self.prompt_count == 0)
        {
            return Err("complete history row has inconsistent request counts".into());
        }
        Ok(())
    }
}
#[derive(Debug)]
pub(super) struct Comparison<'a> {
    pub row: &'a Row,
    pub baseline_count: usize,
    pub throughput_delta: Option<f64>,
    pub ttft_delta: Option<f64>,
    pub classification: &'static str,
}
pub(super) fn compare<'a>(
    current: &'a [Row],
    history: &[Row],
    throughput: f64,
    ttft: f64,
    cancellation: &crate::process::Cancellation,
) -> DynResult<Vec<Comparison<'a>>> {
    let mut cohorts = std::collections::BTreeMap::<&str, Vec<&Row>>::new();
    for row in history {
        check(cancellation)?;
        row.validate()?;
        cohorts.entry(&row.cohort_key).or_default().push(row);
    }
    current
        .iter()
        .map(|row| {
            row.validate()?;
            check(cancellation)?;
            let mut baseline = Vec::new();
            for prior in cohorts.get(row.cohort_key.as_str()).into_iter().flatten() {
                check(cancellation)?;
                if prior.complete && prior.source_sha != row.source_sha {
                    baseline.push(*prior);
                }
            }
            let mut comparison = Comparison {
                row,
                baseline_count: baseline.len(),
                throughput_delta: None,
                ttft_delta: None,
                classification: "insufficient-baseline",
            };
            if !row.complete {
                comparison.classification = "correctness-failure";
            } else if baseline.len() >= 3 {
                let rates = baseline
                    .iter()
                    .map(|row| row.output_tokens_per_second)
                    .collect::<Vec<_>>();
                let times = baseline
                    .iter()
                    .map(|row| row.ttft_ms_mean)
                    .collect::<Vec<_>>();
                let (rate, rate_mad) = median_mad(&rates);
                let (time, time_mad) = median_mad(&times);
                comparison.throughput_delta = delta(row.output_tokens_per_second, rate);
                comparison.ttft_delta = delta(row.ttft_ms_mean, time);
                let rate_bad = comparison.throughput_delta.is_some_and(|d| d < -throughput)
                    && rate - row.output_tokens_per_second > (3.0 * rate_mad).max(rate * 0.01);
                let time_bad = comparison.ttft_delta.is_some_and(|d| d > ttft)
                    && row.ttft_ms_mean - time > (3.0 * time_mad).max(time * 0.01);
                comparison.classification = if rate_bad || time_bad {
                    "performance-regression"
                } else if comparison.throughput_delta.is_some() && comparison.ttft_delta.is_some() {
                    "pass"
                } else {
                    "insufficient-baseline"
                };
            }
            Ok(comparison)
        })
        .collect()
}
fn delta(candidate: f64, base: f64) -> Option<f64> {
    if base > 0.0 {
        Some(candidate / base - 1.0)
    } else {
        None
    }
}
fn median_mad(values: &[f64]) -> (f64, f64) {
    let middle = median(values);
    let deviations = values
        .iter()
        .map(|v| (v - middle).abs())
        .collect::<Vec<_>>();
    (middle, median(&deviations))
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        sorted[n / 2 - 1] / 2.0 + sorted[n / 2] / 2.0
    } else {
        sorted[n / 2]
    }
}
pub(super) fn report(comparisons: &[Comparison<'_>]) -> String {
    let mut text = String::from(
        "# Performance history regression report\n\nComparisons are restricted to exact cohort keys. At least three prior complete runs are required; thresholds are report-only unless `--gate` is used.\n\n| model | backend | c | baseline runs | throughput Δ | TTFT Δ | result |\n|---|---|---:|---:|---:|---:|---|\n",
    );
    for item in comparisons {
        let c = &item.row.cohort;
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            escape(c["model"].as_str().unwrap_or("unknown")),
            escape(c["arm"].as_str().unwrap_or("unknown")),
            c["concurrency"],
            item.baseline_count,
            percentage(item.throughput_delta),
            percentage(item.ttft_delta),
            item.classification
        ));
    }
    text
}
fn escape(value: &str) -> String {
    value.replace('|', "\\|").replace(['\n', '\r'], " ")
}
fn percentage(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |v| format!("{:+.1}%", v * 100.0))
}

fn check(cancellation: &crate::process::Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        Err("performance history cancelled".into())
    } else {
        Ok(())
    }
}
