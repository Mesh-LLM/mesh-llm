use crate::command::DynResult;
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Cell {
    pub concurrency: usize,
    pub trajectories: u64,
    pub requests: u64,
    pub successful_requests: u64,
    pub failed_request_ids: Option<Vec<String>>,
    #[serde(default)]
    pub successful_request_ids: Vec<String>,
    #[serde(default)]
    pub content_sha256_by_request: std::collections::BTreeMap<String, String>,
    pub prompt_tokens_min: Option<u64>,
    pub prompt_tokens_max: Option<u64>,
    #[serde(default)]
    pub completion_tokens: f64,
    #[serde(default)]
    pub prompt_tokens: f64,
    #[serde(default)]
    pub cached_tokens: f64,
    #[serde(default)]
    pub generation_seconds: f64,
    #[serde(default)]
    pub workload_window_seconds: f64,
    #[serde(default)]
    pub ttft_samples: Vec<f64>,
    #[serde(alias = "mean_request_decode_tokens_per_second")]
    pub decode_tokens_per_second: Option<f64>,
    pub agent_steps_per_second: Option<f64>,
    pub workload_output_tokens_per_second: Option<f64>,
    pub ttft_p50_seconds: Option<f64>,
    pub ttft_p95_seconds: Option<f64>,
    pub mean_in_flight: Option<f64>,
    #[serde(default, alias = "exact_output_requests")]
    pub budget_exhausted_requests: u64,
}

pub(super) fn aggregate(cells: &[&Cell]) -> DynResult<serde_json::Value> {
    let mut output = serde_json::Map::new();
    let sum = |select: fn(&Cell) -> f64| -> DynResult<f64> {
        let mut total = 0.0;
        for cell in cells {
            let value = select(cell);
            if !value.is_finite() || value < 0.0 {
                return Err("invalid pooled replay metric".into());
            }
            total += value;
        }
        if !total.is_finite() {
            return Err("pooled replay metric overflow".into());
        }
        Ok(total)
    };
    let completion = sum(|cell| cell.completion_tokens)?;
    let prompt = sum(|cell| cell.prompt_tokens)?;
    let cached = sum(|cell| cell.cached_tokens)?;
    let generation = sum(|cell| cell.generation_seconds)?;
    let window = sum(|cell| cell.workload_window_seconds)?;
    let success = cells.iter().try_fold(0_u64, |total, cell| {
        total
            .checked_add(cell.successful_requests)
            .ok_or("pooled success count overflow")
    })?;
    let requests = cells.iter().try_fold(0_u64, |total, cell| {
        total
            .checked_add(cell.requests)
            .ok_or("pooled request count overflow")
    })?;
    let exhausted = cells.iter().try_fold(0_u64, |total, cell| {
        total
            .checked_add(cell.budget_exhausted_requests)
            .ok_or("pooled exhausted count overflow")
    })?;
    output.insert(
        "success_pct".into(),
        serde_json::json!(ratio(
            100.0 * super::stream_evidence::number(success),
            super::stream_evidence::number(requests)
        )),
    );
    output.insert(
        "budget_exhausted_pct".into(),
        serde_json::json!(ratio(
            100.0 * super::stream_evidence::number(exhausted),
            super::stream_evidence::number(success)
        )),
    );
    output.insert(
        "cache_pct".into(),
        serde_json::json!(ratio(100.0 * cached, prompt)),
    );
    for (metric, select, numerator, denominator) in [
        (
            "agent_steps_per_second",
            (|cell: &Cell| cell.agent_steps_per_second) as fn(&Cell) -> Option<f64>,
            super::stream_evidence::number(success),
            window,
        ),
        (
            "workload_output_tokens_per_second",
            |cell: &Cell| cell.workload_output_tokens_per_second,
            completion,
            window,
        ),
        (
            "decode_tokens_per_second",
            |cell: &Cell| cell.decode_tokens_per_second,
            completion,
            generation,
        ),
    ] {
        let values = samples(cells, select)?;
        output.insert(
            metric.into(),
            serde_json::json!(ratio(numerator, denominator).or_else(|| mean(&values))),
        );
        ranges(&mut output, metric, &values);
    }
    let mut ttft = cells
        .iter()
        .flat_map(|cell| cell.ttft_samples.iter().copied())
        .collect::<Vec<_>>();
    check_samples(&ttft)?;
    ttft.sort_by(f64::total_cmp);
    for (metric, select, percent) in [
        (
            "ttft_p50_seconds",
            (|cell: &Cell| cell.ttft_p50_seconds) as fn(&Cell) -> Option<f64>,
            50,
        ),
        ("ttft_p95_seconds", |cell: &Cell| cell.ttft_p95_seconds, 95),
    ] {
        let values = samples(cells, select)?;
        output.insert(
            metric.into(),
            serde_json::json!(percentile(&ttft, percent)?.or_else(|| mean(&values))),
        );
        if percent == 50 {
            ranges(&mut output, metric, &values);
        }
    }
    let flight = samples(cells, |cell| cell.mean_in_flight)?;
    let weighted = sum(|cell| cell.mean_in_flight.unwrap_or(0.0) * cell.workload_window_seconds)?;
    let realized = ratio(weighted, window).or_else(|| mean(&flight));
    output.insert("mean_in_flight".into(), serde_json::json!(realized));
    let concurrency = cells.first().ok_or("empty pooled cell group")?.concurrency;
    output.insert(
        "concurrency_utilization_pct".into(),
        serde_json::json!(realized.and_then(|value| ratio(
            100.0 * value,
            super::stream_evidence::number(u64::try_from(concurrency).ok()?)
        ))),
    );
    Ok(output.into())
}

fn samples(cells: &[&Cell], select: fn(&Cell) -> Option<f64>) -> DynResult<Vec<f64>> {
    let values = cells
        .iter()
        .filter_map(|cell| select(cell))
        .collect::<Vec<_>>();
    check_samples(&values)?;
    Ok(values)
}
fn check_samples(values: &[f64]) -> DynResult<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("invalid pooled replay sample".into());
    }
    Ok(())
}
fn ranges(output: &mut serde_json::Map<String, serde_json::Value>, metric: &str, values: &[f64]) {
    output.insert(
        format!("{metric}_min"),
        serde_json::json!(values.iter().copied().min_by(f64::total_cmp)),
    );
    output.insert(
        format!("{metric}_max"),
        serde_json::json!(values.iter().copied().max_by(f64::total_cmp)),
    );
}
fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}
fn mean(values: &[f64]) -> Option<f64> {
    ratio(
        values.iter().sum(),
        super::stream_evidence::number(u64::try_from(values.len()).ok()?),
    )
}
fn percentile(values: &[f64], percent: usize) -> DynResult<Option<f64>> {
    if values.is_empty() {
        return Ok(None);
    }
    let rank = values
        .len()
        .checked_mul(percent)
        .ok_or("percentile rank overflow")?
        .div_ceil(100);
    Ok(Some(values[rank.saturating_sub(1).min(values.len() - 1)]))
}
