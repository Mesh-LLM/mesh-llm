//! Per-round request and KV telemetry measurements for waiting-prefix runs.
use super::acceptance::Version;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum Request {
    Failed {
        request_id: u64,
        family: String,
        error: String,
    },
    Completed {
        request_id: u64,
        family: String,
        first_token_ms: f64,
        ttft_ms: f64,
        tokens_predicted: u64,
        #[serde(default)]
        cached_tokens: u64,
    },
}

#[derive(Debug, Deserialize)]
pub(super) struct Event {
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Input {
    pub round: u64,
    pub version: Version,
    requests: Vec<Request>,
    events: Vec<Event>,
    capacity_events: Vec<Event>,
    record_events: Vec<Event>,
    makespan_ms: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Summary {
    pub requests: u64,
    pub successful: u64,
    #[serde(default)]
    pub errors: Vec<Request>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub usage_cached_requests: u64,
    pub matched_prefix_tokens_total: Option<f64>,
    pub suffix_prefill_tokens_total: Option<f64>,
    pub capacity_rejections: u64,
    pub resident_evicted_tokens_total: Option<f64>,
    pub resident_evicted_entries_total: Option<f64>,
    pub predicted_recompute_cost_total: Option<f64>,
    pub ttft_ms_p50: Option<f64>,
    pub ttft_ms_p95: Option<f64>,
    pub makespan_ms: f64,
    pub output_tokens_per_second: f64,
    pub family_switches: u64,
    pub family_service_order: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Cell {
    pub round: u64,
    pub version: Version,
    pub summary: Summary,
}

fn numeric(events: &[Event], key: &str) -> DynResult<Option<f64>> {
    let mut observed = false;
    let mut total = 0.0;
    for event in events {
        if let Some(value) = event.attributes.get(key).and_then(Value::as_f64) {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("invalid nonnegative telemetry measurement {key}").into());
            }
            observed = true;
            total += value;
        }
    }
    if !total.is_finite() {
        return Err(format!("telemetry total overflow for {key}").into());
    }
    Ok(observed.then_some(total))
}

fn status_count(events: &[Event], key: &str, expected: &str) -> u64 {
    events
        .iter()
        .filter(|event| event.attributes.get(key).and_then(Value::as_str) == Some(expected))
        .count() as u64
}

fn combine(left: Option<f64>, right: Option<f64>) -> DynResult<Option<f64>> {
    if left.is_none() && right.is_none() {
        return Ok(None);
    }
    let total = left.unwrap_or(0.0) + right.unwrap_or(0.0);
    if !total.is_finite() {
        return Err("combined eviction measurement overflow".into());
    }
    Ok(Some(total))
}

fn percentile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile).round_ties_even() as usize;
    values.get(index).copied()
}

fn request_summary(requests: Vec<Request>, makespan_ms: f64) -> DynResult<Summary> {
    if !makespan_ms.is_finite() || makespan_ms <= 0.0 {
        return Err("makespan must be a positive finite measurement".into());
    }
    let count = requests.len() as u64;
    let mut errors = Vec::new();
    let mut order = Vec::new();
    let mut ttft = Vec::new();
    let mut output_tokens = 0_u64;
    let mut cached = 0;
    let mut identities = BTreeSet::new();
    for request in requests {
        let identity = match &request {
            Request::Failed { request_id, .. } | Request::Completed { request_id, .. } => {
                *request_id
            }
        };
        if !identities.insert(identity) {
            return Err("duplicate request identity in measured round".into());
        }
        match request {
            Request::Failed { .. } => errors.push(request),
            Request::Completed {
                family,
                first_token_ms,
                ttft_ms,
                tokens_predicted,
                cached_tokens,
                ..
            } => {
                if family.is_empty()
                    || !first_token_ms.is_finite()
                    || first_token_ms < 0.0
                    || !ttft_ms.is_finite()
                    || ttft_ms < 0.0
                {
                    return Err(
                        "successful request requires finite nonnegative latency and a family"
                            .into(),
                    );
                }
                output_tokens = output_tokens
                    .checked_add(tokens_predicted)
                    .ok_or("output token count overflow")?;
                cached += u64::from(cached_tokens > 0);
                order.push((first_token_ms, family));
                ttft.push(ttft_ms);
            }
        }
    }
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let families: Vec<_> = order.into_iter().map(|(_, family)| family).collect();
    let throughput = output_tokens as f64 / (makespan_ms / 1000.0);
    if !throughput.is_finite() {
        return Err("output throughput overflow".into());
    }
    Ok(Summary {
        requests: count,
        successful: count - errors.len() as u64,
        errors,
        cache_hits: 0,
        cache_misses: 0,
        usage_cached_requests: cached,
        matched_prefix_tokens_total: None,
        suffix_prefill_tokens_total: None,
        capacity_rejections: 0,
        resident_evicted_tokens_total: None,
        resident_evicted_entries_total: None,
        predicted_recompute_cost_total: None,
        ttft_ms_p50: percentile(&mut ttft, 0.50),
        ttft_ms_p95: percentile(&mut ttft, 0.95),
        makespan_ms,
        output_tokens_per_second: throughput,
        family_switches: families
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count() as u64,
        family_service_order: families,
    })
}

pub(super) fn summarize(input: Input) -> DynResult<Cell> {
    if input.round == 0 {
        return Err("round numbers must be positive".into());
    }
    let mut summary = request_summary(input.requests, input.makespan_ms)?;
    summary.cache_hits = status_count(&input.events, "skippy.kv.status", "hit");
    summary.cache_misses = status_count(&input.events, "skippy.kv.status", "miss");
    summary.matched_prefix_tokens_total =
        numeric(&input.events, "skippy.kv.matched_prefix_tokens")?;
    summary.suffix_prefill_tokens_total =
        numeric(&input.events, "skippy.kv.suffix_prefill_tokens")?;
    summary.capacity_rejections = status_count(
        &input.capacity_events,
        "skippy.kv.capacity_status",
        "rejected",
    );
    summary.predicted_recompute_cost_total = numeric(
        &input.capacity_events,
        "skippy.kv.capacity_predicted_recompute_cost",
    )?;
    let proactive: Vec<_> = input
        .record_events
        .into_iter()
        .filter(|event| {
            event
                .attributes
                .get("skippy.kv.decision")
                .and_then(Value::as_str)
                == Some("proactive_eviction")
        })
        .collect();
    summary.resident_evicted_tokens_total = combine(
        numeric(&input.capacity_events, "skippy.kv.capacity_evicted_tokens")?,
        numeric(&proactive, "skippy.kv.proactive_evicted_tokens")?,
    )?;
    summary.resident_evicted_entries_total = combine(
        numeric(&input.capacity_events, "skippy.kv.capacity_evicted_entries")?,
        numeric(&proactive, "skippy.kv.proactive_evicted_entries")?,
    )?;
    Ok(Cell {
        round: input.round,
        version: input.version,
        summary,
    })
}
