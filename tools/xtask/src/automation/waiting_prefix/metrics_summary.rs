//! Collector timing statistics, kept separate from client acceptance metrics.
use super::{acceptance::Version, metrics_correlation::Timing};
use crate::command::DynResult;
use serde::Serialize;
use std::{collections::BTreeSet, fmt::Write};

#[derive(Serialize)]
pub(super) struct Summary {
    pub version: Version,
    pub measured_requests: u64,
    pub server_ttft_ms_p50: f64,
    pub server_ttft_ms_p95: f64,
    pub server_request_latency_ms_p50: f64,
    pub server_request_latency_ms_p95: f64,
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile).round_ties_even() as usize;
    values[index]
}

pub(super) fn summarize(version: Version, cells: &[Vec<Timing>]) -> DynResult<Summary> {
    if cells.is_empty() || cells.iter().any(Vec::is_empty) {
        return Err("collector summary requires measured requests in every cell".into());
    }
    let mut ttft = Vec::new();
    let mut latency = Vec::new();
    for cell in cells {
        let mut ids = BTreeSet::new();
        for timing in cell {
            if timing.request_id.is_empty()
                || !ids.insert(&timing.request_id)
                || !timing.server_ttft_ms.is_finite()
                || !timing.server_request_latency_ms.is_finite()
                || timing.server_ttft_ms < 0.0
                || timing.server_request_latency_ms < timing.server_ttft_ms
            {
                return Err(
                    "collector summary contains invalid or duplicate request timing".into(),
                );
            }
            ttft.push(timing.server_ttft_ms);
            latency.push(timing.server_request_latency_ms);
        }
    }
    Ok(Summary {
        version,
        measured_requests: ttft.len() as u64,
        server_ttft_ms_p50: percentile(&mut ttft, 0.50),
        server_ttft_ms_p95: percentile(&mut ttft, 0.95),
        server_request_latency_ms_p50: percentile(&mut latency, 0.50),
        server_request_latency_ms_p95: percentile(&mut latency, 0.95),
    })
}

pub(super) fn render(rows: &[Summary]) -> DynResult<String> {
    if rows.len() != 2 || rows[0].version != Version::Old || rows[1].version != Version::New {
        return Err("collector report requires exactly one old and one new summary".into());
    }
    let mut output = String::from(
        "\nCollector timings over all measured requests, excluding cache seeding.\n\n| Metric | Before | After |\n| --- | ---: | ---: |\n",
    );
    let old = &rows[0];
    let new = &rows[1];
    writeln!(
        output,
        "| Collector measured requests | {} | {} |",
        old.measured_requests, new.measured_requests
    )?;
    for (label, before, after) in [
        (
            "Server TTFT p50 ms",
            old.server_ttft_ms_p50,
            new.server_ttft_ms_p50,
        ),
        (
            "Server TTFT p95 ms",
            old.server_ttft_ms_p95,
            new.server_ttft_ms_p95,
        ),
        (
            "Server request latency p50 ms",
            old.server_request_latency_ms_p50,
            new.server_request_latency_ms_p50,
        ),
        (
            "Server request latency p95 ms",
            old.server_request_latency_ms_p95,
            new.server_request_latency_ms_p95,
        ),
    ] {
        writeln!(output, "| {label} | {before:.1} | {after:.1} |")?;
    }
    output.push_str("\nHardware acceptance uses the client measurements above. Collector timings use server spans and have a different measurement boundary.\n");
    Ok(output)
}

#[cfg(test)]
#[path = "metrics_summary_tests.rs"]
mod tests;
