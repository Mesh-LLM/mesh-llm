//! Round aggregation that preserves unavailable measurements.
use super::{
    acceptance::{Aggregate, Version},
    telemetry::{Cell, Summary},
};
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub(super) struct Input {
    pub cells: Vec<Cell>,
}

fn median(
    rows: &[&Summary],
    measurement: impl Fn(&Summary) -> Option<f64>,
) -> DynResult<Option<f64>> {
    let mut values: Vec<_> = rows.iter().filter_map(|row| measurement(row)).collect();
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("round measurements must be finite and nonnegative".into());
    }
    if values.is_empty() {
        return Ok(None);
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(Some(if values.len().is_multiple_of(2) {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    }))
}

fn total(rows: &[&Summary], count: impl Fn(&Summary) -> u64) -> DynResult<u64> {
    rows.iter().try_fold(0_u64, |sum, row| {
        sum.checked_add(count(row))
            .ok_or_else(|| "aggregate request count overflow".into())
    })
}

fn version(cells: &[Cell], version: Version) -> DynResult<Aggregate> {
    let rows: Vec<_> = cells
        .iter()
        .filter(|cell| cell.version == version)
        .map(|cell| &cell.summary)
        .collect();
    Ok(Aggregate {
        version,
        rounds: rows.len() as u64,
        requests: total(&rows, |row| row.requests)?,
        successful: total(&rows, |row| row.successful)?,
        capacity_rejections: total(&rows, |row| row.capacity_rejections)?,
        cache_hits_median: median(&rows, |row| Some(row.cache_hits as f64))?,
        suffix_prefill_tokens_median: median(&rows, |row| row.suffix_prefill_tokens_total)?,
        resident_evicted_tokens_median: median(&rows, |row| row.resident_evicted_tokens_total)?,
        resident_evicted_entries_median: median(&rows, |row| row.resident_evicted_entries_total)?,
        predicted_recompute_cost_median: median(&rows, |row| row.predicted_recompute_cost_total)?,
        ttft_ms_p50_median: median(&rows, |row| row.ttft_ms_p50)?,
        ttft_ms_p95_median: median(&rows, |row| row.ttft_ms_p95)?,
        makespan_ms_median: median(&rows, |row| Some(row.makespan_ms))?,
        output_tokens_per_second_median: median(&rows, |row| Some(row.output_tokens_per_second))?,
        family_switches_median: median(&rows, |row| Some(row.family_switches as f64))?,
    })
}

pub(super) fn aggregate(input: Input) -> DynResult<Vec<Aggregate>> {
    let mut identities = BTreeSet::new();
    for cell in &input.cells {
        if cell.round == 0 || !identities.insert((cell.version, cell.round)) {
            return Err("aggregate has invalid or duplicate round identity".into());
        }
        if cell.summary.successful > cell.summary.requests {
            return Err("successful request count exceeds observed requests".into());
        }
    }
    Ok(vec![
        version(&input.cells, Version::Old)?,
        version(&input.cells, Version::New)?,
    ])
}
